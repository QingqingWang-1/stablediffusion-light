#class guided conditional diffusion model, use cifar10 dataset
import os 
import numpy as np
from types import SimpleNamespace
from contextlib import nullcontext
import argparse, logging, copy 
import torch 
import torch.nn as nn
from torch import optim 
from matplotlib import pyplot as plt
from tqdm import tqdm
from modules import UNet_conditional, EMA
from utils import *
import wandb

config = SimpleNamespace(
    run_name = "DDPM_conditional_cifar10",
    epochs = 400,  #100
    noise_steps = 1000,
    seed = 42,
    batch_size = 48, #64, train with batch_size, validation with 2*batch_size in DataLoader
    img_size = 32,
    num_classes = 10,
    dataset_path = "data/Cifar10",
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1, #if 1, use the whole dataset; otherwise, split the original dataset to N=slice_size smaller subsets, and choose one to do experiments
    do_validation = False,
    fp16 = True,
    log_images_every_epoch = 10,
    save_models_every_epoch = 100,
    num_workers = 1,
    lr = 1e-3) #5e-3

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, device="cuda", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta 
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
                
        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device 
        self.c_in = c_in
        self.num_classes = num_classes
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x) #normal distribution
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon  #add noise to input image
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    @torch.inference_mode() #sampling is only used at inference stage
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new images...")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale>0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise 
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x 

    def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data_cifar(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, steps_per_epoch=len(self.train_dataloader) // args.batch_size, epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler() #in mix-precision (f16 for weights), this class is used to scale gradients to avoid gradient vanishing or exploring

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()  #loss.backward() in mix-precision context
        self.scaler.step(self.optimizer) #optimizer.step() in mix-precision context
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train:
            self.model.train()
        else:
            self.model.eval()
        pbar = tqdm(self.train_dataloader)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()): #autocast is used to start mix-precision
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_image(images, t)
                if np.random.random()<0.1:
                    labels=None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                wandb.log({
                    "train_mse": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0]
                })
            pbar.set_postfix(MSE=loss.item())

        return avg_loss.mean().item()

    def log_images(self, run_name, epoch):
        "log images to wandb and save them to disk"
        labels = torch.arange(self.num_classes).long().to(self.device)
        sampled_images = self.sample(use_ema=False, labels=labels)
        plot_images(sampled_images) #to display on jupyter if available
        save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))
        wandb.log({
            "sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]
        })
        ema_sampled_images = self.sample(use_ema=True, labels=labels)
        plot_images(ema_sampled_images) #to display on jupyter if available
        save_images(ema_sampled_images, os.path.join("results", run_name, f"ema_{epoch}.jpg"))
        wandb.log({
            "ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]
        })

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))
    
    def save_model(self, run_name, epoch=-1):
        "save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch":epoch})
        at.add_dir(os.path.join("models", run_name))
        wandb.log_artifact(at)
    
    def fit(self, args):
        for epoch in range(1, args.epochs+1):
            logging.info(f"Starting epoch {epoch}/{args.epochs}:")
            _ = self.one_epoch(train=True)

            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                wandb.log({
                    "val_mse": avg_loss
                })
            
            if epoch % args.log_images_every_epoch == 0:
                self.log_images(run_name=args.run_name, epoch=epoch) #sampling images with model and ema_model, and save sampled images
            if epoch % args.save_models_every_epoch == 0:
                self.save_model(run_name=args.run_name, epoch=epoch)
        
def parse_args(config):
    parser = argparse.ArgumentParser(description="process hyper-parameters")
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')

    args = vars(parser.parse_args())
    #update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == "__main__":
    parse_args(config)

    #seed everything
    set_seed(config.seed)

    diffuser = Diffusion(noise_steps=config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    with wandb.init(project="train_sd", group="train", config=config):
        diffuser.prepare(config)
        diffuser.fit(config)










