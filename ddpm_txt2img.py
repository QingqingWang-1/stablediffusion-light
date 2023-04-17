import argparse
import logging as logging
import math 
import os
import random
from pathlib import Path 
from typing import Optional

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami 
from packaging import version 
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from utils import *

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
import xformers
import torch.distributed as dist 
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--dataset_config_name",type=str, default=None, help="The config of the Dataset, leave as None if there is only one config.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Supported platforms are `tensorboard', 'wandb' and 'comet_ml', use all to report to all integrations")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`.")
    
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4", help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, default="lambdalabs/pokemon-blip-captions", help="lambdalabs/pokemon-blip-captions from huggingface")
    parser.add_argument("--train_data_dir", type=str, default=None, help="For training the model on private dataset")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.")
    parser.add_argument("--center_crop", type=bool, default=True, help="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. Images will be resized to the resolution first before cropping.")
    parser.add_argument("--random_flip", type=bool, default=True, help="whether to randomly flip images horizontally.")
    parser.add_argument("--output_dir", type=str, default="results/ddpm-txt2img-pokemon", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", type=str, default="cache", help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="The scheduler type to use. Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")

    parser.add_argument("--local_rank", type=str, default='0', help="For distributed training.")
    parser.add_argument("--rank", type=str, default='0', help="For distributed training.")
    parser.add_argument("--world_size", type=int, default=1, help="Indicate how many processes will be used for distributed training")
    parser.add_argument("--max_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    parser.add_argument("--use_ema", type=bool, default=True, help="whether to use EMA model.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=False, action="store_true", help="Whether or not to use xformers.")

    args = parser.parse_args() 
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args 
dataset_name_mapping = {"lambdalabs/pokemon-blip-captions":("image", "text")}

def main(train=True, prompt=None, save_path=None):
    args = parse_args()
    #since only have one GPU
    os.environ["LOCAL_RANK"]=args.local_rank #rank of current process in current node
    os.environ["RANK"] = args.rank #rank of current process in global environment ( value should be in [0, world_size-1])
    os.environ['MASTER_ADDR'] = 'localhost' #set to the machine runing the master process

    #set distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count()) #count the GPUs in one node to get current GPU No.
    dist.init_process_group(backend='nccl', init_method="tcp://localhost:8888", rank=rank, world_size=args.world_size)
    device = torch.device("cuda") #put model to particular GPU

    if not train:  #Test inference
        model_path = args.output_dir
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.to(device)
        if prompt is None:
            prompt = "yoda"
        save_path = args.output_dir if save_path is None else save_path
        os.makedirs(save_path, exist_ok=True)
        image = pipe(prompt=prompt).images[0]
        image.save(os.path.join(save_path, f"pokemon_{prompt}.png"))
        return

    if dist.get_rank()==0 and args.output_dir is not None:
        os.makedirs(args.output_dir,exist_ok=True)
        os.makedirs(args.cache_dir,exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logging_file = os.path.join(args.output_dir, "log.txt")
    file_handler = logging.FileHandler(logging_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if dist.get_rank()==0: #main process
        datasets.utils.logging.set_verbosity_warning() #set the level of log to warning (only warning and above will be recorded in log file)
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:  #At fine-tuning stage, set to None (if LoRA is used, set to 42) 
        set_seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    #VAE with encoder-decoder used in stable diffusion model.  
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision) 
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    unet = unet.to(device)
    unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[local_rank], output_device=local_rank)
    #Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    #Create EMA for the unet
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
        ema_unet = ema_unet.to(device)
    
    if args.enable_xformers_memory_efficient_attention:
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn("xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please upgrade xFormers to at least 0.0.17")
        unet.enable_xformers_memory_efficient_attention()
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=args.cache_dir)

    column_names = dataset["train"].column_names

    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    image_column = dataset_columns[0] #image
    caption_column = dataset_columns[1] #text

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(f"Caption column '{caption_column}' should contain either strings or lists of strings")
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids 
    
    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples 
    
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples)) #select a small portain of training samples
    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.train_batch_size, 
                                                   pin_memory=True, num_workers=args.dataloader_num_workers, sampler=sampler)

    max_train_steps = args.num_train_epochs * math.ceil(len(train_dataloader) / args.train_batch_size)
    num_steps_per_epoch =  math.ceil(len(train_dataloader) / args.train_batch_size)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=max_train_steps)
    #set optimizer for distributed training
    optimizer = torch.optim.DistributedOptimizer(optimizer, named_parameters=unet.named_parameters())
    
    #cast text_encoder and vae weights to half-precision as they are only used for inference, keeping weights in full precision is not required
    weight_dtype = torch.float32 
    if args.max_precision == "fp16":
        weight_dtype = torch.float16
    elif args.max_precision == "bf16":
        weight_dtype = torch.bfloat16
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f" Training batch size = {args.train_batch_size}")
    logger.info(f"  Total training steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    #resume from the latest checkpoint
    dirs = os.listdir(args.output_dir)
    dirs - [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        logger.info(f"No checkpoint exists. Starting a new training run.")
    else:
        logger.info(f"Resume the latest checkpoint in {path}")
        unet.load_state_dict(os.path.join(args.output_dir, path)) 
        optimizer.load_state_dict(os.path.join(args.output_dir, "optimizer.pt")) 
        if args.use_ema:
            ema_unet.load_state_dict(os.path.join(args.output_dir, f"ema-{path}")) #checkpoints for ema are stored as "ema_checlpoint_***.pt"
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_steps_per_epoch #start training from first_epoch
    assert first_epoch < args.num_train_epochs, f"resumed from step {global_step}/epoch {first_epoch}, which should be smaller than total training epochs {args.num_train_epoch}"
    with wandb.init(project="train_sd", group="train", config=args):
        for epoch in range(first_epoch, args.num_train_epochs):
            logger.info(f"Starting epoch {epoch} / {args.num_train_epochs}:")
            pbar = tqdm(train_dataloader, disable = not dist.get_rank()==0)
            unet.train()
            train_loss = 0.0
            train_dataloader.sampler.set_epoch(epoch)
            for step, batch in enumerate(pbar):
                latents = vae.encode(batch["pixel_values"].to(device).to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor 

                noise = torch.rand_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) #add noise of selected timestep
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                model_predict = unet(noisy_latents, timesteps, encoder_hidden_states).sample 
                loss = F.mse_loss(model_predict.float(), target.float(), reduction="mean")
                train_loss += loss

                #log and send to wandb
                loss_item =  loss.item()
                current_lr = lr_scheduler.get_last_lr()[0]
                wandb.log({
                    "train_MSE_loss": loss_item,
                    "learning_rate": current_lr,
                })
                logger.info(f"Epoch {epoch}/{args.num_train_epochs}, step {step}/{num_steps_per_epoch}: train_MSE_loss-{loss_item}, learning_rate-{current_lr}\n ")
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                pbar.update(1)
                pbar.set_postfix(MSE=loss.item()) 

                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    if dist.get_rank() == 0:
                        torch.save(unet.state_dict(), os.path.join(args.output_dir, f"checkpoint-{global_step}.pt"))
                        torch.save(ema_unet.state_dict(), os.path.join(args.output_dir, f"ema-checkpoint-{global_step}.pt"))
                        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, f"optimizer.pt"))

            avg_loss = train_loss.mean().item()
            wandb.log({"train_mse_mean_per_epoch": avg_loss})
            dist.barrier()
        
        logger.info("Training finished.")
        #create a pipeline using the trained modules and save it 
        if dist.get_rank()==0:
            if args.use_ema:
                ema_unet.copy_to(unet.parameters())
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
            )
            pipeline.save_pretrained(args.output_dir)

if __name__=="__main__":

    main(train=True)
    #main(train=False, prompt="yoda with red hair", save_path="results/pokemon"):


            


            
            



            












    

    


    








        

    


    
   





