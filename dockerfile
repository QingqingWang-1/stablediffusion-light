FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04
ENV TZ=America/New_York
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 
RUN apt-get update && apt-get install -y tzdata && \
    apt-get install -y build-essential cmake curl unzip git wget pkg-config && \
    apt-get install -y python3-dev python3-pip && \
    apt-get install -y libboost-all-dev libtbb-dev libgflags-dev libgoogle-glog-dev libhdf5-dev libgtk2.0-dev libglib2.0-dev libgl1-mesa-glx && \
    pip3 install --no-cache-dir -r requirements.txt && \
    apt-get install -y libopencv-dev python3-opencv &&\
    git clone https://github.com/facebookresearch/xformers.git && \
    cd xformers && \
    git submodule update --init --recursive && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install -e .
