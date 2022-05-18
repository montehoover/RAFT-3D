# Base image
FROM nvcr.io/nvidia/cuda:11.3.1-runtime-ubuntu20.04

# Install system basics
# software-properties-common gives us python3.8
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    python3-pip \
    python3-opencv \
    python-is-python3

# # Install python3.9, pip, and lib dependencies needed for opencv
# RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get install -y \
#     python3.9

# Create the following directory in the docker image and cd into it for future commands
WORKDIR /RAFT-3D

# Copy from pwd on host machine (first dot) into pwd in image (second dot)
COPY . .

# Install python requirements
RUN pip install -r requirements.txt && \ 
    pip install -r requirements2.txt

# Download RAFT-3D pretrained model
RUN gdown https://drive.google.com/uc?id=1Lt14WdzPQIjaOqVLbvNBqdDLtN9wtxbs

