# Base image
FROM nvcr.io/nvidia/cuda:11.3.1-devel-ubuntu20.04

# Avoid interactive questions that come from some apt package installs
ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update

# software-properties-common gives us python3.8 (and add-apt-repository)
RUN apt-get install -y \
    software-properties-common \
    git

# Python stuff. None of these are necessary on a typical Ubuntu user environment,
# but needed here to get the docker image to feel like a virtual environment.
RUN apt-get install -y \
    python3-opencv \
    python3-pip \
    python-is-python3

# Needed for running GUI apps from the container.
RUN apt-get install -y \
    xauth \
    python3-tk 

# Create the following directory in the docker image and cd into it for future commands
WORKDIR /RAFT-3D

# Copy from pwd on host machine (first dot) into pwd in image (second dot)
COPY . .

# Install python requirements
RUN pip install -r requirements.txt && \ 
    pip install -r requirements2.txt

# Download RAFT-3D pretrained model
RUN gdown https://drive.google.com/uc?id=1Lt14WdzPQIjaOqVLbvNBqdDLtN9wtxbs

