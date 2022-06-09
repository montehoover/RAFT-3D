# The 'devel' version of nvidia/cuda contains nvcc but the 'runtime' version does not so we use 'devel'
FROM nvcr.io/nvidia/cuda:11.3.1-devel-ubuntu20.04

# Avoid interactive questions that come from some apt package installs
ARG DEBIAN_FRONTEND="noninteractive"

# software-properties-common gives us python3.8 (and add-apt-repository)
RUN apt-get update && apt-get install -y \
        software-properties-common \
        git \
    && rm -rf /var/lib/apt/lists/*

# Python stuff. None of these are necessary on a typical Ubuntu user environment,
# but needed here to get the docker image to feel like a virtual environment.
# (And python3-opencv gives us libs we need and are typically already present in standard Ubuntu)
RUN apt-get update && apt-get install -y \
        python3-opencv \
        python3-pip \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# ---- Everything above this line sets up a clean py38-cuda11.3 image setup like an empty pip venv -------

# Only needed for running GUI apps from the container.
RUN apt-get update && apt-get install -y \
        xauth \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements. Note this is done in two stages because lietorch in
# requirements2.txt requires torch to already be installed.
# Also apt install ninja-build because it speeds up building the lietorch wheel.
RUN apt-get update && apt-get install -y \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements2.txt ./
RUN pip install -r requirements.txt && \ 
    pip install -r requirements2.txt && \
    rm requirements.txt requirements2.txt

# Delete ninja-build because it was only needed to make the above step go faster.
RUN apt-get remove -y ninja-build 

# Create RAFT-3D directory and cd into it for future commands and for container startup
WORKDIR /RAFT-3D

# Download RAFT-3D pretrained model
RUN gdown https://drive.google.com/uc?id=1Lt14WdzPQIjaOqVLbvNBqdDLtN9wtxbs
