# Base image
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ninja-build \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    libgl1-mesa-glx \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create Conda environment
RUN conda create -n SPARK python=3.9 -y && conda clean -a
SHELL ["conda", "run", "-n", "SPARK", "/bin/bash", "-c"]

# Install dependencies for MultiFLARE
RUN conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y && \
    conda install iopath -c iopath -y && \
    pip install ninja && \
    pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2 && \
    pip install git+https://github.com/NVlabs/nvdiffrast/ && \
    pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch && \
    pip install gpytoolbox opencv-python meshzoo trimesh matplotlib chumpy lpips configargparse open3d wandb && \
    pip install xatlas && \
    pip install git+https://github.com/jonbarron/robust_loss_pytorch

# Install dependencies for TrackerAdaptation
RUN pip install mediapipe==0.10.11 && \
    pip install timm~=0.9.16 adabound~=0.0.5 compress-pickle~=1.2.0 face-alignment==1.3.4 facenet-pytorch~=2.5.1 imgaug==0.4.0 albumentations==1.4.8 scikit-video==1.1.11 && \
    conda install omegaconf~=2.0.6 pytorch-lightning==1.4.9 torchmetrics==0.6.2 hickle==5.0.2 munch~=2.5.0 torchfile==0.1.0 -c conda-forge -y

# Reinstall specific numpy version
RUN pip install numpy==1.23

# Install gdown for downloading pretrained models
RUN pip install gdown

# Activate Conda environment by default
ENV PATH="/opt/conda/envs/SPARK/bin:$PATH"
CMD ["bash"]

