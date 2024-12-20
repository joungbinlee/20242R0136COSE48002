# 20242R0136COSE48002: Industrial-Academic Capstone Design

## Introduction
This repository contains the implementation for our **Industrial-Academic Capstone Design Project**. It includes data preprocessing and model execution code tailored for our specific research and development goals.

## Installation
Run the following commands to set up the environment (details are in `requirements.txt`):

```bash
git clone https://github.com/joungbinlee/20242R0136COSE48002.git
cd 20242R0136COSE48002
conda create -n CapstoneEnv python=3.7 
conda activate CapstoneEnv

pip install -r requirements.txt
pip install -e submodules/custom-bg-depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e ./diffusers
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
pip install --upgrade "protobuf<=3.20.1"
```

## Data Preparation


### 1. Download Dataset
We used talking portrait videos from [AD-NeRF](https://github.com/YudongGuo/AD-NeRF), [GeneFace](https://github.com/yerfor/GeneFace) and [HDTF dataset](https://github.com/MRzzm/HDTF). These are static videos whose average length are about 3~5 minutes.

You can see an example video with the below line:

```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/dataset/vids/Obama.mp4?raw=true -O data/obama/obama.mp4
```

We also used [SynObama](https://grail.cs.washington.edu/projects/AudioToObama/) for cross-driven setting inference.

### 2. Prepare Face-Parsing Model
Download the face-parsing model:

```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
```

### 3. Download 3DMM Model
Download the Basel Face Model 2009 from [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details) and place `01_MorphableModel.mat` in the following directory:

```
data_utils/face_tracking/3DMM/
```

Convert the model and preprocess data:

```bash
cd data_utils/face_tracking
python convert_BFM.py
cd ../../
python data_utils/process.py ${YOUR_DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.mp4
```

### 4. Obtain AU45 for Eye Blinking
Run `FeatureExtraction` in [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), rename and move the output CSV file to the following directory structure:

```
├── (your dataset dir)
│   | (dataset name)
│       ├── gt_imgs
│           ├── 0.jpg
│           ├── 1.jpg
│           ├── 2.jpg
│           ├── ...
│       ├── ori_imgs
│           ├── 0.jpg
│           ├── 0.lms
│           ├── 1.jpg
│           ├── 1.lms
│           ├── ...
│       ├── parsing
│           ├── 0.png
│           ├── 1.png
│           ├── 2.png
│           ├── 3.png
│           ├── ...
│       ├── torso_imgs
│           ├── 0.png
│           ├── 1.png
│           ├── 2.png
│           ├── 3.png
│           ├── ...
│       ├── au.csv
│       ├── aud_ds.npy
│       ├── aud_novel.wav
│       ├── aud_train.wav
│       ├── aud.wav
│       ├── bc.jpg
│       ├── (dataset name).mp4
│       ├── track_params.pt
│       ├── transforms_train.json
│       ├── transforms_val.json
```


## Training

Run the training script with the following command:

```bash
python train.py -s ${YOUR_DATASET_DIR} --model_path ${YOUR_MODEL_DIR} --configs arguments/64_dim_1_transformer.py
```


```bash
python train_stage2.py -s ${YOUR_DATASET_DIR} --model_path ${YOUR_MODEL_DIR} --configs arguments/64_dim_1_transformer.py
```

- `<YOUR_DATASET_DIR>`: Path to the dataset directory.
- `<YOUR_MODEL_DIR>`: Path to save the trained model.
- `arguments/64_dim_1_transformer.py`: Configuration file for training parameters.
