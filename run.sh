CUDA_VISIBLE_DEVICES=0 python train_vae.py \
    -s /media/dataset2/joungbin/GaussianTalker/data/obama \
    --model_path /media/dataset2/joungbin/GaussianTalker/output/vae/obama \
    --configs arguments/64_dim_1_transformer.py 