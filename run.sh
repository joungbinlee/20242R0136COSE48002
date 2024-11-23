export WANDB_API_KEY=04a1e4bd9b4bedf915e7c9ba1c9d460872289c3f

# CUDA_VISIBLE_DEVICES=2 python train_vae.py \
#     -s /media/dataset2/joungbin/GaussianTalker/data/obama \
#     --model_path /media/dataset2/joungbin/GaussianTalker/output/vae/obama \
#     --configs arguments/64_dim_1_transformer.py \
#     --use_wandb \
#     --expname obama_64_dim_1_transformer \


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    train_vae.py \
    -s /media/dataset2/joungbin/GaussianTalker/data/obama \
    --model_path /media/dataset2/joungbin/GaussianTalker/output/vae/obama \
    --configs arguments/64_dim_1_transformer.py \
    --use_wandb \
    --expname obama_64_dim_1_transformer \
    --vae_pretrained /media/dataset2/joungbin/animate_anyone/pretrained_weights/sd-vae-ft-mse \
