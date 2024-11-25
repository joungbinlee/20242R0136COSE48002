export WANDB_API_KEY=04a1e4bd9b4bedf915e7c9ba1c9d460872289c3f
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# CUDA_VISIBLE_DEVICES=2 python train_vae.py \
#     -s /media/dataset1/HDTF/data \
#     --model_path /media/dataset2/joungbin/GaussianTalker/output/vae/test \
#     --configs arguments/64_dim_1_transformer.py \
#     --expname test_64_dim_1_transformer \
#     --vae_pretrained /media/dataset2/joungbin/animate_anyone/pretrained_weights/sd-vae-ft-mse \


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    train_vae.py \
    -s /media/dataset1/HDTF/data \
    --model_path /media/dataset2/joungbin/GaussianTalker/output/vae/train_vae_10_identity \
    --configs arguments/64_dim_1_transformer.py \
    --use_wandb \
    --expname train_vae_10_identity \
    --vae_pretrained /media/dataset2/joungbin/animate_anyone/pretrained_weights/sd-vae-ft-mse \
    
    # --start_checkpoint 21000 \
