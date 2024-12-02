CUDA_VISIBLE_DEVICES=3 \
    python train.py -s /media/dataset2/joungbin/VHAP/data/obama/obama_whiteBg_staticOffset_maskBelowLine \
    --model_path /media/dataset2/joungbin/GaussianTalker/output/Diff_gen/obama \
    --configs arguments/64_dim_1_transformer.py \
    --port 1249 \


# CUDA_VISIBLE_DEVICES=2 \
#     python train.py -s /media/dataset2/joungbin/data/Adam_Schiff \
#     --model_path /media/dataset2/joungbin/GaussianTalker/output/add_token_train_gs/final/64_dim_1_transformer_Adam_Schiff \
#     --configs arguments/64_dim_1_transformer.py \
#     --port 1235


