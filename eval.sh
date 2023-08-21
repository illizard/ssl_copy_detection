sscd/disc_eval.py --disc_path ../dataset/DISC2021_exp/ --gpus=2  \
    --output_path=./   --size=224 --preserve_aspect_ratio=True \
    --backbone=OFFL_DINO --dims=768 \
    --model_state=/hdd/wi/sscd-copy-detection/ckpt/dino/lightning_logs/version_0/checkpoints/epoch=49-step=19499.ckpt

# # disc_eval_modi
# python sscd/disc_eval_modi.py --disc_path ../dataset/DISC2021_mini/ --gpus=2  \
#     --output_path=./   --size=224 --preserve_aspect_ratio \
#     --backbone=OFFL_DINO --dims=768 \
#     --model_state=/hdd/wi/sscd-copy-detection/ckpt/dino/lightning_logs/version_0/checkpoints/epoch=49-step=19499.ckpt