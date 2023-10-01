sscd/disc_eval.py --disc_path /hdd/wi/dataset/DISC2021_exp/ --gpus=2  \
    --output_path=./   --size=224 --preserve_aspect_ratio=True \
    --backbone=OFFL_VIT_TINY --dims=192 \
    --workers=0 \
    --model_state=/hdd/wi/sscd-copy-detection/ckpt/vit_tiny/lightning_logs/version_0/checkpoints/epoch=49-step=19499.ckpt

# disc_eval_modi
# python sscd/disc_eval.py --disc_path ../dataset/DISC2021_mini/ --gpus=2  \
#     --output_path=./   --size=224 --preserve_aspect_ratio=True \
#     --backbone=OFFL_DINO --dims=768 \
#     --workers=0 \
#     --model_state=/hdd/wi/sscd-copy-detection_my/ckpt/dino/lightning_logs/version_0/checkpoints/epoch=49-step=19499.ckpt

# sscd/disc_eval.py --disc_path ../dataset/DISC2021_exp --gpus=2 \
#   --output_path=./ \
#   --size=288 --preserve_aspect_ratio=true \
#   --workers=0 \
#   --backbone=CV_RESNET50 --dims=512 --model_state=/hdd/wi/sscd-copy-detection_my/ckpt/sscd_disc_mixup.classy.pt