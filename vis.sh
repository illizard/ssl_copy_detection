#!/bin/bash
python -m sscd.disc_vis \
    --disc_path /hdd/wi/sscd-copy-detection/sample \
    --backbone vit_base_patch16_224.dino \
    --dims 768 \
    --mode pt \
    --batch_size 1 \
    --pt_model_state /hdd/wi/isc2021/models/dino_vitbase16_pretrain.pth \
    --ft_model_state /hdd/wi/sscd-copy-detection/ckpt/dino/lightning_logs/version_0/checkpoints/epoch=49-step=19499.ckpt \
    --output_path ./ \
    --gpus 2 \
    --size 224

#    --disc_path /hdd/wi/dataset/DISC2021_exp/references/images/references/ \

    # --backbone vit_tiny_patch16_224.augreg_in21k \
    # --dims 192 \
    # --mode timm \

#    --backbone vit_base_patch16_224.dino \
#    --dims 768 \
#    --mode ft \

