MASTER_ADDR="localhost" MASTER_PORT="15088" NODE_RANK="0" WORLD_SIZE=2 \
  ./sscd/train.py --nodes=1 --gpus=2 --batch_size=256 \
  --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/train/images/train \
  --val_dataset_path=/hdd/wi/dataset/DISC2021_exp/ \
  --entropy_weight=30 --epochs=50 --augmentations=ADVANCED --mixup=true  \
  --output_path=./ckpt/my_dtop_vit \
  --backbone=MY_DTOP_VIT --dims=192 \
  #  --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/train_50k/images/train_50k \ 가 왜 필요한지좀 알아내야겠다
