import torch
import torch.nn as nn
import timm
from timm.models.layers import PatchEmbed
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sscd.models.model import Model
import argparse
from collections import OrderedDict
from tqdm import tqdm

import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from sscd.models import mae_vit, dino_vit

# 이미지들을 텐서로 로드하는 함수. 입력: 이미지 디렉터리 경로, 출력: 텐서로 변환된 이미지들

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path


# 거리 행렬을 계산하는 함수. 입력: 패치 크기, 패치 수, 길이, 출력: 거리 행렬
# Compute Distance Matrix
def compute_distance_matrix(patch_size, num_patches, length):
    """Helper function to compute distance matrix."""
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix # (196,196)

# 평균 주의 거리를 계산하는 함수. 입력: 패치 크기, 주의 가중치, 출력: 평균 거리
# 이미지 배치로 받음
# Compute MAD
# 한 인코더 블럭이 아니라 한 헤드당 처리됨
def compute_mean_attention_dist(patch_size, attention_weights):
    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    #attention_weights = (4, 12, 197, 197)
    attention_weights = attention_weights[..., 1:, 1:] # Removing the CLS token
    num_patches = attention_weights.shape[-1] #196
    length = int(np.sqrt(num_patches)) #14
    assert length**2 == num_patches, "Num patches is not perfect square"
    
    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token
    mean_distances = attention_weights * distance_matrix
    # print(f"mean_distances og shape \t\t\t{mean_distances.shape}")     #mean_distances.shape (4, 12, 196, 196) (batch, head, patch, patch)
    # print()
    # print(f"mean_distances before sum \t\t\t{mean_distances[0][0][0][:]}")    #mean_distances[0][0][0][:].shape (196,)
                                                                            # np.sum(mean_distances[0][0][0][:]) 0.004783373888156958
    mean_distances = np.sum(mean_distances, axis=-1)   # sum along last axis to get average distance per token
    # print(f"mean_distances after sum \t\t\t{mean_distances[0][0][:]}")     #mean_distances.shape (4, 12, 196)
                                                                            #mean_distances[0][0][:].shape (196,)
                                                                            # np.mean(mean_distances[0][0][:]) 4.092669845300465
    mean_distances = np.mean(mean_distances, axis=-1)  # now average across all the tokens
    # print(f"mean_distances after intra head mean \t\t\t{mean_distances[0][:]}") 
# mean_distances[:][:]
# array([[4.09266985e+00, 2.79655516e+01, 4.92462419e+01, 7.80558368e+01,
#         6.10430431e-01, 3.37945888e+01, 3.20314235e+01, 7.09147495e+01,
#         3.32613894e+01, 1.15330825e+02, 4.65250868e+01, 1.08756315e+02],
#        [4.24969365e+00, 8.19954375e+01, 6.00211046e+01, 9.97932106e+01,
#         8.20995802e-03, 5.94220744e+01, 5.59052359e+01, 1.01406131e+02,
#         6.66451660e+01, 1.14905680e+02, 1.15605957e+02, 1.06149762e+02],
#        [2.16769696e+00, 4.11639613e+01, 5.17882398e+01, 5.73646012e+01,
#         1.01820320e+00, 3.60785480e+01, 4.13529155e+01, 8.24575647e+01,
#         3.95146253e+01, 1.12879481e+02, 5.28098888e+01, 1.04353818e+02],
#        [3.47455246e+00, 8.52578273e+01, 7.37828362e+01, 9.19375742e+01,
#         1.44176390e+00, 6.31774218e+01, 6.94482065e+01, 8.42286030e+01,
#         6.85798091e+01, 1.06148779e+02, 9.22238656e+00, 9.24141029e+01]])
    mean_distances = np.mean(mean_distances, axis=0)
    # print(f"mean_distances after batch mean \t\t\t{mean_distances[:]}")    
    
    # 배치안에서 평균이 되었음
    return mean_distances #(12,0)


# MAD들을 시각화하는 함수. 입력: 인수, 모든 MAD들, 저장 경로, 출력: 그래프
# Visualization
def visualize_mads(args, all_mads, save_path=None):
    num_blocks = len(all_mads)
    num_heads = len(all_mads[0])  # Get the number of heads from the first block
    plt.figure(figsize=(10, 6))
    
    # 색상 팔레트 생성 (헤드 수에 따라 0~1 사이의 균일한 간격으로 생성)
    # 색상은 RGBA채널
    colors = plt.cm.tab20(np.linspace(0, 1, num_heads))
    # colors = plt.cm.jet(np.linspace(0, 1, num_heads))
    
    # print(all_mads)    
    # [[ 36.69031965  72.49219277   9.54662151]
    #  [ 27.51730782  67.84144572  28.13950712]
    #  [ 42.35682873  28.80423458  63.88725415]
    #  [ 60.54626757  99.77967492  47.16004414]
    #  [ 61.33741689  90.95886426  79.36172727]
    #  [ 57.26353915  85.71266112  50.1735485 ]
    #  [ 91.45550277  78.20184476  84.46877513]
    #  [ 82.64462066  84.21794416  77.38795059]
    #  [115.04350376 126.69304523 123.44919803]
    #  [139.6749332  144.74889937 138.11682429]
    #  [154.62736542 157.11991205 148.65057951]
    #  [150.65509769 149.02569553 145.35579439]]

    for idx in range(num_blocks):
        one_block_data = all_mads[idx]
        print(one_block_data)    

        # Calculate mean and variance for the one_block_data
        mean = np.mean(one_block_data)
        variance = np.var(one_block_data)
        
        # Rest of the existing code
        min = np.min(one_block_data)
        max = np.max(one_block_data)
        median = np.median(one_block_data)
        
        #x 좌표(블럭 인덱스)를 가져오기 위한 리스트 생성 
        x = [idx] * len(one_block_data)
        for head_idx, y_val in enumerate(one_block_data):
            plt.scatter(x[head_idx], y_val, color=colors[head_idx], alpha=1, label=f"Head {head_idx}" if idx == 0 else "")

        # plt.scatter(x=x, y=y, label=f"Block {idx}", alpha=0.6)
        print("=" * 20)
        print(f"block {idx}\n")
        print(f"min: {min}\n")
        print(f"max: {max}\n")
        print(f"mean: {mean}\n")
        print(f"median: {median}\n")
        print(f"variance: {variance}\n")
        print()

    plt.xlabel("Block Index")
    plt.ylabel("MAD")
    plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0))
    plt.title(args.backbone+"_"+args.mode, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


# 메인 함수. 입력: 인수, 출력: 그래프 및 결과 파일
# Main
def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset(args.disc_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.mode == 'timm':
        # timm model
        model = timm.create_model(args.backbone, pretrained=True, num_classes=0)

    elif args.mode == 'pt':
        # pretraining model
        model = timm.create_model(args.backbone, pretrained=False, num_classes=0)
        state = torch.load(args.pt_model_state, map_location=torch.device('cpu'))
        # new_state_dict = OrderedDict((k.replace("backbone.", ""), v) for k, v in state['model'].items())
        model.load_state_dict(state, strict=True)

    elif args.mode == 'ft':
        # fine-tuning model
        model = timm.create_model(args.backbone, pretrained=False, num_classes=0)
        state = torch.load(args.ft_model_state, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict((k.replace("model.backbone.", ""), v) for k, v in state['state_dict'].items())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        raise ValueError("Invalid mode. Choose from ['timm', 'pt', 'ft']")

    model.eval()
    
    # Check if multiple GPUs are available and wrap the model with DataParallel
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)

    # # Send the model to GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.eval()
    
    # from collections import Counter
    #배치 처리 - 한 스텝마다 들어감    
    all_mads_each_step = []
    for idx, (step, image_path) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
        avg_mads_per_batch = []  # Initialize as an empty list
        # 여기 확인
        # print(f"image_path is \t\t{image_path}")
        for block_idx in range(12):
            feature_extractor = create_feature_extractor(
                model, return_nodes=[f'blocks.{block_idx}.attn.softmax'],
                tracer_kwargs={'leaf_modules': [timm.models.layers.PatchEmbed]})
            with torch.no_grad():
                out = feature_extractor(step) #dict{'blocks.0.attn.softmax': tensor([[[[3.1426e-0...7e-03]]]])} 
            #여기를 tensor로 바꿔도 될듯
            # batch (batch, 3, 224, 224 )-> block 당 
            # 배치당 처리됨
            attention_scores = out[f'blocks.{block_idx}.attn.softmax'].numpy() # out['blocks.0.attn.softmax'].shape = (4, 12, 197, 197)
            mad = compute_mean_attention_dist(16, attention_scores) # (12,0) (n_head, 0)
            # 한 블럭안에서 배치들의 헤드들의 평균 mean attn dist(평균의 평균의 평균) (배치안에서의 헤드들의 평균 <- 한 헤드안에서 패치들의 평균 <- 한 패치의 어텐션 거리 평균)
            # mad =모든 배치, 한 블럭안에서, 모든 헤드들 평균 값
            # all_mads[f"block_{N}"].appned(mad) #all_mads  'block_0': [array([ 94.55144187,...46090959])] 'block_1': [array([ 77.37947706,...45892484])]
            avg_mads_per_batch.append(mad) #(blk, head)
        # avg_mads_per_batch = compute_mean_attention_dist_per_batch(idx, step, model) #avg_mads_per_batch - (12, 3)
        # 블럭마다
        # avg_mads_per_batch = list, [array(3,1),array(3,1),array(3,1),array(3,1) - 12개의 블럭개수대로있음] 
        all_mads_each_step.append(np.array(avg_mads_per_batch))  #all_mads - (step, num_enc_blk=12, num_heads = 3)
        
        #step 별 저장re
        # save_path = f"./result_{args.backbone}_{args.disc_path.split('/')[-2]}_{image_path[0].split('/')[-1].split('.')[-2]}_step_{idx}.png"
        save_path = f"./result/sample/result_{args.backbone}_{args.mode}_{args.disc_path.split('/')[-2]}_step_{idx}.png"
        # save_path = f"./result_{args.backbone}_{args.disc_path.split('/')[-2]}_{str(idx)}.png"
        visualize_mads(args, np.array(avg_mads_per_batch), save_path)

    # print(len(all_mads_each_step))
    # print(np.array(all_mads_each_step))
    avg_mads_all_step = np.mean(all_mads_each_step, axis=0) #avg_mads_all_step = (12,3)    
        
    save_path = f"./result/sample/result_{args.backbone}_{args.mode}_{args.disc_path.split('/')[-2]}_fianl.png"
    # save_path = f"./result_{args.backbone}_{args.disc_path.split('/')[-2]}_{str(idx)}.png"
    visualize_mads(args, avg_mads_all_step, save_path)

# 스크립트 실행시 메인 함수 호출
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--disc_path", required=True, type=str, help="Path to the directory containing images.")
    parser.add_argument("--backbone", default="vit_base_patch16_224.dino", type=str)
    parser.add_argument("--dims", default=768, type=int)
    parser.add_argument("--mode",  default='ft', type=str, choices=['timm', 'pt', 'ft'], required=True, help="Choose the model mode.")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--pt_model_state", default="/hdd/wi/isc2021/models/dino_vitbase16_pretrain.pth", type=str)  # checkpoint
    parser.add_argument("--ft_model_state", default="/hdd/wi/sscd-copy-detection/ckpt/dino/lightning_logs/version_0/checkpoints/epoch=49-step=19499.ckpt", type=str)  # checkpoint
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--size", default=224, type=int, help="Image size for inference")
    args = parser.parse_args()

    main(args)
