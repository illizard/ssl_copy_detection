import torch
import timm
from timm.models.layers import PatchEmbed
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sscd.models.model import Model
from sscd.lib.util import call_using_args, parse_bool
import argparse
from collections import OrderedDict

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
        return image


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
    
    mean_distances = np.sum(mean_distances, axis=-1)   # sum along last axis to get average distance per token
    mean_distances = np.mean(mean_distances, axis=-1)  # now average across all the tokens
    mean_distances = np.mean(mean_distances, axis=0)
    # 배치안에서 평균이 되었음
    return mean_distances #(12,0)


# 평균 주의 거리를 계산하는 함수. 입력: 패치 크기, 주의 가중치, 출력: 평균 거리
# Compute MAD for multiple images and visualize
# batch (batch, 3, 224, 224 )-> block 당 
# 배치당 처리됨
def compute_mean_attention_dist_per_batch(idx, step, model):
    
    avg_mads_per_batch = []  # Initialize as an empty list
    # 여기 확인
    for N in range(12):
        feature_extractor = create_feature_extractor(
            model, return_nodes=[f'blocks.{N}.attn.softmax'],
            tracer_kwargs={'leaf_modules': [timm.models.layers.PatchEmbed]})
        
        with torch.no_grad():
            out = feature_extractor(step) #dict{'blocks.0.attn.softmax': tensor([[[[3.1426e-0...7e-03]]]])} 
        
        #여기를 tensor로 바꿔도 될듯
        attention_scores = out[f'blocks.{N}.attn.softmax'].numpy() # out['blocks.0.attn.softmax'].shape = (4, 12, 197, 197)
        mad = compute_mean_attention_dist(16, attention_scores) # (12,0) (n_head, 0)
        # 한 블럭안에서 배치들의 헤드들의 평균 mean attn dist(평균의 평균의 평균) (배치안에서의 헤드들의 평균 <- 한 헤드안에서 패치들의 평균 <- 한 패치의 어텐션 거리 평균)
        # mad = 한 블럭안에서 모든 헤드들
        # all_mads[f"block_{N}"].appned(mad) #all_mads  'block_0': [array([ 94.55144187,...46090959])] 'block_1': [array([ 77.37947706,...45892484])]
        avg_mads_per_batch.append(mad) #(blk, head)
    avg_mads_per_batch = np.array(avg_mads_per_batch)
    # 각 스텝에서 나온 결과 즉, 배치들끼리의 평균을 구해야함
    # Calculate average MADs for all steps
    return avg_mads_per_batch


# MAD들을 시각화하는 함수. 입력: 인수, 모든 MAD들, 저장 경로, 출력: 그래프
# Visualization
def visualize_mads(args, all_mads, save_path=None):
    num_blocks = len(all_mads)
    plt.figure(figsize=(10, 6))

    for idx in range(num_blocks):
        block_data = all_mads[idx]
        
        # Calculate mean and variance for the block_data
        mean = np.mean(block_data)
        variance = np.var(block_data)
        
        # Rest of the existing code
        min = np.min(block_data)
        max = np.max(block_data)
        median = np.median(block_data)
        
        x = [idx] * block_data.shape[0]
        y = block_data
        plt.scatter(x=x, y=y, label=f"Block {idx} Mean Distance", alpha=0.6)
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
    plt.legend(loc="lower right")
    plt.title(args.model_state, fontsize=14)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# 메인 함수. 입력: 인수, 출력: 그래프 및 결과 파일
# Main
def main(args):
    # images = load_images_as_tensors(args.disc_path) #[20]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset(args.disc_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # model = timm.create_model(args.backbone, pretrained=False, num_classes=0)
    # state = torch.load(args.model_state, map_location=torch.device('cpu'))
    # new_state_dict = OrderedDict((k.replace("model.backbone.", ""), v) for k, v in state['state_dict'].items())
    # model.load_state_dict(new_state_dict, strict=True)
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    model.eval()
    # from collections import Counter
    #배치 처리 - 한 스텝마다 들어감    
    all_mads = []
    # final_dict = {}
 
    for idx, step in enumerate(dataloader):
        avg_mads_per_batch = compute_mean_attention_dist_per_batch(idx, step, model) #avg_mads_per_batch - (12, 3)
        # 블럭마다 
        all_mads.append(avg_mads_per_batch)  #all_mads - (step, 12, 3)
    all_mads = np.array(all_mads)
    avg_mads_all_step = np.mean(all_mads, axis=0) #avg_mads_overall = (12,3)    
    
    save_path = f"./result_{args.backbone}_{args.disc_path.split('/')[-2]}.png"
    visualize_mads(args, avg_mads_all_step, save_path)

# 스크립트 실행시 메인 함수 호출
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--disc_path", required=True, type=str, help="Path to the directory containing images.")
    parser.add_argument("--backbone", default="vit_tiny_patch16_224.augreg_in21k_ft_in1k", type=str)
    parser.add_argument("--dims", default=192, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--model_state")  # checkpoint
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--size", default=224, type=int, help="Image size for inference")
    args = parser.parse_args()

    main(args)
