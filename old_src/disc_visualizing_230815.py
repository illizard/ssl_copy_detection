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


def load_images_as_tensors(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset(args.disc_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    images = []
    for batch in dataloader:
        images.append(batch)
    return torch.cat(images, dim=0)


# 거리 행렬을 계산하는 함수. 입력: 패치 크기, 패치 수, 길이, 출력: 거리 행렬
# Compute Distance Matrix
def compute_distance_matrix(patch_size, num_patches, length):
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])
    return distance_matrix

# 평균 주의 거리를 계산하는 함수. 입력: 패치 크기, 주의 가중치, 출력: 평균 거리
# Compute MAD
def compute_mean_attention_dist(patch_size, attention_weights):
    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[..., 1:, 1:]
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    
    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(mean_distances, axis=-1)
    mean_distances = np.mean(mean_distances, axis=-1)
    return mean_distances

# 모든 주의 점수로부터 MAD들을 수집하는 함수. 입력: 주의 점수들, 패치 크기, 출력: 모든 MAD들
# Gather MADs
def gather_mads(attention_scores, patch_size: int = 16):
    all_mean_distances = {
        f"block_{i}_mean_dist": compute_mean_attention_dist(
            patch_size=patch_size, attention_weights=attention_weight
        )
        for i, attention_weight in enumerate(attention_scores)
    }
    return all_mean_distances

# MAD들을 시각화하는 함수. 입력: 인수, 모든 MAD들, 저장 경로, 출력: 그래프
# Visualization
def visualize_mads(args, all_mads, save_path=None):
    num_blocks = len(all_mads)
    plt.figure(figsize=(10, 6))

    for idx in range(num_blocks):
        mean_distance = all_mads[f"block_{idx}"][0]
        x = [idx] * mean_distance.shape[0]
        y = mean_distance
        plt.scatter(x=x, y=y, label=f"Block {idx}")

    plt.xlabel("Block Index")
    plt.ylabel("MAD")
    plt.legend(loc="lower right")  # 오른쪽 아래로 수정
    plt.title(args.model_state, fontsize=14)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# 평균 주의 거리를 계산하는 함수. 입력: 패치 크기, 주의 가중치, 출력: 평균 거리
# Compute MAD for multiple images and visualize
def compute_mean_attention_dist_for_images(images, model):
    all_mads = {f"block_{i}": [] for i in range(12)}

    for image in images:
        image = image.unsqueeze(0)
        for N in range(12):
            feature_extractor = create_feature_extractor(
                model, return_nodes=[f'blocks.{N}.attn.softmax'],
                tracer_kwargs={'leaf_modules': [timm.models.layers.PatchEmbed]})
            
            with torch.no_grad():
                out = feature_extractor(image)
            
            attention_scores = out[f'blocks.{N}.attn.softmax'].numpy()
            mad = compute_mean_attention_dist(16, attention_scores)
            all_mads[f"block_{N}"].append(mad)

    # Calculate average MADs for all images
    avg_mads = {key: np.mean(value, axis=0) for key, value in all_mads.items()}
    return avg_mads

# 메인 함수. 입력: 인수, 출력: 그래프 및 결과 파일
# Main
def main(args):
    images = load_images_as_tensors(args) #[20]

    model = timm.create_model(args.backbone, pretrained=False, num_classes=0)
    state = torch.load(args.model_state, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict((k.replace("model.backbone.", ""), v) for k, v in state['state_dict'].items())
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    avg_mads = compute_mean_attention_dist_for_images(images, model)

    save_path = f"./result_{args.backbone}_{args.disc_path.split('/')[-2]}_gpt.png"
    visualize_mads(args, avg_mads, save_path)

# 스크립트 실행시 메인 함수 호출
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--disc_path", required=True, type=str, help="Path to the directory containing images.")
    parser.add_argument("--backbone", default="vit_base_patch16_224.dino", type=str)
    parser.add_argument("--dims", default=768, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model_state")  # checkpoint
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--size", default=224, type=int, help="Image size for inference")
    args = parser.parse_args()

    main(args)
