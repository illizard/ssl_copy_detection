import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import torchextractor as tx

import timm
from timm.models.vision_transformer import VisionTransformer, Block


class IRB(nn.Module):
    def __init__(self, in_channels):
        super(IRB, self).__init__()
        self.expand_ratio = 2
        hidden_dim = round(in_channels * self.expand_ratio)
        self.use_res_connect = self.expand_ratio == 1

        layers = []
        if self.expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


# # ASPP(Atrous Spatial Pyramid Pooling) Module
# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels, num_classes):
#         super(ASPP, self).__init__()
        
#         # 1번 branch = 1x1 convolution → BatchNorm → ReLu
#         self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

#         # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
#         self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
#         self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

#         # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
#         self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
#         self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)
    
#         # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
#         self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
#         self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

#         # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        
#         self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1) # (1280 = 5*256)
#         self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

#         self.conv_1x1_4 = nn.Conv2d(out_channels, num_classes, kernel_size=1)

#     def forward(self, feature_map):
#         # feature map의 shape은 (batch_size, in_channels, height/output_stride, width/output_stride)

#         feature_map_h = feature_map.size()[2] # (== h/16)
#         feature_map_w = feature_map.size()[3] # (== w/16)

#         # 1번 branch = 1x1 convolution → BatchNorm → ReLu
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
#         # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
#         # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
#         # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

#         # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
#         # shape: (batch_size, in_channels, 1, 1)
#         out_img = self.avg_pool(feature_map) 
#         # shape: (batch_size, out_channels, 1, 1)
#         out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

#         # shape: (batch_size, out_channels * 5, height/output_stride, width/output_stride)
#         out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) 
#         # shape: (batch_size, num_classes, height/output_stride, width/output_stride)
#         out = self.conv_1x1_4(out) 

#         return out


# class WaveBlock(nn.Module):
#     def __init__(self, rw=0.3, rh=1):
#         super(WaveBlock, self).__init__()
#         self.rw = rw
#         self.rh = rh

#     def forward(self, x):
#         # x의 shape: [batch, layer, channel, height, width] = [4, 1, 768, 14, 14]
#         B, L, C, H, W = x.size()
        
#         # 높이에 대한 랜덤 값을 생성합니다.
#         X = random.randint(0, int(H * (1 - self.rw)))
        
#         # 랜덤 값에 따라 feature map을 수정합니다.
#         mask = torch.ones_like(x)
#         mask[:, :, :, X:X + int(H * self.rw), :] = self.rh
#         x_wave = x * mask

#         return x_wave

class EnhancedLocalityModule(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedLocalityModule, self).__init__()
        self.irb = IRB(in_channels)
        # self.aspp = ASPP(in_channels, in_channels, in_channels)

    def forward(self, x):
        x = self.irb(x)
        return x

        # return self.aspp(x)



'''
F_c_dim = class token embedding dimension
'''

    
class GlobalBranch(nn.Module):
    def __init__(self, F_c_dim): #torch.Size([4, 12, 768])
        super(GlobalBranch, self).__init__()
        self.fc = nn.Linear(12, 1)
        self.F_c_dim = F_c_dim
        
    def forward(self, F_c):
        x = self.fc(F_c.transpose(1, 2))
        u_c = x.transpose(1, 2) 
        
        return u_c  #torch.Size([4, 1, 768])
        

# Local Branch
class LocalBranch(nn.Module):
    def __init__(self, k, F_p_dim):
        super(LocalBranch, self).__init__()
        self.k = k
        self.wh = 196
        self.out_channels = 1
        self.F_p_dim = F_p_dim
        # self.conv1x1 = nn.Conv2d(self.k, self.out_channels, kernel_size=1)  # Added 1x1 convolution layer.
        self.conv1x1 = nn.Conv3d(self.k, self.out_channels, kernel_size=(1, 1, 1))  # 3D convolution layer
        self.elm = EnhancedLocalityModule(self.F_p_dim)  # Uncommented this line to use ELM.
        
    def forward(self, x): 
        x = x.permute(0, 1, 3, 2).contiguous().view(x.size(0), x.size(1), x.size(-1), int((x.size(2))**0.5), int((x.size(2))**0.5)) # 3d tensor unfold
        x = self.conv1x1(x)  # Apply 1x1 convolution before ELM. #torch.Size([4, 1, 768, 14, 14])
        x = x.squeeze()
        x_og = x.clone()
        elm_x = self.elm(x)  # Apply ELM.
        x = x_og + elm_x # Fuse
        u_p = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).permute(0, 2, 1)
        
        return u_p

      
class DeepTokenPooling(nn.Module):
    def __init__(self):
        super(DeepTokenPooling, self).__init__()    
        # common
        self.num_blocks = 12
        
        # for global branch
        self.F_c_dim = 192
        self.N = self.F_c_dim * 2 # final descriptor dim (global + local) 
        self.k = 12        
        
        # Initialize self.features as None. We will allocate it dynamically.
        self.features = None
        # Initialize model
        # self.vit = timm.create_model("vit_tiny_patch16_224.augreg_in21k", num_classes=0, pretrained=True)
        self.global_branch = GlobalBranch(self.F_c_dim)
        self.local_branch = LocalBranch(self.k, self.F_c_dim)
        self.hooks = []
        self.block_names = [f"blocks.{i}" for i in range(12)]
        self.vit_backbone = tx.Extractor(timm.create_model("vit_tiny_patch16_224.augreg_in21k", num_classes=0, pretrained=True), self.block_names)
        self.last_fc_layer = nn.Linear(self.N, self.F_c_dim)  # 차원 축소 레이어 추가



    def forward(self, x):
        # x = self.cnn_stem(x)

        # final_layer_feature = self.vit_backbone(x)  # Using the VisionTransformer model
        # pos = self.dpe(x)
        # x = x + pos  # Adding position embeddings to the output of VisionTransformer  
        # print(self.features.shape) #torch.Size([4, 12, 197, 768])
        output, feature = self.vit_backbone(x)
        feature_list = [feature[name] for name in self.block_names]
        features = torch.stack(feature_list, dim=1)  # size: [4, 12, 197, 768]
        
        cls_tokens = features[:, :, 0, :]
        patch_tokens = features[:, :, 1:, :]
        
        global_features = self.global_branch(cls_tokens)
        local_features = self.local_branch(patch_tokens)
        combined_features = torch.cat([global_features, local_features], dim=2).squeeze()
        logit = self.last_fc_layer(combined_features)
    
        return logit
    

def controlRandomness(random_seed=42):

    if random_seed is not None:
        print(f"random seed = {random_seed}")
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # np.random.seed(random_seed)
        random.seed(random_seed)

    else: # random
        print(f"random seed = {random_seed}")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
    controlRandomness()
    # Create the custom model
    model = DeepTokenPooling()
    dummy = torch.randn(4, 3, 224, 224)
    # combined_features, vit_features = model(dummy)
    vit_features = model(dummy)
    print(vit_features.shape)
