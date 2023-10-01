# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#230804
from collections import OrderedDict 

import argparse
import enum
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d
from classy_vision.models import build_model
from .gem_pooling import GlobalGeMPool2d

from sscd.models import mae_vit
from sscd.models import dino_vit
from sscd.models import dtop_vit

import timm

class Implementation(enum.Enum):
    CLASSY_VISION = enum.auto()
    TORCHVISION = enum.auto()
    TORCHVISION_ISC = enum.auto()
    OFFICIAL = enum.auto() ##230708##
    MOBILE = enum.auto() ##230708##
    MY = enum.auto() ##230925##
    
class Backbone(enum.Enum):
    CV_RESNET18 = ("resnet18", 512, Implementation.CLASSY_VISION)
    CV_RESNET50 = ("resnet50", 2048, Implementation.CLASSY_VISION)
    CV_RESNEXT101 = ("resnext101_32x4d", 2048, Implementation.CLASSY_VISION)

    TV_RESNET18 = (resnet18, 512, Implementation.TORCHVISION)
    TV_RESNET50 = (resnet50, 2048, Implementation.TORCHVISION)
    TV_RESNEXT101 = (resnext101_32x8d, 2048, Implementation.TORCHVISION)
    
    MULTI_RESNET50 = ("multigrain_resnet50", 2048, Implementation.TORCHVISION_ISC)

    OFFL_VIT = ('vit_patch_16_base', 768, Implementation.OFFICIAL)     ##230831#    
    OFFL_VIT_TINY = ('vit_patch_16_tiny', 192, Implementation.OFFICIAL)     ##230831#    
    OFFL_DINO = ('dino_patch_16_base', 768, Implementation.OFFICIAL)     ##230708##
    OFFL_MAE = ('mae_patch_16_base', 768, Implementation.OFFICIAL)  
    OFFL_MOBVIT = ('mobilevit_xxs', 192, Implementation.MOBILE)
    MY_DTOP_VIT = ('dtop_vit_tiny', 192, Implementation.MY)  
  
    # dino = ('dino_patch_16_base', 768, Implementation.OFFICIAL)     ##230708##
    
    def build(self, dims: int):
        impl = self.value[2]
        # print(self.value) #('resnet50', 2048, <Implementation.CLASSY_VISION: 1>)
        if impl == Implementation.CLASSY_VISION:
            model = build_model({"name": self.value[0]})
            # Remove head exec wrapper, which we don't need, and breaks pickling
            # (needed for spawn dataloaders).
            return model.classy_model
        if impl == Implementation.TORCHVISION:
            return self.value[0](num_classes=dims, zero_init_residual=True)
        # multigrain 230804
        if impl == Implementation.TORCHVISION_ISC:
            model = resnet50(pretrained=False)
            st = torch.load("/hdd/wi/isc2021/models/multigrain_joint_3B_0.5.pth")
            state_dict = OrderedDict([
                (name[9:], v)
                for name, v in st["model_state"].items() if name.startswith("features.")
            ])
            model.avgpool = nn.Identity()     
            model.fc = nn.Identity()
            # model.avgpool = None # None으로 하면 forward에서 호출하는 게 none이어서 
            # model.fc = None
            model.load_state_dict(state_dict, strict=True)
            return model

        if impl == Implementation.OFFICIAL: #### modi 0722 ###
            if self.value[0] == "vit_patch_16_base":
                model = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=False, num_classes=0)
                return model            
            elif self.value[0] == "vit_patch_16_tiny":
                model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=0)
                return model            
            elif self.value[0] == "dino_patch_16_base":
                model = dino_vit.__dict__['vit_base'](patch_size=16, num_classes=0)
                ckpt = torch.load("/hdd/wi/isc2021/models/dino_vitbase16_pretrain.pth", map_location=torch.device('cpu'))
                # new_ckpt = OrderedDict(("backbone."+k, v) for k, v in ckpt.items())
                # model.load_state_dict(ckpt, strict=True)
                print(model)
                print(f"===="*30)
                print(f"current state is")
                print(f"{ckpt.keys()}\n")
                print(f"===="*30)
                print(f"Model {self.value[0]} built.")
                return model
            
            elif self.value[0] == "mae_patch_16_base":
                model = mae_vit.__dict__['vit_base_patch16'](num_classes=0, drop_path_rate=0.0) # global_pool=args.global_pool)
                ckpt = torch.load("/hdd/wi/isc2021/models/mae_pretrain_vit_base.pth", map_location=torch.device('cpu'))
                model.load_state_dict(ckpt['model'], strict=True)
                print(f"===="*30)
                print(f"current state is")
                print(f"{ckpt['model'].keys()}\n")
                print(f"===="*30)
                print(f"Model {self.value[0]} built.")
                return model

        if impl == Implementation.MOBILE:
            model = timm.create_model("mobilevit_xxs", num_classes=0, pretrained=True)
            return model

        if impl == Implementation.MY:
            if self.value[0] == "dtop_vit_tiny":
                model = dtop_vit.DeepTokenPooling().cuda()
                # MY_DTOP_VIT = ('dtop_vit_tiny', 192, Implementation.MY)  

            return model
                
        else:
            raise AssertionError("Unsupported OFFICIAL model: %s" % (self.value[0]))
        

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)


class Model(nn.Module):
    def __init__(self, backbone: str, dims: int, pool_param: float):
        super().__init__()
        self.backbone_type = Backbone[backbone] # self.backbone_type = <Backbone.CV_RESNET50>
                                                #<Backbone.CV_RESNET50: ('resnet50', 2048, <Implementation.CLASSY_VISION: 1>)>
                                                #Backbone <enum 'Backbone'> // 'CV_RESNET50'
        print(f"self.backbone_type is {self.backbone_type}")
        self.backbone = self.backbone_type.build(dims=dims)
        impl = self.backbone_type.value[2]
        if impl == Implementation.CLASSY_VISION:
            self.embeddings = nn.Sequential(
                GlobalGeMPool2d(pool_param),
                nn.Linear(self.backbone_type.value[1], dims),
                L2Norm(),
            )
        elif impl == Implementation.TORCHVISION:
            if pool_param > 1:
                self.backbone.avgpool = GlobalGeMPool2d(pooling_param=3.0)
                fc = self.backbone.fc
                nn.init.xavier_uniform_(fc.weight)
                nn.init.constant_(fc.bias, 0)
            self.embeddings = L2Norm()
            # self.embeddings = nn.Identity()
        
        ## MODIFIED 230804##
        elif impl == Implementation.TORCHVISION_ISC:
            if pool_param > 1:
                self.backbone.avgpool = GlobalGeMPool2d(pooling_param=3.0)
                self.backbone.fc = nn.Linear(self.backbone_type.value[1], dims)
            self.embeddings = L2Norm()
        #classy vision은 모델에 pooling, fc없는데 torch vision이랑 pooling이 avg로 달려있음
            # self.embeddings = nn.Sequential(
            #     GlobalGeMPool2d(pooling_param=3.0),
            #     L2Norm(),
            # )
        # ValueError: not enough values to unpack (expected 4, got 2) 왜 이런 에러가 나는지 도대체 모르겟네
        ## MODIFIED 230724##
        elif impl == Implementation.OFFICIAL:
            if self.backbone_type.value[0] == "vit_patch_16_base":
                self.embeddings = L2Norm() 
            elif self.backbone_type.value[0] == "vit_patch_16_tiny":
                self.embeddings = L2Norm()    
            elif self.backbone_type.value[0] == "dino_patch_16_base":
                self.embeddings = L2Norm()    
            elif self.backbone_type.value[0] == "mae_patch_16_base":
                self.backbone.head_drop = nn.Identity()
                self.embeddings = L2Norm()
            # cls token // patch embbdig 정하기            

        elif impl == Implementation.MOBILE:
            self.embeddings = L2Norm()
        
        elif impl == Implementation.MY:
            self.embeddings = L2Norm()
            
    def forward(self, x):
        x = self.backbone(x)
        # print(f"x.shape is {x.shape}")

        # return x
        return self.embeddings(x)
        
        
        # print(x[0].shape) # DINO = x.shape = (1,768) //MAE = x.shape = tuple((1,768), (1,768))
           
        # if self.impl == Implementation.OFFICIAL:
        # if self.backbone_type.value[0] == "dino_patch_16_base":
        #     feats = self.backbone.get_intermediate_layers(x, n=1)[0].clone()  # size: (batch_size, num_patches + 1, embed_dim) | 마지막 블록에서 [CLS] 토큰과 패치 임베딩 리턴
        #     cls_output_token = feats[:, 0, :] #  [CLS] token
        
        # ## patch_tokens = feats[:, 1:, :]
        # ## GeM with exponent 4 for output patch tokens
        # # b, h, w, d = x.shape[0], x.shape[-2] // self.backbone.patch_embed.patch_size, x.shape[-1] // self.backbone.patch_embed.patch_size, feats.shape[-1] # GeM pooling with exponent 4을 patch tokens에 적용하고, cls tokens와 연결
        # # feats = feats[:, 1:, :].view(b, h, w, d)
        # # feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
        # # feats = nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1. / 4).view(b, -1)
        
        # elif self.backbone_type.value[0] == "mae_patch_16_base":
        #     # print(f"x.shape is \t\t {x.shape}")
        #     # cls_output_token = self.backbone.forward_features(x) # 모델 안에 이미 클래스 토큰만 나오게 돼있음
        #     cls_output_token = self.backbone.forward_features(x)[0].clone() #패치임베딩까지 같이 꺼낼때 사용
        # else:
        #     raise AssertionError("Unsupported model: %s" % (self.backbone_type.value[0]))

        # cls_tokens = self.embeddings(cls_output_token) # cls token만 나옴 #self.backbone(x).shape torch.Size([256, 768]) | shape is (256, 768)
        # # patch_tokens = self.embeddings(feats)
        
        # # tokens = torch.cat([cls_tokens, patch_tokens], dim=1)  # size: (batch_size, 2 * embed_dim)
        # return cls_tokens # cls 토큰과 + 패치 임베딩 반환            




    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser = parser.add_argument_group("Model")
        parser.add_argument(
            "--backbone", default="TV_RESNET50", choices=[b.name for b in Backbone]
        )
        parser.add_argument("--dims", default=512, type=int)
        parser.add_argument("--pool_param", default=3, type=float)
        # parser.add_argument("--feature", default='cls', type=str, choices=['cls', 'patch', 'cls_plus_patch'])