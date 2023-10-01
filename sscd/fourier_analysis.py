import timm
import torch
import torch.nn as nn
import copy
import requests
import torch
import numpy as np

from PIL import Image
from einops import rearrange, reduce, repeat
import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import argparse
from collections import OrderedDict



class PatchEmbed(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)
        
    def forward(self, x, **kwargs):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        return x


class Residual(nn.Module):
    def __init__(self, *fn):
        super().__init__()
        self.fn = nn.Sequential(*fn)
        
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
    
class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)


def flatten(xs_list):
    return [x for xs in xs_list for x in xs]

def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))


def make_segments(x, y):  # make segment for `plot_segment`
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_segment(marker, ax, xs, ys, cmap_name="plasma"):  # plot with cmap segments
    z = np.linspace(0.0, 1.0, len(ys))
    z = np.asarray(z)
    
    # cmap = cm.get_cmap(cmap_name)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(0.0, 1.0)
    segments = make_segments(xs, ys)
    lc = LineCollection(segments, array=z, cmap=cmap_name, norm=norm,
                        linewidth=2.5, alpha=1.0)
    ax.add_collection(lc)

    colors = [cmap(x) for x in xs]
    ax.scatter(xs, ys, color=colors, marker=marker, zorder=100)
    # ax.scatter(xs, ys, color=colors, zorder=100)

def main(args):
    model = timm.create_model(args.backbone, pretrained=False, num_classes=0)
    state = torch.load(args.ft_model_state, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict((k.replace("model.backbone.", ""), v) for k, v in state['state_dict'].items())
    model.load_state_dict(new_state_dict, strict=True)

    
    # `blocks` is a sequence of blocks
    blocks = [
        PatchEmbed(model),
        *flatten([[Residual(b.norm1, b.attn), Residual(b.norm2, b.mlp)] 
                for b in model.blocks]),
        nn.Sequential(model.norm, Lambda(lambda x: x[:, 0]), model.head),
    ]

    # This cell build off https://github.com/facebookresearch/mae


    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    # load a sample ImageNet-1K image -- use the full val dataset for precise results
    xs = [
        "https://user-images.githubusercontent.com/930317/158025258-e9a5a454-99de-4d22-bc93-b217cdf06abb.jpeg",
    ]
    xs = [Image.open(requests.get(x, stream=True).raw) for x in xs]
    xs = [x.resize((224, 224)) for x in xs]
    xs = [np.array(x) / 255. for x in xs]
    xs = np.stack(xs)

    assert xs.shape[1:] == (224, 224, 3)

    # normalize by ImageNet mean and std
    xs = xs - imagenet_mean
    xs = xs / imagenet_std
    xs = rearrange(torch.tensor(xs, dtype=torch.float32), "b h w c -> b c h w")

    # accumulate `latents` by collecting hidden states of a model
    latents = []
    with torch.no_grad():
        for block in blocks:
            xs = block(xs)
            latents.append(xs)
            
    if args.backbone in [args.backbone, "pit_ti_224"]:  # for ViT: Drop CLS token
        latents = [latent[:,1:] for latent in latents]
    latents = latents[:-1]  # drop logit (output)


    # Fourier transform feature maps
    fourier_latents = []
    for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
        latent = latent.cpu()
        
        if len(latent.shape) == 3:  # for ViT
            b, n, c = latent.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        elif len(latent.shape) == 4:  # for CNN
            b, c, h, w = latent.shape
        else:
            raise Exception("shape: %s" % str(latent.shape))
        latent = fourier(latent)
        latent = shift(latent).mean(dim=(0, 1))
        latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
        latent = latent - latent[0]  # visualize 'relative' log amplitudes 
                                    # (i.e., low-freq amp - high freq amp)
        fourier_latents.append(latent)
        

    # A. Plot Fig 2a: "Relative log amplitudes of Fourier transformed feature maps"
    fig, ax1 = plt.subplots(1, 1, figsize=(3.3, 4), dpi=150)
    for i, latent in enumerate(reversed(fourier_latents[:-1])):
        freq = np.linspace(0, 1, len(latent))
        ax1.plot(freq, latent, color=cm.plasma_r(i / len(fourier_latents)))
        
    ax1.set_xlim(left=0, right=1)

    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("$\Delta$ Log amplitude")

    from matplotlib.ticker import FormatStrFormatter
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))
    save_path_1 = f"./result/sample/result_{args.backbone}_fourier_analysis_1.png"
    plt.savefig(save_path_1, bbox_inches='tight')


    # B. Plot Fig 8: "Relative log amplitudes of high-frequency feature maps"
    if args.backbone == "resnet50":  # for ResNet-50
        pools = [4, 8, 14]
        msas = []
        marker = "D"
    elif args.backbone == "vit_tiny_patch16_224":  # for ViT-Ti
        pools = []
        msas = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,]
        marker = "o"
    elif args.backbone == "vit_base_patch16_224.dino":  # for ViT-Base
        pools = []  # Assuming no pooling layers for ViT-Base, similar to ViT-Ti
        # You can adjust the indices for multi-head self-attention based on the architecture of ViT-Base
        msas = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,]  
        marker = "o"  # Using circle shape for plotting, similar to ViT-Ti
    elif args.backbone == "vit_base_patch16_224.mae":  # for ViT-Base
        pools = []  # Assuming no pooling layers for ViT-Base, similar to ViT-Ti
        # You can adjust the indices for multi-head self-attention based on the architecture of ViT-Base
        msas = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,]  
        marker = "o"  # Using circle shape for plotting, similar to ViT-Ti
    else:
        import warnings
        warnings.warn("The configuration for %s are not implemented." % args.backbone, Warning)
        pools, msas = [], []
        marker = "s"

    depths = range(len(fourier_latents))

    # Normalize
    depth = len(depths) - 1
    depths = (np.array(depths)) / depth
    pools = (np.array(pools)) / depth
    msas = (np.array(msas)) / depth

    fig, ax2 = plt.subplots(1, 1, figsize=(6.5, 4), dpi=120)
    plot_segment(marker, ax2, depths, [latent[-1] for latent in fourier_latents])  # high-frequency component

    for pool in pools:
        ax2.axvspan(pool - 1.0 / depth, pool + 0.0 / depth, color="tab:blue", alpha=0.15, lw=0)
    for msa in msas:
        ax2.axvspan(msa - 1.0 / depth, msa + 0.0 / depth, color="tab:gray", alpha=0.15, lw=0)
        
    ax2.set_xlabel("Normalized depth")
    ax2.set_ylabel("$\Delta$ Log amplitude")
    ax2.set_xlim(0.0, 1.0)

    from matplotlib.ticker import FormatStrFormatter
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    save_path = f"./result/sample/result_{args.backbone}_fourier_analysis.png"
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vit_base_patch16_224.dino", type=str)
    parser.add_argument("--ft_model_state", default="/hdd/wi/sscd-copy-detection/ckpt/dino/lightning_logs/version_0/checkpoints/epoch=49-step=19499.ckpt", type=str)  # checkpoint
    args = parser.parse_args()

    main(args)