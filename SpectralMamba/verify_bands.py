import sys
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import segmentation_models_pytorch as smp

# Add paths for models
sys.path.append(os.getcwd())
from models.vmunet.vmunet import VMUNet

def compute_bands(fmap):
    fmap = fmap.cpu().float()
    fmap = fmap - fmap.mean(dim=(-2, -1), keepdim=True)
    B, C, H, W = fmap.shape
    fft = torch.fft.fft2(fmap)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = torch.abs(fft_shifted) ** 2
    cy, cx = H // 2, W // 2
    y = torch.arange(H).view(1, 1, H, 1)
    x = torch.arange(W).view(1, 1, 1, W)
    dist_y = torch.abs(y - cy) / (H / 2)
    dist_x = torch.abs(x - cx) / (W / 2)
    freq_ratio = torch.max(dist_y.expand(B, C, H, W), dist_x.expand(B, C, H, W))
    mask_low = freq_ratio <= 0.25
    mask_mid = (freq_ratio > 0.25) & (freq_ratio <= 0.75)
    mask_high = freq_ratio > 0.75
    total = power.sum().item()
    if total == 0: return 0., 0., 0.
    return (power * mask_low).sum().item() / total, (power * mask_mid).sum().item() / total, (power * mask_high).sum().item() / total

device = 'cuda'
ckpt_dir = 'best-ckpt/'
tfm = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

unet = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device)
unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-unet-isic18.pth')))
unet.eval()

vmunet = VMUNet().to(device)
vmunet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-vmunet-scratch-isic18.pth')), strict=True)
vmunet.eval()

img_path = sorted(glob.glob('data/isic18/train/images/*.jpg'))[0]
img_pil = Image.open(img_path).convert('RGB')
i256 = tfm(img_pil).unsqueeze(0).to(device)

feats = {}
def hook(m, i, o): feats['f'] = o.detach()

unet.encoder.layer1.register_forward_hook(hook)
unet(i256)
l_u, m_u, h_u = compute_bands(feats['f'])

vmunet.vmunet.layers[0].blocks[-1].register_forward_hook(hook)
vmunet(i256)
f_m = feats['f']
if f_m.dim() == 4 and f_m.shape[-1] in [96, 192, 384, 768]:
    f_m = f_m.permute(0, 3, 1, 2)
l_m, m_m, h_m = compute_bands(f_m)

print(f'UNet Level 1: Low={l_u:.4f}, Mid={m_u:.4f}, High={h_u:.4f}')
print(f'Mamba Level 1: Low={l_m:.4f}, Mid={m_m:.4f}, High={h_m:.4f}')
