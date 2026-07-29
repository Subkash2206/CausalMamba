import sys
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import segmentation_models_pytorch as smp

sys.path.append(os.path.join(os.getcwd(), 'VM-UNet'))
from models.vmunet.vmunet import VMUNet
sys.path.append(os.path.join(os.getcwd(), 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

class MockArgs:
    def __init__(self):
        self.cfg = 'Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml'
        self.opts = None; self.batch_size = 1; self.zip = False; self.cache_mode = 'part'; self.resume = None; self.accumulation_steps = None; self.use_checkpoint = False; self.amp_opt_level = 'O0'; self.tag = 'test'; self.eval = False; self.throughput = False

def compute_avr(fmap):
    fmap = fmap.cpu().float()
    fmap = fmap - fmap.mean(dim=(-2, -1), keepdim=True)
    B, C, H, W = fmap.shape
    fft = torch.fft.fft2(fmap); fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1)); power = torch.abs(fft_shifted) ** 2
    cy, cx = H // 2, W // 2; y = torch.arange(H).view(1, 1, H, 1); x = torch.arange(W).view(1, 1, 1, W)
    mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4); mask = mask.expand(B, C, H, W)
    return (power * mask).sum().item() / power.sum().item() if power.sum() > 0 else 0.0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Starting Unified Stage-wise Spectral Audit (Matched Resolutions) on {device}...')
    ckpt_dir = 'VM-UNet/best-ckpt/'
    
    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    unet = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device)
    unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-unet-isic18.pth'), map_location=device)); unet.eval()
    
    args = MockArgs(); config = get_config(args); swin = SwinUnet(config, img_size=224, num_classes=1).to(device)
    swin.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-swinunet-isic18.pth'), map_location=device)); swin.eval()
    
    vmunet = VMUNet().to(device)
    vmunet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-vmunet-scratch-isic18.pth'), map_location=device), strict=True); vmunet.eval()

    features = {}
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple): features[name] = output[0].detach()
            else: features[name] = output.detach()
        return hook

    for i in range(1, 5):
        # UNet: layer1 to layer4
        getattr(unet.encoder, f'layer{i}').register_forward_hook(get_hook(f'unet_res{i}'))
        
        # Swin: output of the last block in each layer (before downsample)
        swin.swin_unet.layers[i-1].blocks[-1].register_forward_hook(get_hook(f'swin_res{i}'))
        
        # Mamba: output of the last block in each layer (before downsample)
        vmunet.vmunet.layers[i-1].blocks[-1].register_forward_hook(get_hook(f'mamba_res{i}'))

    img_dir = 'VM-UNet/data/isic18/train/images/'
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    import random; random.seed(42); random.shuffle(img_paths)
    split_idx = int(0.8 * len(img_paths))
    val_imgs = img_paths[split_idx:] # Full validation split
    
    audit_results = defaultdict(list); stage_info = {}
    print(f'Auditing {len(val_imgs)} VALIDATION images...')
    for path in val_imgs:
        img = Image.open(path).convert('RGB'); i256 = t256(img).unsqueeze(0).to(device); i224 = t224(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = unet(i256)
            for i in range(1, 5):
                f = features[f'unet_res{i}']; audit_results[f'UNet_res{i}'].append(compute_avr(f)); stage_info[f'UNet_res{i}'] = f'{f.shape[2]}x{f.shape[3]}'
            
            _ = swin(i224)
            for i in range(1, 5):
                f = features[f'swin_res{i}']
                # Swin blocks output might be (B, L, C) or (B, H, W, C)
                if f.dim() == 3:
                    B, L, C = f.shape; H = W = int(np.sqrt(L)); f = f.transpose(1, 2).reshape(B, C, H, W)
                elif f.dim() == 4:
                    if f.shape[-1] > f.shape[1]:
                        f = f.permute(0, 3, 1, 2)
                    B, C, H, W = f.shape
                else:
                    B, C, H, W = f.shape
                audit_results[f'Swin_res{i}'].append(compute_avr(f)); stage_info[f'Swin_res{i}'] = f'{H}x{W}'
                
            _ = vmunet(i256)
            for i in range(1, 5):
                f = features[f'mamba_res{i}']
                # Mamba block output is typically (B, H, W, C)
                if f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]: f = f.permute(0, 3, 1, 2)
                audit_results[f'Mamba_res{i}'].append(compute_avr(f)); stage_info[f'Mamba_res{i}'] = f'{f.shape[2]}x{f.shape[3]}'

    final_data = []
    print('\n' + '='*80); print(f'{"Model":<15} | {"Resolution Level":<18} | {"Actual Res":<12} | {"Mean AVR":<10}'); print('-'*80)
    for model_name in ['UNet', 'Swin', 'Mamba']:
        for i in range(1, 5):
            key = f'{model_name}_res{i}'; mean_avr = np.mean(audit_results[key]); res = stage_info[key]
            print(f'{model_name:<15} | Level {i:<12} | {res:<12} | {mean_avr:.4f}')
            final_data.append(f"{model_name},{i},{res},{mean_avr}")
    print('='*80)
    
    with open('VM-UNet/results/avr_stagewise_results_matched.csv', 'w') as f:
        f.write("model,res_level,resolution,mean_avr\n")
        f.write("\n".join(final_data) + "\n")
        
    print(f'Saved matched resolution results to VM-UNet/results/avr_stagewise_results_matched.csv')

if __name__ == '__main__':
    main()
