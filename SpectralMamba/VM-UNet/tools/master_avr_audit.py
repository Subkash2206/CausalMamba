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

# Add paths for VM-UNet and Swin-UNet
sys.path.append(os.path.join(os.getcwd(), 'VM-UNet'))
from models.vmunet.vmunet import VMUNet

sys.path.append(os.path.join(os.getcwd(), '..', 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

class MockArgs:
    def __init__(self):
        self.cfg = '../Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml'
        self.opts = None
        self.batch_size = 1
        self.zip = False
        self.cache_mode = 'part'
        self.resume = None
        self.accumulation_steps = None
        self.use_checkpoint = False
        self.amp_opt_level = 'O0'
        self.tag = 'test'
        self.eval = False
        self.throughput = False

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
    print(f'Starting Spectral Audit on {device}...')

    # 1. Load Models
    print('Loading Models...')
    
    # UNet
    unet = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device)
    unet.load_state_dict(torch.load('best-ckpt/best-unet-isic18.pth', map_location=device))
    unet.eval()
    
    # Swin-UNet
    args = MockArgs()
    config = get_config(args)
    swin = SwinUnet(config, img_size=224, num_classes=1).to(device)
    swin.load_state_dict(torch.load('best-ckpt/best-swinunet-isic18.pth', map_location=device))
    swin.eval()
    
    # VM-UNet (Mamba)
    vmunet = VMUNet().to(device)
    vmunet.load_state_dict(torch.load('best-ckpt/best-vmunet-scratch-isic18.pth', map_location=device), strict=True)
    vmunet.eval()

    # 2. Setup Hooks
    features = {}
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                features[name] = output[0].detach()
            else:
                features[name] = output.detach()
        return hook

    # Hook Encoders (Stage 1, 2, 3, 4)
    # UNet ResNet50 Stages
    unet.encoder.layer1.register_forward_hook(get_hook('unet_s1'))
    unet.encoder.layer2.register_forward_hook(get_hook('unet_s2'))
    unet.encoder.layer3.register_forward_hook(get_hook('unet_s3'))
    unet.encoder.layer4.register_forward_hook(get_hook('unet_s4'))
    
    # Swin Stages
    swin.swin_unet.layers[0].register_forward_hook(get_hook('swin_s1'))
    swin.swin_unet.layers[1].register_forward_hook(get_hook('swin_s2'))
    swin.swin_unet.layers[2].register_forward_hook(get_hook('swin_s3'))
    swin.swin_unet.layers[3].register_forward_hook(get_hook('swin_s4'))
    
    # Mamba Stages (vmunet.vmunet.layers)
    vmunet.vmunet.layers[0].register_forward_hook(get_hook('mamba_s1'))
    vmunet.vmunet.layers[1].register_forward_hook(get_hook('mamba_s2'))
    vmunet.vmunet.layers[2].register_forward_hook(get_hook('mamba_s3'))
    vmunet.vmunet.layers[3].register_forward_hook(get_hook('mamba_s4'))

    # 3. Data
    img_dir = 'data/isic18/train/images/'
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    
    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    audit_results = defaultdict(list)

    print('Auditing Spectral Signatures...')
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        i256 = t256(img).unsqueeze(0).to(device)
        i224 = t224(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _ = unet(i256)
            audit_results['unet_s1'].append(compute_avr(features['unet_s1']))
            audit_results['unet_s2'].append(compute_avr(features['unet_s2']))
            audit_results['unet_s3'].append(compute_avr(features['unet_s3']))
            audit_results['unet_s4'].append(compute_avr(features['unet_s4']))
            
            _ = swin(i224)
            # Swin outputs can be (B, L, C) or (B, H, W, C)
            for s in range(1, 5):
                f = features[f'swin_s{s}']
                if f.dim() == 3:
                    B, L, C = f.shape; H = W = int(np.sqrt(L)); f = f.transpose(1, 2).reshape(B, C, H, W)
                elif f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]:
                    f = f.permute(0, 3, 1, 2)
                audit_results[f'swin_s{s}'].append(compute_avr(f))
                
            _ = vmunet(i256)
            # Mamba outputs (B, C, H, W) or (B, H, W, C)
            for s in range(1, 5):
                f = features[f'mamba_s{s}']
                if f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]:
                    f = f.permute(0, 3, 1, 2)
                audit_results[f'mamba_s{s}'].append(compute_avr(f))

    # 4. Report
    print('\n' + '='*80)
    print(f'{"Architecture":<20} | {"Stage 1":<10} | {"Stage 2":<10} | {"Stage 3":<10} | {"Stage 4":<10}')
    print('-'*80)
    
    models = [('UNet (CNN)', 'unet'), ('Swin (Trans)', 'swin'), ('VM-UNet (Mamba)', 'mamba')]
    for label, prefix in models:
        s1 = np.mean(audit_results[f'{prefix}_s1'])
        s2 = np.mean(audit_results[f'{prefix}_s2'])
        s3 = np.mean(audit_results[f'{prefix}_s3'])
        s4 = np.mean(audit_results[f'{prefix}_s4'])
        print(f'{label:<20} | {s1:.4f}     | {s2:.4f}     | {s3:.4f}     | {s4:.4f}')
    print('='*80)

if __name__ == '__main__':
    main()
