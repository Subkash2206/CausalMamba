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

# Add paths for VM-UNet and Swin-UNet (Synchronized with run_band_only.py)
sys.path.append(os.getcwd())
from models.vmunet.vmunet import VMUNet

sys.path.append(os.path.join(os.getcwd(), 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

class MockArgs:
    def __init__(self):
        self.cfg = os.path.join(os.getcwd(), 'Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml')
        self.opts = None; self.batch_size = 1; self.zip = False; self.cache_mode = 'part'
        self.resume = None; self.accumulation_steps = None; self.use_checkpoint = False
        self.amp_opt_level = 'O0'; self.tag = 'test'; self.eval = False; self.throughput = False

def flexible_load(model, ckpt_path):
    print(f"Loading weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'model' in state_dict: state_dict = state_dict['model']
    model_dict = model.state_dict()
    has_vmunet_prefix = any(k.startswith('vmunet.') for k in state_dict.keys())
    model_has_vmunet_prefix = any(k.startswith('vmunet.') for k in model_dict.keys())
    
    # Filter metadata and enforce strict loading
    clean_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('total_ops') and not k.endswith('total_params')}
    new_state_dict = {}
    for k, v in clean_state_dict.items():
        new_k = k
        if has_vmunet_prefix and not model_has_vmunet_prefix: new_k = k.replace('vmunet.', '')
        elif not has_vmunet_prefix and model_has_vmunet_prefix: new_k = 'vmunet.' + k
        new_state_dict[new_k] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    print("  SUCCESS: Model loaded with strict=True.")
    return model

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
    ckpt_dir = os.path.join(os.getcwd(), 'VM-UNet/best-ckpt/')
    
    # UNet
    unet = flexible_load(smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device), os.path.join(ckpt_dir, 'best-unet-isic18.pth')).eval()
    
    # Swin-UNet
    args = MockArgs(); config = get_config(args); swin = flexible_load(SwinUnet(config, img_size=224, num_classes=1).to(device), os.path.join(ckpt_dir, 'best-swinunet-isic18.pth')).eval()
    
    # Depths verified from checkpoint: enc=[2,2,9,2], dec=[2,9,2,2]
    vmunet = flexible_load(VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 9, 2, 2]
    ).to(device), os.path.join(ckpt_dir, 'best-vmunet-scratch-isic18.pth')).eval()

    # 2. Setup Hooks
    features = {}
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple): features[name] = output[0].detach()
            else: features[name] = output.detach()
        return hook

    # Hook Encoders
    unet.encoder.layer1.register_forward_hook(get_hook('unet_s1'))
    unet.encoder.layer2.register_forward_hook(get_hook('unet_s2'))
    unet.encoder.layer3.register_forward_hook(get_hook('unet_s3'))
    unet.encoder.layer4.register_forward_hook(get_hook('unet_s4'))
    
    swin.swin_unet.layers[0].register_forward_hook(get_hook('swin_s1'))
    swin.swin_unet.layers[1].register_forward_hook(get_hook('swin_s2'))
    swin.swin_unet.layers[2].register_forward_hook(get_hook('swin_s3'))
    swin.swin_unet.layers[3].register_forward_hook(get_hook('swin_s4'))
    
    vmunet.vmunet.layers[0].register_forward_hook(get_hook('mamba_s1'))
    vmunet.vmunet.layers[1].register_forward_hook(get_hook('mamba_s2'))
    vmunet.vmunet.layers[2].register_forward_hook(get_hook('mamba_s3'))
    vmunet.vmunet.layers[3].register_forward_hook(get_hook('mamba_s4'))

    # 3. Data Split (Fix Data Leakage)
    img_dir = 'VM-UNet/data/isic18/train/images/'
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    import random; random.seed(42); random.shuffle(img_paths)
    split_idx = int(0.8 * len(img_paths))
    val_imgs = img_paths[split_idx:] # Full validation set
    
    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    audit_results = defaultdict(list)
    print(f'Auditing Spectral Signatures on {len(val_imgs)} validation images...')
    for path in val_imgs:
        img = Image.open(path).convert('RGB')
        i256 = t256(img).unsqueeze(0).to(device); i224 = t224(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _ = unet(i256)
            for s in range(1, 5): audit_results[f'unet_s{s}'].append(compute_avr(features[f'unet_s{s}']))
            
            _ = swin(i224)
            for s in range(1, 5):
                f = features[f'swin_s{s}']
                if f.dim() == 3: B, L, C = f.shape; H = W = int(np.sqrt(L)); f = f.transpose(1, 2).reshape(B, C, H, W)
                elif f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]: f = f.permute(0, 3, 1, 2)
                audit_results[f'swin_s{s}'].append(compute_avr(f))
                
            _ = vmunet(i256)
            for s in range(1, 5):
                f = features[f'mamba_s{s}']
                if f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]: f = f.permute(0, 3, 1, 2)
                audit_results[f'mamba_s{s}'].append(compute_avr(f))

    # 4. Report and Export
    print('\n' + '='*80)
    print(f'{"Architecture":<20} | {"Stage 1":<10} | {"Stage 2":<10} | {"Stage 3":<10} | {"Stage 4":<10}')
    print('-'*80)
    
    csv_data = []
    models = [('UNet', 'unet'), ('Swin', 'swin'), ('Mamba', 'mamba')]
    for label, prefix in models:
        s1 = np.mean(audit_results[f'{prefix}_s1'])
        s2 = np.mean(audit_results[f'{prefix}_s2'])
        s3 = np.mean(audit_results[f'{prefix}_s3'])
        s4 = np.mean(audit_results[f'{prefix}_s4'])
        print(f'{label:<20} | {s1:.4f}     | {s2:.4f}     | {s3:.4f}     | {s4:.4f}')
        csv_data.append(f"{label},{s1},{s2},{s3},{s4}")
    print('='*80)
    
    os.makedirs('VM-UNet/results', exist_ok=True)
    with open('VM-UNet/results/avr_stagewise_results_matched.csv', 'w') as f:
        f.write('model,stage1,stage2,stage3,stage4\n')
        f.write('\n'.join(csv_data) + '\n')
    print('Saved results to VM-UNet/results/avr_stagewise_results_matched.csv')

if __name__ == '__main__':
    main()
