import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import glob
from collections import defaultdict
import timm
import math

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. Initialize Swin model
    print('Initializing Swin-Tiny (features_only=True)...')
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True).to(device)
    model.eval()

    # 3. Register forward hooks
    features = {}
    def get_hook(name):
        def hook(module, input, output):
            # Capture output
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
                
            # Swin layers might output (B, L, C) or (B, H, W, C) or (B, C, H, W)
            if out.dim() == 3:
                B, L, C = out.shape
                H = W = int(math.sqrt(L))
                out = out.view(B, H, W, C).permute(0, 3, 1, 2)
            elif out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768]:
                out = out.permute(0, 3, 1, 2)
                
            features[name] = out.detach().cpu()
        return hook

    print('Registering hooks on Swin stages...')
    hook_count = 0
    layer_names = []
    
    # In timm Swin models, the stages are usually under `model.layers`
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            hook_name = f'swin_stage_{i}'
            layer.register_forward_hook(get_hook(hook_name))
            layer_names.append(hook_name)
            hook_count += 1
    print(f'Registered {hook_count} hooks.')

    # 4. Load 50 images
    img_dir = 'data/isic18/train/images/'
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    print(f'Found {len(img_paths)} images for analysis.')
    
    # Swin requires 224x224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    avr_sums = defaultdict(float)
    shapes = {}

    # 5. Run inference and compute AVR
    print('Running inference and computing AVR...')
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Because features_only=True, this returns a list of feature maps directly
            # We can use these directly or use the hooked features.
            # Using the returned features guarantees they are properly formatted (B, C, H, W) by timm's FeatureInfo
            ret_features = model(input_tensor)
            
        # If hooks didn't register or we want to use the cleaner timm output:
        if hook_count == 0 or len(features) == 0:
            for j, fmap in enumerate(ret_features):
                name = f'swin_stage_{j}'
                if name not in layer_names:
                    layer_names.append(name)
                # Ensure shape is (B, C, H, W)
                if fmap.dim() == 4 and fmap.shape[-1] in [96, 192, 384, 768]:
                    fmap = fmap.permute(0, 3, 1, 2)
                elif fmap.dim() == 3:
                    # In case of (B, L, C)
                    B, L, C = fmap.shape
                    H_f = W_f = int(math.sqrt(L))
                    fmap = fmap.view(B, H_f, W_f, C).permute(0, 3, 1, 2)
                    
                features[name] = fmap.detach().cpu()
        else:
            # Optionally overwrite with ret_features to ensure shape is (B, C, H, W)
            for j, fmap in enumerate(ret_features):
                name = f'swin_stage_{j}'
                if fmap.dim() == 4 and fmap.shape[-1] in [96, 192, 384, 768]:
                    fmap = fmap.permute(0, 3, 1, 2)
                features[name] = fmap.detach().cpu()

        for name in layer_names:
            fmap = features[name] # shape (B, C, H, W)
            B, C, H, W = fmap.shape
            shapes[name] = fmap.shape
            
            # Compute 2D FFT
            fft = torch.fft.fft2(fmap)
            fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
            
            # Compute power spectrum
            power = torch.abs(fft_shifted) ** 2
            
            # Define Nyquist mask (> H/4 and > W/4 from center)
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device='cpu').view(1, 1, H, 1)
            x = torch.arange(W, device='cpu').view(1, 1, 1, W)
            
            mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4)
            mask = mask.expand(B, C, H, W)
            
            high_freq_energy = (power * mask).sum()
            total_energy = power.sum()
            
            avr = (high_freq_energy / total_energy).item() if total_energy > 0 else 0.0
            avr_sums[name] += avr

    # 6. Average and Print Table
    print('\nAverage Alias Volume Ratio (AVR) across 50 images (Swin-Tiny):')
    print('-' * 80)
    print(f'{"Layer Name":<40} | {"Resolution":<15} | {"Mean AVR":<10}')
    print('-' * 80)
    for name in layer_names:
        mean_avr = avr_sums[name] / len(img_paths)
        res_str = f'{shapes[name][2]}x{shapes[name][3]}'
        print(f'{name:<40} | {res_str:<15} | {mean_avr:.4f}')
    print('-' * 80)

if __name__ == '__main__':
    main()
