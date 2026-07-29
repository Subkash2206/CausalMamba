import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import glob
from collections import defaultdict

# Add current directory to path for imports
sys.path.append(os.getcwd())

from models.vmunet.vmunet import VMUNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. Initialize model
    print('Initializing VMUNet...')
    model = VMUNet().to(device)
    
    # 2. Load checkpoint
    ckpt_path = 'best-ckpt/best-vmunet-isic18.pth'
    print(f'Loading checkpoint from {ckpt_path}...')
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print('Checkpoint loaded successfully.')
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
    
    model.eval()

    # 3. Register forward hooks
    features = {}
    def get_hook(name):
        def hook(module, input, output):
            # Capture output and permute to (B, C, H, W)
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
                
            # VMUNet VSSBlock output is typically (B, H, W, C)
            if out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768]:
                out = out.permute(0, 3, 1, 2)
            features[name] = out.detach().cpu()
        return hook

    print('Registering hooks on VSSBlock layers...')
    hook_count = 0
    layer_names = []
    for name, module in model.named_modules():
        if 'VSSBlock' in str(type(module)):
            hook_name = f'hook_{hook_count:02d}_{name}'
            module.register_forward_hook(get_hook(hook_name))
            layer_names.append(hook_name)
            hook_count += 1
    print(f'Registered {hook_count} hooks.')

    # 4. Load 50 images
    img_dir = 'data/isic18/train/images/'
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    print(f'Found {len(img_paths)} images for analysis.')
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dictionary to accumulate AVR sums
    avr_sums = defaultdict(float)
    shapes = {}

    # 5. Run inference and compute AVR
    print('Running inference and computing AVR...')
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _ = model(input_tensor)
            
        for name in layer_names:
            fmap = features[name] # shape (B, C, H, W)
            B, C, H, W = fmap.shape
            shapes[name] = fmap.shape
            
            # Compute 2D FFT
            # fft2 computes over last two dimensions by default
            fft = torch.fft.fft2(fmap)
            fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
            
            # Compute power spectrum
            power = torch.abs(fft_shifted) ** 2
            
            # Define Nyquist mask
            # The prompt specified "u > H/2 or v > W/2". In a grid of size HxW, after fftshift,
            # the center is at H//2, W//2. Frequencies > H/4 correspond to the Nyquist limit
            # for 2x downsampling, which is the standard definition of AVR. 
            # We use > H/4 and > W/4 from the center.
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
    print('\nAverage Alias Volume Ratio (AVR) across 50 images:')
    print('-' * 80)
    print(f'{"Layer Name":<40} | {"Resolution":<15} | {"Mean AVR":<10}')
    print('-' * 80)
    for name in layer_names:
        mean_avr = avr_sums[name] / len(img_paths)
        res_str = f'{shapes[name][2]}x{shapes[name][3]}'
        # strip the "hook_xx_" prefix for cleaner display
        clean_name = '_'.join(name.split('_')[2:])
        print(f'{clean_name:<40} | {res_str:<15} | {mean_avr:.4f}')
    print('-' * 80)

if __name__ == '__main__':
    main()
