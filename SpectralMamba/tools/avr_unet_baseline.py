import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import glob
from collections import defaultdict
import segmentation_models_pytorch as smp

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. Initialize SMP UNet model with ResNet50 encoder
    print('Initializing UNet (ResNet50 encoder)...')
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        in_channels=3, 
        classes=1
    ).to(device)
    
    model.eval()

    # 3. Register forward hooks on Encoder layers
    features = {}
    def get_hook(name):
        def hook(module, input, output):
            # Capture output
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # ResNet output is (B, C, H, W)
            features[name] = out.detach().cpu()
        return hook

    print('Registering hooks on encoder layers...')
    # Attach hooks to the main stages of the ResNet50 encoder
    target_layers = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
    hook_count = 0
    layer_names = []
    
    for name, module in model.encoder.named_children():
        if name in target_layers:
            hook_name = f'encoder_{name}'
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
    print('\nAverage Alias Volume Ratio (AVR) across 50 images (UNet-ResNet50):')
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
