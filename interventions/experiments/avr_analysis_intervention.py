"""
Frequency intervention analysis (integrated with SpectralMamba).

This is a copy of avr_analysis.py with minimal modifications to integrate
the FrequencyIntervention framework.

Integration changes are marked with comment lines containing "INTEGRATION".
"""

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

# --- INTEGRATION: Import the existing intervention framework ----------------
# These modules are already implemented and tested independently.
# The interventions package lives one directory above SpectralMamba/.
sys.path.append(os.path.join(os.getcwd(), '..'))  # add project root for interventions/
from interventions.intervention import FrequencyIntervention
# ---------------------------------------------------------------------------

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
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        print('Checkpoint loaded successfully.')
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
    
    model.eval()

    # --- INTEGRATION: Create an identity intervention (all-ones mask) -------
    # Purpose: Verify that introducing the intervention framework does NOT
    # change model predictions beyond floating-point tolerance.
    # An all-ones mask passes every frequency through unchanged.
    identity_mask = lambda h, w, device, dtype: torch.ones(
        1, 1, h, w, device=device, dtype=dtype
    )
    intervention = FrequencyIntervention(identity_mask, check_nan=True)
    print(f'Using identity intervention (all-ones mask).')
    # ------------------------------------------------------------------------

    # 3. Register forward hooks
    # --- INTEGRATION: Use two dicts to disambiguate before/after -------------
    features_before = {}  # original feature BEFORE intervention
    features_after = {}   # modified feature AFTER intervention (fed to graph)
    def get_hook(name, intervention=None):
        def hook(module, input, output):
            # Capture output and permute to (B, C, H, W)
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
                
            # VMUNet VSSBlock output is typically (B, H, W, C)
            if out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768]:
                out = out.permute(0, 3, 1, 2)

            # Store the original (pre-intervention) feature.
            features_before[name] = out.detach().cpu()

            if intervention is not None:
                # Apply the intervention and store the modified version.
                modified = intervention(out)
                features_after[name] = modified.detach().cpu()
                # Return the modified tensor to the forward graph.
                # PyTorch uses the hook's return value as the module output.
                return modified
            else:
                # No intervention: original output flows through unchanged.
                features_after[name] = out.detach().cpu()
        return hook
    # ------------------------------------------------------------------------

    print('Registering hooks on VSSBlock layers...')
    hook_count = 0
    layer_names = []
    for name, module in model.named_modules():
        if 'VSSBlock' in str(type(module)):
            hook_name = f'hook_{hook_count:02d}_{name}'
            # --- INTEGRATION: Pass the intervention to the hook -------------
            module.register_forward_hook(get_hook(hook_name, intervention=intervention))
            # -----------------------------------------------------------------
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

    # Dictionaries to accumulate AVR sums
    avr_sums_before = defaultdict(float)
    avr_sums_after = defaultdict(float)
    shapes = {}

    # 5. Run inference and compute AVR
    print('Running inference and computing AVR...')
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _ = model(input_tensor)
            
        for name in layer_names:
            # --- INTEGRATION: Compute AVR for BOTH before and after ---------
            fmap_before = features_before[name]  # shape (B, C, H, W)
            fmap_after = features_after[name]    # shape (B, C, H, W)
            B, C, H, W = fmap_before.shape
            shapes[name] = fmap_before.shape
            
            # Before-intervention AVR
            fft_b = torch.fft.fft2(fmap_before)
            fft_shifted_b = torch.fft.fftshift(fft_b, dim=(-2, -1))
            power_b = torch.abs(fft_shifted_b) ** 2
            
            # After-intervention AVR
            fft_a = torch.fft.fft2(fmap_after)
            fft_shifted_a = torch.fft.fftshift(fft_a, dim=(-2, -1))
            power_a = torch.abs(fft_shifted_a) ** 2
            
            # Define Nyquist mask (same as original)
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device='cpu').view(1, 1, H, 1)
            x = torch.arange(W, device='cpu').view(1, 1, 1, W)
            
            mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4)
            mask = mask.expand(B, C, H, W)
            
            high_freq_energy_b = (power_b * mask).sum()
            total_energy_b = power_b.sum()
            avr_b = (high_freq_energy_b / total_energy_b).item() if total_energy_b > 0 else 0.0
            avr_sums_before[name] += avr_b

            high_freq_energy_a = (power_a * mask).sum()
            total_energy_a = power_a.sum()
            avr_a = (high_freq_energy_a / total_energy_a).item() if total_energy_a > 0 else 0.0
            avr_sums_after[name] += avr_a
            # -----------------------------------------------------------------

    # 6. Average, compute ΔAVR, and Print Table
    print('\nAVR Comparison (Before vs After Intervention) across 50 images:')
    print('-' * 120)
    print(f'{"Layer Name":<40} | {"Resolution":<15} | {"AVR Before":<12} | {"AVR After":<12} | {"ΔAVR":<10}')
    print('-' * 120)
    for name in layer_names:
        mean_avr_before = avr_sums_before[name] / len(img_paths)
        mean_avr_after = avr_sums_after[name] / len(img_paths)
        delta_avr = mean_avr_after - mean_avr_before
        res_str = f'{shapes[name][2]}x{shapes[name][3]}'
        # strip the "hook_xx_" prefix for cleaner display
        clean_name = '_'.join(name.split('_')[2:])
        print(f'{clean_name:<40} | {res_str:<15} | {mean_avr_before:.6f}   | {mean_avr_after:.6f}   | {delta_avr:+.2e}')
    print('-' * 120)

if __name__ == '__main__':
    main()