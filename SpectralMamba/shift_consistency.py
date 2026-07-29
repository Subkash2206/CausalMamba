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

# Add paths for models
sys.path.append(os.getcwd())
from models.vmunet.vmunet import VMUNet
sys.path.append('Swin-Unet')
from config import get_config
from networks.vision_transformer import SwinUnet

class MockArgs:
    def __init__(self):
        self.cfg = 'Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml'
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

def calculate_iou(pred, target):
    # Check if inputs already look like probabilities (0 to 1)
    if pred.min() >= 0 and pred.max() <= 1:
        p = (pred > 0.5).float()
    else:
        p = (torch.sigmoid(pred) > 0.5).float()
        
    if target.min() >= 0 and target.max() <= 1:
        t = (target > 0.5).float()
    else:
        t = (torch.sigmoid(target) > 0.5).float()
        
    intersection = (p * t).sum()
    union = p.sum() + t.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running Shift Consistency Audit on {device}...')

    ckpt_dir = 'VM-UNet/best-ckpt/'
    
    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print("Loading models...")
    unet = flexible_load(smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device), os.path.join(ckpt_dir, 'best-unet-isic18.pth')).eval()
    
    args = MockArgs(); config = get_config(args); swin = flexible_load(SwinUnet(config, img_size=224, num_classes=1).to(device), os.path.join(ckpt_dir, 'best-swinunet-isic18.pth')).eval()
    
    vmunet = flexible_load(VMUNet().to(device), os.path.join(ckpt_dir, 'best-vmunet-scratch-isic18.pth')).eval()

    ROOT = os.getcwd()
    img_dir = os.path.join(ROOT, 'VM-UNet/data/isic18/train/images/')
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    import random; random.seed(42); random.shuffle(img_paths)
    split_idx = int(0.8 * len(img_paths))
    val_imgs = img_paths[split_idx:] # Full validation set
    
    shifts = [1, 2, 3, 4, 5]
    results = defaultdict(lambda: defaultdict(list))

    print(f'Auditing {len(val_imgs)} images for shift consistency...')
    for idx, path in enumerate(val_imgs):
        if idx % 10 == 0:
            print(f"Processing image {idx}/{len(val_imgs)}...")
            
        img = Image.open(path).convert('RGB')
        i256 = t256(img).unsqueeze(0).to(device)
        i224 = t224(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Baselines
            base_unet = unet(i256)
            base_swin = swin(i224)
            base_mamba = vmunet(i256)
            
            for s in shifts:
                # Shift inputs horizontally (dim=-1 for W)
                shifted_i256 = torch.roll(i256, shifts=s, dims=-1)
                shifted_i224 = torch.roll(i224, shifts=s, dims=-1)
                
                # Inference on shifted
                pred_unet = unet(shifted_i256)
                pred_swin = swin(shifted_i224)
                pred_mamba = vmunet(shifted_i256)
                
                # Shift prediction back to compare with baseline
                pred_unet_unshifted = torch.roll(pred_unet, shifts=-s, dims=-1)
                pred_swin_unshifted = torch.roll(pred_swin, shifts=-s, dims=-1)
                pred_mamba_unshifted = torch.roll(pred_mamba, shifts=-s, dims=-1)
                
                results['UNet'][s].append(calculate_iou(base_unet, pred_unet_unshifted))
                results['Swin'][s].append(calculate_iou(base_swin, pred_swin_unshifted))
                results['Mamba'][s].append(calculate_iou(base_mamba, pred_mamba_unshifted))

    final_data = []
    print('\n' + '='*60)
    print(f'{"Model":<15} | {"Shift Amount":<15} | {"Mean IoU":<10}')
    print('-'*60)
    for model_name in ['UNet', 'Swin', 'Mamba']:
        for s in shifts:
            mean_iou = np.mean(results[model_name][s])
            print(f'{model_name:<15} | {s:<15} | {mean_iou:.4f}')
            final_data.append(f'{model_name},{s},{mean_iou}')
            
    print('='*60)
    
    out_path = 'VM-UNet/results/shift_consistency_results.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('model,shift,mean_iou\n')
        f.write('\n'.join(final_data) + '\n')
    print(f'Saved results to {out_path}')

if __name__ == '__main__':
    main()
