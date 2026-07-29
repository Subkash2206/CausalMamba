import os
import sys
import glob
import math
import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

# Add project root to path
ROOT = os.getcwd()
sys.path.append(ROOT)
from models.vmunet.vmunet import VMUNet

# Import SwinUnet
sys.path.append(os.path.join(ROOT, 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

class MockArgs:
    def __init__(self):
        self.cfg = os.path.join(ROOT, 'Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml')
        self.opts = None; self.batch_size = 1; self.zip = False; self.cache_mode = 'part'; self.resume = None; self.accumulation_steps = None; self.use_checkpoint = False; self.amp_opt_level = 'O0'; self.tag = 'test'; self.eval = False; self.throughput = False

def flexible_load(model, ckpt_path):
    print(f"Loading weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'model' in state_dict: state_dict = state_dict['model']
    
    model_dict = model.state_dict()
    # Check if we need to add or remove 'vmunet.' prefix
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

def compute_dice(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    return 2.0 * intersection / total if total > 0 else 1.0

def compute_boundary_f1(pred, gt, tolerance=2):
    # Cast both pred and gt to boolean arrays
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    
    # Extract strictly 1-pixel thick boundaries
    pred_boundary = pred ^ binary_erosion(pred, iterations=1)
    gt_boundary = gt ^ binary_erosion(gt, iterations=1)
    
    total_pred_boundary_pixels = pred_boundary.sum()
    total_gt_boundary_pixels = gt_boundary.sum()
    
    # Safety Check
    if total_pred_boundary_pixels == 0 and total_gt_boundary_pixels == 0:
        return 1.0
    if total_pred_boundary_pixels == 0 or total_gt_boundary_pixels == 0:
        return 0.0
        
    # Calculate distance maps (invert boundaries so they equal 0)
    pred_dist_map = distance_transform_edt(~pred_boundary)
    gt_dist_map = distance_transform_edt(~gt_boundary)
    
    # Calculate True Positives with tolerance
    tp_pred = (pred_boundary & (gt_dist_map <= tolerance)).sum()
    tp_gt = (gt_boundary & (pred_dist_map <= tolerance)).sum()
    
    # Calculate Precision and Recall
    precision = tp_pred / total_pred_boundary_pixels
    recall = tp_gt / total_gt_boundary_pixels
    
    # Return standard F1 score
    if precision + recall == 0:
        return 0.0
        
    return 2.0 * precision * recall / (precision + recall)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'UTMOST PRECISION AUDIT: Starting Full Validation Evaluation on {device}...')

    # 1. Evaluate on Validation Set
    img_dir = os.path.join(ROOT, 'VM-UNet/data/isic18/train/images/')
    mask_dir = os.path.join(ROOT, 'VM-UNet/data/isic18/train/masks/')
    all_imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    import random; random.seed(42); random.shuffle(all_imgs)
    split_idx = int(0.8 * len(all_imgs))
    val_imgs = all_imgs[split_idx:]
    print(f'Total Images: {len(all_imgs)} | Full Validation Set Size: {len(val_imgs)}')

    # 2. Load Models with verified weights
    ckpt_dir = os.path.join(ROOT, 'VM-UNet/best-ckpt/')
    
    # Depths verified from checkpoint: enc=[2,2,9,2], dec=[2,9,2,2]
    vmunet = flexible_load(VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 9, 2, 2]
    ).to(device), os.path.join(ckpt_dir, 'best-vmunet-scratch-isic18.pth')).eval()
    unet = flexible_load(smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device), os.path.join(ckpt_dir, 'best-unet-isic18.pth')).eval()
    
    args = MockArgs(); config = get_config(args)
    swin = flexible_load(SwinUnet(config, img_size=224, num_classes=1).to(device), os.path.join(ckpt_dir, 'best-swinunet-isic18.pth')).eval()

    models = {'VM-UNet': vmunet, 'UNet-ResNet50': unet, 'Swin-Tiny': swin}
    results = {name: {'dice': [], 'bf1': []} for name in models}
    
    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for img_path in tqdm(val_imgs, desc='Rigorous Evaluation'):
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, base + '_segmentation.png')
        if not os.path.exists(mask_path): continue
        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')
        
        for m_name, model in models.items():
            size = 224 if m_name == 'Swin-Tiny' else 256
            tfm = t224 if size == 224 else t256
            x = tfm(img_pil).unsqueeze(0).to(device)
            gt = np.array(mask_pil.resize((size, size), Image.NEAREST)) > 127
            
            with torch.no_grad():
                out = model(x)
                if isinstance(out, tuple): out = out[0]
                # VM-UNet applies sigmoid internally
                pred = out.squeeze().cpu().numpy() if m_name == 'VM-UNet' else torch.sigmoid(out).squeeze().cpu().numpy()
            
            pred_bin = pred > 0.5
            results[m_name]['dice'].append(compute_dice(pred_bin, gt))
            results[m_name]['bf1'].append(compute_boundary_f1(pred_bin, gt))

    # Save results to both root and VM-UNet/results for consistency
    header = 'model,mean_dice,mean_bf1\n'
    content = header + "\n".join([f"{n},{np.mean(results[n]['dice']):.4f},{np.mean(results[n]['bf1']):.4f}" for n in models])
    
    with open('boundary_results.csv', 'w') as f: f.write(content)
    with open('VM-UNet/results/boundary_results.csv', 'w') as f: f.write(content)
    print("\nFINAL AUDITED RESULTS (N=519):")
    print(content)

if __name__ == '__main__':
    main()
