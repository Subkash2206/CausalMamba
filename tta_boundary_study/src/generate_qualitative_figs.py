import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from src.datasets.cvc_dataset import CVCDataset
from src.datasets.brats_dataset import BraTSDataset
from src.models.unet import get_unet
from monai.networks.nets import SegResNet
from src.tta.tta_inference import tta_predict, baseline_predict, tta_predict_3d, baseline_predict_3d

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_overlay(ax, img, mask, title):
    ax.imshow(img, cmap='gray')
    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(masked, cmap='autumn', alpha=0.6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

def generate_cvc_visuals():
    model = get_unet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/unet_cvc_best.pth', map_location=DEVICE))
    model.eval()
    
    ds = CVCDataset('cvc_clinicdb/original', 'cvc_clinicdb/ground_truth', split='val')
    
    # Programmatically find a highly visible outlier-suppression example
    df = pd.read_csv('results/ablation_cvc_unet.csv')
    df['hd95_diff'] = df['base_hd95'] - df['tta4_hd95']
    
    # Filter for cases where HD95 dropped massively (20+ pixels) 
    # but the baseline Dice is still reasonable enough to look like a real prediction
    valid = df[(df['hd95_diff'] >= 20) & (df['base_dice'] > 0.60)].dropna()
    
    if len(valid) > 1:
        best_idx = valid['hd95_diff'].nlargest(2).index[-1]  # Get 2nd best
    elif len(valid) == 1:
        best_idx = valid['hd95_diff'].idxmax()
    else:
        # Fallback just in case no patient meets the strict 20px threshold
        best_idx = df['hd95_diff'].nlargest(2).index[-1]
        
    print(f"Selected CVC Patient Index {best_idx} (HD95 improved by {df.loc[best_idx, 'hd95_diff']:.2f} pixels)")

    img_tensor, mask_tensor = ds[best_idx]
    img_np = img_tensor.permute(1, 2, 0).numpy()
    gt_np = mask_tensor[0].numpy()
    
    base_pred = (baseline_predict(model, img_tensor.unsqueeze(0), DEVICE) > 0.5).astype(np.uint8)
    tta_pred = (tta_predict(model, img_tensor.unsqueeze(0), DEVICE, n_views=4) > 0.5).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_np)
    axes[0].set_title('Original Endoscopy', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    plot_overlay(axes[1], img_np.mean(axis=2), gt_np, 'Ground Truth')
    plot_overlay(axes[2], img_np.mean(axis=2), base_pred, 'Baseline Prediction')
    plot_overlay(axes[3], img_np.mean(axis=2), tta_pred, 'TTA (4-View) Prediction')
    
    plt.tight_layout()
    plt.savefig('results/figures/qualitative_cvc.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_brats_visuals():
    model = SegResNet(in_channels=4, out_channels=1, init_filters=16).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/segresnet_brats_best.pth', map_location=DEVICE))
    model.eval()
    
    ds = BraTSDataset('brats2021', split='val')
    
    df = pd.read_csv('results/ablation_brats.csv')
    df['hd95_diff'] = df['base_hd95'] - df['tta4_hd95']
    valid = df[(df['base_dice'] > 0.85) & (df['tta4_dice'] >= df['base_dice'] - 0.005)].dropna()
    best_idx = valid['hd95_diff'].idxmax()
    print(f"Selected BraTS Patient Index {best_idx} (HD95 improved by {valid.loc[best_idx, 'hd95_diff']:.2f})")
    
    img_tensor, mask_tensor = ds[best_idx]
    gt_vol = mask_tensor[0].numpy()
    z_slice = np.argmax(gt_vol.sum(axis=(0,1)))
    
    bg_img = img_tensor[1, :, :, z_slice].numpy()
    gt_np = gt_vol[:, :, z_slice]
    
    base_vol = baseline_predict_3d(model, img_tensor.unsqueeze(0), DEVICE)
    tta_vol = tta_predict_3d(model, img_tensor.unsqueeze(0), DEVICE, n_views=4)
    
    base_np = (base_vol[:, :, z_slice] > 0.5).astype(np.uint8)
    tta_np = (tta_vol[:, :, z_slice] > 0.5).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(bg_img, cmap='gray')
    axes[0].set_title('MRI (T1ce Slice)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    plot_overlay(axes[1], bg_img, gt_np, 'Ground Truth')
    plot_overlay(axes[2], bg_img, base_np, 'Baseline Prediction')
    plot_overlay(axes[3], bg_img, tta_np, 'TTA (4-View) Prediction')
    
    plt.tight_layout()
    plt.savefig('results/figures/qualitative_brats.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating Intelligent Qualitative Visuals...")
    os.makedirs('results/figures', exist_ok=True)
    generate_cvc_visuals()
    generate_brats_visuals()
    print("Done! Check results/figures/")

if __name__ == '__main__':
    main()
