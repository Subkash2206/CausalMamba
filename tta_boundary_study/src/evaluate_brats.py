import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.brats_dataset import BraTSDataset
from monai.networks.nets import SegResNet
from src.metrics.boundary_metrics import dice_score, boundary_f1, hausdorff_95
from src.tta.tta_inference import tta_predict_3d, baseline_predict_3d

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print(f"Loading 3D SegResNet to {DEVICE}...")
    model = SegResNet(in_channels=4, out_channels=1, init_filters=16).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/segresnet_brats_best.pth', map_location=DEVICE))
    model.eval()
    
    val_ds = BraTSDataset('brats2021', split='val')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    results = []
    
    for imgs, masks in tqdm(val_loader, desc='Evaluating 3D Volumes'):
        # Extract ground truth to [H, W, D] numpy array
        mask_np = masks[0, 0].cpu().numpy()
        
        base_prob = baseline_predict_3d(model, imgs, DEVICE)
        tta_prob = tta_predict_3d(model, imgs, DEVICE, n_views=8)
        
        res_dict = {
            'baseline_dice': dice_score(base_prob, mask_np),
            'tta_dice': dice_score(tta_prob, mask_np),
            'baseline_bf1': boundary_f1(base_prob, mask_np, thickness=1),
            'tta_bf1': boundary_f1(tta_prob, mask_np, thickness=1),
            'baseline_hd95': hausdorff_95(base_prob, mask_np),
            'tta_hd95': hausdorff_95(tta_prob, mask_np),
        }
        
        res_dict['dice_delta'] = res_dict['tta_dice'] - res_dict['baseline_dice']
        res_dict['bf1_delta'] = res_dict['tta_bf1'] - res_dict['baseline_bf1']
        
        results.append(res_dict)
        
    df = pd.DataFrame(results)
    df.to_csv('results/brats_results.csv', index=False)
    
    print("\n=== FINAL 3D BRATS RESULTS (MEAN) ===")
    print(f"Baseline Dice: {df['baseline_dice'].mean():.4f}")
    print(f"TTA Dice:      {df['tta_dice'].mean():.4f}")
    print(f"Dice Delta:    {df['dice_delta'].mean():.4f}\n")
    
    print(f"Baseline BF1:  {df['baseline_bf1'].mean():.4f}")
    print(f"TTA BF1:       {df['tta_bf1'].mean():.4f}")
    print(f"BF1 Delta:     {df['bf1_delta'].mean():.4f}\n")
    
    print(f"Baseline HD95: {df['baseline_hd95'].dropna().mean():.4f}")
    print(f"TTA HD95:      {df['tta_hd95'].dropna().mean():.4f}")

if __name__ == '__main__':
    main()
