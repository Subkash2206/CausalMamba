import torch
import pandas as pd
from tqdm import tqdm
import os

from src.datasets.cvc_dataset import CVCDataset
from src.datasets.brats_dataset import BraTSDataset
from src.models.unet import get_unet
from src.models.swin_unetr_cvc import get_swin_unetr
from monai.networks.nets import SegResNet
from src.metrics.boundary_metrics import dice_score, boundary_f1, hausdorff_95
from src.tta.tta_inference import tta_predict, baseline_predict, tta_predict_3d, baseline_predict_3d

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_2d(model_name, model, val_loader, csv_out):
    print(f"\nRunning Ablation for: {model_name}")
    results = []
    for imgs, masks in tqdm(val_loader, desc=model_name):
        mask_np = masks[0, 0].cpu().numpy()
        
        base = baseline_predict(model, imgs, DEVICE)
        tta2 = tta_predict(model, imgs, DEVICE, n_views=2)
        tta4 = tta_predict(model, imgs, DEVICE, n_views=4)
        tta8 = tta_predict(model, imgs, DEVICE, n_views=8)
        
        preds = {'base': base, 'tta2': tta2, 'tta4': tta4, 'tta8': tta8}
        res = {}
        for k, p in preds.items():
            res[f'{k}_dice'] = dice_score(p, mask_np)
            res[f'{k}_bf1'] = boundary_f1(p, mask_np, thickness=2)
            res[f'{k}_hd95'] = hausdorff_95(p, mask_np)
        results.append(res)
        
    save_and_print(results, csv_out)

def evaluate_3d(model_name, model, val_loader, csv_out):
    print(f"\nRunning Ablation for: {model_name}")
    results = []
    for imgs, masks in tqdm(val_loader, desc=model_name):
        mask_np = masks[0, 0].cpu().numpy()
        
        base = baseline_predict_3d(model, imgs, DEVICE)
        tta2 = tta_predict_3d(model, imgs, DEVICE, n_views=2)
        tta4 = tta_predict_3d(model, imgs, DEVICE, n_views=4)
        tta8 = tta_predict_3d(model, imgs, DEVICE, n_views=8)
        
        preds = {'base': base, 'tta2': tta2, 'tta4': tta4, 'tta8': tta8}
        res = {}
        for k, p in preds.items():
            res[f'{k}_dice'] = dice_score(p, mask_np)
            res[f'{k}_bf1'] = boundary_f1(p, mask_np, thickness=1) # 3D strict tolerance
            res[f'{k}_hd95'] = hausdorff_95(p, mask_np)
        results.append(res)
        
    save_and_print(results, csv_out)

def save_and_print(results, csv_out):
    df = pd.DataFrame(results)
    df.to_csv(csv_out, index=False)
    
    print(f"{'Metric':<10} | {'Base':<8} | {'2-View':<8} | {'4-View':<8} | {'8-View':<8}")
    print("-" * 55)
    print(f"{'Dice':<10} | {df['base_dice'].mean():.4f}   | {df['tta2_dice'].mean():.4f}   | {df['tta4_dice'].mean():.4f}   | {df['tta8_dice'].mean():.4f}")
    print(f"{'BF1':<10} | {df['base_bf1'].mean():.4f}   | {df['tta2_bf1'].mean():.4f}   | {df['tta4_bf1'].mean():.4f}   | {df['tta8_bf1'].mean():.4f}")
    print(f"{'HD95':<10} | {df['base_hd95'].dropna().mean():.4f}   | {df['tta2_hd95'].dropna().mean():.4f}   | {df['tta4_hd95'].dropna().mean():.4f}   | {df['tta8_hd95'].dropna().mean():.4f}")

def main():
    from torch.utils.data import DataLoader
    
    # 1. CVC UNet
    model_unet = get_unet(pretrained=False).to(DEVICE)
    model_unet.load_state_dict(torch.load('checkpoints/unet_cvc_best.pth', map_location=DEVICE))
    cvc_val = DataLoader(CVCDataset('cvc_clinicdb/original', 'cvc_clinicdb/ground_truth', split='val'), batch_size=1)
    evaluate_2d('CVC UNet', model_unet, cvc_val, 'results/ablation_cvc_unet.csv')
    
    # 2. CVC SwinUNETR
    model_swin = get_swin_unetr().to(DEVICE)
    model_swin.load_state_dict(torch.load('checkpoints/swinunetr_cvc_best.pth', map_location=DEVICE))
    evaluate_2d('CVC SwinUNETR', model_swin, cvc_val, 'results/ablation_cvc_swin.csv')
    
    # 3. BraTS SegResNet
    model_brats = SegResNet(in_channels=4, out_channels=1, init_filters=16).to(DEVICE)
    model_brats.load_state_dict(torch.load('checkpoints/segresnet_brats_best.pth', map_location=DEVICE))
    brats_val = DataLoader(BraTSDataset('brats2021', split='val'), batch_size=1)
    evaluate_3d('BraTS SegResNet', model_brats, brats_val, 'results/ablation_brats.csv')

if __name__ == '__main__':
    main()
