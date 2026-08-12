"""
train_swinunetr_cvc_256.py — Swin-UNETR (ViT) retrained on CVC-ClinicDB at 256x256.

Fixes the resolution confound identified in the WACV audit: the existing
swinunetr_cvc_best.pth was trained at 352 but benchmarked at 256. This script
retrains the MONAI Swin-UNETR (spatial_dims=2) at 256x256 so the cross-
architecture benchmark is apples-to-apples.

Same recipe as tta_boundary_study/src/train_swin_cvc.py: Adam + BCEWithLogits +
cosine anneal, 100 epochs, best-val-loss checkpointing. Batch 8 fits the 6 GB
GPU at 256 (model is ~6M params).

Usage:
    python interventions/train_swinunetr_cvc_256.py
"""

import sys, os, argparse, json, datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_TTA = os.path.join(_REPO, 'tta_boundary_study')
for p in (_TTA,):
    sys.path.insert(0, p)

IMG_DIR = os.path.join(_TTA, 'cvc_clinicdb', 'original')
MASK_DIR = os.path.join(_TTA, 'cvc_clinicdb', 'ground_truth')


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default=os.path.join(
        _REPO, 'interventions', 'results', 'best-swinunetr-cvc-256'))
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from src.datasets.cvc_dataset import CVCDataset
    from src.models.swin_unetr_cvc import get_swin_unetr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('Swin-UNETR CVC retrain @256 (WACV resolution-confound fix)')
    print('=' * 80)
    print(f'Device: {device}')

    train_ds = CVCDataset(IMG_DIR, MASK_DIR, split='train', img_size=args.img_size)
    val_ds = CVCDataset(IMG_DIR, MASK_DIR, split='val', img_size=args.img_size)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | {args.img_size}x{args.img_size}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = get_swin_unetr().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'best-swinunetr-cvc-256.pth')
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred * masks).sum().item()
                union = pred.sum().item() + masks.sum().item()
                val_dice += 2 * inter / max(union, 1e-8)
        val_loss /= max(len(val_loader), 1)
        val_dice /= max(len(val_loader), 1)

        saved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            saved = ' --> Saved best'
        print(f'Epoch {epoch:03d}/{args.epochs} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} | Val Dice {val_dice:.4f}{saved}')

    print(f'\nTraining complete. Best checkpoint: {save_path} (val loss {best_val_loss:.4f})')
    meta = {
        'experiment': 'Swin-UNETR CVC retrain @256 (resolution-confound fix)',
        'model': 'SwinUNETR (MONAI, spatial_dims=2)', 'dataset': 'CVC-ClinicDB',
        'img_size': args.img_size, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'lr': args.lr, 'seed': args.seed, 'device': str(device),
        'checkpoint': save_path,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    }
    with open(os.path.join(args.output_dir, 'metadata_swinunetr_cvc_256.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
