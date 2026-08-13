"""
train_swinunetr_isic18_cvcrecipe.py — Swin-UNETR (ViT) on ISIC2018 with the CVC recipe.

Closes the ViT leg of the cross-dataset comparison. The CVC ViT leg is Swin-UNETR
(MONAI, spatial_dims=2, trained at 256 with the CVC recipe); the ISIC ViT leg was the
*different* Swin-UNet architecture. This retrains Swin-UNETR on ISIC with the EXACT
CVC Swin-UNETR recipe (BCEWithLogits, Adam 1e-4, CosineAnnealing(100), 256, batch 8,
seed 42) so both ViT legs are the same architecture and recipe — only dataset +
normalization (ImageNet for ISIC, [0,1] for CVC) differ, mirroring the CNN leg.

Consumes the fixed 80/10/10 split from carve_splits.py. Selection = best val loss.

Usage:
    python interventions/train_swinunetr_isic18_cvcrecipe.py [--max_batches N] [--resume ...]
"""

import sys, os, json, argparse, datetime, random, time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms.functional as TF

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_REPO, 'tta_boundary_study'))
sys.path.insert(0, _REPO)

from interventions.isic_dataset import ISICCacheDataset

IMG_DIR = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'images')
MASK_DIR = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'masks')
SPLIT_JSON = os.path.join(_REPO, 'interventions', 'results', 'splits', 'isic_split.json')
CKPT_DIR = os.path.join(_REPO, 'interventions', 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(CKPT_DIR, 'swinunetr_isic_cvcrecipe_best.pth')
STATE_PATH = os.path.join(CKPT_DIR, 'swinunetr_isic_cvcrecipe_state_latest.pt')

IMG_SIZE, EPOCHS, BATCH_SIZE, LR, SEED = 256, 100, 8, 1e-4, 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--max_batches', type=int, default=None, help='Smoke/timing: stop each epoch after N batches.')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--resume_epoch', type=int, default=0)
    ap.add_argument('--resume_best_val', type=float, default=float('inf'))
    args, _ = ap.parse_known_args()

    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    train_ds = ISICCacheDataset(split['train'], IMG_DIR, MASK_DIR, is_train=True,
                                cache_name='train')
    val_ds = ISICCacheDataset(split['val'], IMG_DIR, MASK_DIR, is_train=False,
                              cache_name='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f'Swin-UNETR ISIC (CVC recipe) 256px | train={len(train_ds)} val={len(val_ds)} '
          f'| batch {BATCH_SIZE} | {DEVICE}')

    from src.models.swin_unetr_cvc import get_swin_unetr
    model = get_swin_unetr().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_loss = float('inf')
    start_epoch = 0

    if args.resume:
        sd = torch.load(args.resume, map_location=DEVICE)
        if 'model_state_dict' in sd:
            model.load_state_dict(sd['model_state_dict'], strict=True)
            optimizer.load_state_dict(sd['optimizer_state_dict'])
            scheduler.load_state_dict(sd['scheduler_state_dict'])
            start_epoch = sd['epoch']
            best_val_loss = sd['best_val_loss']
            print(f'[resume] full-state from {os.path.basename(args.resume)}: '
                  f'after epoch {start_epoch}, best_val {best_val_loss:.4f}')
        else:
            model.load_state_dict(sd if not isinstance(sd, dict) else
                                  (sd.get('model_state_dict') or sd.get('state_dict') or sd),
                                  strict=True)
            for _ in range(args.resume_epoch):
                scheduler.step()
            start_epoch = args.resume_epoch
            best_val_loss = args.resume_best_val
            print(f'[resume] raw weights from {os.path.basename(args.resume)}: '
                  f'after epoch {start_epoch}, best_val {best_val_loss:.4f}')

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        for i, (imgs, masks) in enumerate(train_loader, 1):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if args.max_batches and i >= args.max_batches:
                break
        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_loss += criterion(model(imgs), masks).item()
                pred = (torch.sigmoid(model(imgs)) > 0.5).float()
                inter = (pred * masks).sum().item()
                union = pred.sum().item() + masks.sum().item()
                val_dice += 2 * inter / max(union, 1e-8)
        val_loss /= max(len(val_loader), 1)
        val_dice /= max(len(val_loader), 1)

        saved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            saved = ' --> saved best'
        if (epoch + 1) % 5 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch + 1, 'best_val_loss': best_val_loss}, STATE_PATH)
        el = (time.time() - t0) / 60
        print(f'Epoch {epoch+1:03d}/{EPOCHS} | Train {train_loss:.4f} | '
              f'Val {val_loss:.4f} | Val Dice {val_dice:.4f} | {el:.1f}min{saved}')

    print(f'Done. Best val loss {best_val_loss:.4f} -> {SAVE_PATH}')
    meta = {'experiment': 'Swin-UNETR ISIC (CVC recipe) — ViT leg closure',
            'model': 'SwinUNETR (MONAI, spatial_dims=2)', 'dataset': 'ISIC2018',
            'img_size': IMG_SIZE, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'lr': LR, 'seed': SEED, 'checkpoint': SAVE_PATH,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    with open(os.path.join(CKPT_DIR, 'metadata_swinunetr_isic_cvcrecipe.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
