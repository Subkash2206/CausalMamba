"""
train_unet_isic18_cvcrecipe.py — ISIC2018 ResNet50-UNet retrain with the CVC recipe.

Phase-0 audit found the existing ISIC ResNet50-UNet was trained with a *different
recipe* (ImageNet init, Dice loss, no scheduler) than the CVC ResNet50-UNet
(random init, BCEWithLogits, cosine). That makes the cross-dataset inversion
confounded by recipe. This script retrains the ISIC model with the EXACT CVC
recipe so the only differences are dataset + normalization.

CVC recipe (train_cvc_256.py): smp.Unet(resnet50, weights=None), BCEWithLogits,
Adam 1e-4, CosineAnnealing(T_max=100), 256, bs8, 100 epochs, seed 42.
ISIC-specific: ImageNet normalization (legitimate dataset difference).

Consumes the fixed 80/10/10 split from carve_splits.py. Selection = best val BCE.

Usage:
    python interventions/train_unet_isic18_cvcrecipe.py
"""

import sys, os, json, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets as tv_datasets
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT = _REPO
IMG_DIR = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'images')
MASK_DIR = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'masks')
SPLIT_JSON = os.path.join(_REPO, 'interventions', 'results', 'splits', 'isic_split.json')
CKPT_DIR = os.path.join(_REPO, 'interventions', 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(CKPT_DIR, 'unet_isic_cvcrecipe_best.pth')

IMG_SIZE, EPOCHS, BATCH_SIZE, LR, SEED = 256, 100, 8, 1e-4, 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ISICDataset(Dataset):
    """ISIC18 with ImageNet normalization (per plan) and CVC-style transforms."""
    def __init__(self, names, img_dir, mask_dir, img_size=256, is_train=True):
        self.names = names
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        base = os.path.splitext(name)[0]
        img = TF.to_tensor(__import__('PIL').Image.open(os.path.join(self.img_dir, name)).convert('RGB').resize((self.img_size, self.img_size), __import__('PIL').Image.BILINEAR))
        mask = TF.to_tensor(__import__('PIL').Image.open(os.path.join(self.mask_dir, base + '_segmentation.png')).convert('L').resize((self.img_size, self.img_size), __import__('PIL').Image.NEAREST))
        mask = (mask > 0.5).float()
        if self.is_train:
            if np.random.rand() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if np.random.rand() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            if np.random.rand() > 0.5:
                img, mask = TF.rotate(img, np.random.uniform(-30, 30)), TF.rotate(mask, np.random.uniform(-30, 30))
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img, mask


import argparse

def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--resume', default=None,
                    help='Resume from a full state file (*.pt, saved every 5 epochs) OR raw '
                         'model weights (*.pth).')
    ap.add_argument('--resume_epoch', type=int, default=0,
                    help='Number of already-completed epochs when --resume points to raw '
                         'weights (scheduler + cosine position).')
    ap.add_argument('--resume_best_val', type=float, default=float('inf'),
                    help='Best val loss achieved before resume (raw-weights resume).')
    args, _ = ap.parse_known_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    train_ds = ISICDataset(split['train'], IMG_DIR, MASK_DIR, is_train=True)
    val_ds = ISICDataset(split['val'], IMG_DIR, MASK_DIR, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    # EXACT CVC recipe: random init, BCEWithLogits, Adam 1e-4, cosine.
    model = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_loss = float('inf')
    start_epoch = 0

    STATE_PATH = os.path.join(CKPT_DIR, 'unet_isic_cvcrecipe_state_latest.pt')
    if args.resume:
        sd = torch.load(args.resume, map_location=DEVICE)
        if 'model_state_dict' in sd:
            # Full state file written by this script (every 5 epochs).
            model.load_state_dict(sd['model_state_dict'], strict=True)
            optimizer.load_state_dict(sd['optimizer_state_dict'])
            scheduler.load_state_dict(sd['scheduler_state_dict'])
            start_epoch = sd['epoch']            # completed epochs -> next epoch = +1
            best_val_loss = sd['best_val_loss']
            print(f'[resume] full-state from {os.path.basename(args.resume)}: '
                  f'resuming after epoch {start_epoch}, best_val {best_val_loss:.4f}')
        else:
            # Raw model weights (e.g., the epoch-32 best saved before resume support).
            model.load_state_dict(sd if not isinstance(sd, dict) else
                                  (sd.get('model_state_dict') or sd.get('state_dict') or sd),
                                  strict=True)
            for _ in range(args.resume_epoch):
                scheduler.step()
            start_epoch = args.resume_epoch
            best_val_loss = args.resume_best_val
            print(f'[resume] raw weights from {os.path.basename(args.resume)}: '
                  f'resuming after epoch {start_epoch}, best_val {best_val_loss:.4f}, '
                  f'lr now {scheduler.get_last_lr()[0]:.2e}')

    print(f'Training ISIC ResNet50-UNet (CVC recipe) {IMG_SIZE}px, {EPOCHS} epochs, '
          f'train={len(train_ds)} val={len(val_ds)} on {DEVICE} '
          f'| resume-from-epoch={start_epoch}')
    import time; t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        model.train(); train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(); loss = criterion(model(imgs), masks)
            loss.backward(); optimizer.step(); train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_loss += criterion(model(imgs), masks).item()
        val_loss /= len(val_loader); scheduler.step()
        ok = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH); ok = ' --> saved best'
        if (epoch + 1) % 5 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch + 1, 'best_val_loss': best_val_loss}, STATE_PATH)
        el = (time.time() - t0) / 60
        print(f'Epoch {epoch+1:03d}/{EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f} | {el:.1f}min{ok}')
    print(f'Done. Best val loss {best_val_loss:.4f} -> {SAVE_PATH}')


if __name__ == '__main__':
    main()
