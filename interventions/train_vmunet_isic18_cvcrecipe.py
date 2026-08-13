"""
train_vmunet_isic18_cvcrecipe.py — ISIC2018 VM-UNet retrain with the CVC recipe,
canonical VSSM implementation.

Why: the existing ISIC VM-UNet was trained with the *repo* VSSM (JIT per-timestep
loop); the CVC VM-UNet used the *canonical* VSSM (vectorized scan). We proved the
two implementations give different outputs for identical weights. To make the SSM
leg a clean, implementation-matched comparison, retrain ISIC with the canonical
implementation and the exact CVC training recipe (from scratch, AdamW 1e-4 wd 1e-4,
BCE + soft-Dice, cosine, best-val-loss, seed 42). Only dataset + normalization
(ImageNet, per the plan) differ from CVC.

Local-run setup (2026-08-13): uses ISICCacheDataset (resized-256 tensors cached on
disk + RAM) so the per-epoch 29MP JPEG decode cost is eliminated; cudnn.benchmark
on; full-state resume every 5 epochs (crash-safe on a laptop that has rebooted).

Usage:
    python interventions/train_vmunet_isic18_cvcrecipe.py \
        [--max_batches N] [--resume state.pt|best.pth] [--resume_epoch N] [--resume_best_val V]

Output: interventions/checkpoints/vmunet_isic_cvcrecipe_best.pth
"""

import sys, os, json, argparse, time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SPECTRAL = os.path.join(_REPO, 'SpectralMamba')
sys.path.insert(0, _SPECTRAL)
sys.path.insert(0, _REPO)

from interventions.isic_dataset import ISICCacheDataset

IMG_DIR = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'images')
MASK_DIR = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'masks')
SPLIT_JSON = os.path.join(_REPO, 'interventions', 'results', 'splits', 'isic_split.json')
CKPT_DIR = os.path.join(_REPO, 'interventions', 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(CKPT_DIR, 'vmunet_isic_cvcrecipe_best.pth')
STATE_PATH = os.path.join(CKPT_DIR, 'vmunet_isic_cvcrecipe_state_latest.pt')

IMG_SIZE, EPOCHS, LR, SEED = 256, 100, 1e-4, 42
MICRO_BATCH, ACC_STEPS = 2, 4           # effective batch 8, matching CVC recipe
AMP, CHECKPOINTING = True, True


def soft_dice_loss(probs, masks, smooth=1.0):
    inter = (probs * masks).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--max_batches', type=int, default=None,
                    help='Smoke/timing: stop each epoch after N micro-batches.')
    ap.add_argument('--max_epochs', type=int, default=None,
                    help='Smoke/timing: stop after this many epochs.')
    ap.add_argument('--resume', default=None,
                    help='Full state file (*.pt) OR raw model weights (*.pth).')
    ap.add_argument('--resume_epoch', type=int, default=0,
                    help='Completed epochs when --resume points to raw weights.')
    ap.add_argument('--resume_best_val', type=float, default=float('inf'))
    args, _ = ap.parse_known_args()

    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    from models.vmunet.vmunet import VMUNet  # canonical VSSM (vectorized scan)

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    train_ds = ISICCacheDataset(split['train'], IMG_DIR, MASK_DIR, is_train=True,
                                cache_name='train')
    val_ds = ISICCacheDataset(split['val'], IMG_DIR, MASK_DIR, is_train=False,
                              cache_name='val')
    train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VMUNet(num_classes=1, input_channels=3,
                   depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                   drop_path_rate=0.2, load_ckpt_path=None,
                   use_checkpoint=CHECKPOINTING).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'VM-UNet (canonical) {n_params:.1f}M params | train={len(train_ds)} '
          f'val={len(val_ds)} | eff-batch {MICRO_BATCH * ACC_STEPS} | {device}')

    bce = nn.BCELoss()

    def combined_loss(preds, masks):
        return bce(preds, masks) + soft_dice_loss(preds, masks)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    autocast = torch.cuda.amp.autocast(enabled=AMP)

    best_val_loss = float('inf')
    start_epoch = 0
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
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

    t_start = time.time()
    print(f'Training {EPOCHS} epochs (resume-from-epoch {start_epoch}) ...')
    for epoch in range(start_epoch, EPOCHS):
        if args.max_epochs and epoch - start_epoch >= args.max_epochs:
            break
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (imgs, masks) in enumerate(train_loader, start=1):
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast:
                probs = model(imgs)  # VMUNet applies sigmoid internally
            loss = combined_loss(probs.float(), masks.float()) / ACC_STEPS
            scaler.scale(loss).backward()
            train_loss += loss.item() * ACC_STEPS
            if batch_idx % ACC_STEPS == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if args.max_batches and batch_idx >= args.max_batches:
                break
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast:
                    probs = model(imgs)
                val_loss += combined_loss(probs.float(), masks.float()).item()
                pred = (probs > 0.5).float()
                inter = (pred * masks).sum().item()
                union = pred.sum().item() + masks.sum().item()
                val_dice += 2 * inter / max(union, 1e-8)
        val_loss /= max(len(val_loader), 1)
        val_dice /= max(len(val_loader), 1)
        scheduler.step()

        saved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            saved = ' --> Saved best'
        if (epoch + 1) % 5 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch + 1, 'best_val_loss': best_val_loss}, STATE_PATH)
        el = (time.time() - t_start) / 3600
        print(f'Epoch {epoch+1:03d}/{EPOCHS} | Train {train_loss:.4f} | '
              f'Val {val_loss:.4f} | Dice {val_dice:.4f} | {el:.1f}h{saved}')
    print(f'Done. Best val loss {best_val_loss:.4f} -> {SAVE_PATH}')


if __name__ == '__main__':
    main()
