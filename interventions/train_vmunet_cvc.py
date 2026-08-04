"""
train_vmunet_cvc.py — VM-UNet training on CVC-ClinicDB (Phase 2).

Prepares the missing VM-UNet CVC-ClinicDB checkpoint. Uses the canonical
30-block VSSM topology (depths=[2,2,9,2], depths_decoder=[2,9,2,2]) and the
same CVCDataset pipeline as tta_boundary_study (352x352, [0,1] normalization).

Training recipe: AdamW + BCE + soft-Dice loss, cosine anneal, 100 epochs,
best-val-loss checkpointing. Initialization from ImageNet-1k pretrained
weights is wired via --load_ckpt (VMUNet.load_from()).

Run overnight (intentionally not executed here):
    cd SpectralMamba
    python ..\\interventions\\train_vmunet_cvc.py --output_dir ..\\checkpoints --seed 42
"""

import sys, os, argparse, random, json, datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SPECTRAL = os.path.join(_REPO, 'SpectralMamba')
_TTA = os.path.join(_REPO, 'tta_boundary_study')
for p in (_SPECTRAL, _TTA):
    sys.path.insert(0, p)

CVC_IMG_DIR = os.path.join(_TTA, 'cvc_clinicdb', 'original')
CVC_MASK_DIR = os.path.join(_TTA, 'cvc_clinicdb', 'ground_truth')


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_dir', default=os.path.join(_TTA, 'checkpoints'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--img_size', type=int, default=352)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--load_ckpt', default=None,
                    help='Optional ImageNet-1k pretrained backbone checkpoint')
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from src.datasets.cvc_dataset import CVCDataset
    from models.vmunet.vmunet import VMUNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('train_vmunet_cvc.py - VM-UNet training on CVC-ClinicDB')
    print('=' * 80)
    print(f'Device: {device}')

    # ------------------------------------------------------------------
    # Datasets (CVC-ClinicDB, 352x352, [0,1] normalization)
    # ------------------------------------------------------------------
    train_ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='train', img_size=args.img_size)
    val_ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='val', img_size=args.img_size)
    print(f'Train: {len(train_ds)} images | Val: {len(val_ds)} images | 352x352')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # ------------------------------------------------------------------
    # Model: canonical 30-block VM-UNet (strictly verified topology)
    # ------------------------------------------------------------------
    _NC = {'num_classes': 1, 'input_channels': 3,
           'depths': [2, 2, 9, 2], 'depths_decoder': [2, 9, 2, 2],
           'drop_path_rate': 0.2,
           'load_ckpt_path': args.load_ckpt}
    model = VMUNet(**_NC).to(device)

    # ImageNet-1k pretrained initialization (matching ISIC methodology).
    if args.load_ckpt and os.path.exists(args.load_ckpt):
        print(f'Loading pretrained backbone from {args.load_ckpt} ...')
        try:
            model.load_from()
            print('Pretrained encoder weights loaded via VMUNet.load_from().')
        except Exception as e:  # noqa: BLE001
            print(f'load_from() failed ({e}); training from scratch.')
    else:
        print('No --load_ckpt supplied - training from scratch '
              '(wire ImageNet weights later via --load_ckpt <path>).')

    # ------------------------------------------------------------------
    # Loss: BCE + soft-Dice (standard for binary segmentation)
    # ------------------------------------------------------------------
    bce = nn.BCEWithLogitsLoss()

    def soft_dice_loss(logits, masks, smooth=1.0):
        probs = torch.sigmoid(logits)
        inter = (probs * masks).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()

    def combined_loss(logits, masks):
        return bce(logits, masks) + soft_dice_loss(logits, masks)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'best-vmunet-cvc.pth')
    best_val_loss = float('inf')

    print(f'\nTraining {args.epochs} epochs (AdamW, BCE+Dice) ...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_loss(logits, masks)
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
                val_loss += combined_loss(logits, masks).item()
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
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
        print(f'Epoch {epoch:03d}/{args.epochs} | Train Loss {train_loss:.4f} '
              f'| Val Loss {val_loss:.4f} | Val Dice {val_dice:.4f}{saved}')

    print(f'\nTraining complete. Best checkpoint: {save_path} (val loss {best_val_loss:.4f})')

    meta = {
        'experiment': 'VM-UNet CVC-ClinicDB training',
        'model': 'VM-UNet', 'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 9, 2, 2], 'dataset': 'CVC-ClinicDB',
        'img_size': args.img_size, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'lr': args.lr, 'seed': args.seed,
        'device': str(device), 'checkpoint': save_path,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    }
    with open(os.path.join(args.output_dir, 'metadata_vmunet_cvc_training.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()