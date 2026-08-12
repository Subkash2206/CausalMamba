"""
train_vmunet_cvc_tsa.py — VM-UNet training on CVC-ClinicDB with Targeted
Spectral Augmentation (Phase 4.2).

Identical to the baseline train_vmunet_cvc.py recipe (canonical 30-block VSSM
topology depths=[2,2,9,2], depths_decoder=[2,9,2,2], AdamW + BCE + soft-Dice,
cosine anneal, AMP, gradient checkpointing, best-val-loss checkpointing) PLUS
a dynamic low-pass input augmentation:

    Each training batch, with probability p=0.5 per image, the input is passed
    through a GPU FFT -> circular low-pass (cutoff=0.25) -> IFFT. This decouples
    the model's reliance on high-frequency boundaries during training (the
    "generational lock-in" objective), so it learns to segment from
    low-frequency content alone and becomes robust to spectral perturbation.

Usage:
    cd SpectralMamba
    python ..\\interventions\\train_vmunet_cvc_tsa.py ^
        --img_size 352 --epochs 100 --batch_size 4 --checkpointing --amp --seed 42 ^
        --output_dir ..\\interventions\\results\\best-vmunet-cvc-tsa
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


@torch.no_grad()
def apply_tsa(images, cutoff=0.25, p=0.5):
    """Targeted Spectral Augmentation: Applies low-pass filter to a random p% of the batch."""
    B, C, H, W = images.shape
    device = images.device

    # Decide which images in the batch get augmented
    mask = torch.rand(B, device=device) < p
    if not mask.any():
        return images

    # Process only the selected images
    imgs_to_aug = images[mask]

    # 2D FFT
    fft_imgs = torch.fft.fftshift(torch.fft.fft2(imgs_to_aug.float(), norm='ortho'))

    # Create circular low-pass mask (keep frequencies <= cutoff)
    center_h, center_w = H // 2, W // 2
    Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    Y = (Y - center_h) / (H / 2)
    X = (X - center_w) / (W / 2)
    radius = torch.sqrt(X ** 2 + Y ** 2)
    lp_mask = (radius <= cutoff).float().unsqueeze(0).unsqueeze(0)

    # Apply mask
    fft_imgs_filtered = fft_imgs * lp_mask

    # Inverse FFT
    imgs_filtered = torch.fft.ifft2(torch.fft.ifftshift(fft_imgs_filtered), norm='ortho').real
    # Clamp to the [0,1] input distribution: trims Gibbs-ringing tails so the
    # blurred images cannot produce degenerate (near-zero-variance) extremes that
    # overflow the fp16 selective-scan path under AMP (crash seen at batch ~186).
    imgs_filtered = imgs_filtered.clamp(0.0, 1.0)

    # Put augmented images back into the batch
    images_out = images.clone()
    images_out[mask] = imgs_filtered.to(images.dtype)
    return images_out



def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_dir', default=os.path.join(_TTA, 'checkpoints'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--img_size', type=int, default=352)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8,
                    help='Micro-batch size (effective batch = batch_size * accumulation_steps)')
    ap.add_argument('--accumulation_steps', type=int, default=1,
                    help='Gradient accumulation steps; preserves effective batch size (default 1)')
    ap.add_argument('--amp', action='store_true',
                    help='Use mixed precision (autocast + GradScaler); no protocol change')
    ap.add_argument('--checkpointing', action='store_true',
                    help='Enable built-in gradient checkpointing (Chen et al. 2016)')
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
    print('train_vmunet_cvc_tsa.py - VM-UNet CVC training + Targeted Spectral Augmentation')
    print('=' * 80)
    print(f'Device: {device}')

    # ------------------------------------------------------------------
    # Datasets (CVC-ClinicDB, 352x352, [0,1] normalization)
    # ------------------------------------------------------------------
    train_ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='train', img_size=args.img_size)
    val_ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='val', img_size=args.img_size)
    print(f'Train: {len(train_ds)} images | Val: {len(val_ds)} images | {args.img_size}x{args.img_size}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # ------------------------------------------------------------------
    # Model: canonical 30-block VM-UNet (strictly verified topology)
    # ------------------------------------------------------------------
    _NC = {'num_classes': 1, 'input_channels': 3,
           'depths': [2, 2, 9, 2], 'depths_decoder': [2, 9, 2, 2],
           'drop_path_rate': 0.2,
           'load_ckpt_path': args.load_ckpt,
           'use_checkpoint': args.checkpointing}
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
    # NOTE: VMUNet.forward applies sigmoid internally when num_classes==1,
    # so model(imgs) returns probabilities in [0,1] -- NOT raw logits.
    bce = nn.BCELoss()  # expects probabilities

    def soft_dice_loss(probs, masks, smooth=1.0):
        inter = (probs * masks).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()

    def combined_loss(preds, masks):
        return bce(preds, masks) + soft_dice_loss(preds, masks)

    # ------------------------------------------------------------------
    # LOAD PRETRAINED BASELINE FOR ROBUSTNESS FINE-TUNING (Phase 4.2)
    # ------------------------------------------------------------------
    baseline_ckpt_path = os.path.join(_REPO, 'tta_boundary_study', 'checkpoints',
                                      'best-vmunet-cvc.pth')  # standard Phase-1 baseline
    print(f'Loading baseline weights from {baseline_ckpt_path} for fine-tuning...')
    if not os.path.exists(baseline_ckpt_path):
        raise FileNotFoundError(f'Baseline checkpoint not found: {baseline_ckpt_path}')
    checkpoint = torch.load(baseline_ckpt_path, map_location=device)
    # Handle DataParallel/DDP module prefixes if they exist
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    print('Baseline weights loaded successfully.')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    autocast = torch.cuda.amp.autocast(enabled=bool(args.amp))

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'best-vmunet-cvc.pth')
    best_val_loss = float('inf')

    acc_steps = max(1, args.accumulation_steps)
    eff_batch = args.batch_size * acc_steps
    print(f'\nTraining {args.epochs} epochs (AdamW, BCE+Dice) ...')
    print(f'  micro-batch={args.batch_size} x accumulation={acc_steps} '
          f'=> effective batch {eff_batch}')
    _ep_start = datetime.datetime.now()
    nan_batches = 0   # Phase 4.2 crash-fix monitor: batches with NaN in model output
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (imgs, masks) in enumerate(train_loader, start=1):
            imgs, masks = imgs.to(device), masks.to(device)
            # Targeted Spectral Augmentation (Phase 4.2): low-pass a random 50% of the batch.
            imgs = apply_tsa(imgs, cutoff=0.25, p=0.5)
            with autocast:
                probs = model(imgs)  # sigmoid probabilities (VMUNet internal sigmoid)
            # Robustness guard (Phase 4.2 crash fix): a degenerate TSA-blurred input
            # can overflow the fp16 scan -> NaN in probs -> BCELoss device-side assert.
            # nan_to_num maps NaN positions to 0.5 (zero grad there), so training
            # continues instead of crashing; we count them for monitoring.
            probs_f32 = probs.float()
            if torch.isnan(probs_f32).any():
                nan_batches += 1
                if nan_batches % 10 == 0:
                    print(f'\n[WARN] NaN in model output (batch {batch_idx}); '
                          f'{int(torch.isnan(probs_f32).sum())} NaN values -> guarded')
                probs_f32 = torch.nan_to_num(probs_f32, nan=0.5)
            # BCELoss is unsafe inside autocast -> compute loss in fp32.
            loss = combined_loss(probs_f32.clamp(1e-6, 1.0 - 1e-6), masks.float()) / acc_steps
            scaler.scale(loss).backward()
            train_loss += loss.item() * acc_steps

            _n = len(train_loader)
            _el = (datetime.datetime.now() - _ep_start).total_seconds()
            _frac = batch_idx / _n
            _w = 24
            _f = int(_w * _frac)
            _eta = _el / max(_frac, 1e-9) * (1 - _frac)
            print(f"\rEpoch {epoch:03d}/{args.epochs} | "
                  f"[{'#' * _f}{'-' * (_w - _f)}] {_frac * 100:5.1f}% "
                  f"| batch {batch_idx:3d}/{_n} "
                  f"| loss {train_loss / batch_idx * acc_steps:7.4f} "
                  f"| {_el:6.0f}s | ETA {_eta:6.0f}s",
                  end='', flush=True)
            if batch_idx == _n:
                print()

            if batch_idx % acc_steps == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)   # unscale + apply if grads finite
                scaler.update()
                optimizer.zero_grad()
        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast:
                    probs = model(imgs)  # already sigmoid-ed
                val_loss += combined_loss(probs.float(), masks.float()).item()
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
    if nan_batches:
        print(f'[WARN] {nan_batches} batches had NaN in model output (guarded with nan_to_num).')

    meta = {
        'experiment': 'VM-UNet CVC-ClinicDB training + Targeted Spectral Augmentation (Phase 4.2)',
        'model': 'VM-UNet', 'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 9, 2, 2], 'dataset': 'CVC-ClinicDB',
        'img_size': args.img_size, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'accumulation_steps': acc_steps,
        'effective_batch_size': eff_batch, 'lr': args.lr, 'seed': args.seed,
        'tsa': {'augmentation': 'targeted_spectral_lowpass',
                'cutoff': 0.25, 'probability': 0.5, 'on': 'train_only'},
        'fine_tuning': {'from': baseline_ckpt_path, 'strict': True,
                        'phase': 'Phase 4.2 robustness fine-tuning'},
        'device': str(device), 'checkpoint': save_path,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    }
    with open(os.path.join(args.output_dir, 'metadata_vmunet_cvc_tsa.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()