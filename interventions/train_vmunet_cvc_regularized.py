"""
train_vmunet_cvc_regularized.py - VM-UNet training on CVC-ClinicDB
with High-Frequency Consistency regularization (Phase 4, WACV Stretch).

Constructive arm: an auxiliary spectral loss stabilises the high-frequency
feature representations of the VM-UNet encoder. For every training batch:

    1. images_perturbed = images + N(0, 0.05)
    2. Forward images  -> capture first-VSSBlock features (features_clean)
    3. Forward images_perturbed -> capture first-VSSBlock features (features_pert)
    4. seg_loss   = BCE + soft-Dice on the standard (clean) outputs
    5. spec_loss  = HighFrequencyConsistencyLoss(features_clean, features_pert)
    6. loss       = seg_loss + lambda_spectral * spec_loss   (lambda_spectral = 0.05)

Otherwise identical to train_vmunet_cvc.py: canonical 30-block VSSM topology
(depths=[2,2,9,2], depths_decoder=[2,9,2,2]), AdamW + BCE + soft-Dice, cosine
anneal, AMP (fp32 FFT safety handled inside the spectral loss), and best-val
checkpointing.

Usage:
    cd SpectralMamba
    python ..\\interventions\\train_vmunet_cvc_regularized.py --amp --output_dir ..\\checkpoints
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
for p in (_REPO, _SPECTRAL, _TTA):
    sys.path.insert(0, p)

from interventions.spectral_loss import HighFrequencyConsistencyLoss

CVC_IMG_DIR = os.path.join(_TTA, 'cvc_clinicdb', 'original')
CVC_MASK_DIR = os.path.join(_TTA, 'cvc_clinicdb', 'ground_truth')

LAMBDA_SPECTRAL = 0.05          # weight of the high-frequency consistency loss
PERTURB_STD = 0.05              # std of Gaussian input perturbation


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_dir', default=os.path.join(_TTA, 'checkpoints'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--img_size', type=int, default=352)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8,
                    help='Micro-batch size (effective batch = batch_size * grad_accum_steps)')
    ap.add_argument('--grad_accum_steps', type=int, default=4,
                    help='Gradient accumulation steps; effective batch = batch_size * grad_accum_steps (default 4)')
    ap.add_argument('--amp', action='store_true',
                    help='Use mixed precision (autocast + GradScaler); no protocol change')
    ap.add_argument('--checkpointing', action='store_true',
                    help='Enable built-in gradient checkpointing (Chen et al. 2016)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--load_ckpt', default=None,
                    help='Optional ImageNet-1k pretrained backbone checkpoint')
    ap.add_argument('--max_batches', type=int, default=0,
                    help='Limit batches per epoch (0 = full epoch); for smoke tests')
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
    print('train_vmunet_cvc_regularized.py - VM-UNet CVC training + spectral consistency')
    print('=' * 80)
    # ------------------------------------------------------------------
    # Datasets (CVC-ClinicDB, [0,1] normalization)
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
    # Segmentation loss: BCE + soft-Dice (VMUNet returns sigmoid probs)
    # ------------------------------------------------------------------
    bce = nn.BCELoss()  # expects probabilities

    def soft_dice_loss(probs, masks, smooth=1.0):
        inter = (probs * masks).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()

    def combined_loss(preds, masks):
        return bce(preds, masks) + soft_dice_loss(preds, masks)

    # ------------------------------------------------------------------
    # Spectral auxiliary loss + feature capture hook (first VSSBlock)
    # ------------------------------------------------------------------
    spectral_criterion = HighFrequencyConsistencyLoss(cutoff_radius=0.25)

    _feat_store = []  # holds the captured first-VSSBlock feature of the last forward

    def _capture_hook(module, inp, out):
        if not module.training:
            return None  # never capture during validation (avoids leaks)
        if isinstance(out, tuple):
            out = out[0]
        # First VSSBlock emits NHWC (B, H, W, C); store as NCHW for the FFT loss.
        is_nhwc = (out.dim() == 4 and out.shape[-1] in {96, 192, 384, 768})
        _feat_store.append(out.permute(0, 3, 1, 2).contiguous() if is_nhwc else out)
        return None  # do NOT modify the forward output

    first_vss_block = model.vmunet.layers[0].blocks[0]
    _hook_handle = first_vss_block.register_forward_hook(_capture_hook)
    print(f'\nSpectral hook registered on {type(first_vss_block).__name__} '
          f'(first encoder VSSBlock).')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    autocast = torch.cuda.amp.autocast(enabled=bool(args.amp))

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'best-vmunet-cvc-reg.pth')
    best_val_loss = float('inf')

    grad_accum_steps = max(1, args.grad_accum_steps)
    eff_batch = args.batch_size * grad_accum_steps
    max_batches = max(0, args.max_batches)
    print(f'\nTraining {args.epochs} epochs (AdamW, BCE+Dice + {LAMBDA_SPECTRAL}*spectral) ...')
    print(f'  micro-batch={args.batch_size} x grad_accum={grad_accum_steps} '
          f'=> effective batch {eff_batch}')
    if max_batches:
        print(f'  LIMITED to {max_batches} batches/epoch (smoke test mode)')

    _ep_start = datetime.datetime.now()
    # Representative params for gradient-flow checks (cheap, no full-model scan)
    _rep_params = [model.vmunet.layers[0].blocks[0].ln_1.weight,   # hooked first VSSBlock
                   model.vmunet.final_conv.weight]                  # output head
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_spec = 0.0
        saw_grad = False
        saw_nan_grad = False
        optimizer.zero_grad()

        n_batches = min(len(train_loader), max_batches) if max_batches else len(train_loader)
        for batch_idx, (imgs, masks) in enumerate(train_loader, start=1):
            if batch_idx > n_batches:
                break
            imgs, masks = imgs.to(device), masks.to(device)

            # ==================================================================
            # PASS 1 (Clean): segmentation supervision only.
            # Backward immediately so the clean graph is freed before PASS 2,
            # keeping only one live computational graph in VRAM at a time.
            # ==================================================================
            _feat_store.clear()                       # hook: clear at forward start
            with autocast:
                outputs_clean = model(imgs)           # sigmoid probabilities
            # Stop-gradient anchor: detached fixed target for the spectral loss.
            feat_clean_anchor = _feat_store.pop().detach().clone()

            seg_loss = combined_loss(outputs_clean.float(), masks.float())
            scaler.scale(seg_loss / grad_accum_steps).backward()
            # --> Clean graph is now freed from VRAM.
            del outputs_clean

            # ==================================================================
            # PASS 2 (Perturbed): spectral consistency only.
            # Gradient flows through the shared encoder (perturbed branch); the
            # detached clean anchor serves as the fixed high-frequency target.
            # ==================================================================
            noise = torch.randn_like(imgs, device=imgs.device) * PERTURB_STD
            images_perturbed = imgs + noise
            _feat_store.clear()                       # hook: clear at forward start
            with autocast:
                outputs_pert = model(images_perturbed)  # output unused; need features
            feat_pert = _feat_store.pop()
            # CRITICAL memory fix: release the perturbed DECODER graph immediately.
            # spec_loss only needs the encoder path (retained via feat_pert); keeping
            # outputs_pert alive would leak the decoder activations until the next
            # batch and roughly double peak VRAM (verified by memory probe).
            del outputs_pert

            spec_loss = spectral_criterion(feat_clean_anchor, feat_pert.float())
            scaler.scale((LAMBDA_SPECTRAL * spec_loss) / grad_accum_steps).backward()
            # --> Perturbed graph is now freed from VRAM.

            train_loss += seg_loss.item() + LAMBDA_SPECTRAL * spec_loss.item()
            train_spec += spec_loss.item()

            # Track gradient flow + finiteness on representative params
            # (checked right after backward, before scaler.step clears grads).
            for rp in _rep_params:
                if rp.grad is not None:
                    saw_grad = True
                    if not torch.isfinite(rp.grad).all().item():
                        saw_nan_grad = True

            # Guarantee no cross-batch retention in the feature store (no leaks).
            _feat_store.clear()

            _el = (datetime.datetime.now() - _ep_start).total_seconds()
            _frac = batch_idx / n_batches
            _w = 24
            _f = int(_w * _frac)
            _eta = _el / max(_frac, 1e-9) * (1 - _frac)
            print(f"\rEpoch {epoch:03d}/{args.epochs} | "
                  f"[{'#' * _f}{'-' * (_w - _f)}] {_frac * 100:5.1f}% "
                  f"| batch {batch_idx:3d}/{n_batches} "
                  f"| loss {train_loss / batch_idx:7.4f} "
                  f"| spec {train_spec / batch_idx:7.4f} "
                  f"| {_el:6.0f}s | ETA {_eta:6.0f}s",
                  end='', flush=True)
            if batch_idx == n_batches:
                print()

            # Gradient accumulation: step only after grad_accum_steps micro-batches
            # (batch_idx is 1-based; the user's "(batch_idx+1) % steps" applies to
            #  0-based indexing, so "batch_idx % steps == 0" is the exact equivalent)
            # or at the end of the epoch (flush the partial group).
            if batch_idx % grad_accum_steps == 0 or batch_idx == n_batches:
                scaler.step(optimizer)   # unscale + apply if grads finite
                scaler.update()
                optimizer.zero_grad()

        train_loss /= max(n_batches, 1)
        train_spec /= max(n_batches, 1)

        # NaN/Inf health check (smoke-test assertion)
        finite_ok = bool(torch.isfinite(torch.tensor(train_loss)).item()
                         and torch.isfinite(torch.tensor(train_spec)).item())
        print(f'  [health] loss_finite={finite_ok} grads_flow={saw_grad} '
              f'grads_finite={not saw_nan_grad}')

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
        _feat_store.clear()  # defensive: val hook skips capture, but keep store empty

        saved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            saved = ' --> Saved best'
        print(f'Epoch {epoch:03d}/{args.epochs} | Train Loss {train_loss:.4f} '
              f'| Train Spec {train_spec:.6f} '
              f'| Val Loss {val_loss:.4f} | Val Dice {val_dice:.4f}{saved}')

    # ------------------------------------------------------------------
    # Cleanup + metadata
    # ------------------------------------------------------------------
    _hook_handle.remove()
    print(f'\nTraining complete. Best checkpoint: {save_path} (val loss {best_val_loss:.4f})')

    meta = {
        'experiment': 'VM-UNet CVC-ClinicDB regularized training (Phase 4)',
        'model': 'VM-UNet', 'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 9, 2, 2], 'dataset': 'CVC-ClinicDB',
        'img_size': args.img_size, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'grad_accum_steps': grad_accum_steps,
        'effective_batch_size': eff_batch, 'lr': args.lr, 'seed': args.seed,
        'amp': bool(args.amp), 'max_batches': max_batches or None,
        'spectral': {'loss': 'HighFrequencyConsistencyLoss',
                     'lambda_spectral': LAMBDA_SPECTRAL,
                     'cutoff_radius': 0.25,
                     'perturb_std': PERTURB_STD,
                     'hook': 'vmunet.layers[0].blocks[0]'},
        'device': str(device), 'checkpoint': save_path,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    }
    with open(os.path.join(args.output_dir, 'metadata_vmunet_cvc_regularized.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
