"""
Experiment 10: UNet-ResNet50 Whole-Network Causal Intervention on CVC-ClinicDB.

Replicates Experiment 7 (whole-network low-pass, cutoff=0.25) on the
CVC-ClinicDB dataset using CVCDataset (352x352, [0,1] normalization) and the
pre-trained UNet-ResNet50 CVC checkpoint.

Usage:
    cd SpectralMamba
    python ..\\interventions\\experiments\\experiment10_resnet50_cvc_causal.py \\
        --output_dir ..\\interventions\\results\\experiment10_resnet50_cvc_causal --seed 42
"""

import sys, os, glob, json, csv, datetime, argparse, random
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_sys_paths = [
    _REPO,
    os.path.join(_REPO, 'SpectralMamba'),
    os.path.join(_REPO, 'tta_boundary_study'),
]
for p in _sys_paths:
    sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Helpers - same definitions as Exps 7-9
# ---------------------------------------------------------------------------

def compute_segmentation_metrics(preds_flat, gts_flat, threshold=0.5):
    y_pre = np.where(preds_flat >= threshold, 1, 0)
    y_true = np.where(gts_flat >= 0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    total = float(np.sum(confusion))
    return {
        'dice': float(2 * TP) / float(2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0,
        'iou': float(TP) / float(TP + FP + FN) if (TP + FP + FN) > 0 else 0.0,
        'accuracy': float(TN + TP) / total if total > 0 else 0.0,
        'sensitivity': float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0,
        'specificity': float(TN) / float(TN + FP) if (TN + FP) > 0 else 0.0,
    }


def compute_avr(fmap):
    B, C, H, W = fmap.shape
    fft = torch.fft.fft2(fmap)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = torch.abs(fft_shifted) ** 2
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device='cpu').view(1, 1, H, 1)
    x = torch.arange(W, device='cpu').view(1, 1, 1, W)
    mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4)
    mask = mask.expand(B, C, H, W)
    high_freq_energy = (power * mask).sum()
    total_energy = power.sum()
    return (high_freq_energy / total_energy).item() if total_energy > 0 else 0.0


def _resolve_path(model, path):
    if '[-1]' in path:
        base, _ = path.rsplit('[-1]', 1)
        module = model
        for part in base.split('.'):
            module = getattr(module, part)
        return module[-1]
    module = model
    for chunk in path.split('.'):
        if '[' in chunk:
            name, idx = chunk.split('[')
            module = getattr(module, name)[int(idx.rstrip(']'))]
        else:
            module = getattr(module, chunk)
    return module


HOOK_TARGETS = [
    ('encoder.block1', 'encoder.layer1[-1]'),
    ('encoder.block2', 'encoder.layer2[-1]'),
    ('encoder.block3', 'encoder.layer3[-1]'),
    ('encoder.block4', 'encoder.layer4[-1]'),
    ('bridge',         'decoder.blocks[0]'),
    ('decoder.block1', 'decoder.blocks[1]'),
    ('decoder.block2', 'decoder.blocks[2]'),
    ('decoder.block3', 'decoder.blocks[3]'),
    ('decoder.block4', 'decoder.blocks[4]'),
]


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--ckpt_path', default=None)
    args, _ = ap.parse_known_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask
    from src.datasets.cvc_dataset import CVCDataset
    import segmentation_models_pytorch as smp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('Experiment 10: UNet-ResNet50 CVC-ClinicDB Whole-Network Causal (cutoff=0.25)')
    print('=' * 80)
    print(f'Device: {device}')

    ckpt_path = args.ckpt_path if args.ckpt_path else os.path.join(
        _REPO, 'tta_boundary_study', 'checkpoints', 'unet_cvc_best.pth')
    img_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
    mask_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')

    # CVC val split (matches the checkpoint's evaluation protocol)
    val_ds = CVCDataset(img_dir, mask_dir, split='val', img_size=352)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f'\nDataset: CVC-ClinicDB (validation split)')
    print(f'  Images: {len(val_ds)} ({img_dir})')
    print(f'  Resolution: 352x352, [0,1] normalization')

    print("\nInitialising UNet-ResNet50 (smp.Unet, encoder_name='resnet50')...")
    model = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                     in_channels=3, classes=1).to(device)
    model.eval()
    if os.path.exists(ckpt_path):
        print(f'Loading checkpoint from {ckpt_path}...')
        ck = torch.load(ckpt_path, map_location=device)
        sd = ck.get('model_state_dict') or ck.get('state_dict') or ck \
            if isinstance(ck, dict) else ck
        if not isinstance(sd, dict):
            sd = ck
        model.load_state_dict(sd, strict=True)
        print('Checkpoint loaded.')
    else:
        print(f'WARNING: Checkpoint not found at {ckpt_path}. Random init.')

    # ==================================================================
    # Protocol A: Baseline (identity)
    # ==================================================================
    print('\n' + '-' * 80)
    print('Protocol A: Baseline (identity intervention)')
    print('-' * 80)
    model_bl = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                        in_channels=3, classes=1).to(device)
    model_bl.load_state_dict(model.state_dict(), strict=True)
    model_bl.eval()

    id_fn = lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
    interv_id = FrequencyIntervention(id_fn, check_nan=True)
    bl_store = {}

    def make_bl_hook(name, interv):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            bl_store[name] = out.detach().cpu()
            return interv(out)
        return hook

    bl_handles = []
    for name, path in HOOK_TARGETS:
        bl_handles.append(_resolve_path(model_bl, path).register_forward_hook(
            make_bl_hook(name, interv_id)))

    bl_preds, bl_gts = [], []
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            bl_store.clear()
            logits = model_bl(imgs)
            prob = torch.sigmoid(logits)
            bl_preds.append(prob.squeeze(1).cpu().detach().numpy())
            bl_gts.append(masks.squeeze(1).cpu().numpy())
            if (i + 1) % 30 == 0:
                print(f'  Baseline: {i + 1}/{len(val_loader)}')
    for h in bl_handles:
        h.remove()

    bl_metrics = compute_segmentation_metrics(
        np.concatenate([p.ravel() for p in bl_preds]),
        np.concatenate([g.ravel() for g in bl_gts]))
    print(f"\n  Baseline Dice: {bl_metrics['dice']:.4f}")
    print(f"  Baseline IoU:  {bl_metrics['iou']:.4f}")

    # ==================================================================
    # Protocol B: Low-pass intervention (cutoff=0.25, all 9 hooks)
    # ==================================================================
    print('\n' + '-' * 80)
    print('Protocol B: Low-pass intervention (cutoff=0.25, all mapped hooks)')
    print('-' * 80)
    model_lp = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                        in_channels=3, classes=1).to(device)
    model_lp.load_state_dict(model.state_dict(), strict=True)
    model_lp.eval()

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
    interv_lp = FrequencyIntervention(lp_fn, check_nan=True)
    lp_store = {}
    lp_avr_before = defaultdict(list)
    lp_avr_after = defaultdict(list)

    def make_lp_hook(name, interv):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            before = out.detach().cpu()
            modified = interv(out)
            after = modified.detach().cpu()
            lp_store[name] = (before, after)
            return modified
        return hook

    lp_handles = []
    for name, path in HOOK_TARGETS:
        lp_handles.append(_resolve_path(model_lp, path).register_forward_hook(
            make_lp_hook(name, interv_lp)))

    lp_preds, lp_gts = [], []
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            lp_store.clear()
            logits = model_lp(imgs)
            prob = torch.sigmoid(logits)
            lp_preds.append(prob.squeeze(1).cpu().detach().numpy())
            lp_gts.append(masks.squeeze(1).cpu().numpy())
            for nm in lp_store:
                b, a = lp_store[nm]
                lp_avr_before[nm].append(compute_avr(b))
                lp_avr_after[nm].append(compute_avr(a))
            if (i + 1) % 30 == 0:
                print(f'  Low-pass: {i + 1}/{len(val_loader)}')
    for h in lp_handles:
        h.remove()

    lp_metrics = compute_segmentation_metrics(
        np.concatenate([p.ravel() for p in lp_preds]),
        np.concatenate([g.ravel() for g in lp_gts]))
    print(f"\n  Low-pass Dice: {lp_metrics['dice']:.4f}")
    print(f"  Low-pass IoU:  {lp_metrics['iou']:.4f}")

    # ==================================================================
    # Results
    # ==================================================================
    print('\n' + '=' * 80)
    print('Results')
    print('=' * 80)

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = args.output_dir if args.output_dir else os.path.join(
        os.getcwd(), 'results', 'experiment10_resnet50_cvc_causal')
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        'experiment': 'Experiment 10: UNet-ResNet50 CVC-ClinicDB Whole-Network Causal',
        'model': 'UNet-ResNet50 (smp)', 'dataset': 'CVC-ClinicDB',
        'checkpoint': ckpt_path, 'intervention_type': 'lowpass',
        'cutoff': 0.25, 'img_size': 352, 'num_images': len(val_ds),
        'device': str(device), 'seed': args.seed, 'timestamp': ts,
        'mapped_hooks': HOOK_TARGETS,
    }
    with open(os.path.join(out_dir, f'metadata_{ts}.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nMetadata saved to {os.path.join(out_dir, f"metadata_{ts}.json")}')

    print('\n--- Per-Layer Spectral Statistics (9 mapped stages) ---')
    print(f"{'Stage':<20} | {'AVR Before':<12} | {'AVR After':<12} | {'Delta AVR':<14} | {'Reduction %':<10}")
    print('-' * 75)
    deltas_all = []
    for nm in sorted(lp_avr_before.keys()):
        mean_bf = np.mean(lp_avr_before[nm])
        mean_af = np.mean(lp_avr_after[nm])
        delta = mean_af - mean_bf
        pct = ((mean_bf - mean_af) / (mean_bf + 1e-12)) * 100.0
        deltas_all.append(abs(delta))
        print(f"{nm:<20} | {mean_bf:<12.6f} | {mean_af:<12.6f} | {delta:<+14.2e} | {pct:<+10.2f}%")

    print('\n--- Segmentation Metrics ---')
    print(f"{'Metric':<20} {'Baseline':<12} {'Low-pass':<12} {'Delta':<14} {'Delta%':<10}")
    print('-' * 70)
    for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
        blv, lpv = bl_metrics[metric], lp_metrics[metric]
        diff = lpv - blv
        pct_ch = (diff / (blv + 1e-12)) * 100.0
        print(f"{metric:<20} {blv:<12.6f} {lpv:<12.6f} {diff:<+14.2e} {pct_ch:<+10.2f}%")

    with open(os.path.join(out_dir, f'per_layer_avr_{ts}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Stage', 'AVR_Before', 'AVR_After', 'Delta_AVR', 'Reduction_Pct'])
        for nm in sorted(lp_avr_before.keys()):
            mean_bf = np.mean(lp_avr_before[nm])
            mean_af = np.mean(lp_avr_after[nm])
            delta = mean_af - mean_bf
            pct = ((mean_bf - mean_af) / (mean_bf + 1e-12)) * 100.0
            w.writerow([nm, f'{mean_bf:.6f}', f'{mean_af:.6f}', f'{delta:.2e}', f'{pct:.2f}'])

    with open(os.path.join(out_dir, f'metrics_{ts}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Baseline', 'LowPass', 'Delta', 'DeltaPct'])
        for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
            blv, lpv = bl_metrics[metric], lp_metrics[metric]
            diff = lpv - blv
            pct_ch = (diff / (blv + 1e-12)) * 100.0
            w.writerow([metric, f'{blv:.6f}', f'{lpv:.6f}', f'{diff:.2e}', f'{pct_ch:.2f}'])

    print(f'\nPer-layer AVR + metrics saved to {out_dir}')

    print('\n--- Summary ---')
    print(f"  Baseline Dice: {bl_metrics['dice']:.4f}")
    print(f"  Low-pass Dice: {lp_metrics['dice']:.4f}")
    print(f"  Dice change:   {lp_metrics['dice'] - bl_metrics['dice']:+.4f}")
    print(f"  IoU change:    {lp_metrics['iou'] - bl_metrics['iou']:+.4f}")
    mean_delta = np.mean(deltas_all) if deltas_all else 0.0
    max_delta = max(deltas_all) if deltas_all else 0.0
    print(f'  Mean |Delta AVR| across 9 stages: {mean_delta:.2e}')
    print(f'  Maximum |Delta AVR|:              {max_delta:.2e}')
    print()


if __name__ == '__main__':
    main()