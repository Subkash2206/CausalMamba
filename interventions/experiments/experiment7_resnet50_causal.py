"""
Experiment 7: UNet-ResNet50 Causal Frequency Intervention (Phase 1).

Cross-architecture comparison vs Phase-0 VM-UNet (Exp 2). Uses the same
FFT -> Mask -> IFFT pipeline (FrequencyIntervention) and the same ISIC2018
protocol (50 images) as Experiment 2.

Protocol A: Baseline (identity intervention - no spectral change).
Protocol B: Low-pass intervention (cutoff=0.25) at every mapped stage
            (encoder.layer1..4[-1], decoder.blocks[0..4]).

Usage:
    cd SpectralMamba
    python ..\\interventions\\experiments\\experiment7_resnet50_causal.py \\
        --output_dir ..\\interventions\\results\\experiment7_resnet50 --seed 42
"""

import sys, os, glob, json, csv, datetime, argparse, random
from collections import defaultdict

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix

# Path setup for the `interventions` package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'SpectralMamba'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Evaluation helpers - IDENTICAL to Phase 0 (Exp 2)
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
    """Average Volume Ratio - identical AVR definition to Phase 0."""
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
    """Resolve 'encoder.layer1[-1]' or 'decoder.blocks[3]' to a module."""
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


# Mapped semantic boundaries (same paths as resnet50_hook_mapping.py)
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
    _ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _ap.add_argument('--output_dir', default=None)
    _ap.add_argument('--seed', type=int, default=None)
    _ap.add_argument('--ckpt_path', default=None)
    _args, _ = _ap.parse_known_args()
    if _args.seed is not None:
        random.seed(_args.seed)
        torch.manual_seed(_args.seed)
        np.random.seed(_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_args.seed)

    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask
    import segmentation_models_pytorch as smp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('Experiment 7: UNet-ResNet50 Causal Low-Pass Intervention (cutoff=0.25)')
    print('=' * 80)
    print(f'Device: {device}')

    vm_unet_root = os.path.join(os.getcwd(), 'VM-UNet')
    img_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'images') + os.sep
    mask_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'masks') + os.sep
    ckpt_path = _args.ckpt_path if _args.ckpt_path else os.path.join(
        vm_unet_root, 'best-ckpt', 'best-unet-isic18.pth')
    n_images = 50

    if not os.path.isdir(img_dir):
        print(f'\nERROR: Image directory not found at {img_dir}')
        sys.exit(1)

    img_paths = sorted(
        glob.glob(os.path.join(img_dir, '*.jpg')) +
        glob.glob(os.path.join(img_dir, '*.png'))
    )[:n_images]
    mask_paths = []
    for imp in img_paths:
        mc = os.path.join(mask_dir, os.path.basename(imp))
        if not os.path.exists(mc):
            mc = mc.replace('.jpg', '.png').replace('.png', '_segmentation.png')
        mask_paths.append(mc)
    n_masks = sum(1 for p in mask_paths if os.path.exists(p))
    print(f'\nDataset: ISIC2018')
    print(f'  Images: {len(img_paths)} ({img_dir})')
    print(f'  Masks:  {n_masks}/{len(img_paths)} found ({mask_dir})')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # ------------------------------------------------------------------
    # Model + checkpoint
    # ------------------------------------------------------------------
    print("\nInitialising UNet-ResNet50 (smp.Unet, encoder_name='resnet50')...")
    model = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                     in_channels=3, classes=1).to(device)
    model.eval()

    if os.path.exists(ckpt_path):
        print(f'Loading checkpoint from {ckpt_path}...')
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
        print('Checkpoint loaded.')
    else:
        print(f'WARNING: Checkpoint not found at {ckpt_path}. Random init.')

    # ==================================================================
    # Protocol A: Baseline (identity intervention)
    # ==================================================================
    print('\n' + '-' * 80)
    print('Protocol A: Baseline (identity intervention - no spectral change)')
    print('-' * 80)

    model_bl = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                        in_channels=3, classes=1).to(device)
    model_bl.load_state_dict(model.state_dict(), strict=True)
    model_bl.eval()

    identity_fn = lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
    intervention_id = FrequencyIntervention(identity_fn, check_nan=True)
    bl_hook_store = {}

    def make_bl_hook(name, intervention):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            bl_hook_store[name] = out.detach().cpu()
            return intervention(out)
        return hook

    bl_handles = []
    for name, path in HOOK_TARGETS:
        mod = _resolve_path(model_bl, path)
        bl_handles.append(mod.register_forward_hook(make_bl_hook(name, intervention_id)))

    bl_preds, bl_gts = [], []
    with torch.no_grad():
        for idx, imp in enumerate(img_paths):
            inp = transform(Image.open(imp).convert('RGB')).unsqueeze(0).to(device)
            has_mask = idx < len(mask_paths) and os.path.exists(mask_paths[idx])
            if has_mask:
                msk_t = mask_transform(Image.open(mask_paths[idx]).convert('L'))
            else:
                msk_t = torch.zeros(1, 256, 256)
            bl_hook_store.clear()
            logits = model_bl(inp)
            prob = torch.sigmoid(logits)
            bl_preds.append(prob.squeeze(1).cpu().detach().numpy())
            bl_gts.append(msk_t.squeeze(0).cpu().numpy())
            if (idx + 1) % 10 == 0:
                print(f'  Baseline: {idx + 1}/{len(img_paths)}')

    for h in bl_handles:
        h.remove()

    bl_metrics = compute_segmentation_metrics(
        np.concatenate([p.ravel() for p in bl_preds]),
        np.concatenate([g.ravel() for g in bl_gts]))
    print(f"\n  Baseline Dice: {bl_metrics['dice']:.4f}")
    print(f"  Baseline IoU:  {bl_metrics['iou']:.4f}")

    # ==================================================================
    # Protocol B: Low-pass intervention (cutoff=0.25, all mapped hooks)
    # ==================================================================
    print('\n' + '-' * 80)
    print('Protocol B: Low-pass intervention (cutoff=0.25, all mapped hooks)')
    print('-' * 80)

    model_lp = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                        in_channels=3, classes=1).to(device)
    model_lp.load_state_dict(model.state_dict(), strict=True)
    model_lp.eval()

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
    intervention_lp = FrequencyIntervention(lp_fn, check_nan=True)
    lp_hook_store = {}
    lp_avr_before = defaultdict(list)
    lp_avr_after = defaultdict(list)

    def make_lp_hook(name, intervention):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            before = out.detach().cpu()
            modified = intervention(out)
            after = modified.detach().cpu()
            lp_hook_store[name] = (before, after)
            return modified
        return hook

    lp_handles = []
    for name, path in HOOK_TARGETS:
        mod = _resolve_path(model_lp, path)
        lp_handles.append(mod.register_forward_hook(make_lp_hook(name, intervention_lp)))

    lp_preds, lp_gts = [], []
    with torch.no_grad():
        for idx, imp in enumerate(img_paths):
            inp = transform(Image.open(imp).convert('RGB')).unsqueeze(0).to(device)
            has_mask = idx < len(mask_paths) and os.path.exists(mask_paths[idx])
            if has_mask:
                msk_t = mask_transform(Image.open(mask_paths[idx]).convert('L'))
            else:
                msk_t = torch.zeros(1, 256, 256)
            lp_hook_store.clear()
            logits = model_lp(inp)
            prob = torch.sigmoid(logits)
            lp_preds.append(prob.squeeze(1).cpu().detach().numpy())
            lp_gts.append(msk_t.squeeze(0).cpu().numpy())

            for nm in lp_hook_store:
                before, after = lp_hook_store[nm]
                lp_avr_before[nm].append(compute_avr(before))
                lp_avr_after[nm].append(compute_avr(after))
            if (idx + 1) % 10 == 0:
                print(f'  Low-pass: {idx + 1}/{len(img_paths)}')

    for h in lp_handles:
        h.remove()

    lp_metrics = compute_segmentation_metrics(
        np.concatenate([p.ravel() for p in lp_preds]),
        np.concatenate([g.ravel() for g in lp_gts]))
    print(f"\n  Low-pass Dice: {lp_metrics['dice']:.4f}")
    print(f"  Low-pass IoU:  {lp_metrics['iou']:.4f}")

    # ==================================================================
    # Results - save to interventions/results/experiment7_resnet50/
    # ==================================================================
    print('\n' + '=' * 80)
    print('Results')
    print('=' * 80)

    _timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _results_dir = _args.output_dir if _args.output_dir else os.path.join(
        os.getcwd(), 'results', 'experiment7_resnet50')
    os.makedirs(_results_dir, exist_ok=True)

    _meta = {
        'experiment': 'Experiment 7: UNet-ResNet50 Causal Low-Pass Intervention',
        'model': 'UNet-ResNet50 (smp)', 'encoder_name': 'resnet50',
        'checkpoint': ckpt_path, 'dataset': 'ISIC2018',
        'dataset_path': img_dir, 'intervention_type': 'lowpass',
        'cutoff': 0.25, 'num_images': len(img_paths), 'num_masks': n_masks,
        'device': str(device), 'seed': _args.seed, 'timestamp': _timestamp,
        'mapped_hooks': HOOK_TARGETS,
    }
    _meta_path = os.path.join(_results_dir, f'metadata_{_timestamp}.json')
    with open(_meta_path, 'w') as f:
        json.dump(_meta, f, indent=2)
    print(f'\nMetadata saved to {_meta_path}')

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

    _csv_path = os.path.join(_results_dir, f'per_layer_avr_{_timestamp}.csv')
    with open(_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Stage', 'AVR_Before', 'AVR_After', 'Delta_AVR', 'Reduction_Pct'])
        for nm in sorted(lp_avr_before.keys()):
            mean_bf = np.mean(lp_avr_before[nm])
            mean_af = np.mean(lp_avr_after[nm])
            delta = mean_af - mean_bf
            pct = ((mean_bf - mean_af) / (mean_bf + 1e-12)) * 100.0
            w.writerow([nm, f'{mean_bf:.6f}', f'{mean_af:.6f}', f'{delta:.2e}', f'{pct:.2f}'])
    print(f'\nPer-layer AVR saved to {_csv_path}')

    _metrics_path = os.path.join(_results_dir, f'metrics_{_timestamp}.csv')
    with open(_metrics_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Baseline', 'LowPass', 'Delta', 'DeltaPct'])
        for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
            blv, lpv = bl_metrics[metric], lp_metrics[metric]
            diff = lpv - blv
            pct_ch = (diff / (blv + 1e-12)) * 100.0
            w.writerow([metric, f'{blv:.6f}', f'{lpv:.6f}', f'{diff:.2e}', f'{pct_ch:.2f}'])
    print(f'Metrics saved to {_metrics_path}')

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