"""
Experiment 12: UNet-ResNet50 Boundary Error Analysis on CVC-ClinicDB.

Replicates Experiment 9 (boundary analysis, whole-network LP 0.25) on the
CVC-ClinicDB dataset using CVCDataset (352x352, [0,1] normalization) and the
pre-trained UNet-ResNet50 CVC checkpoint.

Computes per-region error rates (Boundary +-5/10/20px, Interior, Background)
and the Boundary-to-Interior error ratio for cross-dataset comparison.

Usage:
    cd SpectralMamba
    python ..\\interventions\\experiments\\experiment12_resnet50_cvc_boundary.py \\
        --output_dir ..\\interventions\\results\\experiment12_resnet50_cvc_boundary --seed 42
"""

import sys, os, glob, json, csv, datetime, argparse, random
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy.ndimage import binary_erosion, distance_transform_edt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [_REPO, os.path.join(_REPO, 'SpectralMamba'),
          os.path.join(_REPO, 'tta_boundary_study')]:
    sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Helpers - IDENTICAL to Exp 9
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


def boundary_bands(mask_np, widths=[5, 10, 20]):
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    binary = (mask_np > 0.5).astype(np.uint8)
    eroded = binary_erosion(binary, iterations=1)
    boundary = binary.astype(float) - eroded.astype(float)
    dt = distance_transform_edt(1 - boundary)
    bands = {}
    for w in widths:
        bands[w] = (dt <= w).astype(np.uint8)
    return bands


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
    print('Experiment 12: UNet-ResNet50 CVC-ClinicDB Boundary Error Analysis (cutoff=0.25)')
    print('=' * 80)
    print(f'Device: {device}')

    ckpt_path = args.ckpt_path if args.ckpt_path else os.path.join(
        _REPO, 'tta_boundary_study', 'checkpoints', 'unet_cvc_best.pth')
    img_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
    mask_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')

    val_ds = CVCDataset(img_dir, mask_dir, split='val', img_size=352)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f'\nDataset: CVC-ClinicDB (validation split)')
    print(f'  Images: {len(val_ds)} ({img_dir})')
    print(f'  Resolution: 352x352, [0,1] normalization')

    print("\nLoading model...")
    model_ref = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                         in_channels=3, classes=1).to(device)
    model_ref.eval()
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        sd = ck.get('model_state_dict') or ck.get('state_dict') or ck if isinstance(ck, dict) else ck
        if not isinstance(sd, dict):
            sd = ck
        model_ref.load_state_dict(sd, strict=True)
        print('Checkpoint loaded.')
    else:
        print(f'WARNING: Checkpoint not found at {ckpt_path}. Random init.')

    # ==================================================================
    # Whole-network low-pass intervention (cutoff=0.25, all 9 stages)
    # ==================================================================
    print('\n' + '=' * 80)
    print('Whole-network Low-Pass Intervention (cutoff=0.25, all 9 stages)')
    print('=' * 80)

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
    lp_interv = FrequencyIntervention(lp_fn, check_nan=True)

    model_lp = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                        in_channels=3, classes=1).to(device)
    model_lp.load_state_dict(model_ref.state_dict(), strict=True)
    model_lp.eval()
    lp_handles = []
    for name, path in HOOK_TARGETS:
        lp_handles.append(
            _resolve_path(model_lp, path).register_forward_hook(_make_lp_hook(lp_interv)))

    lp_preds_list, lp_gts_list = [], []
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model_lp(imgs)
            prob = torch.sigmoid(logits)
            pred_np = prob.squeeze(1).cpu().detach().numpy()
            lp_preds_list.append((pred_np[0] > 0.5).astype(np.uint8))
            lp_gts_list.append((masks.squeeze(1)[0].cpu().numpy() > 0.5).astype(np.uint8))
            if (i + 1) % 30 == 0:
                print(f'  Low-pass: {i + 1}/{len(val_loader)}')
    for h in lp_handles:
        h.remove()
    del model_lp

    lp_preds_flat = np.concatenate([p.ravel() for p in lp_preds_list])
    lp_gts_flat = np.concatenate([g.ravel() for g in lp_gts_list])
    lp_metrics = compute_segmentation_metrics(lp_preds_flat, lp_gts_flat)
    print(f'    Dice={lp_metrics["dice"]:.4f}, IoU={lp_metrics["iou"]:.4f}')
    print(f'    Sensitivity={lp_metrics["sensitivity"]:.4f}, Specificity={lp_metrics["specificity"]:.4f}')

    # ==================================================================
    # Boundary error computation - IDENTICAL to Exp 9
    # ==================================================================
    boundary_widths = [5, 10, 20]
    errors_boundary = {w: [] for w in boundary_widths}
    errors_interior = []
    errors_background = []

    for i in range(len(lp_preds_list)):
        gt = lp_gts_list[i]
        pred = lp_preds_list[i]
        if gt.sum() == 0:
            continue

        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)

        bands = boundary_bands(gt, boundary_widths)

        interior = gt.copy()
        interior[bands[max(boundary_widths)] > 0] = 0

        background = 1 - gt.copy()
        background[bands[max(boundary_widths)] > 0] = 0

        for w in boundary_widths:
            b = bands[w]
            total_band_pixels = b.sum()
            if total_band_pixels > 0:
                band_errors = ((fp | fn) & (b > 0)).sum()
                errors_boundary[w].append(band_errors / total_band_pixels)

        if interior.sum() > 0:
            errors_interior.append(((fp | fn) & (interior > 0)).sum() / interior.sum())
        if background.sum() > 0:
            errors_background.append(((fp | fn) & (background > 0)).sum() / background.sum())

    print('\n  Mean error rate by region:')
    for w in boundary_widths:
        mean_b = np.mean(errors_boundary[w]) * 100 if errors_boundary[w] else 0
        print(f'    Boundary (+-{w}px):    {mean_b:.2f}%')
    mean_int = np.mean(errors_interior) * 100 if errors_interior else 0
    mean_bg = np.mean(errors_background) * 100 if errors_background else 0
    print(f'    Interior (foreground): {mean_int:.2f}%')
    print(f'    Background (far):      {mean_bg:.2f}%')

    edge_ratio = mean_b / (mean_int + 1e-8)
    print(f'\n  Boundary-to-Interior error ratio: {edge_ratio:.2f}x')
    if edge_ratio > 2.0:
        print('  -> Errors are disproportionately concentrated at boundaries (>2x).')
        print('  -> Consistent with high frequencies being critical for boundary discrimination.')
    else:
        print('  -> Errors are not strongly concentrated at boundaries.')

    # ==================================================================
    # Save results
    # ==================================================================
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = args.output_dir if args.output_dir else os.path.join(
        os.getcwd(), 'results', 'experiment12_resnet50_cvc_boundary')
    os.makedirs(out_dir, exist_ok=True)

    def _to_python(val):
        if isinstance(val, (np.floating, np.integer)):
            return val.item()
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    results = {
        'whole_network_lp': {
            'cutoff': 0.25,
            'dice': _to_python(lp_metrics['dice']),
            'iou': _to_python(lp_metrics['iou']),
            'accuracy': _to_python(lp_metrics['accuracy']),
            'sensitivity': _to_python(lp_metrics['sensitivity']),
            'specificity': _to_python(lp_metrics['specificity']),
        },
        'boundary_errors': {
            'cutoff': 0.25,
            'boundary_widths': boundary_widths,
            'error_rate_boundary_5': _to_python(np.mean(errors_boundary[5]) * 100) if errors_boundary[5] else 0.0,
            'error_rate_boundary_10': _to_python(np.mean(errors_boundary[10]) * 100) if errors_boundary[10] else 0.0,
            'error_rate_boundary_20': _to_python(np.mean(errors_boundary[20]) * 100) if errors_boundary[20] else 0.0,
            'error_rate_interior': _to_python(mean_int),
            'error_rate_background': _to_python(mean_bg),
            'boundary_interior_ratio': _to_python(edge_ratio),
        },
        'metadata': {
            'experiment': 'Experiment 12: UNet-ResNet50 CVC-ClinicDB Boundary Error Analysis',
            'model': 'UNet-ResNet50 (smp)', 'dataset': 'CVC-ClinicDB',
            'img_size': 352, 'num_images': len(val_ds), 'device': str(device),
            'seed': args.seed, 'timestamp': ts,
        }
    }

    with open(os.path.join(out_dir, f'metadata_{ts}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(out_dir, f'boundary_summary_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region', 'error_rate_pct'])
        for wd in boundary_widths:
            mean_b = np.mean(errors_boundary[wd]) * 100 if errors_boundary[wd] else 0
            w.writerow([f'boundary_{wd}px', f'{mean_b:.4f}'])
        w.writerow(['interior', f'{mean_int:.4f}'])
        w.writerow(['background', f'{mean_bg:.4f}'])
        w.writerow(['boundary_interior_ratio', f'{edge_ratio:.4f}'])
    print(f'\nResults saved to {out_dir}/')
    print()


def _make_lp_hook(intervention):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        return intervention(out)
    return hook


if __name__ == '__main__':
    main()