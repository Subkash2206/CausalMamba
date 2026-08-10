"""
Experiment 15: VM-UNet CVC-ClinicDB Boundary-vs-Interior Error Analysis (Phase 3.3).

Completes the boundary-error grid (VM-UNet x CVC) after Experiments 9 (ResNet50 x ISIC),
6 (VM-UNet x ISIC) and 12 (ResNet50 x CVC). Computes error rates partitioned into
"boundary" (GT contour +-5/10/20 px), "interior" (foreground away from boundary) and
"background" (away from boundary) regions, for the frozen VM-UNet CVC checkpoint:

  --mode baseline   : non-intervened model (identity forward pass)
  --mode intervened : whole-network low-pass intervention (--cutoff_radius 0.25)
                      applied to all 30 VSSBlocks

The boundary vs interior delta quantifies how much high-frequency removal
causally harms boundary pixels.

Usage:
    cd SpectralMamba
    python ..\\interventions\\experiments\\experiment15_vmunet_cvc_boundary.py ^
        --mode baseline --output_dir ..\\interventions\\results\\experiment15_vmunet_cvc_boundary_baseline
    python ..\\interventions\\experiments\\experiment15_vmunet_cvc_boundary.py ^
        --mode intervened --cutoff_radius 0.25 --output_dir ..\\interventions\\results\\experiment15_vmunet_cvc_boundary_intervened
"""

import sys, os, argparse, random, json, csv, datetime

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
# Helpers - identical to Experiments 12 / 14
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
    """Return dilated bands around the GT contour (in pixels)."""
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


# 30 VSSBlock names - same mapping as Experiment 14
_VMAMBA_LAYER_NAMES = [
    "encoder.block1.blk0", "encoder.block1.blk1",
    "encoder.block2.blk0", "encoder.block2.blk1",
    "encoder.block3.blk0", "encoder.block3.blk1", "encoder.block3.blk2",
    "encoder.block3.blk3", "encoder.block3.blk4", "encoder.block3.blk5",
    "encoder.block3.blk6", "encoder.block3.blk7", "encoder.block3.blk8",
    "encoder.block4.blk0", "encoder.block4.blk1",
    "bridge.blk0", "bridge.blk1",
    "decoder.block1.blk0", "decoder.block1.blk1", "decoder.block1.blk2",
    "decoder.block1.blk3", "decoder.block1.blk4", "decoder.block1.blk5",
    "decoder.block1.blk6", "decoder.block1.blk7", "decoder.block1.blk8",
    "decoder.block2.blk0", "decoder.block2.blk1",
    "decoder.block3.blk0", "decoder.block3.blk1",
]
assert len(_VMAMBA_LAYER_NAMES) == 30, f"Expected 30 layer names, got {len(_VMAMBA_LAYER_NAMES)}"


def make_lp_hook(name, interv):
    """VM-UNet VSSBlock hook (same as Exp 14): permute NHWC<->NCHW before intervening."""
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        is_vm = (out.dim() == 4 and out.shape[-1] in {96, 192, 384, 768})
        if is_vm:
            fmap = out.permute(0, 3, 1, 2).contiguous()
        else:
            fmap = out
        modified = interv(fmap)
        return modified.permute(0, 2, 3, 1).contiguous() if is_vm else modified
    return hook


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--mode', choices=['baseline', 'intervened'], default='baseline',
                    help="'baseline' = no intervention; 'intervened' = low-pass on all 30 VSSBlocks")
    ap.add_argument('--cutoff_radius', type=float, default=0.25,
                    help='Low-pass cutoff radius as a fraction of Nyquist (default: 0.25)')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--ckpt_path', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args, _ = ap.parse_known_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from models.vmunet.vmunet import VMUNet
    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask
    from src.datasets.cvc_dataset import CVCDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('Experiment 15: VM-UNet CVC-ClinicDB Boundary-vs-Interior Error Analysis')
    print(f'Mode: {args.mode}  |  cutoff_radius: {args.cutoff_radius}')
    print('=' * 80)
    print(f'Device: {device}')

    # ------------------------------------------------------------------
    # CVC dataset + frozen checkpoint
    # ------------------------------------------------------------------
    ckpt_path = args.ckpt_path if args.ckpt_path else os.path.join(
        _REPO, 'tta_boundary_study', 'checkpoints', 'best-vmunet-cvc.pth')
    img_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
    mask_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')

    val_ds = CVCDataset(img_dir, mask_dir, split='val', img_size=256)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f'\nDataset: CVC-ClinicDB (validation split), {len(val_ds)} images @ 256x256')

    # ------------------------------------------------------------------
    # Canonical 30-block VM-UNet + frozen weights
    # ------------------------------------------------------------------
    _NC = {'num_classes': 1, 'input_channels': 3,
           'depths': [2, 2, 9, 2], 'depths_decoder': [2, 9, 2, 2],
           'drop_path_rate': 0.2}
    model = VMUNet(**_NC).to(device)
    model.eval()

    if os.path.exists(ckpt_path):
        print(f'Loading frozen checkpoint: {ckpt_path}')
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and 'vmunet.layers.0.blocks.0.ln_1.weight' not in sd:
            sd = sd.get('model_state_dict') or sd.get('state_dict') or sd
        model.load_state_dict(sd, strict=True)
        print('Checkpoint loaded (strict=True).')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    # ------------------------------------------------------------------
    # Optional: whole-network low-pass intervention
    # ------------------------------------------------------------------
    if args.mode == 'intervened':
        print(f'\nAttaching low-pass intervention (cutoff={args.cutoff_radius}) to all 30 VSSBlocks...')
        lp_fn = lambda h, w, dev, dt: lowpass_mask(
            h, w, args.cutoff_radius, device=dev, dtype=dt)
        interv_lp = FrequencyIntervention(lp_fn, check_nan=True)
        lp_handles = []
        blk_idx = 0
        for nm, mod in model.named_modules():
            if 'VSSBlock' in type(mod).__name__:
                hname = _VMAMBA_LAYER_NAMES[blk_idx] if blk_idx < len(_VMAMBA_LAYER_NAMES) else f'vss_{blk_idx}'
                lp_handles.append(mod.register_forward_hook(make_lp_hook(hname, interv_lp)))
                blk_idx += 1
        print(f'  Hooked {blk_idx} VSSBlocks.')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print(f'\n--- Inference ({args.mode}) ---')
    preds_list, gts_list = [], []
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            probs = model(imgs)                      # VMUNet returns probabilities in [0,1]
            preds_list.append((probs.squeeze(1)[0].cpu().numpy() > 0.5).astype(np.uint8))
            gts_list.append((masks.squeeze(1)[0].cpu().numpy() > 0.5).astype(np.uint8))
            if (i + 1) % 30 == 0:
                print(f'  {i + 1}/{len(val_loader)}')

    if args.mode == 'intervened':
        for h in lp_handles:
            h.remove()

    preds_flat = np.concatenate([p.ravel() for p in preds_list])
    gts_flat = np.concatenate([g.ravel() for g in gts_list])
    seg_metrics = compute_segmentation_metrics(preds_flat, gts_flat)
    print(f'\nSegmentation: Dice={seg_metrics["dice"]:.4f}, IoU={seg_metrics["iou"]:.4f}, '
          f'Acc={seg_metrics["accuracy"]:.4f}, '
          f'Sens={seg_metrics["sensitivity"]:.4f}, Spec={seg_metrics["specificity"]:.4f}')

    # ------------------------------------------------------------------
    # Boundary / interior / background error rates (same as Exp 12)
    # ------------------------------------------------------------------
    boundary_widths = [5, 10, 20]
    errors_boundary = {w: [] for w in boundary_widths}
    errors_interior = []
    errors_background = []

    for i in range(len(preds_list)):
        gt = gts_list[i]
        pred = preds_list[i]
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
        print(f'    Boundary (+-{w}px):  {mean_b:.2f}%')
    mean_int = np.mean(errors_interior) * 100 if errors_interior else 0
    mean_bg = np.mean(errors_background) * 100 if errors_background else 0
    print(f'    Interior (foreground): {mean_int:.2f}%')
    print(f'    Background (far):      {mean_bg:.2f}%')

    edge_ratio = mean_b / (mean_int + 1e-8)
    print(f'\n  Boundary-to-Interior error ratio: {edge_ratio:.2f}x')

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = args.output_dir if args.output_dir else os.path.join(
        os.path.dirname(__file__), '..', 'results', 'experiment15_vmunet_cvc_boundary')
    os.makedirs(out_dir, exist_ok=True)

    def _to_python(val):
        if isinstance(val, (np.floating, np.integer)):
            return val.item()
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    results = {
        'mode': args.mode,
        'cutoff_radius': args.cutoff_radius,
        'segmentation': {k: _to_python(v) for k, v in seg_metrics.items()},
        'boundary_errors': {
            'boundary_widths': boundary_widths,
            'error_rate_boundary_5': _to_python(np.mean(errors_boundary[5]) * 100) if errors_boundary[5] else 0.0,
            'error_rate_boundary_10': _to_python(np.mean(errors_boundary[10]) * 100) if errors_boundary[10] else 0.0,
            'error_rate_boundary_20': _to_python(np.mean(errors_boundary[20]) * 100) if errors_boundary[20] else 0.0,
            'error_rate_interior': _to_python(mean_int),
            'error_rate_background': _to_python(mean_bg),
            'boundary_interior_ratio': _to_python(edge_ratio),
        },
        'metadata': {
            'experiment': 'Experiment 15: VM-UNet CVC-ClinicDB Boundary-vs-Interior Error Analysis',
            'model': 'VM-UNet (30-block)', 'dataset': 'CVC-ClinicDB',
            'checkpoint': ckpt_path, 'img_size': 256, 'num_images': len(val_ds),
            'device': str(device), 'seed': args.seed, 'timestamp': ts,
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


if __name__ == '__main__':
    main()

