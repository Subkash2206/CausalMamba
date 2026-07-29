"""
Experiment 6: DC-Only Baseline and Boundary Error Analysis.

Closes two logical gaps identified in the paper review:
    1. What is the performance floor? (cutoff=0.01, DC-only)
    2. Where do errors concentrate? (boundary vs interior)

Configuration:
    - Dataset: ISIC2018 (50 images, same as Exp 2)
    - Model: VM-UNet pre-trained checkpoint
    - Protocol A: DC-only (cutoff=0.01) on all VSSBlocks
    - Protocol B: Boundary error maps for whole-network LP (cutoff=0.25)

Usage:
    python tools/experiment6_dc_baseline.py
"""

import sys
import os
import glob
import json
import csv
import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), '..'))


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
    """Compute boundary band masks at multiple widths."""
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


def compute_avr(fmap):
    B, C, H, W = fmap.shape
    fft = torch.fft.fft2(fmap)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = torch.abs(fft_shifted) ** 2
    cy, cx = H // 2, W // 2
    y = torch.arange(H).view(1, 1, H, 1)
    x = torch.arange(W).view(1, 1, 1, W)
    mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4)
    mask = mask.expand(B, C, H, W)
    high_freq_energy = (power * mask).sum()
    total_energy = power.sum()
    return (high_freq_energy / total_energy).item() if total_energy > 0 else 0.0


def main():
    from models.vmunet.vmunet import VMUNet
    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("Experiment 6: DC-Only Baseline & Boundary Error Analysis")
    print("=" * 80)
    print(f"Device: {device}")

    vm_unet_root = os.path.join(os.getcwd(), 'VM-UNet')
    img_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'images') + os.sep
    mask_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'masks') + os.sep
    ckpt_path = os.path.join(vm_unet_root, 'best-ckpt', 'best-vmunet-isic18.pth')
    n_images = 50
    _NC = {'num_classes': 1, 'input_channels': 3, 'depths': [2, 2, 2, 2],
           'depths_decoder': [2, 2, 2, 1], 'drop_path_rate': 0.2}

    if not os.path.isdir(img_dir):
        print(f"ERROR: Image directory not found at {img_dir}")
        sys.exit(1)

    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                        glob.glob(os.path.join(img_dir, '*.png')))[:n_images]
    mask_paths = []
    for imp in img_paths:
        mc = os.path.join(mask_dir, os.path.basename(imp))
        if not os.path.exists(mc):
            mc = mc.replace('.jpg', '.png').replace('.png', '_segmentation.png')
        mask_paths.append(mc)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    print("Loading model...")
    model_ref = VMUNet(**_NC).to(device)
    model_ref.eval()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
        mapped = {k: v for k, v in sd.items() if 'total_ops' not in k and 'total_params' not in k}
        model_ref.load_state_dict(mapped)
        print("Checkpoint loaded.")

    vssblock_names = [nm for nm, mod in model_ref.named_modules()
                       if 'VSSBlock' in str(type(mod))]

    # ======================================================================
    # Protocol A: DC-only baseline (cutoff=0.01)
    # ======================================================================
    print("\n" + "=" * 80)
    print("Protocol A: DC-Only Baseline (cutoff=0.01)")
    print("=" * 80)

    dc_interv = FrequencyIntervention(
        lambda h, w, dev, dt: lowpass_mask(h, w, 0.01, device=dev, dtype=dt)
    )
    id_interv = FrequencyIntervention(
        lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
    )

    dc_preds_list, dc_gts_list = [], []

    for cutoff_val, label in [(0.01, 'DC-only'), (0.25, 'LP(0.25)')]:
        print(f"\n  --- {label} ---")
        model = VMUNet(**_NC).to(device)
        model.load_state_dict(model_ref.state_dict())
        model.eval()
        interv = FrequencyIntervention(
            lambda h, w, dev, dt, c=cutoff_val: lowpass_mask(h, w, c, device=dev, dtype=dt)
        )
        handles = []
        _store = {}
        for nm, mod in model.named_modules():
            if 'VSSBlock' in str(type(mod)):
                def make_hook(name, iv):
                    def hook(module, inp, out):
                        if isinstance(out, tuple):
                            out = out[0]
                        is_vmamba = (out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768])
                        if is_vmamba:
                            fmap = out.permute(0, 3, 1, 2).contiguous()
                        else:
                            fmap = out
                        _store[name] = fmap.detach().cpu()
                        modified = iv(fmap)
                        if is_vmamba:
                            return modified.permute(0, 2, 3, 1).contiguous()
                        return modified
                    return hook
                handles.append(mod.register_forward_hook(make_hook(nm, interv)))

        preds_list, gts_list = [], []
        with torch.no_grad():
            for idx, imp in enumerate(img_paths):
                img = Image.open(imp).convert('RGB')
                inp = transform(img).unsqueeze(0).to(device)
                has_mask = idx < len(mask_paths) and os.path.exists(mask_paths[idx])
                if has_mask:
                    msk = Image.open(mask_paths[idx]).convert('L')
                    msk_t = mask_transform(msk)
                else:
                    msk_t = torch.zeros(1, 256, 256)
                _store.clear()
                pred = model(inp)
                preds_list.append(pred.squeeze(1).cpu().detach().numpy())
                gts_list.append(msk_t.squeeze(0).cpu().numpy())
        for h in handles:
            h.remove()
        del model

        preds_flat = np.concatenate([p.ravel() for p in preds_list])
        gts_flat = np.concatenate([g.ravel() for g in gts_list])
        metrics = compute_segmentation_metrics(preds_flat, gts_flat)
        mean_pred = np.mean(preds_flat)
        std_pred = np.std(preds_flat)

        print(f"    Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")
        print(f"    Sensitivity={metrics['sensitivity']:.4f}, Specificity={metrics['specificity']:.4f}")
        print(f"    Prediction stats: mean={mean_pred:.4f}, std={std_pred:.4f}")

        if cutoff_val == 0.01:
            dc_metrics = metrics
            dc_preds_list = preds_list
            dc_pred_mean = mean_pred
            dc_pred_std = std_pred

    # Check if DC-only enters degenerate regime
    print(f"\n  Degenerate regime check:")
    print(f"    Mean prediction value: {dc_pred_mean:.4f}")
    if dc_pred_mean < 0.05 or dc_pred_mean > 0.95:
        print(f"    → Degenerate: predictions are near-{dc_pred_mean:.2f} (near-uniform)")
    elif dc_pred_std < 0.05:
        print(f"    → Degenerate: predictions have very low variance ({dc_pred_std:.4f})")
    else:
        print(f"    → Non-degenerate: predictions vary across pixels (std={dc_pred_std:.4f})")

    # ======================================================================
    # Protocol B: Boundary error analysis for LP(0.25)
    # ======================================================================
    print("\n" + "=" * 80)
    print("Protocol B: Boundary Error Analysis (cutoff=0.25)")
    print("=" * 80)

    # Re-run LP(0.25) to get per-image predictions with masks
    model_lp = VMUNet(**_NC).to(device)
    model_lp.load_state_dict(model_ref.state_dict())
    model_lp.eval()
    lp_025 = FrequencyIntervention(
        lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
    )
    handles_lp = []
    _store_lp = {}
    for nm, mod in model_lp.named_modules():
        if 'VSSBlock' in str(type(mod)):
            def make_hook_lp(name):
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    is_vmamba = (out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768])
                    if is_vmamba:
                        fmap = out.permute(0, 3, 1, 2).contiguous()
                    else:
                        fmap = out
                    _store_lp[name] = fmap.detach().cpu()
                    modified = lp_025(fmap)
                    if is_vmamba:
                        return modified.permute(0, 2, 3, 1).contiguous()
                    return modified
                return hook
            handles_lp.append(mod.register_forward_hook(make_hook_lp(nm)))

    lp_preds_list, lp_gts_list = [], []
    lp_pred_raw = []

    with torch.no_grad():
        for idx, imp in enumerate(img_paths):
            img = Image.open(imp).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            has_mask = idx < len(mask_paths) and os.path.exists(mask_paths[idx])
            if has_mask:
                msk = Image.open(mask_paths[idx]).convert('L')
                msk_t = mask_transform(msk)
            else:
                msk_t = torch.zeros(1, 256, 256)
            _store_lp.clear()
            pred = model_lp(inp)
            pred_np = pred.squeeze(1).cpu().detach().numpy()
            lp_preds_list.append((pred_np[0] > 0.5).astype(np.uint8))
            lp_gts_list.append((msk_t[0].cpu().numpy() > 0.5).astype(np.uint8))
            lp_pred_raw.append(pred_np[0])
    for h in handles_lp:
        h.remove()
    del model_lp

    # Boundary error computation
    boundary_widths = [5, 10, 20]
    errors_boundary = {w: [] for w in boundary_widths}
    errors_interior = []
    errors_background = []

    for i in range(len(lp_preds_list)):
        gt = lp_gts_list[i]
        pred = lp_preds_list[i]
        if gt.sum() == 0:
            continue

        # Error map: 1 = false positive, -1 = false negative, 0 = correct
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)

        # Boundary bands
        bands = boundary_bands(gt, boundary_widths)

        # Interior = foreground minus widest boundary band
        interior = gt.copy()
        interior[bands[max(boundary_widths)] > 0] = 0

        # Background = everything outside widest boundary band
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

    print(f"\n  Mean error rate by region:")
    for w in boundary_widths:
        mean_b = np.mean(errors_boundary[w]) * 100 if errors_boundary[w] else 0
        print(f"    Boundary (±{w}px):    {mean_b:.2f}%")
    mean_int = np.mean(errors_interior) * 100 if errors_interior else 0
    mean_bg = np.mean(errors_background) * 100 if errors_background else 0
    print(f"    Interior (foreground): {mean_int:.2f}%")
    print(f"    Background (far):      {mean_bg:.2f}%")

    edge_ratio = mean_b / (mean_int + 1e-8)
    print(f"\n  Boundary-to-Interior error ratio: {edge_ratio:.2f}x")
    if edge_ratio > 2.0:
        print(f"  → Errors are disproportionately concentrated at boundaries (>2x).")
        print(f"  → Consistent with high frequencies being critical for boundary discrimination.")
    else:
        print(f"  → Errors are not strongly concentrated at boundaries.")

    # ======================================================================
    # Save results
    # ======================================================================
    _ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _out_dir = os.path.join(os.getcwd(), 'results', 'experiment6_dc_baseline')
    os.makedirs(_out_dir, exist_ok=True)

    def _to_python(val):
        if isinstance(val, (np.floating, np.integer)):
            return val.item()
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    results = {
        'dc_only': {
            'cutoff': 0.01,
            'dice': _to_python(dc_metrics['dice']),
            'iou': _to_python(dc_metrics['iou']),
            'accuracy': _to_python(dc_metrics['accuracy']),
            'sensitivity': _to_python(dc_metrics['sensitivity']),
            'specificity': _to_python(dc_metrics['specificity']),
            'pred_mean': _to_python(dc_pred_mean),
            'pred_std': _to_python(dc_pred_std),
            'degenerate': bool(dc_pred_std < 0.05 or dc_pred_mean < 0.05 or dc_pred_mean > 0.95),
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
            'experiment': 'Experiment 6: DC-Only Baseline & Boundary Error Analysis',
            'model': 'VM-UNet', 'dataset': 'ISIC2018',
            'num_images': n_images, 'device': str(device), 'timestamp': _ts,
        }
    }

    with open(os.path.join(_out_dir, f'metadata_{_ts}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {_out_dir}/")
    print()


if __name__ == '__main__':
    main()