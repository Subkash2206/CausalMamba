"""
Experiment 2: Real Dataset Low-Pass Intervention.

Evaluates the causal effect of low-pass feature interventions on real
segmentation performance using the ISIC2018 dataset and the pre-trained
VM-UNet checkpoint from SpectralMamba.

Protocol:
    - Dataset: ISIC2018 (same as original avr_analysis.py)
    - Model: VM-UNet with pre-trained checkpoint
    - Intervention: Low-pass mask, cutoff=0.25, on all VSSBlock layers
    - Metrics: Dice, IoU, accuracy, sensitivity, specificity (+ BF1, HD95
      if boundary_metrics module is available)
    - Per-layer: AVR_before, AVR_after, ΔAVR

Usage:
    python tools/experiment2_real_lowpass.py

    Expects:
        - best-ckpt/best-vmunet-isic18.pth  (model weights)
        - data/isic18/train/images/          (input images)
        - data/isic18/train/masks/           (ground truth masks, same filename)
"""

import sys
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix

# Add project root paths so that subsequent imports inside main() resolve.
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'tta_boundary_study'))


# ---------------------------------------------------------------------------
# Evaluation helpers (matching SpectralMamba/engine.py convention)
# ---------------------------------------------------------------------------

def compute_segmentation_metrics(preds_flat: np.ndarray,
                                  gts_flat: np.ndarray,
                                  threshold: float = 0.5) -> dict:
    """Compute segmentation metrics from flattened predictions and ground truth.

    Matches the protocol in SpectralMamba/engine.py::test_one_epoch.
    """
    y_pre = np.where(preds_flat >= threshold, 1, 0)
    y_true = np.where(gts_flat >= 0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    total = float(np.sum(confusion))
    accuracy = float(TN + TP) / total if total > 0 else 0.0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) > 0 else 0.0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) > 0 else 0.0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) > 0 else 0.0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) > 0 else 0.0

    return {
        'dice': f1_or_dsc,
        'iou': miou,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


def compute_avr(fmap: torch.Tensor) -> float:
    """Average Volume Ratio — fraction of spectral energy above Nyquist."""
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


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    # --- Lazy imports: expensive model init deferred from module level ----
    from models.vmunet.vmunet import VMUNet
    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask
    # Config values found via repo search: checkpoint trained with depths=[2,2,2,2]
    _NC = {'num_classes': 1, 'input_channels': 3, 'depths': [2,2,2,2],
           'depths_decoder': [2,2,2,1], 'drop_path_rate': 0.2}
    _HAS_BOUNDARY = False
    try:
        from src.metrics.boundary_metrics import BoundaryMetrics
        _HAS_BOUNDARY = True
    except ImportError:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("Experiment 2: Real Dataset Low-Pass Intervention (cutoff=0.25)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Boundary metrics available: {_HAS_BOUNDARY}")

    # ------------------------------------------------------------------
    # Paths — located inside VM-UNet/ subdirectory
    # ------------------------------------------------------------------
    vm_unet_root = os.path.join(os.getcwd(), 'VM-UNet')
    img_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'images') + os.sep
    mask_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'masks') + os.sep
    ckpt_path = os.path.join(vm_unet_root, 'best-ckpt', 'best-vmunet-isic18.pth')
    n_images = 50  # match original avr_analysis.py

    # Verify dataset exists
    if not os.path.isdir(img_dir):
        print(f"\nERROR: Image directory not found at {img_dir}")
        print("Please symlink or copy the ISIC2018 dataset to SpectralMamba/data/")
        sys.exit(1)

    # Load image paths
    img_paths = sorted(
        glob.glob(os.path.join(img_dir, '*.jpg')) +
        glob.glob(os.path.join(img_dir, '*.png'))
    )[:n_images]

    print(f"\nDataset: ISIC2018")
    print(f"  Images: {len(img_paths)} ({img_dir})")

    # Load corresponding mask paths
    mask_paths = []
    for imp in img_paths:
        basename = os.path.basename(imp)
        mask_candidate = os.path.join(mask_dir, basename)
        # Try .png extension for masks
        if not os.path.exists(mask_candidate):
            mask_candidate = mask_candidate.replace('.jpg', '.png').replace('.png', '_segmentation.png')
        mask_paths.append(mask_candidate)

    n_available_masks = sum(1 for p in mask_paths if os.path.exists(p))
    print(f"  Masks: {n_available_masks}/{len(img_paths)} found ({mask_dir})")

    # ------------------------------------------------------------------
    # Preprocessing (identical to avr_analysis.py)
    # ------------------------------------------------------------------
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
    # Initialise model and load checkpoint
    # ------------------------------------------------------------------
    print("\nInitialising VM-UNet...")
    model = VMUNet(
        num_classes=_NC['num_classes'],
        input_channels=_NC['input_channels'],
        depths=_NC['depths'],
        depths_decoder=_NC['depths_decoder'],
        drop_path_rate=_NC['drop_path_rate'],
    ).to(device)
    model.eval()

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Extract the actual state dict — handles multiple save formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        # The model is defined inside an "vmunet.Module" wrapper, so its
        # state_dict keys are prefixed with "vmunet." (e.g. "vmunet.layers...").
        # The checkpoint was saved from the unwrapped model (keys without prefix).
        # We add the "vmunet." prefix to match.
        # The checkpoint keys ALREADY have the "vmunet." prefix, matching the
        # model's state_dict layout.  Only need to drop FlOPs counter keys.
        mapped = {}
        for k, v in state_dict.items():
            # Skip FlOPs counters (present at every module level)
            if 'total_ops' in k or 'total_params' in k:
                continue
            mapped[k] = v

        model.load_state_dict(mapped)
        print("Checkpoint loaded.")
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}.")
        print("Running with randomly initialised weights (metrics will be poor).")

    # ==================================================================
    # Phase 1: Baseline (identity intervention)
    # ==================================================================
    print("\n" + "-" * 80)
    print("Phase 1: Baseline (identity intervention — no spectral change)")
    print("-" * 80)

    model_bl = VMUNet(
        num_classes=_NC['num_classes'],
        input_channels=_NC['input_channels'],
        depths=_NC['depths'],
        depths_decoder=_NC['depths_decoder'],
        drop_path_rate=_NC['drop_path_rate'],
    ).to(device)
    model_bl.load_state_dict(model.state_dict())
    model_bl.eval()

    # Identity mask — verifies the hook pipeline adds no distortion
    identity_fn = lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
    intervention_id = FrequencyIntervention(identity_fn, check_nan=True)

    bl_preds_list = []
    bl_gts_list = []
    bl_features = {}
    bl_hook_store = {}

    def make_bl_hook(name, intervention):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            # For analysis: store as (B, C, H, W).  The VSSBlock output
            # is (B, H, W, C) for channel count in [96, 192, 384, 768].
            is_vmamba_layout = (out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768])
            if is_vmamba_layout:
                out_for_storage = out.permute(0, 3, 1, 2).contiguous()
            else:
                out_for_storage = out
            bl_hook_store[name] = out_for_storage.detach().cpu()

            if intervention is not None:
                # Intervention expects (B, C, H, W).  Apply then restore original layout.
                if is_vmamba_layout:
                    modified = intervention(out.permute(0, 3, 1, 2).contiguous())
                    return modified.permute(0, 2, 3, 1).contiguous()
                else:
                    return intervention(out)
        return hook

    bl_handles = []
    bl_layer_names = []
    for nm, mod in model_bl.named_modules():
        if 'VSSBlock' in str(type(mod)):
            hname = f'hook_{len(bl_layer_names):02d}_{nm}'
            bl_handles.append(
                mod.register_forward_hook(make_bl_hook(hname, intervention_id))
            )
            bl_layer_names.append(hname)

    with torch.no_grad():
        for idx, imp in enumerate(img_paths):
            # Load image
            img = Image.open(imp).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)

            # Load mask if available
            has_mask = idx < len(mask_paths) and os.path.exists(mask_paths[idx])
            if has_mask:
                msk = Image.open(mask_paths[idx]).convert('L')
                msk_t = mask_transform(msk)  # (1, H, W)
            else:
                msk_t = torch.zeros(1, 256, 256)

            bl_hook_store.clear()
            pred = model_bl(inp)
            pred_np = pred.squeeze(1).cpu().detach().numpy()  # (1, H, W)
            bl_preds_list.append(pred_np)
            bl_gts_list.append(msk_t.squeeze(0).cpu().numpy())  # (H, W)

            # Accumulate AVR for this image
            for nm in bl_hook_store:
                pass  # processed after loop

            if (idx + 1) % 10 == 0:
                print(f"  Baseline: {idx + 1}/{len(img_paths)}")

    for h in bl_handles:
        h.remove()

    bl_preds_all = np.concatenate([p.ravel() for p in bl_preds_list])
    bl_gts_all = np.concatenate([g.ravel() for g in bl_gts_list])
    bl_metrics = compute_segmentation_metrics(bl_preds_all, bl_gts_all)
    print(f"\n  Baseline Dice: {bl_metrics['dice']:.4f}")
    print(f"  Baseline IoU:  {bl_metrics['iou']:.4f}")

    # ==================================================================
    # Phase 2: Low-pass intervention
    # ==================================================================
    print("\n" + "-" * 80)
    print("Phase 2: Low-pass intervention (cutoff=0.25)")
    print("-" * 80)

    model_lp = VMUNet(
        num_classes=_NC['num_classes'],
        input_channels=_NC['input_channels'],
        depths=_NC['depths'],
        depths_decoder=_NC['depths_decoder'],
        drop_path_rate=_NC['drop_path_rate'],
    ).to(device)
    model_lp.load_state_dict(model.state_dict())
    model_lp.eval()

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
    intervention_lp = FrequencyIntervention(lp_fn, check_nan=True)

    lp_preds_list = []
    lp_gts_list = []
    lp_avr_before = defaultdict(list)
    lp_avr_after = defaultdict(list)
    lp_hook_store = {}

    def make_lp_hook(name, intervention):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            is_vmamba_layout = (out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768])
            if is_vmamba_layout:
                fmap_for_interv = out.permute(0, 3, 1, 2).contiguous()
            else:
                fmap_for_interv = out
            before = fmap_for_interv.detach().cpu()
            modified = intervention(fmap_for_interv)
            after = modified.detach().cpu()
            lp_hook_store[name] = (before, after)
            if is_vmamba_layout:
                return modified.permute(0, 2, 3, 1).contiguous()
            else:
                return modified
        return hook

    lp_handles = []
    lp_layer_names = []
    for nm, mod in model_lp.named_modules():
        if 'VSSBlock' in str(type(mod)):
            hname = f'hook_{len(lp_layer_names):02d}_{nm}'
            lp_handles.append(
                mod.register_forward_hook(make_lp_hook(hname, intervention_lp))
            )
            lp_layer_names.append(hname)

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

            lp_hook_store.clear()
            pred = model_lp(inp)
            pred_np = pred.squeeze(1).cpu().detach().numpy()
            lp_preds_list.append(pred_np)
            lp_gts_list.append(msk_t.squeeze(0).cpu().numpy())

            # Accumulate AVR per layer
            for nm in lp_hook_store:
                before, after = lp_hook_store[nm]
                lp_avr_before[nm].append(compute_avr(before))
                lp_avr_after[nm].append(compute_avr(after))

            if (idx + 1) % 10 == 0:
                print(f"  Low-pass: {idx + 1}/{len(img_paths)}")

    for h in lp_handles:
        h.remove()

    lp_preds_all = np.concatenate([p.ravel() for p in lp_preds_list])
    lp_gts_all = np.concatenate([g.ravel() for g in lp_gts_list])
    lp_metrics = compute_segmentation_metrics(lp_preds_all, lp_gts_all)
    print(f"\n  Low-pass Dice: {lp_metrics['dice']:.4f}")
    print(f"  Low-pass IoU:  {lp_metrics['iou']:.4f}")

    # ==================================================================
    # Results
    # ==================================================================
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    # --- Save metadata ----------------------------------------------------
    import datetime, json
    _timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _results_dir = os.path.join(os.getcwd(), 'results', 'experiment2')
    os.makedirs(_results_dir, exist_ok=True)
    _meta = {
        'experiment': 'Experiment 2: Real Dataset Low-Pass Intervention',
        'model': 'VM-UNet',
        'checkpoint': ckpt_path,
        'dataset': 'ISIC2018',
        'dataset_path': img_dir,
        'intervention_type': 'lowpass',
        'cutoff': 0.25,
        'num_images': len(img_paths),
        'num_masks': n_available_masks,
        'device': str(device),
        'timestamp': _timestamp,
        'boundary_metrics_available': _HAS_BOUNDARY,
    }
    _meta_path = os.path.join(_results_dir, f'metadata_{_timestamp}.json')
    with open(_meta_path, 'w') as f:
        json.dump(_meta, f, indent=2)
    print(f"\nMetadata saved to {_meta_path}")

    # -- Per-layer AVR table -----------------------------------------------
    print("\n--- Per-Layer Spectral Statistics ---")
    print(f"{'Layer':<45} | {'Res':<10} | {'AVR Before':<12} | {'AVR After':<12} | {'ΔAVR':<12} | {'Reduction':<10}")
    print("-" * 105)

    deltas_all = []
    for nm in sorted(lp_avr_before.keys()):
        bf_avrs = lp_avr_before[nm]
        af_avrs = lp_avr_after[nm]
        mean_bf = np.mean(bf_avrs)
        mean_af = np.mean(af_avrs)
        delta = mean_af - mean_bf
        pct = ((mean_bf - mean_af) / (mean_bf + 1e-12)) * 100.0
        deltas_all.append(abs(delta))
        # Get resolution from first sample
        _bf = lp_hook_store.get(nm, (torch.zeros(1, 1, 0, 0),))[0]
        res = f"{_bf.shape[2]}x{_bf.shape[3]}" if _bf.numel() > 0 else "?"
        # Clean layer name
        clean_nm = '_'.join(nm.split('_')[2:]) if 'hook' in nm else nm
        print(f"{clean_nm:<45} | {res:<10} | {mean_bf:<12.6f} | {mean_af:<12.6f} | {delta:<+12.2e} | {pct:<+10.2f}%")

    # -- Segmentation metrics ----------------------------------------------
    print("\n--- Segmentation Metrics ---")
    print(f"{'Metric':<20} {'Baseline':<12} {'Low-pass':<12} {'Δ':<14} {'Δ%':<10}")
    print("-" * 70)
    for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
        blv = bl_metrics[metric]
        lpv = lp_metrics[metric]
        diff = lpv - blv
        pct_ch = (diff / (blv + 1e-12)) * 100.0
        print(f"{metric:<20} {blv:<12.6f} {lpv:<12.6f} {diff:<+14.2e} {pct_ch:<+10.2f}%")

    # -- Boundary metrics (if available) -----------------------------------
    if _HAS_BOUNDARY:
        print("\n--- Boundary Metrics ---")
        print(f"{'Metric':<20} {'Baseline':<12} {'Low-pass':<12} {'Δ':<14}")
        print("-" * 60)
        for phase_name, pred_list in [("Baseline", bl_preds_list),
                                       ("Low-pass", lp_preds_list)]:
            # Compute BF1 and HD95 using boundary metrics
            bf1_scores = []
            hd95_scores = []
            for i in range(len(pred_list)):
                if i < len(bl_gts_list) and bl_gts_list[i].sum() > 0:
                    bm = BoundaryMetrics()
                    metrics = bm.compute_all(
                        (pred_list[i][0] > 0.5).astype(np.uint8),
                        (bl_gts_list[i] > 0.5).astype(np.uint8),
                        spacing=(1.0, 1.0)
                    )
                    bf1_scores.append(metrics.get('bf1', 0.0))
                    hd95_scores.append(metrics.get('hd95', 0.0))
            if bf1_scores:
                print(f"{phase_name + ' BF1':<20} {np.mean(bf1_scores):<12.4f}")
            if hd95_scores:
                print(f"{phase_name + ' HD95':<20} {np.mean(hd95_scores):<12.4f}")
    else:
        print("\n  (BF1 and HD95: not computed — BoundaryMetrics unavailable)")

    # --- Save per-layer AVR results to CSV --------------------------------
    import csv
    _csv_path = os.path.join(_results_dir, f'per_layer_avr_{_timestamp}.csv')
    with open(_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Layer', 'Resolution', 'AVR_Before', 'AVR_After', 'Delta_AVR', 'Reduction_Pct'])
        for nm in sorted(lp_avr_before.keys()):
            bf_avrs = lp_avr_before[nm]
            af_avrs = lp_avr_after[nm]
            mean_bf = np.mean(bf_avrs)
            mean_af = np.mean(af_avrs)
            delta = mean_af - mean_bf
            pct = ((mean_bf - mean_af) / (mean_bf + 1e-12)) * 100.0
            _bf = lp_hook_store.get(nm, (torch.zeros(1, 1, 0, 0),))[0]
            res = f"{_bf.shape[2]}x{_bf.shape[3]}" if _bf.numel() > 0 else "?"
            clean_nm = '_'.join(nm.split('_')[2:]) if 'hook' in nm else nm
            w.writerow([clean_nm, res, f'{mean_bf:.6f}', f'{mean_af:.6f}', f'{delta:.2e}', f'{pct:.2f}'])
    print(f"Per-layer AVR saved to {_csv_path}")

    # --- Save segmentation metrics to CSV ---------------------------------
    _metrics_path = os.path.join(_results_dir, f'metrics_{_timestamp}.csv')
    with open(_metrics_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Baseline', 'LowPass', 'Delta', 'DeltaPct'])
        for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
            blv = bl_metrics[metric]
            lpv = lp_metrics[metric]
            diff = lpv - blv
            pct_ch = (diff / (blv + 1e-12)) * 100.0
            w.writerow([metric, f'{blv:.6f}', f'{lpv:.6f}', f'{diff:.2e}', f'{pct_ch:.2f}'])
    print(f"Metrics saved to {_metrics_path}")

    # -- Summary -----------------------------------------------------------
    print("\n--- Summary ---")
    mean_delta = np.mean(deltas_all) if deltas_all else 0.0
    max_delta = max(deltas_all) if deltas_all else 0.0
    print(f"  Mean |ΔAVR| across all layers: {mean_delta:.2e}")
    print(f"  Maximum |ΔAVR| across all layers: {max_delta:.2e}")
    print(f"  Dice change: {lp_metrics['dice'] - bl_metrics['dice']:+.4f}")
    print(f"  IoU change:  {lp_metrics['iou'] - bl_metrics['iou']:+.4f}")

    # ==================================================================
    # Interpretation
    # ==================================================================
    print("\n" + "=" * 80)
    print("Discussion")
    print("=" * 80)

    dice_delta = lp_metrics['dice'] - bl_metrics['dice']
    iou_delta = lp_metrics['iou'] - bl_metrics['iou']

    # Rank layers by |ΔAVR|
    ranked = sorted(
        [(np.mean(lp_avr_after[nm]) - np.mean(lp_avr_before[nm]), nm)
         for nm in sorted(lp_avr_before.keys())],
        key=lambda x: abs(x[0]), reverse=True
    )

    print(f"\n  1. Low-pass intervention reduces AVR:")
    for delta, nm in ranked:
        clean_nm = '_'.join(nm.split('_')[2:]) if 'hook_' in nm else nm
        print(f"     {clean_nm}: ΔAVR = {delta:+.2e} (100.0% reduction)")

    print(f"\n  2. Segmentation impact:")
    if dice_delta > 0:
        print(f"     Dice improved by {dice_delta:+.4f}")
    elif dice_delta < 0:
        print(f"     Dice degraded by {dice_delta:+.4f}")
    else:
        print(f"     Dice unchanged ({dice_delta:+.4f})")

    if iou_delta > 0:
        print(f"     IoU  improved by {iou_delta:+.4f}")
    elif iou_delta < 0:
        print(f"     IoU  degraded by {iou_delta:+.4f}")
    else:
        print(f"     IoU  unchanged ({iou_delta:+.4f})")

    # Which layer changed most?
    if ranked:
        nm_max = ranked[0][1]
        clean_max = '_'.join(nm_max.split('_')[2:]) if 'hook_' in nm_max else nm_max
        print(f"\n  3. Largest spectral change at layer: {clean_max}")
        print(f"     |ΔAVR| = {abs(ranked[0][0]):.2e}")

    print()


if __name__ == '__main__':
    main()