"""
Experiment 3: Layer-wise Low-Pass Causal Intervention.

Measures the causal contribution of each individual VSSBlock by intervening
on one block at a time while leaving all other blocks unchanged.

Protocol:
    - Dataset: ISIC2018 (same as avr_analysis.py)
    - Model: VM-UNet with pre-trained checkpoint
    - For each VSSBlock independently:
        - Low-pass (cutoff=0.25) on the target block
        - Identity (all-ones) on all other blocks
    - Metrics: Dice, IoU, accuracy, sensitivity, specificity
    - Per-layer: AVR_before, AVR_after, ΔAVR
    - Visualization: ΔDice bar chart, ΔIoU bar chart, |ΔAVR| vs |ΔDice| scatter

Usage:
    python tools/experiment3_layerwise.py
"""

import sys
import os
import glob
import json
import csv
import datetime
import math
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'SpectralMamba'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------------------------------------------------------------------------
# Evaluation helpers
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
    y = torch.arange(H).view(1, 1, H, 1)
    x = torch.arange(W).view(1, 1, 1, W)
    mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4)
    mask = mask.expand(B, C, H, W)
    high_freq_energy = (power * mask).sum()
    total_energy = power.sum()
    return (high_freq_energy / total_energy).item() if total_energy > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Lazy imports
    from models.vmunet.vmunet import VMUNet
    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("Experiment 3: Layer-wise Low-Pass Causal Intervention")
    print("=" * 80)
    print(f"Device: {device}")

    # -- Paths --------------------------------------------------------------
    vm_unet_root = os.path.join(os.getcwd(), 'VM-UNet')
    img_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'images') + os.sep
    mask_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'masks') + os.sep
    ckpt_path = os.path.join(vm_unet_root, 'best-ckpt', 'best-vmunet-isic18.pth')
    n_images = 50
    _NC = {'num_classes': 1, 'input_channels': 3, 'depths': [2, 2, 2, 2],
           'depths_decoder': [2, 2, 2, 1], 'drop_path_rate': 0.2}

    # -- Verify prerequisites -----------------------------------------------
    if not os.path.isdir(img_dir):
        print(f"ERROR: Image directory not found at {img_dir}"); sys.exit(1)
    if not os.path.exists(ckpt_path):
        print(f"WARNING: Checkpoint not found. Running with random weights.")

    # -- Dataset ------------------------------------------------------------
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                        glob.glob(os.path.join(img_dir, '*.png')))[:n_images]
    mask_paths = []
    for imp in img_paths:
        mc = os.path.join(mask_dir, os.path.basename(imp))
        if not os.path.exists(mc):
            mc = mc.replace('.jpg', '.png').replace('.png', '_segmentation.png')
        mask_paths.append(mc)
    n_masks = sum(1 for p in mask_paths if os.path.exists(p))
    if n_masks == 0:
        print(f"WARNING: No masks found at {mask_dir}. Using dummy masks.")
    print(f"\nDataset: {len(img_paths)} images, {n_masks} masks")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # -- Load model ---------------------------------------------------------
    print("Loading model...")
    model_ref = VMUNet(**_NC).to(device)
    model_ref.eval()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
        mapped = {k: v for k, v in sd.items() if 'total_ops' not in k and 'total_params' not in k}
        model_ref.load_state_dict(mapped)
        print("Checkpoint loaded.")

    # -- Identify VSSBlock layers -------------------------------------------
    vssblock_names = []
    for nm, mod in model_ref.named_modules():
        if 'VSSBlock' in str(type(mod)):
            vssblock_names.append(nm)
    print(f"Found {len(vssblock_names)} VSSBlock layers.")

    # -- Helper functions for hooks -----------------------------------------
    def make_hook(name, is_lowpass, all_names):
        """Create a hook that applies low-pass to `name` and identity to others."""
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            is_vmamba = (out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768])
            if is_vmamba:
                fmap = out.permute(0, 3, 1, 2).contiguous()
            else:
                fmap = out
            _hook_store[name] = fmap.detach().cpu()
            if is_lowpass:
                modified = _lowpass_intervention(fmap)
            else:
                modified = _identity_intervention(fmap)
            if is_vmamba:
                return modified.permute(0, 2, 3, 1).contiguous()
            return modified
        return hook

    # Pre-create interventions
    _identity_intervention = FrequencyIntervention(
        lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
    )
    _lowpass_intervention = FrequencyIntervention(
        lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
    )

    # -- Run baseline (all identity) ----------------------------------------
    print("\n--- Baseline (all identity) ---")
    _hook_store = {}

    def run_one_config(lowpass_layer_name):
        """Run inference with low-pass on one layer, identity on all others."""
        model = VMUNet(**_NC).to(device)
        model.load_state_dict(model_ref.state_dict())
        model.eval()

        handles = []
        _hook_store.clear()
        for nm, mod in model.named_modules():
            if 'VSSBlock' in str(type(mod)):
                is_lp = (nm == lowpass_layer_name)
                handles.append(
                    mod.register_forward_hook(make_hook(nm, is_lp, vssblock_names))
                )

        preds_list, gts_list = [], []
        lp_avr_before, lp_avr_after = defaultdict(list), defaultdict(list)

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

                _hook_store.clear()
                pred = model(inp)
                pred_np = pred.squeeze(1).cpu().detach().numpy()
                preds_list.append(pred_np)
                gts_list.append(msk_t.squeeze(0).cpu().numpy())

                # Store AVR for the lowpass layer only
                if lowpass_layer_name in _hook_store:
                    fmap = _hook_store[lowpass_layer_name]
                    # AVR_before = before intervention (same as after since identity
                    # restores exactly, but we store pre-intervention separately)
                    # For the lowpass layer, the stored tensor IS post-intervention.
                    # We need before and after — but we only stored the result.
                    # Re-run the intervention to get before:
                    pass  # handled separately now

                if (idx + 1) % 25 == 0:
                    label = lowpass_layer_name if lowpass_layer_name else 'All Identity'
                    print(f"  {label[:40]:40s}: {idx+1}/{len(img_paths)}")

        for h in handles:
            h.remove()

        preds_all = np.concatenate([p.ravel() for p in preds_list])
        gts_all = np.concatenate([g.ravel() for g in gts_list])
        metrics = compute_segmentation_metrics(preds_all, gts_all)
        return metrics

    # -- Run baseline -------------------------------------------------------
    bl_metrics = run_one_config(None)
    print(f"  Baseline Dice: {bl_metrics['dice']:.4f}, IoU: {bl_metrics['iou']:.4f}")

    # -- Run per-layer interventions ----------------------------------------
    print("\n--- Layer-wise Interventions ---")
    results = []

    for i, ln in enumerate(vssblock_names):
        print(f"\n  [{i+1}/{len(vssblock_names)}] Intervening on: {ln}")
        metrics = run_one_config(ln)

        # Compute AVR for this layer using a separate forward pass
        # (Recording both before/after in the same pass would need hook changes)
        model_tmp = VMUNet(**_NC).to(device)
        model_tmp.load_state_dict(model_ref.state_dict())
        model_tmp.eval()
        lp_interv = FrequencyIntervention(
            lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
        )
        id_interv = FrequencyIntervention(
            lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
        )
        _hook_store_avr = {}
        avr_before_vals, avr_after_vals = [], []
        handles_avr = []
        for nm, mod in model_tmp.named_modules():
            if 'VSSBlock' in str(type(mod)):
                is_lp = (nm == ln)
                def make_avr_hook(target_name, is_lp):
                    def hook(module, inp, out):
                        if isinstance(out, tuple):
                            out = out[0]
                        is_vmamba = (out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768])
                        if is_vmamba:
                            fmap = out.permute(0, 3, 1, 2).contiguous()
                        else:
                            fmap = out
                        before = fmap.detach().cpu()
                        if is_lp:
                            modified = lp_interv(fmap)
                        else:
                            modified = id_interv(fmap)
                        after = modified.detach().cpu()
                        _hook_store_avr[target_name] = (before, after)
                        if is_vmamba:
                            return modified.permute(0, 2, 3, 1).contiguous()
                        return modified
                    return hook
                handles_avr.append(
                    mod.register_forward_hook(make_avr_hook(nm, is_lp))
                )
        with torch.no_grad():
            for idx in range(min(10, len(img_paths))):
                imp = img_paths[idx]
                img = Image.open(imp).convert('RGB')
                inp = transform(img).unsqueeze(0).to(device)
                _hook_store_avr.clear()
                model_tmp(inp)
                if ln in _hook_store_avr:
                    bf, af = _hook_store_avr[ln]
                    avr_before_vals.append(compute_avr(bf))
                    avr_after_vals.append(compute_avr(af))
        for h in handles_avr:
            h.remove()
        del model_tmp

        mean_avr_before = np.mean(avr_before_vals) if avr_before_vals else 0.0
        mean_avr_after = np.mean(avr_after_vals) if avr_after_vals else 0.0
        delta_avr = mean_avr_after - mean_avr_before
        pct_reduction = ((mean_avr_before - mean_avr_after) / (mean_avr_before + 1e-12)) * 100.0

        # Stage classification
        if 'layers.' in ln and 'layers_up.' not in ln:
            stage = 'Encoder'
        elif 'layers_up.' in ln:
            stage = 'Decoder'
        else:
            stage = 'Other'

        # Resolution from layer index heuristic
        parts = ln.split('.')
        try:
            layer_idx = int(parts[2]) if 'layers_up' in ln else int(parts[1])
        except (IndexError, ValueError):
            layer_idx = 0
        res_map = {0: 64, 1: 32, 2: 16, 3: 8}
        resolution = res_map.get(layer_idx, '?')

        delta_dice = metrics['dice'] - bl_metrics['dice']
        delta_iou = metrics['iou'] - bl_metrics['iou']

        results.append({
            'layer': ln,
            'stage': stage,
            'resolution': f'{resolution}x{resolution}',
            'dice': metrics['dice'],
            'delta_dice': delta_dice,
            'iou': metrics['iou'],
            'delta_iou': delta_iou,
            'accuracy': metrics['accuracy'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'avr_before': mean_avr_before,
            'avr_after': mean_avr_after,
            'delta_avr': delta_avr,
            'reduction_pct': pct_reduction,
        })
        print(f"    Dice: {metrics['dice']:.4f} (Δ={delta_dice:+.4f}), IoU: {metrics['iou']:.4f}, |ΔAVR|={abs(delta_avr):.2e}")

    # -- Sort by |ΔDice| descending -----------------------------------------
    results.sort(key=lambda r: abs(r['delta_dice']), reverse=True)

    # -- Save results -------------------------------------------------------
    _ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _out_dir = os.path.join(os.getcwd(), 'results', 'experiment3_layerwise')
    os.makedirs(_out_dir, exist_ok=True)

    # Metadata
    meta = {
        'experiment': 'Experiment 3: Layer-wise Low-Pass Intervention',
        'model': 'VM-UNet', 'dataset': 'ISIC2018', 'cutoff': 0.25,
        'num_images': n_images, 'device': str(device), 'timestamp': _ts,
    }
    with open(os.path.join(_out_dir, f'metadata_{_ts}.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # CSV
    csv_path = os.path.join(_out_dir, f'layerwise_results_{_ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'layer', 'stage', 'resolution', 'dice', 'delta_dice', 'iou',
            'delta_iou', 'accuracy', 'sensitivity', 'specificity',
            'avr_before', 'avr_after', 'delta_avr', 'reduction_pct'
        ])
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # -- Print table --------------------------------------------------------
    print("\n" + "=" * 120)
    print("Layer-wise Results (sorted by |ΔDice|)")
    print("=" * 120)
    print(f"{'Layer':<50} | {'Stage':<10} | {'Res':<8} | {'Dice':<8} | {'ΔDice':<10} | {'IoU':<8} | {'ΔIoU':<10} | {'AVR_b':<8} | {'ΔAVR':<10}")
    print("-" * 120)
    for r in results:
        print(f"{r['layer'][:48]:<50} | {r['stage']:<10} | {r['resolution']:<8} | {r['dice']:<8.4f} | {r['delta_dice']:<+10.4f} | {r['iou']:<8.4f} | {r['delta_iou']:<+10.4f} | {r['avr_before']:<8.4f} | {r['delta_avr']:<+10.2e}")
    print("-" * 120)

    # -- Visualization ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Bar chart: ΔDice
        layers_short = [r['layer'].split('.')[-1][:20] for r in results]
        delta_dice_vals = [r['delta_dice'] for r in results]
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in delta_dice_vals]
        axes[0, 0].barh(range(len(results)), delta_dice_vals, color=colors)
        axes[0, 0].set_yticks(range(len(results)))
        axes[0, 0].set_yticklabels(layers_short, fontsize=8)
        axes[0, 0].axvline(0, color='gray', linestyle='--')
        axes[0, 0].set_xlabel('ΔDice')
        axes[0, 0].set_title('ΔDice by Layer')

        # Bar chart: ΔIoU
        delta_iou_vals = [r['delta_iou'] for r in results]
        colors_iou = ['#e74c3c' if v < 0 else '#2ecc71' for v in delta_iou_vals]
        axes[0, 1].barh(range(len(results)), delta_iou_vals, color=colors_iou)
        axes[0, 1].set_yticks(range(len(results)))
        axes[0, 1].set_yticklabels(layers_short, fontsize=8)
        axes[0, 1].axvline(0, color='gray', linestyle='--')
        axes[0, 1].set_xlabel('ΔIoU')
        axes[0, 1].set_title('ΔIoU by Layer')

        # Scatter: |ΔAVR| vs |ΔDice|
        abs_delta_avr = [abs(r['delta_avr']) for r in results]
        abs_delta_dice = [abs(r['delta_dice']) for r in results]
        sc = axes[1, 0].scatter(abs_delta_avr, abs_delta_dice, c=range(len(results)),
                                 cmap='viridis', s=60)
        axes[1, 0].set_xlabel('|ΔAVR|')
        axes[1, 0].set_ylabel('|ΔDice|')
        axes[1, 0].set_title('|ΔAVR| vs |ΔDice|')
        cbar = plt.colorbar(sc, ax=axes[1, 0])
        cbar.set_label('Layer index')

        # Pearson correlation
        if len(abs_delta_avr) >= 2:
            corr, pval = pearsonr(abs_delta_avr, abs_delta_dice)
            axes[1, 0].text(0.05, 0.95, f'Pearson r = {corr:.4f}\np = {pval:.4e}',
                             transform=axes[1, 0].transAxes, fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Text summary
        axes[1, 1].axis('off')
        summary_lines = [
            f"Layers tested: {len(results)}",
            f"Baseline Dice: {bl_metrics['dice']:.4f}",
            f"Baseline IoU: {bl_metrics['iou']:.4f}",
            "",
            "Largest |ΔDice|:",
        ]
        for i, r in enumerate(results[:3]):
            summary_lines.append(f"  {i+1}. {r['layer'][:30]}: {r['delta_dice']:+.4f}")
        summary_lines.append("")
        summary_lines.append("Largest |ΔIoU|:")
        sorted_by_iou = sorted(results, key=lambda x: abs(x['delta_iou']), reverse=True)
        for i, r in enumerate(sorted_by_iou[:3]):
            summary_lines.append(f"  {i+1}. {r['layer'][:30]}: {r['delta_iou']:+.4f}")
        if len(abs_delta_avr) >= 2:
            summary_lines.append("")
            summary_lines.append(f"Pearson r(|ΔAVR|, |ΔDice|): {corr:.4f}")
            summary_lines.append(f"p-value: {pval:.4e}")
        axes[1, 1].text(0.05, 0.95, '\n'.join(summary_lines), transform=axes[1, 1].transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        fig_path = os.path.join(_out_dir, f'layerwise_plots_{_ts}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure saved to {fig_path}")

    except ImportError as e:
        print(f"Visualization skipped: {e}")

    # -- Discussion ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Discussion")
    print("=" * 80)

    # 1. Most sensitive layer
    most_sensitive = results[0]
    print(f"\n  1. Most sensitive to high-frequency removal:")
    print(f"     {most_sensitive['layer']}")
    print(f"     ΔDice = {most_sensitive['delta_dice']:+.4f}, ΔIoU = {most_sensitive['delta_iou']:+.4f}")

    # 2. Encoder vs decoder
    enc_deltas = [r['delta_dice'] for r in results if r['stage'] == 'Encoder']
    dec_deltas = [r['delta_dice'] for r in results if r['stage'] == 'Decoder']
    print(f"\n  2. Encoder vs Decoder sensitivity:")
    print(f"     Encoder mean ΔDice: {np.mean(enc_deltas):+.4f} (n={len(enc_deltas)})")
    print(f"     Decoder mean ΔDice: {np.mean(dec_deltas):+.4f} (n={len(dec_deltas)})")

    # 3. Does highest baseline AVR = largest performance drop?
    sorted_by_avr = sorted(results, key=lambda r: r['avr_before'], reverse=True)
    top_avr = sorted_by_avr[0]
    print(f"\n  3. Layer with highest baseline AVR:")
    print(f"     {top_avr['layer']} (AVR_before={top_avr['avr_before']:.4f})")
    print(f"     Its ΔDice = {top_avr['delta_dice']:+.4f}")
    print(f"     Rank by |ΔDice|: {next(i+1 for i, r in enumerate(results) if r['layer'] == top_avr['layer'])}/{len(results)}")

    # 4. Correlation
    if len(abs_delta_avr) >= 2:
        print(f"\n  4. Correlation between |ΔAVR| and |ΔDice|:")
        print(f"     Pearson r = {corr:.4f}, p = {pval:.4e}")
        if pval < 0.05:
            print(f"     Statistically significant (p < 0.05).")
        else:
            print(f"     Not statistically significant (p >= 0.05).")
    else:
        print(f"\n  4. Not enough data points for correlation analysis.")

    print()


if __name__ == '__main__':
    main()