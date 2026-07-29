"""
Experiment 4: Low-Pass Cutoff Sweep.

Quantifies how segmentation performance changes as progressively more
high-frequency information is removed.

Protocols:
    1. Whole-network intervention (all VSSBlocks)
    2. Single-layer intervention on layers.0.blocks.0 (most sensitive from Exp 3)

Cutoffs evaluated: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

For every cutoff:
    - Dice, IoU, accuracy, sensitivity, specificity
    - AVR_before, AVR_after, ΔAVR

Usage:
    python tools/experiment4_cutoff_sweep.py
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


def get_mask_fn(cutoff):
    """Return a mask_fn for a given cutoff value."""
    from interventions.masks import lowpass_mask
    return lambda h, w, dev, dt: lowpass_mask(h, w, cutoff, device=dev, dtype=dt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from models.vmunet.vmunet import VMUNet
    from interventions.intervention import FrequencyIntervention

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("Experiment 4: Low-Pass Cutoff Sweep")
    print("=" * 80)
    print(f"Device: {device}")

    # -- Paths --------------------------------------------------------------
    vm_unet_root = os.path.join(os.getcwd(), 'VM-UNet')
    img_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'images') + os.sep
    mask_dir = os.path.join(vm_unet_root, 'data', 'isic18', 'train', 'masks') + os.sep
    ckpt_path = os.path.join(vm_unet_root, 'best-ckpt', 'best-vmunet-isic18.pth')
    n_images = 50
    cutoffs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    _NC = {'num_classes': 1, 'input_channels': 3, 'depths': [2, 2, 2, 2],
           'depths_decoder': [2, 2, 2, 1], 'drop_path_rate': 0.2}

    # -- Verify prerequisites -----------------------------------------------
    if not os.path.isdir(img_dir):
        print(f"ERROR: Image directory not found at {img_dir}")
        sys.exit(1)
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
    print(f"\nDataset: {len(img_paths)} images, {n_masks} masks")
    print(f"Cutoffs: {cutoffs}")

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
    target_layer = 'vmunet.layers.0.blocks.0'
    print(f"Found {len(vssblock_names)} VSSBlock layers.")
    print(f"Single-layer target: {target_layer}")

    # -- Results container --------------------------------------------------
    results = []  # list of dicts

    for cutoff in cutoffs:
        print(f"\n  --- Cutoff = {cutoff:.2f} ---")

        # --- Protocol A: Whole-network intervention ------------------------
        _hook_store_all = {}
        _store_avr_all = {}

        def make_hook_all(cutoff_val):
            mask_fn = get_mask_fn(cutoff_val)
            interv = FrequencyIntervention(mask_fn)
            def make_hook_func(name):
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    is_vmamba = (out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768])
                    if is_vmamba:
                        fmap = out.permute(0, 3, 1, 2).contiguous()
                    else:
                        fmap = out
                    _hook_store_all[name] = fmap.detach().cpu()
                    modified = interv(fmap)
                    if is_vmamba:
                        return modified.permute(0, 2, 3, 1).contiguous()
                    return modified
                return hook
            return make_hook_func

        model_all = VMUNet(**_NC).to(device)
        model_all.load_state_dict(model_ref.state_dict())
        model_all.eval()
        handles_all = []
        _hook_store_all.clear()
        hook_maker = make_hook_all(cutoff)
        for nm, mod in model_all.named_modules():
            if 'VSSBlock' in str(type(mod)):
                handles_all.append(mod.register_forward_hook(hook_maker(nm)))

        preds_all_list, gts_all_list = [], []
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
                _hook_store_all.clear()
                pred = model_all(inp)
                pred_np = pred.squeeze(1).cpu().detach().numpy()
                preds_all_list.append(pred_np)
                gts_all_list.append(msk_t.squeeze(0).cpu().numpy())
        for h in handles_all:
            h.remove()
        del model_all

        preds_all_flat = np.concatenate([p.ravel() for p in preds_all_list])
        gts_all_flat = np.concatenate([g.ravel() for g in gts_all_list])
        metrics_all = compute_segmentation_metrics(preds_all_flat, gts_all_flat)

        # --- Protocol B: Single-layer intervention (layers.0.blocks.0) -----
        _hook_store_single = {}

        def make_hook_single(cutoff_val, target):
            mask_fn = get_mask_fn(cutoff_val)
            lp_interv = FrequencyIntervention(mask_fn)
            id_interv = FrequencyIntervention(
                lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
            )
            def make_hook_func(name):
                is_target = (name == target)
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    is_vmamba = (out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768])
                    if is_vmamba:
                        fmap = out.permute(0, 3, 1, 2).contiguous()
                    else:
                        fmap = out
                    _hook_store_single[name] = fmap.detach().cpu()
                    if is_target:
                        modified = lp_interv(fmap)
                    else:
                        modified = id_interv(fmap)
                    if is_vmamba:
                        return modified.permute(0, 2, 3, 1).contiguous()
                    return modified
                return hook
            return make_hook_func

        model_single = VMUNet(**_NC).to(device)
        model_single.load_state_dict(model_ref.state_dict())
        model_single.eval()
        handles_single = []
        _hook_store_single.clear()
        hook_maker_s = make_hook_single(cutoff, target_layer)
        for nm, mod in model_single.named_modules():
            if 'VSSBlock' in str(type(mod)):
                handles_single.append(mod.register_forward_hook(hook_maker_s(nm)))

        preds_single_list, gts_single_list = [], []
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
                _hook_store_single.clear()
                pred = model_single(inp)
                pred_np = pred.squeeze(1).cpu().detach().numpy()
                preds_single_list.append(pred_np)
                gts_single_list.append(msk_t.squeeze(0).cpu().numpy())
        for h in handles_single:
            h.remove()
        del model_single

        preds_single_flat = np.concatenate([p.ravel() for p in preds_single_list])
        gts_single_flat = np.concatenate([g.ravel() for g in gts_single_list])
        metrics_single = compute_segmentation_metrics(preds_single_flat, gts_single_flat)

        # --- Compute AVR for this cutoff -----------------------------------
        # Whole-network: average AVR across all layers from a separate forward pass
        avr_before_all_vals, avr_after_all_vals = [], []
        avr_before_single_vals, avr_after_single_vals = [], []

        model_avr = VMUNet(**_NC).to(device)
        model_avr.load_state_dict(model_ref.state_dict())
        model_avr.eval()
        mask_fn_avr = get_mask_fn(cutoff)
        lp_avr = FrequencyIntervention(mask_fn_avr)
        id_avr = FrequencyIntervention(
            lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
        )
        _avr_store = {}

        def make_avr_hook(name, is_target):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                is_vmamba = (out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768])
                if is_vmamba:
                    fmap = out.permute(0, 3, 1, 2).contiguous()
                else:
                    fmap = out
                before = fmap.detach().cpu()
                if is_target:
                    modified = lp_avr(fmap)
                else:
                    modified = id_avr(fmap)
                after = modified.detach().cpu()
                _avr_store[name] = (before, after)
                if is_vmamba:
                    return modified.permute(0, 2, 3, 1).contiguous()
                return modified
            return hook

        handles_avr = []
        for nm, mod in model_avr.named_modules():
            if 'VSSBlock' in str(type(mod)):
                handles_avr.append(
                    mod.register_forward_hook(make_avr_hook(nm, nm == target_layer))
                )
        with torch.no_grad():
            for idx in range(min(10, len(img_paths))):
                imp = img_paths[idx]
                img = Image.open(imp).convert('RGB')
                inp = transform(img).unsqueeze(0).to(device)
                _avr_store.clear()
                model_avr(inp)
                for nm in _avr_store:
                    bf, af = _avr_store[nm]
                    avr_before_all_vals.append(compute_avr(bf))
                    avr_after_all_vals.append(compute_avr(af))
                    if nm == target_layer:
                        avr_before_single_vals.append(compute_avr(bf))
                        avr_after_single_vals.append(compute_avr(af))
        for h in handles_avr:
            h.remove()
        del model_avr

        mean_bf_all = np.mean(avr_before_all_vals) if avr_before_all_vals else 0.0
        mean_af_all = np.mean(avr_after_all_vals) if avr_after_all_vals else 0.0
        mean_bf_single = np.mean(avr_before_single_vals) if avr_before_single_vals else 0.0
        mean_af_single = np.mean(avr_after_single_vals) if avr_after_single_vals else 0.0

        row = {
            'cutoff': cutoff,
            # Whole-network
            'dice_all': metrics_all['dice'], 'iou_all': metrics_all['iou'],
            'acc_all': metrics_all['accuracy'], 'sens_all': metrics_all['sensitivity'],
            'spec_all': metrics_all['specificity'],
            'avr_before_all': mean_bf_all, 'avr_after_all': mean_af_all,
            'delta_avr_all': mean_af_all - mean_bf_all,
            # Single-layer
            'dice_single': metrics_single['dice'], 'iou_single': metrics_single['iou'],
            'acc_single': metrics_single['accuracy'], 'sens_single': metrics_single['sensitivity'],
            'spec_single': metrics_single['specificity'],
            'avr_before_single': mean_bf_single, 'avr_after_single': mean_af_single,
            'delta_avr_single': mean_af_single - mean_bf_single,
        }
        results.append(row)

        print(f"    Whole: Dice={metrics_all['dice']:.4f}, IoU={metrics_all['iou']:.4f}, "
              f"ΔAVR={row['delta_avr_all']:+.2e}")
        print(f"    Single: Dice={metrics_single['dice']:.4f}, IoU={metrics_single['iou']:.4f}, "
              f"ΔAVR={row['delta_avr_single']:+.2e}")

    # -- Save results -------------------------------------------------------
    _ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _out_dir = os.path.join(os.getcwd(), 'results', 'experiment4_cutoff_sweep')
    os.makedirs(_out_dir, exist_ok=True)

    meta = {
        'experiment': 'Experiment 4: Low-Pass Cutoff Sweep',
        'model': 'VM-UNet', 'dataset': 'ISIC2018',
        'cutoffs': cutoffs, 'num_images': n_images,
        'device': str(device), 'timestamp': _ts,
        'single_layer_target': target_layer,
    }
    with open(os.path.join(_out_dir, f'metadata_{_ts}.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    csv_path = os.path.join(_out_dir, f'cutoff_sweep_{_ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        if results:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # -- Print table --------------------------------------------------------
    print("\n" + "=" * 120)
    print("Cutoff Sweep Results")
    print("=" * 120)
    print(f"{'Cutoff':<8} | {'Protocol':<10} | {'Dice':<8} | {'ΔDice':<10} | {'IoU':<8} | {'ΔIoU':<10} | {'AVR_b':<8} | {'AVR_a':<8} | {'ΔAVR':<10}")
    print("-" * 120)
    bl_dice_all = results[0]['dice_all']  # closest to identity
    bl_dice_single = results[0]['dice_single']
    bl_iou_all = results[0]['iou_all']
    bl_iou_single = results[0]['iou_single']

    for r in results:
        for protocol, prefix in [('Whole', 'all'), ('Single', 'single')]:
            dice_key = f'dice_{prefix}'
            iou_key = f'iou_{prefix}'
            bl_dice = bl_dice_all if prefix == 'all' else bl_dice_single
            bl_iou = bl_iou_all if prefix == 'all' else bl_iou_single
            print(f"{r['cutoff']:<8.2f} | {protocol:<10} | {r[dice_key]:<8.4f} | {r[dice_key]-bl_dice:<+10.4f} | "
                  f"{r[iou_key]:<8.4f} | {r[iou_key]-bl_iou:<+10.4f} | "
                  f"{r[f'avr_before_{prefix}']:<8.4f} | {r[f'avr_after_{prefix}']:<8.4f} | "
                  f"{r[f'delta_avr_{prefix}']:<+10.2e}")

    # -- Visualization ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        xs = [r['cutoff'] for r in results]

        # 1. Dice vs cutoff
        dice_all = [r['dice_all'] for r in results]
        dice_single = [r['dice_single'] for r in results]
        axes[0, 0].plot(xs, dice_all, 'o-', color='#e74c3c', label='Whole-network', linewidth=2)
        axes[0, 0].plot(xs, dice_single, 's--', color='#3498db', label='Single (layers.0.blocks.0)', linewidth=2)
        axes[0, 0].axhline(y=dice_all[0], color='gray', linestyle=':', alpha=0.5, label=f'Baseline ({dice_all[0]:.4f})')
        axes[0, 0].set_xlabel('Cutoff')
        axes[0, 0].set_ylabel('Dice')
        axes[0, 0].set_title('Dice vs Cutoff')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_xaxis()  # high cutoff = less filtering on left

        # 2. IoU vs cutoff
        iou_all = [r['iou_all'] for r in results]
        iou_single = [r['iou_single'] for r in results]
        axes[0, 1].plot(xs, iou_all, 'o-', color='#e74c3c', label='Whole-network', linewidth=2)
        axes[0, 1].plot(xs, iou_single, 's--', color='#3498db', label='Single (layers.0.blocks.0)', linewidth=2)
        axes[0, 1].axhline(y=iou_all[0], color='gray', linestyle=':', alpha=0.5, label=f'Baseline ({iou_all[0]:.4f})')
        axes[0, 1].set_xlabel('Cutoff')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('IoU vs Cutoff')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_xaxis()

        # 3. ΔAVR vs cutoff
        davr_all = [r['delta_avr_all'] for r in results]
        davr_single = [r['delta_avr_single'] for r in results]
        axes[1, 0].plot(xs, davr_all, 'o-', color='#e74c3c', label='Whole-network', linewidth=2)
        axes[1, 0].plot(xs, davr_single, 's--', color='#3498db', label='Single (layers.0.blocks.0)', linewidth=2)
        axes[1, 0].axhline(0, color='gray', linestyle=':', alpha=0.5)
        axes[1, 0].set_xlabel('Cutoff')
        axes[1, 0].set_ylabel('ΔAVR')
        axes[1, 0].set_title('ΔAVR vs Cutoff')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].invert_xaxis()

        # 4. Dice vs ΔAVR
        abs_davr_all = [abs(r['delta_avr_all']) for r in results]
        abs_davr_single = [abs(r['delta_avr_single']) for r in results]
        axes[1, 1].plot(abs_davr_all, dice_all, 'o-', color='#e74c3c', label='Whole-network', linewidth=2)
        axes[1, 1].plot(abs_davr_single, dice_single, 's--', color='#3498db', label='Single (layers.0.blocks.0)', linewidth=2)
        axes[1, 1].set_xlabel('|ΔAVR|')
        axes[1, 1].set_ylabel('Dice')
        axes[1, 1].set_title('Dice vs |ΔAVR|')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        # Annotate cutoff values
        for i, cutoff in enumerate(xs):
            axes[1, 1].annotate(f'{cutoff:.2f}', (abs_davr_all[i], dice_all[i]),
                                fontsize=7, ha='center', va='bottom')

        plt.tight_layout()
        fig_path = os.path.join(_out_dir, f'sweep_plots_{_ts}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure saved to {fig_path}")

    except ImportError as e:
        print(f"Visualization skipped: {e}")

    # -- Discussion ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Discussion")
    print("=" * 80)

    # Compute Dice changes from first cutoff (least aggressive)
    baseline_all = results[0]['dice_all']
    baseline_single = results[0]['dice_single']

    # Find where Dice drops below 90% of baseline
    threshold_all = baseline_all * 0.90
    threshold_single = baseline_single * 0.90
    critical_cutoff_all = None
    critical_cutoff_single = None
    for r in results:
        if r['dice_all'] < threshold_all and critical_cutoff_all is None:
            critical_cutoff_all = r['cutoff']
        if r['dice_single'] < threshold_single and critical_cutoff_single is None:
            critical_cutoff_single = r['cutoff']

    print(f"\n  1. Whole-network intervention:")
    print(f"     Dice ranges from {results[0]['dice_all']:.4f} (cutoff=0.10) to {results[-1]['dice_all']:.4f} (cutoff=0.80)")
    dice_drop_all = results[-1]['dice_all'] - results[0]['dice_all']
    print(f"     Total ΔDice across sweep: {dice_drop_all:+.4f}")
    if critical_cutoff_all is not None:
        print(f"     Critical cutoff (Dice drops below 90% of max): {critical_cutoff_all:.2f}")
    else:
        print(f"     No critical cutoff — Dice never drops below 90% of baseline.")

    print(f"\n  2. Single-layer intervention (layers.0.blocks.0):")
    print(f"     Dice ranges from {results[0]['dice_single']:.4f} (cutoff=0.10) to {results[-1]['dice_single']:.4f} (cutoff=0.80)")
    dice_drop_single = results[-1]['dice_single'] - results[0]['dice_single']
    print(f"     Total ΔDice across sweep: {dice_drop_single:+.4f}")
    if critical_cutoff_single is not None:
        print(f"     Critical cutoff (Dice drops below 90% of max): {critical_cutoff_single:.2f}")
    else:
        print(f"     No critical cutoff — Dice never drops below 90% of baseline.")

    print(f"\n  3. Threshold / nonlinear transition analysis:")
    # Detect nonlinear transition by checking consecutive differences
    dice_diffs_all = [abs(results[i+1]['dice_all'] - results[i]['dice_all']) for i in range(len(results)-1)]
    max_gap_idx = np.argmax(dice_diffs_all) if dice_diffs_all else 0
    max_gap = max(dice_diffs_all) if dice_diffs_all else 0.0
    print(f"     Largest single-step Dice drop (whole-network): {max_gap:.4f} "
          f"between cutoff {results[max_gap_idx]['cutoff']:.2f} and {results[max_gap_idx+1]['cutoff']:.2f}")
    if max_gap > 0.02:
        print(f"     Nonlinear transition detected: performance rapidly deteriorates "
              f"at cutoff ~{results[max_gap_idx]['cutoff']:.2f}–{results[max_gap_idx+1]['cutoff']:.2f}.")
    else:
        print(f"     No sharp nonlinear transition — performance degrades smoothly.")

    print(f"\n  4. Comparison between protocols:")
    ratio = dice_drop_single / dice_drop_all if dice_drop_all != 0 else 0
    print(f"     Single-layer ΔDice / Whole-network ΔDice = {ratio:.2f}")
    if ratio > 0.5:
        print(f"     The first encoder block accounts for most of the whole-network degradation.")
    elif ratio > 0.2:
        print(f"     The first encoder block accounts for a substantial portion of degradation.")
    else:
        print(f"     Degradation is distributed across layers — no single layer is dominant.")

    print()


if __name__ == '__main__':
    main()