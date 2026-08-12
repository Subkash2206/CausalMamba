"""
Experiment 17: VM-UNet Regularized (HF-Consistency) Whole-Network Causal
Intervention on CVC-ClinicDB.

Evaluates the Phase-4 regularized checkpoint (trained with the
High-Frequency Consistency Loss) under the exact same protocol as
Experiment 14, to quantify robustness against spectral perturbation:

Protocol A: Baseline (identity intervention).
Protocol B: Low-pass intervention (cutoff=0.25) at all 30 VSSBlocks.
Metrics: Dice, IoU, accuracy, sensitivity, specificity, per-block AVR.

Usage:
    cd SpectralMamba
    python ..\\interventions\\experiments\\experiment17_vmunet_cvc_reg_eval.py ^
        --output_dir ..\\interventions\\results\\experiment17_vmunet_cvc_reg_eval --seed 42
"""

import sys, os, argparse, random, json, csv, datetime
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [_REPO, os.path.join(_REPO, 'SpectralMamba'), os.path.join(_REPO, 'tta_boundary_study')]:
    sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Helpers (identical to Phase 0 Experiment 2)
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


# VM-UNet deep-layer stage names (matched to ResNet50's 9 semantic blocks)
# The 30 VSSBlocks are distributed as: layers[0]:2, layers[1]:2, layers[2]:9, layers[3]:2
#                                    layers_up[0]:2, layers_up[1]:9, layers_up[2]:2, layers_up[3]:2
# Total = 2+2+9+2 + 2+9+2+2 = 30
# We label them by their structural position within the encoder/decoder stages.
_VMAMBA_LAYER_NAMES = [
    # Encoder: layers[0] (depth 2) -> encoder.block1
    "encoder.block1.blk0", "encoder.block1.blk1",
    # Encoder: layers[1] (depth 2) -> encoder.block2
    "encoder.block2.blk0", "encoder.block2.blk1",
    # Encoder: layers[2] (depth 9) -> encoder.block3
    "encoder.block3.blk0", "encoder.block3.blk1", "encoder.block3.blk2",
    "encoder.block3.blk3", "encoder.block3.blk4", "encoder.block3.blk5",
    "encoder.block3.blk6", "encoder.block3.blk7", "encoder.block3.blk8",
    # Encoder: layers[3] (depth 2) -> encoder.block4
    "encoder.block4.blk0", "encoder.block4.blk1",
    # Decoder: layers_up[0] (depth 2, no upsample) -> bridge
    "bridge.blk0", "bridge.blk1",
    # Decoder: layers_up[1] (depth 9) -> decoder.block1
    "decoder.block1.blk0", "decoder.block1.blk1", "decoder.block1.blk2",
    "decoder.block1.blk3", "decoder.block1.blk4", "decoder.block1.blk5",
    "decoder.block1.blk6", "decoder.block1.blk7", "decoder.block1.blk8",
    # Decoder: layers_up[2] (depth 2) -> decoder.block2
    "decoder.block2.blk0", "decoder.block2.blk1",
    # Decoder: layers_up[3] (depth 2) -> decoder.block3
    "decoder.block3.blk0", "decoder.block3.blk1",
]
assert len(_VMAMBA_LAYER_NAMES) == 30, f"Expected 30 layer names, got {len(_VMAMBA_LAYER_NAMES)}"


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--ckpt_path', default=None)
    ap.add_argument('--img_size', type=int, default=256,
                    help='Evaluation resolution (regularized ckpt trained at 128)')
    args, _ = ap.parse_known_args()
    if args.seed is not None:
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
    print('Experiment 17: VM-UNet Regularized CVC Whole-Network Causal (cutoff=0.25)')
    print('=' * 80)
    print(f'Device: {device}')

    # ------------------------------------------------------------------
    # CVC dataset + checkpoint
    # ------------------------------------------------------------------
    ckpt_path = args.ckpt_path if args.ckpt_path else os.path.join(
        _REPO, 'interventions', 'results', 'best-vmunet-cvc-reg', 'best-vmunet-cvc-reg.pth')
    img_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
    mask_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')

    val_ds = CVCDataset(img_dir, mask_dir, split='val', img_size=args.img_size)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f'\nDataset: CVC-ClinicDB (validation split)')
    print(f'  Images: {len(val_ds)} | {args.img_size}x{args.img_size}, [0,1] normalization')

    # ------------------------------------------------------------------
    # Model: canonical 30-block VM-UNet
    # ------------------------------------------------------------------
    _NC = {'num_classes': 1, 'input_channels': 3,
           'depths': [2, 2, 9, 2], 'depths_decoder': [2, 9, 2, 2],
           'drop_path_rate': 0.2}
    model = VMUNet(**_NC).to(device)
    model.eval()

    if os.path.exists(ckpt_path):
        print(f'Loading checkpoint from {ckpt_path}...')
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and 'vmunet.layers.0.blocks.0.ln_1.weight' not in sd:
            sd = sd.get('model_state_dict') or sd.get('state_dict') or sd
        model.load_state_dict(sd, strict=True)
        print('Checkpoint loaded (strict=True).')
    else:
        print(f'WARNING: Checkpoint not found at {ckpt_path}. Random init.')

    # ==================================================================
    # Protocol A: Baseline (identity)
    # ==================================================================
    print('\n' + '-' * 80)
    print('Protocol A: Baseline (identity intervention)')
    print('-' * 80)

    model_bl = VMUNet(**_NC).to(device)
    model_bl.load_state_dict(model.state_dict(), strict=True)
    model_bl.eval()

    id_fn = lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
    interv_id = FrequencyIntervention(id_fn, check_nan=True)
    bl_store = {}

    def make_bl_hook(name, interv):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            # VSSBlock output: (B, H, W, C). Permute to (B, C, H, W) for AVR.
            is_vm = (out.dim() == 4 and out.shape[-1] in {96, 192, 384, 768})
            if is_vm:
                fmap = out.permute(0, 3, 1, 2).contiguous()
            else:
                fmap = out
            bl_store[name] = fmap.detach().cpu()
            return interv(fmap).permute(0, 2, 3, 1).contiguous() if is_vm else interv(fmap)
        return hook

    bl_handles = []
    blk_idx = 0
    for nm, mod in model_bl.named_modules():
        if 'VSSBlock' in type(mod).__name__:
            hname = _VMAMBA_LAYER_NAMES[blk_idx] if blk_idx < len(_VMAMBA_LAYER_NAMES) else f'vss_{blk_idx}'
            bl_handles.append(mod.register_forward_hook(make_bl_hook(hname, interv_id)))
            blk_idx += 1

    bl_preds, bl_gts = [], []
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            bl_store.clear()
            probs = model_bl(imgs)  # internal sigmoid
            bl_preds.append(probs.squeeze(1).cpu().detach().numpy())
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
    # Protocol B: Low-pass intervention (cutoff=0.25)
    # ==================================================================
    print('\n' + '-' * 80)
    print('Protocol B: Low-pass intervention (cutoff=0.25, all 30 VSSBlocks)')
    print('-' * 80)

    model_lp = VMUNet(**_NC).to(device)
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
            is_vm = (out.dim() == 4 and out.shape[-1] in {96, 192, 384, 768})
            if is_vm:
                fmap = out.permute(0, 3, 1, 2).contiguous()
            else:
                fmap = out
            before = fmap.detach().cpu()
            modified = interv(fmap)
            after = modified.detach().cpu()
            lp_store[name] = (before, after)
            return modified.permute(0, 2, 3, 1).contiguous() if is_vm else modified
        return hook

    lp_handles = []
    blk_idx = 0
    for nm, mod in model_lp.named_modules():
        if 'VSSBlock' in type(mod).__name__:
            hname = _VMAMBA_LAYER_NAMES[blk_idx] if blk_idx < len(_VMAMBA_LAYER_NAMES) else f'vss_{blk_idx}'
            lp_handles.append(mod.register_forward_hook(make_lp_hook(hname, interv_lp)))
            blk_idx += 1

    lp_preds, lp_gts = [], []
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            lp_store.clear()
            probs = model_lp(imgs)
            lp_preds.append(probs.squeeze(1).cpu().detach().numpy())
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
        os.path.dirname(__file__), '..', 'results', 'experiment17_vmunet_cvc_reg_eval')
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        'experiment': 'Experiment 17: VM-UNet Regularized CVC Whole-Network Causal',
        'model': 'VM-UNet (30-block)', 'dataset': 'CVC-ClinicDB',
        'checkpoint': ckpt_path, 'intervention_type': 'lowpass',
        'cutoff': 0.25, 'img_size': args.img_size, 'num_images': len(val_ds),
        'device': str(device), 'seed': args.seed, 'timestamp': ts,
    }
    with open(os.path.join(out_dir, f'metadata_{ts}.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nMetadata saved to {os.path.join(out_dir, f"metadata_{ts}.json")}')

    print('\n--- Per-Block Spectral Statistics (30 VSSBlocks) ---')
    print(f"{'Block':<25} | {'AVR Before':<12} | {'AVR After':<12} | {'Delta AVR':<14} | {'Reduction %':<10}")
    print('-' * 80)
    deltas_all = []
    for nm in _VMAMBA_LAYER_NAMES:
        if nm not in lp_avr_before:
            continue
        mean_bf = np.mean(lp_avr_before[nm])
        mean_af = np.mean(lp_avr_after[nm])
        delta = mean_af - mean_bf
        pct = ((mean_bf - mean_af) / (mean_bf + 1e-12)) * 100.0
        deltas_all.append(abs(delta))
        print(f"{nm:<25} | {mean_bf:<12.6f} | {mean_af:<12.6f} | {delta:<+14.2e} | {pct:<+10.2f}%")

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
        w.writerow(['Block', 'AVR_Before', 'AVR_After', 'Delta_AVR', 'Reduction_Pct'])
        for nm in _VMAMBA_LAYER_NAMES:
            if nm not in lp_avr_before:
                continue
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

    print(f'\nPer-block AVR + metrics saved to {out_dir}')

    print('\n--- Summary ---')
    print(f"  Baseline Dice: {bl_metrics['dice']:.4f}")
    print(f"  Low-pass Dice: {lp_metrics['dice']:.4f}")
    print(f"  Dice change:   {lp_metrics['dice'] - bl_metrics['dice']:+.4f}")
    print(f"  IoU change:    {lp_metrics['iou'] - bl_metrics['iou']:+.4f}")
    mean_delta = np.mean(deltas_all) if deltas_all else 0.0
    max_delta = max(deltas_all) if deltas_all else 0.0
    print(f'  Mean |Delta AVR| across 30 blocks: {mean_delta:.2e}')
    print(f'  Maximum |Delta AVR|:               {max_delta:.2e}')
    print()


if __name__ == '__main__':
    main()