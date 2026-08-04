"""
Experiment 11: UNet-ResNet50 Layer-wise Causal Intervention on CVC-ClinicDB.

Replicates Experiment 8 (layer-wise low-pass, cutoff=0.25) on the
CVC-ClinicDB dataset using CVCDataset (352x352, [0,1] normalization) and the
pre-trained UNet-ResNet50 CVC checkpoint.

For each of the 9 mapped stages, applies lowpass_mask(cutoff=0.25) ONLY to the
target stage, keeping all others as identity. Reports causal Delta Dice.

Usage:
    cd SpectralMamba
    python ..\\interventions\\experiments\\experiment11_resnet50_cvc_layerwise.py \\
        --output_dir ..\\interventions\\results\\experiment11_resnet50_cvc_layerwise --seed 42
"""

import sys, os, glob, json, csv, datetime, argparse, random
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [_REPO, os.path.join(_REPO, 'SpectralMamba'),
          os.path.join(_REPO, 'tta_boundary_study')]:
    sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Helpers - IDENTICAL to Exps 8
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


ENCODER_STAGES = [
    ('encoder.block1', 'encoder.layer1[-1]'),
    ('encoder.block2', 'encoder.layer2[-1]'),
    ('encoder.block3', 'encoder.layer3[-1]'),
    ('encoder.block4', 'encoder.layer4[-1]'),
]
BRIDGE_STAGES = [('bridge', 'decoder.blocks[0]')]
DECODER_STAGES = [
    ('decoder.block1', 'decoder.blocks[1]'),
    ('decoder.block2', 'decoder.blocks[2]'),
    ('decoder.block3', 'decoder.blocks[3]'),
    ('decoder.block4', 'decoder.blocks[4]'),
]
ALL_STAGES = ENCODER_STAGES + BRIDGE_STAGES + DECODER_STAGES


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
    print('Experiment 11: UNet-ResNet50 CVC-ClinicDB Layer-wise Causal (cutoff=0.25)')
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

    # ------------------------------------------------------------------
    # Baseline (all identity)
    # ------------------------------------------------------------------
    print('\n--- Baseline (all identity) ---')
    id_fn = lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt)
    identity_interv = FrequencyIntervention(id_fn, check_nan=True)

    model_bl = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                        in_channels=3, classes=1).to(device)
    model_bl.load_state_dict(model_ref.state_dict(), strict=True)
    model_bl.eval()
    bl_handles = []
    for name, path in ALL_STAGES:
        bl_handles.append(_resolve_path(model_bl, path).register_forward_hook(
            _make_interv_hook(identity_interv, None)))

    bl_preds, bl_gts = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model_bl(imgs)
            prob = torch.sigmoid(logits)
            bl_preds.append(prob.squeeze(1).cpu().detach().numpy())
            bl_gts.append(masks.squeeze(1).cpu().numpy())
    for h in bl_handles:
        h.remove()

    bl_metrics = compute_segmentation_metrics(
        np.concatenate([p.ravel() for p in bl_preds]),
        np.concatenate([g.ravel() for g in bl_gts]))
    print(f"  Baseline Dice: {bl_metrics['dice']:.4f}, IoU: {bl_metrics['iou']:.4f}")

    # ------------------------------------------------------------------
    # Layer-wise interventions
    # ------------------------------------------------------------------
    print('\n--- Layer-wise Interventions (low-pass on one stage, identity elsewhere) ---')
    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
    lp_interv = FrequencyIntervention(lp_fn, check_nan=True)

    results = []
    for idx_stage, (stage_name, path) in enumerate(ALL_STAGES):
        print(f'\n  [{idx_stage + 1}/{len(ALL_STAGES)}] TARGET: {stage_name} -> {path}')

        model = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                         in_channels=3, classes=1).to(device)
        model.load_state_dict(model_ref.state_dict(), strict=True)
        model.eval()

        avr_store = {}
        handles = []
        for name, p in ALL_STAGES:
            is_target = (name == stage_name)
            handles.append(_resolve_path(model, p).register_forward_hook(
                _make_interv_hook(lp_interv if is_target else identity_interv,
                                  avr_store if is_target else None)))

        preds, gts = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                avr_store.clear() if avr_store is not None else None
                logits = model(imgs)
                prob = torch.sigmoid(logits)
                preds.append(prob.squeeze(1).cpu().detach().numpy())
                gts.append(masks.squeeze(1).cpu().numpy())
        for h in handles:
            h.remove()
        del model

        m = compute_segmentation_metrics(
            np.concatenate([p.ravel() for p in preds]),
            np.concatenate([g.ravel() for g in gts]))
        delta_dice = m['dice'] - bl_metrics['dice']
        delta_iou = m['iou'] - bl_metrics['iou']
        delta_avr = (np.mean(avr_store.get('before', [0.0])) - np.mean(avr_store.get('after', [0.0]))) if avr_store else 0.0

        results.append({
            'stage': stage_name, 'path': path,
            'dice': m['dice'], 'delta_dice': delta_dice,
            'iou': m['iou'], 'delta_iou': delta_iou,
            'delta_avr': delta_avr,
        })
        print(f'    Dice: {m["dice"]:.4f} (Delta={delta_dice:+.4f}), IoU: {m["iou"]:.4f}')

    # ------------------------------------------------------------------
    # Sort by |Delta Dice| descending and save
    # ------------------------------------------------------------------
    results.sort(key=lambda r: abs(r['delta_dice']), reverse=True)

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = args.output_dir if args.output_dir else os.path.join(
        os.getcwd(), 'results', 'experiment11_resnet50_cvc_layerwise')
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        'experiment': 'Experiment 11: UNet-ResNet50 CVC-ClinicDB Layer-wise Causal',
        'model': 'UNet-ResNet50 (smp)', 'dataset': 'CVC-ClinicDB',
        'cutoff': 0.25, 'img_size': 352, 'num_images': len(val_ds),
        'device': str(device), 'seed': args.seed, 'timestamp': ts,
    }
    with open(os.path.join(out_dir, f'metadata_{ts}.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    csv_path = os.path.join(out_dir, f'layerwise_results_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['stage', 'path', 'dice', 'delta_dice',
                                          'iou', 'delta_iou', 'delta_avr'])
        w.writeheader()
        w.writerows(results)
    print(f'\nResults saved to {csv_path}')

    print('\n' + '=' * 90)
    print('Layer-wise Results (sorted by |Delta Dice|)')
    print('=' * 90)
    print(f"{'Stage':<18} | {'Module Path':<22} | {'Dice':<8} | {'DeltaDice':<10} | {'IoU':<8} | {'DeltaIoU':<10}")
    print('-' * 90)
    for r in results:
        print(f"{r['stage']:<18} | {r['path']:<22} | {r['dice']:<8.4f} | "
              f"{r['delta_dice']:<+10.4f} | {r['iou']:<8.4f} | {r['delta_iou']:<+10.4f}")
    print('-' * 90)

    # --- Plot --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        stages = [r['stage'] for r in results]
        deltas = [r['delta_dice'] for r in results]
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in deltas]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(len(results)), deltas, color=colors)
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(stages, fontsize=9)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_xlabel('Delta Dice (low-pass 0.25 on single stage)')
        ax.set_title('UNet-ResNet50 (CVC-ClinicDB): Causal Sensitivity by Stage')
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f'layerwise_plot_{ts}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Figure saved to {fig_path}')
    except ImportError:
        print('Visualization skipped (matplotlib unavailable).')

    print()


def _make_interv_hook(intervention, avr_store):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if avr_store is not None:
            before = out.detach().cpu()
            modified = intervention(out)
            after = modified.detach().cpu()
            avr_store.setdefault('before', []).append(compute_avr(before))
            avr_store.setdefault('after', []).append(compute_avr(after))
            return modified
        return intervention(out)
    return hook


if __name__ == '__main__':
    main()