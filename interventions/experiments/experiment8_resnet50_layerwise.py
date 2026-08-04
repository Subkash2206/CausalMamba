"""
Experiment 8: UNet-ResNet50 Layer-wise Causal Low-Pass Intervention (Phase 1).

Replicates Phase-0 Experiment 3 logic against the UNet-ResNet50 architecture
using the 9 mapped semantic boundaries from resnet50_hook_mapping.py:

    Encoder 1-4 (encoder.layer1..4[-1]), Bridge (decoder.blocks[0]),
    Decoder 1-4 (decoder.blocks[1..4]).

For each of the 9 stages, applies the lowpass_mask(cutoff=0.25) to ONLY the
target stage, keeping all other stages as identity. Reports the causal Delta
Dice per stage, sorted by |Delta Dice| descending.

Usage:
    cd SpectralMamba
    python ..\\interventions\\experiments\\experiment8_resnet50_layerwise.py \\
        --output_dir ..\\interventions\\results\\experiment8_resnet50_layerwise --seed 42
"""

import sys, os, glob, json, csv, datetime, argparse, random
from collections import defaultdict

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'SpectralMamba'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Evaluation helpers - IDENTICAL to Phase 0 (Exp 3)
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


# ---------------------------------------------------------------------------
# Mapped semantic boundaries (identical to resnet50_hook_mapping.py)
# ---------------------------------------------------------------------------

ENCODER_STAGES = [
    ('encoder.block1', 'encoder.layer1[-1]'),
    ('encoder.block2', 'encoder.layer2[-1]'),
    ('encoder.block3', 'encoder.layer3[-1]'),
    ('encoder.block4', 'encoder.layer4[-1]'),
]
BRIDGE_STAGES = [
    ('bridge', 'decoder.blocks[0]'),
]
DECODER_STAGES = [
    ('decoder.block1', 'decoder.blocks[1]'),
    ('decoder.block2', 'decoder.blocks[2]'),
    ('decoder.block3', 'decoder.blocks[3]'),
    ('decoder.block4', 'decoder.blocks[4]'),
]
ALL_STAGES = ENCODER_STAGES + BRIDGE_STAGES + DECODER_STAGES


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
    print('Experiment 8: UNet-ResNet50 Layer-wise Causal Intervention (cutoff=0.25)')
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
    # Load reference model + checkpoint
    # ------------------------------------------------------------------
    print("\nLoading model...")
    model_ref = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                         in_channels=3, classes=1).to(device)
    model_ref.eval()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            sd = ckpt
        model_ref.load_state_dict(sd, strict=True)
        print('Checkpoint loaded.')
    else:
        print(f'WARNING: Checkpoint not found at {ckpt_path}. Random init.')

    # ------------------------------------------------------------------
    # Baseline (all identity) - Protocol A
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
        bl_handles.append(
            _resolve_path(model_bl, path).register_forward_hook(
                _make_interv_hook(identity_interv, None)))

    bl_preds, bl_gts = [], []
    with torch.no_grad():
        for idx, imp in enumerate(img_paths):
            inp = transform(Image.open(imp).convert('RGB')).unsqueeze(0).to(device)
            has_mask = idx < len(mask_paths) and os.path.exists(mask_paths[idx])
            msk_t = mask_transform(Image.open(mask_paths[idx]).convert('L')) if has_mask else torch.zeros(1, 256, 256)
            logits = model_bl(inp)
            prob = torch.sigmoid(logits)
            bl_preds.append(prob.squeeze(1).cpu().detach().numpy())
            bl_gts.append(msk_t.squeeze(0).cpu().numpy())
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
    for stage_name, path in ALL_STAGES:
        print(f'\n  [{ALL_STAGES.index((stage_name, path)) + 1}/{len(ALL_STAGES)}] '
              f'TARGET: {stage_name} -> {path}')

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
            for idx, imp in enumerate(img_paths):
                inp = transform(Image.open(imp).convert('RGB')).unsqueeze(0).to(device)
                has_mask = idx < len(mask_paths) and os.path.exists(mask_paths[idx])
                msk_t = mask_transform(Image.open(mask_paths[idx]).convert('L')) if has_mask else torch.zeros(1, 256, 256)
                avr_store.clear() if avr_store is not None else None
                logits = model(inp)
                prob = torch.sigmoid(logits)
                preds.append(prob.squeeze(1).cpu().detach().numpy())
                gts.append(msk_t.squeeze(0).cpu().numpy())

        for h in handles:
            h.remove()
        del model

        m = compute_segmentation_metrics(
            np.concatenate([p.ravel() for p in preds]),
            np.concatenate([g.ravel() for g in gts]))
        delta_dice = m['dice'] - bl_metrics['dice']
        delta_iou = m['iou'] - bl_metrics['iou']
        delta_avr = np.mean(avr_store.get('before', [0.0])) - np.mean(avr_store.get('after', [0.0])) if avr_store else 0.0

        results.append({
            'stage': stage_name,
            'path': path,
            'dice': m['dice'],
            'delta_dice': delta_dice,
            'iou': m['iou'],
            'delta_iou': delta_iou,
        })
        print(f'    Dice: {m["dice"]:.4f} (Δ={delta_dice:+.4f}), IoU: {m["iou"]:.4f}')

    # ------------------------------------------------------------------
    # Sort by |Delta Dice| descending and save
    # ------------------------------------------------------------------
    results.sort(key=lambda r: abs(r['delta_dice']), reverse=True)

    _ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _out_dir = _args.output_dir if _args.output_dir else os.path.join(
        os.getcwd(), 'results', 'experiment8_resnet50_layerwise')
    os.makedirs(_out_dir, exist_ok=True)

    meta = {
        'experiment': 'Experiment 8: UNet-ResNet50 Layer-wise Causal Intervention',
        'model': 'UNet-ResNet50 (smp)', 'dataset': 'ISIC2018',
        'cutoff': 0.25, 'num_images': n_images,
        'device': str(device), 'seed': _args.seed, 'timestamp': _ts,
    }
    with open(os.path.join(_out_dir, f'metadata_{_ts}.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    csv_path = os.path.join(_out_dir, f'layerwise_results_{_ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['stage', 'path', 'dice', 'delta_dice',
                                          'iou', 'delta_iou'])
        w.writeheader()
        w.writerows(results)
    print(f'\nResults saved to {csv_path}')

    print('\n' + '=' * 90)
    print('Layer-wise Results (sorted by |ΔDice|)')
    print('=' * 90)
    print(f"{'Stage':<18} | {'Module Path':<22} | {'Dice':<8} | {'ΔDice':<10} | {'IoU':<8} | {'ΔIoU':<10}")
    print('-' * 90)
    for r in results:
        print(f"{r['stage']:<18} | {r['path']:<22} | {r['dice']:<8.4f} | "
              f"{r['delta_dice']:<+10.4f} | {r['iou']:<8.4f} | {r['delta_iou']:<+10.4f}")
    print('-' * 90)

    # --- Plot (bar chart of Delta Dice) -----------------------------------
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
        ax.set_title('UNet-ResNet50: Causal Sensitivity by Stage')
        plt.tight_layout()
        fig_path = os.path.join(_out_dir, f'layerwise_plot_{_ts}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Figure saved to {fig_path}')
    except ImportError:
        print('Visualization skipped (matplotlib unavailable).')

    print()


def _make_interv_hook(intervention, avr_store):
    """Return a forward hook applying `intervention` to the module output.

    If avr_store is not None, records AVR before/after for the target stage.
    """
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