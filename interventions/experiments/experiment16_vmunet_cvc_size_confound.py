"""
Experiment 16: VM-UNet CVC-ClinicDB Lesion-Size Confound Check (Phase 3.4).

Verifies whether the low-pass intervention (cutoff=0.25) performance drop is
disproportionately driven by polyp size or is a universal vulnerability across
all polyp sizes in CVC-ClinicDB.

Protocol (single run, both modes):
  1. Load the CVC-ClinicDB validation split and compute the ground-truth polyp
     pixel area (at 256x256) for every image.
  2. Bin the 123 validation images into 3 quantile terciles by area:
     Small / Medium / Large.
  3. Evaluate the frozen VM-UNet (best-vmunet-cvc.pth) with:
       --mode baseline  : identity forward pass
       --mode intervened: whole-network low-pass (cutoff=0.25) on all 30 VSSBlocks
  4. Report per-bin Baseline Dice, Intervened Dice and Delta Dice.

Usage:
    python interventions/experiments/experiment16_vmunet_cvc_size_confound.py
    (optionally: --mode baseline|intervened|both, --cutoff_radius, --output_dir)
"""

import sys, os, argparse, random, json, csv, datetime

import torch
import numpy as np
from torch.utils.data import DataLoader

# Robust console output on Windows (cp1252 default): allow Δ/≤ characters.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [_REPO, os.path.join(_REPO, 'SpectralMamba'),
          os.path.join(_REPO, 'tta_boundary_study')]:
    sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_dice(pred_bin, gt_bin):
    """Per-image Dice on binary maps. Both empty => perfect (1.0)."""
    inter = float((pred_bin & gt_bin).sum())
    denom = float(pred_bin.sum() + gt_bin.sum())
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def quantile_bins(areas):
    """Bin area values into Small/Medium/Large terciles (quantile thresholds)."""
    thresholds = np.quantile(np.asarray(areas, dtype=float), [1.0 / 3.0, 2.0 / 3.0])
    lo, hi = thresholds[0], thresholds[1]

    def label(a):
        if a <= lo:
            return 'Small'
        if a <= hi:
            return 'Medium'
        return 'Large'
    return lo, hi, [label(a) for a in areas]


# 30 VSSBlock names - same mapping as Experiments 14 / 15
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
    """VM-UNet VSSBlock hook (same as Exp 14/15): permute NHWC<->NCHW before intervening."""
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
    ap.add_argument('--mode', choices=['baseline', 'intervened', 'both'], default='both',
                    help="'baseline'/'intervened'/'both' (default: both)")
    ap.add_argument('--cutoff_radius', type=float, default=0.25)
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
    print('Experiment 16: VM-UNet CVC-ClinicDB Lesion-Size Confound Check')
    print('=' * 80)
    print(f'Device: {device} | mode={args.mode} | cutoff_radius={args.cutoff_radius}')

    # ------------------------------------------------------------------
    # Dataset + GT polyp area
    # ------------------------------------------------------------------
    ckpt_path = args.ckpt_path if args.ckpt_path else os.path.join(
        _REPO, 'tta_boundary_study', 'checkpoints', 'best-vmunet-cvc.pth')
    img_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
    mask_dir = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')

    val_ds = CVCDataset(img_dir, mask_dir, split='val', img_size=256)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f'\nDataset: CVC-ClinicDB (validation split), {len(val_ds)} images @ 256x256')

    areas = []      # GT polyp pixel area per image
    gts = []        # binarized GT masks
    for i, (_, masks) in enumerate(val_loader):
        gt = (masks.squeeze(1)[0].cpu().numpy() > 0.5).astype(np.uint8)
        gts.append(gt)
        areas.append(int(gt.sum()))
    areas = np.asarray(areas)

    lo, hi, bin_labels = quantile_bins(areas)
    print(f'\nGT polyp area (px @256): min={areas.min()}, median={np.median(areas):.0f}, '
          f'max={areas.max()}')
    print(f'Quantile thresholds: Small<= {lo:.0f}px | Medium<= {hi:.0f}px | Large > {hi:.0f}px')
    for b in ['Small', 'Medium', 'Large']:
        idx = [k for k, lb in enumerate(bin_labels) if lb == b]
        print(f'  {b:<7}: n={len(idx):>3}  area range [{areas[idx].min()}, {areas[idx].max()}]')

    # ------------------------------------------------------------------
    # Frozen VM-UNet
    # ------------------------------------------------------------------
    _NC = {'num_classes': 1, 'input_channels': 3,
           'depths': [2, 2, 9, 2], 'depths_decoder': [2, 9, 2, 2],
           'drop_path_rate': 0.2}
    model = VMUNet(**_NC).to(device)
    model.eval()

    if os.path.exists(ckpt_path):
        print(f'\nLoading frozen checkpoint: {ckpt_path}')
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and 'vmunet.layers.0.blocks.0.ln_1.weight' not in sd:
            sd = sd.get('model_state_dict') or sd.get('state_dict') or sd
        model.load_state_dict(sd, strict=True)
        print('Checkpoint loaded (strict=True).')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    baseline_dice = [None] * len(val_ds)
    intervened_dice = [None] * len(val_ds)

    if args.mode in ('baseline', 'both'):
        print(f'\n--- Inference: baseline ---')
        with torch.no_grad():
            for i, (imgs, _) in enumerate(val_loader):
                imgs = imgs.to(device)
                probs = model(imgs)                      # probabilities in [0,1]
                pred = (probs.squeeze(1)[0].cpu().numpy() > 0.5).astype(np.uint8)
                baseline_dice[i] = compute_dice(pred, gts[i])
                if (i + 1) % 30 == 0:
                    print(f'  {i + 1}/{len(val_ds)}')

    if args.mode in ('intervened', 'both'):
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

        print(f'\n--- Inference: intervened ---')
        with torch.no_grad():
            for i, (imgs, _) in enumerate(val_loader):
                imgs = imgs.to(device)
                probs = model(imgs)
                pred = (probs.squeeze(1)[0].cpu().numpy() > 0.5).astype(np.uint8)
                intervened_dice[i] = compute_dice(pred, gts[i])
                if (i + 1) % 30 == 0:
                    print(f'  {i + 1}/{len(val_ds)}')
        for h in lp_handles:
            h.remove()

    # ------------------------------------------------------------------
    # Per-bin summary
    # ------------------------------------------------------------------
    bins = ['Small', 'Medium', 'Large']
    summary = {}
    for b in bins:
        idx = [k for k, lb in enumerate(bin_labels) if lb == b]
        bd = [baseline_dice[k] for k in idx if baseline_dice[k] is not None]
        iv = [intervened_dice[k] for k in idx if intervened_dice[k] is not None]
        summary[b] = {
            'n': len(idx),
            'area_min': int(areas[idx].min()),
            'area_max': int(areas[idx].max()),
            'baseline_dice': float(np.mean(bd)) if bd else float('nan'),
            'intervened_dice': float(np.mean(iv)) if iv else float('nan'),
        }
        summary[b]['delta_dice'] = summary[b]['intervened_dice'] - summary[b]['baseline_dice']

    print('\n' + '=' * 78)
    print('Per-size-bin Dice (mean over images in bin):')
    print('=' * 78)
    hdr = f'{"Bin":<8}{"n":>5}{"Area range":>16}{"Baseline Dice":>15}{"Intervened Dice":>16}{"Delta Dice":>11}'
    print(hdr)
    print('-' * len(hdr))
    for b in bins:
        s = summary[b]
        print(f'{b:<8}{s["n"]:>5}{f"[{s["area_min"]}, {s["area_max"]}]":>16}'
              f'{s["baseline_dice"]:>15.4f}{s["intervened_dice"]:>16.4f}{s["delta_dice"]:>+11.4f}')

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = args.output_dir if args.output_dir else os.path.join(
        os.path.dirname(__file__), '..', 'results', 'experiment16_vmunet_cvc_size_confound')
    os.makedirs(out_dir, exist_ok=True)

    results = {
        'cutoff_radius': args.cutoff_radius,
        'quantile_thresholds_px': {'small_medium': float(lo), 'medium_large': float(hi)},
        'per_bin': {b: summary[b] for b in bins},
        'metadata': {
            'experiment': 'Experiment 16: VM-UNet CVC-ClinicDB Lesion-Size Confound Check',
            'model': 'VM-UNet (30-block)', 'dataset': 'CVC-ClinicDB',
            'checkpoint': ckpt_path, 'img_size': 256, 'num_images': len(val_ds),
            'device': str(device), 'seed': args.seed, 'timestamp': ts,
        }
    }
    with open(os.path.join(out_dir, f'metadata_{ts}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    per_image_path = os.path.join(out_dir, f'per_image_{ts}.csv')
    with open(per_image_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_index', 'area_px', 'size_bin', 'baseline_dice', 'intervened_dice'])
        for k in range(len(val_ds)):
            w.writerow([k, int(areas[k]), bin_labels[k],
                        f'{baseline_dice[k]:.4f}' if baseline_dice[k] is not None else '',
                        f'{intervened_dice[k]:.4f}' if intervened_dice[k] is not None else ''])

    # ------------------------------------------------------------------
    # Markdown table for the manuscript
    # ------------------------------------------------------------------
    print('\n' + '=' * 78)
    print('Manuscript markdown table:')
    print('=' * 78)
    print('| Polyp size | n | Baseline Dice | Intervened Dice | Δ Dice |')
    print('|---|---|---:|---:|---:|')
    for b in bins:
        s = summary[b]
        print(f'| {b} | {s["n"]} | {s["baseline_dice"]:.3f} | {s["intervened_dice"]:.3f} | {s["delta_dice"]:+.3f} |')
    print(f'\nQuantile thresholds (GT area @ 256x256): Small ≤ {lo:.0f} px | '
          f'Medium ≤ {hi:.0f} px | Large > {hi:.0f} px')
    print(f'\nResults saved to {out_dir}/')


if __name__ == '__main__':
    main()
