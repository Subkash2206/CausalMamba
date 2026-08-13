"""
eval_isic_heldout.py — Cross-architecture Fourier benchmark on the HELD-OUT test
split (Phase 2 of the spectral-vulnerability plan).

Same protocol as eval_isic_cross_arch.py (feature-space LP cutoff 0.25, input-space
blur, pooled + per-image Dice) but evaluated on the untouched 260-image test split
carved by carve_splits.py (never used for model selection).

Models:
  - VM-UNet        (SSM, repo impl): best-vmunet-scratch-isic18.pth   @256
  - ResNet50-UNet  (CNN, ISIC recipe): best-unet-isic18.pth            @256
  - ResNet50-UNet  (CNN, CVC recipe / standardized): unet_isic_cvcrecipe_best.pth @256 (included if present)
  - Swin-UNet      (ViT): best-swinunet-isic18.pth                     @224

Results -> interventions/results/isic_heldout_eval.json

Usage:
    python interventions/experiments/eval_isic_heldout.py
"""

import sys, os, json, datetime

import torch
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'tta_boundary_study'))
sys.path.insert(0, os.path.join(_REPO, 'SpectralMamba'))
sys.path.insert(0, _REPO)

from interventions.experiments.eval_isic_cross_arch import (
    build_model, attach_lp, IMAGENET_MEAN, IMAGENET_STD, CUTOFF, IMG_DIR, MASK_DIR,
    compute_segmentation_metrics, per_image_dice)

from interventions.intervention import FrequencyIntervention
from interventions.masks import lowpass_mask

SPLIT_JSON = os.path.join(_REPO, 'interventions', 'results', 'splits', 'isic_split.json')
STD_RN50_CKPT = os.path.join(_REPO, 'interventions', 'checkpoints', 'unet_isic_cvcrecipe_best.pth')
STD_SWINUNETR_CKPT = os.path.join(_REPO, 'interventions', 'checkpoints',
                                  'swinunetr_isic_cvcrecipe_best.pth')


def build_model_heldout(arch, ckpt, device):
    """build_model plus the ISIC-trained MONAI Swin-UNETR (ViT leg closure)."""
    if arch == 'swinunetr':
        from src.models.swin_unetr_cvc import get_swin_unetr
        model = get_swin_unetr().to(device)
        sd = torch.load(ckpt, map_location=device)
        if isinstance(sd, dict):
            sd = sd.get('model_state_dict') or sd.get('state_dict') or sd
        model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=True)
        model.eval()
        return model
    return build_model(arch, ckpt, device)


def attach_lp_heldout(model, arch, interv):
    """attach_lp plus the CVC-protocol Swin-UNETR hook layout (swinViT blocks)."""
    if arch == 'swinunetr':
        from interventions.experiments.cross_arch_cvc_eval import make_lp_hook
        handles = []
        swin = model.swinViT
        for i in range(1, 5):
            layer = getattr(swin, f'layers{i}')[0]
            for blk in layer.blocks:
                handles.append(blk.register_forward_hook(make_lp_hook(interv, nhwc=True)))
        return handles
    return attach_lp(model, arch, interv)


def load_isic_pairs(names):
    pairs = []
    for name in names:
        base = os.path.splitext(name)[0]
        mask_p = os.path.join(MASK_DIR, base + '_segmentation.png')
        if not os.path.exists(mask_p):
            raise FileNotFoundError(f'Missing mask for {name}')
        p = os.path.join(IMG_DIR, name)
        # Native ISIC images are ~29MP; pre-resize to 256 to bound memory
        # (evaluate() re-resizes to model resolution, a no-op at 256).
        img = np.array(__import__('PIL').Image.open(p).convert('RGB').resize(
            (256, 256), __import__('PIL').Image.BILINEAR), dtype=np.float32) / 255.0
        msk = np.array(__import__('PIL').Image.open(mask_p).convert('L').resize(
            (256, 256), __import__('PIL').Image.NEAREST), dtype=np.float32) / 255.0
        pairs.append((torch.from_numpy(img).permute(2, 0, 1),
                      (torch.from_numpy(msk) > 0.5).float().unsqueeze(0)))
    return pairs


def hd95_fast(pred, target, threshold=0.5):
    """Symmetric Hausdorff via EDT (O(N)) — EXACTLY equal to
    max(directed_hausdorff(pred,target), directed_hausdorff(target,pred)),
    i.e. the same metric as tta_boundary_study's hausdorff_95() (which is a
    max-Hausdorff, not a true 95th percentile) but fast enough for large
    masks (directed_hausdorff is O(N*M) and takes minutes per ISIC lesion).
    """
    from scipy.ndimage import distance_transform_edt
    pred_bin = (pred > threshold).astype(bool)
    target_bin = target.astype(bool)
    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return float('inf')
    d_pred = distance_transform_edt(~target_bin)[pred_bin]
    d_tgt = distance_transform_edt(~pred_bin)[target_bin]
    return float(max(d_pred.max(), d_tgt.max()))


def boundary_metrics_for(per_preds, per_gts):
    """BF1 + HD95 per-image (well-defined only when preds contain positive class)."""
    from tta_boundary_study.src.metrics.boundary_metrics import boundary_f1
    bf1s, hd95s = [], []
    for p, g in zip(per_preds, per_gts):
        if p.max() == 0:
            bf1s.append(0.0); hd95s.append(np.inf)
            continue
        bf1s.append(boundary_f1(p, g))
        hd95s.append(hd95_fast(p, g))
    return {'bf1_mean': float(np.nanmean(bf1s)), 'bf1_std': float(np.nanstd(bf1s)),
            'hd95_mean': float(np.nanmean([h for h in hd95s if np.isfinite(h)]))
                        if any(np.isfinite(h) for h in hd95s) else None,
            'hd95_defined_n': int(sum(np.isfinite(h) for h in hd95s)),
            'bf1_defined_n': len(bf1s)}


def evaluate_full(model, pairs, device, size, needs_sigmoid, mode='clean'):
    """Like eval_isic_cross_arch.evaluate but returns per-image pred/gt arrays for
    boundary metrics and per-image stats."""
    import torch.nn.functional as F
    per_dice, per_preds, per_gts = [], [], []
    with torch.no_grad():
        for img01, gt in pairs:
            img_rs = F.interpolate(img01.unsqueeze(0), size=(size, size),
                                   mode='bilinear', align_corners=False).squeeze(0)
            gt_rs = F.interpolate(gt.unsqueeze(0), size=(size, size),
                                  mode='nearest').squeeze(0)
            if mode == 'blur':
                from interventions.train_vmunet_cvc_tsa import apply_tsa
                img_rs = apply_tsa(img_rs.unsqueeze(0), cutoff=CUTOFF, p=1.0).squeeze(0)
            inp = torch.zeros_like(img_rs)
            for c in range(3):
                inp[c] = (img_rs[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            out = model(inp.unsqueeze(0).to(device))
            probs = torch.sigmoid(out) if needs_sigmoid else out
            p_np = probs.squeeze(1).squeeze(0).cpu().numpy()
            gt_np = gt_rs.squeeze(0).cpu().numpy()
            per_dice.append(per_image_dice(p_np, gt_np))
            per_preds.append(p_np)
            per_gts.append(gt_np)
    pooled = compute_segmentation_metrics(
        np.concatenate([p.ravel() for p in per_preds]),
        np.concatenate([g.ravel() for g in per_gts]))
    return (pooled, float(np.mean(per_dice)), float(np.std(per_dice)), per_dice,
            per_preds, per_gts)


import argparse

def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--max_images', type=int, default=None,
                    help='Limit evaluation to the first N test images (smoke-test mode).')
    ap.add_argument('--models', default='all', choices=['all', 'existing', 'standardized'],
                    help="'all' = existing checkpoints + standardized CNN (if present); "
                         "'existing' = repo checkpoints only; 'standardized' = new CVC-recipe "
                         "checkpoints only (rerun after retraining completes).")
    ap.add_argument('--output_json', default=os.path.join(
        _REPO, 'interventions', 'results', 'isic_heldout_eval.json'))
    args, _ = ap.parse_known_args()

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    test_names = split['test']
    if args.max_images:
        test_names = test_names[:args.max_images]
    print(f'Held-out ISIC test set: {len(test_names)} images')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pairs = load_isic_pairs(test_names)

    MODELS = [
        ('VM-UNet (repo-impl SSM)', 'SSM', 'vmunet',
         os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                      'best-vmunet-scratch-isic18.pth'), 256, False),
        ('ResNet50-UNet (ISIC recipe)', 'CNN', 'resnet50',
         os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                      'best-unet-isic18.pth'), 256, True),
        ('Swin-UNet (ViT)', 'ViT', 'swinunet',
         os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                      'best-swinunet-isic18.pth'), 224, True),
    ]
    STD_MODELS = []
    if os.path.exists(STD_RN50_CKPT):
        STD_MODELS.append(('ResNet50-UNet (CVC recipe, standardized)', 'CNN', 'resnet50',
                           STD_RN50_CKPT, 256, True))
    if os.path.exists(STD_SWINUNETR_CKPT):
        STD_MODELS.append(('Swin-UNETR (CVC recipe, ViT leg closure)', 'ViT', 'swinunetr',
                           STD_SWINUNETR_CKPT, 256, True))
    if args.models == 'existing':
        pass  # only repo checkpoints
    elif args.models == 'standardized':
        MODELS = STD_MODELS
    else:  # 'all'
        MODELS = MODELS[:2] + STD_MODELS + MODELS[2:]
    if args.models in ('all', 'standardized'):
        print(f'Standardized ResNet50-UNet (CVC recipe): '
              f'{"included" if STD_MODELS else "checkpoint not found yet; rerun after training completes"}')

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, CUTOFF, device=dev, dtype=dt)
    interv_lp = FrequencyIntervention(lp_fn, check_nan=True)

    results = {'experiment': 'ISIC held-out Fourier benchmark (Phase 2)',
               'cutoff': CUTOFF, 'num_images': len(pairs),
               'device': str(device), 'timestamp': datetime.datetime.now().isoformat(),
               'models': []}

    for name, family, arch, ckpt, size, needs_sigmoid in MODELS:
        print(f'\n--- {name} ({family}, {size}x{size}) ---')
        model = build_model_heldout(arch, ckpt, device)

        # Clean pass must run WITHOUT hooks (intervention is deterministic; attaching
        # hooks first would make clean == feat, silently killing the comparison).
        clean = evaluate_full(model, pairs, device, size, needs_sigmoid, 'clean')
        handles = attach_lp_heldout(model, arch, interv_lp)
        feat = evaluate_full(model, pairs, device, size, needs_sigmoid, 'feat')
        for h in handles:
            h.remove()
        blur = evaluate_full(model, pairs, device, size, needs_sigmoid, 'blur')

        entry = {
            'name': name, 'family': family, 'architecture': arch, 'checkpoint': ckpt,
            'clean': clean[0], 'feature_lp': feat[0], 'input_blur': blur[0],
            'per_image_dice_mean': {'clean': clean[1], 'feature_lp': feat[1], 'input_blur': blur[1]},
            'per_image_dice_std': {'clean': clean[2], 'feature_lp': feat[2], 'input_blur': blur[2]},
            'delta_pct_feature': (feat[0]['dice'] - clean[0]['dice']) / (clean[0]['dice'] + 1e-12) * 100.0,
            'per_image_dice': {'clean': clean[3], 'feature_lp': feat[3], 'input_blur': blur[3]},
        }
        if family == 'SSM':
            entry['boundary'] = {
                'clean': boundary_metrics_for(clean[4], clean[5]),
                'feature_lp': boundary_metrics_for(feat[4], feat[5]),
                'input_blur': boundary_metrics_for(blur[4], blur[5])}
        results['models'].append(entry)
        print(f"  Clean Dice: {clean[0]['dice']:.4f} | Feat-LP Dice: {feat[0]['dice']:.4f} "
              f"({entry['delta_pct_feature']:+.1f}%) | Input-LP: {blur[0]['dice']:.4f}")

    out_json = args.output_json
    results['max_images'] = args.max_images
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {out_json}')


if __name__ == '__main__':
    main()
