"""
eval_cvc_heldout.py — Cross-architecture Fourier benchmark on the CVC-ClinicDB
HELD-OUT test split (62 frames from carve_splits.py).

Same protocol as cross_arch_cvc_eval.py (feature-space LP cutoff 0.25) but evaluated
on the untouched carved test split, giving Table-2 symmetry with the ISIC held-out
eval. Models are the same checkpoints used for the 123-val cross-arch benchmark.

Results -> interventions/results/cvc_heldout_eval.json

Usage:
    python interventions/experiments/eval_cvc_heldout.py
"""

import sys, os, json, argparse, datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'tta_boundary_study'))
sys.path.insert(0, os.path.join(_REPO, 'SpectralMamba'))
sys.path.insert(0, _REPO)

from interventions.experiments.cross_arch_cvc_eval import (
    build_model, attach_lp, compute_segmentation_metrics, CUTOFF, IMG_SIZE,
    CVC_IMG_DIR, CVC_MASK_DIR)
from interventions.experiments.eval_isic_heldout import hd95_fast
from tta_boundary_study.src.metrics.boundary_metrics import boundary_f1

from interventions.intervention import FrequencyIntervention
from interventions.masks import lowpass_mask

SPLIT_JSON = os.path.join(_REPO, 'interventions', 'results', 'splits', 'cvc_split.json')


class CVCListDataset(Dataset):
    """CVC frames from an explicit filename list, [0,1] normalization (CVC recipe)."""
    def __init__(self, names, img_dir, mask_dir, img_size=256):
        self.names, self.img_dir, self.mask_dir, self.img_size = names, img_dir, mask_dir, img_size
        import cv2
        self.cv2 = cv2

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        cv2 = self.cv2
        name = self.names[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, name)),
                           cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        return (torch.from_numpy(img).permute(2, 0, 1).float(),
                torch.from_numpy(mask).unsqueeze(0).float())


def evaluate_full(model, loader, device, needs_sigmoid, mode='clean'):
    """Pooled + per-image metrics; mode 'blur' low-passes the input first."""
    per_dice, preds, gts, per_preds, per_gts = [], [], [], [], []
    with torch.no_grad():
        for img01, gt in loader:
            if mode == 'blur':
                from interventions.train_vmunet_cvc_tsa import apply_tsa
                img01 = apply_tsa(img01, cutoff=CUTOFF, p=1.0)
            img01, gt = img01.to(device), gt.to(device)
            out = model(img01)
            probs = torch.sigmoid(out) if needs_sigmoid else out
            p_np = probs.squeeze(1).squeeze(0).cpu().numpy()
            gt_np = gt.squeeze(1).squeeze(0).cpu().numpy()
            inter = ((p_np > 0.5) & (gt_np > 0.5)).sum()
            union = (p_np > 0.5).sum() + (gt_np > 0.5).sum()
            per_dice.append(2 * inter / max(union, 1e-8))
            preds.append(p_np.ravel()); gts.append(gt_np.ravel())
            per_preds.append(p_np); per_gts.append(gt_np)
    pooled = compute_segmentation_metrics(np.concatenate(preds), np.concatenate(gts))
    return pooled, float(np.mean(per_dice)), per_dice, per_preds, per_gts


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_json', default=os.path.join(
        _REPO, 'interventions', 'results', 'cvc_heldout_eval.json'))
    args, _ = ap.parse_known_args()

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    test_names = split['test']
    ds = CVCListDataset(test_names, CVC_IMG_DIR, CVC_MASK_DIR, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print(f'CVC held-out test set: {len(ds)} images @ {IMG_SIZE}x{IMG_SIZE}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CKPTS = [
        ('VM-UNet', 'SSM', 'vmunet',
         os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'best-vmunet-cvc.pth'),
         False),
        ('VM-UNet-TSA (Defense)', 'SSM', 'vmunet',
         os.path.join(_REPO, 'interventions', 'results', 'best-vmunet-cvc-tsa-finetune',
                      'best-vmunet-cvc.pth'), False),
        ('ResNet50-UNet', 'CNN', 'resnet50',
         os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'unet_cvc_best_256.pth'),
         True),
        ('Swin-UNETR', 'ViT', 'swinunetr',
         os.path.join(_REPO, 'interventions', 'results', 'best-swinunetr-cvc-256',
                      'best-swinunetr-cvc-256.pth'), True),
    ]

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, CUTOFF, device=dev, dtype=dt)
    interv_lp = FrequencyIntervention(lp_fn, check_nan=True)

    results = {'experiment': 'CVC held-out Fourier benchmark (Phase 2)',
               'cutoff': CUTOFF, 'img_size': IMG_SIZE, 'num_images': len(ds),
               'device': str(device), 'timestamp': datetime.datetime.now().isoformat(),
               'models': []}

    for name, family, arch, ckpt, needs_sigmoid in CKPTS:
        print(f'\n--- {name} ({family}) ---')
        model = build_model(arch, ckpt, device)

        clean = evaluate_full(model, loader, device, needs_sigmoid, 'clean')
        handles = attach_lp(model, arch, interv_lp)
        feat = evaluate_full(model, loader, device, needs_sigmoid, 'feat')
        for h in handles:
            h.remove()
        blur = evaluate_full(model, loader, device, needs_sigmoid, 'blur')

        entry = {
            'name': name, 'family': family, 'architecture': arch, 'checkpoint': ckpt,
            'clean': clean[0], 'feature_lp': feat[0], 'input_blur': blur[0],
            'per_image_dice_mean': {'clean': clean[1], 'feature_lp': feat[1]},
            'per_image_dice': {'clean': clean[2], 'feature_lp': feat[2], 'input_blur': blur[2]},
            'delta_pct_feature': (feat[0]['dice'] - clean[0]['dice']) / (clean[0]['dice'] + 1e-12) * 100.0,
        }
        if family == 'SSM':
            b = {}
            for mode, per in [('clean', clean), ('feature_lp', feat), ('input_blur', blur)]:
                bf1s, hd95s = [], []
                for p, g in zip(per[3], per[4]):
                    if p.max() == 0:
                        bf1s.append(0.0); hd95s.append(np.inf)
                        continue
                    bf1s.append(boundary_f1(p, g)); hd95s.append(hd95_fast(p, g))
                b[mode] = {'bf1_mean': float(np.nanmean(bf1s)),
                           'hd95_mean': float(np.nanmean([h for h in hd95s if np.isfinite(h)]))
                           if any(np.isfinite(h) for h in hd95s) else None}
            entry['boundary'] = b
        results['models'].append(entry)
        print(f"  Clean Dice: {clean[0]['dice']:.4f} | Feat-LP: {feat[0]['dice']:.4f} "
              f"({entry['delta_pct_feature']:+.1f}%) | Input-LP: {blur[0]['dice']:.4f}")

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {args.output_json}')


if __name__ == '__main__':
    main()

