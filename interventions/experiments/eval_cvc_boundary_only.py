"""
eval_cvc_boundary_only.py — Focused re-run of boundary metrics (BF1, HD95) for the
4 CVC models under Clean and Feature-space LP (cutoff 0.25).

Uses the FIXED aggregation from eval_cvc_robustness.evaluate (HD95 NaN-guarded on
empty predictions, aggregated with nanmean), so empty-prediction images no longer
poison the mean.

Usage:
    python interventions/experiments/eval_cvc_boundary_only.py
"""

import sys, os, json

import torch
from torch.utils.data import DataLoader

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'tta_boundary_study'))
sys.path.insert(0, os.path.join(_REPO, 'SpectralMamba'))
sys.path.insert(0, _REPO)

from interventions.experiments.eval_cvc_robustness import (
    build_model, evaluate, attach_lp, CVC_IMG_DIR, CVC_MASK_DIR, IMG_SIZE)
from src.datasets.cvc_dataset import CVCDataset
from interventions.intervention import FrequencyIntervention
from interventions.masks import lowpass_mask

RES = os.path.join(_REPO, 'interventions', 'results')
JSON_PATH = os.path.join(RES, 'cvc_robustness_eval.json')

MODELS = [
    ('VM-UNet', 'vmunet',
     os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'best-vmunet-cvc.pth'), False),
    ('VM-UNet-TSA', 'vmunet',
     os.path.join(_REPO, 'interventions', 'results', 'best-vmunet-cvc-tsa-finetune',
                  'best-vmunet-cvc.pth'), False),
    ('ResNet50-UNet', 'resnet50',
     os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'unet_cvc_best_256.pth'), True),
    ('Swin-UNETR', 'swinunetr',
     os.path.join(_REPO, 'interventions', 'results', 'best-swinunetr-cvc-256',
                  'best-swinunetr-cvc-256.pth'), True),
]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 78)
    print('Boundary metrics re-run (fixed HD95 NaN handling) | @256')
    print('=' * 78)
    print(f'Device: {device}')

    val_ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='val', img_size=IMG_SIZE)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
    interv = FrequencyIntervention(lp_fn, check_nan=True)

    updated = {}
    for name, arch, ckpt, sig in MODELS:
        model = build_model(arch, ckpt, device)
        clean = evaluate(model, loader, device, sig, boundary=True)
        handles = attach_lp(model, arch, interv)
        lp = evaluate(model, loader, device, sig, boundary=True)
        for h in handles:
            h.remove()
        del model

        print(f'  {name:<16} clean BF1={clean["boundary_f1_mean"]:.4f} '
              f'HD95={clean["hd95_mean"]:.2f} | LP BF1={lp["boundary_f1_mean"]:.4f} '
              f'HD95={lp["hd95_mean"]:.2f}')
        updated[name] = {
            'clean_f1': clean['boundary_f1_mean'], 'clean_hd95': clean['hd95_mean'],
            'lp_f1': lp['boundary_f1_mean'], 'lp_hd95': lp['hd95_mean'],
            'lp_dice': lp['pooled']['dice'],
        }

    # Merge into cvc_robustness_eval.json
    with open(JSON_PATH) as f:
        j = json.load(f)
    for m in j['models']:
        u = updated.get(m['name'])
        if not u:
            continue
        m['clean']['boundary_f1'] = u['clean_f1']
        m['clean']['hd95'] = u['clean_hd95']
        m['boundary'] = {'dice': u['lp_dice'], 'boundary_f1': u['lp_f1'],
                         'hd95': u['lp_hd95']}
    with open(JSON_PATH, 'w') as f:
        json.dump(j, f, indent=2)
    print(f'\nUpdated -> {JSON_PATH}')

    print('\nCorrected boundary table (clean vs feature-LP 0.25)')
    print('| Model | Clean BF1 | Clean HD95 | LP BF1 | LP HD95 |')
    print('|---|---:|---:|---:|---:|')
    for m in j['models']:
        u = updated.get(m['name'], {})
        print(f'| {m["name"]} | {u.get("clean_f1", "n/a"):.3f} | '
              f'{u.get("clean_hd95", float("nan")):.1f} | '
              f'{u.get("lp_f1", "n/a"):.3f} | {u.get("lp_hd95", float("nan")):.1f} |')


if __name__ == '__main__':
    main()
