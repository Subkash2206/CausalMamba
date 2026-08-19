"""
make_isic_robustness_figures.py — Qualitative ISIC figure, mirror of the CVC panels in
make_robustness_figures.py, rendered with the Phase-2 held-out checkpoints.

For three representative ISIC2018 held-out test images (small / medium / large lesion
fraction), renders rows Clean / Input-LP (cutoff 0.25) / Feature-LP (cutoff 0.25) and
one column per held-out model:

    VM-UNet (SSM) @256 | ResNet50-UNet (ISIC recipe) @256 | ResNet50-UNet (CVC recipe)
    @256 | Swin-UNETR (ViT) @256 | Swin-UNet (ViT) @224

Protocol identical to eval_isic_heldout.py (ImageNet normalization, per-model resize,
feature-space LP via forward hooks, input-space LP via apply_tsa at the same cutoff),
so the panels match the audited numbers.

Outputs 300-dpi PNGs to interventions/results/paper_v2/figures/.

Usage:
    python interventions/experiments/make_isic_robustness_figures.py
"""

import sys, os, json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'tta_boundary_study'))
sys.path.insert(0, os.path.join(_REPO, 'SpectralMamba'))
sys.path.insert(0, _REPO)

from interventions.experiments.eval_isic_heldout import (
    build_model_heldout, attach_lp_heldout, load_isic_pairs,
    IMAGENET_MEAN, IMAGENET_STD, CUTOFF, SPLIT_JSON)
from interventions.intervention import FrequencyIntervention
from interventions.masks import lowpass_mask
from interventions.train_vmunet_cvc_tsa import apply_tsa

FIG_DIR = os.path.join(_REPO, 'interventions', 'results', 'paper_v2', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

STD_RN50_CKPT = os.path.join(_REPO, 'interventions', 'checkpoints',
                             'unet_isic_cvcrecipe_best.pth')
STD_SWINUNETR_CKPT = os.path.join(_REPO, 'interventions', 'checkpoints',
                                  'swinunetr_isic_cvcrecipe_best.pth')

MODELS = [
    ('VM-UNet (SSM)', 'vmunet',
     os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                  'best-vmunet-scratch-isic18.pth'), 256, False),
    ('ResNet50-UNet (ISIC recipe)', 'resnet50',
     os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                  'best-unet-isic18.pth'), 256, True),
    ('ResNet50-UNet (CVC recipe)', 'resnet50', STD_RN50_CKPT, 256, True),
    ('Swin-UNETR (ViT)', 'swinunetr', STD_SWINUNETR_CKPT, 256, True),
    ('Swin-UNet (ViT)', 'swinunet',
     os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                  'best-swinunet-isic18.pth'), 224, True),
]

CONDITIONS = [('Clean', False, False), ('Input-LP', True, False),
              ('Feature-LP', False, True)]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    test_names = split['test']
    pairs = load_isic_pairs(test_names)          # pre-resized to 256x256
    fracs = [float(p[1].float().mean()) for p in pairs]

    # Pick one small, one medium, one large lesion (skip empty / near-full masks).
    order = sorted(range(len(pairs)), key=lambda i: fracs[i])
    valid = [i for i in order if 0.01 < fracs[i] < 0.9]
    if len(valid) < 3:
        valid = order
    picks = [('small', valid[0]), ('medium', valid[len(valid) // 2]),
             ('large', valid[-1])]

    print(f'Building {len(MODELS)} held-out models...')
    models = [(mname, arch, size, sig,
               build_model_heldout(arch, ckpt, device))
              for mname, arch, ckpt, size, sig in MODELS]

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, CUTOFF, device=dev, dtype=dt)
    interv = FrequencyIntervention(lp_fn, check_nan=True)

    for lab, idx in picks:
        img01, _ = pairs[idx]                     # (3,256,256), (1,256,256)
        rows = []
        for cond, blur, feat in CONDITIONS:
            disp = img01
            preds = []
            for mname, arch, size, sig, model in models:
                handles = attach_lp_heldout(model, arch, interv) if feat else []
                with torch.no_grad():
                    inp = F.interpolate(img01.unsqueeze(0), size=(size, size),
                                        mode='bilinear', align_corners=False)
                    if blur:
                        inp = apply_tsa(inp, cutoff=CUTOFF, p=1.0)
                    norm = torch.zeros_like(inp)
                    for c in range(3):
                        norm[:, c] = (inp[:, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
                    out = model(norm.to(device))
                for h in handles:
                    h.remove()
                probs = torch.sigmoid(out) if sig else out
                p_np = probs.squeeze(1).squeeze(0).cpu().numpy()
                if p_np.shape[0] != 256:          # Swin-UNet native 224 -> uniform grid
                    p_np = np.array(__import__('PIL').Image.fromarray(
                        (p_np * 255).astype(np.uint8)).resize(
                            (256, 256), __import__('PIL').Image.NEAREST),
                        dtype=np.float32) / 255.0
                preds.append(p_np)
            rows.append((cond, disp, preds))

        n_cols = 1 + len(MODELS)
        fig, axes = plt.subplots(len(CONDITIONS), n_cols, figsize=(3.0 * n_cols, 9.5))
        for r, (cond, disp, preds) in enumerate(rows):
            img_np = disp.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            axes[r][0].imshow(np.clip(img_np, 0, 1))
            axes[r][0].set_ylabel(cond, rotation=0, labelpad=32, fontsize=11)
            axes[r][0].axis('off')
            for c, p in enumerate(preds):
                axes[r][c + 1].imshow(p, cmap='gray', vmin=0, vmax=1)
                axes[r][c + 1].set_title(MODELS[c][0], fontsize=11)
                axes[r][c + 1].axis('off')
        axes[0][0].set_title('Input', fontsize=11)
        plt.tight_layout()
        out_p = os.path.join(FIG_DIR, f'isic_sample_{lab}.png')
        plt.savefig(out_p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  saved {out_p}  (image {test_names[idx]}, lesion frac {fracs[idx]:.3f})')

    print(f'\nDone -> {FIG_DIR}')


if __name__ == '__main__':
    main()

