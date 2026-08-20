"""eval_band_energy.py - FFT band-energy ratios (low/mid/high + AVR) of native
intermediate feature maps on the CURRENT matched checkpoints and HELD-OUT splits.

Replaces the stale SpectralMamba/run_band_only.py (legacy UNet/Swin checkpoints,
random 80/20 dev split) with the paper's matched legs. Band definition identical to
run_band_only.compute_bands (low<=0.25, mid 0.25-0.75, high>0.75, avr>0.5 radial).

Usage:
    python interventions/experiments/eval_band_energy.py [--limit N]
"""

import sys, os, json, argparse, datetime

import torch
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [_REPO, os.path.join(_REPO, 'SpectralMamba'),
          os.path.join(_REPO, 'tta_boundary_study')]:
    sys.path.insert(0, p)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from interventions.experiments.eval_isic_heldout import (
    build_model_heldout, load_isic_pairs, IMAGENET_MEAN, IMAGENET_STD)
from interventions.experiments.eval_cvc_heldout import CVCListDataset

CVC_SPLIT = os.path.join(_REPO, 'interventions', 'results', 'splits', 'cvc_split.json')
ISIC_SPLIT = os.path.join(_REPO, 'interventions', 'results', 'splits', 'isic_split.json')
OUT_DIR = os.path.join(_REPO, 'interventions', 'results', 'band_energy')
os.makedirs(OUT_DIR, exist_ok=True)

LEGS = [
    ('cnn_cvc', 'resnet50', os.path.join(_REPO, 'tta_boundary_study', 'checkpoints',
                                         'unet_cvc_best_256.pth'), 'cvc', True),
    ('cnn_isic', 'resnet50', os.path.join(_REPO, 'interventions', 'checkpoints',
                                          'unet_isic_cvcrecipe_best.pth'), 'isic', True),
    ('ssm_cvc', 'vmunet', os.path.join(_REPO, 'tta_boundary_study', 'checkpoints',
                                       'best-vmunet-cvc.pth'), 'cvc', False),
    ('ssm_isic', 'vmunet', os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                                        'best-vmunet-scratch-isic18.pth'), 'isic', False),
    ('vit_cvc', 'swinunetr', os.path.join(_REPO, 'interventions', 'results',
                                          'best-swinunetr-cvc-256',
                                          'best-swinunetr-cvc-256.pth'), 'cvc', True),
    ('vit_isic', 'swinunetr', os.path.join(_REPO, 'interventions', 'checkpoints',
                                           'swinunetr_isic_cvcrecipe_best.pth'), 'isic', True),
]
def band_ratios(fmap):
    f = fmap.float()
    if f.shape[-1] in (96, 192, 384, 768):
        f = f.permute(0, 3, 1, 2)
    f = f.mean(dim=1, keepdim=True)
    f = f - f.mean(dim=(-2, -1), keepdim=True)
    B, C, H, W = f.shape
    spec = torch.fft.fftshift(torch.fft.fft2(f), dim=(-2, -1))
    power = torch.abs(spec) ** 2
    y = torch.arange(H).view(1, 1, H, 1).to(power)
    x = torch.arange(W).view(1, 1, 1, W).to(power)
    fr = torch.max(torch.abs(y - H // 2) / (H / 2), torch.abs(x - W // 2) / (W / 2))
    tot = power.sum().item()
    if tot == 0:
        return 0.0, 0.0, 0.0, 0.0
    low = (power * (fr <= 0.25)).sum().item() / tot
    mid = (power * ((fr > 0.25) & (fr <= 0.75))).sum().item() / tot
    high = (power * (fr > 0.75)).sum().item() / tot
    avr = (power * (fr > 0.5)).sum().item() / tot
    return low, mid, high, avr


def stage_targets(model, arch):
    if arch == 'resnet50':
        return [('level1', model.encoder.layer1, 'nchw'),
                ('level2', model.encoder.layer2, 'nchw'),
                ('level3', model.encoder.layer3, 'nchw'),
                ('level4', model.encoder.layer4, 'nchw')]
    if arch == 'vmunet':
        out = []
        for i in range(4):
            out.append((f'level{i + 1}', model.vmunet.layers[i].blocks[-1], 'nhwc'))
        return out
    if arch == 'swinunetr':
        out = []
        swin = model.swinViT
        for i in range(1, 5):
            out.append((f'level{i}', getattr(swin, f'layers{i}')[0].blocks[-1], 'nhwc'))
        return out
    raise ValueError(arch)


def prep(img01, gt, dataset, device):
    img01, gt = img01.to(device), gt.to(device)
    if dataset == 'isic':
        import torch.nn.functional as F
        img01 = F.interpolate(img01.unsqueeze(0), size=(256, 256),
                              mode='bilinear', align_corners=False)
        norm = torch.zeros_like(img01)
        for c in range(3):
            norm[:, c] = (img01[:, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        img01 = norm
    else:
        img01 = img01.unsqueeze(0)
    return img01, gt
def run_leg(label, arch, ckpt, dataset, sigmoid, limit, device):
    print(f'\n===== {label}: {arch} {dataset.upper()} =====')
    print(f'  ckpt: {ckpt}')
    model = build_model_heldout(arch, ckpt, device)
    targets = stage_targets(model, arch)

    if dataset == 'cvc':
        with open(CVC_SPLIT) as f:
            names = json.load(f)['test']
        if limit:
            names = names[:limit]
        ds = CVCListDataset(names, os.path.join(_REPO, 'tta_boundary_study',
                            'cvc_clinicdb', 'original'), os.path.join(_REPO,
                            'tta_boundary_study', 'cvc_clinicdb', 'ground_truth'),
                            img_size=256)
        items = [(ds[i][0], ds[i][1]) for i in range(len(ds))]
    else:
        with open(ISIC_SPLIT) as f:
            names = json.load(f)['test']
        if limit:
            names = names[:limit]
        items = load_isic_pairs(names)
    print(f'  images: {len(items)}')

    accum = {nm: [] for nm, _, _ in targets}
    resos = {}
    with torch.no_grad():
        for img01, gt in items:
            img01, _ = prep(img01, gt, dataset, device)
            feats = {}
            handles = []
            for nm, mod, _ in targets:
                def mk(nm=nm, mod=mod):
                    def hk(module, inp, out):
                        o = out[0] if isinstance(out, tuple) else out
                        feats[nm] = o.detach()
                    return hk
                handles.append(mod.register_forward_hook(mk()))
            out = model(img01)
            for h in handles:
                h.remove()
            for nm, _, _ in targets:
                if nm not in feats:
                    continue
                f = feats[nm]
                resos[nm] = tuple(f.shape[-2:])
                accum[nm].append(band_ratios(f))
    rows = []
    for nm, _, _ in targets:
        arr = np.array(accum[nm])
        if arr.size == 0:
            print(f'  {nm}: NO FEATURES captured (hook mismatch?)')
            continue
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
        rows.append({'leg': label, 'stage': nm, 'resolution': 'x'.join(map(str, resos[nm])),
                     'low_band_ratio': float(m[0]), 'mid_band_ratio': float(m[1]),
                     'high_band_ratio': float(m[2]), 'avr_ratio': float(m[3]),
                     'low_std': float(s[0]), 'mid_std': float(s[1]),
                     'high_std': float(s[2]), 'n': len(arr)})
        print(f'  {nm:<7s} {rows[-1]["resolution"]:<9s} low={m[0]:.4f} mid={m[1]:.4f} '
              f'high={m[2]:.4f} avr={m[3]:.4f}')
    return rows


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--limit', type=int, default=0)
    args, _ = ap.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    all_rows = []
    for label, arch, ckpt, dataset, sigmoid in LEGS:
        all_rows += run_leg(label, arch, ckpt, dataset, sigmoid, args.limit, device)
    out_p = os.path.join(OUT_DIR, 'band_energy_results.csv')
    with open(out_p, 'w') as f:
        f.write('leg,stage,resolution,low_band_ratio,mid_band_ratio,high_band_ratio,'
                'avr_ratio,low_std,mid_std,high_std,n\n')
        for r in all_rows:
            f.write(','.join(str(r[k]) for k in
                             ['leg', 'stage', 'resolution', 'low_band_ratio',
                              'mid_band_ratio', 'high_band_ratio', 'avr_ratio',
                              'low_std', 'mid_std', 'high_std', 'n']) + '\n')
    with open(os.path.join(OUT_DIR, 'band_energy_results.json'), 'w') as f:
        json.dump(all_rows, f, indent=2)
    print(f'\nsaved -> {out_p}')


if __name__ == '__main__':
    main()
