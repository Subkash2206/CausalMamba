"""Re-run of stale exp11/exp8/exp3 layer-wise LP localization on current paper
checkpoints and HELD-OUT splits. Targets:
  cnn_cvc : ResNet50-UNet unet_cvc_best_256.pth on CVC 62-test
  cnn_isic: ResNet50-UNet CVC-recipe unet_isic_cvcrecipe_best.pth on ISIC 260-test
  ssm_isic: VM-UNet legacy best-vmunet-scratch-isic18.pth on ISIC 260-test
Usage:
  python interventions/experiments/eval_layerwise_heldout.py --target cnn_cvc
  ... --target cnn_isic | --target ssm_isic | --limit 8 (smoke)
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

from interventions.experiments.cross_arch_cvc_eval import (
    HOOK_TARGETS, make_lp_hook, _resolve_path,
    compute_segmentation_metrics, CUTOFF, IMG_SIZE)
from interventions.experiments.eval_isic_cross_arch import build_model  # repo VM-UNet for ISIC
from interventions.experiments.eval_cvc_heldout import CVCListDataset
from interventions.experiments.eval_isic_heldout import load_isic_pairs, \
    IMAGENET_MEAN, IMAGENET_STD
from interventions.intervention import FrequencyIntervention
from interventions.masks import lowpass_mask

CVC_SPLIT = os.path.join(_REPO, 'interventions', 'results', 'splits', 'cvc_split.json')
ISIC_SPLIT = os.path.join(_REPO, 'interventions', 'results', 'splits', 'isic_split.json')
OUT_DIR = os.path.join(_REPO, 'interventions', 'results', 'layerwise_heldout')
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = {
    'cnn_cvc': {'arch': 'resnet50', 'dataset': 'cvc',
                'ckpt': os.path.join(_REPO, 'tta_boundary_study', 'checkpoints',
                                     'unet_cvc_best_256.pth'), 'sigmoid': True},
    'cnn_isic': {'arch': 'resnet50', 'dataset': 'isic',
                 'ckpt': os.path.join(_REPO, 'interventions', 'checkpoints',
                                      'unet_isic_cvcrecipe_best.pth'), 'sigmoid': True},
    'ssm_isic': {'arch': 'vmunet', 'dataset': 'isic',
                 'ckpt': os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                                      'best-vmunet-scratch-isic18.pth'), 'sigmoid': False},
}
def avr_ratio(fmap):
    """Fraction of power above radial freq 0.5 on the mean-over-channel spectrum."""
    f = fmap.float()
    if f.shape[-1] in (96, 192, 384, 768):
        f = f.permute(0, 3, 1, 2)
    if f.dim() == 4:
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
        return 0.0
    return float((power * (fr > 0.5)).sum().item() / tot)


def forward(model, img, sigmoid):
    with torch.no_grad():
        out = model(img)
        return torch.sigmoid(out) if sigmoid else out


def prep_input(img01, gt, dataset, device):
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
def run_target(target, limit, device):
    cfg = TARGETS[target]
    arch, dataset = cfg['arch'], cfg['dataset']
    print('=' * 90)
    print(f'{target}: {arch} on {dataset.upper()}  | rho={CUTOFF}')
    print(f'  ckpt: {cfg["ckpt"]}')
    print('=' * 90)

    model = build_model(arch, cfg['ckpt'], device)
    if arch == 'vmunet':
        stages = [(nm, mod) for nm, mod in model.named_modules()
                  if 'VSSBlock' in type(mod).__name__]
        nhwc = True
    else:
        stages = HOOK_TARGETS[:]
        nhwc = False

    if dataset == 'cvc':
        with open(CVC_SPLIT) as f:
            names = json.load(f)['test']
        if limit:
            names = names[:limit]
        ds = CVCListDataset(names, os.path.join(_REPO, 'tta_boundary_study',
                            'cvc_clinicdb', 'original'), os.path.join(_REPO,
                            'tta_boundary_study', 'cvc_clinicdb', 'ground_truth'),
                            img_size=IMG_SIZE)
        items = [(ds[i][0], ds[i][1]) for i in range(len(ds))]
    else:
        with open(ISIC_SPLIT) as f:
            names = json.load(f)['test']
        if limit:
            names = names[:limit]
        items = load_isic_pairs(names)

    print(f'  images: {len(items)}  stages: {len(stages)}')
    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, CUTOFF, device=dev, dtype=dt)
    interv = FrequencyIntervention(lp_fn, check_nan=True)

    def stage_mod(st):
        if isinstance(st, tuple) and hasattr(st[1], 'register_forward_hook'):
            return st[1]
        return _resolve_path(model, st[1])

    avr_before = {}
    captures = []
    for st in stages:
        nm, mod = st[0], stage_mod(st)
        def mk(nm=nm, mod=mod):
            def hk(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                avr_before[nm] = avr_ratio(o.detach())
            return hk
        captures.append(mod.register_forward_hook(mk()))
    preds, gts = [], []
    for img01, gt in items:
        img01, gt = prep_input(img01, gt, dataset, device)
        out = forward(model, img01, cfg['sigmoid'])
        p = out.squeeze(1).squeeze(0).cpu().numpy()
        g = gt.squeeze(1).squeeze(0).cpu().numpy()
        preds.append(p.ravel()); gts.append(g.ravel())
    for h in captures:
        h.remove()
    clean_metrics = compute_segmentation_metrics(np.concatenate(preds), np.concatenate(gts))
    expect = {'cnn_cvc': 0.960, 'cnn_isic': 0.892, 'ssm_isic': 0.915}[target]
    print(f'  clean pooled Dice = {clean_metrics["dice"]:.4f} (Table-2 expect {expect})')

    rows = []
    for i, st in enumerate(stages):
        nm, mod = st[0], stage_mod(st)
        store = {}
        def hk_lp(module, inp, out, nm=nm, mod=mod):
            o = out[0] if isinstance(out, tuple) else out
            mod_out = make_lp_hook(interv, nhwc)(module, inp, o)
            store['avr_after'] = avr_ratio(mod_out.detach())
            return mod_out
        h = mod.register_forward_hook(hk_lp)
        preds2, gts2 = [], []
        for img01, gt in items:
            img01, gt = prep_input(img01, gt, dataset, device)
            out = forward(model, img01, cfg['sigmoid'])
            p = out.squeeze(1).squeeze(0).cpu().numpy()
            g = gt.squeeze(1).squeeze(0).cpu().numpy()
            preds2.append(p.ravel()); gts2.append(g.ravel())
        h.remove()
        m = compute_segmentation_metrics(np.concatenate(preds2), np.concatenate(gts2))
        avr_after = store.get('avr_after')
        rows.append({'stage': nm, 'dice': m['dice'],
                     'delta_dice': m['dice'] - clean_metrics['dice'],
                     'iou': m['iou'],
                     'delta_iou': m['iou'] - clean_metrics['iou'],
                     'avr_before': avr_before.get(nm),
                     'avr_after': avr_after,
                     'delta_avr': (avr_after - avr_before.get(nm, 0.0))
                     if avr_after is not None and nm in avr_before else None})
        print(f'  [{i + 1}/{len(stages)}] {nm:<42s} dice={m["dice"]:.4f} '
              f'({rows[-1]["delta_dice"]:+.4f})', flush=True)

    rows.sort(key=lambda r: r['delta_dice'])
    print('\nstage,dice,delta_dice,iou,delta_iou,avr_before,avr_after')
    for r in rows:
        ab = r['avr_before'] if r['avr_before'] is not None else float('nan')
        aa = r['avr_after'] if r['avr_after'] is not None else float('nan')
        print(f'{r["stage"]},{r["dice"]:.4f},{r["delta_dice"]:.4f},{r["iou"]:.4f},'
              f'{r["delta_iou"]:.4f},{ab:.4f},{aa:.4f}')

    out = {'target': target, 'checkpoint': cfg['ckpt'], 'dataset': dataset.upper(),
           'split': 'held-out test', 'cutoff': CUTOFF, 'n_images': len(items),
           'clean_dice': clean_metrics['dice'],
           'timestamp': datetime.datetime.now().isoformat(), 'stages': rows}
    p = os.path.join(OUT_DIR, f'{target}_layerwise_results.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved -> {p}')


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--target', default='cnn_cvc', choices=list(TARGETS))
    ap.add_argument('--limit', type=int, default=0)
    args, _ = ap.parse_known_args()
    run_target(args.target, args.limit,
               torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


if __name__ == '__main__':
    main()
