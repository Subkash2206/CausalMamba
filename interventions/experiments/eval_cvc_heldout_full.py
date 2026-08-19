"""
eval_cvc_heldout_full.py — Full robustness protocol on the CVC-ClinicDB HELD-OUT
test split (62 frames from carve_splits.py, interventions/results/splits/cvc_split.json).

Closes the Table-1/Table-3 gap: the headline Table 2 uses the 62-image held-out
test split, but the dose-response sweep (Table 1), the input-LP / boundary /
defense characterization (Table 3) were previously computed on the 123-image
VALIDATION split. This script runs the same protocol on the 62-image held-out
TEST split, with the same four checkpoints (no retraining):

  1. VM-UNet          (SSM)  - tta_boundary_study/checkpoints/best-vmunet-cvc.pth
  2. VM-UNet-TSA      (SSM)  - interventions/results/best-vmunet-cvc-tsa-finetune/best-vmunet-cvc.pth
  3. ResNet50-UNet    (CNN)  - tta_boundary_study/checkpoints/unet_cvc_best_256.pth
  4. Swin-UNETR       (ViT)  - interventions/results/best-swinunetr-cvc-256/best-swinunetr-cvc-256.pth

Protocol (mechanics identical to eval_cvc_robustness.py, split = held-out test):
  A. Feature-space LP dose-response : cutoffs [0.10,0.15,0.20,0.25,0.30,0.40]
     (FFT->mask->IFFT on every semantic stage). The CLEAN pass runs BEFORE any
     hooks are attached; hooks are attached per-cutoff, used, then removed.
  B. Input-space LP at 0.25 (apply_tsa) — matches Table 3.
  C. Boundary metrics (clean + feat-LP 0.25): boundary-F1 and HD95 for ALL FOUR
     models. HD95 uses the fast EDT implementation hd95_fast (validated to equal
     symmetric directed_hausdorff exactly in an earlier session) — NOT the
     O(N*M) scipy directed_hausdorff call.

Guardrails:
  * --limit N runs a smoke test on the first N images: prints per-image clean vs
    feat-LP Dice to prove hook order is correct (clean != feat), and re-runs clean
    after hooks are removed to prove they did not leak. No JSON is written.
  * After the full sweep the rho=0.25 row is cross-checked against the known
    Table-2 held-out numbers (ResNet50 0.960->0.000, VM-UNet 0.912->0.244,
    Swin-UNETR 0.824->0.569). On any mismatch the script exits non-zero WITHOUT
    writing the JSON.

Results -> interventions/results/cvc_heldout_full.json (drop-in structure of
cvc_robustness_eval.json: clean / feature_dose / input_dose / boundary + per_dice).

Usage:
    python interventions/experiments/eval_cvc_heldout_full.py            # full run
    python interventions/experiments/eval_cvc_heldout_full.py --limit 4  # smoke test
"""

import sys, os, json, argparse, datetime

import torch
import numpy as np
from torch.utils.data import DataLoader

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
    build_model, attach_lp, compute_segmentation_metrics, CUTOFF, IMG_SIZE,
    CVC_IMG_DIR, CVC_MASK_DIR)
from interventions.experiments.eval_cvc_heldout import CVCListDataset, SPLIT_JSON
from interventions.experiments.eval_isic_heldout import hd95_fast
from tta_boundary_study.src.metrics.boundary_metrics import boundary_f1

from interventions.intervention import FrequencyIntervention
from interventions.masks import lowpass_mask
from interventions.train_vmunet_cvc_tsa import apply_tsa

FEAT_CUTOFFS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

# Same four checkpoints as eval_cvc_heldout.py (Table-2 numbers).
CKPTS = [
    ('VM-UNet', 'SSM', 'vmunet',
     os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'best-vmunet-cvc.pth'),
     False),
    ('VM-UNet-TSA', 'SSM', 'vmunet',
     os.path.join(_REPO, 'interventions', 'results', 'best-vmunet-cvc-tsa-finetune',
                  'best-vmunet-cvc.pth'), False),
    ('ResNet50-UNet', 'CNN', 'resnet50',
     os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'unet_cvc_best_256.pth'),
     True),
    ('Swin-UNETR', 'ViT', 'swinunetr',
     os.path.join(_REPO, 'interventions', 'results', 'best-swinunetr-cvc-256',
                  'best-swinunetr-cvc-256.pth'), True),
]

# Known Table-2 held-out numbers for the rho=0.25 cross-check (pooled Dice, 3 dp).
T2_EXPECTED = {'ResNet50-UNet': (0.960, 0.000), 'VM-UNet': (0.912, 0.244),
               'Swin-UNETR': (0.824, 0.569)}

# Draft table labels (family suffix, matching ISBI_PAPER_DRAFT.md Table 1/3 rows).
TABLE_LABEL = {'ResNet50-UNet': 'ResNet50-UNet (CNN)', 'VM-UNet': 'VM-UNet (SSM)',
               'VM-UNet-TSA': 'VM-UNet-TSA', 'Swin-UNETR': 'Swin-UNETR (ViT)'}


def evaluate(model, loader, device, needs_sigmoid, mode='clean'):
    """Pooled + per-image metrics; mode 'blur' low-passes the input first (rho=0.25)."""
    per_dice, preds, gts, per_preds, per_gts = [], [], [], [], []
    with torch.no_grad():
        for img01, gt in loader:
            if mode == 'blur':
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
    return {'pooled': pooled, 'dice_mean': float(np.mean(per_dice)),
            'dice_std': float(np.std(per_dice)),
            'per_dice': [float(x) for x in per_dice],
            'per_preds': per_preds, 'per_gts': per_gts}


def boundary_metrics(per_preds, per_gts):
    """BF1 + HD95 per image via the fast EDT HD95 (empty preds -> BF1 0 / HD95 inf)."""
    bf1s, hd95s = [], []
    for p, g in zip(per_preds, per_gts):
        if p.max() == 0:
            bf1s.append(0.0); hd95s.append(float('inf'))
            continue
        bf1s.append(boundary_f1(p, g))
        hd95s.append(hd95_fast(p, g))
    fin = [h for h in hd95s if np.isfinite(h)]
    return {'bf1_mean': float(np.nanmean(bf1s)),
            'hd95_mean': float(np.mean(fin)) if fin else None,
            'hd95_defined_n': int(len(fin)),
            'hd95_undefined_n': int(len(hd95s) - len(fin))}


def smoke_test(names, limit, device):
    """Guardrail: prove clean != feat-LP and that hooks are fully removed."""
    ds = CVCListDataset(names[:limit], CVC_IMG_DIR, CVC_MASK_DIR, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print(f'\n===== SMOKE TEST: first {limit} images =====')
    ok_all = True
    for name, family, arch, ckpt, needs_sigmoid in CKPTS:
        model = build_model(arch, ckpt, device)
        clean1 = evaluate(model, loader, device, needs_sigmoid, 'clean')
        handles = attach_lp(model, arch,
                            FrequencyIntervention(
                                lambda h, w, dev, dt: lowpass_mask(
                                    h, w, CUTOFF, device=dev, dtype=dt),
                                check_nan=True))
        feat = evaluate(model, loader, device, needs_sigmoid, 'clean')
        for h in handles:
            h.remove()
        clean2 = evaluate(model, loader, device, needs_sigmoid, 'clean')  # leak probe
        c1, f = np.array(clean1['per_dice']), np.array(feat['per_dice'])
        c2 = np.array(clean2['per_dice'])
        differ = int((np.abs(c1 - f) > 1e-12).sum())
        leak = int((np.abs(c1 - c2) > 1e-12).sum())
        print(f'  {name:<15s} clean vs feat: {differ}/{limit} images differ '
              f'(max |d| = {np.abs(c1 - f).max():.4f}) | clean rerun leak: {leak}/'
              f'{limit} images differ')
        if differ == 0 or leak > 0:
            ok_all = False
    print('SMOKE TEST:', 'PASS' if ok_all else 'FAIL - hook order/leak problem')
    if not ok_all:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_json', default=os.path.join(
        _REPO, 'interventions', 'results', 'cvc_heldout_full.json'))
    ap.add_argument('--limit', type=int, default=0,
                    help='smoke-test mode: only first N images, no JSON written')
    args, _ = ap.parse_known_args()

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    test_names = split['test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.limit:
        smoke_test(test_names, args.limit, device)
        return

    ds = CVCListDataset(test_names, CVC_IMG_DIR, CVC_MASK_DIR, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print('=' * 90)
    print('CVC held-out FULL protocol: dose-response + input-LP + boundary  |  @256')
    print('=' * 90)
    print(f'Device: {device} | split: held-out test ({len(ds)} images)')

    results = {'experiment':
               'CVC held-out full protocol (dose-response/input-LP/boundary, 62-image test)',
               'img_size': IMG_SIZE, 'num_images': len(ds), 'cutoff_set': FEAT_CUTOFFS,
               'split': 'held-out test (splits/cvc_split.json)',
               'device': str(device), 'timestamp': datetime.datetime.now().isoformat(),
               'models': []}

    for name, family, arch, ckpt, needs_sigmoid in CKPTS:
        print(f'\n===== {name} ({family}) =====')
        model = build_model(arch, ckpt, device)

        # 1) CLEAN pass BEFORE any hooks (guardrail).
        clean = evaluate(model, loader, device, needs_sigmoid, 'clean')
        cbm = boundary_metrics(clean['per_preds'], clean['per_gts'])
        entry = {'name': name, 'family': family, 'architecture': arch,
                 'checkpoint': ckpt,
                 'clean': {'dice': clean['pooled']['dice'], 'iou': clean['pooled']['iou'],
                           'dice_mean': clean['dice_mean'], 'dice_std': clean['dice_std'],
                           'boundary_f1': cbm['bf1_mean'], 'hd95': cbm['hd95_mean'],
                           'per_dice': clean['per_dice']},
                 'feature_dose': {}, 'input_dose': {}, 'boundary': {}}

        # 2) Feature-space LP dose-response (hooks attached per cutoff, removed after).
        print('  Feature-space LP:', end='')
        for c in FEAT_CUTOFFS:
            lp_fn = lambda h, w, dev, dt, c=c: lowpass_mask(h, w, c, device=dev, dtype=dt)
            handles = attach_lp(model, arch, FrequencyIntervention(lp_fn, check_nan=True))
            r = evaluate(model, loader, device, needs_sigmoid, 'clean')
            for h in handles:
                h.remove()
            entry['feature_dose'][str(c)] = {
                'dice': r['pooled']['dice'], 'dice_mean': r['dice_mean'],
                'dice_std': r['dice_std'], 'per_dice': r['per_dice']}
            if abs(c - 0.25) < 1e-9:
                bm = boundary_metrics(r['per_preds'], r['per_gts'])
                entry['boundary'] = {'dice': r['pooled']['dice'],
                                     'boundary_f1': bm['bf1_mean'],
                                     'hd95': bm['hd95_mean'],
                                     'hd95_defined_n': bm['hd95_defined_n'],
                                     'hd95_undefined_n': bm['hd95_undefined_n']}
            print(f' {c:.2f}:{r["pooled"]["dice"]:.3f}', end='', flush=True)
        print()

        # 3) Input-space LP at rho=0.25 (matches Table 3).
        inp = evaluate(model, loader, device, needs_sigmoid, 'blur')
        entry['input_dose']['0.25'] = {'dice': inp['pooled']['dice'],
                                       'dice_mean': inp['dice_mean'],
                                       'dice_std': inp['dice_std'],
                                       'per_dice': inp['per_dice']}
        dlt = (entry['feature_dose']['0.25']['dice'] - entry['clean']['dice']) \
            / entry['clean']['dice'] * 100
        print(f'  Clean: Dice={entry["clean"]["dice"]:.4f} '
              f'| Feat-LP0.25: {entry["feature_dose"]["0.25"]["dice"]:.4f} ({dlt:+.1f}%) '
              f'| Input-LP0.25: {inp["pooled"]["dice"]:.4f}')
        hd_c = f'{entry["clean"]["hd95"]:.1f}' if entry['clean'].get('hd95') is not None else 'n/a'
        hd_b = f'{entry["boundary"]["hd95"]:.1f}' if entry['boundary'].get('hd95') is not None else 'n/a'
        print(f'  Boundary BF1 clean->LP: {entry["clean"]["boundary_f1"]:.4f} -> '
              f'{entry["boundary"]["boundary_f1"]:.4f} | HD95 clean->LP: '
              f'{hd_c} -> {hd_b}')
        results['models'].append(entry)
        # Crash-resilience: persist completed models after each one.
        with open(args.output_json + '.partial', 'w') as f:
            json.dump(results, f, indent=2)

    # ---- Guardrail: rho=0.25 must reproduce Table-2 held-out numbers exactly. ----
    print('\n' + '=' * 90)
    print('GUARDRAIL: rho=0.25 row vs known Table-2 held-out numbers')
    print('=' * 90)
    gate_ok = True
    for name, (clean_exp, feat_exp) in T2_EXPECTED.items():
        m = next(x for x in results['models'] if x['name'] == name)
        c, f = m['clean']['dice'], m['feature_dose']['0.25']['dice']
        ok = abs(c - clean_exp) <= 0.0005 and abs(f - feat_exp) <= 0.0005
        gate_ok &= ok
        print(f'  {name:<15s} clean {c:.4f} (expect {clean_exp:.3f}) | '
              f'feat {f:.4f} (expect {feat_exp:.3f}) | {"OK" if ok else "MISMATCH"}')
    if not gate_ok:
        raise SystemExit('Table-2 reproduction FAILED - do not trust the sweep; '
                         'find the discrepancy before proceeding.')

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    if os.path.exists(args.output_json + '.partial'):
        os.remove(args.output_json + '.partial')
    print(f'\nSaved -> {args.output_json}')

    # ---- Ready-to-paste markdown tables (draft Table 1 & 3 column layout). ----
    print('\n' + '=' * 90)
    print('TABLE 1 (62-image held-out test) - paste into ISBI_PAPER_DRAFT.md')
    print('=' * 90)
    print('| Model | Clean | rho=0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for m in results['models']:
        row = f'| {TABLE_LABEL[m["name"]]} | {m["clean"]["dice"]:.3f}'
        for c in FEAT_CUTOFFS:
            row += f' | {m["feature_dose"][str(c)]["dice"]:.3f}'
        print(row + ' |')

    print('\n' + '=' * 90)
    print('TABLE 3 (62-image held-out test) - paste into ISBI_PAPER_DRAFT.md')
    print('=' * 90)
    print('| Model | Clean | Input-LP | Feat-LP | Delta% feat | BF1 clean->LP | HD95 clean->LP |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for m in results['models']:
        cl, lp, inp = m['clean'], m['feature_dose']['0.25'], m['input_dose']['0.25']['dice']
        bnd = m['boundary']
        dlt = (lp['dice'] - cl['dice']) / cl['dice'] * 100
        hd_lp = f'{bnd["hd95"]:.1f}' if bnd.get('hd95') is not None else '— (undefined)'
        hd_c = f'{cl["hd95"]:.1f}' if cl.get('hd95') is not None else 'n/a'
        print(f'| {TABLE_LABEL[m["name"]]} | {cl["dice"]:.3f} | {inp:.3f} | {lp["dice"]:.3f} '
              f'| {dlt:.1f}% | {cl["boundary_f1"]:.3f}->{bnd["boundary_f1"]:.3f} '
              f'| {hd_c}->{hd_lp} |')

    print('\n' + '=' * 90)
    print('TABLE 1 & 3 LaTeX tabular rows (62-image held-out test)')
    print('=' * 90)
    for m in results['models']:
        row = ' & '.join([TABLE_LABEL[m['name']], f"{m['clean']['dice']:.3f}"]
                         + [f"{m['feature_dose'][str(c)]['dice']:.3f}" for c in FEAT_CUTOFFS])
        print('T1  ' + row + ' \\\\')
    for m in results['models']:
        cl, lp, inp = m['clean'], m['feature_dose']['0.25'], m['input_dose']['0.25']['dice']
        bnd = m['boundary']
        dlt = (lp['dice'] - cl['dice']) / cl['dice'] * 100
        hd_lp = f"{bnd['hd95']:.1f}" if bnd.get('hd95') is not None else '--'
        hd_c = f"{cl['hd95']:.1f}" if cl.get('hd95') is not None else '--'
        print(f'T3  {TABLE_LABEL[m["name"]]} & {cl["dice"]:.3f} & {inp:.3f} & '
              f'{lp["dice"]:.3f} & {dlt:.1f} & {cl["boundary_f1"]:.3f} & '
              f'{bnd["boundary_f1"]:.3f} & {hd_c} & {hd_lp} \\\\')


if __name__ == '__main__':
    main()

