"""
report_split_and_ci.py — Consolidates the split documentation + per-image CIs into
a manuscript-ready markdown report (CPU-only; reads the eval JSON artifacts).

Documents:
  - deterministic train/val split (seed 42, 80/20) for CVC (489/123) and ISIC (50-val)
  - pooled Dice vs per-image mean +/- SD
  - bootstrap 95% CI of the mean per-image Dice
  - headline feature-space + input-space benchmark tables (CVC + ISIC)

Output: interventions/results/robustness_report.md

Usage:
    python interventions/experiments/report_split_and_ci.py
"""

import sys, os, json

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(_REPO, 'interventions', 'results')


def boot_ci(vals, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals)
    means = [float(np.mean(rng.choice(v, size=len(v), replace=True))) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    md = []
    md.append('# Spectral Robustness — Consolidated Report\n')
    md.append('## Data splits (deterministic, documented for reproducibility)\n')
    md.append('- **CVC-ClinicDB**: `CVCDataset` 80/20 split via `train_test_split(test_size=0.2, '
              'random_state=42)` → **489 train / 123 val**. The 123-image val is used for '
              'best-val-loss model selection *and* all reported metrics (a mild selection-on-test '
              'caveat to state in the paper).\n')
    md.append('- **ISIC2018**: 50-image validation subset (first 50 sorted images of the 2,594 '
              'training images), matching the Phase-0 protocol (`experiment2`).\n')
    md.append('- All checkpoints loaded `strict=True`; all evals at 256×256 (Swin-UNet @224).\n')

    # ---------- CVC ----------
    p = os.path.join(RES, 'cvc_robustness_eval.json')
    if os.path.exists(p):
        j = json.load(open(p))
        md.append('\n## CVC-ClinicDB — per-image Dice (mean ± SD, bootstrap 95% CI)\n')
        md.append('| Model | Cond | Pooled | Per-img μ±σ | 95% CI |\n|---|---|---|---:|---:|')
        for m in j['models']:
            clean = m['clean']['per_dice']
            lo, hi = boot_ci(clean)
            md.append(f"| {m['name']} | Clean | {m['clean']['dice']:.4f} | "
                      f"{m['clean']['dice_mean']:.4f}±{m['clean']['dice_std']:.4f} | "
                      f"[{lo:.4f}, {hi:.4f}] |")
            for c in ['0.25']:
                fd = m['feature_dose'][c]['per_dice']
                lo, hi = boot_ci(fd)
                md.append(f"| {m['name']} | Feat-LP {c} | {m['feature_dose'][c]['dice']:.4f} | "
                          f"{m['feature_dose'][c]['dice_mean']:.4f}±{m['feature_dose'][c]['dice_std']:.4f} | "
                          f"[{lo:.4f}, {hi:.4f}] |")
                id_ = m['input_dose'][c]['per_dice']
                lo, hi = boot_ci(id_)
                md.append(f"| {m['name']} | Input-LP {c} | {m['input_dose'][c]['dice']:.4f} | "
                          f"{m['input_dose'][c]['dice_mean']:.4f}±{m['input_dose'][c]['dice_std']:.4f} | "
                          f"[{lo:.4f}, {hi:.4f}] |")
        md.append('\n### Feature-space LP dose-response (pooled Dice)\n')
        md.append('| Model | Clean | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |\n'
                  '|---|---:|---:|---:|---:|---:|---:|---:|')
        for m in j['models']:
            row = f"| {m['name']} | {m['clean']['dice']:.3f}"
            for c in ['0.1', '0.15', '0.2', '0.25', '0.3', '0.4']:
                row += f" | {m['feature_dose'][c]['dice']:.3f}"
            md.append(row + ' |')
        md.append('\n### Input-space degradations (pooled Dice)\n')
        md.append('| Model | Clean | LP.25 | G2 | G4 | G8 | J80 | J50 | J20 | M5 | M10 |\n'
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for m in j['models']:
            d = m['degradations']
            md.append(f"| {m['name']} | {m['clean']['dice']:.3f} | "
                      f"{m['input_dose']['0.25']['dice']:.3f} | {d['gauss_2']:.3f} | "
                      f"{d['gauss_4']:.3f} | {d['gauss_8']:.3f} | {d['jpeg_80']:.3f} | "
                      f"{d['jpeg_50']:.3f} | {d['jpeg_20']:.3f} | {d['motion_5']:.3f} | "
                      f"{d['motion_10']:.3f} |")
        md.append('\n### Boundary metrics (clean vs feature-LP 0.25)\n')
        md.append('| Model | Clean BF1 | Clean HD95 | LP BF1 | LP HD95 |\n|---|---:|---:|---:|---:|')
        for m in j['models']:
            b = m['boundary']
            md.append(f"| {m['name']} | {m['clean']['boundary_f1']:.3f} | "
                      f"{m['clean']['hd95']:.1f} | {b['boundary_f1']:.3f} | {b['hd95']:.1f} |")

    # ---------- ISIC ----------
    p = os.path.join(RES, 'isic_cross_arch_eval.json')
    if os.path.exists(p):
        j = json.load(open(p))
        md.append('\n## ISIC2018 — cross-architecture benchmark (feature-space vs input-space)\n')
        md.append('| Architecture | Family | Clean | Feat-LP | Δ% | Input-LP | Δ% |\n'
                  '|---|---|---:|---:|---:|---:|---:|')
        for m in j['models']:
            md.append(f"| {m['name']} | {m['family']} | {m['clean']['dice']:.4f} | "
                      f"{m['feature_lp']['dice']:.4f} | {m['delta_pct_feature']:+.1f}% | "
                      f"{m['input_blur']['dice']:.4f} | {m['delta_pct_input']:+.1f}% |")

    out_p = os.path.join(RES, 'robustness_report.md')
    with open(out_p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'Saved -> {out_p}')
    print('Report generated (see robustness_report.md for full tables).')


if __name__ == '__main__':
    main()
