"""
compute_robustness_stats.py — Paired statistics for the low-pass robustness results.

Reads per-image Dice from the JSON artifacts and computes, per model and per
intervention (feature-space LP 0.25 / input-space blur 0.25):
  - paired Wilcoxon signed-rank test (clean vs intervened)
  - mean +/- SD of the per-image delta
  - bootstrapped 95% CI of the mean delta (10k resamples)

Sources:
  - CVC : interventions/results/cvc_robustness_eval.json
  - ISIC: interventions/results/isic_cross_arch_eval.json

Usage:
    python interventions/experiments/compute_robustness_stats.py
"""

import sys, os, json

import numpy as np
from scipy import stats

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(_REPO, 'interventions', 'results')


def bootstrap_ci(deltas, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(rng.choice(deltas, size=len(deltas), replace=True)))
            for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return lo, hi


def analyze(clean, inter):
    d = np.asarray(clean) - np.asarray(inter)   # positive = intervention hurt
    if len(d) < 5 or np.std(d) == 0:
        return None
    w = stats.wilcoxon(d)                        # H0: median delta = 0
    lo, hi = bootstrap_ci(d)
    return {'mean_delta': float(np.mean(d)), 'std_delta': float(np.std(d)),
            'wilcoxon_stat': float(w.statistic), 'p_value': float(w.pvalue),
            'ci95': [float(lo), float(hi)]}


def main():
    out = {'experiment': 'Paired robustness statistics (per-image Dice)',
           'models': []}

    # ---------- CVC ----------
    cvc_path = os.path.join(RES, 'cvc_robustness_eval.json')
    if os.path.exists(cvc_path):
        with open(cvc_path) as f:
            j = json.load(f)
        for m in j['models']:
            clean = m['clean']['per_dice']
            f_lp = m['feature_dose']['0.25']['per_dice']
            i_lp = m['input_dose']['0.25']['per_dice']
            row = {'dataset': 'CVC', 'model': m['name'],
                   'feature_lp': analyze(clean, f_lp),
                   'input_blur': analyze(clean, i_lp)}
            out['models'].append(row)
            print(f"CVC {m['name']}: "
                  f"feat-LP p={row['feature_lp']['p_value']:.2e} "
                  f"d={row['feature_lp']['mean_delta']:+.4f}±{row['feature_lp']['std_delta']:.4f} "
                  f"| input p={row['input_blur']['p_value']:.2e} "
                  f"d={row['input_blur']['mean_delta']:+.4f}")

    # ---------- ISIC ----------
    isic_path = os.path.join(RES, 'isic_cross_arch_eval.json')
    if os.path.exists(isic_path):
        with open(isic_path) as f:
            j = json.load(f)
        for m in j['models']:
            clean = m['per_image_dice']['clean']
            f_lp = m['per_image_dice']['feature_lp']
            i_lp = m['per_image_dice']['input_blur']
            row = {'dataset': 'ISIC', 'model': m['name'],
                   'feature_lp': analyze(clean, f_lp),
                   'input_blur': analyze(clean, i_lp)}
            out['models'].append(row)
            print(f"ISIC {m['name']}: "
                  f"feat-LP p={row['feature_lp']['p_value']:.2e} "
                  f"d={row['feature_lp']['mean_delta']:+.4f}±{row['feature_lp']['std_delta']:.4f} "
                  f"| input p={row['input_blur']['p_value']:.2e} "
                  f"d={row['input_blur']['mean_delta']:+.4f}")

    with open(os.path.join(RES, 'robustness_stats.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved -> {os.path.join(RES, "robustness_stats.json")}')


if __name__ == '__main__':
    main()
