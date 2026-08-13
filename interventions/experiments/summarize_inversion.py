"""
summarize_inversion.py — Assemble the cross-dataset inversion table (Table 2) from the
CVC robustness eval and the ISIC held-out eval.

    python interventions/experiments/summarize_inversion.py

Inputs:
  interventions/results/cross_arch_cvc_eval.json      (CVC, val 123, feature-LP 0.25)
  interventions/results/isic_heldout_eval.json        (ISIC, held-out test 260)
Outputs: console table + per-image Wilcoxon + prints the Table-2 markdown rows.
"""

import sys, os, json

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CVC_JSON = os.path.join(_REPO, 'interventions', 'results', 'cross_arch_cvc_eval.json')
ISIC_JSON = os.path.join(_REPO, 'interventions', 'results', 'isic_heldout_eval.json')


def wilcoxon(a, b):
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float('nan')
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float('nan')


def main():
    cvc = json.load(open(CVC_JSON))
    isic = json.load(open(ISIC_JSON))

    # Map model family -> CVC entry by name.
    cvc_by_family = {}
    for m in cvc['models']:
        fam = m['family']
        if fam not in cvc_by_family:
            cvc_by_family[fam] = m

    isic_by_family = {}
    for m in isic['models']:
        fam = m['family']
        isic_by_family.setdefault(fam, []).append(m)

    print('=' * 100)
    print(f"Cross-dataset inversion at feature-LP cutoff 0.25 | CVC n={cvc['num_images']} "
          f"| ISIC held-out n={isic['num_images']}")
    print('=' * 100)
    print(f"{'Architecture':<34}{'CVC clean':>10}{'CVC feat-LP':>12}{'CVC d%':>9}"
          f"{'ISIC clean':>11}{'ISIC feat-LP':>13}{'ISIC d%':>9}  leg")
    print('-' * 100)
    for fam, cvc_m in cvc_by_family.items():
        isic_rows = isic_by_family.get(fam, [])
        # pick the first non-standardized row as the primary ISIC entry
        isic_primary = isic_rows[0] if isic_rows else None
        cvc_d = (cvc_m['lowpass']['dice'] - cvc_m['clean']['dice']) / (cvc_m['clean']['dice'] + 1e-12) * 100
        if isic_primary:
            isic_d = (isic_primary['feature_lp']['dice'] - isic_primary['clean']['dice']) / \
                     (isic_primary['clean']['dice'] + 1e-12) * 100
            print(f"{cvc_m['name']:<34}{cvc_m['clean']['dice']:>10.3f}{cvc_m['lowpass']['dice']:>12.3f}"
                  f"{cvc_d:>+9.1f}{isic_primary['clean']['dice']:>11.3f}"
                  f"{isic_primary['feature_lp']['dice']:>13.3f}{isic_d:>+9.1f}  {isic_primary['name'][:30]}")
        else:
            print(f"{cvc_m['name']:<34}{cvc_m['clean']['dice']:>10.3f}{cvc_m['lowpass']['dice']:>12.3f}"
                  f"{cvc_d:>+9.1f}{'(no ISIC row)':>33}")

    print()
    print('Per-image Wilcoxon (clean vs feat-LP) — ISIC held-out:')
    for m in isic['models']:
        c = m['per_image_dice']['clean']; f = m['per_image_dice']['feature_lp']
        print(f"  {m['name'][:45]:<47} p={wilcoxon(c, f):.2e}  mean {np.mean(c):.4f}->{np.mean(f):.4f}")

    print()
    print('Table-2 markdown rows (CVC clean->LP / ISIC clean->LP / leg status):')
    for fam, cvc_m in cvc_by_family.items():
        isic_rows = isic_by_family.get(fam, [])
        isic_primary = isic_rows[0] if isic_rows else None
        cvc_d = (cvc_m['lowpass']['dice'] - cvc_m['clean']['dice']) / (cvc_m['clean']['dice'] + 1e-12) * 100
        cvc_str = f"{cvc_m['clean']['dice']:.3f} -> {cvc_m['lowpass']['dice']:.3f} ({cvc_d:+.1f}%)"
        if isic_primary:
            isic_d = (isic_primary['feature_lp']['dice'] - isic_primary['clean']['dice']) / \
                     (isic_primary['clean']['dice'] + 1e-12) * 100
            isic_str = f"{isic_primary['clean']['dice']:.3f} -> {isic_primary['feature_lp']['dice']:.3f} ({isic_d:+.1f}%)"
        else:
            isic_str = '—'
        print(f"| {cvc_m['name']} | {cvc_str} | {isic_str} |")


if __name__ == '__main__':
    main()
