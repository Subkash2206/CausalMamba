"""
bootstrap_cvc_isic_ci.py — Two-sample bootstrap 95% CI on the cross-dataset
difference in per-image Delta-Dice (CVC minus ISIC) at rho=0.25, per matched
architecture.

Sources (per-image arrays):
  CVC : interventions/results/cvc_heldout_full.json (Task-1 output; 62-image
        held-out test split; clean.per_dice and feature_dose['0.25'].per_dice)
  ISIC: interventions/results/isic_heldout_eval.json (260-image held-out test
        split; per_image_dice.clean / per_image_dice.feature_lp)

For each matched leg (CNN / SSM / ViT):
  delta_CVC_i  = clean_i - feature_lp_i   (per image, rho=0.25)
  delta_ISIC_i = clean_i - feature_lp_i   (per image, rho=0.25)
  two-sample bootstrap (B=10000, seed 42): resample each dataset's per-image
  array INDEPENDENTLY with replacement, mean each resample, take the difference
  of the two means; 95% CI = [P2.5, P97.5] of that difference distribution.
  CVC and ISIC are independent, non-paired image sets — never paired index-wise.

Prints the ready-to-paste sentence:
  CNN [lo, hi], SSM [lo, hi], ViT [lo, hi]
(all values 4 decimal places), plus per-leg context and a zero-exclusion flag.

Usage:
    python interventions/experiments/bootstrap_cvc_isic_ci.py
"""

import os, json, sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
R = os.path.join(_REPO, 'interventions', 'results')

CVC_FULL = os.path.join(R, 'cvc_heldout_full.json')
ISIC_HO = os.path.join(R, 'isic_heldout_eval.json')

SEED = 42
B = 10000

# Matched legs: CVC name in cvc_heldout_full.json <-> ISIC name in isic_heldout_eval.json
LEGS = [
    ('CNN', 'ResNet50-UNet', 'ResNet50-UNet (CVC recipe, standardized)'),
    ('SSM', 'VM-UNet', 'VM-UNet (repo-impl SSM)'),
    ('ViT', 'Swin-UNETR', 'Swin-UNETR (CVC recipe, ViT leg closure)'),
]


def load(p):
    return json.load(open(p))


def find(d, prefix):
    for m in d['models']:
        if m['name'] == prefix or m['name'].startswith(prefix):
            return m
    raise KeyError(prefix)


def two_sample_ci(a, b, B=B, seed=SEED):
    """Independent resampling with replacement; CI on (mean(a*) - mean(b*))."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diffs = np.empty(B)
    for k in range(B):
        ma = rng.choice(a, size=len(a), replace=True).mean()
        mb = rng.choice(b, size=len(b), replace=True).mean()
        diffs[k] = ma - mb
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    cvc = load(CVC_FULL)
    isic = load(ISIC_HO)
    print('=' * 92)
    print('Two-sample bootstrap CIs on cross-dataset Delta-Dice difference '
          '(CVC minus ISIC) at rho=0.25 | B=%d, seed %d' % (B, SEED))
    print('=' * 92)

    results = []
    for label, cvc_name, isic_name in LEGS:
        c = find(cvc, cvc_name)
        i = find(isic, isic_name)
        d_cvc = np.array(c['clean']['per_dice']) - np.array(c['feature_dose']['0.25']['per_dice'])
        d_isic = np.array(i['per_image_dice']['clean']) - np.array(i['per_image_dice']['feature_lp'])
        lo, hi = two_sample_ci(d_cvc, d_isic)
        excludes_zero = lo > 0 or hi < 0
        results.append((label, lo, hi, d_cvc, d_isic, excludes_zero))
        print('  %-3s | CVC n=%d meanDelta=%.4f | ISIC n=%d meanDelta=%.4f | '
              'diff-of-means 95%%CI=[%.4f, %.4f] | excludes 0: %s'
              % (label, len(d_cvc), d_cvc.mean(), len(d_isic), d_isic.mean(),
                 lo, hi, excludes_zero))

    print('\nREADY-TO-PASTE:')
    print('  ' + ', '.join('%s [%.4f, %.4f]' % (l, lo, hi)
                           for l, lo, hi, *_ in results))
    if all(z for _, _, _, _, _, z in results):
        print('\nAll three CIs exclude zero - the dataset-dependent claim is '
              'statistically resolvable (CVC Delta-Dice > ISIC Delta-Dice).')
    else:
        print('\nWARNING: at least one CI includes zero - the dataset-dependent '
              'claim needs its wording walked back for that leg.')


if __name__ == '__main__':
    main()
