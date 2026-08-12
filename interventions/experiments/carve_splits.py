"""
carve_splits.py — Fixed, reproducible held-out splits for ISIC2018 (and CVC).

Phase 2 of the spectral-vulnerability plan: create genuine held-out test splits
that were untouched during model selection, so the cross-dataset inversion can be
reported on an untouched test set.

ISIC: 2,594 images -> train 80% / val 10% / test 10%  (seed 42, deterministic).
CVC : 612 frames   -> train 80% / val 10% / test 10%  (from the CVCDataset pool).

Outputs JSONs to interventions/results/splits/ that the retrain scripts consume,
and prints the test-set filenames for the audit trail.

Usage:
    python interventions/experiments/carve_splits.py
"""

import sys, os, json, glob, random

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SPLIT_DIR = os.path.join(_REPO, 'interventions', 'results', 'splits')
os.makedirs(SPLIT_DIR, exist_ok=True)

ISIC_IMG = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'images')
ISIC_MASK = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'masks')
CVC_IMG = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
CVC_MASK = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')


def carve(name, names, seed=42):
    rng = random.Random(seed)
    idx = list(range(len(names)))
    rng.shuffle(idx)
    n = len(idx)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    split = {'train': [names[i] for i in idx[:n_tr]],
             'val': [names[i] for i in idx[n_tr:n_tr + n_va]],
             'test': [names[i] for i in idx[n_tr + n_va:]]}
    with open(os.path.join(SPLIT_DIR, f'{name}_split.json'), 'w') as f:
        json.dump(split, f, indent=2)
    print(f'{name}: train={len(split["train"])} val={len(split["val"])} test={len(split["test"])}')
    return split


def main():
    random.seed(42); np.random.seed(42)
    isic = sorted(glob.glob(os.path.join(ISIC_IMG, '*.jpg')))
    isic = [os.path.basename(p) for p in isic]
    s_isic = carve('isic', isic)

    cvc = sorted(os.listdir(CVC_IMG))
    cvc = [f for f in cvc if f.lower().endswith(('.tif', '.tiff'))]
    s_cvc = carve('cvc', cvc)

    print('\nISIC test (first 5):', s_isic['test'][:5])
    print('CVC  test (first 5):', s_cvc['test'][:5])
    print(f'\nSplits saved -> {SPLIT_DIR}')


if __name__ == '__main__':
    main()
