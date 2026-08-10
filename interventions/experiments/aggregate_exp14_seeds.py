"""
Phase 3.1: Aggregate Experiment 14 (VM-UNet CVC whole-network causal) results
across seeds 42, 123, 2027 and report Mean ± Std for Baseline and Low-pass.

Reads metrics_*.csv from each run directory (chooses the most recent file if a
directory contains several), then computes per-metric mean/std over seeds for
Dice, IoU, Sensitivity, Specificity (accuracy is also loaded for completeness).
"""
import csv
import glob
import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS = os.path.join(_REPO, 'interventions', 'results')

RUNS = [
    ('Seed 42',   'experiment14_vmunet_cvc_causal'),
    ('Seed 123',  'experiment14_vmunet_cvc_causal_seed123'),
    ('Seed 2027', 'experiment14_vmunet_cvc_causal_seed2027'),
]

METRICS = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
CONDITIONS = ['Baseline', 'LowPass']


def load_latest_metrics(run_dir):
    """Return dict[metric] -> {'Baseline': float, 'LowPass': float} for the newest CSV."""
    csvs = sorted(glob.glob(os.path.join(run_dir, 'metrics_*.csv')))
    if not csvs:
        raise FileNotFoundError(f'No metrics_*.csv found in {run_dir}')
    path = csvs[-1]  # most recent timestamp sorts last lexicographically
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            m = row['Metric'].strip().lower()
            if m in METRICS:
                out[m] = {
                    'Baseline': float(row['Baseline']),
                    'LowPass': float(row['LowPass']),
                }
    return out


def main():
    print('=' * 88)
    print('Experiment 14 aggregation: VM-UNet CVC-ClinicDB whole-network causal (cutoff=0.25)')
    print('=' * 88)

    per_seed = {}   # seed_label -> dict[metric] -> dict[condition] -> float
    for label, sub in RUNS:
        run_dir = os.path.join(RESULTS, sub)
        if not os.path.isdir(run_dir):
            print(f'[MISSING] {label:<8} {sub}  -> directory not found')
            continue
        data = load_latest_metrics(run_dir)
        per_seed[label] = data
        print(f'[OK]      {label:<8} {sub}  -> {sorted(data.keys())}')

    if len(per_seed) < 3:
        print(f'\nWARNING: only {len(per_seed)} of 3 runs available; Mean±Std will use {len(per_seed)} seeds.')

    # ------------------------------------------------------------------
    # Per-seed table
    # ------------------------------------------------------------------
    print('\n--- Per-seed metrics ---')
    hdr = f'{"Seed":<9}' + ''.join(f'{" " + m:>16}' for m in METRICS)
    print(hdr)
    print('-' * len(hdr))
    for label in per_seed:
        row = f'{label:<9}'
        for m in METRICS:
            v = per_seed[label].get(m, {}).get('Baseline')
            row += f'{"B: " + format(v, ".4f") if v is not None else "B: ----":>16}'
        print(row)
        row = f'{"":<9}'
        for m in METRICS:
            v = per_seed[label].get(m, {}).get('LowPass')
            row += f'{"L: " + format(v, ".4f") if v is not None else "L: ----":>16}'
        print(row)

    # ------------------------------------------------------------------
    # Mean +/- Std across seeds (for the manuscript)
    # ------------------------------------------------------------------
    print('\n--- Mean ± Std across seeds ---')
    stats = {}
    for cond in CONDITIONS:
        stats[cond] = {}
        for m in METRICS:
            vals = [per_seed[s][m][cond] for s in per_seed if m in per_seed[s] and cond in per_seed[s][m]]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)      # population std (n = seeds)
                std = var ** 0.5
                stats[cond][m] = (mean, std, len(vals))

    for cond in CONDITIONS:
        print(f'\n{cond}:')
        for m in METRICS:
            if m in stats[cond]:
                mean, std, n = stats[cond][m]
                print(f'  {m:<13} {mean:.4f} ± {std:.4f}   (n={n})')

    # ------------------------------------------------------------------
    # Manuscript-ready lines
    # ------------------------------------------------------------------
    print('\n--- Manuscript-ready (Mean ± Std, 3 seeds) ---')
    for m in ['dice', 'iou', 'sensitivity', 'specificity']:
        bm, bs, _ = stats['Baseline'][m]
        lm, ls, _ = stats['LowPass'][m]
        print(f'{m.capitalize()}: Baseline {bm:.4f} ± {bs:.4f}  |  Low-pass {lm:.4f} ± {ls:.4f}')

    print('\n--- Delta (Low-pass - Baseline) on mean values ---')
    for m in ['dice', 'iou', 'sensitivity', 'specificity']:
        bm, _, _ = stats['Baseline'][m]
        lm, _, _ = stats['LowPass'][m]
        delta = lm - bm
        pct = (delta / bm) * 100 if bm else float('nan')
        print(f'{m.capitalize()}: {delta:+.4f} ({pct:+.2f}%)')


if __name__ == '__main__':
    main()
