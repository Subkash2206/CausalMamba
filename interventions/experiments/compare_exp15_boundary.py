"""Phase 3.3: Side-by-side comparison of Experiment 15 boundary error rates
(baseline vs intervened), parsed from the two boundary_summary_*.csv files."""
import csv
import glob
import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS = os.path.join(_REPO, 'interventions', 'results')

RUNS = [
    ('Baseline',   'experiment15_vmunet_cvc_boundary_baseline'),
    ('Intervened', 'experiment15_vmunet_cvc_boundary_intervened'),
]

REGIONS = ['boundary_5px', 'boundary_10px', 'boundary_20px', 'interior', 'background']
LABELS = {
    'boundary_5px': 'Boundary (±5 px)',
    'boundary_10px': 'Boundary (±10 px)',
    'boundary_20px': 'Boundary (±20 px)',
    'interior': 'Interior (foreground)',
    'background': 'Background (far)',
}


def load_latest_summary(run_dir):
    csvs = sorted(glob.glob(os.path.join(run_dir, 'boundary_summary_*.csv')))
    if not csvs:
        raise FileNotFoundError(f'No boundary_summary_*.csv in {run_dir}')
    out = {}
    with open(csvs[-1], newline='') as f:
        for row in csv.DictReader(f):
            out[row['region'].strip()] = float(row['error_rate_pct'])
    return out


def main():
    data = {}
    for label, sub in RUNS:
        run_dir = os.path.join(RESULTS, sub)
        if not os.path.isdir(run_dir):
            print(f'[MISSING] {label}: {run_dir}')
            continue
        data[label] = load_latest_summary(run_dir)
        print(f'[OK] {label:<11} {sub}')

    if len(data) < 2:
        print('\nERROR: need both runs to compare.')
        return

    bl = data['Baseline']
    iv = data['Intervened']

    print('\n' + '=' * 78)
    print('Experiment 15 — VM-UNet CVC: Boundary-vs-Interior Error Rate (%)')
    print('=' * 78)
    hdr = f'{"Region":<22}{"Baseline":>12}{"Intervened":>14}{"Delta":>12}{"Ratio":>9}'
    print(hdr)
    print('-' * len(hdr))
    for r in REGIONS:
        b, i = bl[r], iv[r]
        print(f'{LABELS[r]:<22}{b:>12.2f}{i:>14.2f}{i - b:>+12.2f}{(i / b if b else float("nan")):>8.2f}x')
    print('-' * len(hdr))
    print(f'{"Boundary/Interior ratio":<22}{bl["boundary_interior_ratio"]:>12.2f}'
          f'{iv["boundary_interior_ratio"]:>14.2f}'
          f'{iv["boundary_interior_ratio"] - bl["boundary_interior_ratio"]:>+12.2f}')

    print('\nKey causal takeaway (boundary vs interior, ±5px band):')
    b_delta = iv['boundary_5px'] - bl['boundary_5px']
    i_delta = iv['interior'] - bl['interior']
    print(f'  Boundary error: {bl["boundary_5px"]:.2f}% -> {iv["boundary_5px"]:.2f}%  (Δ {b_delta:+.2f} pts)')
    print(f'  Interior error: {bl["interior"]:.2f}% -> {iv["interior"]:.2f}%  (Δ {i_delta:+.2f} pts)')
    print(f'  Boundary-to-Interior ratio: {bl["boundary_interior_ratio"]:.2f}x -> {iv["boundary_interior_ratio"]:.2f}x')


if __name__ == '__main__':
    main()
