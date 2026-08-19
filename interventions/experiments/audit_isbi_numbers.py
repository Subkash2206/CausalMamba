"""Audit ISBI_PAPER_DRAFT.md numbers vs source JSONs + bootstrap CIs on held-out evals."""
import os, re, json, sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
R = os.path.join(_REPO, 'interventions', 'results')
DRAFT = os.path.join(R, 'paper_v2', 'ISBI_PAPER_DRAFT.md')
CVC_ROB = os.path.join(R, 'cvc_robustness_eval.json')
CVC_HO = os.path.join(R, 'cvc_heldout_eval.json')
ISIC_HO = os.path.join(R, 'isic_heldout_eval.json')
CUTOFFS = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.4']
T1 = ['ResNet50-UNet', 'VM-UNet', 'VM-UNet-TSA', 'Swin-UNETR']


def load(p):
    return json.load(open(p))


def model(d, name):
    for m in d['models']:
        if m['name'].startswith(name):
            return m
    raise KeyError(name)


def boot_ci(delta, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(delta, float)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def norm(s):
    return s.replace('\u2212', '-').replace('\u2192', '->').replace('rho=', '').strip()


def numbers(s):
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', norm(s))]


def table_rows(draft, header):
    lines = draft.splitlines()
    out = {}
    active = False
    for ln in lines:
        if ln.strip().startswith('### '):
            active = header in ln
            continue
        if ln.startswith('|') and active:
            cells = [c.strip() for c in ln.strip('|').split('|')]
            if cells and cells[0]:
                out[cells[0]] = cells[1:]
    return out

def main():
    cvcr, cvch, isich = load(CVC_ROB), load(CVC_HO), load(ISIC_HO)
    draft = open(DRAFT, encoding='utf-8').read()

    print('=' * 92)
    print('PART 4 - 95% bootstrap CIs on mean per-image Delta-Dice @ rho=0.25 (B=10000, seed 42)')
    print('=' * 92)
    for tag, d in [('CVC held-out (62)', cvch), ('ISIC held-out (260)', isich)]:
        for m in d['models']:
            c = np.array(m['per_image_dice']['clean'])
            f = np.array(m['per_image_dice']['feature_lp'])
            delta = c - f
            lo, hi = boot_ci(delta)
            pct = (m['feature_lp']['dice'] - m['clean']['dice']) / m['clean']['dice'] * 100
            print('  %s | %-42s meanDelta=%+.4f 95%%CI=[%+.4f, %+.4f]  pooled=%+.1f%%' % (
                tag, m['name'][:42], delta.mean(), lo, hi, pct))

    print()
    print('=' * 92)
    print('PART 3 - numeric verification of draft tables vs source JSONs')
    print('=' * 92)
    n_ok = n_bad = 0
    def cmp(label, draft_val, json_val, tol):
        nonlocal n_ok, n_bad
        if draft_val is None or json_val is None:
            n_bad += 1
            print('  [MISS] %s | draft=%r json=%r' % (label, draft_val, json_val))
        elif abs(draft_val - json_val) <= tol:
            n_ok += 1
        else:
            n_bad += 1
            print('  [MISMATCH] %s | draft=%.4f json=%.4f' % (label, draft_val, json_val))

    # ---- Table 1 ----
    rows = table_rows(draft, 'Table 1')
    print('Table 1 (CVC dose-response):')
    for name in T1:
        m = model(cvcr, name)
        expected = [m['clean']['dice']] + [m['feature_dose'][c]['dice'] for c in CUTOFFS]
        drow = next((k, v) for k, v in rows.items() if k.startswith(name))
        dcells = [numbers(c) for c in drow[1]]
        dv = [c[0] if c else None for c in dcells]
        if len(dv) != len(expected):
            print('  [MISS] %s row cell count draft=%d json=%d' % (name, len(dv), len(expected)))
            n_bad += 1
            continue
        for i, (dcell, jcell) in enumerate(zip(dv, expected)):
            lbl = 'T1 %s[%s]' % (name, 'clean' if i == 0 else 'c' + CUTOFFS[i - 1])
            cmp(lbl, dcell, jcell, 0.0005)

    # ---- Table 2 ----
    rows2 = table_rows(draft, 'Table 2')
    legs = [
        ('ResNet50-UNet', lambda m: m['name'].startswith('ResNet50-UNet')
         and ('CVC recipe' in m['name'] or m['name'] == 'ResNet50-UNet')),
        ('VM-UNet', lambda m: m['name'].startswith('VM-UNet') and 'TSA' not in m['name']),
        ('Swin-UNETR', lambda m: m['name'].startswith('Swin-UNETR')),
    ]
    print('Table 2 (cross-dataset):')
    for label, guard in legs:
        c = next(m for m in cvch['models'] if guard(m))
        i = next(m for m in isich['models'] if guard(m))
        cd = (c['feature_lp']['dice'] - c['clean']['dice']) / c['clean']['dice'] * 100
        idp = (i['feature_lp']['dice'] - i['clean']['dice']) / i['clean']['dice'] * 100
        drow = next((k, v) for k, v in rows2.items() if k.startswith(label))
        cvc_n = numbers(drow[1][0])   # e.g. [0.960, 0.000, -100.0]
        isic_n = numbers(drow[1][1])
        cmp('T2 %s CVC clean' % label, cvc_n[0], c['clean']['dice'], 0.0005)
        cmp('T2 %s CVC LP' % label, cvc_n[1], c['feature_lp']['dice'], 0.0005)
        cmp('T2 %s CVC d%%' % label, cvc_n[2], cd, 0.05)
        cmp('T2 %s ISIC clean' % label, isic_n[0], i['clean']['dice'], 0.0005)
        cmp('T2 %s ISIC LP' % label, isic_n[1], i['feature_lp']['dice'], 0.0005)
        cmp('T2 %s ISIC d%%' % label, isic_n[2], idp, 0.05)

    # ---- Table 3 ----
    rows3 = table_rows(draft, 'Table 3')
    print('Table 3 (input vs feature + defense):')
    for name in T1:
        m = model(cvcr, name)
        cl, lp, inp = m['clean'], m['feature_dose']['0.25'], m['input_dose']['0.25']['dice']
        bnd = m.get('boundary', {})
        dlt = (lp['dice'] - cl['dice']) / cl['dice'] * 100
        drow = next((k, v) for k, v in rows3.items() if k.startswith(name))
        dv = [numbers(c) for c in drow[1]]
        # draft columns: name, clean, input, feat, d%, bf1_clean->lp, hd95_clean->lp
        seq = [('clean', 0.0005), ('input', 0.0005), ('feat', 0.0005), ('d%', 0.1)]
        for (l, tol), dc in zip(seq, dv):
            if dc:
                jc = {'clean': cl['dice'], 'input': inp, 'feat': lp['dice'], 'd%': dlt}[l]
                cmp('T3 %s %s' % (name, l), dc[0], jc, tol)
        # arrow pairs: BF1 clean->lp , HD95 clean->lp
        bf1_n, hd95_n = dv[4], dv[5]
        if len(bf1_n) >= 2:
            cmp('T3 %s BF1-clean' % name, bf1_n[0], cl.get('boundary_f1'), 0.0005)
            cmp('T3 %s BF1-LP' % name, bf1_n[1], bnd.get('boundary_f1'), 0.0005)
        if len(hd95_n) >= 2:
            cmp('T3 %s HD95-clean' % name, hd95_n[0], cl.get('hd95'), 0.05)
            cmp('T3 %s HD95-LP' % name, hd95_n[1], bnd.get('hd95'), 0.05)

    print()
    print('Verification: %d numeric cells OK, %d issues.' % (n_ok, n_bad))

    print()
    print('PART 3b - abstract claims present in draft:')
    for probe in ['-100%', '-73%', '-31%', '-9.4%', '-10.3%', '-0.6%', '86.5', '-66', '10.6', '7.1', '52']:
        found = probe in norm(draft)
        print('   %-8s in draft: %s' % (probe, found))


if __name__ == '__main__':
    main()

