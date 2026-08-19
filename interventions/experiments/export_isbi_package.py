"""
export_isbi_package.py — Generate the ISBI paper's tables (CSV) and figures (300 dpi)
straight from the result JSONs (no hardcoded numbers).

Outputs to interventions/results/paper_v2/{tables,figures}/

    python interventions/experiments/export_isbi_package.py          # everything
    python interventions/experiments/export_isbi_package.py --fig2   # Fig. 2 only
"""

import os, sys, csv, json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
R = os.path.join(_REPO, 'interventions', 'results')
OUT_T = os.path.join(R, 'paper_v2', 'tables')
OUT_F = os.path.join(R, 'paper_v2', 'figures')
os.makedirs(OUT_T, exist_ok=True)
os.makedirs(OUT_F, exist_ok=True)

CVC_ROB = os.path.join(R, 'cvc_heldout_full.json')  # 62-image held-out test (Tables 1 & 3)
CVC_HO = os.path.join(R, 'cvc_heldout_eval.json')           # 62-test (Table 2)
ISIC_HO = os.path.join(R, 'isic_heldout_eval.json')         # 260-test (Table 2)

CUTOFFS = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.4']
T1_ORDER = ['ResNet50-UNet', 'VM-UNet', 'VM-UNet-TSA', 'Swin-UNETR']


def _load(p):
    with open(p) as f:
        return json.load(f)


def _model(d, name):
    for m in d['models']:
        if m['name'].startswith(name):
            return m
    raise KeyError(name)


def export_table1():
    d = _load(CVC_ROB)
    rows = [['Model', 'Clean'] + [f'rho={c}' for c in CUTOFFS]]
    for name in T1_ORDER:
        m = _model(d, name)
        row = [name, round(m['clean']['dice'], 3)]
        for c in CUTOFFS:
            row.append(round(m['feature_dose'][c]['dice'], 3))
        rows.append(row)
    p = os.path.join(OUT_T, 'table1_cvc_dose_response.csv')
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print('wrote', p)


def export_table2():
    cvc = _load(CVC_HO)
    isic = _load(ISIC_HO)

    def matched(d, family_guard):
        """Pick the recipe-matched row (CVC-recipe CNN, Swin-UNETR) over legacy ones."""
        for m in d['models']:
            if family_guard(m):
                return m
        return None

    rows = [['Architecture', 'CVC held-out clean', 'CVC held-out feat-LP', 'CVC d%',
             'ISIC held-out clean', 'ISIC held-out feat-LP', 'ISIC d%', 'Leg status']]
    legs = [
        ('ResNet50-UNet (CNN)',
         lambda m: m['name'].startswith('ResNet50-UNet')
         and ('CVC recipe' in m['name'] or m['name'] == 'ResNet50-UNet'),
         'matched recipe'),
        ('VM-UNet (SSM)', lambda m: m['name'].startswith('VM-UNet') and 'TSA' not in m['name'],
         'CVC canonical; ISIC legacy VSSM'),
        ('Swin-UNETR (ViT)', lambda m: m['name'].startswith('Swin-UNETR'),
         'matched recipe'),
    ]
    for label, guard, status in legs:
        c = matched(cvc, guard)
        i = matched(isic, guard)
        cd = (c['feature_lp']['dice'] - c['clean']['dice']) / c['clean']['dice'] * 100
        idp = (i['feature_lp']['dice'] - i['clean']['dice']) / i['clean']['dice'] * 100
        rows.append([label, round(c['clean']['dice'], 3), round(c['feature_lp']['dice'], 3),
                     round(cd, 1), round(i['clean']['dice'], 3),
                     round(i['feature_lp']['dice'], 3), round(idp, 1), status])
    p = os.path.join(OUT_T, 'table2_cross_dataset_fragility.csv')
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print('wrote', p)


def export_table3():
    d = _load(CVC_ROB)
    rows = [['Model', 'Clean', 'Input-LP', 'Feat-LP', 'd% feat',
             'BF1 clean', 'BF1 feat-LP', 'HD95 clean', 'HD95 feat-LP']]
    for name in T1_ORDER:
        m = _model(d, name)
        cl = m['clean']
        lp = m['feature_dose']['0.25']
        inp = m['input_dose']['0.25']['dice']
        fp = lp['dice']
        dlt = (fp - cl['dice']) / cl['dice'] * 100
        bnd = m.get('boundary', {})                      # feature-LP boundary metrics
        bnd_lp = bnd.get('boundary_f1')
        hd_lp = bnd.get('hd95')
        rows.append([name, round(cl['dice'], 3), round(inp, 3), round(fp, 3), round(dlt, 1),
                     round(cl.get('boundary_f1'), 3) if cl.get('boundary_f1') is not None else 'nan',
                     round(bnd_lp, 3) if bnd_lp is not None else 'nan',
                     round(cl.get('hd95'), 1) if cl.get('hd95') is not None else 'nan',
                     round(hd_lp, 1) if hd_lp is not None else 'nan'])
    p = os.path.join(OUT_T, 'table3_input_feature_defense.csv')
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print('wrote', p)


def fig1_dose_response():
    d = _load(CVC_ROB)
    fig, ax = plt.subplots(figsize=(3.6, 2.7), dpi=300)
    colors = {'ResNet50-UNet': '#d62728', 'VM-UNet': '#1f77b4',
              'VM-UNet-TSA': '#2ca02c', 'Swin-UNETR': '#9467bd'}
    xs = [0.0] + [float(c) for c in CUTOFFS]
    for name in T1_ORDER:
        m = _model(d, name)
        ys = [m['clean']['dice']] + [m['feature_dose'][c]['dice'] for c in CUTOFFS]
        ax.plot(xs, ys, '-o', ms=3, lw=1.4, label=name, color=colors[name])
    ax.axvline(0.25, ls='--', color='grey', lw=0.8)
    ax.text(0.25, 0.02, 'rho=0.25', ha='center', fontsize=6, color='grey')
    ax.set_xlabel('Feature-space low-pass cutoff  $\\rho$')
    ax.set_ylabel('Pooled Dice')
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc='center left')
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25)
    p = os.path.join(OUT_F, 'fig1_cvc_dose_response.png')
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)

def fig2_cross_dataset_heatmap():
    cvc = _load(CVC_HO)
    isic = _load(ISIC_HO)

    def matched(d, guard):
        for m in d['models']:
            if guard(m):
                return m
        return None

    archs = [
        ('CNN  (ResNet50-UNet)',
         lambda m: m['name'].startswith('ResNet50-UNet')
         and ('CVC recipe' in m['name'] or m['name'] == 'ResNet50-UNet')),
        ('SSM  (VM-UNet)', lambda m: m['name'].startswith('VM-UNet') and 'TSA' not in m['name']),
        ('ViT  (Swin-UNETR)', lambda m: m['name'].startswith('Swin-UNETR')),
    ]
    labels = ['CVC (n=62)', 'ISIC (n=260)']
    vals = np.zeros((3, 2))
    for i, (_, guard) in enumerate(archs):
        c, j = matched(cvc, guard), matched(isic, guard)
        vals[i, 0] = (c['feature_lp']['dice'] - c['clean']['dice']) / c['clean']['dice'] * 100
        vals[i, 1] = (j['feature_lp']['dice'] - j['clean']['dice']) / j['clean']['dice'] * 100

    fig, ax = plt.subplots(figsize=(3.6, 2.4), dpi=300)
    im = ax.imshow(vals, cmap='Reds', vmin=-100, vmax=0, aspect='auto')
    ax.set_xticks(range(2)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels([a[0] for a in archs], fontsize=7)
    ax.tick_params(length=0)
    # Pick the annotation color from each cell's actual luminance so labels stay
    # visible on both light ('Reds' low-end, e.g. the -100% cell) and dark cells.
    norm = plt.Normalize(vmin=-100, vmax=0)
    cell_rgba = im.cmap(norm(vals))                     # (3, 2, 4)
    lum = (0.299 * cell_rgba[..., 0] + 0.587 * cell_rgba[..., 1]
           + 0.114 * cell_rgba[..., 2])
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f'{vals[i, j]:.0f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if lum[i, j] < 0.6 else 'black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Feature-LP $\\Delta$ Dice (%)', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.set_title('Feature-spectral fragility at $\\rho=0.25$', fontsize=8)
    p = os.path.join(OUT_F, 'fig2_cross_dataset_fragility.png')
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)


def main():
    # --fig2: regenerate only the cross-dataset fragility heatmap (Fig. 2), e.g.
    # after a styling tweak, without touching tables / fig1.
    if len(sys.argv) > 1 and sys.argv[1] == '--fig2':
        fig2_cross_dataset_heatmap()
        return
    export_table1()
    export_table2()
    export_table3()
    fig1_dose_response()
    fig2_cross_dataset_heatmap()
    print('Done. Tables ->', OUT_T, '| Figures ->', OUT_F)


if __name__ == '__main__':
    main()

