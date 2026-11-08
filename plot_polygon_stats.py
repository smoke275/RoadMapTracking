"""Effect of the polygon descriptors on tracking performance.

Reads the latest benchmark_results/*_polygon_stats.csv (run_polygon_stats.py)
and plots every descriptor against the per-polygon tracking metrics of the
paper for the proposed Min-Max +Vis strategy (Table I: mean worst-corner
ratio and %LOS against the skeleton evader; adversarial table: %LOS and
escapes per trial) and the oracle bound s*.  One point per polygon, Spearman
rank correlation in each panel, plus a summary heat map of the correlations.

Run:  python plot_polygon_stats.py [--out benchmark_results/polygon_stats]
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_SCRIPT_DIR, 'benchmark_results')

# ---- paper numbers, Min-Max +Vis, sections/07_experiments.tex ---------------
POLYS = ['poly1', 'poly2', 'poly3', 'poly4', 'poly5', 'poly6', 'poly7', 'poly8', 'poly9']
METRICS = {
    # Table I (skeleton evader, 50 seeds, s_p/s_e = 0.8)
    'ᾱ (skeleton)':      [2.48, 1.39, 1.56, 1.59, 2.09, 3.07, 1.73, 4.74, 1.52],
    '%LOS (skeleton)':   [87.9, 98.8, 90.8, 90.0, 93.7, 84.3, 86.2, 65.6, 87.5],
    # adversarial table (+V columns)
    '%LOS (adversarial)': [85.0, 86.6, 96.9, 93.9, 90.1, 92.6, 89.5, 93.8, 80.8],
    'N_esc (adversarial)': [0.64, 0.48, 0.12, 0.24, 0.30, 0.18, 0.46, 0.30, 0.56],
}
DESCRIPTORS = [            # csv column, axis label
    ('reflex',        'reflex vertices r'),
    ('essential',     'essential cuts m'),
    ('link_diameter', 'link diameter'),
    ('mean_vis',      'mean visibility fraction'),
    ('watchman_norm', 'watchman route L_w / √A'),
    ('ker_norm',      'KER roadmap L / √A'),
    ('passage_norm',  'narrowest passage / √A'),
    ('sstar_ker',     'critical speed s* (KER)'),
]

INK, MUTED, GRID = '0.15', '0.45', '0.88'
POINT = '#1f77b4'          # one series, one hue; polygons are direct-labelled


def load_stats(path=None) -> dict:
    path = path or sorted(glob.glob(os.path.join(RESULTS, '*_polygon_stats.csv')))[-1]
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r['polygon']] = r
    return rows


def col(rows, key):
    return np.array([float(rows[p][key]) if rows[p][key] not in ('', 'None') else np.nan
                     for p in POLYS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None, help='polygon_stats.csv (default: latest)')
    ap.add_argument('--out', default=os.path.join(RESULTS, 'polygon_stats'),
                    help='output prefix; writes <out>_scatter.pdf/.png and <out>_corr.pdf/.png')
    args = ap.parse_args()
    rows = load_stats(args.csv)
    metrics = dict(METRICS)
    metrics['s* (KER roadmap)'] = list(col(rows, 'sstar_ker'))
    descriptors = [d for d in DESCRIPTORS if d[0] != 'sstar_ker'] + [('sstar_ker', 'critical speed s* (KER)')]

    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})

    # ---- scatter grid ---------------------------------------------------------
    nr, nc = len(metrics), len(descriptors)
    fig, axes = plt.subplots(nr, nc, figsize=(1.75 * nc, 1.6 * nr), squeeze=False)
    rho = np.full((nr, nc), np.nan)
    for i, (mname, mvals) in enumerate(metrics.items()):
        y = np.array(mvals, dtype=float)
        for j, (key, label) in enumerate(descriptors):
            ax = axes[i, j]
            x = col(rows, key)
            if key == 'sstar_ker' and mname == 's* (KER roadmap)':
                ax.set_visible(False)
                continue
            ok = np.isfinite(x) & np.isfinite(y)
            r_, p_ = spearmanr(x[ok], y[ok])
            rho[i, j] = r_
            ax.scatter(x[ok], y[ok], s=22, color=POINT, edgecolor='white', linewidth=0.6, zorder=3)
            for p, xx, yy in zip(POLYS, x, y):
                if np.isfinite(xx) and np.isfinite(yy):
                    ax.annotate(p[4:], (xx, yy), xytext=(3, 2), textcoords='offset points',
                                fontsize=6, color=MUTED)
            ax.text(0.03, 0.95, f'ρ = {r_:+.2f}' + (' *' if p_ < 0.05 else ''),
                    transform=ax.transAxes, ha='left', va='top', fontsize=7, color=INK)
            ax.grid(color=GRID, linewidth=0.5, zorder=0)
            ax.tick_params(length=2, labelsize=6)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if i == nr - 1 or (i == nr - 2 and key == 'sstar_ker'):
                ax.set_xlabel(label, fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(mname, fontsize=7)
            else:
                ax.set_yticklabels([])
    fig.suptitle('Polygon descriptors vs. tracking performance of Min-Max +Vis '
                 '(one point per benchmark polygon; Spearman ρ, * p < 0.05)', fontsize=8, y=1.0)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}_scatter.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---- correlation heat map: diverging, two hues + neutral midpoint ---------
    fig, ax = plt.subplots(figsize=(0.9 * nc + 1.6, 0.45 * nr + 1.2))
    im = ax.imshow(rho, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(nc)); ax.set_xticklabels([d[1] for d in descriptors], rotation=35, ha='right', fontsize=7)
    ax.set_yticks(range(nr)); ax.set_yticklabels(list(metrics), fontsize=7)
    for i in range(nr):
        for j in range(nc):
            if np.isfinite(rho[i, j]):
                ax.text(j, i, f'{rho[i, j]:+.2f}', ha='center', va='center', fontsize=6.5,
                        color='white' if abs(rho[i, j]) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Spearman ρ', fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_title('Rank correlation between polygon descriptors and tracking metrics (n = 9)', fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}_corr.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print('Spearman ρ (rows: metrics, cols: descriptors)')
    print(f'{"":22s}' + ''.join(f'{d[0][:12]:>13s}' for d in descriptors))
    for i, m in enumerate(metrics):
        print(f'{m:22s}' + ''.join(f'{rho[i, j]:+13.2f}' if np.isfinite(rho[i, j]) else f'{"":>13s}'
                                   for j in range(nc)))
    print(f'[SAVED] {args.out}_scatter.pdf/.png  {args.out}_corr.pdf/.png')


if __name__ == '__main__':
    main()
