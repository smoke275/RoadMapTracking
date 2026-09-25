"""
Plot the k-pursuer / m-evader results of run_multi_evader.py.

    python plot_multi_evader.py benchmark_results/<stamp>_multi_evader.json \
        --out RoadmapBasedTracking/figures/multi_evader.pdf

Three panels, one line per k, m on the x-axis: the oracle speed ratio
s*_{k,m}, the mean achieved team alpha in trials, and the mean fraction of
evaders in line of sight of at least one pursuer.
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOURS = {1: '#1f4e9c', 2: '#d9822b', 3: '#2e8b57', 4: '#8b2e8b'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json')
    ap.add_argument('--out', default='multi_evader.pdf')
    ap.add_argument('--polygon', default=None)
    args = ap.parse_args()
    res = json.load(open(args.json))
    poly = args.polygon or res['oracle'][0]['polygon']
    oracle = [r for r in res['oracle'] if r['polygon'] == poly]
    trials = [r for r in res['trials'] if r['polygon'] == poly]
    ks = sorted({r['k'] for r in oracle})
    ms = sorted({r['m'] for r in oracle})

    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7})
    fig, axes = plt.subplots(1, 3 if trials else 1, figsize=(7.0, 2.1), constrained_layout=True)
    axes = list(axes) if trials else [axes]

    ax = axes[0]
    for k in ks:
        ys = [next(r['max'] for r in oracle if r['k'] == k and r['m'] == m) for m in ms]
        ax.plot(ms, ys, 'o-', color=COLOURS.get(k, 'k'), label=f'$k={k}$', ms=4)
    ax.axhline(1.0, color='0.6', lw=0.8, ls='--')
    ax.set_xlabel('evaders $m$'); ax.set_ylabel('$s^\\star_{k,m}$')
    ax.set_title('oracle speed ratio', fontsize=8)
    ax.set_xticks(ms); ax.legend(frameon=False)

    if trials:
        ax = axes[1]
        for k in ks:
            rows = [next(r for r in trials if r['k'] == k and r['m'] == m) for m in ms]
            ax.errorbar(ms, [r['mean_alpha'][0] for r in rows], yerr=[r['mean_alpha'][1] for r in rows],
                        fmt='o-', color=COLOURS.get(k, 'k'), ms=4, capsize=2, label=f'$k={k}$')
        ax.axhline(1.0, color='0.6', lw=0.8, ls='--')
        ax.set_xlabel('evaders $m$'); ax.set_ylabel('mean achieved $\\alpha$')
        ax.set_title('trials: timing', fontsize=8); ax.set_xticks(ms)

        ax = axes[2]
        for k in ks:
            rows = [next(r for r in trials if r['k'] == k and r['m'] == m) for m in ms]
            ax.errorbar(ms, [r['in_view_pct'][0] for r in rows], yerr=[r['in_view_pct'][1] for r in rows],
                        fmt='o-', color=COLOURS.get(k, 'k'), ms=4, capsize=2, label=f'$k={k}$')
        ax.set_xlabel('evaders $m$'); ax.set_ylabel('evaders in view (%)')
        ax.set_title('trials: line of sight', fontsize=8); ax.set_xticks(ms); ax.set_ylim(0, 100)

    fig.savefig(args.out)
    print(f'[SAVED] {args.out}')


if __name__ == '__main__':
    main()
