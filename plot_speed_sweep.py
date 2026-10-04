"""Speed-sweep figure (paper Fig. speed_sweep): tracking performance vs
s_p/s_e on poly9, with the critical ratios s* (k=1) and s*_k marked.

Panels:
  left    skeleton %LOS vs ratio (Min-Max, Min-Max +Vis)
  middle  adversarial escapes/trial vs ratio (Min-Max, Min-Max +Vis)
  right   team trials (k = 1,2,3 vs one evader): % in view vs ratio
          (only if --team-jsons are given)

Run:  python plot_speed_sweep.py --out RoadmapBasedTracking/figures/speed_sweep.pdf \
          [--team-jsons benchmark_results/..._multi_evader.json ...]
"""
import argparse
import csv
import glob
import json
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S_STAR = {1: 1.682, 2: 1.134, 3: 1.004}   # poly9, Table IV


def load_k1_sweep():
    """ratio -> {(strategy, evader): {metric: mean}} from the sweep runs."""
    out = {}
    for f in sorted(glob.glob('benchmark_results/*_table.txt')):
        head = open(f).readline()
        if "strategies=['minmax-alpha', 'minmax-alpha-vis']" not in head:
            continue
        sp = float(head.split('sp=')[1].split()[0])
        rows = list(csv.DictReader(open(f.replace('_table.txt', '_raw.csv'))))
        agg = {}
        for key in ('los_pct', 'n_escape'):
            for s in ('minmax-alpha', 'minmax-alpha-vis'):
                for e in ('skeleton', 'adversarial'):
                    v = [float(r[key]) for r in rows
                         if r['strategy'] == s and r['evader'] == e]
                    if v:
                        agg[(s, e, key)] = st.mean(v)
        out[sp] = agg
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='RoadmapBasedTracking/figures/speed_sweep.pdf')
    ap.add_argument('--team-jsons', nargs='*', default=[])
    args = ap.parse_args()

    sweep = load_k1_sweep()
    ratios = list(sweep)
    n_panels = 3 if args.team_jsons else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 2.9))
    plt.rcParams.update({'font.size': 11})

    def mark_sstar(ax, ks=(1,)):
        for k in ks:
            ax.axvline(S_STAR[k], color='gray', ls=':', lw=1.2)
            ax.text(S_STAR[k], ax.get_ylim()[1], f' $s^\\star_{{{k}}}$' if k > 1
                    else ' $s^\\star$', ha='left', va='top', fontsize=10, color='gray')

    styles = {('minmax-alpha'): dict(color='C0', marker='o', label='Min-Max'),
              ('minmax-alpha-vis'): dict(color='C1', marker='s', label='Min-Max +Vis')}

    ax = axes[0]
    for s, sty in styles.items():
        ax.plot(ratios, [sweep[r][(s, 'skeleton', 'los_pct')] for r in ratios],
                lw=1.6, ms=4, **sty)
    ax.set_xlabel('$s_p/s_e$'); ax.set_ylabel('%LOS (skeleton)')
    ax.legend(fontsize=9, loc='lower right'); mark_sstar(ax)

    ax = axes[1]
    for s, sty in styles.items():
        ax.plot(ratios, [sweep[r][(s, 'adversarial', 'n_escape')] for r in ratios],
                lw=1.6, ms=4, **sty)
    ax.set_xlabel('$s_p/s_e$'); ax.set_ylabel('escapes/trial (adversarial)')
    mark_sstar(ax)

    if args.team_jsons:
        team = {}   # sp -> {k: in_view}
        for f in args.team_jsons:
            j = json.load(open(f))
            sp = j['args']['sp']
            team[sp] = {t['k']: t['in_view_pct'][0] for t in j['trials']}
        team = dict(sorted(team.items()))
        ax = axes[2]
        for k, c in [(1, 'C0'), (2, 'C2'), (3, 'C3')]:
            ax.plot(list(team), [team[r][k] for r in team], color=c,
                    marker='o', ms=4, lw=1.6, label=f'$k={k}$')
        ax.set_xlabel('$s_p/s_e$'); ax.set_ylabel('% in view ($m=1$)')
        ax.legend(fontsize=9, loc='lower right'); mark_sstar(ax, ks=(1, 2, 3))

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    print(f'[SAVED] {args.out}')


if __name__ == '__main__':
    main()
