"""Analyze Gazebo case-study trials (ros2_sim/logs/trials/) and build the
paper figure.

Per strategy (mean +/- std over seeds), from the 30 Hz high-frequency logs
(identical control tick for every strategy, so accel/jerk are comparable):
  accel RMS, jerk RMS         drone translational smoothness
  cmd reversal rate           commanded-velocity direction changes > 90deg
                              per minute (planner-induced chattering)
and from the planner logs:
  %LOS drone / virtual, mean lag, held fraction.

Figure (two panels): commanded-speed/heading trace sample (chattering), and
accel/jerk + LOS summary bars.

Run (from repo root, roadmaptracking image is fine):
  python ros2_sim/scripts/analyze_trials.py --out RoadmapBasedTracking/figures/gazebo_tracking.pdf
"""
import argparse
import csv
import glob
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'alpha-guard': 'Min-Max', 'alpha-vis': 'Min-Max +Vis',
          'geo-follow': 'Geo-Follow', 'naive-dijkstra': 'Naive Dijkstra'}
ORDER = ['alpha-guard', 'alpha-vis', 'geo-follow', 'naive-dijkstra']
T_SETTLE = 15.0


def strategy_of(path):
    base = os.path.basename(path)
    return base.split('_seed')[0].split('_', 2)[2]


def hf_metrics(path):
    rows = [r for r in csv.DictReader(open(path)) if float(r['t']) > T_SETTLE]
    if len(rows) < 50:
        return None
    t = [float(r['t']) for r in rows]
    x = [float(r['x']) for r in rows]
    y = [float(r['y']) for r in rows]
    cvx = [float(r['cmd_vx']) for r in rows]
    cvy = [float(r['cmd_vy']) for r in rows]
    # resample positions onto a uniform 0.1 s grid before differentiating:
    # raw ticks can arrive in bursts (near-duplicate stamps), and finite
    # differences of mm-rounded positions over ~ms intervals are garbage
    DT = 0.1
    grid_t, grid_x, grid_y = [], [], []
    tg, i = t[0], 0
    while tg <= t[-1]:
        while i + 1 < len(t) and t[i+1] <= tg:
            i += 1
        if i + 1 < len(t) and t[i+1] > t[i]:
            w = (tg - t[i]) / (t[i+1] - t[i])
            grid_x.append(x[i] + w * (x[i+1] - x[i]))
            grid_y.append(y[i] + w * (y[i+1] - y[i]))
            grid_t.append(tg)
        tg += DT
    v = [((grid_x[i]-grid_x[i-1])/DT, (grid_y[i]-grid_y[i-1])/DT)
         for i in range(1, len(grid_t))]
    a = [((v[i][0]-v[i-1][0])/DT, (v[i][1]-v[i-1][1])/DT)
         for i in range(1, len(v))]
    j = [math.hypot((a[i][0]-a[i-1][0])/DT, (a[i][1]-a[i-1][1])/DT)
         for i in range(1, len(a))]
    arms = math.sqrt(st.mean(ax*ax + ay*ay for ax, ay in a)) if a else float('nan')
    jrms = math.sqrt(st.mean(q*q for q in j)) if j else float('nan')
    # commanded-direction reversals (> 90 deg between consecutive commands
    # that are both above a speed floor), per minute
    rev = 0
    prev = None
    for ux, uy in zip(cvx, cvy):
        n = math.hypot(ux, uy)
        if n < 0.15:
            continue
        cur = (ux/n, uy/n)
        if prev is not None and (cur[0]*prev[0] + cur[1]*prev[1]) < 0.0:
            rev += 1
        prev = cur
    dur_min = (t[-1] - t[0]) / 60.0
    return dict(arms=arms, jrms=jrms, rev=rev/dur_min if dur_min > 0 else float('nan'))


def plan_metrics(path):
    rows = [r for r in csv.DictReader(open(path)) if float(r['t']) > T_SETTLE]
    if len(rows) < 20:
        return None
    return dict(
        losd=100*st.mean(int(r['los_drone']) for r in rows),
        losv=100*st.mean(int(r['los_virt']) for r in rows),
        lag=st.mean(float(r['lag_m']) for r in rows),
        held=100*st.mean(int(r['held']) for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials-dir', default='ros2_sim/logs/trials')
    ap.add_argument('--out', default='RoadmapBasedTracking/figures/gazebo_tracking.pdf')
    args = ap.parse_args()

    agg = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(args.trials_dir, '*_seed*.csv'))):
        if f.endswith('_hf.csv'):
            m = hf_metrics(f)
        else:
            m = plan_metrics(f)
        if m is None:
            continue
        s = strategy_of(f.replace('_hf.csv', '.csv'))
        for k, val in m.items():
            agg[s][k].append(val)

    print(f"{'strategy':<15} {'aRMS':>10} {'jerkRMS':>10} {'rev/min':>8} "
          f"{'%LOS dr':>8} {'%LOS alg':>9} {'lag':>6} {'held%':>6}")
    stats = {}
    for s in ORDER:
        if s not in agg:
            continue
        d = agg[s]
        f = lambda k: (st.mean(d[k]), st.pstdev(d[k])) if d.get(k) else (float('nan'), 0)
        stats[s] = {k: f(k) for k in ('arms', 'jrms', 'rev', 'losd', 'losv', 'lag', 'held')}
        p = stats[s]
        print(f"{s:<15} {p['arms'][0]:7.2f}±{p['arms'][1]:.2f} "
              f"{p['jrms'][0]:7.1f}±{p['jrms'][1]:.1f} {p['rev'][0]:8.1f} "
              f"{p['losd'][0]:8.1f} {p['losv'][0]:9.1f} {p['lag'][0]:6.2f} "
              f"{p['held'][0]:6.1f}")

    # ---- figure: three summary bar panels --------------------------------
    plt.rcParams.update({'font.size': 10})
    strategies = [s for s in ORDER if s in stats]
    names = [LABELS[s] for s in strategies]
    xs = list(range(len(strategies)))
    colors = ['C0', 'C9', 'C2', 'C3']
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6))

    def bars(ax, key, ylabel, scale=1.0):
        vals = [stats[s][key][0] * scale for s in strategies]
        errs = [stats[s][key][1] * scale for s in strategies]
        ax.bar(xs, vals, 0.62, yerr=errs, color=colors[:len(xs)], capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=20, fontsize=8, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis='y')

    bars(axes[0], 'arms', 'accel RMS (m/s$^2$)')
    bars(axes[1], 'rev', 'command reversals / min')
    ax = axes[2]
    losd = [stats[s]['losd'][0] for s in strategies]
    losv = [stats[s]['losv'][0] for s in strategies]
    ax.bar([x - 0.19 for x in xs], losv, 0.36, color='0.65',
           label='planner (ideal)')
    ax.bar([x + 0.19 for x in xs], losd, 0.36, color='C2', label='drone')
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, fontsize=8, ha='right')
    ax.set_ylabel('%LOS')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    print(f'[SAVED] {args.out}')


if __name__ == '__main__':
    main()
