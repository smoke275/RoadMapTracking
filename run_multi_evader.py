"""
k pursuers vs m evaders — headless evaluation on one polygon.

Two experiments, both using the stored corner partition for each k
(run_groups.py; ILP preferred) and the nearest-evader rule
L_c = min_j d_geo(e_j, c) of corner_groups.compute_path_lengths_multi:

  oracle   critical speed ratio s*_{k,m}: the worst, over sampled m-evader
           configurations, of the team alpha (max over groups of the group's
           optimal alpha). Configurations are uniform in the polygon with
           every evader at least a standoff from every corner (alpha diverges
           as an evader reaches a corner; the standoff defaults to the 25x25
           grid's own minimum so m = 1 is comparable to the paper's grid
           sweep). Also reports the 95th percentile and mean.
  trials   T-frame simulations: pursuers spawn at their guards, evaders
           wander the Voronoi skeleton (SkeletonEvader with destination
           memory), pursuers drive toward their per-frame guards with the
           stable-node controller. Per frame we record the achieved team
           alpha (max over corners of min over pursuers), the fraction of
           evaders in line of sight of at least one pursuer, and breach
           events (an evader within BREACH_RADIUS of a corner while no
           pursuer can beat it there). Speeds follow the paper: s_p = 0.8,
           s_e = 1.0 world units per frame.

Run:  python run_multi_evader.py --polygons poly9 --k 1 2 3 --m 1 2 3
      python run_multi_evader.py --seeds 20 --frames 600 --workers 24
Outputs benchmark_results/<stamp>_multi_evader.{txt,json}; plot with
plot_multi_evader.py.
"""
import argparse
import json
import math
import multiprocessing
import os
import random
import statistics
import time

from shapely.geometry import LineString

import ker_pipeline
from benchmark.evaders import SkeletonEvader
from benchmark.harness import load_poly
from benchmark.metrics import BREACH_RADIUS
from config import GROUP_AFFINITY_GRID, GROUP_TRACE_STEP
from corner_groups import (_alphas_at, compute_frame_multi, compute_optimal_guard,
                           compute_path_lengths_multi, evader_grid, make_grouping)
from geometry import get_random_point_in_polygon
from pursuer_motion import StableNodeController


# ---------------------------------------------------------------------------
# Oracle speed ratio over sampled evader configurations
# ---------------------------------------------------------------------------
def grid_standoff(data, grid_n: int = 25) -> float:
    """Smallest distance from any point of the grid_n x grid_n interior grid
    to any reflex corner. alpha_c -> infinity as an evader approaches c, so
    the grid's spacing implicitly bounds the single-evader sweep; sampled
    configurations must respect the same bound to be comparable."""
    best = math.inf
    for x, y in evader_grid(data, grid_n):
        for c in data.corners:
            best = min(best, math.hypot(x - data.poly[c].x(), y - data.poly[c].y()))
    return best


def sample_configs(data, m: int, n: int, seed: int = 0, standoff: float = 0.0) -> list:
    """n configurations of m evaders uniform in the polygon, each evader at
    least `standoff` from every reflex corner (rejection sampling)."""
    state = random.getstate()
    random.seed(seed)
    try:
        out = []
        while len(out) < n:
            evs = []
            while len(evs) < m:
                p = get_random_point_in_polygon(data.shapely_env)
                if all(math.hypot(p.x - data.poly[c].x(), p.y - data.poly[c].y()) >= standoff
                       for c in data.corners):
                    evs.append((p.x, p.y))
            out.append(evs)
        return out
    finally:
        random.setstate(state)


def oracle_speed(data, groups: list, configs: list) -> dict:
    vals = []
    for evs in configs:
        pl, _, _ = compute_path_lengths_multi(evs, data)
        vals.append(max(compute_optimal_guard(pl, data, corners=g)[2] for g in groups))
    vals.sort()
    return {'max': vals[-1], 'p95': vals[int(0.95 * (len(vals) - 1))],
            'mean': statistics.mean(vals), 'n': len(vals)}


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------
_W = {}   # worker globals (fork-inherited)


def _init(data, groups_by_k):
    _W['data'], _W['groups'] = data, groups_by_k


def run_trial(data, groups: list, m: int, seed: int, frames: int,
              s_p: float, s_e: float) -> dict:
    random.seed(seed)
    evs = [get_random_point_in_polygon(data.shapely_env) for _ in range(m)]
    agents = [SkeletonEvader(data.skel_nodes, data.skel_adj, data.shapely_env,
                             (p.x, p.y), s_e, avoid_recent=3) for p in evs]
    evs = [(p.x, p.y) for p in evs]
    seed_mf = compute_frame_multi(evs, None, [evs[0]] * len(groups), data, groups)
    ctrls = [StableNodeController((fc.guard.x, fc.guard.y)) for fc in seed_mf.frames]

    alphas, in_view, breaches = [], [], 0
    in_breach = set()
    for _ in range(frames):
        positions = [tuple(c.pos) for c in ctrls]
        mf = compute_frame_multi(evs, None, positions, data, groups)
        alphas.append(mf.achieved_alpha if math.isfinite(mf.achieved_alpha) else float('nan'))
        # line of sight and breaches per evader
        per_p = [_alphas_at(p, mf.path_lengths, data) for p in positions]
        seen = 0
        now = set()
        for j, e in enumerate(evs):
            if any(data.shapely_env.covers(LineString([p, e])) for p in positions):
                seen += 1
            own = mf.per_evader_pl[j]
            for c in data.corners:
                if own[c] < BREACH_RADIUS:
                    a = min(pp[c] * mf.path_lengths[c] / own[c] for pp in per_p)
                    if a > 1.0:
                        now.add((j, c))
        in_view.append(seen / m)
        breaches += len(now - in_breach)
        in_breach = now
        for ctrl, fc in zip(ctrls, mf.frames):
            ctrl.step(data.graph, (fc.guard.x, fc.guard.y), s_p)
        evs = [tuple(a.step(1.0)) for a in agents]
    fin = [a for a in alphas if not math.isnan(a)]
    return {'mean_alpha': statistics.mean(fin) if fin else float('nan'),
            'peak_alpha': max(fin) if fin else float('nan'),
            'in_view_pct': 100.0 * statistics.mean(in_view),
            'n_breach': breaches}


def _job(args):
    k, m, seed, frames, s_p, s_e = args
    return k, m, seed, run_trial(_W['data'], _W['groups'][k], m, seed, frames, s_p, s_e)


def main():
    ap = argparse.ArgumentParser(description='k pursuers vs m evaders')
    ap.add_argument('--polygons', nargs='+', default=['poly9'])
    ap.add_argument('--k', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--m', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--configs', type=int, default=400, help='Sampled evader configurations per m')
    ap.add_argument('--standoff', type=float, default=None,
                    help='Min evader-to-corner distance in sampled configurations '
                         '(default: the 25x25 grid\'s own minimum)')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--frames', type=int, default=600)
    ap.add_argument('--sp', type=float, default=0.8)
    ap.add_argument('--se', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--no-trials', action='store_true')
    args = ap.parse_args()

    results = {'args': vars(args), 'oracle': [], 'trials': []}
    for name in args.polygons:
        print(f'=== {name} ===', flush=True)
        data = ker_pipeline.build(load_poly(name), renderer=None, force_recompute=True)
        groups_by_k = {k: make_grouping(data, k, GROUP_AFFINITY_GRID, GROUP_TRACE_STEP).groups
                       for k in args.k}
        for k, g in groups_by_k.items():
            print(f'  k={k} partition: {g}', flush=True)

        # ---- oracle ---------------------------------------------------
        standoff = args.standoff if args.standoff is not None else grid_standoff(data, 25)
        print(f'  corner standoff for sampled configurations: {standoff:.1f} units', flush=True)
        for m in args.m:
            t0 = time.time()
            configs = sample_configs(data, m, args.configs, standoff=standoff)
            for k in args.k:
                o = oracle_speed(data, groups_by_k[k], configs)
                results['oracle'].append({'polygon': name, 'k': k, 'm': m, 'standoff': standoff, **o})
                print(f'  oracle k={k} m={m}: s*={o["max"]:.3f}  p95={o["p95"]:.3f}  '
                      f'mean={o["mean"]:.3f}  ({o["n"]} configs, {time.time() - t0:.0f}s)', flush=True)

        # ---- trials ---------------------------------------------------
        if not args.no_trials:
            jobs = [(k, m, s, args.frames, args.sp, args.se)
                    for k in args.k for m in args.m for s in range(args.seeds)]
            n_workers = args.workers or multiprocessing.cpu_count()
            ctx = multiprocessing.get_context('fork')
            per = {}
            t0 = time.time()
            with ctx.Pool(n_workers, initializer=_init, initargs=(data, groups_by_k)) as pool:
                for i, (k, m, seed, met) in enumerate(pool.imap_unordered(_job, jobs), 1):
                    per.setdefault((k, m), []).append(met)
                    if i % 10 == 0 or i == len(jobs):
                        print(f'  trials {i}/{len(jobs)} ({time.time() - t0:.0f}s)', flush=True)
            for (k, m), mets in sorted(per.items()):
                agg = {}
                for key in mets[0]:
                    vals = [x[key] for x in mets if not (isinstance(x[key], float) and math.isnan(x[key]))]
                    agg[key] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
                results['trials'].append({'polygon': name, 'k': k, 'm': m, 'seeds': len(mets),
                                          **{key: list(v) for key, v in agg.items()}})

    # ---- report -------------------------------------------------------
    lines = [f'{"Polygon":<8} {"k":>2} {"m":>2} {"s*_(k,m)":>9} {"p95":>7} {"mean":>7}', '-' * 42]
    for r in results['oracle']:
        lines.append(f'{r["polygon"]:<8} {r["k"]:>2} {r["m"]:>2} {r["max"]:>9.3f} {r["p95"]:>7.3f} {r["mean"]:>7.3f}')
    if results['trials']:
        lines += ['', f'{"Polygon":<8} {"k":>2} {"m":>2} {"mean α":>13} {"α_max":>13} {"%in view":>14} {"N_breach":>11}',
                  '-' * 70]
        for r in results['trials']:
            f = lambda key: f'{r[key][0]:.2f}±{r[key][1]:.2f}'
            lines.append(f'{r["polygon"]:<8} {r["k"]:>2} {r["m"]:>2} {f("mean_alpha"):>13} '
                         f'{f("peak_alpha"):>13} {f("in_view_pct"):>13}% {f("n_breach"):>11}')
    lines += ['', f'oracle: worst / p95 / mean team alpha over {args.configs} sampled m-evader '
                  f'configurations (corner standoff as printed), L_c = min over evaders; '
                  f'trials: {args.seeds} seeds x '
                  f'{args.frames} frames, s_p={args.sp}, s_e={args.se}, skeleton evaders.']
    table = '\n'.join(lines)
    print('\n' + table)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results')
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(out_dir, f'{stamp}_multi_evader.txt'), 'w') as f:
        f.write(table + '\n')
    with open(os.path.join(out_dir, f'{stamp}_multi_evader.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\n[SAVED] {out_dir}/{stamp}_multi_evader.txt / .json')


if __name__ == '__main__':
    main()
