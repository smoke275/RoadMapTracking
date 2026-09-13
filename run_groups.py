"""
k-pursuer corner partition: exact optimisation (default) or the clustering
heuristic, and the speed ratio / line-of-sight each partition achieves.

For each polygon and each k:
  --method ilp (default)   partition_ilp.optimise_partition: MILP over corner
                           assignments and pursuer positions; minimises the
                           team critical speed ratio, then maximises the
                           parked-pursuer line-of-sight fraction Lambda at
                           that bound. --relax adds Pareto points at bounds
                           s_k (1 + r).
  --method heuristic       corner_groups.make_grouping: LOS-overlap affinity
                           + average-linkage clustering; --refine adds the
                           greedy local search on the speed ratio.
Every partition is then re-measured on a fine evader grid with the exact
continuous guard optimiser (the k-pursuer analogue of run_max_speed.py) and
stored in resources/corner_groups.pkl, where the GUI (config.NUM_PURSUERS)
picks it up.

Run:  python run_groups.py                                 # current polygon, k = 1..3, ILP
      python run_groups.py --polygons poly9 --k 2 3 --relax 0.1 0.25
      python run_groups.py --method heuristic --refine
      python run_groups.py --method heuristic --raw --show-affinity

Note: each polygon build overwrites the single-slot pipeline cache
(resources/ker_cache.pkl), same as run_benchmark.py.
"""
import argparse
import os
import time

import ker_pipeline
from benchmark.harness import load_poly
from config import GROUP_AFFINITY_GRID, GROUP_TRACE_STEP
from corner_groups import (evaluate_groups, format_affinity, grid_path_lengths,
                           make_grouping, refine_groups, save_partition)


def main():
    parser = argparse.ArgumentParser(description='k-pursuer corner partition report')
    parser.add_argument('--polygons', nargs='+', default=['current'],
                        help="'current', preset names (polyN), or CSV paths")
    parser.add_argument('--k', nargs='+', type=int, default=[1, 2, 3],
                        help='Team sizes (default 1 2 3)')
    parser.add_argument('--method', choices=['ilp', 'heuristic'], default='ilp')
    parser.add_argument('--grid', type=int, default=25,
                        help='Fine evader grid per axis for the final speed-ratio sweep')
    # ILP options
    parser.add_argument('--ilp-grid', type=int, default=8,
                        help='Evader grid per axis inside the MILP (default 8)')
    parser.add_argument('--spacing', type=float, default=15.0,
                        help='Candidate pursuer positions every this many units along edges')
    parser.add_argument('--relax', nargs='*', type=float, default=[0.1],
                        help='Extra Pareto points: max sight under speed bound s_k(1+r)')
    parser.add_argument('--time-limit', type=float, default=600.0)
    # heuristic options
    parser.add_argument('--aff-grid', type=int, default=GROUP_AFFINITY_GRID)
    parser.add_argument('--step', type=float, default=GROUP_TRACE_STEP)
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--refine-grid', type=int, default=10)
    parser.add_argument('--raw', action='store_true',
                        help='Heuristic only: ignore stored partitions, report the raw clustering')
    parser.add_argument('--recompute-affinity', action='store_true')
    parser.add_argument('--show-affinity', action='store_true')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not store the partitions for the GUI')
    args = parser.parse_args()

    rows = []
    for name in args.polygons:
        print(f'=== {name} ===', flush=True)
        data = ker_pipeline.build(load_poly(name), renderer=None, force_recompute=True)
        t0 = time.time()
        fine = grid_path_lengths(data, args.grid)
        print(f'  fine grid: {len(fine)} interior points ({time.time() - t0:.0f}s)', flush=True)

        if args.method == 'ilp':
            from partition_ilp import optimise_partition
            for k in args.k:
                t0 = time.time()
                out = optimise_partition(data, k, grid_n=args.ilp_grid, spacing=args.spacing,
                                         relax=tuple(args.relax), time_limit=args.time_limit,
                                         log=lambda m: print('  ' + m, flush=True))
                if 'lex' not in out:
                    continue
                variants = [('ILP min-speed', out['lex'])] + \
                           [(f'ILP bound +{r:.0%}', out[f'relax_{r}']) for r in args.relax
                            if out.get(f'relax_{r}') is not None and out[f'relax_{r}'].feasible]
                for label, res in variants:
                    alpha, x, y, per = evaluate_groups(data, res.groups, fine)
                    rows.append((name, k, label, res.groups, alpha, res.sight, per))
                    print(f'  k={k} {label}: s*={alpha:.3f} (fine grid)  Lambda={res.sight:.3f}  '
                          f'groups={res.groups}', flush=True)
                if not args.no_save:
                    save_partition(data, args.aff_grid, args.step, k, out['lex'].groups, method='ilp')
                    print(f'  stored ILP partition for k={k}; the GUI will use it.', flush=True)
                print(f'  k={k} done in {time.time() - t0:.0f}s', flush=True)
        else:
            refine_pl = grid_path_lengths(data, args.refine_grid) if args.refine else None
            shown = False
            for k in args.k:
                grouping = make_grouping(
                    data, k, args.aff_grid, args.step, force=args.recompute_affinity and not shown,
                    progress=lambda d, t: print(f'\r  affinity {d}/{t}', end='', flush=True),
                    use_refined=not (args.refine or args.raw))
                print()
                if args.show_affinity and not shown:
                    print(format_affinity(grouping, list(data.corners)))
                shown = True
                groups, label = grouping.groups, 'heuristic raw'
                if args.refine and len(groups) > 1:
                    groups, label = refine_groups(data, groups, refine_pl), 'heuristic refined'
                    if not args.no_save:
                        save_partition(data, args.aff_grid, args.step, k, groups, method='refined')
                alpha, x, y, per = evaluate_groups(data, groups, fine)
                rows.append((name, k, label, groups, alpha, float('nan'), per))
                print(f'  k={k} {label}: s*={alpha:.3f} at ({x:.1f}, {y:.1f})  groups={groups}  '
                      f'per-group worst={[round(a, 2) for a in per]}', flush=True)

    header = (f'{"Polygon":<8} {"k":>2} {"Method":<18} {"s*_k":>7} {"Lambda":>7}   '
              f'groups (per-group worst alpha)')
    lines = [header, '-' * len(header)]
    for name, k, label, groups, alpha, lam, per in rows:
        gs = '  '.join(f'{g} ({a:.2f})' for g, a in zip(groups, per))
        lam_s = f'{lam:.3f}' if lam == lam else '   -  '
        lines.append(f'{name:<8} {k:>2} {label:<18} {alpha:>7.3f} {lam_s:>7}   {gs}')
    lines.append('')
    lines.append(f's*_k: max over a {args.grid}x{args.grid} evader grid of the k-pursuer alpha '
                 f'(max over groups of that group\'s optimal alpha); evader speed = 1, '
                 f'zero-transit pursuers. Lambda: nearness-weighted fraction of (evader '
                 f'position, corner) pairs whose pursuer keeps line of sight for the whole '
                 f'escape from its chosen position (ILP grid {args.ilp_grid}x{args.ilp_grid}, '
                 f'positions every {args.spacing} units).')
    table = '\n'.join(lines)
    print()
    print(table)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, time.strftime('%Y%m%d_%H%M%S') + '_groups.txt')
    with open(out_path, 'w') as f:
        f.write(table + '\n')
    print(f'\n[SAVED] {out_path}')


if __name__ == '__main__':
    main()
