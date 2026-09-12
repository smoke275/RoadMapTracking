"""
k-pursuer corner grouping and the oracle speed ratio it needs.

For each polygon and each k: partition the reflex corners into k groups
(corner_groups.make_grouping: LOS-overlap affinity + agglomerative
clustering), then sweep an evader grid and report the max over the grid of
the k-pursuer alpha (max over groups of the group's optimal alpha). That is
the smallest s_p/s_e for which k zero-transit pursuers, each confined to its
own corner group, guarantee corner cutoff everywhere — the k-pursuer
counterpart of run_max_speed.py's single-pursuer bound.

Run:  python run_groups.py                          # current polygon, k = 1..3
      python run_groups.py --polygons poly3 poly9 --k 1 2 3 4
      python run_groups.py --refine                 # local-search the grouping
      python run_groups.py --show-affinity          # print the corner-pair table

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
                           make_grouping, refine_groups, save_refined)


def main():
    parser = argparse.ArgumentParser(description='k-pursuer corner grouping report')
    parser.add_argument('--polygons', nargs='+', default=['current'],
                        help="'current', preset names (polyN), or CSV paths")
    parser.add_argument('--k', nargs='+', type=int, default=[1, 2, 3],
                        help='Team sizes to evaluate (default 1 2 3)')
    parser.add_argument('--grid', type=int, default=25,
                        help='Evader grid per axis for the speed-ratio sweep (default 25)')
    parser.add_argument('--aff-grid', type=int, default=GROUP_AFFINITY_GRID,
                        help='Evader grid per axis for the overlap affinity')
    parser.add_argument('--step', type=float, default=GROUP_TRACE_STEP,
                        help='Trace sampling step along escape paths')
    parser.add_argument('--refine', action='store_true',
                        help='Greedy local search on the grouping (uses --refine-grid)')
    parser.add_argument('--refine-grid', type=int, default=10)
    parser.add_argument('--recompute-affinity', action='store_true')
    parser.add_argument('--show-affinity', action='store_true')
    parser.add_argument('--raw', action='store_true',
                        help='Ignore cached refined groupings; report the raw clustering')
    args = parser.parse_args()

    rows = []
    for name in args.polygons:
        print(f'=== {name} ===', flush=True)
        data = ker_pipeline.build(load_poly(name), renderer=None, force_recompute=True)
        t0 = time.time()
        grid_pl = grid_path_lengths(data, args.grid)
        print(f'  sweep grid: {len(grid_pl)} interior points ({time.time() - t0:.0f}s)', flush=True)
        refine_pl = grid_path_lengths(data, args.refine_grid) if args.refine else None

        shown = False
        for k in args.k:
            t0 = time.time()
            grouping = make_grouping(
                data, k, args.aff_grid, args.step, force=args.recompute_affinity and not shown,
                progress=lambda d, t: print(f'\r  affinity {d}/{t}', end='', flush=True),
                use_refined=not (args.refine or args.raw))
            print()
            if args.show_affinity and not shown:
                print(format_affinity(grouping, list(data.corners)))
            shown = True
            groups = grouping.groups
            if args.refine and len(groups) > 1:
                groups = refine_groups(data, groups, refine_pl)
                save_refined(data, args.aff_grid, args.step, k, groups)
                print(f'  refined grouping saved; the GUI (config.NUM_PURSUERS={k}) will use it.')
            alpha, x, y, per_group = evaluate_groups(data, groups, grid_pl)
            rows.append((name, k, groups, alpha, x, y, per_group, time.time() - t0))
            print(f'  k={k}: min s_p/s_e = {alpha:.3f} at ({x:.1f}, {y:.1f})   groups={groups}   '
                  f'per-group worst={[round(a, 2) for a in per_group]}', flush=True)

    header = f'{"Polygon":<8} {"k":>2} {"min s_p/s_e":>11} {"worst evader":>18}   groups (per-group worst alpha)'
    lines = [header, '-' * len(header)]
    for name, k, groups, alpha, x, y, per_group, _ in rows:
        gs = '  '.join(f'{g} ({a:.2f})' for g, a in zip(groups, per_group))
        lines.append(f'{name:<8} {k:>2} {alpha:>11.3f} {f"({x:.1f}, {y:.1f})":>18}   {gs}')
    lines.append('')
    lines.append(f'min s_p/s_e: max over a {args.grid}x{args.grid} evader grid of the k-pursuer '
                 f'alpha (max over corner groups of that group\'s optimal alpha); evader '
                 f'speed = 1, zero-transit pursuers. Groups from LOS-overlap affinity '
                 f'(grid {args.aff_grid}, step {args.step})'
                 + (', refined by local search.' if args.refine else
                    ' (raw clustering).' if args.raw else
                    ' (refined groupings from cache where available).'))
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
