"""
Headless benchmark runner for the tracking-performance tables (paper Table 3).

Run:  python run_benchmark.py                       # benchmarks the current polygon
      python run_benchmark.py --polygons my_env.csv poly2 \
          --strategies minmax-alpha geo-follow tsp-patrol \
          --evaders skeleton adversarial

The default polygon is 'current' — whatever config.FILE_NAME points at,
i.e. the polygon last saved by the drawing tool (python main.py). Draw your
environment there first, then run the benchmark on it.

Note: each polygon build overwrites the single-slot pipeline cache
(resources/ker_cache.pkl); the interactive app will simply recompute on its
next run if its polygon differs.
"""
import argparse
import multiprocessing
import os

import ker_pipeline
from benchmark.harness import load_poly, run_config, run_config_parallel
from benchmark.pursuers import PURSUER_CLASSES
from benchmark.report import format_table, save_results


def main():
    parser = argparse.ArgumentParser(description='Headless tracking benchmark')
    parser.add_argument('--polygons', nargs='+', default=['current'],
                        help="Polygons to benchmark: 'current' (the polygon saved "
                             'by the drawing tool), preset names (polyN), or CSV paths')
    parser.add_argument('--strategies', nargs='+', default=['minmax-alpha'],
                        choices=sorted(PURSUER_CLASSES),
                        help='Pursuer strategies to benchmark')
    parser.add_argument('--evaders', nargs='+', default=['skeleton'],
                        choices=['skeleton', 'adversarial'],
                        help='Evader behavior models')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Independent trials per configuration')
    parser.add_argument('--frames', type=int, default=600,
                        help='Simulation frames per trial')
    parser.add_argument('--sp', type=float, default=0.8,
                        help='Pursuer speed (world units per frame)')
    parser.add_argument('--se', type=float, default=1.0,
                        help='Evader speed (world units per frame)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel worker processes (default: all CPU cores). '
                             'Each trial is ~1-2 min of real computation, so this '
                             'matters a lot once --seeds is in the hundreds.')
    parser.add_argument('--sequential', action='store_true',
                        help='Disable multiprocessing (debugging only)')
    args = parser.parse_args()

    n_workers = args.workers or multiprocessing.cpu_count()

    results = []
    for polygon in args.polygons:
        print(f'=== {polygon} ({n_workers} workers) ===', flush=True)
        data = ker_pipeline.build(load_poly(polygon), renderer=None,
                                  force_recompute=True)
        for evader_model in args.evaders:
            for strategy in args.strategies:
                if args.sequential:
                    results.append(run_config(
                        polygon, strategy=strategy, evader_model=evader_model,
                        n_seeds=args.seeds, frames=args.frames,
                        s_p=args.sp, s_e=args.se, data=data))
                else:
                    results.append(run_config_parallel(
                        polygon, strategy=strategy, evader_model=evader_model,
                        n_seeds=args.seeds, frames=args.frames,
                        s_p=args.sp, s_e=args.se, data=data,
                        n_workers=n_workers))

    print()
    print(format_table(results))
    table_path, csv_path = save_results(results, run_args=vars(args))
    print(f'\n[SAVED] {table_path}\n[SAVED] {csv_path}')


if __name__ == '__main__':
    main()
