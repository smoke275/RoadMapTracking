"""
Benchmark-environment statistics (paper Table I and the KER+ILP column of
Table II): vertices and reflex corners after cleaning, KER candidates,
guards, enriched patrol-graph edges, roadmap length and guard coverage.

Run:  python run_environment_stats.py --polygons poly1 poly2 ...
"""
import argparse
import os
import pickle
import time

import ker_pipeline
from benchmark.harness import load_poly
from config import CACHE_FILE


def main():
    ap = argparse.ArgumentParser(description='Benchmark environment statistics')
    ap.add_argument('--polygons', nargs='+', required=True)
    args = ap.parse_args()

    header = (f'{"Polygon":<8} {"n":>4} {"|C|":>4} {"|K|":>6} {"|S|":>4} '
              f'{"|E|":>5} {"Length":>9} {"%Area":>7}')
    lines = [header, '-' * len(header)]
    for name in args.polygons:
        data = ker_pipeline.build(load_poly(name), renderer=None, force_recompute=True)
        with open(CACHE_FILE, 'rb') as f:
            n_ker = len(pickle.load(f)['data']['KER_coords'])
        length = sum(line.length for line in data.path_lines)
        row = (f'{name:<8} {len(data.poly):>4} {len(data.corners):>4} {n_ker:>6} '
               f'{len(data.guards):>4} {len(data.total_edges):>5} {length:>9.1f} '
               f'{data.coverage_pct:>6.1f}%')
        lines.append(row)
        print(row, flush=True)
    lines += ['', 'n, |C| after boundary cleaning; |K| KER candidates before domination '
                  'pre-processing; |E| edges of the enriched patrol graph; Length = total '
                  'patrol-path length; %Area = guard visibility coverage.']
    table = '\n'.join(lines)
    print('\n' + table)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime('%Y%m%d_%H%M%S') + '_environments.txt')
    with open(path, 'w') as f:
        f.write(table + '\n')
    print(f'\n[SAVED] {path}')


if __name__ == '__main__':
    main()
