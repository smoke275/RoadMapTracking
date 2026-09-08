"""
Minimum pursuer speed required by each polygon's guard placement.

For every polygon, places a virtual evader at each interior point of a grid
and evaluates the min-max alpha at that point: the ratio d_G / d_geo the
pursuer would need to reach the worst corner in time, assuming it is already
at its optimal patrol-edge position. The max of those per-point minimums is
the smallest s_p/s_e (evader speed normalised to 1) that guarantees corner
cutoff everywhere in the polygon. This is an oracle bound: a real pursuer
that has to travel to its guard position needs at least this much.

Run:  python run_max_speed.py                 # every resources/sites_poly*.csv
      python run_max_speed.py --polygons poly3 poly9 --grid 40

Note: each polygon build overwrites the single-slot pipeline cache
(resources/ker_cache.pkl), same as run_benchmark.py.
"""
import argparse
import glob
import os
import re
import time

import ker_pipeline
from benchmark.harness import RESOURCES, load_poly

PAPER_SPEED_RATIO = 0.8 / 1.0   # s_p = 0.8, s_e = 1.0 in sections/07_experiments.tex


def _discover_polygons() -> list:
    names = []
    for path in glob.glob(os.path.join(RESOURCES, 'sites_poly*.csv')):
        m = re.search(r'sites_(poly\d+)\.csv$', path)
        if m:
            names.append(m.group(1))
    return sorted(names, key=lambda n: int(n[4:]))


def main():
    parser = argparse.ArgumentParser(
        description='Minimum pursuer speed ratio required per polygon')
    parser.add_argument('--polygons', nargs='+', default=None,
                        help='Preset names (polyN) or CSV paths '
                             '(default: every resources/sites_poly*.csv)')
    parser.add_argument('--grid', type=int, default=25,
                        help='Grid resolution per axis (default: 25)')
    args = parser.parse_args()

    polygons = args.polygons or _discover_polygons()
    if not polygons:
        raise SystemExit('no polygons found')

    rows = []
    for name in polygons:
        print(f'=== {name} ===', flush=True)
        data = ker_pipeline.build(load_poly(name), renderer=None, force_recompute=True)
        t0 = time.time()
        result = ker_pipeline.sweep_max_alpha(data, data.shapely_env, args.grid)
        elapsed = time.time() - t0
        if result is None:
            print(f'  no interior grid points at grid={args.grid}', flush=True)
            continue
        x, y, alpha, n_pts = result
        rows.append((name, len(data.poly), len(data.corners), len(data.guards),
                     n_pts, alpha, x, y, elapsed))
        print(f'  min s_p/s_e = {alpha:.3f} at ({x:.1f}, {y:.1f})  '
              f'[{n_pts} pts, {elapsed:.0f}s]', flush=True)

    header = (f'{"Polygon":<8} {"Verts":>5} {"Corners":>7} {"Guards":>6} '
              f'{"Grid pts":>8} {"min s_p/s_e":>11} {"worst evader (x, y)":>22} '
              f'{"paper 0.8 ok?":>13}')
    lines = [header, '-' * len(header)]
    for name, nv, nc, ng, n_pts, alpha, x, y, _ in rows:
        ok = 'yes' if PAPER_SPEED_RATIO >= alpha else 'no'
        lines.append(f'{name:<8} {nv:>5} {nc:>7} {ng:>6} {n_pts:>8} {alpha:>11.3f} '
                     f'{f"({x:.1f}, {y:.1f})":>22} {ok:>13}')
    lines.append('')
    lines.append(f'min s_p/s_e: max over the grid of the min-max alpha at each evader '
                 f'position (evader speed = 1, grid {args.grid}x{args.grid}). '
                 f'"paper 0.8 ok?" compares against the s_p/s_e = {PAPER_SPEED_RATIO:.1f} '
                 f'regime used in the experiments section — an oracle bound, so "no" '
                 f'means even a zero-transit pursuer cannot guarantee cutoff everywhere.')
    table = '\n'.join(lines)

    print()
    print(table)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, time.strftime('%Y%m%d_%H%M%S') + '_max_speed.txt')
    with open(out_path, 'w') as f:
        f.write(table + '\n')
    print(f'\n[SAVED] {out_path}')


if __name__ == '__main__':
    main()
