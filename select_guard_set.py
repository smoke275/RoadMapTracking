"""Second stage of the guard selection: among minimum guard sets, keep the one
whose roadmap has the smallest critical speed ratio.

Algorithm 1 fixes only the number of guards. Many guard sets of that size
exist, and the critical speed ratio s* of the resulting roadmap differs a lot
between them. This script draws minimum guard sets (sample 0 is the plain
witness selection, sample s > 0 breaks the ties of the 0/1 program at random
with seed s), builds the roadmap of each, evaluates s* on the grid of
run_max_speed.py, and stores the best set in resources/guard_sets.json, keyed
by the polygon fingerprint. ker_pipeline.build uses the stored set when
config.GUARD_SELECTION = 'witness-minspeed'.

    python select_guard_set.py --polygons poly9 --samples 100 --workers 22
"""
import argparse
import contextlib
import glob
import json
import os
import re
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def _evaluate(task):
    name, sample, grid = task
    import ker_pipeline
    from benchmark.harness import load_poly
    t0 = time.time()
    try:
        with open(os.devnull, 'w') as null, contextlib.redirect_stdout(null):
            data = ker_pipeline.build(load_poly(name), renderer=None,
                                      guard_sample=sample)
            res = ker_pipeline.sweep_max_alpha(data, data.shapely_env, grid)
        if res is None:
            raise RuntimeError('no interior grid points')
        return dict(sample=sample, guards=[list(g) for g in data.guards],
                    s_star=float(res[2]), coverage=float(data.coverage_pct),
                    length=float(sum(l.length for l in data.path_lines)),
                    sec=round(time.time() - t0, 1))
    except Exception as e:                      # one bad sample must not stop the search
        return dict(sample=sample, error=repr(e)[:300])


def main():
    import config
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--polygons', nargs='+', default=None)
    ap.add_argument('--samples', type=int, default=config.GUARD_SAMPLES)
    ap.add_argument('--grid', type=int, default=25)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    import ker_pipeline
    from cache import poly_fingerprint
    from geometry import clean_polygon
    from benchmark.harness import load_poly

    polygons = args.polygons or sorted(
        (re.search(r'sites_(poly\d+)\.csv', p).group(1)
         for p in glob.glob(os.path.join(HERE, 'resources', 'sites_poly*.csv'))),
        key=lambda s: int(s[4:]))
    try:
        with open(ker_pipeline.GUARD_SETS_FILE) as f:
            store = json.load(f)
    except (OSError, ValueError):
        store = {}

    lines = [f'# samples={args.samples} (plus the plain witness selection) grid={args.grid}',
             f'{"Polygon":<8} {"Guards":>6} {"Valid":>5} {"Distinct":>8} {"witness s*":>10} '
             f'{"best s*":>8} {"median":>7} {"worst":>7} {"best sample":>11}',
             '-' * 80]
    with Pool(args.workers) as pool:
        for name in polygons:
            t0 = time.time()
            tasks = [(name, s, args.grid) for s in range(args.samples + 1)]
            recs = pool.map(_evaluate, tasks, chunksize=1)
            ok = [r for r in recs if 'error' not in r]
            for r in recs:
                if 'error' in r:
                    print(f'  [{name}] sample {r["sample"]} failed: {r["error"]}', flush=True)
            base = next((r for r in ok if r['sample'] == 0), None)
            if base is None:
                print(f'=== {name}: the plain witness selection failed, skipped', flush=True)
                continue
            k = min(len(r['guards']) for r in ok)
            # same number of guards, and no less coverage than the plain selection
            valid = [r for r in ok if len(r['guards']) == k
                     and r['coverage'] >= base['coverage'] - 100.0 * config.COVERAGE_ETA]
            best = min(valid, key=lambda r: (round(r['s_star'], 6), r['length'], r['sample']))
            vals = sorted(r['s_star'] for r in valid)
            distinct = len({tuple(sorted(map(tuple, r['guards']))) for r in valid})
            fp = poly_fingerprint(clean_polygon(load_poly(name)))
            store[fp] = dict(polygon=name, guards=best['guards'], s_star=best['s_star'],
                             coverage=best['coverage'], sample=best['sample'],
                             samples=args.samples, grid=args.grid,
                             witness_s_star=base['s_star'],
                             all_s_star=[r['s_star'] for r in valid])
            with open(ker_pipeline.GUARD_SETS_FILE, 'w') as f:
                json.dump(store, f, indent=1)
            lines.append(f'{name:<8} {k:>6} {len(valid):>5} {distinct:>8} {base["s_star"]:>10.3f} '
                         f'{best["s_star"]:>8.3f} {vals[len(vals) // 2]:>7.3f} {vals[-1]:>7.3f} '
                         f'{best["sample"]:>11}')
            print(f'=== {name}: {k} guards, {len(valid)} valid samples ({distinct} distinct sets), '
                  f's* witness {base["s_star"]:.3f} -> best {best["s_star"]:.3f} '
                  f'(sample {best["sample"]}), worst {vals[-1]:.3f}  [{time.time() - t0:.0f}s]',
                  flush=True)

    lines += ['', 'witness s*: the set returned by the plain witness selection. best/median/worst: '
              'over the valid samples (minimum number of guards, coverage not below the plain '
              'selection). The best set is stored in resources/guard_sets.json.']
    table = '\n'.join(lines)
    print('\n' + table)
    out = os.path.join(HERE, 'benchmark_results', time.strftime('%Y%m%d_%H%M%S') + '_guard_search.txt')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(table + '\n')
    print(f'\n[SAVED] {out}')


if __name__ == '__main__':
    main()
