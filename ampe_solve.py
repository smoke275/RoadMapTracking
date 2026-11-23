"""Offline solve for the minimax baseline (baselines/ampe): the visibility-based
pursuit-evasion solver of Antonio, Becerra and Murrieta-Cid, run on our polygons.

For every polygon, mode and speed ratio it writes the value table and the
pursuer neighbor lists to resources/ampe/, where benchmark/ampe_pursuer.py
reads them. Run inside the roadmaptracking image:

    python ampe_solve.py --polygons poly9 --ratios 0.8 --samples 3000
"""
import argparse
import contextlib
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AMPE = os.path.join(HERE, 'baselines', 'ampe')
OUT = os.path.join(HERE, 'resources', 'ampe')


def table_prefix(fingerprint: str, mode: str, ratio: float) -> str:
    return os.path.join(OUT, f'{fingerprint[:16]}_{mode}_r{ratio:.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--polygons', nargs='+', default=None)
    ap.add_argument('--modes', nargs='+', default=['roadmap', 'free'], choices=['roadmap', 'free'])
    ap.add_argument('--ratios', nargs='+', type=float, default=[0.8], help='s_p / s_e')
    ap.add_argument('--samples', type=int, default=3000, help='evader samples (and pursuer samples in free mode)')
    ap.add_argument('--spacing', type=float, default=3.0, help='spacing of the pursuer samples on the roadmap')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    import ker_pipeline
    from cache import poly_fingerprint
    from benchmark.harness import load_poly

    subprocess.run(['bash', os.path.join(AMPE, 'build.sh')], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    os.makedirs(OUT, exist_ok=True)
    polygons = args.polygons or sorted(
        (re.search(r'sites_(poly\d+)\.csv', p).group(1)
         for p in glob.glob(os.path.join(HERE, 'resources', 'sites_poly*.csv'))),
        key=lambda s: int(s[4:]))
    for name in polygons:
        with open(os.devnull, 'w') as null, contextlib.redirect_stdout(null):
            data = ker_pipeline.build(load_poly(name), renderer=None)
        fp = poly_fingerprint(data.poly)
        xs = list(data.shapely_env.exterior.coords)[:-1]
        polyf = os.path.join(OUT, f'{fp[:16]}_poly.txt')
        roadf = os.path.join(OUT, f'{fp[:16]}_roadmap.txt')
        with open(polyf, 'w') as f:
            f.write(f'{len(xs)}\n' + ''.join(f'{x:.6f} {y:.6f}\n' for x, y in xs))
        with open(roadf, 'w') as f:      # the dense graph on which the controller of the paper runs
            f.write(f'{len(data.total_edges)}\n' + ''.join(
                f'{data.vertices[a].x:.6f} {data.vertices[a].y:.6f} '
                f'{data.vertices[b].x:.6f} {data.vertices[b].y:.6f}\n' for a, b in data.total_edges))
        for mode in args.modes:
            for ratio in args.ratios:
                prefix = table_prefix(fp, mode, ratio)
                if os.path.exists(prefix + '.meta') and not args.force:
                    print(f'[{name} {mode} r={ratio}] exists, skipped', flush=True)
                    continue
                t0 = time.time()
                r = subprocess.run([os.path.join(AMPE, 'ampe_poly'), polyf, roadf, str(args.samples), mode,
                                    str(ratio), str(args.seed), str(args.spacing), prefix],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                res = [l for l in r.stdout.splitlines() if l.startswith('RESULT')]
                if r.returncode != 0 or not res:
                    print(f'[{name} {mode} r={ratio}] FAILED (exit {r.returncode})', flush=True)
                    continue
                with open(prefix + '.meta') as f:
                    meta = json.load(f)
                meta['polygon'] = name
                with open(prefix + '.meta', 'w') as f:
                    json.dump(meta, f)
                print(f'[{name} {mode} r={ratio}] {res[0][7:]}  [{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    sys.exit(main())
