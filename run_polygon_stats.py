"""Standard geometric descriptors of the benchmark polygons.

Beyond the pipeline counts of Table I (n, |C|, |K|, |J|, |E|, %Area, s*),
computational geometry characterises a simple polygon by

  m        number of essential cuts (minimal pockets; watchman_route.py)
  L_w      shortest watchman route length, also normalised by sqrt(area)
  D_geo    geodesic diameter (longest shortest path; attained at vertices)
  D_link   link diameter: largest number of straight segments needed to join two
           points (estimated on polygon vertices + an interior grid as bend
           candidates, so it is an upper bound that is exact at this resolution)
  V_mean   mean visibility fraction: average over interior grid points of the
           visible area divided by the polygon area
  w_min    narrowest passage: twice the largest inward offset that keeps the
           interior connected (same definition as ros2_sim/scripts/generate_world.py)
  class    star-shaped / monotone / orthogonal / spiral membership

The KER columns (|J|, |E|, roadmap length, %Area) and s* under both roadmaps
are read from the most recent benchmark_results/*_environments.txt,
*_max_speed.txt and *_max_speed_watchman.txt so nothing is rebuilt here.

Run:  python run_polygon_stats.py                      # all resources/sites_poly*.csv
      python run_polygon_stats.py --polygons poly9 --grid 30
"""
import argparse
import csv
import glob
import math
import os
import re
import time

import numpy as np
import shapely
import visilibity as vis
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import Point, Polygon as ShapelyPolygon

from config import EPSILON
from geometry import poly_to_points, suppress_output
from watchman_route import (PolygonGeodesics, _load_csv, all_cuts, essential_cuts,
                            get_watchman_route, reflex_vertices)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES   = os.path.join(_SCRIPT_DIR, 'resources')
RESULTS     = os.path.join(_SCRIPT_DIR, 'benchmark_results')


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------
def interior_grid(poly: ShapelyPolygon, n: int) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, n + 2)[1:-1]
    ys = np.linspace(miny, maxy, n + 2)[1:-1]
    G = np.array([(x, y) for x in xs for y in ys])
    inside = shapely.contains_xy(poly.buffer(-1e-3), G[:, 0], G[:, 1])
    return G[inside]


def link_diameter(geo: PolygonGeodesics, grid: np.ndarray) -> int:
    """Max over node pairs of the unweighted visibility-graph distance, with
    polygon vertices and grid points as bend candidates."""
    P = np.vstack([geo.V, grid])
    k = len(P)
    I, J = np.triu_indices(k, 1)
    seen = geo._covers(P[I], P[J])
    A = np.zeros((k, k), dtype=np.int8)
    A[I[seen], J[seen]] = 1
    A[J[seen], I[seen]] = 1
    D = shortest_path(A, unweighted=True, directed=False)
    return int(D.max())


def mean_visibility(pts, poly: ShapelyPolygon, grid: np.ndarray) -> float:
    walls = vis.Polygon([vis.Point(float(x), float(y)) for x, y in pts])
    if walls.area() < 0:
        walls.reverse()
    env = vis.Environment([walls])
    env.PRINTING_DEBUG_DATA = False
    fr = []
    for x, y in grid:
        with suppress_output():
            vp = vis.Visibility_Polygon(vis.Point(float(x), float(y)), env, EPSILON)
        vx, vy = poly_to_points(vp)
        if len(vx) < 3:
            continue
        shp = ShapelyPolygon(list(zip(vx, vy)))
        if not shp.is_valid:
            shp = shp.buffer(0)
        fr.append(shp.intersection(poly).area / poly.area)
    return float(np.mean(fr))


def narrowest_passage(poly: ShapelyPolygon) -> float:
    lo, hi = 0.0, min(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])
    for _ in range(30):
        mid = (lo + hi) / 2
        d = poly.buffer(-mid)
        if d.geom_type == 'Polygon' and not d.is_empty:
            lo = mid
        else:
            hi = mid
    return 2 * lo


def is_star_shaped(V: np.ndarray) -> bool:
    """Kernel = intersection of the inner half-planes of all edges (ccw)."""
    minx, miny = V.min(axis=0) - 10
    maxx, maxy = V.max(axis=0) + 10
    ker = ShapelyPolygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    R = 10 * max(maxx - minx, maxy - miny)
    n = len(V)
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        d = b - a
        d = d / np.linalg.norm(d)
        nrm = np.array([-d[1], d[0]])          # inward normal for a ccw polygon
        half = ShapelyPolygon([a - R * d, b + R * d, b + R * d + R * nrm, a - R * d + R * nrm])
        ker = ker.intersection(half)
        if ker.is_empty or ker.area < 1e-9:
            return False
    return True


def is_monotone(V: np.ndarray, n_dirs: int = 180) -> bool:
    """Monotone with respect to some direction: the boundary projected on it
    has one increasing and one decreasing run."""
    for k in range(n_dirs):
        th = math.pi * k / n_dirs
        p = V @ np.array([math.cos(th), math.sin(th)])
        s = np.sign(np.roll(p, -1) - p)
        s = s[s != 0]
        if len(s) == 0:
            return True
        changes = int(np.sum(s != np.roll(s, 1)))
        if changes <= 2:
            return True
    return False


def is_orthogonal(V: np.ndarray, tol: float = 1e-6) -> bool:
    E = np.roll(V, -1, axis=0) - V
    return bool(np.all((np.abs(E[:, 0]) < tol) | (np.abs(E[:, 1]) < tol)))


def is_spiral(V: np.ndarray) -> bool:
    """Reflex vertices form one contiguous run along the boundary."""
    n = len(V)
    r = set(reflex_vertices(V))
    if not r:
        return True
    runs = sum(1 for i in range(n) if i in r and (i - 1) % n not in r)
    return runs == 1


def polygon_class(V: np.ndarray) -> str:
    tags = []
    if is_star_shaped(V):
        tags.append('star')
    if is_monotone(V):
        tags.append('monotone')
    if is_orthogonal(V):
        tags.append('orthogonal')
    if is_spiral(V):
        tags.append('spiral')
    return '+'.join(tags) if tags else 'general'


# ---------------------------------------------------------------------------
# previously computed pipeline columns
# ---------------------------------------------------------------------------
def _latest(pattern: str):
    files = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    return files[-1] if files else None


def read_environments() -> dict:
    path = _latest('*_environments.txt')
    out = {}
    if path is None:
        return out
    with open(path) as f:
        for line in f:
            m = re.match(r'(poly\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%', line)
            if m:
                out[m.group(1)] = dict(guards=int(m.group(5)), edges=int(m.group(6)),
                                       ker_length=float(m.group(7)), area_pct=float(m.group(8)))
    return out


def read_sstar(pattern: str) -> dict:
    path = _latest(pattern)
    out = {}
    if path is None:
        return out
    with open(path) as f:
        for line in f:
            m = re.match(r'(poly\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+)', line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Geometric descriptors of the benchmark polygons')
    ap.add_argument('--polygons', nargs='+', default=None)
    ap.add_argument('--grid', type=int, default=25, help='interior grid per axis')
    args = ap.parse_args()
    names = args.polygons or sorted(
        (re.search(r'sites_(poly\d+)\.csv$', p).group(1)
         for p in glob.glob(os.path.join(RESOURCES, 'sites_poly*.csv'))),
        key=lambda s: int(s[4:]))

    envs   = read_environments()
    s_ker  = read_sstar('*_max_speed.txt')
    s_wm   = read_sstar('*_max_speed_watchman.txt')

    rows = []
    for name in names:
        t0 = time.time()
        pts  = _load_csv(os.path.join(RESOURCES, f'sites_{name}.csv'))
        geo  = PolygonGeodesics(pts)
        poly = geo.poly
        V    = geo.V
        grid = interior_grid(poly, args.grid)
        wr   = get_watchman_route(pts)
        area, perim = poly.area, poly.length
        row = dict(
            polygon=name, n=len(V), reflex=len(reflex_vertices(V)),
            cuts=len(all_cuts(geo)), essential=len(wr.cuts),
            area=area, perimeter=perim,
            watchman_len=wr.length, watchman_norm=wr.length / math.sqrt(area),
            geo_diameter=float(np.max(geo.D[np.isfinite(geo.D)])),
            geo_diam_norm=float(np.max(geo.D[np.isfinite(geo.D)])) / math.sqrt(area),
            link_diameter=link_diameter(geo, grid),
            mean_vis=mean_visibility(pts, poly, grid),
            passage=narrowest_passage(poly),
            passage_norm=narrowest_passage(poly) / math.sqrt(area),
            cls=polygon_class(V),
            guards=envs.get(name, {}).get('guards'),
            edges=envs.get(name, {}).get('edges'),
            ker_length=envs.get(name, {}).get('ker_length'),
            ker_norm=(envs[name]['ker_length'] / math.sqrt(area)) if name in envs else None,
            area_pct=envs.get(name, {}).get('area_pct'),
            sstar_ker=s_ker.get(name), sstar_watchman=s_wm.get(name),
        )
        rows.append(row)
        print(f'{name}: n={row["n"]} r={row["reflex"]} m={row["essential"]} '
              f'L_w={row["watchman_len"]:.0f} D_geo={row["geo_diameter"]:.0f} '
              f'D_link={row["link_diameter"]} V_mean={row["mean_vis"]:.3f} '
              f'w_min={row["passage"]:.1f} class={row["cls"]} ({time.time() - t0:.0f}s)',
              flush=True)

    def f(v, w, d=1):
        return f'{"-":>{w}}' if v is None else f'{v:>{w}.{d}f}'

    header = (f'{"Polygon":<7} {"n":>3} {"r":>3} {"m":>3} {"|J|":>4} {"|E|":>4} '
              f'{"L_ker":>6} {"L_w":>6} {"L_w/√A":>6} {"D_geo/√A":>8} {"D_link":>6} '
              f'{"V_mean":>6} {"w_min/√A":>8} {"s*ker":>6} {"s*wm":>6}  class')
    lines = [header, '-' * len(header)]
    for r in rows:
        lines.append(
            f'{r["polygon"]:<7} {r["n"]:>3} {r["reflex"]:>3} {r["essential"]:>3} '
            f'{f(r["guards"], 4, 0)} {f(r["edges"], 4, 0)} {f(r["ker_length"], 6, 0)} '
            f'{r["watchman_len"]:>6.0f} {r["watchman_norm"]:>6.2f} {r["geo_diam_norm"]:>8.2f} '
            f'{r["link_diameter"]:>6d} {r["mean_vis"]:>6.3f} {r["passage_norm"]:>8.3f} '
            f'{f(r["sstar_ker"], 6, 3)} {f(r["sstar_watchman"], 6, 3)}  {r["cls"]}')
    lines += ['', 'n vertices, r reflex vertices, m essential cuts; |J| guards, |E| patrol-graph '
                  'edges and L_ker roadmap length of the KER roadmap; L_w shortest watchman route; '
                  'D_geo geodesic diameter; D_link link diameter (vertex+grid estimate); V_mean mean '
                  'visibility fraction over the interior grid; w_min narrowest passage; s* critical '
                  f'speed ratio on the KER roadmap and on the watchman cycle. Grid {args.grid}x{args.grid}. '
                  'Lengths normalised by sqrt(area).']
    table = '\n'.join(lines)
    print('\n' + table)

    os.makedirs(RESULTS, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(RESULTS, f'{stamp}_polygon_stats.txt'), 'w') as fh:
        fh.write(table + '\n')
    with open(os.path.join(RESULTS, f'{stamp}_polygon_stats.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[SAVED] {RESULTS}/{stamp}_polygon_stats.txt and .csv')


if __name__ == '__main__':
    main()
