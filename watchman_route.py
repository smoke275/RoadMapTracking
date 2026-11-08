"""Shortest watchman route in a simple polygon.

A *watchman route* is a closed curve from which every point of the polygon
is seen at some moment.  For a simple polygon the shortest one is
characterised by its *essential cuts* (Chin & Ntafos 1988; Carlsson,
Jonsson & Nilsson 1999; Tan 2001; Dror, Efrat, Lubiw & Mitchell 2003):

* At every reflex vertex v, extend each of the two incident edges through v
  until it meets the boundary.  Each extension is a *cut*; it splits the
  polygon into two parts, and the *pocket* of the cut is the part that
  contains the other incident edge (the region hidden behind the wall).
* A closed route sees the whole polygon iff it touches the closed pocket of
  every cut.  If pocket(c1) is contained in pocket(c2), touching c1's pocket
  already touches c2's, so only the cuts with minimal pockets, the
  *essential cuts*, constrain the route.
* The optimal route touches the essential cuts in the order in which their
  pockets appear along the boundary, and between consecutive touch points it
  follows a geodesic shortest path.  Finding the optimal touch points is the
  "touring a sequence of segments" problem.

We implement that structure exactly (cuts, pockets, dominance, ordering,
geodesic legs) and solve the touring step numerically: a dynamic program
over sampled points on the ordered cuts gives the best closed tour on the
samples, and coordinate descent with progressive zoom then moves every touch
point continuously along its cut until the length stops improving.  On the
benchmark polygons (10-29 reflex corners) this takes a few seconds and
converges to the continuous optimum to well below a hundredth of a world
unit.

Geodesic distances between arbitrary points are computed through the vertex
visibility graph (a shortest path in a simple polygon bends only at reflex
vertices), with Floyd-Warshall over the vertices done once per polygon.

    python watchman_route.py --skip-draw              # polygon from config.FILE_NAME
    python watchman_route.py --polygon resources/sites_poly9.csv --png out.png
"""
import argparse
import hashlib
import math
import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.sparse.csgraph import floyd_warshall
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon
from shapely.geometry.polygon import orient

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE  = os.path.join(_SCRIPT_DIR, 'resources', 'watchman_cache.pkl')

_COVER_EPS   = 1e-6    # tolerance so boundary-grazing segments count as inside
_RAY_START   = 1e-3    # cut rays start this far past the reflex vertex
_MIN_CUT_LEN = 1e-6    # cuts shorter than this are degenerate and dropped
_POCKET_TOL  = 1e-6    # tolerance for pocket containment along the boundary


# ---------------------------------------------------------------------------
# Geodesic distances through the vertex visibility graph
# ---------------------------------------------------------------------------
class PolygonGeodesics:
    """Shortest-path distances inside a simple polygon, vectorised over point
    sets.  Vertices are stored counter-clockwise."""

    def __init__(self, pts):
        poly = orient(ShapelyPolygon([(float(x), float(y)) for x, y in pts]), sign=1.0)
        if not poly.is_valid:
            raise ValueError('polygon is not simple')
        self.poly  = poly
        self.ring  = poly.exterior
        self.V     = np.asarray(self.ring.coords[:-1], dtype=float)
        self.n     = len(self.V)
        self.cover = poly.buffer(_COVER_EPS)

        I, J = np.triu_indices(self.n, 1)
        seen = self._covers(self.V[I], self.V[J])
        W = np.full((self.n, self.n), np.inf)
        d = np.linalg.norm(self.V[I] - self.V[J], axis=1)
        W[I[seen], J[seen]] = d[seen]
        W[J[seen], I[seen]] = d[seen]
        np.fill_diagonal(W, 0.0)
        self.D, self.pred = floyd_warshall(W, directed=False, return_predecessors=True)

    # -- primitive visibility -------------------------------------------------
    def _covers(self, A, B):
        """Elementwise: does the polygon contain segment A[i]B[i]?"""
        A = np.asarray(A, dtype=float).reshape(-1, 2)
        B = np.asarray(B, dtype=float).reshape(-1, 2)
        segs = shapely.linestrings(np.stack([A, B], axis=1))
        return shapely.covers(self.cover, segs)

    def visible_vertices(self, P):
        """Boolean (len(P), n): which polygon vertices each point sees."""
        P = np.asarray(P, dtype=float).reshape(-1, 2)
        A = np.repeat(P, self.n, axis=0)
        B = np.tile(self.V, (len(P), 1))
        return self._covers(A, B).reshape(len(P), self.n)

    # -- distances --------------------------------------------------------------
    def _through_vertices(self, A, visA, B, visB):
        """(len(A), len(B)) geodesic lengths that bend at >= 1 vertex, and the
        (first, last) vertex of the best such path for each pair."""
        dA = np.linalg.norm(A[:, None, :] - self.V[None, :, :], axis=2)   # (a, n)
        dB = np.linalg.norm(B[:, None, :] - self.V[None, :, :], axis=2)   # (b, n)
        dA = np.where(visA, dA, np.inf)
        dB = np.where(visB, dB, np.inf)
        # tb[b, u] = min_v D[u, v] + |v b|
        TB = self.D[None, :, :] + dB[:, None, :]                              # (b, n, n)
        tb_v = np.argmin(TB, axis=2)                                           # (b, n)
        tb   = np.take_along_axis(TB, tb_v[:, :, None], axis=2)[:, :, 0]      # (b, n)
        tot  = dA[:, None, :] + tb[None, :, :]                                 # (a, b, n)
        u    = np.argmin(tot, axis=2)                                          # (a, b)
        best = np.take_along_axis(tot, u[:, :, None], axis=2)[:, :, 0]
        v    = tb_v[np.arange(len(B))[None, :], u]                             # (a, b)
        return best, u, v

    def distances(self, A, B, visA=None, visB=None):
        """(len(A), len(B)) geodesic distances between two point sets."""
        A = np.asarray(A, dtype=float).reshape(-1, 2)
        B = np.asarray(B, dtype=float).reshape(-1, 2)
        if visA is None:
            visA = self.visible_vertices(A)
        if visB is None:
            visB = self.visible_vertices(B)
        a, b = len(A), len(B)
        direct = self._covers(np.repeat(A, b, axis=0), np.tile(B, (a, 1))).reshape(a, b)
        eucl   = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        bent, _, _ = self._through_vertices(A, visA, B, visB)
        return np.where(direct, eucl, bent)

    def path(self, a, b):
        """Geodesic shortest path a -> b as a list of (x, y) including both ends."""
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        if self._covers(a, b)[0]:
            return [tuple(a), tuple(b)]
        visA = self.visible_vertices(a); visB = self.visible_vertices(b)
        _, u, v = self._through_vertices(a[None], visA, b[None], visB)
        u, v = int(u[0, 0]), int(v[0, 0])
        chain = [v]
        while chain[-1] != u:
            chain.append(int(self.pred[u, chain[-1]]))
        chain.reverse()
        return [tuple(a)] + [tuple(self.V[k]) for k in chain] + [tuple(b)]


# ---------------------------------------------------------------------------
# Cuts and pockets
# ---------------------------------------------------------------------------
@dataclass
class Cut:
    vertex:     int            # reflex vertex index (ccw numbering of PolygonGeodesics.V)
    p0:         tuple          # the reflex vertex
    p1:         tuple          # where the extension meets the boundary
    pocket_s:   float          # boundary parameter where the pocket chain starts (ccw)
    pocket_len: float          # length of the pocket's boundary chain
    essential:  bool = False

    def point(self, t: float) -> tuple:
        return (self.p0[0] + t * (self.p1[0] - self.p0[0]),
                self.p0[1] + t * (self.p1[1] - self.p0[1]))

    @property
    def length(self) -> float:
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])


def reflex_vertices(V) -> list:
    """Indices of reflex vertices of a counter-clockwise polygon."""
    n = len(V); out = []
    for i in range(n):
        a, b, c = V[i - 1], V[i], V[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross < -1e-9:          # right turn in a ccw polygon
            out.append(i)
    return out


def _first_boundary_hit(geo: PolygonGeodesics, origin, direction):
    """First point where the ray origin + t*direction (t > 0) meets the boundary."""
    R = 4.0 * (geo.ring.length + 1.0)
    o = np.asarray(origin, dtype=float); d = np.asarray(direction, dtype=float)
    ray = LineString([tuple(o + _RAY_START * d), tuple(o + R * d)])
    hit = ray.intersection(geo.ring)
    if hit.is_empty:
        return None
    pts = shapely.get_coordinates(hit)
    if len(pts) == 0:
        return None
    k = np.argmin(np.linalg.norm(pts - o, axis=1))
    return (float(pts[k][0]), float(pts[k][1]))


def all_cuts(geo: PolygonGeodesics) -> list:
    """Both edge-extension cuts at every reflex vertex, with their pockets."""
    V, n, ring = geo.V, geo.n, geo.ring
    L = ring.length
    cuts = []
    for i in reflex_vertices(V):
        v = V[i]
        t_v = ring.project(Point(v))
        for side in (-1, +1):
            nb = V[(i + side) % n]
            d  = v - nb
            d  = d / np.linalg.norm(d)
            hit = _first_boundary_hit(geo, v, d)
            if hit is None or math.hypot(hit[0] - v[0], hit[1] - v[1]) < _MIN_CUT_LEN:
                continue
            t_p = ring.project(Point(hit))
            if side == -1:
                # extension of the incoming edge: pocket is the chain v -> ... -> hit (ccw)
                s, ln = t_v, (t_p - t_v) % L
            else:
                # extension of the outgoing edge: pocket is the chain hit -> ... -> v (ccw)
                s, ln = t_p, (t_v - t_p) % L
            cuts.append(Cut(i, (float(v[0]), float(v[1])), hit, float(s), float(ln)))
    return cuts


def _contains(outer: Cut, inner: Cut, L: float) -> bool:
    """Is inner's pocket chain inside outer's (cyclic boundary intervals)?"""
    off = (inner.pocket_s - outer.pocket_s) % L
    if off > L - _POCKET_TOL:
        off = 0.0
    return off + inner.pocket_len <= outer.pocket_len + _POCKET_TOL


def essential_cuts(cuts: list, L: float) -> list:
    """Cuts whose pocket contains no other cut's pocket, in boundary order."""
    ess = []
    for i, c in enumerate(cuts):
        dominated = False
        for j, o in enumerate(cuts):
            if i == j:
                continue
            if _contains(c, o, L):
                same = _contains(o, c, L)
                if not same or j < i:      # strictly smaller, or an equal duplicate
                    dominated = True
                    break
        c.essential = not dominated
        if c.essential:
            ess.append(c)
    ess.sort(key=lambda c: c.pocket_s)
    return ess


# ---------------------------------------------------------------------------
# Shortest closed tour touching the essential cuts in order
# ---------------------------------------------------------------------------
@dataclass
class WatchmanRoute:
    touch:     list                      # (x, y) touch point per essential cut, in order
    params:    list                      # position t in [0, 1] along each cut
    route:     list                      # closed polyline (x, y); last point == first
    length:    float
    cuts:      list                      # essential cuts, in order
    all_cuts:  list = field(default_factory=list)
    seconds:   float = 0.0

    def point_at(self, s: float) -> tuple:
        """Point at arc length s (mod length) along the closed route."""
        if self.length <= 0 or len(self.route) < 2:
            return tuple(self.route[0])
        s = s % self.length
        for (x0, y0), (x1, y1) in zip(self.route[:-1], self.route[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            if s <= seg:
                f = s / seg if seg > 0 else 0.0
                return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
            s -= seg
        return tuple(self.route[-1])

    def nearest(self, p) -> tuple:
        """(arc length, point) of the route point closest to p (Euclidean)."""
        best_s, best_pt, best_d, acc = 0.0, tuple(self.route[0]), float('inf'), 0.0
        for (x0, y0), (x1, y1) in zip(self.route[:-1], self.route[1:]):
            dx, dy = x1 - x0, y1 - y0
            seg = math.hypot(dx, dy)
            f = 0.0 if seg == 0 else max(0.0, min(1.0, ((p[0] - x0) * dx + (p[1] - y0) * dy) / seg ** 2))
            qx, qy = x0 + f * dx, y0 + f * dy
            d = math.hypot(p[0] - qx, p[1] - qy)
            if d < best_d:
                best_d, best_s, best_pt = d, acc + f * seg, (qx, qy)
            acc += seg
        return best_s, best_pt


def _closed_tour_dp(M: list):
    """M[i] is the (K_i x K_{i+1}) cost matrix from samples on cut i to samples
    on cut i+1 (cyclically; the last matrix closes back to cut 0).  Returns
    the cheapest closed tour as one index per cut."""
    m = len(M)
    K0 = M[0].shape[0]
    dp = M[0].copy()                               # (K0 starts, K1)
    back = []
    for i in range(1, m - 1):
        tot = dp[:, :, None] + M[i][None, :, :]    # (K0, K_i, K_{i+1})
        arg = np.argmin(tot, axis=1)               # (K0, K_{i+1})
        dp  = np.take_along_axis(tot, arg[:, None, :], axis=1)[:, 0, :]
        back.append(arg)
    # close: tour[s] = min_j dp[s, j] + M[m-1][j, s]
    close = dp + M[m - 1].T                        # (K0, K_{m-1})
    s = int(np.argmin(np.min(close, axis=1)))
    j = int(np.argmin(close[s]))
    best = float(close[s, j])
    idx = [j]
    for arg in reversed(back):
        idx.append(int(arg[s, idx[-1]]))
    idx.append(s)
    idx.reverse()                                  # [s, i_1, ..., i_{m-1}]
    return idx, best


def _refine(geo: PolygonGeodesics, cuts: list, params: list, passes: int = 30,
            tol: float = 1e-7):
    """Coordinate descent: move each touch point along its cut to minimise the
    two adjacent legs, zooming the search window three times."""
    m = len(cuts)
    params = list(params)

    def tour_length(ps):
        P = np.array([cuts[i].point(ps[i]) for i in range(m)])
        return float(sum(geo.distances(P[i][None], P[(i + 1) % m][None])[0, 0]
                         for i in range(m)))

    best = tour_length(params)
    for _ in range(passes):
        improved = False
        for i in range(m):
            a = np.array(cuts[(i - 1) % m].point(params[(i - 1) % m]))[None]
            b = np.array(cuts[(i + 1) % m].point(params[(i + 1) % m]))[None]
            visA = geo.visible_vertices(a); visB = geo.visible_vertices(b)
            lo, hi, t_best = 0.0, 1.0, params[i]
            f_best = None
            for level, nsamp in enumerate((65, 33, 33, 33)):
                ts = np.linspace(lo, hi, nsamp)
                C  = np.array([cuts[i].point(t) for t in ts])
                visC = geo.visible_vertices(C)
                f = geo.distances(a, C, visA, visC)[0] + geo.distances(C, b, visC, visB)[:, 0]
                k = int(np.argmin(f))
                if f_best is None or f[k] < f_best - 1e-12:
                    f_best, t_best = float(f[k]), float(ts[k])
                step = (hi - lo) / (nsamp - 1)
                lo, hi = max(0.0, t_best - step), min(1.0, t_best + step)
            if abs(t_best - params[i]) > 0:
                params[i] = t_best
        new = tour_length(params)
        if new < best - tol:
            improved = True
        best = min(best, new)
        if not improved:
            break
    return params, best


def shortest_watchman_route(pts, samples: int = 24, refine: bool = True) -> WatchmanRoute:
    """Shortest closed route from which all of the simple polygon `pts`
    (list of (x, y)) is seen.  See the module docstring."""
    t0 = time.perf_counter()
    geo = PolygonGeodesics(pts)
    cuts = all_cuts(geo)
    ess  = essential_cuts(cuts, geo.ring.length)
    m = len(ess)

    if m == 0:                                   # convex: any point sees everything
        c = geo.poly.representative_point()
        p = (float(c.x), float(c.y))
        return WatchmanRoute([p], [], [p, p], 0.0, [], cuts, time.perf_counter() - t0)
    if m == 1:                                   # one essential cut: stand on it
        p = ess[0].point(0.5)
        return WatchmanRoute([p], [0.5], [p, p], 0.0, ess, cuts, time.perf_counter() - t0)

    # -- dynamic program over sampled cut points ----------------------------------
    ts = np.linspace(0.0, 1.0, samples)
    S  = [np.array([c.point(t) for t in ts]) for c in ess]
    vis = [geo.visible_vertices(s) for s in S]
    M  = [geo.distances(S[i], S[(i + 1) % m], vis[i], vis[(i + 1) % m]) for i in range(m)]
    idx, length = _closed_tour_dp(M)
    params = [float(ts[k]) for k in idx]

    # -- continuous refinement ------------------------------------------------------
    if refine:
        params, length = _refine(geo, ess, params)

    touch = [ess[i].point(params[i]) for i in range(m)]
    route = []
    for i in range(m):
        leg = geo.path(touch[i], touch[(i + 1) % m])
        route.extend(leg[:-1])
    route.append(route[0])
    length = sum(math.hypot(x1 - x0, y1 - y0)
                 for (x0, y0), (x1, y1) in zip(route[:-1], route[1:]))
    return WatchmanRoute(touch, params, route, float(length), ess, cuts,
                         time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Cache keyed by polygon, and the entry point used by the GUI
# ---------------------------------------------------------------------------
def _fingerprint(pts) -> str:
    h = hashlib.sha256()
    for x, y in pts:
        h.update(f'{float(x):.6f},{float(y):.6f};'.encode())
    return h.hexdigest()


def get_watchman_route(pts, force: bool = False, **kw) -> WatchmanRoute:
    """Cached shortest_watchman_route for a list of (x, y) or vis.Point."""
    pts = [(p.x(), p.y()) if hasattr(p, 'x') and callable(p.x) else (p[0], p[1]) for p in pts]
    key = _fingerprint(pts)
    blob = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                blob = pickle.load(f)
        except Exception:
            blob = {}
    if not force and key in blob:
        return blob[key]
    wr = shortest_watchman_route(pts, **kw)
    blob[key] = wr
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(blob, f)
    except OSError as e:
        print(f'[WATCHMAN] cache not written: {e}')
    return wr


# ---------------------------------------------------------------------------
# Verification and plotting
# ---------------------------------------------------------------------------
def coverage_fraction(pts, route: list, step: float = 2.0) -> float:
    """Fraction of polygon area seen from points sampled every `step` units
    along the route (exact visibility polygons via VisiLibity)."""
    import visilibity as vis
    from geometry import poly_to_points, suppress_output
    from config import EPSILON
    poly = ShapelyPolygon(pts)
    walls = vis.Polygon([vis.Point(float(x), float(y)) for x, y in pts])
    if walls.area() < 0:
        walls.reverse()
    env = vis.Environment([walls])
    env.PRINTING_DEBUG_DATA = False
    # sample along the route
    samples = []
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(route[:-1], route[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        s = 0.0
        while s <= seg:
            f = s / seg if seg > 0 else 0.0
            samples.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
            s += step
        acc += seg
    samples.append(tuple(route[-1]))
    # pull sample points a hair inside so VisiLibity accepts boundary points
    inner = poly.buffer(-1e-3)
    shapes = []
    for (x, y) in samples:
        p = Point(x, y)
        if not inner.covers(p):
            q = inner.exterior.interpolate(inner.exterior.project(p))
            x, y = q.x, q.y
        with suppress_output():
            vp = vis.Visibility_Polygon(vis.Point(x, y), env, EPSILON)
        vx, vy = poly_to_points(vp)
        if len(vx) >= 3:
            shp = ShapelyPolygon(list(zip(vx, vy)))
            shapes.append(shp if shp.is_valid else shp.buffer(0))
    seen = shapely.unary_union(shapes).intersection(poly)
    return seen.area / poly.area


def plot_route(pts, wr: WatchmanRoute, png: str, title: str = ''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.fill(xs, ys, color='#eef2fb', zorder=0)
    ax.plot(xs, ys, color='#30405a', lw=2, zorder=3)
    for c in wr.all_cuts:
        if not c.essential:
            ax.plot([c.p0[0], c.p1[0]], [c.p0[1], c.p1[1]], color='#b0b8c8', lw=0.8,
                    ls=':', zorder=1)
    for k, c in enumerate(wr.cuts):
        ax.plot([c.p0[0], c.p1[0]], [c.p0[1], c.p1[1]], color='#d08a00', lw=1.4,
                ls='--', zorder=2)
        ax.annotate(str(k + 1), c.p1, fontsize=8, color='#d08a00',
                    xytext=(3, 3), textcoords='offset points')
    rx = [p[0] for p in wr.route]; ry = [p[1] for p in wr.route]
    ax.plot(rx, ry, color='#00897b', lw=3, zorder=4,
            label=f'shortest watchman route, L = {wr.length:.1f}')
    tx = [p[0] for p in wr.touch]; ty = [p[1] for p in wr.touch]
    ax.scatter(tx, ty, s=40, color='#00897b', edgecolor='white', zorder=5,
               label=f'{len(wr.cuts)} essential cuts touched')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or 'Shortest watchman route')
    ax.legend(loc='lower left', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def _load_csv(path):
    import csv
    pts = []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip():
                pts.append((float(row[0]), float(row[1])))
    # same cleaning as the pipeline: drop consecutive duplicates, closing point
    clean = []
    for p in pts:
        if not clean or p != clean[-1]:
            clean.append(p)
    while len(clean) > 1 and clean[-1] == clean[0]:
        clean.pop()
    return clean


def main():
    ap = argparse.ArgumentParser(description='Shortest watchman route of a simple polygon')
    ap.add_argument('--polygon', help='CSV of x,y vertices (default: config.FILE_NAME)')
    ap.add_argument('-s', '--skip-draw', action='store_true',
                    help='accepted for symmetry with main.py; the CSV is always loaded')
    ap.add_argument('--samples', type=int, default=24, help='DP samples per cut')
    ap.add_argument('--no-refine', action='store_true', help='skip continuous refinement')
    ap.add_argument('--png', help='write a figure of the route')
    ap.add_argument('--check', action='store_true',
                    help='verify coverage with VisiLibity visibility polygons along the route')
    ap.add_argument('-r', '--recompute', action='store_true', help='ignore the cache')
    args = ap.parse_args()

    if args.polygon is None:
        from config import FILE_NAME
        args.polygon = FILE_NAME
    pts = _load_csv(args.polygon)
    name = os.path.splitext(os.path.basename(args.polygon))[0]

    wr = get_watchman_route(pts, force=args.recompute, samples=args.samples,
                            refine=not args.no_refine)
    nref = len(reflex_vertices(PolygonGeodesics(pts).V))
    print(f'[WATCHMAN] {name}: {len(pts)} vertices, {nref} reflex, '
          f'{len(wr.all_cuts)} cuts, {len(wr.cuts)} essential')
    print(f'[WATCHMAN] route length {wr.length:.3f} through {len(wr.route) - 1} segments '
          f'({wr.seconds:.1f} s)')
    for k, (c, t, p) in enumerate(zip(wr.cuts, wr.params, wr.touch)):
        print(f'   cut {k + 1:2d}: reflex v{c.vertex:<3d} len {c.length:7.2f}  '
              f'touch t={t:.4f} at ({p[0]:8.2f}, {p[1]:8.2f})')
    if args.check:
        cov = coverage_fraction(pts, wr.route)
        print(f'[WATCHMAN] visibility coverage along the route: {100 * cov:.3f}% of area')
    if args.png:
        plot_route(pts, wr, args.png, title=f'{name}: shortest watchman route (L = {wr.length:.1f})')
        print(f'[WATCHMAN] figure written to {args.png}')


if __name__ == '__main__':
    main()
