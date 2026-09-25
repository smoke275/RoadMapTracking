"""
Corner grouping for a k-pursuer team.

The single-pursuer optimiser (ker_pipeline.compute_optimal_guard) is
separable by corner: it builds one tent per reflex corner and takes the
minimax. Partition the corners into k groups and each group is an
independent single-pursuer problem on the same roadmap; the k-pursuer alpha
for an evader position is the max over groups of each group's own optimum.
So the multi-pursuer question is "which corners belong together?", and the
roadmap regions each pursuer ends up patrolling fall out afterwards.

Grouping signal — line-of-sight overlap (roadmap_los):
  For an evader position e and corner c, V(e, c) is the part of the roadmap
  that stays visible to the evader for its whole escape to c. The length of
  V(e, c1) ∩ V(e, c2) says how much roadmap a single pursuer could stand on
  and keep sight of the evader whichever of the two corners it runs to.
  Averaged over an evader grid it is a corner-pair affinity that encodes the
  visibility structure roadmap distance misses (two corners on either side
  of a thin wall are near on the roadmap but never co-watchable; two corners
  down one open hall are far apart but are).

Clustering: agglomerative (average linkage) on a distance derived from the
affinity, with a small roadmap-distance term to break ties between pairs
that never overlap anywhere. Cut into exactly k groups.

Evaluation / refinement: the oracle k-pursuer speed ratio is the max over an
evader grid of the max over groups of the group's optimal alpha (same
semantics as ker_pipeline.sweep_max_alpha). refine_groups does a greedy
local search moving one corner at a time between groups when that lowers
the worst case.
"""
from dataclasses import dataclass, field
import math
import os
import pickle

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from shapely.geometry import Point

from cache import poly_fingerprint
from config import GROUPS_FILE
from geometry import interpolate_point
from graph import dijkstra
from ker_pipeline import (FrameComputed, _opt_offset, compute_optimal_guard,
                          compute_path_lengths)
from roadmap_los import VisCache, lines_only, persistent_visible_all_corners

_OVERLAP_TOL   = 0.05   # buffer when intersecting two common-visible sets
_DIST_TIEBREAK = 0.05   # weight of the roadmap-distance term in the cluster distance


# ---------------------------------------------------------------------------
# Corner-pair affinity
# ---------------------------------------------------------------------------
def evader_grid(data, grid_n: int) -> list:
    """Interior points of a grid_n x grid_n grid over the polygon's bbox
    (same layout as ker_pipeline.sweep_max_alpha)."""
    minx, miny, maxx, maxy = data.shapely_env.bounds
    pts = []
    for i in range(grid_n):
        x = minx + (i + 0.5) * (maxx - minx) / grid_n
        for j in range(grid_n):
            y = miny + (j + 0.5) * (maxy - miny) / grid_n
            if data.shapely_env.contains(Point(x, y)):
                pts.append((x, y))
    return pts


def corner_affinity(data, grid_n: int = 10, step: float = 5.0,
                    progress=None) -> np.ndarray:
    """Mean over evader grid points of len(V(e, ci) ∩ V(e, cj)), indexed like
    data.corners. Diagonal = mean len(V(e, ci)). `progress(done, total)` is
    called after each grid point.

    V(e, c) is exact (roadmap_los.persistent_visible_roadmap): only the
    escape path's vertices are viewpoints, and every viewpoint is either a
    grid point or a reflex corner, so one VisCache serves the whole sweep —
    |grid| + |C| visibility polygons in total. `step` is accepted for
    signature compatibility and unused."""
    corners = list(data.corners)
    n = len(corners)
    aff = np.zeros((n, n))
    pts = evader_grid(data, grid_n)
    cache = VisCache(data)
    for k, (x, y) in enumerate(pts):
        vis = persistent_visible_all_corners(x, y, data, cache)
        commons = [vis[c] for c in corners]
        for i in range(n):
            aff[i, i] += commons[i].length
            if commons[i].is_empty:
                continue
            for j in range(i + 1, n):
                if commons[j].is_empty:
                    continue
                inter = lines_only(commons[i].intersection(commons[j].buffer(_OVERLAP_TOL)))
                aff[i, j] += inter.length
                aff[j, i] += inter.length
        if progress:
            progress(k + 1, len(pts))
    if pts:
        aff /= len(pts)
    return aff


def corner_roadmap_distance(data) -> np.ndarray:
    """Roadmap (patrol-graph) distance between corners' access points, indexed
    like data.corners. Uses the pipeline's node-to-corner distance table
    (vectors_org) from each corner's access vertices."""
    corners = list(data.corners)
    n = len(corners)

    def access_idx(c):
        out = []
        for pt in data.intersection_points.get(c, []):
            for vi, v in enumerate(data.vertices):
                if v.equals_exact(pt, 1e-6):
                    out.append(vi)
                    break
        return out

    acc = {c: access_idx(c) for c in corners}
    d = np.full((n, n), np.inf)
    np.fill_diagonal(d, 0.0)
    for i, ci in enumerate(corners):
        for j, cj in enumerate(corners):
            if i == j:
                continue
            best = math.inf
            for vi in acc[ci]:
                best = min(best, data.vectors_org[vi][cj])
            for vj in acc[cj]:
                best = min(best, data.vectors_org[vj][ci])
            d[i, j] = best
    finite = d[np.isfinite(d)]
    dmax = finite.max() if finite.size else 1.0
    d[~np.isfinite(d)] = dmax
    return d


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def cluster_corners(aff: np.ndarray, road_d: np.ndarray, corners: list, k: int) -> list:
    """Agglomerative clustering into exactly k groups. Returns a list of k
    lists of corner indices, ordered by their smallest corner."""
    n = len(corners)
    k = max(1, min(k, n))
    if k == 1:
        return [list(corners)]
    if k == n:
        return [[c] for c in corners]

    amax = aff.max() if aff.max() > 0 else 1.0
    dmax = road_d.max() if road_d.max() > 0 else 1.0
    dist = (1.0 - aff / amax) + _DIST_TIEBREAK * (road_d / dmax)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)

    Z = linkage(squareform(dist, checks=False), method='average')
    labels = cut_tree(Z, n_clusters=k).ravel()
    groups = {}
    for c, lab in zip(corners, labels):
        groups.setdefault(int(lab), []).append(c)
    return sorted(groups.values(), key=lambda g: min(g))


# ---------------------------------------------------------------------------
# Evaluation (oracle k-pursuer speed ratio) and refinement
# ---------------------------------------------------------------------------
def grid_path_lengths(data, grid_n: int) -> list:
    """[(x, y, path_lengths)] for every interior grid point — the expensive
    geodesic part of a sweep, computed once and reused across groupings."""
    return [(x, y, compute_path_lengths(x, y, data)) for x, y in evader_grid(data, grid_n)]


def evaluate_groups(data, groups: list, grid_pl: list) -> tuple:
    """Max over grid points of the k-pursuer alpha (max over groups of the
    group's optimal alpha). Returns (alpha, x, y, per_group_worst) where
    per_group_worst[i] is group i's own worst alpha over the grid."""
    worst = (-math.inf, None, None)
    per_group = [-math.inf] * len(groups)
    for x, y, pl in grid_pl:
        a_pt = -math.inf
        for gi, g in enumerate(groups):
            _, _, a = compute_optimal_guard(pl, data, corners=g)
            per_group[gi] = max(per_group[gi], a)
            a_pt = max(a_pt, a)
        if a_pt > worst[0]:
            worst = (a_pt, x, y)
    return worst[0], worst[1], worst[2], per_group


def refine_groups(data, groups: list, grid_pl: list, max_iter: int = 10,
                  log=print) -> list:
    """Greedy local search: move one corner to another group whenever that
    lowers the worst-case k-pursuer alpha. Groups never become empty."""
    groups = [list(g) for g in groups]
    best, *_ = evaluate_groups(data, groups, grid_pl)
    for it in range(max_iter):
        improved = False
        for gi, g in enumerate(groups):
            if len(g) <= 1:
                continue
            for c in list(g):
                for gj in range(len(groups)):
                    if gj == gi:
                        continue
                    cand = [list(x) for x in groups]
                    cand[gi].remove(c)
                    cand[gj].append(c)
                    a, *_ = evaluate_groups(data, cand, grid_pl)
                    if a < best - 1e-9:
                        log(f'[GROUPS] iter {it}: corner {c} {gi}->{gj}  alpha {best:.3f} -> {a:.3f}')
                        best, groups, improved = a, cand, True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return [sorted(g) for g in sorted(groups, key=lambda g: min(g))]


# ---------------------------------------------------------------------------
# Cache + top-level entry
# ---------------------------------------------------------------------------
@dataclass
class Grouping:
    k: int
    groups: list          # list of lists of corner indices
    affinity: np.ndarray
    road_dist: np.ndarray

    def group_of(self, corner: int) -> int:
        for i, g in enumerate(self.groups):
            if corner in g:
                return i
        return -1


def _cache_key(data, grid_n, step):
    # 'exact' tags the vertex-only V(e, c) computation; `step` no longer
    # influences the affinity (kept in the signature for callers).
    return (poly_fingerprint(data.poly), int(grid_n), 'exact')


def _load_cache() -> dict:
    try:
        if os.path.exists(GROUPS_FILE):
            with open(GROUPS_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f'[GROUPS] cache load failed: {e}')
    return {}


def _save_cache(blob: dict):
    try:
        os.makedirs(os.path.dirname(GROUPS_FILE), exist_ok=True)
        with open(GROUPS_FILE, 'wb') as f:
            pickle.dump(blob, f, protocol=4)
    except Exception as e:
        print(f'[GROUPS] cache save failed: {e}')


def get_affinity(data, grid_n: int, step: float, force: bool = False,
                 progress=None) -> tuple:
    """(affinity, road_dist) for this polygon, from the on-disk cache when the
    polygon and parameters match."""
    key = _cache_key(data, grid_n, step)
    blob = _load_cache()
    if not force and key in blob:
        print('[GROUPS] affinity loaded from cache.')
        return blob[key]['affinity'], blob[key]['road_dist']
    aff = corner_affinity(data, grid_n, step, progress=progress)
    road = corner_roadmap_distance(data)
    blob[key] = {'affinity': aff, 'road_dist': road}
    _save_cache(blob)
    return aff, road


# Partition sources, in order of preference when the GUI asks for k groups:
#   'ilp'      exact optimum from partition_ilp.py (run_groups.py, default)
#   'refined'  heuristic clustering + local search (run_groups.py --method heuristic --refine)
#   raw        heuristic clustering alone, computed on the spot
_METHOD_ORDER = ('ilp', 'refined')


def load_partition(data, grid_n: int, step: float, k: int, methods=_METHOD_ORDER):
    """(method, groups) saved by save_partition for this polygon and k, first
    available in `methods` order, or (None, None)."""
    entry = _load_cache().get(_cache_key(data, grid_n, step))
    if entry:
        for m in methods:
            if k in entry.get(m, {}):
                return m, entry[m][k]
    return None, None


def save_partition(data, grid_n: int, step: float, k: int, groups: list,
                   method: str = 'refined'):
    """Persist a partition so the GUI can use it instead of the raw clustering."""
    key = _cache_key(data, grid_n, step)
    blob = _load_cache()
    entry = blob.setdefault(key, {})
    entry.setdefault(method, {})[k] = [list(g) for g in groups]
    _save_cache(blob)


# Backwards-compatible names
def load_refined(data, grid_n, step, k):
    return load_partition(data, grid_n, step, k, methods=('refined',))[1]


def save_refined(data, grid_n, step, k, groups):
    save_partition(data, grid_n, step, k, groups, method='refined')


def make_grouping(data, k: int, grid_n: int, step: float,
                  force: bool = False, progress=None,
                  use_refined: bool = True) -> Grouping:
    """Return k corner groups: a stored partition (ILP optimum preferred,
    then the refined heuristic) if run_groups.py has produced one for this
    polygon and k, else the raw affinity clustering."""
    aff, road = get_affinity(data, grid_n, step, force=force, progress=progress)
    method, stored = load_partition(data, grid_n, step, k) if use_refined else (None, None)
    if stored is not None:
        print(f'[GROUPS] using stored {method} partition for k={k}.')
        groups = [sorted(g) for g in stored]
    else:
        groups = cluster_corners(aff, road, list(data.corners), k)
    return Grouping(k=len(groups), groups=groups, affinity=aff, road_dist=road)


def format_affinity(g: Grouping, corners: list) -> str:
    w = max(4, max(len(str(c)) for c in corners) + 1)
    lines = [' ' * w + ''.join(f'{c:>{w}}' for c in corners)]
    for i, ci in enumerate(corners):
        lines.append(f'{ci:>{w}}' + ''.join(f'{g.affinity[i, j]:>{w}.0f}' for j in range(len(corners))))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Per-frame computation for k pursuers
# ---------------------------------------------------------------------------
@dataclass
class MultiFrame:
    path_lengths: dict    # corner -> L_c = min over evaders of geodesic distance
    frames: list          # FrameComputed per group (guard, opt alpha, route)
    combined_alphas: dict # corner -> min over pursuers of d_G(p_i, c) / L_c
    team_opt_alpha: float # max over groups of group opt alpha (oracle)
    achieved_alpha: float # max over corners of combined_alphas (actual positions)
    evaders: list = field(default_factory=list)          # [(x, y), ...]
    per_evader_pl: list = field(default_factory=list)    # path_lengths dict per evader
    nearest_evader: dict = field(default_factory=dict)   # corner -> index of the evader defining L_c


def compute_path_lengths_multi(evaders: list, data) -> tuple:
    """Nearest-evader geodesics. For m evaders the pursuer must beat the one
    that can reach each corner first, so L_c = min_j d_geo(e_j, c) and the
    single-evader optimiser applies unchanged. Returns
    (path_lengths, per_evader, nearest) with per_evader[j] the j-th evader's
    own dict and nearest[c] the index attaining the minimum."""
    per_evader = [compute_path_lengths(x, y, data) for x, y in evaders]
    pl, nearest = {}, {}
    for c in data.corners:
        j = min(range(len(evaders)), key=lambda j: per_evader[j][c])
        pl[c], nearest[c] = per_evader[j][c], j
    return pl, per_evader, nearest


def _alphas_at(pos, path_lengths, data) -> dict:
    """Paper Eq. 1 alpha per corner at a roadmap position (same construction
    as benchmark.metrics.alphas_at, inlined to avoid importing the benchmark
    package from the GUI)."""
    px, py = pos
    best = None
    for v1, v2 in data.total_edges:
        a, b = data.vertices[v1], data.vertices[v2]
        abx, aby = b.x - a.x, b.y - a.y
        seg2 = abx * abx + aby * aby
        t = 0.0 if seg2 < 1e-12 else max(0.0, min(1.0, ((px - a.x) * abx + (py - a.y) * aby) / seg2))
        d = math.hypot(px - (a.x + t * abx), py - (a.y + t * aby))
        if best is None or d < best[0]:
            best = (d, v1, v2)
    _, u, w = best
    d_u = math.hypot(px - data.vertices[u].x, py - data.vertices[u].y)
    d_w = math.hypot(px - data.vertices[w].x, py - data.vertices[w].y)
    return {c: min(d_u + data.vectors_org[u][c], d_w + data.vectors_org[w][c]) / path_lengths[c]
            for c in data.corners}


def compute_frame_multi(ex, ey, positions: list, data, groups: list) -> MultiFrame:
    """k pursuers at `positions`, corner partition `groups`, and either one
    evader (ex, ey as floats) or several (ex = [(x, y), ...], ey ignored).
    With several evaders each corner uses its nearest one (see
    compute_path_lengths_multi); everything else is the single-evader path."""
    evaders = list(ex) if isinstance(ex, (list, tuple)) else [(float(ex), float(ey))]
    pl, per_evader, nearest = compute_path_lengths_multi(evaders, data)
    frames = []
    for (px, py), g in zip(positions, groups):
        v1, v2, a = compute_optimal_guard(pl, data, corners=g)
        guard = interpolate_point(data.vertices[v1], data.vertices[v2],
                                  _opt_offset(pl, v1, v2, data, corners=g))
        _, route = dijkstra(data.graph, (px, py), Point(guard.x, guard.y))
        frames.append(FrameComputed(guard=guard, opt_edge_v1=v1, opt_edge_v2=v2,
                                    opt_alpha=a, path_lengths=pl, alphas=[],
                                    obs_to_guard_path=route))
    per_pursuer = [_alphas_at(p, pl, data) for p in positions]
    combined = {c: min(a[c] for a in per_pursuer) for c in data.corners}
    finite = [v for v in combined.values() if math.isfinite(v)]
    return MultiFrame(
        path_lengths=pl,
        frames=frames,
        combined_alphas=combined,
        team_opt_alpha=max(f.opt_alpha for f in frames),
        achieved_alpha=max(finite) if finite else math.inf,
        evaders=evaders,
        per_evader_pl=per_evader,
        nearest_evader=nearest,
    )
