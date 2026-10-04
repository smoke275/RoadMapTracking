"""Pursuer tracking strategies for the benchmark (paper §7.3.2).

Each strategy picks a per-frame target on the patrol roadmap; the shared
StableNodeController then drives the pursuer toward it along the roadmap.
"""
import itertools
import math

from shapely.geometry import Point
from shapely.prepared import prep

from geometry import interpolate_point
from graph import dijkstra
from ker_pipeline import (_build_vis_shape, compute_optimal_guard,
                          compute_path_lengths, _opt_offset)
from pursuer_motion import StableNodeController
from benchmark.kernel_pursuer import KernelWeightedPursuer


def _evader_vis_prepared(evader_pos, data):
    """Prepared visibility polygon of the evader (buffered a hair), or None.
    A roadmap point covered by it has line of sight to the evader."""
    shape = _build_vis_shape(evader_pos[0], evader_pos[1], data.env)
    if shape is None:
        return None
    return prep(shape.buffer(0.1))


class MinMaxAlphaPursuer:
    """Proposed strategy: chase g* = argmin_q max_c alpha_c(q, e)."""

    name = 'minmax-alpha'

    def __init__(self, data, evader_start, speed):
        self.data = data
        self.speed = speed
        pl0 = compute_path_lengths(evader_start[0], evader_start[1], data)
        self.ctrl = StableNodeController(self._guard(pl0))

    def _guard(self, path_lengths):
        v1, v2, _ = compute_optimal_guard(path_lengths, self.data)
        g = interpolate_point(self.data.vertices[v1], self.data.vertices[v2],
                              _opt_offset(path_lengths, v1, v2, self.data))
        return g.x, g.y

    def target(self, evader_pos, path_lengths, cur_pos=None):
        return self._guard(path_lengths)

    def step(self, evader_pos, path_lengths):
        return self.ctrl.step(self.data.graph,
                              self.target(evader_pos, path_lengths),
                              self.speed)

    @property
    def pos(self):
        return self.ctrl.pos


class MinMaxAlphaVisPursuer(MinMaxAlphaPursuer):
    """Epsilon-slack vision layer (proposed, timing first + sight with the
    slack): if g* itself sees the evader, chase g*; otherwise chase the
    roadmap vertex with the lowest worst-corner alpha among those whose
    alpha is within (1 + EPS) of the optimum AND that currently see the
    evader; with no such vertex, fall back to g*. The single-pursuer
    analogue of the team ILP's relaxed speed bound."""

    name = 'minmax-alpha-vis'
    EPS = 0.10

    def target(self, evader_pos, path_lengths, cur_pos=None):
        v1, v2, opt_alpha = compute_optimal_guard(path_lengths, self.data)
        g = interpolate_point(self.data.vertices[v1], self.data.vertices[v2],
                              _opt_offset(path_lengths, v1, v2, self.data))
        gpos = (g.x, g.y)
        vp = _evader_vis_prepared(evader_pos, self.data)
        if vp is None or vp.covers(Point(gpos)):
            return gpos
        bound = opt_alpha * (1.0 + self.EPS)
        best = None
        for i, vtx in enumerate(self.data.vertices):
            vec = self.data.vectors_org[i]
            a = 0.0
            for c, length in path_lengths.items():
                r = vec[c] / length
                if r > a:
                    a = r
                    if a > bound:
                        break
            if a <= bound and vp.covers(Point((vtx.x, vtx.y))):
                if best is None or a < best[0]:
                    best = (a, (vtx.x, vtx.y))
        return best[1] if best is not None else gpos

    def step(self, evader_pos, path_lengths):
        return self.ctrl.step(self.data.graph,
                              self.target(evader_pos, path_lengths),
                              self.speed)


class GeoFollowPursuer:
    """Pure-pursuit baseline: drive to the roadmap vertex closest in geodesic
    distance to the evader. Candidates are narrowed to the K Euclidean-nearest
    roadmap vertices (Euclidean distance lower-bounds geodesic distance, so
    the geodesic-nearest vertex is almost surely among them), then ranked by
    exact geodesic distance.
    """

    name = 'geo-follow'
    K = 15

    def __init__(self, data, evader_start, speed):
        self.data = data
        self.speed = speed
        self._verts = [(v.x, v.y) for v in data.vertices]
        self.ctrl = StableNodeController(self._target(evader_start))

    def _target(self, evader_pos):
        ex, ey = evader_pos
        cands = sorted(self._verts,
                       key=lambda v: math.hypot(v[0] - ex, v[1] - ey))[:self.K]

        def geo_d(v):
            try:
                return self.data.geodesic.get_distance((ex, ey), v)
            except Exception:
                return math.hypot(v[0] - ex, v[1] - ey)

        return min(cands, key=geo_d)

    def target(self, evader_pos, path_lengths=None, cur_pos=None):
        return self._target(evader_pos)

    def step(self, evader_pos, path_lengths):
        return self.ctrl.step(self.data.graph, self._target(evader_pos),
                              self.speed)

    @property
    def pos(self):
        return self.ctrl.pos


class GreedyLOSPursuer:
    """Sight-only baseline: drive to the roadmap vertex that currently sees
    the evader and is closest to it (for a visible vertex the geodesic
    distance equals the Euclidean distance); with no visible vertex, fall
    back to the geodesically closest vertex to reacquire. Ignores the
    timing ratio entirely — the ablation counterpart of Min-Max."""

    name = 'greedy-los'
    K = 15                     # Euclidean prefilter for the geodesic fallback

    def __init__(self, data, evader_start, speed):
        self.data = data
        self.speed = speed
        self._verts = [(v.x, v.y) for v in data.vertices]
        self._geo_fallback = GeoFollowPursuer._target
        self.ctrl = StableNodeController(self._target(evader_start))

    def _target(self, evader_pos):
        ex, ey = evader_pos
        vp = _evader_vis_prepared(evader_pos, self.data)
        if vp is not None:
            visible = [v for v in self._verts if vp.covers(Point(v))]
            if visible:
                return min(visible,
                           key=lambda v: math.hypot(v[0] - ex, v[1] - ey))
        return self._geo_fallback(self, evader_pos)

    def target(self, evader_pos, path_lengths=None, cur_pos=None):
        return self._target(evader_pos)

    def step(self, evader_pos, path_lengths):
        return self.ctrl.step(self.data.graph, self._target(evader_pos),
                              self.speed)

    @property
    def pos(self):
        return self.ctrl.pos


class TSPPatrolPursuer:
    """Uninformed baseline: cyclically traverse a minimum-length TSP tour
    over the selected guard set, ignoring the evader entirely. The tour is
    solved exactly (guard sets are small, |S| <= ~9) over roadmap
    shortest-path distances.
    """

    name = 'tsp-patrol'
    ARRIVE_TOL = 1.0

    def __init__(self, data, evader_start, speed):
        self.data = data
        self.speed = speed
        self.tour = self._solve_tour([tuple(g) for g in data.guards])
        self._leg = 0
        self.ctrl = StableNodeController(self.tour[0])

    def _solve_tour(self, guards):
        n = len(guards)
        if n <= 1:
            return guards or [(0.0, 0.0)]
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d, _ = dijkstra(self.data.graph, guards[i], guards[j])
                if not math.isfinite(d):
                    d = math.hypot(guards[i][0] - guards[j][0],
                                   guards[i][1] - guards[j][1])
                dist[i][j] = dist[j][i] = d
        if n == 2:
            return list(guards)
        # Held-Karp DP over subsets: exact in O(2^n n^2), fine for n <= ~16
        # (the old brute force was O(n!), hopeless at poly8's 13 guards).
        FULL = 1 << (n - 1)              # subsets of {1..n-1}, city 0 fixed
        dp = [[math.inf] * (n - 1) for _ in range(FULL)]
        par = [[-1] * (n - 1) for _ in range(FULL)]
        for j in range(n - 1):
            dp[1 << j][j] = dist[0][j + 1]
        for mask in range(FULL):
            for j in range(n - 1):
                cur = dp[mask][j]
                if not math.isfinite(cur) or not (mask >> j) & 1:
                    continue
                for nxt in range(n - 1):
                    if (mask >> nxt) & 1:
                        continue
                    nm = mask | (1 << nxt)
                    cand = cur + dist[j + 1][nxt + 1]
                    if cand < dp[nm][nxt]:
                        dp[nm][nxt] = cand
                        par[nm][nxt] = j
        full = FULL - 1
        j = min(range(n - 1), key=lambda t: dp[full][t] + dist[t + 1][0])
        order, mask = [], full
        while j != -1:
            order.append(j + 1)
            j, mask = par[mask][j], mask ^ (1 << j)
        order.append(0)
        order.reverse()
        return [guards[i] for i in order]

    def target(self, evader_pos=None, path_lengths=None, cur_pos=None):
        """Current tour node, advancing the leg on arrival. cur_pos defaults
        to this pursuer's own controller position (benchmark use); the GUI
        passes its pursuer position explicitly."""
        px, py = cur_pos if cur_pos is not None else self.ctrl.pos
        tx, ty = self.tour[self._leg]
        if math.hypot(px - tx, py - ty) < self.ARRIVE_TOL:
            self._leg = (self._leg + 1) % len(self.tour)
            tx, ty = self.tour[self._leg]
        return (tx, ty)

    def step(self, evader_pos, path_lengths):
        return self.ctrl.step(self.data.graph, self.target(), self.speed)

    @property
    def pos(self):
        return self.ctrl.pos


PURSUER_CLASSES = {
    'minmax-alpha': MinMaxAlphaPursuer,
    'minmax-alpha-vis': MinMaxAlphaVisPursuer,
    'kernel-control': KernelWeightedPursuer,
    'geo-follow': GeoFollowPursuer,
    'greedy-los': GreedyLOSPursuer,
    'tsp-patrol': TSPPatrolPursuer,
}
