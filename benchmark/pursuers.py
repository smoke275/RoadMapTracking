"""Pursuer tracking strategies for the benchmark (paper §7.3.2).

Each strategy picks a per-frame target on the patrol roadmap; the shared
StableNodeController then drives the pursuer toward it along the roadmap.
"""
import itertools
import math

from geometry import interpolate_point
from graph import dijkstra
from ker_pipeline import compute_optimal_guard, compute_path_lengths, _opt_offset
from pursuer_motion import StableNodeController
from benchmark.kernel_pursuer import KernelWeightedPursuer


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
        best_perm, best_cost = None, float('inf')
        for perm in itertools.permutations(range(1, n)):
            order = (0,) + perm
            cost = sum(dist[order[k]][order[(k + 1) % n]] for k in range(n))
            if cost < best_cost:
                best_cost, best_perm = cost, order
        return [guards[i] for i in best_perm]

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
    'kernel-control': KernelWeightedPursuer,
    'geo-follow': GeoFollowPursuer,
    'tsp-patrol': TSPPatrolPursuer,
}
