"""Evader behavior models for the headless benchmark harness."""
import math

import pyvisgraph as vg
import visilibity as vis
from shapely.geometry import Point

from config import EPSILON
from geometry import suppress_output
from skeleton import nearest_node, pick_destination, skeleton_path


class SkeletonEvader:
    """Wanders the Voronoi skeleton: walks a shortest path to a randomly
    picked distant node, then picks a new destination on arrival.

    Same segment-walking algorithm as the GUI's auto-evader (window.py),
    stripped of Qt state.
    """

    def __init__(self, skel_nodes, skel_adj, shapely_env, start_pos, speed):
        self.nodes = skel_nodes
        self.adj = skel_adj
        self.env = shapely_env
        self.pos = list(start_pos)
        self.speed = speed          # world units per second
        self._path: list = []
        self._seg_idx = 0
        self._seg_pos = 0.0

    def step(self, dt: float, pursuer_alphas: dict = None):
        """Advance up to speed*dt along the skeleton. Returns new (x, y).
        pursuer_alphas is ignored (this evader is pursuer-oblivious)."""
        budget = self.speed * dt
        while budget > 0:
            if not self._path or self._seg_idx >= len(self._path) - 1:
                cur = nearest_node(self.nodes, self.pos[0], self.pos[1])
                dst = pick_destination(self.nodes, self.adj, cur)
                new_path = skeleton_path(self.nodes, self.adj, cur, dst)
                if not new_path or len(new_path) <= 1:
                    break
                self._path = new_path
                self._seg_idx = 0
                self._seg_pos = 0.0

            i0 = self._path[self._seg_idx]
            i1 = self._path[self._seg_idx + 1]
            x0, y0 = self.nodes[i0]
            x1, y1 = self.nodes[i1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            remaining = seg_len - self._seg_pos

            if budget >= remaining:
                budget -= remaining
                self._seg_idx += 1
                self._seg_pos = 0.0
                nx, ny = x1, y1
            else:
                self._seg_pos += budget
                t = self._seg_pos / seg_len if seg_len > 0 else 0
                nx = x0 + t * (x1 - x0)
                ny = y0 + t * (y1 - y0)
                budget = 0

            if self.env.contains(Point(nx, ny)):
                self.pos = [nx, ny]

        return self.pos[0], self.pos[1]


class AdversarialEvader:
    """Targets the reflex corner with the poorest pursuer timing margin
    (c* = argmax_c alpha_c(p, e)) and moves geodesically toward it,
    retargeting every frame (paper §7.3.3). Stops within `standoff` of the
    corner so alpha stays bounded while breach detection (radius 1.5) can
    still trigger.
    """

    def __init__(self, data, start_pos, speed, standoff=0.5):
        self.data = data
        self.pos = list(start_pos)
        self.speed = speed
        self.standoff = standoff

    def _geo_path(self, target):
        try:
            sp = self.data.geodesic.shortest_path(self.pos, list(target))
        except KeyError:
            with suppress_output():
                raw = self.data.env.shortest_path(
                    vis.Point(self.pos[0], self.pos[1]),
                    vis.Point(target[0], target[1]), EPSILON)
            sp = [vg.Point(p.x(), p.y()) for p in raw.path()]
        return [(p.x, p.y) for p in sp]

    def step(self, dt, pursuer_alphas: dict = None):
        """Advance toward the currently worst-covered corner.
        pursuer_alphas — per-corner alpha at the pursuer's current position."""
        if not pursuer_alphas:
            return self.pos[0], self.pos[1]
        finite = {c: a for c, a in pursuer_alphas.items() if math.isfinite(a)}
        if not finite:
            return self.pos[0], self.pos[1]
        c_star = max(finite, key=finite.get)
        target = (self.data.poly[c_star].x(), self.data.poly[c_star].y())

        path = self._geo_path(target)
        budget = self.speed * dt
        px, py = self.pos
        for nx, ny in path[1:]:
            d_corner = math.hypot(target[0] - px, target[1] - py)
            if d_corner <= self.standoff or budget <= 0:
                break
            seg = math.hypot(nx - px, ny - py)
            if seg < 1e-9:
                continue
            step = min(budget, seg,
                       max(0.0, d_corner - self.standoff))
            px += (nx - px) / seg * step
            py += (ny - py) / seg * step
            budget -= step
            if step < seg:      # budget or standoff exhausted mid-segment
                break
        self.pos = [px, py]
        return px, py
