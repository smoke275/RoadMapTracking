"""Evader behavior models for the headless benchmark harness."""
from collections import deque
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

    avoid_recent — remember this many past destinations and exclude them
    (and their neighbours) when picking the next one. Without it the
    "farthest node" rule ping-pongs between the same two spots (see
    skeleton.pick_destination). 0 keeps the historical benchmark behaviour;
    the GUI uses 3 like window.py.
    """

    def __init__(self, skel_nodes, skel_adj, shapely_env, start_pos, speed,
                 avoid_recent: int = 0):
        self.nodes = skel_nodes
        self.adj = skel_adj
        self.env = shapely_env
        self.pos = list(start_pos)
        self.speed = speed          # world units per second
        self._path: list = []
        self._seg_idx = 0
        self._seg_pos = 0.0
        self._dest_history = deque(maxlen=avoid_recent) if avoid_recent > 0 else None

    def step(self, dt: float, pursuer_alphas: dict = None,
             pursuer_pos=None):
        """Advance up to speed*dt along the skeleton. Returns new (x, y).
        pursuer_alphas/pursuer_pos are ignored (pursuer-oblivious)."""
        budget = self.speed * dt
        while budget > 0:
            if not self._path or self._seg_idx >= len(self._path) - 1:
                cur = nearest_node(self.nodes, self.pos[0], self.pos[1])
                dst = pick_destination(self.nodes, self.adj, cur,
                                       avoid=self._dest_history)
                new_path = skeleton_path(self.nodes, self.adj, cur, dst)
                if not new_path or len(new_path) <= 1:
                    break
                if self._dest_history is not None:
                    self._dest_history.append(dst)
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


class EscapingEvader:
    """The threat of the necessity lemma, made executable: whenever the
    evader wins a corner race (alpha_c > s_p/s_e for some corner), it
    COMMITS to that corner, runs its geodesic to it, and on arrival steps
    into the shadow of the corner with respect to the pursuer's current
    position (the extension of the pursuer->corner ray, the escape
    direction Lemma 1 guarantees exists when the pursuer is outside the
    association region). It holds there briefly, then re-targets. When no
    corner race is won, it pressures the currently worst-covered corner
    like AdversarialEvader, at a standoff.

    `ratio` is s_p/s_e: the evader knows the speeds, the strongest fair
    adversary consistent with the paper's information model.
    """

    HOLD_FRAMES = 45          # dwell in the shadow before re-targeting
    SHADOW_DEPTH = 3.0        # how far past the corner to step
    ARRIVE = 0.8

    def __init__(self, data, start_pos, speed, ratio=0.8, standoff=0.5):
        self.data = data
        self.pos = list(start_pos)
        self.speed = speed
        self.ratio = ratio
        self.standoff = standoff
        self._committed = None         # corner index, or None
        self._shadow_target = None
        self._hold = 0
        self._recent = deque(maxlen=2)

    def _shadow_point(self, c_xy, pursuer_pos):
        """A point past the corner along the extension of the pursuer->
        corner ray, clipped inside the polygon."""
        dx, dy = c_xy[0] - pursuer_pos[0], c_xy[1] - pursuer_pos[1]
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return c_xy
        dx, dy = dx / n, dy / n
        from shapely.geometry import LineString, Point as ShPoint
        for frac in (1.0, 0.75, 0.5, 0.3, 0.15):
            L = self.SHADOW_DEPTH * frac
            p = (c_xy[0] + dx * L, c_xy[1] + dy * L)
            try:
                if (self.data.shapely_env.covers(ShPoint(p)) and
                        self.data.shapely_env.covers(
                            LineString([c_xy, p]))):
                    return p
            except Exception:
                continue
        return c_xy

    def _walk(self, path, budget, stop_short=0.0, goal=None):
        px, py = self.pos
        for nx, ny in path[1:]:
            if budget <= 0:
                break
            if goal is not None and stop_short > 0:
                d_goal = math.hypot(goal[0] - px, goal[1] - py)
                if d_goal <= stop_short:
                    break
            seg = math.hypot(nx - px, ny - py)
            if seg < 1e-9:
                continue
            step = min(budget, seg)
            if goal is not None and stop_short > 0:
                d_goal = math.hypot(goal[0] - px, goal[1] - py)
                step = min(step, max(0.0, d_goal - stop_short))
            px += (nx - px) / seg * step
            py += (ny - py) / seg * step
            budget -= step
            if step < seg:
                break
        self.pos = [px, py]

    def step(self, dt, pursuer_alphas: dict = None, pursuer_pos=None):
        if not pursuer_alphas:
            return self.pos[0], self.pos[1]
        finite = {c: a for c, a in pursuer_alphas.items()
                  if math.isfinite(a)}
        if not finite:
            return self.pos[0], self.pos[1]
        budget = self.speed * dt

        if self._committed is None:
            cands = {c: a for c, a in finite.items()
                     if c not in self._recent}
            c_star = max(cands or finite, key=(cands or finite).get)
            if finite[c_star] > self.ratio:
                self._committed = c_star
                self._shadow_target = None
                self._hold = 0
            else:
                # no race won: pressure the worst corner at a standoff
                target = (self.data.poly[c_star].x(),
                          self.data.poly[c_star].y())
                self._walk(self._geo_path(target), budget,
                           stop_short=self.standoff, goal=target)
                return self.pos[0], self.pos[1]

        c = self._committed
        c_xy = (self.data.poly[c].x(), self.data.poly[c].y())
        d_c = math.hypot(c_xy[0] - self.pos[0], c_xy[1] - self.pos[1])
        if self._shadow_target is None:
            if d_c > self.ARRIVE:
                self._walk(self._geo_path(c_xy), budget)
                return self.pos[0], self.pos[1]
            self._shadow_target = self._shadow_point(
                c_xy, pursuer_pos if pursuer_pos is not None else self.pos)
        t = self._shadow_target
        d_t = math.hypot(t[0] - self.pos[0], t[1] - self.pos[1])
        if d_t > 0.2:
            self._walk([tuple(self.pos), t], budget)
            return self.pos[0], self.pos[1]
        self._hold += 1
        if self._hold >= self.HOLD_FRAMES:
            self._recent.append(c)
            self._committed = None
            self._shadow_target = None
            self._hold = 0
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
        except (KeyError, ValueError):   # ValueError: pyvisgraph acos
                                         # domain error on near-collinear
                                         # viewpoints at polygon vertices
            with suppress_output():
                raw = self.data.env.shortest_path(
                    vis.Point(self.pos[0], self.pos[1]),
                    vis.Point(target[0], target[1]), EPSILON)
            sp = [vg.Point(p.x(), p.y()) for p in raw.path()]
        return [(p.x, p.y) for p in sp]

    def step(self, dt, pursuer_alphas: dict = None, pursuer_pos=None):
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


# EscapingEvader shares AdversarialEvader's geodesic-path helper.
EscapingEvader._geo_path = AdversarialEvader._geo_path
