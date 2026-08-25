"""Free-motion pursuer using the kernel-weighted control law from
"Scalable Multi-Agent Surveillance: A Kernel-Based Approach"
(Mandal & Bhattacharya, ICRA 2025) — Eq. 1-2. Reuses our own ILP guard
cover (data.guards) as the static anchor set; only the per-frame motion
law differs from the roadmap-based strategies in pursuers.py. Kept as a
separately-labeled benchmark strategy so results never mix with those.

Simplifications from the paper, agreed with the user:
- gamma_i(p, e): straight-line direction to corner i when directly
  visible, else the direction of the first leg of the geodesic shortest
  path to it — rather than the cited single-corner optimal-control law
  (Bhattacharya & Hutchinson, IJRR 2011), which this paper doesn't derive
  in-text and which we don't have on hand.
- C (the corner set summed over in Eq. 1) is taken as the gap-corner set,
  matching Eq. 2's own domain — q_i is only defined over gap-corners, so
  every term in the weighted sum has a matching weight.
"""
import math

import visilibity as vis
from shapely.geometry import Point, Polygon as ShapelyPolygon

from config import EPSILON
from geometry import poly_to_points, suppress_output


def _vis_shape(px, py, env):
    vp = vis.Visibility_Polygon(vis.Point(px, py), env, EPSILON)
    vx, vy = poly_to_points(vp)
    if len(vx) < 3:
        return None
    shp = ShapelyPolygon(list(zip(vx, vy)))
    return shp if shp.is_valid else shp.buffer(0)


def _edge_visible(vis_shape, corner_xy, neighbor_xy) -> bool:
    """True if a point just inside the corner->neighbor edge is visible
    from the viewpoint that produced vis_shape."""
    cx, cy = corner_xy
    nx, ny = neighbor_xy
    t = 0.02
    px, py = cx + t * (nx - cx), cy + t * (ny - cy)
    return vis_shape.contains(Point(px, py))


def gap_corners(guard_pos, data) -> list:
    """Reflex corners currently visible to the guard with exactly one of
    their two incident edges also visible (paper's "gap-corner")."""
    vshape = _vis_shape(guard_pos[0], guard_pos[1], data.env)
    if vshape is None:
        return []
    n = len(data.poly)
    out = []
    for i in data.corners:
        cx, cy = data.poly[i].x(), data.poly[i].y()
        if not vshape.buffer(1).contains(Point(cx, cy)):
            continue  # corner not currently visible to this guard
        prev_xy = (data.poly[(i - 1) % n].x(), data.poly[(i - 1) % n].y())
        next_xy = (data.poly[(i + 1) % n].x(), data.poly[(i + 1) % n].y())
        vis_prev = _edge_visible(vshape, (cx, cy), prev_xy)
        vis_next = _edge_visible(vshape, (cx, cy), next_xy)
        if vis_prev != vis_next:   # exactly one of the two visible
            out.append(i)
    return out


def _direction_to_corner(guard_pos, corner_idx, data, vshape):
    """Straight line if visible, else the first leg of the geodesic
    shortest path — per the user's simplification of gamma_i."""
    cx, cy = data.poly[corner_idx].x(), data.poly[corner_idx].y()
    px, py = guard_pos
    if vshape is not None and vshape.buffer(1).contains(Point(cx, cy)):
        dx, dy = cx - px, cy - py
    else:
        with suppress_output():
            sp = data.geodesic.shortest_path((px, py), (cx, cy))
        if len(sp) >= 2:
            dx, dy = sp[1].x - px, sp[1].y - py
        else:
            dx, dy = cx - px, cy - py
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 1e-9 else (0.0, 0.0)


def _risk_weights(gap_idxs, evader_pos, data) -> dict:
    """Eq. 2: Gaussian weighting over gap-corners by geodesic distance
    from the evader, peaking at the nearest (most urgent) corner."""
    if len(gap_idxs) == 1:
        return {gap_idxs[0]: 1.0}
    dists = {}
    ex, ey = evader_pos
    for i in gap_idxs:
        cx, cy = data.poly[i].x(), data.poly[i].y()
        with suppress_output():
            dists[i] = data.geodesic.get_distance((ex, ey), (cx, cy))
    ordered = sorted(gap_idxs, key=lambda i: dists[i])
    l0, lm = dists[ordered[0]], dists[ordered[-1]]
    sigma = abs(l0 - lm) / 2
    if sigma < 1e-9:
        return {i: 1.0 / len(gap_idxs) for i in gap_idxs}
    raw = {i: math.exp(-0.5 * ((dists[i] - l0) / sigma) ** 2) for i in gap_idxs}
    total = sum(raw.values())
    return {i: v / total for i, v in raw.items()}


class KernelWeightedPursuer:
    """Free (non-roadmap) pursuer using the kernel-based paper's Eq. 1-2
    control law, anchored by our own ILP-computed guard cover."""

    name = 'kernel-control'

    def __init__(self, data, evader_start, speed):
        self.data = data
        self.speed = speed
        if data.guards:
            gx, gy = min(data.guards, key=lambda g: math.hypot(
                g[0] - evader_start[0], g[1] - evader_start[1]))
        else:
            gx, gy = evader_start
        self.pos = [gx, gy]

    def step(self, evader_pos, path_lengths=None):
        gaps = gap_corners(self.pos, self.data)
        if not gaps:
            return self.pos[0], self.pos[1]
        weights = _risk_weights(gaps, evader_pos, self.data)
        vshape = _vis_shape(self.pos[0], self.pos[1], self.data.env)
        vx = vy = 0.0
        for i in gaps:
            dx, dy = _direction_to_corner(self.pos, i, self.data, vshape)
            vx += dx * weights[i]
            vy += dy * weights[i]
        norm = math.hypot(vx, vy)
        if norm > 1e-9:
            vx, vy = vx / norm, vy / norm
        nx, ny = self.pos[0] + vx * self.speed, self.pos[1] + vy * self.speed
        if self.data.shapely_env.buffer(1).contains(Point(nx, ny)):
            self.pos = [nx, ny]
        return self.pos[0], self.pos[1]
