"""Per-frame measurements and per-trial metric reductions (paper §7.4)."""
import math
from dataclasses import dataclass, field

import visilibity as vis
from shapely.geometry import Point, Polygon as ShapelyPolygon

from config import EPSILON
from geometry import poly_to_points

BREACH_RADIUS = 1.5   # d_geo(e, c) below which a corner counts as reached


def _point_segment_dist(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    seg2 = abx * abx + aby * aby
    if seg2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / seg2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def alphas_at(pos, path_lengths: dict, data) -> dict:
    """Per-corner alpha at a roadmap position (paper Eq. 1):
        alpha_c(p, e) = d_G(p, c) / d_geo(e, c)

    Locates `pos` on the dense patrol graph (data.total_edges — the same
    edge set compute_optimal_guard optimises over, where corner access
    points are explicit vertices) and combines the distance to each edge
    endpoint with the precomputed endpoint-to-corner distances
    (data.vectors_org), mirroring the tent-function construction.
    """
    px, py = pos
    best = None
    for v1, v2 in data.total_edges:
        a, b = data.vertices[v1], data.vertices[v2]
        d = _point_segment_dist(px, py, a.x, a.y, b.x, b.y)
        if best is None or d < best[0]:
            best = (d, v1, v2)
    _, u_i, w_i = best
    d_u = math.hypot(px - data.vertices[u_i].x, py - data.vertices[u_i].y)
    d_w = math.hypot(px - data.vertices[w_i].x, py - data.vertices[w_i].y)

    alphas = {}
    for corn in data.corners:
        d_g = min(d_u + data.vectors_org[u_i][corn],
                  d_w + data.vectors_org[w_i][corn])
        alphas[corn] = d_g / path_lengths[corn]
    return alphas


def has_los(pursuer_pos, evader_pos, data) -> bool:
    """True if the evader is inside the pursuer's visibility polygon."""
    try:
        vp = vis.Visibility_Polygon(
            vis.Point(pursuer_pos[0], pursuer_pos[1]), data.env, EPSILON)
        xs, ys = poly_to_points(vp)
        if len(xs) < 3:
            return False
        shape = ShapelyPolygon(list(zip(xs, ys)))
        if not shape.is_valid:
            shape = shape.buffer(0)
        return shape.buffer(EPSILON * 10).covers(Point(evader_pos))
    except Exception:
        return False


@dataclass
class TrialRecorder:
    """Accumulates per-frame observations and reduces them to trial metrics."""
    max_alphas: list = field(default_factory=list)
    los_frames: int = 0
    frames: int = 0
    breach_count: int = 0
    _in_breach: set = field(default_factory=set)

    def record(self, pursuer_pos, evader_pos, path_lengths, data):
        alphas = alphas_at(pursuer_pos, path_lengths, data)
        finite = [a for a in alphas.values() if math.isfinite(a)]
        if finite:
            self.max_alphas.append(max(finite))
        if has_los(pursuer_pos, evader_pos, data):
            self.los_frames += 1
        # Breach: evader arrives near corner c while alpha_c > 1 — count
        # entry events, not frames, so a lingering evader counts once.
        now_breach = {c for c in data.corners
                      if path_lengths[c] < BREACH_RADIUS
                      and math.isfinite(alphas[c]) and alphas[c] > 1.0}
        self.breach_count += len(now_breach - self._in_breach)
        self._in_breach = now_breach
        self.frames += 1

    def metrics(self) -> dict:
        n = max(self.frames, 1)
        return {
            'mean_alpha': (sum(self.max_alphas) / len(self.max_alphas)
                           if self.max_alphas else float('nan')),
            'peak_alpha': max(self.max_alphas) if self.max_alphas else float('nan'),
            'los_pct': 100.0 * self.los_frames / n,
            'n_breach': self.breach_count,
        }
