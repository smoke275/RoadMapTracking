"""
Line-of-sight forecast score.

Given the current frame — pursuer on the roadmap heading for the optimal
guard g*, evader somewhere in the polygon — answer, for one reflex corner c:

    If the evader escapes toward c along its geodesic, what fraction of the
    time until the pursuer reaches g* does the pursuer keep line of sight?

and then aggregate that over all corners.

This is a read-only score. It does not change the guard placement, the
controller, or any per-frame computation: it only consumes what the pipeline
already produces (the pursuer's roadmap route from compute_frame, the
evader's geodesic to each corner, the polygon) and replays both paths at
their speeds, testing visibility at each sampled instant.

Visibility test: for a simple polygon without holes, two points see each
other exactly when the segment between them lies inside the polygon, so the
test is shapely `covers(LineString)` rather than a visibility-polygon build.
`covers` (not `contains`) so that a segment touching the boundary — e.g. the
evader sitting on a reflex vertex — still counts as visible.

Time is measured in frames, matching the speeds s_p / s_e in world units
per frame used throughout the simulation.
"""
from dataclasses import dataclass, field
import math

import pyvisgraph as vg
import visilibity as vis
from shapely.geometry import LineString

from config import EPSILON
from geometry import suppress_output
from graph import dijkstra


@dataclass
class LOSProfile:
    corner: int
    times: list = field(default_factory=list)       # sampled instants (frames)
    los: list = field(default_factory=list)         # bool per sample
    score: float = 0.0                              # fraction of samples with LOS
    horizon: float = 0.0                            # last sampled instant
    pursuer_arrival: float = 0.0                    # frames until pursuer reaches g*
    evader_arrival: float = 0.0                     # frames until evader reaches c
    blind_intervals: list = field(default_factory=list)  # [(t_start, t_end), ...]
    los_at_end: bool = False                        # LOS at the horizon instant
    evader_path: list = field(default_factory=list)  # geodesic to the corner, [(x, y), ...]
    evader_cum: list = field(default_factory=list)   # cumulative arc length of evader_path

    def evader_position(self, t: float) -> tuple:
        """Evader position at time t (frames) along its geodesic, given s_e
        was used to build this profile: arc length = s_e * t."""
        return _point_at(self.evader_path, self.evader_cum, self._s_e * t)

    _s_e: float = 1.0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def evader_path_to_corner(ex: float, ey: float, corner_idx: int, data) -> list:
    """Geodesic from the evader to corner `corner_idx` as [(x, y), ...].
    Same lookup and fallback as ker_pipeline.compute_path_lengths."""
    cx, cy = data.poly[corner_idx].x(), data.poly[corner_idx].y()
    try:
        sp = data.geodesic.shortest_path([ex, ey], [cx, cy])
    except KeyError:
        with suppress_output():
            raw = data.env.shortest_path(vis.Point(ex, ey), vis.Point(cx, cy), EPSILON)
        sp = [vg.Point(p.x(), p.y()) for p in raw.path()]
    return [(float(p.x), float(p.y)) for p in sp]


def pursuer_path_to_guard(px: float, py: float, gx: float, gy: float, data) -> list:
    """Roadmap route pursuer -> guard as [(x, y), ...]. dijkstra() works on a
    deep copy of data.graph, so this never mutates shared state. Falls back to
    the pursuer's own position (stationary) if no route exists."""
    _, path = dijkstra(data.graph, (px, py), (gx, gy))
    pts = [(float(x), float(y)) for x, y in path] if path else []
    return pts or [(float(px), float(py))]


def _cumulative_lengths(pts: list) -> list:
    cum = [0.0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
    return cum


def _point_at(pts: list, cum: list, s: float) -> tuple:
    """Point at arc length s along the polyline; clamps to the endpoints."""
    if s <= 0 or len(pts) == 1:
        return pts[0]
    if s >= cum[-1]:
        return pts[-1]
    for i in range(1, len(cum)):
        if s <= cum[i]:
            seg = cum[i] - cum[i - 1]
            f = 0.0 if seg == 0 else (s - cum[i - 1]) / seg
            (x0, y0), (x1, y1) = pts[i - 1], pts[i]
            return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
    return pts[-1]


def has_los(a: tuple, b: tuple, shapely_env) -> bool:
    return shapely_env.covers(LineString([a, b]))


# ---------------------------------------------------------------------------
# Single corner
# ---------------------------------------------------------------------------
def los_profile_for_corner(pursuer_path: list, ex: float, ey: float,
                           corner_idx: int, data,
                           s_p: float, s_e: float,
                           dt: float = 1.0,
                           clip_to_evader: bool = True) -> LOSProfile:
    """Replay the evader escaping to `corner_idx` and the pursuer travelling
    `pursuer_path` (its roadmap route to g*), sampling every `dt` frames.

    Horizon: the pursuer's arrival at g*. If `clip_to_evader`, it is cut
    at the evader's arrival at the corner when that comes first — past the
    corner the evader's path is unknown, so nothing is assumed about it.
    """
    e_pts = evader_path_to_corner(ex, ey, corner_idx, data)
    e_cum = _cumulative_lengths(e_pts)
    p_pts = pursuer_path if pursuer_path else [(ex, ey)]
    p_cum = _cumulative_lengths(p_pts)

    t_p = p_cum[-1] / s_p if s_p > 0 else 0.0
    t_e = e_cum[-1] / s_e if s_e > 0 else 0.0
    horizon = min(t_p, t_e) if clip_to_evader else t_p

    n = max(1, int(math.floor(horizon / dt)) + 1)
    times = [k * dt for k in range(n)]
    if times[-1] < horizon:
        times.append(horizon)

    los = []
    for t in times:
        p = _point_at(p_pts, p_cum, s_p * t)
        e = _point_at(e_pts, e_cum, s_e * t)
        los.append(has_los(p, e, data.shapely_env))

    blind, start = [], None
    for t, ok in zip(times, los):
        if not ok and start is None:
            start = t
        elif ok and start is not None:
            blind.append((start, t))
            start = None
    if start is not None:
        blind.append((start, times[-1]))

    return LOSProfile(
        corner=corner_idx,
        times=times,
        los=los,
        score=sum(los) / len(los),
        horizon=times[-1],
        pursuer_arrival=t_p,
        evader_arrival=t_e,
        blind_intervals=blind,
        los_at_end=los[-1],
        evader_path=e_pts,
        evader_cum=e_cum,
        _s_e=s_e,
    )


# ---------------------------------------------------------------------------
# All corners
# ---------------------------------------------------------------------------
def nearness_weights(path_lengths: dict) -> dict:
    """Corner weights by nearness to the evader: w_c proportional to 1/L_c,
    normalized to sum to 1 — the same inverse-geodesic weighting the guard
    optimization uses (w_c = 1/L_c, Section 5 of the paper). A corner the
    evader can reach in half the distance counts twice as much. path_lengths
    is compute_frame's fc.path_lengths (already floored away from zero)."""
    inv = {c: 1.0 / L for c, L in path_lengths.items() if L > 0}
    total = sum(inv.values())
    return {c: v / total for c, v in inv.items()} if total > 0 else {}


def los_scores(pursuer_path: list, ex: float, ey: float, data,
               s_p: float, s_e: float, alphas: list | None = None,
               path_lengths: dict | None = None,
               dt: float = 1.0, clip_to_evader: bool = True) -> dict:
    """Per-corner profiles plus aggregates:
      'fitness'        — THE single score: mean over corners weighted by
                         nearness of each corner to the evader (w_c ∝ 1/L_c,
                         see nearness_weights); only if `path_lengths`
                         (compute_frame's fc.path_lengths) is given
      'worst'          — lowest score over corners (the corner that would hurt most)
      'mean'           — unweighted mean over corners
      'alpha_weighted' — mean weighted by each corner's current alpha (higher
                         alpha = more exposed corner = more weight); only if
                         `alphas` (compute_frame's fc.alphas, same order as
                         data.corners) is given
      'critical'       — score for the corner with the largest alpha, i.e.
                         the one the Adversarial Evader would actually pick
    """
    profiles = {
        c: los_profile_for_corner(pursuer_path, ex, ey, c, data, s_p, s_e,
                                  dt=dt, clip_to_evader=clip_to_evader)
        for c in data.corners
    }
    scores = {c: pr.score for c, pr in profiles.items()}
    out = {
        'profiles': profiles,
        'per_corner': scores,
        'worst': min(scores.values()),
        'worst_corner': min(scores, key=scores.get),
        'mean': sum(scores.values()) / len(scores),
    }
    if path_lengths is not None:
        w = nearness_weights({c: path_lengths[c] for c in data.corners if c in path_lengths})
        if w:
            out['nearness_weights'] = w
            out['fitness'] = sum(scores[c] * wc for c, wc in w.items())
    if alphas is not None and len(alphas) == len(data.corners):
        finite = [(c, a) for c, a in zip(data.corners, alphas) if math.isfinite(a)]
        total = sum(a for _, a in finite)
        if finite and total > 0:
            out['alpha_weighted'] = sum(scores[c] * a for c, a in finite) / total
        c_crit = max(finite, key=lambda ca: ca[1])[0] if finite else out['worst_corner']
        out['critical'] = scores[c_crit]
        out['critical_corner'] = c_crit
    return out


def score_frame(fc, ex: float, ey: float, data, s_p: float, s_e: float, **kw) -> dict:
    """Convenience wrapper around a FrameComputed from ker_pipeline.compute_frame:
    uses its obs_to_guard_path as the pursuer route, its path_lengths for the
    nearness weights, and its alphas for the alpha-weighted aggregate."""
    return los_scores(list(fc.obs_to_guard_path), ex, ey, data, s_p, s_e,
                      alphas=list(fc.alphas), path_lengths=dict(fc.path_lengths), **kw)


def fitness(fc, ex: float, ey: float, data, s_p: float, s_e: float, **kw) -> float:
    """The single fitness score for the current frame, in [0, 1]: expected
    fraction of the trip to g* during which the pursuer keeps line of sight,
    with each corner's escape weighted by its nearness to the evader."""
    return score_frame(fc, ex, ey, data, s_p, s_e, **kw)['fitness']
