"""
Common visible roadmap along an evader escape.

Given an evader position and a target reflex corner, walk the evader along
its geodesic to that corner and, at every sampled instant, intersect the
patrol roadmap with the evader's visibility polygon. The running intersection
of those per-sample pieces is the part of the roadmap that stays visible to
the evader for the *whole* trip: a pursuer standing anywhere on it never
loses sight of the evader from the moment it leaves until it reaches the
corner.

Pure computation, no Qt. Reuses what the pipeline already produces:
  - data.path_lines            the patrol roadmap (shapely LineStrings)
  - los_score.evader_path_to_corner   the evader's geodesic to a corner
  - ker_pipeline._build_vis_shape     visilibity visibility polygon -> shapely

The intersection needs only the path's vertices plus one subtlety. In a
simple polygon, if a roadmap point q sees both endpoints of a straight path
leg, any occlusion of the leg's interior can only come from the obstacle
wedges at the leg's own endpoints: a boundary intrusion into the triangle
q-u-v would have to cross one of its sides (which lie in the polygon) or
enter through a vertex. The wedge at a bend vertex occludes a contiguous
stretch of the leg starting at that vertex, so q sees the whole leg iff it
sees a point a hair along the leg from each endpoint. We therefore evaluate
visibility at viewpoints nudged _NUDGE units along each incident leg of
every path vertex (not at the vertex itself, and not along the interior
bisector, both of which can be visible while the leg's first stretch is
not). Geodesics bend only at reflex vertices, so the bend-side viewpoints
form a small fixed set per polygon and `VisCache` memoises the clipped
roadmap per viewpoint, which keeps sweeping many evader positions cheap;
configurations whose blocked stretch is shorter than _NUDGE are missed,
an error of the same order as the _VIS_TOL buffering.

`trace_common_visible_roadmap` additionally samples the path every `step`
units purely so a viewer can replay the walk; the samples do not affect the
result.
"""
from dataclasses import dataclass
import math

from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

from ker_pipeline import _build_vis_shape
from los_score import evader_path_to_corner, _cumulative_lengths, _point_at

# Buffer applied to each visibility polygon before intersecting with the
# roadmap, so a roadmap segment lying exactly on the polygon's boundary (for
# example collinear with a sight ray through a reflex vertex) is not dropped
# by floating-point noise.
_VIS_TOL = 0.01

# How far to nudge a sample that sits exactly on the polygon boundary (a
# reflex vertex the geodesic bends around, or the destination corner itself)
# back along the path into the interior before asking for its visibility
# polygon. visilibity is happier with strictly interior viewpoints.
_NUDGE = 0.1

# Roadmap pieces shorter than this after clipping are numerical debris.
_DEGENERATE_LEN = 1e-6


@dataclass
class SampleVis:
    s: float              # arc length from the evader's start
    pos: tuple            # (x, y) the visibility polygon was evaluated at
    vis_shape: object     # shapely Polygon, or None if the build failed
    visible: object       # roadmap ∩ vis_shape, lines only (MultiLineString)
    visible_len: float    # total length of `visible`


@dataclass
class RoadmapTrace:
    corner: int
    evader_path: list     # [(x, y), ...] geodesic start -> corner
    evader_cum: list      # cumulative arc length along evader_path
    roadmap: object       # MultiLineString of the whole roadmap
    roadmap_len: float
    samples: list         # [SampleVis], in path order
    common: object        # running intersection over all samples, lines only
    common_len: float
    n_failed: int         # samples whose visibility polygon could not be built

    @property
    def path_len(self) -> float:
        return self.evader_cum[-1] if self.evader_cum else 0.0

    @property
    def common_frac(self) -> float:
        return self.common_len / self.roadmap_len if self.roadmap_len > 0 else 0.0

    def evader_position(self, s: float) -> tuple:
        return _point_at(self.evader_path, self.evader_cum, s)

    def sample_at(self, s: float) -> SampleVis:
        """Last sample at or before arc length s."""
        best = self.samples[0]
        for smp in self.samples:
            if smp.s <= s:
                best = smp
            else:
                break
        return best


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def lines_only(geom) -> MultiLineString:
    """Keep only the LineString parts of an arbitrary shapely geometry.
    Intersections can produce GeometryCollections with stray Points, which
    carry no length and only get in the way of further intersections."""
    if geom is None or geom.is_empty:
        return MultiLineString([])
    parts = []
    for g in getattr(geom, 'geoms', [geom]):
        if g.geom_type == 'LineString':
            parts.append(g)
        elif g.geom_type == 'MultiLineString':
            parts.extend(g.geoms)
    # Clipping a line exactly at a polygon vertex can leave a zero-length
    # LineString behind; it carries no visibility information.
    return MultiLineString([g for g in parts if g.length > _DEGENERATE_LEN])


def roadmap_geometry(data) -> MultiLineString:
    """The patrol roadmap as one MultiLineString (noded at crossings)."""
    return lines_only(unary_union(list(data.path_lines)))


def _corner_interior_point(idx: int, data) -> tuple:
    """A point a hair inside the polygon at reflex vertex `idx`, along the
    interior bisector, so its visibility polygon is well defined."""
    n = len(data.poly)
    a, b, c = data.poly[(idx - 1) % n], data.poly[idx], data.poly[(idx + 1) % n]
    u1 = (a.x() - b.x(), a.y() - b.y())
    u2 = (c.x() - b.x(), c.y() - b.y())
    n1 = math.hypot(*u1) or 1.0
    n2 = math.hypot(*u2) or 1.0
    bx = u1[0] / n1 + u2[0] / n2
    by = u1[1] / n1 + u2[1] / n2
    nb = math.hypot(bx, by)
    if nb < 1e-9:                      # collinear edges: use the left normal
        bx, by = -u1[1] / n1, u1[0] / n1
        nb = 1.0
    for sign in (-1.0, 1.0):           # reflex: interior is opposite the edge wedge
        cand = (b.x() + sign * bx / nb * _NUDGE, b.y() + sign * by / nb * _NUDGE)
        if data.shapely_env.contains(Point(cand)):
            return cand
    return (b.x(), b.y())


class VisCache:
    """Memoised roadmap ∩ visibility polygon per viewpoint, for one
    SimulationData. Viewpoints are keyed by rounded coordinates; corners are
    pre-nudged inside via their index."""

    def __init__(self, data, roadmap=None):
        self.data = data
        self.roadmap = roadmap if roadmap is not None else roadmap_geometry(data)
        self._by_key: dict = {}
        self._corner_pos = {c: _corner_interior_point(c, data) for c in data.corners}
        self._corner_by_xy = {(round(data.poly[c].x(), 4), round(data.poly[c].y(), 4)): c
                              for c in data.corners}
        self.misses = 0

    @staticmethod
    def _key(pos):
        return (round(pos[0], 3), round(pos[1], 3))

    def visible_roadmap(self, pos: tuple):
        """Clipped roadmap for an interior viewpoint (built once)."""
        k = self._key(pos)
        if k not in self._by_key:
            self.misses += 1
            vis_shape = _build_vis_shape(pos[0], pos[1], self.data.env)
            self._by_key[k] = (lines_only(self.roadmap.intersection(vis_shape.buffer(_VIS_TOL)))
                               if vis_shape is not None else None)
        return self._by_key[k]

    def visible_roadmap_at_corner(self, corner_idx: int):
        return self.visible_roadmap(self._corner_pos[corner_idx])

    def corner_index_at(self, pos: tuple):
        return self._corner_by_xy.get((round(pos[0], 4), round(pos[1], 4)))


def _leg_viewpoints(path: list, i: int, shapely_env) -> list:
    """Viewpoints certifying sight of the legs incident to path[i]: the
    vertex nudged _NUDGE units along each incident leg. The obstacle wedge at
    a bend (or at the destination corner) occludes a contiguous stretch of a
    leg starting at the vertex, from viewpoints that may still see the vertex
    itself, so the vertex (or a bisector nudge) is not a valid certificate
    for the leg."""
    pos = path[i]
    if i == 0 and shapely_env.contains(Point(pos)):
        return [pos]                     # the start is an interior point: no wedge
    views = []
    for j in (i - 1, i + 1):
        if j < 0 or j >= len(path):
            continue
        dx, dy = path[j][0] - pos[0], path[j][1] - pos[1]
        d = math.hypot(dx, dy)
        if d < 1e-9:
            continue
        cand = (pos[0] + dx / d * _NUDGE, pos[1] + dy / d * _NUDGE)
        if shapely_env.contains(Point(cand)):
            views.append(cand)
    return views or [pos]


def persistent_visible_roadmap(ex: float, ey: float, corner_idx: int, data,
                               cache: VisCache, path: list = None):
    """Exact V(e, c): roadmap visible for the whole escape from (ex, ey) to
    corner `corner_idx` — the intersection of the clipped roadmap at
    viewpoints nudged along each leg incident to each path vertex (see
    module docstring). Returns (MultiLineString, path)."""
    if path is None:
        path = evader_path_to_corner(ex, ey, corner_idx, data)
    common = cache.roadmap
    for i in range(len(path)):
        for view in _leg_viewpoints(path, i, data.shapely_env):
            vis = cache.visible_roadmap(view)
            if vis is None:
                continue
            common = lines_only(common.intersection(vis.buffer(_VIS_TOL)))
            if common.is_empty:
                return common, path
    return common, path


def sample_path(pts: list, cum: list, step: float) -> list:
    """Arc lengths to sample: every `step` units, plus every bend point and
    both endpoints. Returns a sorted list of floats."""
    total = cum[-1]
    ss = {0.0, total}
    ss.update(cum)
    if step > 0:
        k = 1
        while k * step < total:
            ss.add(k * step)
            k += 1
    return sorted(ss)


def _interior_viewpoint(pos: tuple, prev_pos: tuple, next_pos: tuple, shapely_env):
    """Return `pos` if strictly inside the polygon, else the same point nudged
    slightly along the path (back first, then forward) into the interior."""
    p = Point(pos)
    if shapely_env.contains(p):
        return pos
    for other in (prev_pos, next_pos):
        if other is None:
            continue
        dx, dy = other[0] - pos[0], other[1] - pos[1]
        d = math.hypot(dx, dy)
        if d < 1e-9:
            continue
        cand = (pos[0] + dx / d * _NUDGE, pos[1] + dy / d * _NUDGE)
        if shapely_env.contains(Point(cand)):
            return cand
    return pos


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def trace_common_visible_roadmap(ex: float, ey: float, corner_idx: int, data,
                                 step: float = 5.0, cache: VisCache = None) -> RoadmapTrace:
    """Walk the evader from (ex, ey) to reflex corner `corner_idx` along its
    geodesic. `common` is the exact persistently visible roadmap (see
    persistent_visible_roadmap); the per-sample pieces every `step` units
    are for replaying the walk in a viewer."""
    if cache is None:
        cache = VisCache(data)
    common, path = persistent_visible_roadmap(ex, ey, corner_idx, data, cache)
    cum = _cumulative_lengths(path)
    roadmap = cache.roadmap
    roadmap_len = roadmap.length

    samples = []
    n_failed = 0
    ss = sample_path(path, cum, step)
    for i, s in enumerate(ss):
        pos = _point_at(path, cum, s)
        prev_pos = _point_at(path, cum, ss[i - 1]) if i > 0 else None
        next_pos = _point_at(path, cum, ss[i + 1]) if i + 1 < len(ss) else None
        view = _interior_viewpoint(pos, prev_pos, next_pos, data.shapely_env)

        vis_shape = _build_vis_shape(view[0], view[1], data.env)
        if vis_shape is None:
            n_failed += 1
            samples.append(SampleVis(s=s, pos=view, vis_shape=None,
                                     visible=MultiLineString([]), visible_len=0.0))
            continue
        visible = lines_only(roadmap.intersection(vis_shape.buffer(_VIS_TOL)))
        samples.append(SampleVis(s=s, pos=view, vis_shape=vis_shape,
                                 visible=visible, visible_len=visible.length))

    return RoadmapTrace(
        corner=corner_idx,
        evader_path=path,
        evader_cum=cum,
        roadmap=roadmap,
        roadmap_len=roadmap_len,
        samples=samples,
        common=common,
        common_len=common.length,
        n_failed=n_failed,
    )


def trace_all_corners(ex: float, ey: float, data, step: float = 5.0,
                      cache: VisCache = None) -> dict:
    """Convenience: one RoadmapTrace per reflex corner, keyed by corner index."""
    cache = cache or VisCache(data)
    return {c: trace_common_visible_roadmap(ex, ey, c, data, step=step, cache=cache)
            for c in data.corners}


def persistent_visible_all_corners(ex: float, ey: float, data, cache: VisCache) -> dict:
    """Exact V(e, c) for every corner, no sampling: {corner: MultiLineString}."""
    return {c: persistent_visible_roadmap(ex, ey, c, data, cache)[0] for c in data.corners}
