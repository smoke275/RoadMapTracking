"""Geodesic path queries, graph construction, and Dijkstra over the patrol graph."""
import copy
import heapq
import random

import numpy as np
import pyvisgraph as vg
import visilibity as vis
from bidict import bidict
from scipy.sparse.csgraph import csgraph_from_dense
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon

from config import BOUNDARY_X, BOUNDARY_Y, EPSILON
from geometry import add_unique_point, distance_vg, suppress_output

# pyvisgraph's sweep-line visibility-graph builder has known crashes on
# symmetric/degenerate input (e.g. a plain axis-aligned rectangle) — its
# Edge.__lt__ tie-break assumes any two edges equidistant from the scan
# point share a vertex, which isn't always true, and other code paths
# divide by zero on aligned geometry. A tiny deterministic jitter breaks
# the exact float ties that trigger these, at a scale far below anything
# visually or geometrically meaningful.
_JITTER_MAG  = 1e-4   # world units
_JITTER_SEED = 1337   # fixed, so the same polygon always gets the same jitter


def _jitter(pts):
    rng = random.Random(_JITTER_SEED)
    return [vg.Point(p.x + rng.uniform(-_JITTER_MAG, _JITTER_MAG),
                     p.y + rng.uniform(-_JITTER_MAG, _JITTER_MAG))
            for p in pts]


class Geodesic:
    """Builds a pyvisgraph visibility graph and answers shortest-path queries."""

    def __init__(self, poly):
        self._poly = poly
        self.graph: vg.VisGraph = None
        self._room  = None   # lazy: buffered shapely shape, for path validation
        self._venv  = None   # lazy: visilibity Environment, for the fallback below

    def build(self):
        poly   = self._poly
        maxx_i = max(range(len(poly)), key=lambda i: poly[i].x())
        vgpoly = [vg.Point(pt.x(), pt.y()) for pt in poly]

        # The rim must strictly enclose the polygon with real margin, or the
        # hole-bridging construction below (splicing the polygon into a
        # bounding rectangle so pyvisgraph treats the polygon's interior as
        # free space) produces an invalid/self-touching input and the
        # resulting visibility graph can include edges that cut outside the
        # polygon. BOUNDARY_X/Y (500) is also the drawing tool's canvas
        # extent, so a legitimately-drawn polygon can reach right up to it —
        # a fixed rim at exactly that size leaves no safety margin. Compute
        # the rim from the actual polygon's extent instead, always with
        # genuine headroom beyond it.
        xs = [p.x() for p in poly]
        ys = [p.y() for p in poly]
        margin = max(max(max(abs(v) for v in xs), max(abs(v) for v in ys)) * 0.5, 100)
        rim_x  = max(BOUNDARY_X, max(abs(v) for v in xs) + margin)
        rim_y  = max(BOUNDARY_Y, max(abs(v) for v in ys) + margin)

        rim = [
            vg.Point(poly[maxx_i].x(),  rim_y),
            vg.Point(-rim_x,             rim_y),
            vg.Point(-rim_x,            -rim_y),
            vg.Point(poly[maxx_i].x(), -rim_y),
        ]
        vgpoly = vgpoly[:maxx_i + 1] + rim + vgpoly[maxx_i:]
        vgpoly = _jitter(vgpoly)
        self.graph = vg.VisGraph()
        self.graph.build([vgpoly])

    def shortest_path(self, a, b):
        sp = self.graph.shortest_path(vg.Point(a[0], a[1]), vg.Point(b[0], b[1]))
        if self._path_stays_inside(sp):
            return sp
        # pyvisgraph's dynamic on-the-fly visibility check (used whenever the
        # query points aren't exact existing graph vertices — always true
        # here, since build() jitters them) has its own correctness bugs
        # independent of the sweep-line crash worked around there: it can
        # report a false line-of-sight straight across a real reflex-corner
        # notch. Fall back to visilibity's independent shortest-path
        # implementation, which doesn't share that bug.
        with suppress_output():
            raw = self._env().shortest_path(vis.Point(a[0], a[1]), vis.Point(b[0], b[1]), EPSILON)
        return [vg.Point(p.x(), p.y()) for p in raw.path()]

    def get_distance(self, a, b):
        sp = self.shortest_path(a, b)
        return sum(distance_vg(sp[i], sp[i + 1]) for i in range(len(sp) - 1))

    def _path_stays_inside(self, sp) -> bool:
        if self._room is None:
            self._room = ShapelyPolygon([(p.x(), p.y()) for p in self._poly]).buffer(2)
        for i in range(len(sp) - 1):
            seg = LineString([(sp[i].x, sp[i].y), (sp[i + 1].x, sp[i + 1].y)])
            if not self._room.contains(seg):
                return False
        return True

    def _env(self) -> vis.Environment:
        if self._venv is None:
            walls = vis.Polygon(list(reversed(self._poly)))
            self._venv = vis.Environment([walls])
            self._venv.PRINTING_DEBUG_DATA = False
        return self._venv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _are_close(p1, p2, eps=1):
    return abs(p1[0] - p2[0]) < eps and abs(p1[1] - p2[1]) < eps


def _find_close(point, graph):
    for p in graph:
        if _are_close(p, point):
            return p
    return None


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def construct_graph(linestrings: list) -> dict:
    graph = {}
    for line in linestrings:
        rs, re = line.coords[0], line.coords[-1]
        s = _find_close(rs, graph) or rs
        e = _find_close(re, graph) or re
        graph.setdefault(s, {})[e] = line.length
        graph.setdefault(e, {})[s] = line.length
    return graph


def construct_graph_dense(linestrings, intersection_points):
    graph     = construct_graph(linestrings)
    raw_verts = list(graph.keys())
    num_orig  = len(raw_verts)
    vertices  = [Point(pt[0], pt[1]) for pt in raw_verts]

    for pts in intersection_points.values():
        for pt in pts:
            add_unique_point(vertices, pt)

    n     = len(vertices)
    dense = np.full((n, n), -1.0)
    bi    = bidict({i: vertices[i].coords[0] for i in range(n)})

    for u, nbrs in graph.items():
        for v in nbrs:
            i, j = bi.inverse[u], bi.inverse[v]
            dense[i][j] = dense[j][i] = Point(u).distance(Point(v))
    np.fill_diagonal(dense, 0)

    for i in range(num_orig, n):
        p_pt = vertices[i]
        pt_c = p_pt.coords[0]
        edge = find_edge_for_point(pt_c, graph)
        if edge is None:
            continue
        u, v   = edge
        pu, pv = Point(u), Point(v)
        if p_pt.equals_exact(pu, 1e-6) or p_pt.equals_exact(pv, 1e-6):
            continue
        graph[pt_c] = {}
        graph[v].pop(u, None)
        graph[u].pop(v, None)
        graph[pt_c][u] = graph[u][pt_c] = p_pt.distance(pu)
        graph[pt_c][v] = graph[v][pt_c] = p_pt.distance(pv)
        iu, iv, ip = bi.inverse[u], bi.inverse[v], bi.inverse[pt_c]
        dense[iu][ip] = dense[ip][iu] = p_pt.distance(pu)
        dense[iv][ip] = dense[ip][iv] = p_pt.distance(pv)
        dense[iu][iv] = dense[iv][iu] = -1

    scipy_g   = csgraph_from_dense(dense, null_value=-1)
    scipy_src = {}
    for corner, pts in intersection_points.items():
        scipy_src[corner] = [
            next((k for k in range(n) if vertices[k].equals_exact(pt, 1e-6)), None)
            for pt in pts
        ]
    return dense, scipy_g, scipy_src, bi, vertices


def add_observer_to_graph(dense, vertices, og_graph, bi_map, observer: Point):
    k         = len(vertices)
    new_dense = np.full((k + 1, k + 1), -1.0)
    new_dense[:k, :k] = dense
    new_dense[k - 1][k - 1] = 0

    pt    = observer.coords[0]
    graph = copy.deepcopy(og_graph)
    edge  = find_edge_for_point(pt, graph)
    if edge is None:
        return csgraph_from_dense(dense, null_value=-1)

    u, v   = edge
    pu, pv = Point(u), Point(v)
    if not pu.equals_exact(observer, 1e-6) and not pv.equals_exact(observer, 1e-6):
        graph[pt] = {}
        graph[v].pop(u, None)
        graph[u].pop(v, None)
        graph[pt][u] = graph[u][pt] = observer.distance(pu)
        graph[pt][v] = graph[v][pt] = observer.distance(pv)
        iu, iv = bi_map.inverse[u], bi_map.inverse[v]
        new_dense[iu][k] = new_dense[k][iu] = observer.distance(pu)
        new_dense[iv][k] = new_dense[k][iv] = observer.distance(pv)
        new_dense[iu][iv] = new_dense[iv][iu] = -1

    return csgraph_from_dense(new_dense, null_value=-1)


def point_on_edge(point, edge, eps=3):
    return LineString([edge[0], edge[1]]).distance(Point(point)) < eps


def find_edge_for_point(point, graph):
    for vertex, nbrs in graph.items():
        for nb in nbrs:
            if point_on_edge(point, (vertex, nb)):
                return vertex, nb
    return None


# ---------------------------------------------------------------------------
# Dijkstra over the patrol graph dict
# ---------------------------------------------------------------------------

def dijkstra(graph, start, end):
    se = find_edge_for_point(start, graph)
    ee = find_edge_for_point(end,   graph)
    if not se or not ee:
        return float('inf'), []

    graph = copy.deepcopy(graph)
    if 'start' not in graph:
        graph['start'] = {
            se[0]: Point(start).distance(Point(se[0])),
            se[1]: Point(start).distance(Point(se[1])),
        }
        graph[se[0]]['start'] = graph['start'][se[0]]
        graph[se[1]]['start'] = graph['start'][se[1]]
    if 'end' not in graph:
        graph['end'] = {
            ee[0]: Point(end).distance(Point(ee[0])),
            ee[1]: Point(end).distance(Point(ee[1])),
        }
        graph[ee[0]]['end'] = graph['end'][ee[0]]
        graph[ee[1]]['end'] = graph['end'][ee[1]]

    dist    = {v: float('inf') for v in graph}
    prev    = {v: None for v in graph}
    dist['start'] = 0
    counter = 0
    pq      = [(0, counter, 'start')]

    while pq:
        d, _, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for nb, w in graph[u].items():
            nd = d + w
            if nd < dist[nb]:
                dist[nb] = nd
                prev[nb] = u
                counter += 1
                heapq.heappush(pq, (nd, counter, nb))

    path = []
    node = 'end'
    while node is not None:
        if node == 'start':
            path.append(start.coords[0] if hasattr(start, 'coords') else tuple(start))
        elif node == 'end':
            path.append(end.coords[0] if hasattr(end, 'coords') else tuple(end))
        else:
            path.append(node)
        node = prev.get(node)
    path.reverse()

    return dist['end'], path
