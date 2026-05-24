"""
ker_simulation.py
KER-based pursuit-evasion simulation — refactored for clarity and better graphics.

Controls (interactive mode):
  Drag RED dot     — move evader anywhere inside polygon
  Drag GREEN dot   — slide observer/pursuer along the patrol path
  Drag CYAN dot    — cycle active corner (shows its patrol intersection points)
  A                — toggle auto-evader (Voronoi skeleton routing)
  Esc              — quit
"""

import atexit
import contextlib
import copy
import csv
from collections import deque
from enum import Enum, auto
import hashlib
import heapq
import itertools
import math
import os
import pickle
import random
import sys
import threading
import time

import numpy as np
import pyclipper
import pygame
import seaborn as sns
from bidict import bidict
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect, QPoint, QTimer
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygon, QColor, QFont, QTransform
from PyQt5.QtWidgets import QApplication, QMainWindow
from scipy.sparse import find as sp_find
from scipy.sparse.csgraph import csgraph_from_dense, floyd_warshall
from shapely.geometry import LineString, Polygon, Point
import shapely.geometry as sp_geom
from scipy.spatial import Voronoi as SciVoronoi
import visilibity as vis
import pyvisgraph as vg

from function_generator import process_functions

# --- pursuer debug log (append mode so each run accumulates) ---
_DBG_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pursuer_debug.log')
_dbg_log = open(_DBG_LOG_PATH, 'a', buffering=1)  # line-buffered
_dbg_log.write(f'\n{"="*60}\n[RUN START] {time.strftime("%Y-%m-%d %H:%M:%S")}\n{"="*60}\n')
atexit.register(_dbg_log.close)


def _dbg(msg: str):
    """Write a timestamped line to the pursuer debug log."""
    _dbg_log.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPSILON      = 1e-7
BOUNDARY_X   = 500
BOUNDARY_Y   = 500
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FILE_NAME    = os.path.join(_SCRIPT_DIR, 'resources', 'sites_poly9.csv')
POINT_RADIUS = 8
WINDOW_SIZE  = 750

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
random.seed(43)

CACHE_FILE   = os.path.join(_SCRIPT_DIR, 'resources', 'ker_cache.pkl')
GEO_FILE     = os.path.join(_SCRIPT_DIR, 'resources', 'ker_geodesic.graph')


def _poly_fingerprint(poly) -> str:
    """SHA-256 of the cleaned polygon vertex coords (rounded to 2 dp)."""
    coords = tuple((round(p.x(), 2), round(p.y(), 2)) for p in poly)
    return hashlib.sha256(str(coords).encode()).hexdigest()


def _save_cache(fingerprint: str, payload: dict, geodesic):
    """Pickle payload and save geodesic graph to disk."""
    try:
        os.makedirs('resources', exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'fingerprint': fingerprint, 'data': payload}, f, protocol=4)
        geodesic.graph.save(GEO_FILE)
        print('[CACHE] Saved.')
    except Exception as e:
        print(f'[CACHE] Save failed: {e}')


def _load_cache(fingerprint: str):
    """Return payload dict and Geodesic if cache is valid, else (None, None)."""
    try:
        if not os.path.exists(CACHE_FILE) or not os.path.exists(GEO_FILE):
            return None, None
        with open(CACHE_FILE, 'rb') as f:
            blob = pickle.load(f)
        if blob.get('fingerprint') != fingerprint:
            print('[CACHE] Polygon changed — recomputing.')
            return None, None
        geo = vg.VisGraph()
        geo.load(GEO_FILE)
        print('[CACHE] Loaded.')
        return blob['data'], geo
    except Exception as e:
        print(f'[CACHE] Load failed: {e}')
        return None, None

# ---------------------------------------------------------------------------
# Shared mutable state (written by simulation thread, read by Qt paint thread)
# ---------------------------------------------------------------------------
_sem    = threading.Semaphore()
_env    = None           # vis.Environment
_p_walls = None          # vis.Polygon
_asso   = {}             # corner-index → clipped visibility polygon (list of (x,y))
_corners = []            # list of concave corner indices

# ---------------------------------------------------------------------------
# Drawing command tags
# ---------------------------------------------------------------------------
class Op(Enum):
    point          = auto()
    line           = auto()
    dotted_line    = auto()
    circle         = auto()
    filled_circle  = auto()
    border_circle  = auto()
    polygon        = auto()
    filled_polygon = auto()
    dotted_polygon = auto()
    border_polygon = auto()
    draw_text      = auto()


# ---------------------------------------------------------------------------
# Utility: suppress C-level stdout/stderr (visilibity SWIG noise)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_output():
    devnull   = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(sys.stdout.fileno())
    saved_err = os.dup(sys.stderr.fileno())
    try:
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        yield
    finally:
        os.dup2(saved_out, sys.stdout.fileno())
        os.dup2(saved_err, sys.stderr.fileno())
        for fd in (devnull, saved_out, saved_err):
            os.close(fd)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def find_slope_and_intercept(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:
        return float('inf'), x1
    m = (y2 - y1) / (x2 - x1)
    return m, y1 - m * x1


def interpolate_point(a: Point, b: Point, s: float) -> Point:
    """Return the point s units along the segment a→b."""
    d = a.distance(b)
    if d == 0:
        return a
    t = s / d
    return Point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))


def add_unique_point(points, new_point, decimal=2):
    tol = 10 ** (-decimal)
    for p in points:
        if p.equals_exact(new_point, tol):
            return
    points.append(new_point)


def add_unique_linestring(linestrings, new_ls, tolerance=0.01):
    p21, p22 = Point(new_ls.coords[0]), Point(new_ls.coords[1])
    for ls in linestrings:
        p11, p12 = Point(ls.coords[0]), Point(ls.coords[1])
        if (p11.equals_exact(p21, tolerance) and p12.equals_exact(p22, tolerance)) or \
           (p11.equals_exact(p22, tolerance) and p12.equals_exact(p21, tolerance)):
            return
    linestrings.append(new_ls)


def distance_vg(p1: vg.Point, p2: vg.Point) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def poly_to_points(polygon) -> tuple:
    return ([polygon[i].x() for i in range(polygon.n())],
            [polygon[i].y() for i in range(polygon.n())])


def minimum_distance(pt_a: vis.Point, pt_b: vis.Point, pt_e: vis.Point) -> float:
    AB = np.array([pt_b.x() - pt_a.x(), pt_b.y() - pt_a.y()])
    AE = np.array([pt_e.x() - pt_a.x(), pt_e.y() - pt_a.y()])
    BE = np.array([pt_e.x() - pt_b.x(), pt_e.y() - pt_b.y()])
    if np.dot(AB, BE) > 0:
        return math.hypot(pt_e.x() - pt_b.x(), pt_e.y() - pt_b.y())
    if np.dot(AB, AE) < 0:
        return math.hypot(pt_e.x() - pt_a.x(), pt_e.y() - pt_a.y())
    mod = np.linalg.norm(AB)
    return abs(AB[0] * AE[1] - AB[1] * AE[0]) / mod


def find_intersection(polygon, point: Point, direction):
    boundary  = LineString(polygon)
    poly_shape = Polygon(polygon)
    far = 1e6

    def _pick(ray, skip=0):
        inter = boundary.intersection(ray)
        if inter.is_empty:
            return None, inter
        if inter.geom_type == 'MultiPoint':
            pts = sorted(inter.geoms, key=lambda p: point.distance(p))
            if skip == 0:
                return pts[0].coords[0], inter
            mid = Point((pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2)
            return (pts[skip].coords[0] if mid.within(poly_shape) else None), inter
        if inter.geom_type == 'Point':
            return (inter.coords[0] if skip == 0 else None), inter
        if inter.geom_type == 'LineString':
            return inter.coords[0], inter
        if inter.geom_type == 'MultiLineString':
            return inter.geoms[0].coords[0], inter
        if inter.geom_type == 'GeometryCollection':
            return inter.geoms[1].coords[0], inter
        return None, inter

    fwd = LineString([point, (point.x + direction[0] * far, point.y + direction[1] * far)])
    if point.touches(poly_shape):
        return _pick(fwd, skip=1)
    if point.within(poly_shape):
        return _pick(fwd)
    bwd = LineString([point, (point.x - direction[0] * far, point.y - direction[1] * far)])
    candidates = [pt for r in (fwd, bwd) for pt, _ in [_pick(r)] if pt is not None]
    if not candidates:
        return None, None
    return min(candidates, key=lambda p: point.distance(Point(p))), None


# ---------------------------------------------------------------------------
# Geodesic (visibility-graph shortest paths)
# ---------------------------------------------------------------------------
class Geodesic:
    """Builds a pyvisgraph visibility graph and answers shortest-path queries."""

    def __init__(self, poly):
        self._poly = poly
        self.graph: vg.VisGraph = None

    def build(self):
        poly = self._poly
        maxx_i = max(range(len(poly)), key=lambda i: poly[i].x())
        vgpoly  = [vg.Point(pt.x(), pt.y()) for pt in poly]
        rim = [
            vg.Point(poly[maxx_i].x(),  BOUNDARY_Y),
            vg.Point(-BOUNDARY_X,        BOUNDARY_Y),
            vg.Point(-BOUNDARY_X,       -BOUNDARY_Y),
            vg.Point(poly[maxx_i].x(), -BOUNDARY_Y),
        ]
        vgpoly = vgpoly[:maxx_i + 1] + rim + vgpoly[maxx_i:]
        self.graph = vg.VisGraph()
        self.graph.build([vgpoly])

    def shortest_path(self, a, b):
        return self.graph.shortest_path(vg.Point(a[0], a[1]), vg.Point(b[0], b[1]))

    def get_distance(self, a, b):
        sp = self.shortest_path(a, b)
        return sum(distance_vg(sp[i], sp[i + 1]) for i in range(len(sp) - 1))


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def _are_close(p1, p2, eps=1):
    return abs(p1[0] - p2[0]) < eps and abs(p1[1] - p2[1]) < eps


def _find_close(point, graph):
    for p in graph:
        if _are_close(p, point):
            return p
    return None


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
    graph    = construct_graph(linestrings)
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
        p_pt   = vertices[i]
        pt_c   = p_pt.coords[0]
        edge   = find_edge_for_point(pt_c, graph)
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

    scipy_g = csgraph_from_dense(dense, null_value=-1)

    scipy_src = {}
    for corner, pts in intersection_points.items():
        scipy_src[corner] = [
            next((k for k in range(n) if vertices[k].equals_exact(pt, 1e-6)), None)
            for pt in pts
        ]
    return dense, scipy_g, scipy_src, bi, vertices


def add_observer_to_graph(dense, vertices, og_graph, bi_map, observer: Point):
    k        = len(vertices)
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

    dist = {v: float('inf') for v in graph}
    prev = {v: None for v in graph}
    dist['start'] = 0
    counter = 0
    pq = [(0, counter, 'start')]
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

    # Reconstruct path as coordinate tuples
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


# ---------------------------------------------------------------------------
# Polygon interaction helpers
# ---------------------------------------------------------------------------
def get_random_point_in_polygon(pol):
    minx, miny, maxx, maxy = pol.bounds
    while True:
        p = sp_geom.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if pol.contains(p):
            return p


# ---------------------------------------------------------------------------
# Voronoi skeleton inside polygon
# ---------------------------------------------------------------------------
def build_voronoi_skeleton(shapely_poly: Polygon, samples: int = 300):
    """
    Sample the polygon boundary, compute interior Voronoi edges, return:
      - nodes: list of (x, y) tuples (unique Voronoi vertices inside poly)
      - adj:   dict { node_idx: {neighbour_idx: edge_length, ...}, ... }
      - edges: list of ((x1,y1), (x2,y2)) for drawing
    """
    poly_shrunken = shapely_poly.buffer(-2)   # tiny inset to avoid boundary noise
    if poly_shrunken.is_empty:
        poly_shrunken = shapely_poly

    # Sample boundary points densely
    boundary = shapely_poly.exterior
    step = boundary.length / samples
    pts  = [boundary.interpolate(i * step) for i in range(samples)]
    coords = np.array([(p.x, p.y) for p in pts])

    vor = SciVoronoi(coords)

    edges = []
    node_set = {}   # (x,y) → int index

    def _add_node(xy):
        key = (round(xy[0], 1), round(xy[1], 1))
        if key not in node_set:
            node_set[key] = len(node_set)
        return node_set[key]

    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        p1 = tuple(vor.vertices[ridge[0]])
        p2 = tuple(vor.vertices[ridge[1]])
        seg = LineString([p1, p2])
        # Keep only edges whose midpoint is inside the polygon
        mid = seg.interpolate(0.5, normalized=True)
        if not shapely_poly.contains(mid):
            continue
        # Clip segment to polygon
        clipped = seg.intersection(shapely_poly)
        if clipped.is_empty or clipped.geom_type not in ('LineString', 'MultiLineString'):
            continue
        segs = [clipped] if clipped.geom_type == 'LineString' else list(clipped.geoms)
        for s in segs:
            if s.length < 1:
                continue
            c0 = (s.coords[0][0], s.coords[0][1])
            c1 = (s.coords[-1][0], s.coords[-1][1])
            edges.append((c0, c1))
            _add_node(c0)
            _add_node(c1)

    # Build adjacency
    idx_to_xy = {v: k for k, v in node_set.items()}
    nodes = [idx_to_xy[i] for i in range(len(idx_to_xy))]
    adj: dict[int, dict[int, float]] = {i: {} for i in range(len(nodes))}

    for c0, c1 in edges:
        k0 = (round(c0[0], 1), round(c0[1], 1))
        k1 = (round(c1[0], 1), round(c1[1], 1))
        i0, i1 = node_set[k0], node_set[k1]
        d = math.hypot(c0[0] - c1[0], c0[1] - c1[1])
        adj[i0][i1] = d
        adj[i1][i0] = d

    return nodes, adj, edges


def _skeleton_nearest_node(nodes, x, y):
    """Return index of skeleton node nearest to (x, y)."""
    best, best_d = 0, float('inf')
    for i, (nx_, ny_) in enumerate(nodes):
        d = math.hypot(x - nx_, y - ny_)
        if d < best_d:
            best_d, best = d, i
    return best


def _skeleton_path(nodes, adj, src_idx, dst_idx):
    """Dijkstra on skeleton adj dict; returns list of node indices."""
    dist = {i: float('inf') for i in range(len(nodes))}
    prev = {}
    dist[src_idx] = 0
    pq = [(0.0, src_idx)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst_idx:
            break
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    # Reconstruct
    path = []
    cur = dst_idx
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(src_idx)
    path.reverse()
    return path if path[0] == src_idx else []


def _skeleton_pick_destination(nodes, adj, src_idx, min_hops: int = 8):
    """BFS from src_idx; pick a random node at least min_hops away."""
    visited = {src_idx: 0}
    q = deque([src_idx])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in visited:
                visited[v] = visited[u] + 1
                q.append(v)
    far = [n for n, hops in visited.items() if hops >= min_hops]
    if not far:
        far = [n for n in visited if n != src_idx] or [src_idx]
    return random.choice(far)


# ---------------------------------------------------------------------------
# KER corner association (clipped angular visibility wedge per concave corner)
# ---------------------------------------------------------------------------
def compute_corner_asso(poly, corners, env) -> dict:
    """For each concave corner, compute the clipped angular visibility wedge."""
    asso = {}
    for j in corners:
        vispol = vis.Visibility_Polygon(poly[j], env, EPSILON)
        a, b, c = poly[(j - 1) % len(poly)], poly[j], poly[(j + 1) % len(poly)]
        ex, ey = poly_to_points(vispol)
        ex.reverse()
        ey.reverse()

        v1 = np.array([b.x() - a.x(), b.y() - a.y()])
        v2 = np.array([b.x() - c.x(), b.y() - c.y()])
        v1 = v1 / np.linalg.norm(v1) * 10000
        v2 = v2 / np.linalg.norm(v2) * 10000

        subj = [(ex[k], ey[k]) for k in range(len(ex))]
        clip = [(b.x(), b.y()), (v1[0] + b.x(), v1[1] + b.y()), (v2[0] + b.x(), v2[1] + b.y())]
        pc = pyclipper.Pyclipper()
        pc.AddPath(clip, pyclipper.PT_CLIP, True)
        pc.AddPath(subj, pyclipper.PT_SUBJECT, True)
        sol = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        asso[j] = sol[0] if sol else None
    return asso


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
# Palette for interactive HUD
C_POLYGON   = Qt.white
C_EVADER    = Qt.red
C_OBSERVER  = QColor(0, 220, 80)     # bright green
C_CORNER    = Qt.cyan
C_GUARD     = QColor(255, 50, 255)   # magenta
C_PATH      = QColor(240, 200, 0)    # golden yellow
C_OPT_EDGE  = QColor(80, 160, 255)   # sky blue
C_INTER_PT  = QColor(80, 180, 255)   # light blue dots
C_TEXT      = Qt.white


def _qcolor(r, g, b, a=255) -> QColor:
    c = QColor()
    c.setRgb(r, g, b, a)
    return c


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class Window(QMainWindow):
    """PyQt5 window that renders the simulation and handles interactive drag."""

    def __init__(self):
        super().__init__()
        self.title = "KER Simulation"

        self._draw_queue: list = []   # written by simulation thread
        self._main_stack: list = []   # swapped into painter

        # Interactive state — defaults until run() populates them
        self.poly                   = None
        self.corners                = []
        self.shapely_polygon        = None
        self.lines                  = []
        self.draggable_point_evader  = QPoint(0, 0)
        self.draggable_point_observer = QPoint(0, 0)
        self.draggable_point_corner  = None
        self.dragging_evader  = False
        self.dragging_observer = False
        self.dragging_corner  = False
        self.point_radius     = POINT_RADIUS

        # Auto-evader state
        self.auto_evader         = False   # toggled by 'A'
        self._skel_nodes         = []      # list of (x, y)
        self._skel_adj           = {}      # adjacency dict
        self._skel_edges         = []      # list of ((x1,y1),(x2,y2)) for drawing
        self._skel_path: list    = []      # current planned path (node indices)
        self._skel_path_pos: float = 0.0  # distance travelled along current segment
        self._skel_seg_idx: int  = 0      # current segment index in _skel_path
        self._evader_speed: float = 3.0   # world-units per frame

        # Autonomous observer state (free 2D, only active when auto_evader is on)
        self._auto_observer_pos   = None   # [float x, float y]
        self._pursuer_speed: float = 2.0   # world-units per frame

        # Roadmap pursuer state (toggled by 'P', only active when auto_evader is on)
        self._roadmap_pursuer: bool    = False  # toggled by 'P'
        self._roadmap_obs_pos          = None   # [float x, float y] current position
        # Stable-node path: only the fixed base graph nodes between pursuer and guard.
        # wp[0] (pursuer virtual) and wp[-1] (guard virtual) are stripped — these are
        # the real, stable nodes the pursuer commits to visiting in order.
        self._roadmap_base_path: list  = []     # [(x,y), ...] base nodes to visit
        self._roadmap_base_idx: int    = 0      # index of current target node
        self._roadmap_guard_pos        = None   # (gx, gy) final dest after base nodes
        self._roadmap_edge_behind      = None   # last graph node the pursuer physically passed through
        self._roadmap_direct: bool     = False  # True = guard confirmed on same edge; allow partial step
        # --- oscillation debug ---
        self._dbg_prev_pursuer_pos     = None   # pos at start of previous frame
        self._dbg_frame_count: int     = 0      # total simulation frames
        self._dbg_dir_flips: int       = 0      # consecutive direction-reversal count
        self._dbg_prev_move_vec        = None   # (dx,dy) unit vec of last movement

        self._init_window()
        timer = QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(33)          # ~30 fps repaint

    def _init_window(self):
        self.setWindowTitle(self.title)
        self.setGeometry(200, 100, WINDOW_SIZE, WINDOW_SIZE)
        self.setFixedSize(self.size())
        self.show()

    # ------------------------------------------------------------------
    # Draw command queue (called from simulation thread)
    # ------------------------------------------------------------------
    def draw(self, cmd):
        self._draw_queue.append(cmd)

    def execute(self):
        """Flush draw queue to the main stack (thread-safe)."""
        _sem.acquire()
        try:
            self._main_stack = self._draw_queue
            self._draw_queue = []
        finally:
            _sem.release()
        self.update()

    # ------------------------------------------------------------------
    # Convenience draw helpers
    # ------------------------------------------------------------------
    def _d_polygon(self, xs, ys, color=Qt.white, width=2):
        self.draw([Op.polygon, xs, ys, width, color])

    def _d_filled_polygon(self, xs, ys, color, width=1):
        self.draw([Op.filled_polygon, xs, ys, width, color])

    def _d_line(self, x1, y1, x2, y2, color=Qt.white, width=2):
        self.draw([Op.line, x1, y1, x2, y2, width, color])

    def _d_dot(self, x, y, r, color):
        self.draw([Op.filled_circle, x, y, r, 1, color])

    def _d_ring(self, x, y, r, color, width=2):
        self.draw([Op.circle, x, y, r, width, color])

    def _d_text(self, x, y, text, size=14):
        self.draw([Op.draw_text, x, y, size, text])

    def _d_vertex_labels(self, poly):
        """Draw index label for every vertex, offset outward from polygon centroid."""
        cx = sum(p.x() for p in poly) / len(poly)
        cy = sum(p.y() for p in poly) / len(poly)
        for i, p in enumerate(poly):
            dx = p.x() - cx
            dy = p.y() - cy
            norm = (dx**2 + dy**2) ** 0.5 or 1
            ox = int(p.x() + dx / norm * 16)
            oy = int(p.y() + dy / norm * 16)
            self._d_text(ox, oy, str(i), size=9)

    def _d_diamond(self, x, y, r, fill_color, border_color=Qt.white):
        """Draw a diamond (rotated square) as a filled border_polygon."""
        xs = [x,     x + r, x,     x - r]
        ys = [y + r, y,     y - r, y    ]
        self.draw([Op.border_polygon, xs, ys, 2, fill_color])

    def _d_glow_line(self, x1, y1, x2, y2, color, width=4):
        """Two-pass line for a subtle glow: thick dim first, thin bright on top."""
        dim = QColor(color)
        dim.setAlpha(80)
        self.draw([Op.line, x1, y1, x2, y2, width + 4, dim])
        self.draw([Op.line, x1, y1, x2, y2, width, color])

    # ------------------------------------------------------------------
    # Paint event
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setWindow(QRect(-BOUNDARY_X, -BOUNDARY_Y, 2 * BOUNDARY_X, 2 * BOUNDARY_Y))
        painter.setViewport(QRect(0, 0, WINDOW_SIZE, WINDOW_SIZE))
        painter.scale(1, -1)
        painter.setRenderHint(QPainter.Antialiasing)

        # Black background
        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
        bg = QPolygon([QPoint(-500, 500), QPoint(500, 500),
                       QPoint(500, -500), QPoint(-500, -500)])
        painter.drawPolygon(bg)

        _sem.acquire()
        try:
            stack = list(self._main_stack)
        finally:
            _sem.release()

        for cmd in stack:
            op = cmd[0]
            if op == Op.point:
                painter.setPen(QPen(cmd[4], cmd[3], Qt.SolidLine))
                painter.drawPoint(int(cmd[1]), int(cmd[2]))

            elif op == Op.line:
                painter.setPen(QPen(cmd[6], cmd[5], Qt.SolidLine))
                painter.drawLine(int(cmd[1]), int(cmd[2]), int(cmd[3]), int(cmd[4]))

            elif op == Op.dotted_line:
                painter.setPen(QPen(cmd[6], cmd[5], Qt.DotLine))
                painter.drawLine(int(cmd[1]), int(cmd[2]), int(cmd[3]), int(cmd[4]))

            elif op == Op.circle:
                painter.setPen(QPen(cmd[5], cmd[4], Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPoint(int(cmd[1]), int(cmd[2])), int(cmd[3]), int(cmd[3]))

            elif op == Op.filled_circle:
                painter.setPen(QPen(cmd[5], cmd[4], Qt.SolidLine))
                painter.setBrush(QBrush(cmd[5], Qt.SolidPattern))
                painter.drawEllipse(QPoint(int(cmd[1]), int(cmd[2])), int(cmd[3]), int(cmd[3]))

            elif op == Op.border_circle:
                painter.setPen(QPen(Qt.black, cmd[4], Qt.SolidLine))
                painter.setBrush(QBrush(cmd[5], Qt.SolidPattern))
                painter.drawEllipse(QPoint(int(cmd[1]), int(cmd[2])), int(cmd[3]), int(cmd[3]))

            elif op in (Op.polygon, Op.dotted_polygon):
                pen_style = Qt.DotLine if op == Op.dotted_polygon else Qt.SolidLine
                painter.setPen(QPen(cmd[4], cmd[3], pen_style))
                painter.setBrush(Qt.NoBrush)
                pts = [QPoint(int(cmd[1][j]), int(cmd[2][j])) for j in range(len(cmd[1]))]
                painter.drawPolygon(QPolygon(pts))

            elif op == Op.filled_polygon:
                painter.setPen(QPen(cmd[4], cmd[3], Qt.SolidLine))
                painter.setBrush(QBrush(cmd[4], Qt.SolidPattern))
                pts = [QPoint(int(cmd[1][j]), int(cmd[2][j])) for j in range(len(cmd[1]))]
                painter.drawPolygon(QPolygon(pts))

            elif op == Op.border_polygon:
                painter.setPen(QPen(Qt.black, cmd[3], Qt.SolidLine))
                painter.setBrush(QBrush(cmd[4], Qt.SolidPattern))
                pts = [QPoint(int(cmd[1][j]), int(cmd[2][j])) for j in range(len(cmd[1]))]
                painter.drawPolygon(QPolygon(pts))

            elif op == Op.draw_text:
                painter.setPen(QPen(C_TEXT, 1, Qt.SolidLine))
                font = QFont("Consolas", cmd[3])
                painter.setFont(font)
                pos = QPoint(cmd[1], -cmd[2])
                painter.save()
                transform = QTransform()
                transform.scale(1, -1)
                painter.setTransform(transform, True)
                painter.drawText(pos, cmd[4])
                painter.restore()

    # ------------------------------------------------------------------
    # Keyboard / Mouse
    # ------------------------------------------------------------------
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_A:
            self.auto_evader = not self.auto_evader
            print(f'[AUTO-EVADER] {"ON" if self.auto_evader else "OFF"}')
            if self.auto_evader and self._skel_nodes and not self._skel_path:
                # Kick off from current evader position
                ex, ey = self.draggable_point_evader.x(), self.draggable_point_evader.y()
                src = _skeleton_nearest_node(self._skel_nodes, ex, ey)
                dst = _skeleton_pick_destination(self._skel_nodes, self._skel_adj, src)
                self._skel_path    = _skeleton_path(self._skel_nodes, self._skel_adj, src, dst)
                self._skel_seg_idx = 0
                self._skel_path_pos = 0.0
                # Initialise autonomous observer at current draggable observer position
                self._auto_observer_pos = [
                    float(self.draggable_point_observer.x()),
                    float(self.draggable_point_observer.y()),
                ]
                # Initialise roadmap observer position too
                self._roadmap_obs_pos    = [
                    float(self.draggable_point_observer.x()),
                    float(self.draggable_point_observer.y()),
                ]
                self._roadmap_base_path  = []
                self._roadmap_base_idx   = 0
                self._roadmap_guard_pos  = None
                self._roadmap_edge_behind = None
                self._roadmap_direct     = False
            if not self.auto_evader:
                self._auto_observer_pos = None
                self._roadmap_obs_pos   = None
        elif e.key() == Qt.Key_P:
            self._roadmap_pursuer = not self._roadmap_pursuer
            print(f'[ROADMAP-PURSUER] {"ON" if self._roadmap_pursuer else "OFF"}')
            if self.auto_evader:
                if self._roadmap_pursuer and self._roadmap_obs_pos is None:
                    self._roadmap_obs_pos = [
                        float(self.draggable_point_observer.x()),
                        float(self.draggable_point_observer.y()),
                    ]
                    self._roadmap_base_path  = []
                    self._roadmap_base_idx   = 0
                    self._roadmap_guard_pos  = None
                    self._roadmap_edge_behind = None
                    self._roadmap_direct     = False
                elif not self._roadmap_pursuer and self._auto_observer_pos is None:
                    self._auto_observer_pos = [
                        float(self.draggable_point_observer.x()),
                        float(self.draggable_point_observer.y()),
                    ]
        super().keyPressEvent(e)

    def _map_mouse(self, event):
        W, H = self.width(), self.height()
        mx = (event.x() / W) * 2 * BOUNDARY_X - BOUNDARY_X
        my = ((H - event.y()) / W) * 2 * BOUNDARY_Y - BOUNDARY_Y
        return mx, my

    def mousePressEvent(self, event):
        if self.poly is None:
            return
        mx, my = self._map_mouse(event)
        tp = QPoint(int(mx), int(my))
        cp = QPoint(int(self.poly[self.draggable_point_corner].x()),
                    int(self.poly[self.draggable_point_corner].y()))
        r  = self.point_radius + 6   # slightly generous hit radius

        if (tp - self.draggable_point_evader).manhattanLength() <= r:
            self.dragging_evader = True
        elif (tp - self.draggable_point_observer).manhattanLength() <= r:
            self.dragging_observer = True
        elif (tp - cp).manhattanLength() <= r:
            self.dragging_corner = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging_evader  = False
        self.dragging_observer = False
        self.dragging_corner  = False

    def mouseMoveEvent(self, event):
        if self.poly is None:
            return
        mx, my = self._map_mouse(event)

        if self.dragging_evader:
            pt = Point(mx, my)
            if pt.within(self.shapely_polygon):
                self.draggable_point_evader = QPoint(int(mx), int(my))
            else:
                self.dragging_evader = False

        elif self.dragging_observer:
            pt = Point(mx, my)
            best, best_d = None, float('inf')
            for line in self.lines:
                proj = line.interpolate(line.project(pt))
                d    = pt.distance(proj)
                if d < best_d:
                    best_d, best = d, proj
            if best:
                self.draggable_point_observer = QPoint(int(best.x), int(best.y))

        elif self.dragging_corner:
            tp = QPoint(int(mx), int(my))
            best_i, best_d = None, float('inf')
            for idx in self.corners:
                cp = QPoint(int(self.poly[idx].x()), int(self.poly[idx].y()))
                d  = (tp - cp).manhattanLength()
                if d < best_d:
                    best_d, best_i = d, idx
            if best_i is not None and self.draggable_point_corner != best_i:
                print(f'Active corner → {best_i}')
                self.draggable_point_corner = best_i

    # ------------------------------------------------------------------
    # Apriori helpers (kept but unused by default)
    # ------------------------------------------------------------------
    def _apriori_check_intersection(self, values):
        if not values:
            return None
        if len(values) == 1:
            return _asso[values[0]]
        pc  = pyclipper.Pyclipper()
        pol = _asso[values[0]]
        for i in range(1, len(values)):
            pc.AddPath(pol, pyclipper.PT_CLIP, True)
            pc.AddPath(_asso[values[i]], pyclipper.PT_SUBJECT, True)
            sol = pc.Execute(pyclipper.CT_INTERSECTION,
                             pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            pc.Clear()
            if not sol or not sol[0]:
                return None
            pol = sol[0]
        return pol

    # ------------------------------------------------------------------
    # Run (simulation thread entry point)
    # ------------------------------------------------------------------
    def run(self, poly):
        global _env, _p_walls, _asso, _corners

        # Remove duplicate adjacent vertices (handles CSV with closing point)
        clean = []
        for p in poly:
            if not clean or p.x() != clean[-1].x() or p.y() != clean[-1].y():
                clean.append(p)
        while len(clean) > 1 and clean[-1].x() == clean[0].x() and clean[-1].y() == clean[0].y():
            clean.pop()
        poly = clean

        x = [p.x() for p in poly]
        y = [p.y() for p in poly]

        # ---- Environment setup (always needed) -----------------------
        walls  = vis.Polygon([p for p in reversed(poly)])
        _p_walls = walls
        _env    = vis.Environment([walls])
        _env.PRINTING_DEBUG_DATA = False

        # ---- Try loading from cache ----------------------------------
        fingerprint = _poly_fingerprint(poly)
        _cached, _cached_geo = _load_cache(fingerprint)
        if _cached is not None:
            # Restore globals and skip all computation
            _corners[:] = _cached['corners']
            _asso.update(_cached['asso'])
            KER              = [vis.Point(cx, cy) for cx, cy in _cached['KER_coords']]
            lst_edges_KER    = _cached['lst_edges_KER']
            chosed_set       = _cached['chosed_set']
            path_lines       = _cached['path_lines']
            intersection_points = _cached['intersection_points']
            dist_matrix      = _cached['dist_matrix']
            vectors_org      = _cached['vectors_org']
            dense            = _cached['dense']
            scipy_src        = _cached['scipy_src']
            bi_map           = _cached['bi_map']
            vertices         = _cached['vertices']
            graph            = _cached['graph']
            geodesic         = Geodesic(poly)
            geodesic.graph   = _cached_geo
            scipy_g          = csgraph_from_dense(dense, null_value=-1)
            shapely_env      = Polygon([(p.x(), p.y()) for p in poly])
            palette          = sns.color_palette("husl", len(_corners))

            # Show polygon, asso regions, patrol path quickly
            for j, n in zip(_corners, range(len(_corners))):
                vp = _asso.get(j)
                if vp is None:
                    continue
                vx = [vp[k][0] for k in range(len(vp))]
                vy = [vp[k][1] for k in range(len(vp))]
                col = QColor()
                col.setRgbF(palette[n][0], palette[n][1], palette[n][2], 0.35)
                self._d_filled_polygon(vx, vy, col)
            self._d_polygon(x, y, Qt.white, 3)
            self._d_vertex_labels(poly)
            for b in _corners:
                self._d_dot(poly[b].x(), poly[b].y(), 6, C_CORNER)
            for p in path_lines:
                self._d_glow_line(p.coords[0][0], p.coords[0][1],
                                  p.coords[1][0], p.coords[1][1], C_PATH, width=3)
            self.execute()
            time.sleep(0.5)
            # Jump straight to interactive loop
            rows, cols, _ = sp_find(scipy_g)
            total_edges = list(zip(cols, rows))
        else:
            # ---- Detect concave corners ------------------------------
            for i in range(len(poly)):
                a, b, c = poly[(i-1) % len(poly)], poly[i], poly[(i+1) % len(poly)]
                s = [b.x()-a.x(), b.y()-a.y()]
                t = [c.x()-b.x(), c.y()-b.y()]
                if s[0]*t[1] - t[0]*s[1] > 0:
                    _corners.append(i)

            # Initial display: polygon + corners
            self._d_polygon(x, y, Qt.white, 3)
            self._d_vertex_labels(poly)
            for b in _corners:
                self._d_dot(poly[b].x(), poly[b].y(), 6, C_CORNER)
            self.execute()
            time.sleep(1.5)

            # ---- Angular visibility wedges (asso) --------------------
            _asso = compute_corner_asso(poly, _corners, _env)
            print(f'Corners: {_corners}')

            # ---- KER boundary contraction ---------------------------
            boundary        = list(range(len(poly)))
            current_corners = _corners.copy()
            corner_clusters = {k: [k] for k in _corners}
            vertex_clusters = {k: {k} for k in _corners}

            random.seed(90)
            while current_corners:
                if len(current_corners) == 1:
                    lc = current_corners[0]
                    for ii in boundary:
                        vertex_clusters[lc].add(ii)
    
                mod_k = None
                for (j, k) in itertools.product(current_corners, [-1, 1]):
                    d  = boundary.index(j)
                    m  = (d + 2 * k) % len(boundary)
                    m1 = (d + k)     % len(boundary)
                    with suppress_output():
                        sp = _env.shortest_path(poly[j], poly[boundary[m]], EPSILON)
                    if len(sp.path()) == 2:
                        corner_clusters[j].extend([boundary[m], boundary[m1]])
                        vertex_clusters[j].add(boundary[m1])
                        mod_k = (d + k) % len(boundary)
                        break
    
                if mod_k is None:
                    break
                boundary = [b for i, b in enumerate(boundary) if i != mod_k]
    
                removal = []
                for corner in current_corners:
                    d  = boundary.index(corner)
                    a  = poly[boundary[(d-1) % len(boundary)]]
                    b  = poly[boundary[d]]
                    c  = poly[boundary[(d+1) % len(boundary)]]
                    s  = [b.x()-a.x(), b.y()-a.y()]
                    t  = [c.x()-b.x(), c.y()-b.y()]
                    if s[0]*t[1] - t[0]*s[1] > 0:
                        removal.append(corner)
                current_corners = removal
    
                bx = [poly[k].x() for k in boundary]
                by = [poly[k].y() for k in boundary]
                self._d_polygon(x,  y,  Qt.white,     3)
                self._d_vertex_labels(poly)
                self._d_polygon(bx, by, Qt.darkGreen,  2)
                for b in current_corners:
                    self._d_dot(poly[b].x(), poly[b].y(), 6, C_CORNER)
                self.execute()
    
            # ---- Refine asso with vertex cluster visibility intersections -
            for key, cluster in vertex_clusters.items():
                for i in cluster:
                    vispol = vis.Visibility_Polygon(poly[i], _env, EPSILON)
                    px, py = poly_to_points(vispol)
                    px.reverse(); py.reverse()
                    clip = [(px[k], py[k]) for k in range(len(px))]
                    if _asso[key] is None:
                        continue
                    pc = pyclipper.Pyclipper()
                    pc.AddPath(clip, pyclipper.PT_CLIP, True)
                    pc.AddPath(_asso[key], pyclipper.PT_SUBJECT, True)
                    sol = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                    pc.Clear()
                    _asso[key] = sol[0] if sol else None
    
            # ---- Display coloured asso regions ---------------------------
            palette = sns.color_palette("husl", len(_corners))
            for j, n in zip(_corners, range(len(_corners))):
                vp = _asso[j]
                if vp is None:
                    continue
                vx = [vp[k][0] for k in range(len(vp))]
                vy = [vp[k][1] for k in range(len(vp))]
                col = QColor()
                col.setRgbF(palette[n][0], palette[n][1], palette[n][2], 0.35)
                self._d_filled_polygon(vx, vy, col)
                self._d_polygon(x, y, Qt.white, 3)
                self._d_vertex_labels(poly)
                self._d_dot(poly[j].x(), poly[j].y(), 6, C_CORNER)
                self.execute()
    
            # Final all-corners view
            for j, n in zip(_corners, range(len(_corners))):
                vp = _asso[j]
                if vp is None:
                    continue
                vx = [vp[k][0] for k in range(len(vp))]
                vy = [vp[k][1] for k in range(len(vp))]
                col = QColor()
                col.setRgbF(palette[n][0], palette[n][1], palette[n][2], 0.35)
                self._d_filled_polygon(vx, vy, col)
            self._d_polygon(x, y, Qt.white, 3)
            self._d_vertex_labels(poly)
            self.execute()
            time.sleep(0.5)
    
            # ---- Visibility graph + KER points ---------------------------
            vis_graph  = vis.Visibility_Graph(_env, EPSILON)
            n_vg       = vis_graph.n()
            edges      = {frozenset([n_vg-1-j, n_vg-1-k])
                          for j in range(n_vg)
                          for k in range(n_vg)
                          if bool(vis_graph(j, k)) and j != k}
    
            # Adjacency list per vertex
            vis_list = [[] for _ in range(len(poly))]
            for it in range(len(poly)):
                for edge in edges:
                    if it in edge:
                        nb = list(edge)
                        nb.remove(it)
                        vis_list[it].append(nb[0])
            for it in range(len(poly)):
                vis_list[it].sort()
    
            # Seed KER with the concave corner positions
            KER = [vis.Point(poly[b].x(), poly[b].y()) for b in _corners]
    
            # Extend KER along each corner's outgoing visibility rays
            lst_edges_KER = []
            for it in range(len(vis_list)):
                if it not in _corners:
                    continue
                for jt in vis_list[it]:
                    a1 = np.array([poly[jt].x(), poly[jt].y()])
                    b1 = np.array([poly[it].x(), poly[it].y()])
                    c1 = (b1 - a1) / np.linalg.norm(b1 - a1)
                    poly_coords = [(p.x(), p.y()) for p in poly]
                    last, _ = find_intersection(poly_coords, Point(b1[0], b1[1]), (c1[0], c1[1]))
                    if last is not None:
                        lp = vis.Point(last[0], last[1])
                        KER.append(lp)
                        lst_edges_KER.append(frozenset([poly[it], lp]))
    
            # Add intersection points of KER edges
            for it in range(len(lst_edges_KER)):
                for jt in range(it + 1, len(lst_edges_KER)):
                    [ia, ib] = list(lst_edges_KER[it])
                    [ja, jb] = list(lst_edges_KER[jt])
                    l1 = LineString([(ia.x(), ia.y()), (ib.x(), ib.y())])
                    l2 = LineString([(ja.x(), ja.y()), (jb.x(), jb.y())])
                    pt = l1.intersection(l2)
                    if not pt.is_empty and pt.geom_type == 'Point':
                        KER.append(vis.Point(pt.x, pt.y))
    
            # Draw KER rays and points
            self._d_polygon(x, y, Qt.white, 3)
            self._d_vertex_labels(poly)
            for edge in lst_edges_KER:
                [j, k] = list(edge)
                self.draw([Op.line, j.x(), j.y(), k.x(), k.y(), 1, Qt.darkRed])
            for pt in KER:
                self._d_dot(pt.x(), pt.y(), 3, QColor(0, 200, 100))
            print(f'KER points: {len(KER)}')
            self.execute()
            time.sleep(1.5)
    
            # ---- Map KER points to corners they cover --------------------
            ts = time.time()
            KER_corner_list = [[] for _ in range(len(KER))]
            for it in _asso:
                pol = _asso[it]
                if pol is None:
                    continue
                pv  = [vis.Point(k[0], k[1]) for k in pol]
                pw  = vis.Polygon(list(pv))
                for jt in range(len(KER)):
                    if KER[jt]._in(pw, 1e-5):
                        KER_corner_list[jt].append(it)
                    else:
                        for kt in range(len(pv)):
                            if minimum_distance(pv[kt], pv[(kt+1) % len(pv)], KER[jt]) < 1:
                                KER_corner_list[jt].append(it)
                                break
            print(f'KER coverage computed in {time.time()-ts:.2f}s')
    
            with open(os.path.join(_SCRIPT_DIR, 'resources', 'output.txt'), 'w') as f:
                for e in KER_corner_list:
                    f.write(f'{e}\n')
    
            # ---- Greedy set cover ----------------------------------------
            print(f'Total: {len(_corners)} corners — {_corners}')
            cleared_set    = []
            chosed_set     = []
            working_list   = [list(row) for row in KER_corner_list]
    
            while len(cleared_set) < len(_corners):
                max_size, index = -1, -1
                for it in range(len(KER)):
                    working_list[it] = [c for c in working_list[it] if c not in cleared_set]
                    if len(working_list[it]) > max_size:
                        max_size = len(working_list[it])
                        index    = it
                if index == -1 or max_size == 0:
                    remaining = [c for c in _corners if c not in cleared_set]
                    print(f'[WARNING] Set cover stalled. Uncovered: {remaining}')
                    break
                chosed_set.append(index)
                cleared_set.extend(working_list[index])
    
            # ---- Build geodesic graph ------------------------------------
            geodesic = Geodesic(poly)
            try:
                geodesic.build()
                print('[INFO] Geodesic graph built.')
            except Exception as e:
                print(f'[ERROR] geodesic.build(): {e}')
                raise
    
            # Show selected KER visibility regions
            shapely_env = Polygon([(p.x(), p.y()) for p in poly])
            self._d_polygon(x, y, Qt.white, 3)
            self._d_vertex_labels(poly)
            for pt_i in chosed_set:
                ker_pt = KER[pt_i]
                kp     = Point(ker_pt.x(), ker_pt.y())
                if not shapely_env.buffer(1).contains(kp):
                    continue
                try:
                    evis = vis.Visibility_Polygon(ker_pt, _env, EPSILON)
                    evx, evy = poly_to_points(evis)
                    self.draw([Op.filled_polygon, evx, evy, 1, QColor(255, 220, 0, 60)])
                except Exception as e:
                    print(f'[ERROR] vis poly for KER[{pt_i}]: {e}')
            for pt_i in chosed_set:
                self._d_dot(KER[pt_i].x(), KER[pt_i].y(), 9, QColor(0, 160, 255))
            self.execute()
            time.sleep(1.5)
    
            # ---- Patrol path (shortest paths between guard positions) ----
            path_lines = []
            for it in range(len(chosed_set)):
                for jt in range(it + 1, len(chosed_set)):
                    ia, ib  = chosed_set[it], chosed_set[jt]
                    pa, pb  = [KER[ia].x(), KER[ia].y()], [KER[ib].x(), KER[ib].y()]
                    try:
                        sp = geodesic.shortest_path(pa, pb)
                    except KeyError:
                        with suppress_output():
                            raw = _env.shortest_path(KER[ia], KER[ib], EPSILON)
                        sp = [vg.Point(p.x(), p.y()) for p in raw.path()]
                    for kt in range(len(sp) - 1):
                        add_unique_linestring(path_lines,
                            LineString([Point(sp[kt].x, sp[kt].y),
                                        Point(sp[kt+1].x, sp[kt+1].y)]))
    
            # ---- Corner intersection points with patrol path -------------
            intersection_points = {}
            for j in _corners:
                vispol  = vis.Visibility_Polygon(poly[j], _env, EPSILON)
                a, b, c = poly[(j-1) % len(poly)], poly[j], poly[(j+1) % len(poly)]
                ex, ey  = poly_to_points(vispol)
                ex.reverse(); ey.reverse()
    
                v1 = np.array([b.x()-a.x(), b.y()-a.y()])
                v2 = np.array([b.x()-c.x(), b.y()-c.y()])
                v1 = v1 / np.linalg.norm(v1) * 10000
                v2 = v2 / np.linalg.norm(v2) * 10000
    
                subj = [(ex[k], ey[k]) for k in range(len(ex))]
                clip = [(b.x(), b.y()), (v1[0]+b.x(), v1[1]+b.y()), (v2[0]+b.x(), v2[1]+b.y())]
                pc   = pyclipper.Pyclipper()
                pc.AddPath(clip, pyclipper.PT_CLIP, True)
                pc.AddPath(subj, pyclipper.PT_SUBJECT, True)
                sol  = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                visi_pol = sol[0]
    
                seg_poly      = Polygon([(visi_pol[k][0], visi_pol[k][1]) for k in range(len(visi_pol))])
                seg_poly_buff = seg_poly.buffer(1)
    
                intersection_points[j] = []
                for seg in path_lines:
                    inter = seg.intersection(seg_poly_buff)
                    if not inter.is_empty and inter.geom_type == 'LineString':
                        for coord in [inter.coords[0], inter.coords[1]]:
                            add_unique_point(intersection_points[j], Point(coord))
                        intersection_points[j] = [
                            pt for pt in intersection_points[j]
                            if seg_poly.boundary.distance(pt) < 2
                        ]
    
            # Patrol path display
            self._d_polygon(x, y, Qt.white, 3)
            self._d_vertex_labels(poly)
            for p in path_lines:
                self._d_glow_line(p.coords[0][0], p.coords[0][1],
                                  p.coords[1][0], p.coords[1][1],
                                  C_PATH, width=3)
            self.execute()
            time.sleep(1.5)
    
            # ---- Build graph data structures for observer queries --------
            graph  = construct_graph(path_lines)
            dense, scipy_g, scipy_src, bi_map, vertices = \
                construct_graph_dense(path_lines, intersection_points)
            dist_matrix = floyd_warshall(scipy_g)
    
            # Precompute per-vertex distance vectors to each corner's sources
            vectors_org = []
            for ver_id in range(len(vertices)):
                vect = {}
                for corn in _corners:
                    min_d = float('inf')
                    for src in scipy_src[corn]:
                        if src is not None and dist_matrix[ver_id][src] < min_d:
                            min_d = dist_matrix[ver_id][src]
                    vect[corn] = min_d
                vectors_org.append(vect)
    
            rows, cols, _ = sp_find(scipy_g)
            total_edges   = list(zip(cols, rows))
            # ---- Save cache for next run -------------------------
            _save_cache(fingerprint, {
                'corners':             list(_corners),
                'asso':                dict(_asso),
                'KER_coords':          [(p.x(), p.y()) for p in KER],
                'lst_edges_KER':       [[(a.x(), a.y()), (b.x(), b.y())] for e in lst_edges_KER for a, b in [list(e)]],
                'chosed_set':          list(chosed_set),
                'path_lines':          list(path_lines),
                'intersection_points': {k: list(v) for k, v in intersection_points.items()},
                'dist_matrix':         dist_matrix,
                'vectors_org':         list(vectors_org),
                'dense':               dense,
                'scipy_src':           dict(scipy_src),
                'bi_map':              bi_map,
                'vertices':            list(vertices),
                'graph':               dict(graph),
            }, geodesic)

        # ---- Initial interactive positions ---------------------------
        evader_pt = Point(0, 0)
        while not evader_pt.within(shapely_env):
            evader_pt = Point(random.randint(-480, 480), random.randint(-480, 480))

        self.draggable_point_evader   = QPoint(int(evader_pt.x), int(evader_pt.y))
        self.draggable_point_observer = QPoint(int(path_lines[0].coords[0][0]),
                                               int(path_lines[0].coords[0][1]))
        self.draggable_point_corner   = _corners[0]
        self.point_radius             = POINT_RADIUS
        self.shapely_polygon          = shapely_env
        self.lines                    = path_lines
        self.poly                     = poly
        self.corners                  = _corners

        # ---- Build Voronoi skeleton ----------------------------------
        print('[SKEL] Building Voronoi skeleton…')
        skel_nodes, skel_adj, skel_edges = build_voronoi_skeleton(shapely_env)
        self._skel_nodes = skel_nodes
        self._skel_adj   = skel_adj
        self._skel_edges = skel_edges
        print(f'[SKEL] {len(skel_nodes)} nodes, {len(skel_edges)} edges')

        # ---- Interactive loop ----------------------------------------
        while True:
            ex_ = self.draggable_point_evader.x()
            ey_ = self.draggable_point_evader.y()
            ox_ = self.draggable_point_observer.x()
            oy_ = self.draggable_point_observer.y()
            act = self.draggable_point_corner

            # In auto mode, project the free-moving observer onto the roadmap
            # for all guard-opt and α computations.
            if self.auto_evader and self._roadmap_pursuer and self._roadmap_obs_pos is not None:
                # Roadmap pursuer: position is already on the path — use directly
                ao_x, ao_y = self._roadmap_obs_pos
                ox_comp, oy_comp = ao_x, ao_y
            elif self.auto_evader and self._auto_observer_pos is not None:
                ao_x, ao_y = self._auto_observer_pos
                ao_pt = Point(ao_x, ao_y)
                best_proj, best_d = None, float('inf')
                for _pl in path_lines:
                    _proj = _pl.interpolate(_pl.project(ao_pt))
                    _d = ao_pt.distance(_proj)
                    if _d < best_d:
                        best_d, best_proj = _d, _proj
                ox_comp = best_proj.x if best_proj else ao_x
                oy_comp = best_proj.y if best_proj else ao_y
            else:
                ao_x, ao_y = float(ox_), float(oy_)
                ox_comp, oy_comp = float(ox_), float(oy_)

            # --- Polygon + patrol paths ---
            self._d_polygon(x, y, Qt.white, 3)
            self._d_vertex_labels(poly)
            for p in path_lines:
                self._d_glow_line(p.coords[0][0], p.coords[0][1],
                                  p.coords[1][0], p.coords[1][1],
                                  C_PATH, width=3)

            # --- Intersection points for active corner ---
            for pt in list(intersection_points[act]):
                self._d_dot(pt.x, pt.y, 5, C_INTER_PT)

            # --- Evader (red circle) ---
            er = 14 if self.dragging_evader else POINT_RADIUS + 2
            self._d_ring(ex_, ey_, er + 3, QColor(200, 0, 0, 80), width=6)
            self._d_dot(ex_, ey_, er, C_EVADER)

            # --- Observer/pursuer (green circle) ---
            or_ = 14 if self.dragging_observer else POINT_RADIUS + 2
            self._d_ring(ox_, oy_, or_ + 3, QColor(0, 150, 50, 80), width=6)
            self._d_dot(ox_, oy_, or_, C_OBSERVER)

            # --- Active corner (cyan, larger) ---
            cr = 14 if self.dragging_corner else POINT_RADIUS + 2
            self._d_ring(poly[act].x(), poly[act].y(), cr + 3, QColor(0, 180, 200, 80), width=6)
            self._d_dot(poly[act].x(), poly[act].y(), cr, C_CORNER)

            # --- Compute evader geodesic distance to each corner ---
            path_lengths = {}
            for cors in _corners:
                pt_a = [ex_, ey_]
                pt_b = [poly[cors].x(), poly[cors].y()]
                try:
                    sp = geodesic.shortest_path(pt_a, pt_b)
                except KeyError:
                    with suppress_output():
                        raw = _env.shortest_path(
                            vis.Point(ex_, ey_), vis.Point(poly[cors].x(), poly[cors].y()), EPSILON)
                    sp = [vg.Point(p.x(), p.y()) for p in raw.path()]
                pl = sum(Point(sp[i].x, sp[i].y).distance(Point(sp[i+1].x, sp[i+1].y))
                         for i in range(len(sp) - 1))
                path_lengths[cors] = pl if pl > 0 else 1e-9

            # --- Compute optimal guard edge (minimise α) ---
            edge_points = []
            for edge in total_edges:
                store = []
                for corn in _corners:
                    d1    = vectors_org[edge[0]][corn] / path_lengths[corn]
                    d2    = vectors_org[edge[1]][corn] / path_lengths[corn]
                    d1_r  = vectors_org[edge[0]][corn]
                    d2_r  = vectors_org[edge[1]][corn]
                    elen  = vertices[edge[0]].distance(vertices[edge[1]])
                    v1, v2 = edge[0], edge[1]
                    types  = ['increasing', 'decreasing', 'mixed']

                    if d1 < d2:
                        b = (d2_r + elen - d1_r) / 2
                        m1, c1 = find_slope_and_intercept((0, d1), (b, d1 + b / path_lengths[corn]))
                        if np.isclose(elen, b, atol=1e-4):
                            store.append((types[0], m1, c1))
                        else:
                            m2, c2 = find_slope_and_intercept((elen, d2), (b, d1 + b / path_lengths[corn]))
                            store.append((types[2], m1, c1, m2, c2, b))
                    else:
                        b = elen - (d1_r + elen - d2_r) / 2
                        m1, c1 = find_slope_and_intercept((elen, d2), (b, d1 + b / path_lengths[corn]))
                        if np.isclose(0, b, atol=1e-4):
                            store.append((types[1], m1, c1))
                        else:
                            m2, c2 = find_slope_and_intercept((0, d1), (b, d1 + b / path_lengths[corn]))
                            store.append((types[2], m2, c2, m1, c1, b))

                min_pt = process_functions(store, x_min=0, x_max=elen)
                edge_points.append((min_pt[0], min_pt[1], v1, v2))

            opt = min(edge_points, key=lambda p: p[1])

            # Draw optimal edge (sky-blue highlight)
            v2x, v2y = vertices[opt[2]].x, vertices[opt[2]].y
            v3x, v3y = vertices[opt[3]].x, vertices[opt[3]].y
            self._d_glow_line(v2x, v2y, v3x, v3y, C_OPT_EDGE, width=8)
            self._d_dot(v2x, v2y, 10, C_OPT_EDGE)
            self._d_dot(v3x, v3y, 10, C_OPT_EDGE)

            # Guard position (magenta diamond — distinct from evader)
            guard = interpolate_point(vertices[opt[2]], vertices[opt[3]], opt[0])
            self._d_ring(guard.x, guard.y, 16, QColor(255, 50, 255, 100), width=8)
            self._d_diamond(guard.x, guard.y, 12, C_GUARD)

            # --- Shortest path on graph: observer → guard_opt ---
            _, _obs_to_guard_path = dijkstra(graph, (ox_comp, oy_comp), Point(guard.x, guard.y))
            if len(_obs_to_guard_path) >= 2:
                C_SP_LINE = QColor(0, 220, 255, 200)
                for _si in range(len(_obs_to_guard_path) - 1):
                    _ax, _ay = _obs_to_guard_path[_si]
                    _bx, _by = _obs_to_guard_path[_si + 1]
                    self._d_glow_line(_ax, _ay, _bx, _by, C_SP_LINE, width=3)

            # --- α values per corner via Dijkstra ---
            # Use roadmap projection of actual observer in auto mode
            add_observer_to_graph(dense, vertices, graph, bi_map,
                                  Point(ox_comp, oy_comp))
            alphas = []
            path_length_ref = max(path_lengths.values()) or 1
            for ct in _corners:
                dists = []
                for pt in list(intersection_points[ct]):
                    sp_d, _ = dijkstra(graph, pt,
                                       Point(ox_comp, oy_comp))
                    dists.append(round(sp_d / path_length_ref, 4))
                alphas.append(min(dists) if dists else float('inf'))

            # --- HUD overlay ---
            alpha_lines = '  '.join(f'\u03B1{i}={a:.2f}' for i, a in enumerate(alphas))
            opt_alpha   = opt[1]
            self._d_text(-490, -460, f'\u03B1 opt = {opt_alpha:.4f}', size=14)
            self._d_text(-490, -490, alpha_lines, size=11)
            if self.auto_evader and self._roadmap_pursuer and self._roadmap_obs_pos is not None:
                self._d_text(-490, 470, f'Observer roadmap ({ao_x:.0f},{ao_y:.0f})', size=11)
                self._d_text(-490, 490, f'Guard ({guard.x:.0f},{guard.y:.0f})', size=10)
            elif self.auto_evader and self._auto_observer_pos is not None:
                self._d_text(-490, 470, f'Observer free ({ao_x:.0f},{ao_y:.0f})', size=11)
                self._d_text(-490, 490, f'Obs roadmap ({ox_comp:.0f},{oy_comp:.0f})  Guard ({guard.x:.0f},{guard.y:.0f})', size=10)
            else:
                self._d_text(-490,  470, f'Observer ({ox_:.0f}, {oy_:.0f})', size=11)
                self._d_text(-490,  490, f'Guard    ({guard.x:.0f}, {guard.y:.0f})', size=11)

            # Legend (top-right)
            lx, ly, ls = 270, 480, 12
            self._d_dot(lx,      ly,      8, C_EVADER);   self._d_text(lx+15, ly-5,  'Evader',        ls)
            self._d_dot(lx,      ly - 28, 8, C_OBSERVER); self._d_text(lx+15, ly-33, 'Observer',      ls)
            self._d_dot(lx,      ly - 56, 8, C_CORNER);   self._d_text(lx+15, ly-61, 'Corner',        ls)
            self._d_diamond(lx,  ly - 84, 8, C_GUARD);    self._d_text(lx+15, ly-89, 'Guard opt',     ls)
            self._d_glow_line(lx-8, ly-108, lx+8, ly-108, QColor(0, 220, 255, 200), width=3); self._d_text(lx+15, ly-113, 'Obs→Guard path', ls)
            # Auto-mode entries (always shown so user knows what A does)
            C_AUTO_OBS = QColor(255, 220, 0)
            C_SKEL_LEG = QColor(80, 80, 180, 200)
            C_PATH_SKEL_LEG = QColor(255, 120, 0, 220)
            self._d_dot(lx,      ly - 136, 8, C_AUTO_OBS);  self._d_text(lx+15, ly-141, 'Auto observer', ls)
            self._d_glow_line(lx-8, ly-160, lx+8, ly-160, C_SKEL_LEG, width=1);     self._d_text(lx+15, ly-165, 'Skeleton',      ls)
            self._d_glow_line(lx-8, ly-184, lx+8, ly-184, C_PATH_SKEL_LEG, width=2); self._d_text(lx+15, ly-189, 'Evader path',   ls)
            self._d_ring(lx,     ly - 208, 6, QColor(255, 180, 0, 200), width=2);   self._d_text(lx+15, ly-213, 'Evader dest',   ls)

            # --- Voronoi skeleton (faint) + auto-evader mode ----------
            C_SKEL = QColor(80, 80, 120, 120)
            for (sx1, sy1), (sx2, sy2) in self._skel_edges:
                self._d_glow_line(sx1, sy1, sx2, sy2, C_SKEL, width=1)

            if self.auto_evader:
                self._d_text(-490, -430, '[A] AUTO EVADER ON', size=11)
                pursuer_label = 'ROADMAP' if self._roadmap_pursuer else 'FREE'
                self._d_text(-490, -410, f'[P] pursuer: {pursuer_label}', size=11)
                # Step evader along skeleton path
                speed = self._evader_speed
                while speed > 0 and self._skel_path:
                    if self._skel_seg_idx >= len(self._skel_path) - 1:
                        # Reached destination — pick new one
                        cur_node = self._skel_path[-1]
                        dst = _skeleton_pick_destination(
                            self._skel_nodes, self._skel_adj, cur_node)
                        new_path = _skeleton_path(
                            self._skel_nodes, self._skel_adj, cur_node, dst)
                        if new_path and len(new_path) > 1:
                            self._skel_path    = new_path
                            self._skel_seg_idx = 0
                            self._skel_path_pos = 0.0
                        else:
                            break

                    i0 = self._skel_path[self._skel_seg_idx]
                    i1 = self._skel_path[self._skel_seg_idx + 1]
                    x0, y0 = self._skel_nodes[i0]
                    x1_, y1_ = self._skel_nodes[i1]
                    seg_len = math.hypot(x1_ - x0, y1_ - y0)
                    remaining = seg_len - self._skel_path_pos

                    if speed >= remaining:
                        speed -= remaining
                        self._skel_seg_idx  += 1
                        self._skel_path_pos  = 0.0
                        nx_x, nx_y = x1_, y1_
                    else:
                        self._skel_path_pos += speed
                        t = self._skel_path_pos / seg_len if seg_len > 0 else 0
                        nx_x = x0 + t * (x1_ - x0)
                        nx_y = y0 + t * (y1_ - y0)
                        speed = 0

                    pt = Point(nx_x, nx_y)
                    if self.shapely_polygon and self.shapely_polygon.contains(pt):
                        self.draggable_point_evader = QPoint(int(nx_x), int(nx_y))

                # Highlight current path on skeleton
                C_PATH_SKEL = QColor(255, 120, 0, 200)
                for si in range(len(self._skel_path) - 1):
                    n0 = self._skel_path[si]
                    n1 = self._skel_path[si + 1]
                    p0 = self._skel_nodes[n0]
                    p1 = self._skel_nodes[n1]
                    self._d_glow_line(p0[0], p0[1], p1[0], p1[1], C_PATH_SKEL, width=2)
                # Destination marker
                if self._skel_path:
                    dst_n = self._skel_path[-1]
                    dx_, dy_ = self._skel_nodes[dst_n]
                    self._d_ring(dx_, dy_, 12, QColor(255, 180, 0, 180), width=2)

                # --- Step autonomous observer toward guard_opt ---
                if self._roadmap_pursuer and self._roadmap_obs_pos is not None:
                    pspeed = self._pursuer_speed
                    gx, gy = guard.x, guard.y
                    px, py = self._roadmap_obs_pos
                    self._dbg_frame_count += 1

                    # ── Replan every frame (stable-node approach) ──────────────────────
                    # Dijkstra inserts virtual nodes for pursuer (p) and guard (g),
                    # splitting whichever edges they fall on. The returned path is:
                    #   [p_virtual, base_node_1, base_node_2, ..., g_virtual]
                    # We strip the two virtual endpoints and keep only the fixed base
                    # graph nodes as the committed sequence.
                    _plan_dist, wp = dijkstra(graph, (px, py), Point(gx, gy))
                    if len(wp) < 2:
                        nearest = min(graph.keys(),
                                      key=lambda n: math.hypot(n[0] - gx, n[1] - gy))
                        _plan_dist, wp = dijkstra(graph, (px, py), Point(*nearest))

                    # Base nodes = everything between virtual start and virtual end
                    new_base = wp[1:-1]  # may be empty if pursuer & guard share an edge

                    # When pursuer is exactly at a graph node, Dijkstra produces
                    # wp = [start_virtual=node, node, ..., end] so new_base[0] == pursuer
                    # pos. Strip those leading "at-position" nodes so we don't waste a
                    # frame targeting our own location.
                    # IMPORTANT: also advance base_idx past that node so that _cur_target
                    # (computed below) reflects the node AFTER the one we're at. Without
                    # this, _cur_target still points at the AT node, making the next node
                    # look like a NODE-CHANGE rather than a TAIL-UPDATE.
                    while new_base and math.hypot(new_base[0][0] - px, new_base[0][1] - py) < 0.5:
                        _at_node = new_base[0]
                        _dbg(f'[PURSUER] STRIP AT-POS node={_at_node} (pursuer already here)')
                        self._roadmap_edge_behind = _at_node
                        # advance base_idx past this node if the old path still targets it
                        if (self._roadmap_base_idx < len(self._roadmap_base_path) and
                                math.hypot(self._roadmap_base_path[self._roadmap_base_idx][0] - _at_node[0],
                                           self._roadmap_base_path[self._roadmap_base_idx][1] - _at_node[1]) < 0.5):
                            self._roadmap_base_idx += 1
                        new_base = new_base[1:]

                    if len(new_base) == 0:
                        # Same edge as guard — go straight to guard, reset base path
                        if self._roadmap_base_path:
                            _dbg(f'[PURSUER][frame={self._dbg_frame_count}] '
                                 f'DIRECT (same edge) pursuer=({px:.1f},{py:.1f}) '
                                 f'guard=({gx:.1f},{gy:.1f})')
                        self._roadmap_base_path = []
                        self._roadmap_base_idx  = 0
                        self._roadmap_direct    = True
                    else:
                        _cur_target = (self._roadmap_base_path[self._roadmap_base_idx]
                                       if self._roadmap_base_idx < len(self._roadmap_base_path)
                                       else None)

                        # Node identity helper
                        def _node_eq(a, b, tol=0.5):
                            return math.hypot(a[0]-b[0], a[1]-b[1]) < tol

                        # ── Strip ONLY the Dijkstra artifact ──────────────────────────────
                        # Dijkstra injects a virtual start for the pursuer, splitting the
                        # current edge (behind_node → forward_node). It always returns BOTH
                        # endpoints: [behind_node, forward_node, ...]. Strip behind_node ONLY
                        # when new_base[1] == _cur_target (i.e., the path immediately doubles
                        # back then continues forward — pure artifact). If behind_node is
                        # present but new_base[1] differs, it is a genuine reroute (go
                        # backward along the current edge then take a different branch);
                        # in that case keep it so the pursuer physically retraces the edge.
                        _new_base = new_base[:]
                        if (self._roadmap_edge_behind is not None and
                                _new_base and
                                _node_eq(_new_base[0], self._roadmap_edge_behind) and
                                (
                                    # Case A: mid-edge — behind_node[1] == cur_target
                                    (_cur_target is not None and
                                     len(_new_base) >= 2 and
                                     _node_eq(_new_base[1], _cur_target))
                                    or
                                    # Case B: DIRECT mode — we just departed from behind_node
                                    # along a confirmed edge; find_edge_for_point(eps=3) near
                                    # that node is ambiguous and may return a wrong edge, making
                                    # Dijkstra route us back. Strip it — we're already past it.
                                    (_cur_target is None and self._roadmap_direct)
                                )):
                            _dbg(f'[PURSUER][frame={self._dbg_frame_count}] '
                                 f'STRIP ARTIFACT node={_new_base[0]}')
                            _new_base.pop(0)

                        if not _new_base:
                            # Pursuer and guard are on the same edge — go straight to guard
                            self._roadmap_base_path = []
                            self._roadmap_base_idx  = 0
                            self._roadmap_direct    = True
                            _dbg(f'[PURSUER][frame={self._dbg_frame_count}] '
                                 f'SAME EDGE — direct to guard')
                        elif _cur_target is not None and _node_eq(_new_base[0], _cur_target):
                            # Immediate next node is unchanged — only update the tail
                            self._roadmap_base_path = (
                                self._roadmap_base_path[:self._roadmap_base_idx] + _new_base
                            )
                            self._roadmap_direct = False
                            _dbg(f'[PURSUER][frame={self._dbg_frame_count}] '
                                 f'TAIL-UPDATE next={_cur_target} '
                                 f'new_tail_len={len(_new_base)}')
                        else:
                            # Genuine route change — commit to new sequence
                            self._roadmap_direct = False
                            _dbg(f'[PURSUER][frame={self._dbg_frame_count}] '
                                 f'NODE-CHANGE old_next={_cur_target} '
                                 f'new_next={_new_base[0]} '
                                 f'pursuer=({px:.1f},{py:.1f}) '
                                 f'guard=({gx:.1f},{gy:.1f}) '
                                 f'path_len={_plan_dist:.1f} '
                                 f'base_nodes={_new_base}')
                            self._roadmap_base_path = _new_base
                            self._roadmap_base_idx  = 0

                    self._roadmap_guard_pos = (gx, gy)

                    # ── Walk toward current committed target ───────────────────────────
                    _px_start, _py_start = px, py
                    rem = pspeed
                    while rem > 0:
                        if self._roadmap_base_idx < len(self._roadmap_base_path):
                            tx, ty = self._roadmap_base_path[self._roadmap_base_idx]
                        elif self._roadmap_guard_pos is not None:
                            tx, ty = self._roadmap_guard_pos
                        else:
                            break
                        dx_, dy_ = tx - px, ty - py
                        d_wp = math.hypot(dx_, dy_)
                        if d_wp < 1e-4:          # already at this node
                            if self._roadmap_base_idx < len(self._roadmap_base_path):
                                self._roadmap_edge_behind = (tx, ty)
                                self._roadmap_base_idx += 1
                            else:
                                break
                            continue
                        if d_wp <= rem:
                            px, py = tx, ty
                            rem -= d_wp
                            if self._roadmap_base_idx < len(self._roadmap_base_path):
                                # Record the node we just physically arrived at
                                self._roadmap_edge_behind = (tx, ty)
                                self._roadmap_base_idx += 1
                            else:
                                break   # reached guard, stop
                        else:
                            # Partial step toward guard allowed only when:
                            #   a) still between two base nodes (always safe, on-edge), OR
                            #   b) DIRECT mode — guard is confirmed on the same edge as
                            #      pursuer, so moving toward it stays on-graph.
                            # In all other cases (base exhausted by walking through nodes)
                            # stop here; Dijkstra will re-plan from this stable position.
                            if (self._roadmap_base_idx < len(self._roadmap_base_path)
                                    or self._roadmap_direct):
                                px += (dx_ / d_wp) * rem
                                py += (dy_ / d_wp) * rem
                            rem = 0
                    self._roadmap_obs_pos = [px, py]

                    # ── Oscillation detector ──────────────────────────────────────────
                    _moved = math.hypot(px - _px_start, py - _py_start)
                    if _moved > 0.01:
                        _cur_vec = ((px - _px_start) / _moved, (py - _py_start) / _moved)
                        if self._dbg_prev_move_vec is not None:
                            _dot = (_cur_vec[0] * self._dbg_prev_move_vec[0] +
                                    _cur_vec[1] * self._dbg_prev_move_vec[1])
                            if _dot < -0.5:
                                self._dbg_dir_flips += 1
                                _dbg(f'[PURSUER][frame={self._dbg_frame_count}] '
                                     f'DIRECTION FLIP #{self._dbg_dir_flips} '
                                     f'dot={_dot:.2f} moved={_moved:.2f} '
                                     f'pos=({px:.1f},{py:.1f}) guard=({gx:.1f},{gy:.1f}) '
                                     f'base_idx={self._roadmap_base_idx}/{len(self._roadmap_base_path)} '
                                     f'base_path={self._roadmap_base_path}')
                            else:
                                self._dbg_dir_flips = 0
                        self._dbg_prev_move_vec = _cur_vec
                    self._dbg_prev_pursuer_pos = [_px_start, _py_start]

                    # Sync green dot
                    self.draggable_point_observer = QPoint(int(px), int(py))
                    # Draw committed base-node path
                    C_RM_PATH = QColor(255, 220, 0, 160)
                    _draw_pts = ([( px,  py)] +
                                 self._roadmap_base_path[self._roadmap_base_idx:] +
                                 [(gx, gy)])
                    for si in range(len(_draw_pts) - 1):
                        ax, ay = _draw_pts[si]
                        bx, by = _draw_pts[si + 1]
                        self._d_glow_line(ax, ay, bx, by, C_RM_PATH, width=2)
                    # Dot + index label on each remaining base node
                    _remaining_nodes = self._roadmap_base_path[self._roadmap_base_idx:]
                    for _ni, _bn in enumerate(_remaining_nodes):
                        self._d_dot(_bn[0], _bn[1], 5, QColor(255, 220, 0, 200))
                        self._d_text(int(_bn[0]) + 7, int(_bn[1]) + 7,
                                     str(_ni + 1), size=9)

                    # HUD path string: p → 1 → 2 → ... → g
                    _node_strs = ([f'p({px:.0f},{py:.0f})'] +
                                  [f'{_ni+1}({_bn[0]:.0f},{_bn[1]:.0f})'
                                   for _ni, _bn in enumerate(_remaining_nodes)] +
                                  [f'g({gx:.0f},{gy:.0f})'])
                    _path_str = ' → '.join(_node_strs)
                    self._d_text(-490, 450, _path_str, size=9)
                    _dbg(f'[PURSUER][frame={self._dbg_frame_count}] PATH {_path_str}')

                elif self._auto_observer_pos is not None:
                    pspeed = self._pursuer_speed
                    gx, gy = guard.x, guard.y
                    px, py = self._auto_observer_pos
                    ddx, ddy = gx - px, gy - py
                    dist_to_guard = math.hypot(ddx, ddy)
                    if dist_to_guard > pspeed:
                        px += (ddx / dist_to_guard) * pspeed
                        py += (ddy / dist_to_guard) * pspeed
                    else:
                        px, py = gx, gy
                    self._auto_observer_pos = [px, py]

                    # Draw autonomous observer — yellow hexagon approximated as ring+dot
                    C_AUTO_OBS = QColor(255, 220, 0)
                    self._d_ring(px, py, POINT_RADIUS + 4, QColor(255, 220, 0, 80), width=5)
                    self._d_dot(px, py, POINT_RADIUS + 1, C_AUTO_OBS)
                    # Dashed line: auto-observer → roadmap projection
                    self.draw([Op.dotted_line, px, py, ox_comp, oy_comp, 1,
                               QColor(255, 220, 0, 120)])
                    # Roadmap projection marker
                    self._d_ring(ox_comp, oy_comp, 5, QColor(255, 220, 0, 180), width=2)
                    # Line: roadmap projection → guard
                    self._d_glow_line(ox_comp, oy_comp, guard.x, guard.y,
                                      QColor(255, 220, 0, 160), width=1)
            else:
                self._d_text(-490, -430, '[A] auto evader off', size=11)
                pursuer_label = 'ROADMAP' if self._roadmap_pursuer else 'FREE'
                self._d_text(-490, -410, f'[P] pursuer: {pursuer_label}', size=11)

            self.execute()


# ---------------------------------------------------------------------------
# Polygon drawing (pygame) → returns list[vis.Point]
# ---------------------------------------------------------------------------
def draw_polygon() -> list:
    pygame.init()
    W, H   = 700, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Draw Polygon — SPACE to close, BACKSPACE to undo, ESC to quit")

    WHITE = (255, 255, 255)
    RED   = (220, 50, 50)
    GRAY  = (60, 60, 60)

    def to_px(px, py):
        return int((px + 500) * W / 1000), int(H - (py + 500) * H / 1000)

    def from_px(px, py):
        return px * 1000 / W - 500, 500 - py * 1000 / H

    def shoelace(pts):
        n = len(pts)
        return sum((pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1]) for i in range(n)) / 2

    points  = []
    closed  = False
    font    = pygame.font.SysFont("monospace", 16)

    with open(FILE_NAME) as f:
        for row in csv.reader(f):
            points.append((float(row[0]), float(row[1])))
    if points:
        points.append(points[0])

    running = True
    while running:
        screen.fill(GRAY)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif not closed and event.type == pygame.MOUSEBUTTONDOWN:
                points.append(from_px(*pygame.mouse.get_pos()))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and len(points) > 2:
                    points.append(points[0])
                    closed = True
                    if shoelace(points[:-1]) > 0:
                        points = list(reversed(points))
                    print(f'Area = {shoelace(points[:-1]):.1f}')
                elif event.key == pygame.K_p:
                    print(points)
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    points, closed = [], False
                elif event.key == pygame.K_BACKSPACE and points:
                    points.pop()

        px_pts = [to_px(px, py) for px, py in points]
        if len(px_pts) > 1:
            pygame.draw.lines(screen, RED, False, px_pts, 2)

        hint = "SPACE=close  BACKSPACE=undo  ESC=done"
        screen.blit(font.render(hint, True, WHITE), (10, 10))
        pygame.display.flip()

    pygame.quit()

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
        poly = [vis.Point(p[0], p[1]) for p in points]
        with open(FILE_NAME, 'w') as f:
            for p in points:
                f.write(f'{p[0]:.2f}, {p[1]:.2f}\n')
        return poly
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def startup():
    poly = draw_polygon()
    if poly is None:
        print('[ERROR] No polygon drawn — exiting.')
        return
    app    = QApplication(sys.argv)
    window = Window()
    thread = threading.Thread(target=window.run, args=(poly,), daemon=True)
    thread.start()
    sys.exit(app.exec())


if __name__ == '__main__':
    startup()
