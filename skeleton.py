"""Voronoi-skeleton construction and navigation helpers for the auto-evader."""
import heapq
import math
import random
from collections import deque

import numpy as np
from shapely.geometry import LineString, Polygon
from scipy.spatial import Voronoi as SciVoronoi


def build_voronoi_skeleton(shapely_poly: Polygon, samples: int = 300):
    """
    Sample the polygon boundary, compute interior Voronoi edges, return:
      nodes  — list of (x, y) unique Voronoi vertices inside the polygon
      adj    — dict { node_idx: {neighbour_idx: edge_length, ...} }
      edges  — list of ((x1,y1), (x2,y2)) for drawing
    """
    poly_shrunken = shapely_poly.buffer(-2)
    if poly_shrunken.is_empty:
        poly_shrunken = shapely_poly

    boundary = shapely_poly.exterior
    step     = boundary.length / samples
    pts      = [boundary.interpolate(i * step) for i in range(samples)]
    coords   = np.array([(p.x, p.y) for p in pts])

    vor      = SciVoronoi(coords)
    edges    = []
    node_set = {}   # (x,y) rounded → int index

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
        mid = seg.interpolate(0.5, normalized=True)
        if not shapely_poly.contains(mid):
            continue
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

    idx_to_xy = {v: k for k, v in node_set.items()}
    nodes     = [idx_to_xy[i] for i in range(len(idx_to_xy))]
    adj: dict[int, dict[int, float]] = {i: {} for i in range(len(nodes))}

    for c0, c1 in edges:
        k0 = (round(c0[0], 1), round(c0[1], 1))
        k1 = (round(c1[0], 1), round(c1[1], 1))
        i0, i1 = node_set[k0], node_set[k1]
        d = math.hypot(c0[0] - c1[0], c0[1] - c1[1])
        adj[i0][i1] = d
        adj[i1][i0] = d

    return nodes, adj, edges


def nearest_node(nodes, x, y) -> int:
    """Return index of skeleton node nearest to (x, y)."""
    best, best_d = 0, float('inf')
    for i, (nx_, ny_) in enumerate(nodes):
        d = math.hypot(x - nx_, y - ny_)
        if d < best_d:
            best_d, best = d, i
    return best


def skeleton_path(nodes, adj, src_idx, dst_idx) -> list:
    """Dijkstra on the skeleton adjacency dict; returns list of node indices."""
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
    path, cur = [], dst_idx
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(src_idx)
    path.reverse()
    return path if path[0] == src_idx else []


def pick_destination(nodes, adj, src_idx, min_hops: int = 8) -> int:
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
