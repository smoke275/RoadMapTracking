"""Low-level geometry and math utilities — no Qt, no simulation state."""
import contextlib
import math
import os
import random
import sys

import numpy as np
import pyvisgraph as vg
import visilibity as vis
import shapely.geometry as sp_geom
from shapely.geometry import LineString, Point, Polygon

from config import EPSILON


@contextlib.contextmanager
def suppress_output():
    """Suppress C-level stdout/stderr (visilibity SWIG noise)."""
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
    boundary   = LineString(polygon)
    poly_shape = Polygon(polygon)
    far        = 1e6

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


def get_random_point_in_polygon(pol: Polygon) -> Point:
    minx, miny, maxx, maxy = pol.bounds
    while True:
        p = sp_geom.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if pol.contains(p):
            return p
