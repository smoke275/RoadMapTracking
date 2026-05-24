"""
KER pipeline: one-time setup (corners → asso → patrol path → graph)
and per-frame pure computations (optimal guard, alpha values).

`build(poly, renderer)` runs the full pipeline, displays intermediate states
through the renderer, and returns a frozen SimulationData object.
`compute_frame(...)` is called every simulation frame to derive guard and α.
"""
import itertools
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pyclipper
import seaborn as sns
import visilibity as vis
import pyvisgraph as vg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from scipy.sparse import find as sp_find
from scipy.sparse.csgraph import csgraph_from_dense, floyd_warshall
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon

from cache import load_cache, poly_fingerprint, save_cache
from config import EPSILON
from geometry import (add_unique_linestring, add_unique_point,
                      find_intersection, find_slope_and_intercept,
                      minimum_distance, poly_to_points, suppress_output)
from graph import (Geodesic, add_observer_to_graph, construct_graph,
                   construct_graph_dense, dijkstra)
from skeleton import build_voronoi_skeleton
from function_generator import process_functions

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_MILP = True
except ImportError:
    _HAS_MILP = False


def _greedy_set_cover(KER_corner_list, corners):
    """Greedy ln|C|-approximation. Used only as a fallback when MILP is
    unavailable or returns infeasible."""
    universe = set(corners)
    chosen   = []
    covered  = set()
    working  = [set(s) & universe for s in KER_corner_list]
    while covered < universe:
        best_i, best_n = -1, 0
        for i in range(len(working)):
            n = len(working[i] - covered)
            if n > best_n:
                best_n, best_i = n, i
        if best_i < 0:
            break
        chosen.append(best_i)
        covered |= working[best_i]
    return chosen


def solve_set_cover_ilp(KER_corner_list, corners):
    """
    Exact minimum set cover via integer linear programming.

    Selects the smallest subset of candidate indices (into ``KER_corner_list``)
    such that every corner in ``corners`` is covered by at least one selected
    candidate.

    A domination pre-processing step removes any candidate whose coverage is
    a (non-strict) subset of another candidate's coverage; these can never
    appear in an optimal cover. The reduced instance is solved exactly with
    ``scipy.optimize.milp``.

    Falls back to the greedy ``ln|C|``-approximation if ``scipy.optimize.milp``
    is unavailable or the ILP returns infeasible (which signals that the
    candidate set fails to cover some corner — a construction-level bug worth
    surfacing).
    """
    universe = list(corners)
    if not universe:
        return []

    n_cands  = len(KER_corner_list)
    uni_set  = frozenset(universe)
    cov_sets = [frozenset(KER_corner_list[i]) & uni_set for i in range(n_cands)]

    # ---- Domination pre-processing -------------------------------------
    # i is dominated by j (j ≠ i) if cov_sets[i] ⊆ cov_sets[j]; when sets
    # tie, keep the lower-indexed candidate.
    keep = []
    for i in range(n_cands):
        if not cov_sets[i]:
            continue
        dominated = False
        for j in range(n_cands):
            if i == j:
                continue
            if cov_sets[i] < cov_sets[j]:
                dominated = True
                break
            if cov_sets[i] == cov_sets[j] and j < i:
                dominated = True
                break
        if not dominated:
            keep.append(i)

    if not keep:
        print('[WARNING] All KER candidates have empty coverage; falling back to greedy.')
        return _greedy_set_cover(KER_corner_list, universe)

    if not _HAS_MILP:
        print('[INFO] scipy.optimize.milp unavailable; using greedy set cover.')
        return _greedy_set_cover(KER_corner_list, universe)

    # ---- ILP -----------------------------------------------------------
    n_red = len(keep)
    n_c   = len(universe)
    cidx  = {c: k for k, c in enumerate(universe)}

    # A[c, i] = 1 iff candidate keep[i] covers corner c.
    A = np.zeros((n_c, n_red), dtype=float)
    for ki, orig in enumerate(keep):
        for c in cov_sets[orig]:
            A[cidx[c], ki] = 1.0

    res = milp(
        c           = np.ones(n_red),
        constraints = LinearConstraint(A, lb=1.0, ub=np.inf),
        integrality = np.ones(n_red),
        bounds      = Bounds(lb=0.0, ub=1.0),
    )
    if not res.success or res.x is None:
        print('[WARNING] ILP infeasible (uncovered corners in K); falling back to greedy.')
        return _greedy_set_cover(KER_corner_list, universe)

    chosen = [keep[i] for i in range(n_red) if res.x[i] > 0.5]
    print(f'[INFO] ILP cover: {len(chosen)}/{n_cands} candidates '
          f'({n_red} after domination pre-processing).')
    return chosen


# ---------------------------------------------------------------------------
# Colours referenced during pipeline display
# ---------------------------------------------------------------------------
C_CORNER = Qt.cyan
C_PATH   = QColor(240, 200, 0)


# ---------------------------------------------------------------------------
# Result dataclass — everything the interactive loop needs from the pipeline
# ---------------------------------------------------------------------------
@dataclass
class SimulationData:
    poly:                list          # cleaned list of vis.Point
    x:                   list          # poly x-coords for drawing
    y:                   list          # poly y-coords for drawing
    shapely_env:         Polygon
    corners:             list          # concave corner indices
    asso:                dict          # corner_idx → clipped vis polygon
    env:                 vis.Environment
    graph:               dict          # patrol path dict graph
    dense:               np.ndarray
    bi_map:              object        # bidict
    vertices:            list          # list of shapely Point
    dist_matrix:         np.ndarray
    vectors_org:         list
    path_lines:          list          # list of LineString
    intersection_points: dict
    total_edges:         list
    geodesic:            Geodesic
    skel_nodes:          list
    skel_adj:            dict
    skel_edges:          list


# ---------------------------------------------------------------------------
# Per-frame pure computation results
# ---------------------------------------------------------------------------
@dataclass
class FrameComputed:
    guard:             Point
    opt_edge_v1:       int
    opt_edge_v2:       int
    opt_alpha:         float
    path_lengths:      dict
    alphas:            list
    obs_to_guard_path: list


# ---------------------------------------------------------------------------
# Corner association (clipped angular visibility wedge)
# ---------------------------------------------------------------------------
def compute_corner_asso(poly, corners, env) -> dict:
    """For each concave corner compute the clipped angular visibility wedge."""
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
# One-time pipeline build
# ---------------------------------------------------------------------------
def build(poly, renderer=None, force_recompute: bool = False) -> SimulationData:
    """
    Run the full KER pipeline and return a SimulationData object.

    renderer  — a Window instance (or any object with _d_polygon/_d_dot/etc.
                convenience methods and execute()).  Pass None to skip all
                intermediate display (useful for headless testing).
    """
    def _render(*args, **kwargs):
        """Call renderer method if renderer is available."""
        pass  # replaced below

    def _show():
        if renderer:
            renderer.execute()

    # ---- Clean polygon --------------------------------------------------
    clean = []
    for p in poly:
        if not clean or p.x() != clean[-1].x() or p.y() != clean[-1].y():
            clean.append(p)
    while len(clean) > 1 and clean[-1].x() == clean[0].x() and clean[-1].y() == clean[0].y():
        clean.pop()
    poly = clean

    x = [p.x() for p in poly]
    y = [p.y() for p in poly]

    # ---- Environment ----------------------------------------------------
    walls = vis.Polygon([p for p in reversed(poly)])
    env   = vis.Environment([walls])
    env.PRINTING_DEBUG_DATA = False

    shapely_env = ShapelyPolygon([(p.x(), p.y()) for p in poly])

    # ---- Try cache -------------------------------------------------------
    fingerprint        = poly_fingerprint(poly)
    _cached, _cached_geo = (None, None) if force_recompute else load_cache(fingerprint)

    if _cached is not None:
        corners             = list(_cached['corners'])
        asso                = dict(_cached['asso'])
        path_lines          = _cached['path_lines']
        intersection_points = _cached['intersection_points']
        dist_matrix         = _cached['dist_matrix']
        vectors_org         = _cached['vectors_org']
        dense               = _cached['dense']
        bi_map              = _cached['bi_map']
        vertices            = _cached['vertices']
        graph               = _cached['graph']
        geodesic            = Geodesic(poly)
        geodesic.graph      = _cached_geo
        scipy_g             = csgraph_from_dense(dense, null_value=-1)
        palette             = sns.color_palette('husl', len(corners))

        if renderer:
            for j, n in zip(corners, range(len(corners))):
                vp = asso.get(j)
                if vp is None:
                    continue
                vx = [vp[k][0] for k in range(len(vp))]
                vy = [vp[k][1] for k in range(len(vp))]
                col = QColor()
                col.setRgbF(palette[n][0], palette[n][1], palette[n][2], 0.35)
                renderer._d_filled_polygon(vx, vy, col)
            renderer._d_polygon(x, y, Qt.white, 3)
            renderer._d_vertex_labels(poly)
            for b in corners:
                renderer._d_dot(poly[b].x(), poly[b].y(), 6, C_CORNER)
            for p in path_lines:
                renderer._d_glow_line(p.coords[0][0], p.coords[0][1],
                                      p.coords[1][0], p.coords[1][1], C_PATH, width=3)
            _show()
            time.sleep(0.5)

        rows, cols_sp, _ = sp_find(scipy_g)
        total_edges = list(zip(cols_sp, rows))

    else:
        # ---- Detect concave corners -------------------------------------
        corners = []
        for i in range(len(poly)):
            a, b, c = poly[(i - 1) % len(poly)], poly[i], poly[(i + 1) % len(poly)]
            s = [b.x() - a.x(), b.y() - a.y()]
            t = [c.x() - b.x(), c.y() - b.y()]
            if s[0] * t[1] - t[0] * s[1] > 0:
                corners.append(i)

        if renderer:
            renderer._d_polygon(x, y, Qt.white, 3)
            renderer._d_vertex_labels(poly)
            for b in corners:
                renderer._d_dot(poly[b].x(), poly[b].y(), 6, C_CORNER)
            _show()
            time.sleep(1.5)

        # ---- Angular visibility wedges ----------------------------------
        asso = compute_corner_asso(poly, corners, env)
        print(f'Corners: {corners}')

        # ---- KER boundary contraction -----------------------------------
        boundary        = list(range(len(poly)))
        current_corners = corners.copy()
        corner_clusters = {k: [k] for k in corners}
        vertex_clusters = {k: {k} for k in corners}

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
                    sp = env.shortest_path(poly[j], poly[boundary[m]], EPSILON)
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
                d = boundary.index(corner)
                a = poly[boundary[(d - 1) % len(boundary)]]
                b = poly[boundary[d]]
                c = poly[boundary[(d + 1) % len(boundary)]]
                s = [b.x() - a.x(), b.y() - a.y()]
                t = [c.x() - b.x(), c.y() - b.y()]
                if s[0] * t[1] - t[0] * s[1] > 0:
                    removal.append(corner)
            current_corners = removal

            if renderer:
                bx = [poly[k].x() for k in boundary]
                by = [poly[k].y() for k in boundary]
                renderer._d_polygon(x, y, Qt.white, 3)
                renderer._d_vertex_labels(poly)
                renderer._d_polygon(bx, by, Qt.darkGreen, 2)
                for b in current_corners:
                    renderer._d_dot(poly[b].x(), poly[b].y(), 6, C_CORNER)
                _show()

        # ---- Refine asso with vertex cluster visibility intersections ---
        for key, cluster in vertex_clusters.items():
            for i in cluster:
                vispol = vis.Visibility_Polygon(poly[i], env, EPSILON)
                px_, py_ = poly_to_points(vispol)
                px_.reverse()
                py_.reverse()
                clip = [(px_[k], py_[k]) for k in range(len(px_))]
                if asso[key] is None:
                    continue
                pc = pyclipper.Pyclipper()
                pc.AddPath(clip, pyclipper.PT_CLIP, True)
                pc.AddPath(asso[key], pyclipper.PT_SUBJECT, True)
                sol = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                pc.Clear()
                asso[key] = sol[0] if sol else None

        # ---- Display coloured asso regions ------------------------------
        palette = sns.color_palette('husl', len(corners))
        if renderer:
            for j, n in zip(corners, range(len(corners))):
                vp = asso[j]
                if vp is None:
                    continue
                vx = [vp[k][0] for k in range(len(vp))]
                vy = [vp[k][1] for k in range(len(vp))]
                col = QColor()
                col.setRgbF(palette[n][0], palette[n][1], palette[n][2], 0.35)
                renderer._d_filled_polygon(vx, vy, col)
                renderer._d_polygon(x, y, Qt.white, 3)
                renderer._d_vertex_labels(poly)
                renderer._d_dot(poly[j].x(), poly[j].y(), 6, C_CORNER)
                _show()

            for j, n in zip(corners, range(len(corners))):
                vp = asso[j]
                if vp is None:
                    continue
                vx = [vp[k][0] for k in range(len(vp))]
                vy = [vp[k][1] for k in range(len(vp))]
                col = QColor()
                col.setRgbF(palette[n][0], palette[n][1], palette[n][2], 0.35)
                renderer._d_filled_polygon(vx, vy, col)
            renderer._d_polygon(x, y, Qt.white, 3)
            renderer._d_vertex_labels(poly)
            _show()
            time.sleep(0.5)

        # ---- Visibility graph + KER points ------------------------------
        vis_graph = vis.Visibility_Graph(env, EPSILON)
        n_vg      = vis_graph.n()
        edges     = {frozenset([n_vg - 1 - j, n_vg - 1 - k])
                     for j in range(n_vg)
                     for k in range(n_vg)
                     if bool(vis_graph(j, k)) and j != k}

        vis_list = [[] for _ in range(len(poly))]
        for it in range(len(poly)):
            for edge in edges:
                if it in edge:
                    nb = list(edge)
                    nb.remove(it)
                    vis_list[it].append(nb[0])
        for it in range(len(poly)):
            vis_list[it].sort()

        KER = [vis.Point(poly[b].x(), poly[b].y()) for b in corners]

        lst_edges_KER = []
        for it in range(len(vis_list)):
            if it not in corners:
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

        for it in range(len(lst_edges_KER)):
            for jt in range(it + 1, len(lst_edges_KER)):
                [ia, ib] = list(lst_edges_KER[it])
                [ja, jb] = list(lst_edges_KER[jt])
                l1 = LineString([(ia.x(), ia.y()), (ib.x(), ib.y())])
                l2 = LineString([(ja.x(), ja.y()), (jb.x(), jb.y())])
                pt = l1.intersection(l2)
                if not pt.is_empty and pt.geom_type == 'Point':
                    KER.append(vis.Point(pt.x, pt.y))

        if renderer:
            renderer._d_polygon(x, y, Qt.white, 3)
            renderer._d_vertex_labels(poly)
            for edge in lst_edges_KER:
                [j, k] = list(edge)
                renderer._d_line(j.x(), j.y(), k.x(), k.y(), Qt.darkRed, 1)
            for pt in KER:
                renderer._d_dot(pt.x(), pt.y(), 3, QColor(0, 200, 100))
            print(f'KER points: {len(KER)}')
            _show()
            time.sleep(1.5)

        # ---- Map KER to corners they cover ------------------------------
        ts = time.time()
        KER_corner_list = [[] for _ in range(len(KER))]
        for it in asso:
            pol = asso[it]
            if pol is None:
                continue
            pv = [vis.Point(k[0], k[1]) for k in pol]
            pw = vis.Polygon(list(pv))
            for jt in range(len(KER)):
                if KER[jt]._in(pw, 1e-5):
                    KER_corner_list[jt].append(it)
                else:
                    for kt in range(len(pv)):
                        if minimum_distance(pv[kt], pv[(kt + 1) % len(pv)], KER[jt]) < 1:
                            KER_corner_list[jt].append(it)
                            break
        print(f'KER coverage computed in {time.time() - ts:.2f}s')

        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'resources', 'output.txt')
        with open(output_path, 'w') as f:
            for e in KER_corner_list:
                f.write(f'{e}\n')

        # ---- Set cover via ILP ------------------------------------------
        print(f'Total: {len(corners)} corners — {corners}')
        ts = time.time()
        chosed_set = solve_set_cover_ilp(KER_corner_list, corners)
        print(f'Set cover solved in {time.time()-ts:.3f}s '
              f'({len(chosed_set)} guards selected)')

        covered = set()
        for i in chosed_set:
            covered.update(KER_corner_list[i])
        uncovered = [c for c in corners if c not in covered]
        if uncovered:
            print(f'[WARNING] Cover incomplete. Uncovered corners: {uncovered}')

        # ---- Build geodesic graph ---------------------------------------
        geodesic = Geodesic(poly)
        try:
            geodesic.build()
            print('[INFO] Geodesic graph built.')
        except Exception as e:
            print(f'[ERROR] geodesic.build(): {e}')
            raise

        if renderer:
            renderer._d_polygon(x, y, Qt.white, 3)
            renderer._d_vertex_labels(poly)
            for pt_i in chosed_set:
                ker_pt = KER[pt_i]
                kp     = Point(ker_pt.x(), ker_pt.y())
                if not shapely_env.buffer(1).contains(kp):
                    continue
                try:
                    evis = vis.Visibility_Polygon(ker_pt, env, EPSILON)
                    evx, evy = poly_to_points(evis)
                    renderer._d_filled_polygon(evx, evy, QColor(255, 220, 0, 60))
                except Exception as e:
                    print(f'[ERROR] vis poly for KER[{pt_i}]: {e}')
            for pt_i in chosed_set:
                renderer._d_dot(KER[pt_i].x(), KER[pt_i].y(), 9, QColor(0, 160, 255))
            _show()
            time.sleep(1.5)

        # ---- Sanity check: union of guard vis-polys covers env ----------
        _vis_union = None
        for pt_i in chosed_set:
            ker_pt = KER[pt_i]
            if not shapely_env.buffer(1).contains(Point(ker_pt.x(), ker_pt.y())):
                continue
            try:
                evis = vis.Visibility_Polygon(ker_pt, env, EPSILON)
                evx, evy = poly_to_points(evis)
                _vp = ShapelyPolygon(list(zip(evx, evy)))
                _vis_union = _vp if _vis_union is None else _vis_union.union(_vp)
            except Exception as _e:
                print(f'[WARNING] Sanity: vis poly for KER[{pt_i}] failed: {_e}')
        if _vis_union is None:
            print('[WARNING] Sanity check: no visibility polygons computed.')
        else:
            _uncov = shapely_env.difference(_vis_union).area
            _total = shapely_env.area
            _pct   = 100.0 * (1.0 - _uncov / _total) if _total > 0 else 0.0
            if _uncov < 1e-3 * _total:
                print(f'[OK] Coverage sanity check passed: guard set covers '
                      f'{_pct:.4f}% of the environment '
                      f'(uncovered area {_uncov:.4f} sq units).')
            else:
                print(f'[WARNING] Coverage gap: {_uncov:.4f} sq units uncovered '
                      f'({100.0 - _pct:.4f}% of total area {_total:.2f}).')
        # -----------------------------------------------------------------

        # ---- Patrol path ------------------------------------------------
        path_lines = []
        for it in range(len(chosed_set)):
            for jt in range(it + 1, len(chosed_set)):
                ia, ib = chosed_set[it], chosed_set[jt]
                pa = [KER[ia].x(), KER[ia].y()]
                pb = [KER[ib].x(), KER[ib].y()]
                try:
                    sp = geodesic.shortest_path(pa, pb)
                except KeyError:
                    with suppress_output():
                        raw = env.shortest_path(KER[ia], KER[ib], EPSILON)
                    sp = [vg.Point(p.x(), p.y()) for p in raw.path()]
                for kt in range(len(sp) - 1):
                    add_unique_linestring(
                        path_lines,
                        LineString([Point(sp[kt].x, sp[kt].y),
                                    Point(sp[kt + 1].x, sp[kt + 1].y)]))

        # ---- Corner intersection points with patrol path ----------------
        intersection_points = {}
        for j in corners:
            vispol  = vis.Visibility_Polygon(poly[j], env, EPSILON)
            a, b, c = poly[(j - 1) % len(poly)], poly[j], poly[(j + 1) % len(poly)]
            ex_, ey_ = poly_to_points(vispol)
            ex_.reverse()
            ey_.reverse()

            v1 = np.array([b.x() - a.x(), b.y() - a.y()])
            v2 = np.array([b.x() - c.x(), b.y() - c.y()])
            v1 = v1 / np.linalg.norm(v1) * 10000
            v2 = v2 / np.linalg.norm(v2) * 10000

            subj     = [(ex_[k], ey_[k]) for k in range(len(ex_))]
            clip     = [(b.x(), b.y()), (v1[0] + b.x(), v1[1] + b.y()), (v2[0] + b.x(), v2[1] + b.y())]
            pc       = pyclipper.Pyclipper()
            pc.AddPath(clip, pyclipper.PT_CLIP, True)
            pc.AddPath(subj, pyclipper.PT_SUBJECT, True)
            sol      = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            visi_pol = sol[0]

            seg_poly      = ShapelyPolygon([(visi_pol[k][0], visi_pol[k][1]) for k in range(len(visi_pol))])
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

        if renderer:
            renderer._d_polygon(x, y, Qt.white, 3)
            renderer._d_vertex_labels(poly)
            for p in path_lines:
                renderer._d_glow_line(p.coords[0][0], p.coords[0][1],
                                      p.coords[1][0], p.coords[1][1], C_PATH, width=3)
            _show()
            time.sleep(1.5)

        # ---- Build graph data structures --------------------------------
        graph = construct_graph(path_lines)
        dense, scipy_g, scipy_src, bi_map, vertices = \
            construct_graph_dense(path_lines, intersection_points)
        dist_matrix = floyd_warshall(scipy_g)

        vectors_org = []
        for ver_id in range(len(vertices)):
            vect = {}
            for corn in corners:
                min_d = float('inf')
                for src in scipy_src[corn]:
                    if src is not None and dist_matrix[ver_id][src] < min_d:
                        min_d = dist_matrix[ver_id][src]
                vect[corn] = min_d
            vectors_org.append(vect)

        rows, cols_sp, _ = sp_find(scipy_g)
        total_edges      = list(zip(cols_sp, rows))

        save_cache(fingerprint, {
            'corners':             list(corners),
            'asso':                dict(asso),
            'KER_coords':          [(p.x(), p.y()) for p in KER],
            'lst_edges_KER':       [[(a.x(), a.y()), (b.x(), b.y())]
                                    for e in lst_edges_KER for a, b in [list(e)]],
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

    # ---- Build Voronoi skeleton -----------------------------------------
    print('[SKEL] Building Voronoi skeleton…')
    skel_nodes, skel_adj, skel_edges = build_voronoi_skeleton(shapely_env)
    print(f'[SKEL] {len(skel_nodes)} nodes, {len(skel_edges)} edges')

    return SimulationData(
        poly=poly,
        x=x,
        y=y,
        shapely_env=shapely_env,
        corners=corners,
        asso=asso,
        env=env,
        graph=graph,
        dense=dense,
        bi_map=bi_map,
        vertices=vertices,
        dist_matrix=dist_matrix,
        vectors_org=vectors_org,
        path_lines=path_lines,
        intersection_points=intersection_points,
        total_edges=total_edges,
        geodesic=geodesic,
        skel_nodes=skel_nodes,
        skel_adj=skel_adj,
        skel_edges=skel_edges,
    )


# ---------------------------------------------------------------------------
# Per-frame pure computations
# ---------------------------------------------------------------------------

def compute_path_lengths(ex: float, ey: float, data: SimulationData) -> dict:
    """Geodesic distance from evader position to each concave corner."""
    path_lengths = {}
    for cors in data.corners:
        pt_a = [ex, ey]
        pt_b = [data.poly[cors].x(), data.poly[cors].y()]
        try:
            sp = data.geodesic.shortest_path(pt_a, pt_b)
        except KeyError:
            with suppress_output():
                raw = data.env.shortest_path(
                    vis.Point(ex, ey),
                    vis.Point(data.poly[cors].x(), data.poly[cors].y()),
                    EPSILON)
            sp = [vg.Point(p.x(), p.y()) for p in raw.path()]
        pl = sum(Point(sp[i].x, sp[i].y).distance(Point(sp[i + 1].x, sp[i + 1].y))
                 for i in range(len(sp) - 1))
        path_lengths[cors] = pl if pl > 0 else 1e-9
    return path_lengths


def compute_optimal_guard(path_lengths: dict, data: SimulationData):
    """
    Find the patrol-edge position minimising max(distance_ratio) across corners.
    Returns (guard_point, v1_idx, v2_idx, opt_alpha).
    """
    edge_points = []
    for edge in data.total_edges:
        store = []
        for corn in data.corners:
            d1   = data.vectors_org[edge[0]][corn] / path_lengths[corn]
            d2   = data.vectors_org[edge[1]][corn] / path_lengths[corn]
            d1_r = data.vectors_org[edge[0]][corn]
            d2_r = data.vectors_org[edge[1]][corn]
            elen = data.vertices[edge[0]].distance(data.vertices[edge[1]])
            v1, v2 = edge[0], edge[1]

            if d1 < d2:
                b = (d2_r + elen - d1_r) / 2
                m1, c1 = find_slope_and_intercept((0, d1), (b, d1 + b / path_lengths[corn]))
                if np.isclose(elen, b, atol=1e-4):
                    store.append(('increasing', m1, c1))
                else:
                    m2, c2 = find_slope_and_intercept((elen, d2), (b, d1 + b / path_lengths[corn]))
                    store.append(('mixed', m1, c1, m2, c2, b))
            else:
                b = elen - (d1_r + elen - d2_r) / 2
                m1, c1 = find_slope_and_intercept((elen, d2), (b, d1 + b / path_lengths[corn]))
                if np.isclose(0, b, atol=1e-4):
                    store.append(('decreasing', m1, c1))
                else:
                    m2, c2 = find_slope_and_intercept((0, d1), (b, d1 + b / path_lengths[corn]))
                    store.append(('mixed', m2, c2, m1, c1, b))

        min_pt = process_functions(store, x_min=0, x_max=elen)
        edge_points.append((min_pt[0], min_pt[1], v1, v2))

    opt = min(edge_points, key=lambda p: p[1])
    return opt[2], opt[3], opt[1]   # v1_idx, v2_idx, opt_alpha


def compute_alphas(ox_comp: float, oy_comp: float,
                   path_lengths: dict, data: SimulationData) -> list:
    """Observer α values — normalised graph distance to each corner's patrol points."""
    add_observer_to_graph(data.dense, data.vertices, data.graph,
                          data.bi_map, Point(ox_comp, oy_comp))
    path_length_ref = max(path_lengths.values()) or 1
    alphas = []
    for ct in data.corners:
        dists = []
        for pt in list(data.intersection_points[ct]):
            sp_d, _ = dijkstra(data.graph, pt, Point(ox_comp, oy_comp))
            dists.append(round(sp_d / path_length_ref, 4))
        alphas.append(min(dists) if dists else float('inf'))
    return alphas


def compute_frame(ex: float, ey: float,
                  ox_comp: float, oy_comp: float,
                  data: SimulationData) -> FrameComputed:
    """Run all per-frame pure computations and return a FrameComputed bundle."""
    from geometry import interpolate_point

    path_lengths = compute_path_lengths(ex, ey, data)
    v1_idx, v2_idx, opt_alpha = compute_optimal_guard(path_lengths, data)

    guard = interpolate_point(data.vertices[v1_idx], data.vertices[v2_idx],
                              _opt_offset(path_lengths, v1_idx, v2_idx, data))

    alphas = compute_alphas(ox_comp, oy_comp, path_lengths, data)
    _, obs_to_guard_path = dijkstra(data.graph, (ox_comp, oy_comp),
                                    Point(guard.x, guard.y))

    return FrameComputed(
        guard=guard,
        opt_edge_v1=v1_idx,
        opt_edge_v2=v2_idx,
        opt_alpha=opt_alpha,
        path_lengths=path_lengths,
        alphas=alphas,
        obs_to_guard_path=obs_to_guard_path,
    )


def _opt_offset(path_lengths, v1_idx, v2_idx, data):
    """Re-derive the scalar offset along (v1→v2) for the optimal guard point."""
    edge_points = []
    for edge in data.total_edges:
        if edge[0] == v1_idx and edge[1] == v2_idx:
            store = []
            for corn in data.corners:
                d1   = data.vectors_org[edge[0]][corn] / path_lengths[corn]
                d2   = data.vectors_org[edge[1]][corn] / path_lengths[corn]
                d1_r = data.vectors_org[edge[0]][corn]
                d2_r = data.vectors_org[edge[1]][corn]
                elen = data.vertices[edge[0]].distance(data.vertices[edge[1]])
                if d1 < d2:
                    b = (d2_r + elen - d1_r) / 2
                    m1, c1 = find_slope_and_intercept((0, d1), (b, d1 + b / path_lengths[corn]))
                    if np.isclose(elen, b, atol=1e-4):
                        store.append(('increasing', m1, c1))
                    else:
                        m2, c2 = find_slope_and_intercept((elen, d2), (b, d1 + b / path_lengths[corn]))
                        store.append(('mixed', m1, c1, m2, c2, b))
                else:
                    b = elen - (d1_r + elen - d2_r) / 2
                    m1, c1 = find_slope_and_intercept((elen, d2), (b, d1 + b / path_lengths[corn]))
                    if np.isclose(0, b, atol=1e-4):
                        store.append(('decreasing', m1, c1))
                    else:
                        m2, c2 = find_slope_and_intercept((0, d1), (b, d1 + b / path_lengths[corn]))
                        store.append(('mixed', m2, c2, m1, c1, b))
            min_pt = process_functions(store, x_min=0, x_max=elen)
            return min_pt[0]
    return 0.0
