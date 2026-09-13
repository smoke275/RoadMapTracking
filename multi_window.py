"""
k-pursuer tracking window.

Same environment, roadmap and evader as the single-pursuer demo (window.py),
but the reflex corners are partitioned into NUM_PURSUERS groups
(corner_groups.py) and each group gets its own roadmap pursuer. Every frame
each pursuer solves the single-pursuer minimax guard problem restricted to
its own corners and drives toward that guard with the same stable-node
controller the demo uses.

Selected by main.py when config.NUM_PURSUERS > 1.

Controls:
  Drag RED dot     — move evader (also re-seats the auto-evader)
  A                — toggle autonomous evader (Voronoi skeleton)
  P                — toggle pursuer motion (off = freeze pursuers, guards still update)
  V                — toggle pursuer visibility polygons
  R                — respawn each pursuer at its current guard point
  Esc              — quit

Readout:
  team α*     — max over groups of the group's optimal alpha: what k oracle
                (zero-transit) pursuers would achieve for this evader position
  achieved α  — max over corners of min over pursuers of d_G(p_i, c) / L_c
                at the pursuers' actual positions
"""
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMainWindow
from shapely.geometry import Point
import visilibity as vis

import ker_pipeline
from benchmark.evaders import SkeletonEvader
from config import (EPSILON, GROUP_AFFINITY_GRID, GROUP_TRACE_STEP, NUM_PURSUERS,
                    POINT_RADIUS)
from corner_groups import compute_frame_multi, make_grouping
from pursuer_motion import StableNodeController
from window import Window, C_POLYGON, C_EVADER, C_PATH

GROUP_COLOURS = [
    QColor(0, 255, 130),     # mint
    QColor(255, 90, 230),    # magenta
    QColor(255, 160, 0),     # orange
    QColor(60, 170, 255),    # sky
    QColor(255, 240, 60),    # yellow
    QColor(180, 120, 255),   # violet
    QColor(0, 230, 230),     # cyan
    QColor(255, 110, 110),   # salmon
]


def _col(i, alpha=255):
    c = QColor(GROUP_COLOURS[i % len(GROUP_COLOURS)])
    c.setAlpha(alpha)
    return c


class MultiPursuitWindow(Window):
    def __init__(self, k: int = NUM_PURSUERS):
        super().__init__()
        self.setWindowTitle(f'KER Simulation — {k} pursuers')
        self._max_alpha_btn.hide()

        self._k = k
        self._grouping = None
        self._ctrls: list = []          # StableNodeController per group
        self._pursuers_on = True
        self._show_vis = True
        self._evader_agent = None       # SkeletonEvader when auto-evader is on
        self._respawn = False
        self._pursuer_speed = 20.0      # px / s (same as the demo)
        self._evader_speed = 30.0

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Escape:
            self.close()
        elif k == Qt.Key_A:
            self.auto_evader = not self.auto_evader
            print(f'[AUTO-EVADER] {"ON" if self.auto_evader else "OFF"}')
            self._evader_agent = None       # rebuilt from the current spot
        elif k == Qt.Key_P:
            self._pursuers_on = not self._pursuers_on
            print(f'[PURSUERS] {"MOVING" if self._pursuers_on else "FROZEN"}')
        elif k == Qt.Key_V:
            self._show_vis = not self._show_vis
        elif k == Qt.Key_R:
            self._respawn = True
        QMainWindow.keyPressEvent(self, e)

    def mousePressEvent(self, event):
        if self.poly is None:
            return
        mx, my = self._map_mouse(event)
        tp = QPoint(int(mx), int(my))
        if (tp - self.draggable_point_evader).manhattanLength() <= self.point_radius + 6:
            self.dragging_evader = True
        QMainWindow.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        if self.dragging_evader:
            self._evader_agent = None   # auto-evader restarts from the new spot
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _draw_env(self, data):
        self._d_filled_polygon(data.x, data.y, QColor(18, 45, 110, 45))
        self._d_polygon(data.x, data.y, C_POLYGON, 3)
        self._d_vertex_labels(data.poly)
        for p in data.path_lines:
            self._d_glow_line(p.coords[0][0], p.coords[0][1],
                              p.coords[1][0], p.coords[1][1], C_PATH, width=3)

    def _status(self, data, text):
        self._draw_env(data)
        self._d_text(-490, -460, text, size=13)
        self.execute()

    def _d_vis_polygon(self, px, py, data, colour):
        try:
            vp = vis.Visibility_Polygon(vis.Point(px, py), data.env, EPSILON)
            xs = [vp[i].x() for i in range(vp.n())]
            ys = [vp[i].y() for i in range(vp.n())]
            fill = QColor(colour); fill.setAlpha(28)
            edge = QColor(colour); edge.setAlpha(90)
            self._d_filled_polygon(xs, ys, fill)
            self._d_polygon(xs, ys, edge, width=1)
        except Exception:
            pass

    def _spawn_pursuers(self, data, mf):
        self._ctrls = [StableNodeController((fc.guard.x, fc.guard.y)) for fc in mf.frames]

    # ------------------------------------------------------------------
    # Main loop (background thread)
    # ------------------------------------------------------------------
    def run(self, poly, force_recompute: bool = False):
        data = ker_pipeline.build(poly, renderer=self, force_recompute=force_recompute)
        self._data = data

        # ---- Corner grouping (cached per polygon) -----------------------
        self._status(data, f'grouping corners for {self._k} pursuers…')
        grouping = make_grouping(
            data, self._k, GROUP_AFFINITY_GRID, GROUP_TRACE_STEP,
            progress=lambda d, t: self._status(
                data, f'grouping corners for {self._k} pursuers…  LOS traces {d}/{t}'))
        self._grouping = grouping
        groups = grouping.groups
        print(f'[GROUPS] k={grouping.k}: {groups}')
        print('[GROUPS] (run `python run_groups.py --k N` once per polygon to store the '
              'exact ILP partition; otherwise the affinity clustering is used)')

        # ---- Interactive state ------------------------------------------
        evader_pt = Point(0, 0)
        while not evader_pt.within(data.shapely_env):
            evader_pt = Point(random.randint(-480, 480), random.randint(-480, 480))
        self.draggable_point_evader = QPoint(int(evader_pt.x), int(evader_pt.y))
        self.draggable_point_observer = QPoint(-10 ** 6, -10 ** 6)   # unused
        self.draggable_point_corner = data.corners[0]
        self.shapely_polygon = data.shapely_env
        self.lines = data.path_lines
        self.poly = data.poly
        self.corners = data.corners

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ker_multi')
        future = None
        mf = None
        last_tick = time.monotonic()

        while self._running:
            now = time.monotonic()
            dt = min(now - last_tick, 0.1)
            last_tick = now

            # ---- Evader ----------------------------------------------
            if self.auto_evader and not self.dragging_evader:
                if self._evader_agent is None:
                    self._evader_agent = SkeletonEvader(
                        data.skel_nodes, data.skel_adj, data.shapely_env,
                        (float(self.draggable_point_evader.x()),
                         float(self.draggable_point_evader.y())),
                        self._evader_speed, avoid_recent=3)
                nx, ny = self._evader_agent.step(dt)
                self.draggable_point_evader = QPoint(int(nx), int(ny))
            ex, ey = float(self.draggable_point_evader.x()), float(self.draggable_point_evader.y())

            # ---- Frame computation (async, one in flight) ---------------
            positions = [tuple(c.pos) for c in self._ctrls] if self._ctrls else None
            if future is not None and future.done():
                try:
                    mf = future.result()
                except Exception as exc:
                    print(f'[MULTI] frame failed: {exc}')
                future = None
            if future is None:
                if positions is None:
                    # First frame: block so pursuers can be seated at their guards.
                    seed = compute_frame_multi(ex, ey, [(ex, ey)] * len(groups), data, groups)
                    self._spawn_pursuers(data, seed)
                    positions = [tuple(c.pos) for c in self._ctrls]
                future = pool.submit(compute_frame_multi, ex, ey, positions, data, groups)
                if mf is None:
                    mf = future.result()
                    future = None

            if self._respawn:
                self._spawn_pursuers(data, mf)
                self._respawn = False

            # ---- Scene -------------------------------------------------
            self._draw_env(data)
            for gi, g in enumerate(groups):
                for c in g:
                    cx, cy = data.poly[c].x(), data.poly[c].y()
                    self._d_ring(cx, cy, 9, _col(gi, 150), width=2)
                    self._d_dot(cx, cy, 5, _col(gi))

            # Pursuers: visibility, guard, route, dot
            for gi, (ctrl, fc) in enumerate(zip(self._ctrls, mf.frames)):
                px, py = ctrl.pos
                gx, gy = fc.guard.x, fc.guard.y
                colour = _col(gi)
                if self._show_vis:
                    self._d_vis_polygon(px, py, data, colour)
                v1, v2 = data.vertices[fc.opt_edge_v1], data.vertices[fc.opt_edge_v2]
                self._d_glow_line(v1.x, v1.y, v2.x, v2.y, _col(gi, 120), width=6)
                route = fc.obs_to_guard_path
                for (ax, ay), (bx, by) in zip(route, route[1:]):
                    self._d_glow_line(ax, ay, bx, by, _col(gi, 200), width=3)
                self._d_ring(gx, gy, 14, _col(gi, 90), width=6)
                self._d_diamond(gx, gy, 10, colour)
                self._d_ring(px, py, POINT_RADIUS + 5, _col(gi, 80), width=6)
                self._d_dot(px, py, POINT_RADIUS + 2, colour)
                self._d_text(int(px) + 12, int(py) + 12, f'P{gi + 1}', size=10)

                # Step toward the guard (uses the frame's guard, which may lag
                # one worker cycle behind the evader — same as the demo).
                if self._pursuers_on:
                    ctrl.step(data.graph, (gx, gy), self._pursuer_speed * dt)

            # Evader last so it stays on top
            er = 14 if self.dragging_evader else POINT_RADIUS + 2
            self._d_ring(ex, ey, er + 3, QColor(200, 0, 0, 80), width=6)
            self._d_dot(ex, ey, er, C_EVADER)

            # ---- HUD ---------------------------------------------------
            # Bottom-up: per-corner combined alphas, one line per pursuer, team line.
            y = -494
            items = [f'α{c}={mf.combined_alphas[c]:.2f}' for c in data.corners]
            per_row = 10
            for i in range(0, len(items), per_row):
                self._d_text(-490, y, '  '.join(items[i:i + per_row]), size=10)
                y += 16
            for gi, (fc, g) in enumerate(zip(mf.frames, groups)):
                self._d_text(-490, y, f'P{gi + 1}: α* = {fc.opt_alpha:.2f}   corners {g}', size=10)
                y += 16
            self._d_text(-490, y + 4,
                         f'team α* = {mf.team_opt_alpha:.3f}   achieved α = {mf.achieved_alpha:.3f}'
                         f'   (k = {len(groups)})', size=14)

            self._d_text(-490, 490, f'[A] auto evader: {"ON" if self.auto_evader else "off"}   '
                                    f'[P] pursuers: {"moving" if self._pursuers_on else "frozen"}   '
                                    f'[V] vis: {"on" if self._show_vis else "off"}   [R] respawn',
                         size=10)
            lx, ly, ls = 300, 470, 11
            self._d_dot(lx, ly, 7, C_EVADER); self._d_text(lx + 14, ly - 5, 'Evader', ls)
            for gi in range(len(groups)):
                yy = ly - 24 * (gi + 1)
                self._d_dot(lx, yy, 7, _col(gi)); self._d_text(lx + 14, yy - 5, f'Pursuer {gi + 1} / corners', ls)

            self.execute()
            time.sleep(0.016)
