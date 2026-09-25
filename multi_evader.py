"""
k pursuers, m evaders — interactive simulation.

Same environment, roadmap, corner partition and pursuer controller as the
k-pursuer window (multi_window.py), with several evaders. The pursuers do
not track evaders: they guard corners. For each reflex corner the relevant
evader is the one that can reach it first, so
    L_c = min_j d_geo(e_j, c)
and every pursuer runs the unchanged single-evader minimax optimiser on its
own corner group (corner_groups.compute_path_lengths_multi). Nothing is
assigned at run time.

Run:  python multi_evader.py --skip-draw                    # config.NUM_PURSUERS / NUM_EVADERS
      python multi_evader.py --skip-draw -k 3 -m 3           # 3 pursuers, 3 evaders
      python multi_evader.py --skip-draw -k 2 -m 4 --auto    # start with evaders wandering

Controls:
  Drag any RED dot — move that evader
  A                — toggle autonomous evaders (each wanders the Voronoi skeleton)
  P                — freeze / unfreeze pursuers
  V                — toggle pursuer visibility polygons
  N                — toggle nearest-evader links (corner -> the evader that defines its L_c)
  R                — respawn each pursuer at its current guard point
  Esc              — quit

Readout:
  team α*      max over groups of the group's optimal alpha (oracle, this frame)
  achieved α   max over corners of min over pursuers of d_G(p_i, c) / L_c
  in view      how many evaders at least one pursuer can currently see
  per evader   that evader's own worst corner alpha against the pursuers'
               actual positions, and whether it is in view
"""
import argparse
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMainWindow
from shapely.geometry import LineString, Point

import ker_pipeline
from benchmark.evaders import SkeletonEvader
from config import (GROUP_AFFINITY_GRID, GROUP_TRACE_STEP, NUM_EVADERS, NUM_PURSUERS,
                    POINT_RADIUS)
from corner_groups import _alphas_at, compute_frame_multi, make_grouping
from draw_polygon import draw_polygon, load_polygon
from multi_window import MultiPursuitWindow, _col
from window import Op

EVADER_COLOURS = [
    QColor(255, 75, 75),     # red
    QColor(255, 150, 60),    # amber
    QColor(255, 230, 80),    # yellow
    QColor(255, 110, 200),   # pink
    QColor(255, 200, 160),   # peach
    QColor(220, 80, 120),    # rose
]


def _ecol(j, alpha=255):
    c = QColor(EVADER_COLOURS[j % len(EVADER_COLOURS)])
    c.setAlpha(alpha)
    return c


def _sees(p, e, shapely_env) -> bool:
    return shapely_env.covers(LineString([p, e]))


class MultiEvaderWindow(MultiPursuitWindow):
    def __init__(self, k: int = NUM_PURSUERS, m: int = NUM_EVADERS, auto: bool = False):
        super().__init__(k)
        self.setWindowTitle(f'KER Simulation — {k} pursuers, {m} evaders')
        self._m = m
        self._evaders: list = []          # [QPoint]
        self._agents: list = []           # SkeletonEvader or None per evader
        self._drag_idx = None
        self._show_links = True
        self.auto_evader = auto

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_N:
            self._show_links = not self._show_links
            QMainWindow.keyPressEvent(self, e)
            return
        if e.key() == Qt.Key_A:
            self._agents = [None] * self._m       # rebuilt from current spots
        super().keyPressEvent(e)

    def mousePressEvent(self, event):
        if self.poly is None:
            return
        mx, my = self._map_mouse(event)
        tp = QPoint(int(mx), int(my))
        for j, ev in enumerate(self._evaders):
            if (tp - ev).manhattanLength() <= self.point_radius + 6:
                self._drag_idx = j
                self.dragging_evader = True
                break
        QMainWindow.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self.poly is None or self._drag_idx is None:
            return
        mx, my = self._map_mouse(event)
        if Point(mx, my).within(self.shapely_polygon):
            self._evaders[self._drag_idx] = QPoint(int(mx), int(my))

    def mouseReleaseEvent(self, event):
        if self._drag_idx is not None:
            self._agents[self._drag_idx] = None   # that evader restarts wandering from here
        self._drag_idx = None
        self.dragging_evader = False

    # ------------------------------------------------------------------
    # Main loop (background thread)
    # ------------------------------------------------------------------
    def run(self, poly, force_recompute: bool = False):
        data = ker_pipeline.build(poly, renderer=self, force_recompute=force_recompute)
        self._data = data

        self._status(data, f'grouping corners for {self._k} pursuers…')
        grouping = make_grouping(
            data, self._k, GROUP_AFFINITY_GRID, GROUP_TRACE_STEP,
            progress=lambda d, t: self._status(
                data, f'grouping corners for {self._k} pursuers…  LOS traces {d}/{t}'))
        self._grouping = grouping
        groups = grouping.groups
        print(f'[GROUPS] k={grouping.k}: {groups}')

        rng = random.Random(7)
        self._evaders = []
        while len(self._evaders) < self._m:
            pt = Point(rng.randint(-480, 480), rng.randint(-480, 480))
            if pt.within(data.shapely_env):
                self._evaders.append(QPoint(int(pt.x), int(pt.y)))
        self._agents = [None] * self._m
        self.draggable_point_evader = self._evaders[0]
        self.draggable_point_observer = QPoint(-10 ** 6, -10 ** 6)
        self.draggable_point_corner = data.corners[0]
        self.shapely_polygon = data.shapely_env
        self.lines = data.path_lines
        self.poly = data.poly
        self.corners = data.corners

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ker_me')
        future, mf = None, None
        last_tick = time.monotonic()

        while self._running:
            now = time.monotonic()
            dt = min(now - last_tick, 0.1)
            last_tick = now

            # ---- Evaders ---------------------------------------------
            if self.auto_evader:
                for j in range(self._m):
                    if j == self._drag_idx:
                        continue
                    if self._agents[j] is None:
                        self._agents[j] = SkeletonEvader(
                            data.skel_nodes, data.skel_adj, data.shapely_env,
                            (float(self._evaders[j].x()), float(self._evaders[j].y())),
                            self._evader_speed, avoid_recent=3)
                    nx, ny = self._agents[j].step(dt)
                    self._evaders[j] = QPoint(int(nx), int(ny))
            evs = [(float(q.x()), float(q.y())) for q in self._evaders]

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
                    seed = compute_frame_multi(evs, None, [evs[0]] * len(groups), data, groups)
                    self._spawn_pursuers(data, seed)
                    positions = [tuple(c.pos) for c in self._ctrls]
                future = pool.submit(compute_frame_multi, evs, None, positions, data, groups)
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

            # Which evader defines each corner (thin link corner -> evader)
            if self._show_links:
                for c, j in mf.nearest_evader.items():
                    if j < len(mf.evaders):
                        cx, cy = data.poly[c].x(), data.poly[c].y()
                        ex, ey = mf.evaders[j]
                        self.draw([Op.dotted_line, cx, cy, ex, ey, 1, _ecol(j, 70)])

            # Pursuers
            pursuer_pos = []
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
                pursuer_pos.append((px, py))
                if self._pursuers_on:
                    ctrl.step(data.graph, (gx, gy), self._pursuer_speed * dt)

            # Evaders: dot, label, sight lines, own worst alpha
            in_view = 0
            ev_lines = []
            per_p_alpha = [_alphas_at(p, mf.path_lengths, data) for p in pursuer_pos]
            for j, (ex, ey) in enumerate(evs):
                seen_by = [i for i, p in enumerate(pursuer_pos) if _sees(p, (ex, ey), data.shapely_env)]
                if seen_by:
                    in_view += 1
                    for i in seen_by:
                        self.draw([Op.dotted_line, pursuer_pos[i][0], pursuer_pos[i][1],
                                   ex, ey, 1, _col(i, 110)])
                er = 14 if self._drag_idx == j else POINT_RADIUS + 2
                self._d_ring(ex, ey, er + 3, _ecol(j, 80), width=6)
                self._d_dot(ex, ey, er, _ecol(j))
                self._d_text(int(ex) + 12, int(ey) - 18, f'E{j + 1}', size=10)
                # this evader's own worst corner against the pursuers' actual positions
                own = mf.per_evader_pl[j]
                worst = max(min(a[c] * mf.path_lengths[c] / own[c] for a in per_p_alpha)
                            for c in data.corners) if per_p_alpha else math.inf
                ev_lines.append(f'E{j + 1}: worst α = {worst:.2f}  '
                                f'{"in view (P" + ",".join(str(i + 1) for i in seen_by) + ")" if seen_by else "HIDDEN"}')

            # ---- HUD ---------------------------------------------------
            y = -494
            items = [f'α{c}={mf.combined_alphas[c]:.2f}' for c in data.corners]
            for i in range(0, len(items), 10):
                self._d_text(-490, y, '  '.join(items[i:i + 10]), size=10); y += 16
            for gi, (fc, g) in enumerate(zip(mf.frames, groups)):
                self._d_text(-490, y, f'P{gi + 1}: α* = {fc.opt_alpha:.2f}   corners {g}', size=10); y += 16
            for line in ev_lines:
                self._d_text(-490, y, line, size=10); y += 16
            self._d_text(-490, y + 4,
                         f'team α* = {mf.team_opt_alpha:.3f}   achieved α = {mf.achieved_alpha:.3f}   '
                         f'in view {in_view}/{self._m}   (k = {len(groups)}, m = {self._m})', size=14)
            self._d_text(-490, 490, f'[A] auto evaders: {"ON" if self.auto_evader else "off"}   '
                                    f'[P] pursuers: {"moving" if self._pursuers_on else "frozen"}   '
                                    f'[V] vis   [N] nearest links   [R] respawn', size=10)
            lx, ly, ls = 300, 470, 11
            for j in range(self._m):
                self._d_dot(lx, ly - 24 * j, 7, _ecol(j)); self._d_text(lx + 14, ly - 24 * j - 5, f'Evader {j + 1}', ls)
            for gi in range(len(groups)):
                yy = ly - 24 * (self._m + gi)
                self._d_dot(lx, yy, 7, _col(gi)); self._d_text(lx + 14, yy - 5, f'Pursuer {gi + 1} / corners', ls)

            self.execute()
            time.sleep(0.016)


def startup():
    parser = argparse.ArgumentParser(description='k pursuers, m evaders')
    parser.add_argument('-k', '--pursuers', type=int, default=NUM_PURSUERS)
    parser.add_argument('-m', '--evaders', type=int, default=NUM_EVADERS)
    parser.add_argument('--auto', action='store_true', help='Start with evaders wandering')
    parser.add_argument('-r', '--recompute', action='store_true')
    parser.add_argument('-s', '--skip-draw', action='store_true',
                        help='Skip the drawing tool; load config.FILE_NAME')
    args, qt_args = parser.parse_known_args()

    poly = load_polygon() if args.skip_draw else draw_polygon()
    if poly is None:
        print('[ERROR] no polygon — exiting.')
        return
    app = QApplication([sys.argv[0]] + qt_args)
    window = MultiEvaderWindow(args.pursuers, args.evaders, auto=args.auto)
    threading.Thread(target=window.run, args=(poly,),
                     kwargs={'force_recompute': args.recompute}, daemon=True).start()
    sys.exit(app.exec())


if __name__ == '__main__':
    startup()
