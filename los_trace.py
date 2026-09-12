"""
Roadmap line-of-sight trace — a stripped-down cousin of main.py.

No pursuer, no tracking. Place the evader (drag the red dot), pick a reflex
corner (click it, or press C to cycle), and the tool walks the evader along
its geodesic to that corner and shows the part of the patrol roadmap that
stays visible to it for the whole trip — the intersection, over every
sampled instant of the escape, of the visible portion of the roadmap.

Run:  python los_trace.py                # draw a polygon first (as main.py)
      python los_trace.py --skip-draw    # use config.FILE_NAME directly
      python los_trace.py -s --step 3    # finer sampling along the path

Controls:
  Drag RED dot     — move evader
  Click a corner   — select the target corner (or press C to cycle)
  SPACE            — play / pause the evader walking its escape path
  [ / ]            — coarser / finer sampling step
  Esc              — quit

Colours:
  gold  (dim)      — the whole roadmap
  green (bright)   — roadmap visible for the WHOLE trip (the intersection)
  magenta          — roadmap visible from the evader's current playback spot
  yellow fill      — visibility polygon at the playback spot
  orange line      — the evader's geodesic to the corner, dots at samples
"""
import argparse
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMainWindow
from shapely.geometry import Point

import ker_pipeline
from config import POINT_RADIUS
from draw_polygon import draw_polygon, load_polygon
from roadmap_los import trace_common_visible_roadmap
from window import Window, C_POLYGON, C_EVADER, C_CORNER, C_PATH

C_COMMON   = QColor(0, 255, 120)          # roadmap visible for the whole trip
C_NOW      = QColor(255, 90, 230)         # roadmap visible at the playback spot
                                          # (magenta: cyan blends to green on
                                          # top of the yellow visibility fill)
C_VIS_FILL = QColor(255, 220, 0, 35)
C_VIS_EDGE = QColor(255, 220, 0, 90)
C_EPATH    = QColor(255, 140, 0, 230)
C_SAMPLE   = QColor(255, 140, 0, 160)
C_WALKER   = QColor(255, 200, 120)

PLAY_SPEED = 60.0        # world units per second during playback
STEP_MIN, STEP_MAX = 0.5, 50.0


class TraceWindow(Window):
    """Reuses Window's rendering, draw queue, and evader drag; replaces the
    simulation loop with the LOS trace."""

    def __init__(self, step: float = 5.0):
        super().__init__()
        self.setWindowTitle('Roadmap LOS trace')
        self._max_alpha_btn.hide()      # tracking-only control, not used here

        self._step = step
        self._trace = None              # last finished RoadmapTrace
        self._trace_key = None          # (ex, ey, corner, step) it was computed for
        self._pending_key = None        # key of the trace currently in flight
        self._trace_future = None
        self._playing = False
        self._play_s = 0.0              # playback arc length along the escape path

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Escape:
            self.close()
        elif k == Qt.Key_C and self.corners:
            i = self.corners.index(self.draggable_point_corner)
            self.draggable_point_corner = self.corners[(i + 1) % len(self.corners)]
            print(f'Active corner → {self.draggable_point_corner}')
        elif k == Qt.Key_Space:
            self._playing = not self._playing
        elif k == Qt.Key_BracketLeft:
            self._step = min(STEP_MAX, self._step * 1.5)
            print(f'[STEP] {self._step:.2f}')
        elif k == Qt.Key_BracketRight:
            self._step = max(STEP_MIN, self._step / 1.5)
            print(f'[STEP] {self._step:.2f}')
        QMainWindow.keyPressEvent(self, e)

    def mousePressEvent(self, event):
        if self.poly is None:
            return
        mx, my = self._map_mouse(event)
        tp = QPoint(int(mx), int(my))
        r = self.point_radius + 6
        if (tp - self.draggable_point_evader).manhattanLength() <= r:
            self.dragging_evader = True
            return
        # Click any reflex corner to select it; keep dragging to slide the
        # selection across corners (Window.mouseMoveEvent handles the drag).
        best_i, best_d = None, float('inf')
        for idx in self.corners:
            cp = QPoint(int(self.poly[idx].x()), int(self.poly[idx].y()))
            d = (tp - cp).manhattanLength()
            if d < best_d:
                best_d, best_i = d, idx
        if best_i is not None and best_d <= 2 * r:
            if best_i != self.draggable_point_corner:
                print(f'Active corner → {best_i}')
            self.draggable_point_corner = best_i
            self.dragging_corner = True
        QMainWindow.mousePressEvent(self, event)

    # ------------------------------------------------------------------
    # Drawing helpers for shapely geometry
    # ------------------------------------------------------------------
    def _d_lines(self, mls, color, width):
        for line in getattr(mls, 'geoms', [mls]):
            c = list(line.coords)
            for (x0, y0), (x1, y1) in zip(c, c[1:]):
                self._d_glow_line(x0, y0, x1, y1, color, width=width)

    def _d_shape(self, poly_shape, fill, edge):
        if poly_shape is None or poly_shape.is_empty:
            return
        for part in getattr(poly_shape, 'geoms', [poly_shape]):
            if part.geom_type != 'Polygon':
                continue
            xs, ys = zip(*part.exterior.coords)
            self._d_filled_polygon(list(xs), list(ys), fill)
            self._d_polygon(list(xs), list(ys), edge, width=1)

    # ------------------------------------------------------------------
    # Main loop (background thread)
    # ------------------------------------------------------------------
    def run(self, poly, force_recompute: bool = False):
        data = ker_pipeline.build(poly, renderer=self, force_recompute=force_recompute)
        self._data = data

        evader_pt = Point(0, 0)
        while not evader_pt.within(data.shapely_env):
            evader_pt = Point(random.randint(-480, 480), random.randint(-480, 480))
        self.draggable_point_evader = QPoint(int(evader_pt.x), int(evader_pt.y))
        self.draggable_point_observer = QPoint(-10 ** 6, -10 ** 6)   # unused, out of reach
        self.draggable_point_corner = data.corners[0]
        self.shapely_polygon = data.shapely_env
        self.lines = data.path_lines
        self.poly = data.poly
        self.corners = data.corners

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='los_trace')
        last_tick = time.monotonic()

        while self._running:
            now = time.monotonic()
            dt = min(now - last_tick, 0.1)
            last_tick = now

            ex, ey = float(self.draggable_point_evader.x()), float(self.draggable_point_evader.y())
            act = self.draggable_point_corner
            key = (ex, ey, act, round(self._step, 4))

            # ---- Trace computation: one in flight at a time -------------
            if self._trace_future is not None and self._trace_future.done():
                try:
                    self._trace = self._trace_future.result()
                    self._trace_key = self._pending_key
                except Exception as exc:
                    print(f'[TRACE] failed: {exc}')
                self._trace_future = None
            if self._trace_future is None and key != self._trace_key:
                self._pending_key = key
                self._trace_future = pool.submit(
                    trace_common_visible_roadmap, ex, ey, act, data, self._step)

            # ---- Static scene ------------------------------------------
            self._d_filled_polygon(data.x, data.y, QColor(18, 45, 110, 45))
            self._d_polygon(data.x, data.y, C_POLYGON, 3)
            self._d_vertex_labels(data.poly)
            dim_path = QColor(C_PATH); dim_path.setAlpha(110)
            for p in data.path_lines:
                self._d_glow_line(p.coords[0][0], p.coords[0][1],
                                  p.coords[1][0], p.coords[1][1], dim_path, width=2)
            for c in data.corners:
                self._d_dot(data.poly[c].x(), data.poly[c].y(), 5, C_CORNER)

            # ---- Trace overlay -----------------------------------------
            tr = self._trace
            if tr is not None:
                stale = (self._trace_key != key)

                # Evader's escape path with sample dots
                pc = tr.evader_path
                for (x0, y0), (x1, y1) in zip(pc, pc[1:]):
                    self._d_line(x0, y0, x1, y1, C_EPATH, width=2)
                for smp in tr.samples:
                    self._d_dot(smp.pos[0], smp.pos[1], 2, C_SAMPLE)

                # Playback: current viewpoint, its visibility, what it sees
                if self._playing and tr.path_len > 0:
                    self._play_s = (self._play_s + PLAY_SPEED * dt) % tr.path_len
                play_s = min(self._play_s, tr.path_len)
                smp = tr.sample_at(play_s)
                self._d_shape(smp.vis_shape, C_VIS_FILL, C_VIS_EDGE)
                self._d_lines(smp.visible, C_NOW, width=3)
                wx, wy = tr.evader_position(play_s)
                self._d_ring(wx, wy, POINT_RADIUS + 2, QColor(255, 200, 120, 160), width=3)
                self._d_dot(wx, wy, POINT_RADIUS - 2, C_WALKER)

                # The answer: roadmap visible for the whole trip
                self._d_lines(tr.common, C_COMMON, width=6)

                # HUD
                min_vis = min((s.visible_len for s in tr.samples), default=0.0)
                self._d_text(-490, -460,
                             f'corner {tr.corner}   path {tr.path_len:.0f}   '
                             f'samples {len(tr.samples)} (step {self._step:.1f})'
                             + ('   updating…' if stale else ''), size=12)
                self._d_text(-490, -480,
                             f'common visible roadmap: {tr.common_len:.0f} / {tr.roadmap_len:.0f} '
                             f'({tr.common_frac * 100:.1f}%)   '
                             f'min single-instant visible: {min_vis / tr.roadmap_len * 100:.1f}%'
                             + (f'   [{tr.n_failed} vis-poly failures]' if tr.n_failed else ''),
                             size=12)
                self._d_text(-490, -500,
                             f'playback {"▶" if self._playing else "‖"} s={play_s:.0f}   '
                             f'visible now: {smp.visible_len:.0f} '
                             f'({smp.visible_len / tr.roadmap_len * 100:.1f}%)', size=11)
            else:
                self._d_text(-490, -460, 'computing trace…', size=12)

            # ---- Evader and active corner ------------------------------
            er = 14 if self.dragging_evader else POINT_RADIUS + 2
            self._d_ring(ex, ey, er + 3, QColor(200, 0, 0, 80), width=6)
            self._d_dot(ex, ey, er, C_EVADER)
            cr = 14 if self.dragging_corner else POINT_RADIUS + 2
            cx, cy = data.poly[act].x(), data.poly[act].y()
            self._d_ring(cx, cy, cr + 3, QColor(0, 180, 200, 80), width=6)
            self._d_dot(cx, cy, cr, C_CORNER)

            # Legend (top-right) and key hints (top-left)
            lx, ly, ls = 250, 480, 11
            self._d_dot(lx, ly, 7, C_EVADER);            self._d_text(lx + 14, ly - 5, 'Evader', ls)
            self._d_dot(lx, ly - 24, 7, C_CORNER);       self._d_text(lx + 14, ly - 29, 'Target corner', ls)
            self._d_glow_line(lx - 8, ly - 48, lx + 8, ly - 48, dim_path, width=2)
            self._d_text(lx + 14, ly - 53, 'Roadmap', ls)
            self._d_glow_line(lx - 8, ly - 72, lx + 8, ly - 72, C_COMMON, width=6)
            self._d_text(lx + 14, ly - 77, 'Visible whole trip', ls)
            self._d_glow_line(lx - 8, ly - 96, lx + 8, ly - 96, C_NOW, width=3)
            self._d_text(lx + 14, ly - 101, 'Visible now', ls)
            self._d_line(lx - 8, ly - 120, lx + 8, ly - 120, C_EPATH, width=2)
            self._d_text(lx + 14, ly - 125, 'Escape path', ls)
            self._d_text(-490, 490, 'drag red: evader   click corner / C: target   '
                                    'SPACE: play   [ ]: step', size=10)

            self.execute()
            time.sleep(0.016)


def startup():
    parser = argparse.ArgumentParser(description='Roadmap line-of-sight trace')
    parser.add_argument('-r', '--recompute', action='store_true',
                        help='Ignore cached pipeline results and rebuild')
    parser.add_argument('-s', '--skip-draw', action='store_true',
                        help='Skip the drawing tool; load config.FILE_NAME')
    parser.add_argument('--step', type=float, default=5.0,
                        help='Sampling step along the escape path, world units (default 5)')
    args, qt_args = parser.parse_known_args()

    if args.skip_draw:
        poly = load_polygon()
        if poly is None:
            print('[ERROR] --skip-draw: polygon file has fewer than 3 points — exiting.')
            return
    else:
        poly = draw_polygon()
        if poly is None:
            print('[ERROR] No polygon drawn — exiting.')
            return

    app = QApplication([sys.argv[0]] + qt_args)
    window = TraceWindow(step=args.step)
    thread = threading.Thread(target=window.run, args=(poly,),
                              kwargs={'force_recompute': args.recompute}, daemon=True)
    thread.start()
    sys.exit(app.exec())


if __name__ == '__main__':
    startup()
