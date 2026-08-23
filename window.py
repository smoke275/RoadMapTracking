"""
PyQt5 window, Op enum, colour constants, and the simulation interactive loop.
Imports ker_pipeline.build() / compute_frame() via duck-typed renderer pattern.
"""
import atexit
import math
import os
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyclipper
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect, QPoint, QTimer
from PyQt5.QtGui import (QPainter, QBrush, QPen, QPolygon,
                          QColor, QFont, QTransform, QRadialGradient)
from PyQt5.QtWidgets import QMainWindow
from enum import Enum, auto
from shapely.geometry import LineString, Point

from config import BOUNDARY_X, BOUNDARY_Y, POINT_RADIUS, WINDOW_SIZE, EPSILON
from geometry import interpolate_point
from graph import dijkstra
import ker_pipeline
import visilibity as vis
from skeleton import nearest_node as _skeleton_nearest_node
from skeleton import skeleton_path as _skeleton_path
from skeleton import pick_destination as _skeleton_pick_destination
from skeleton import build_prm as _build_prm
from pursuer_motion import StableNodeController

# ---------------------------------------------------------------------------
# Pursuer debug log
# ---------------------------------------------------------------------------
_DBG_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pursuer_debug.log')
_dbg_log = open(_DBG_LOG_PATH, 'a', buffering=1)
_dbg_log.write(f'\n{"="*60}\n[RUN START] {time.strftime("%Y-%m-%d %H:%M:%S")}\n{"="*60}\n')
atexit.register(_dbg_log.close)


def _dbg(msg: str):
    _dbg_log.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')


# ---------------------------------------------------------------------------
# Thread semaphore (guards _main_stack between Qt paint thread & sim thread)
# ---------------------------------------------------------------------------
_sem = threading.Semaphore(1)


# ---------------------------------------------------------------------------
# Draw operation tags
# ---------------------------------------------------------------------------
class Op(Enum):
    point          = auto()
    line           = auto()
    dotted_line    = auto()
    circle         = auto()
    filled_circle  = auto()
    border_circle  = auto()
    polygon        = auto()
    dotted_polygon = auto()
    filled_polygon = auto()
    border_polygon = auto()
    draw_text      = auto()
    gradient_circle = auto()
    hud_panel       = auto()


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_POLYGON   = QColor(210, 225, 255)     # cool white
C_EVADER    = QColor(255, 75,  75)      # vivid red
C_OBSERVER  = QColor(0,  255, 130)      # bright mint-green
C_CORNER    = QColor(0,  240, 255)      # electric cyan
C_GUARD     = QColor(210, 50, 255)      # vivid violet
C_PATH      = QColor(255, 210,  0)      # pure gold
C_OPT_EDGE  = QColor(60,  155, 255)     # sky blue
C_INTER_PT  = QColor(90,  190, 255)     # pale blue
C_TEXT      = QColor(230, 240, 255)     # soft white


def _qcolor(r, g, b, a=255) -> QColor:
    c = QColor()
    c.setRgb(r, g, b, a)
    return c


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
class Window(QMainWindow):
    """PyQt5 window that renders the simulation and handles interactive drag."""

    def __init__(self):
        super().__init__()
        self.title = "KER Simulation"

        self._draw_queue: list = []
        self._main_stack: list = []
        self._running: bool = True
        self._data   = None   # set in run() after build()

        # Interactive state — defaults until run() populates them
        self.poly                    = None
        self.corners                 = []
        self.shapely_polygon         = None
        self.lines                   = []
        self.draggable_point_evader  = QPoint(0, 0)
        self.draggable_point_observer = QPoint(0, 0)
        self.draggable_point_corner  = None
        self.dragging_evader  = False
        self.dragging_observer = False
        self.dragging_corner  = False
        self.point_radius     = POINT_RADIUS

        # Auto-evader state
        self.auto_evader        = False
        self._skel_nodes        = []
        self._skel_adj          = {}
        self._skel_edges        = []
        self._skel_path: list   = []
        self._skel_path_pos: float = 0.0
        self._skel_seg_idx: int = 0
        self._evader_speed: float = 30.0   # px / second

        # Autonomous observer state
        self._auto_observer_pos  = None
        self._pursuer_speed: float = 20.0  # px / second

        # PRM for free pursuer navigation
        self._prm_nodes: list        = []
        self._prm_adj: dict          = {}
        self._prm_path: list         = []
        self._prm_seg_idx: int       = 0
        self._prm_seg_pos: float     = 0.0
        self._prm_last_guard         = None
        self._show_prm: bool         = False
        self._gating_enabled: bool   = False
        self._queued_dest: int | None = None  # skeleton node index queued by click

        # Roadmap pursuer state
        self._roadmap_pursuer: bool   = False
        self._pursuer_strategy: str   = 'minmax-alpha'  # cycled with S key
        self._strategy_obj            = None            # lazy benchmark pursuer
        self._roadmap_obs_pos         = None
        self._rm_ctrl                 = StableNodeController()
        self._dbg_prev_pursuer_pos    = None
        self._dbg_frame_count: int    = 0
        self._dbg_dir_flips: int      = 0
        self._dbg_prev_move_vec       = None

        self._init_window()
        timer = QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(33)

    def _init_window(self):
        self.setWindowTitle(self.title)
        self.setGeometry(200, 100, WINDOW_SIZE, WINDOW_SIZE)
        self.setFixedSize(self.size())
        self.show()

    # ------------------------------------------------------------------
    # Draw command queue
    # ------------------------------------------------------------------
    def draw(self, cmd):
        self._draw_queue.append(cmd)

    def execute(self):
        _sem.acquire()
        try:
            self._main_stack = self._draw_queue
            self._draw_queue = []
        finally:
            _sem.release()
        try:
            self.update()
        except RuntimeError:
            pass  # window already destroyed

    def closeEvent(self, event):
        self._running = False
        super().closeEvent(event)

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
        xs = [x,     x + r, x,     x - r]
        ys = [y + r, y,     y - r, y    ]
        self.draw([Op.border_polygon, xs, ys, 2, fill_color])

    def _d_glow_line(self, x1, y1, x2, y2, color, width=4):
        """Three-pass glow: wide-dim halo → mid bloom → sharp bright core."""
        outer = QColor(color); outer.setAlpha(28)
        mid   = QColor(color); mid.setAlpha(85)
        self.draw([Op.line, x1, y1, x2, y2, width + 12, outer])
        self.draw([Op.line, x1, y1, x2, y2, width + 3,   mid])
        self.draw([Op.line, x1, y1, x2, y2, width,        color])

    # ------------------------------------------------------------------
    # Paint event
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setWindow(QRect(-BOUNDARY_X, -BOUNDARY_Y, 2 * BOUNDARY_X, 2 * BOUNDARY_Y))
        painter.setViewport(QRect(0, 0, WINDOW_SIZE, WINDOW_SIZE))
        painter.scale(1, -1)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Dark radial-gradient background
        _bg = QRadialGradient(0, 0, 590)
        _bg.setColorAt(0.0,  QColor(10, 15, 40))
        _bg.setColorAt(0.55, QColor(5,  8,  22))
        _bg.setColorAt(1.0,  QColor(0,  0,  5))
        painter.setBrush(QBrush(_bg))
        painter.setPen(Qt.NoPen)
        bg = QPolygon([QPoint(-500, 500), QPoint(500, 500),
                       QPoint(500, -500), QPoint(-500, -500)])
        painter.drawPolygon(bg)
        # Subtle dot grid
        painter.setPen(QPen(QColor(255, 255, 255, 13), 2, Qt.SolidLine))
        for _gx in range(-450, 451, 50):
            for _gy in range(-450, 451, 50):
                painter.drawPoint(_gx, _gy)

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
                _col = QColor(cmd[5])
                _glow = QColor(_col.red(), _col.green(), _col.blue(),
                               max(_col.alpha() // 5, 22))
                painter.setPen(QPen(_glow, cmd[4] + 9, Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPoint(int(cmd[1]), int(cmd[2])), int(cmd[3]), int(cmd[3]))
                painter.setPen(QPen(_col, cmd[4], Qt.SolidLine))
                painter.drawEllipse(QPoint(int(cmd[1]), int(cmd[2])), int(cmd[3]), int(cmd[3]))

            elif op == Op.filled_circle:
                _cx, _cy, _r = int(cmd[1]), int(cmd[2]), int(cmd[3])
                _col = QColor(cmd[5])
                _bright = QColor(min(_col.red()   + 90, 255),
                                 min(_col.green() + 90, 255),
                                 min(_col.blue()  + 90, 255), _col.alpha())
                _fade   = QColor(_col.red() // 3, _col.green() // 3,
                                 _col.blue() // 3, 0)
                _grad = QRadialGradient(_cx, _cy, _r)
                _grad.setColorAt(0.0, _bright)
                _grad.setColorAt(0.45, _col)
                _grad.setColorAt(1.0,  _fade)
                painter.setBrush(QBrush(_grad))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPoint(_cx, _cy), _r, _r)

            elif op == Op.border_circle:
                painter.setPen(QPen(Qt.black, cmd[4], Qt.SolidLine))
                painter.setBrush(QBrush(cmd[5], Qt.SolidPattern))
                painter.drawEllipse(QPoint(int(cmd[1]), int(cmd[2])), int(cmd[3]), int(cmd[3]))

            elif op in (Op.polygon, Op.dotted_polygon):
                pen_style = Qt.DotLine if op == Op.dotted_polygon else Qt.SolidLine
                _col = QColor(cmd[4])
                pts  = [QPoint(int(cmd[1][j]), int(cmd[2][j])) for j in range(len(cmd[1]))]
                if op == Op.polygon and cmd[3] >= 2:
                    _gc = QColor(_col.red(), _col.green(), _col.blue(), 30)
                    painter.setPen(QPen(_gc, cmd[3] + 7, Qt.SolidLine))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPolygon(QPolygon(pts))
                painter.setPen(QPen(_col, cmd[3], pen_style))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(QPolygon(pts))

            elif op == Op.filled_polygon:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(cmd[4]), Qt.SolidPattern))
                pts = [QPoint(int(cmd[1][j]), int(cmd[2][j])) for j in range(len(cmd[1]))]
                painter.drawPolygon(QPolygon(pts))

            elif op == Op.border_polygon:
                _col = QColor(cmd[4])
                pts  = [QPoint(int(cmd[1][j]), int(cmd[2][j])) for j in range(len(cmd[1]))]
                # Colour glow halo
                _gc  = QColor(_col.red(), _col.green(), _col.blue(), 55)
                painter.setPen(QPen(_gc, cmd[3] + 6, Qt.SolidLine))
                painter.setBrush(QBrush(_col, Qt.SolidPattern))
                painter.drawPolygon(QPolygon(pts))
                # Crisp bright border
                painter.setPen(QPen(QColor(255, 255, 255, 210), cmd[3], Qt.SolidLine))
                painter.drawPolygon(QPolygon(pts))

            elif op == Op.draw_text:
                font = QFont("Consolas", cmd[3])
                painter.setFont(font)
                pos = QPoint(cmd[1], -cmd[2])
                painter.save()
                transform = QTransform()
                transform.scale(1, -1)
                painter.setTransform(transform, True)
                # Drop shadow for legibility
                painter.setPen(QPen(QColor(0, 0, 0, 200), 1, Qt.SolidLine))
                painter.drawText(pos + QPoint(1, 1), cmd[4])
                painter.setPen(QPen(C_TEXT, 1, Qt.SolidLine))
                painter.drawText(pos, cmd[4])
                painter.restore()

    # ------------------------------------------------------------------
    # Pick the patrol-graph node closest to (ex, ey) with LOS
    # ------------------------------------------------------------------
    def _best_los_spawn(self, ex: float, ey: float):
        """Return [x, y] of the patrol-graph node that has line-of-sight to
        (ex, ey) and is closest.  Falls back to the nearest node if none
        has clear LOS."""
        if self._data is None:
            return [float(self.draggable_point_observer.x()),
                    float(self.draggable_point_observer.y())]
        best, best_d   = None, float('inf')
        fallback, fb_d = None, float('inf')
        for node in self._data.graph:
            nx, ny = node
            d = math.hypot(nx - ex, ny - ey)
            if d < fb_d:
                fb_d, fallback = d, [nx, ny]
            seg = LineString([(nx, ny), (ex, ey)])
            if self._data.shapely_env.contains(seg) and d < best_d:
                best_d, best = d, [nx, ny]
        result = best if best is not None else fallback
        return result or [float(self.draggable_point_observer.x()),
                          float(self.draggable_point_observer.y())]

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
                ex, ey = self.draggable_point_evader.x(), self.draggable_point_evader.y()
                src = _skeleton_nearest_node(self._skel_nodes, ex, ey)
                dst = _skeleton_pick_destination(self._skel_nodes, self._skel_adj, src)
                self._skel_path     = _skeleton_path(self._skel_nodes, self._skel_adj, src, dst)
                self._skel_seg_idx  = 0
                self._skel_path_pos = 0.0
                spawn = self._best_los_spawn(float(ex), float(ey))
                self._auto_observer_pos  = list(spawn)
                self._roadmap_obs_pos    = list(spawn)
                self._rm_ctrl.reset(spawn)
                self._prm_path           = []
                self._prm_seg_idx        = 0
                self._prm_seg_pos        = 0.0
                self._prm_last_guard     = None
            if not self.auto_evader:
                self._auto_observer_pos  = None
                self._roadmap_obs_pos    = None
                self._prm_last_guard     = None
                self._queued_dest        = None  # drop stale click-destination
        elif e.key() == Qt.Key_G:
            self._show_prm = not self._show_prm
            print(f'[PRM-GRAPH] {"ON" if self._show_prm else "OFF"}')
        elif e.key() == Qt.Key_S:
            _cycle = ['minmax-alpha', 'geo-follow', 'tsp-patrol']
            _i = _cycle.index(self._pursuer_strategy)
            self._pursuer_strategy = _cycle[(_i + 1) % len(_cycle)]
            self._strategy_obj = None   # rebuilt lazily for the new strategy
            print(f'[STRATEGY] {self._pursuer_strategy}')
        elif e.key() == Qt.Key_V:
            self._gating_enabled = not self._gating_enabled
            ker_pipeline.set_gating_enabled(self._gating_enabled)
            print(f'[VIS-GATING] {"ON" if self._gating_enabled else "OFF"}')
        elif e.key() == Qt.Key_P:
            self._roadmap_pursuer = not self._roadmap_pursuer
            print(f'[ROADMAP-PURSUER] {"ON" if self._roadmap_pursuer else "OFF"}')
            if self.auto_evader:
                if self._roadmap_pursuer and self._roadmap_obs_pos is None:
                    ex, ey = self.draggable_point_evader.x(), self.draggable_point_evader.y()
                    spawn = self._best_los_spawn(float(ex), float(ey))
                    self._roadmap_obs_pos    = list(spawn)
                    self._rm_ctrl.reset(spawn)
                elif not self._roadmap_pursuer and self._auto_observer_pos is None:
                    ex, ey = self.draggable_point_evader.x(), self.draggable_point_evader.y()
                    spawn = self._best_los_spawn(float(ex), float(ey))
                    self._auto_observer_pos  = list(spawn)
                    self._prm_path           = []
                    self._prm_seg_idx        = 0
                    self._prm_seg_pos        = 0.0
                    self._prm_last_guard     = None
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
        r  = self.point_radius + 6

        if (tp - self.draggable_point_evader).manhattanLength() <= r:
            self.dragging_evader = True
        elif (tp - self.draggable_point_observer).manhattanLength() <= r:
            self.dragging_observer = True
        elif (tp - cp).manhattanLength() <= r:
            self.dragging_corner = True
        elif self.auto_evader and self._skel_nodes:
            # Queue the clicked skeleton node as the next evader destination
            node_idx = _skeleton_nearest_node(self._skel_nodes, mx, my)
            self._queued_dest = node_idx
            nx, ny = self._skel_nodes[node_idx]
            print(f'[EVADER-QUEUE] node {node_idx} at ({nx:.1f}, {ny:.1f})')
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging_evader   = False
        self.dragging_observer = False
        self.dragging_corner   = False

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
    def _apriori_check_intersection(self, values, asso):
        if not values:
            return None
        if len(values) == 1:
            return asso[values[0]]
        pc  = pyclipper.Pyclipper()
        pol = asso[values[0]]
        for i in range(1, len(values)):
            pc.AddPath(pol, pyclipper.PT_CLIP, True)
            pc.AddPath(asso[values[i]], pyclipper.PT_SUBJECT, True)
            sol = pc.Execute(pyclipper.CT_INTERSECTION,
                             pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            pc.Clear()
            if not sol or not sol[0]:
                return None
            pol = sol[0]
        return pol

    # ------------------------------------------------------------------
    # Main simulation entry point (runs on background thread)
    # ------------------------------------------------------------------
    def run(self, poly, force_recompute: bool = False):
        # ---- Build KER pipeline (one-time) ---------------------------
        data = ker_pipeline.build(poly, renderer=self, force_recompute=force_recompute)
        self._data = data   # expose to keyPressEvent

        # ---- Expose to interactive handlers --------------------------
        evader_pt = Point(0, 0)
        while not evader_pt.within(data.shapely_env):
            evader_pt = Point(random.randint(-480, 480), random.randint(-480, 480))

        self.draggable_point_evader   = QPoint(int(evader_pt.x), int(evader_pt.y))
        self.draggable_point_observer = QPoint(int(data.path_lines[0].coords[0][0]),
                                               int(data.path_lines[0].coords[0][1]))
        self.draggable_point_corner   = data.corners[0]
        self.point_radius             = POINT_RADIUS
        self.shapely_polygon          = data.shapely_env
        self.lines                    = data.path_lines
        self.poly                     = data.poly
        self.corners                  = data.corners
        self._skel_nodes              = data.skel_nodes
        self._skel_adj                = data.skel_adj
        self._skel_edges              = data.skel_edges

        # ---- Build PRM for free pursuer ----------------------------------
        print('[PRM] Building free-pursuer roadmap…')
        self._prm_nodes, self._prm_adj = _build_prm(data.shapely_env, n_samples=800)
        print(f'[PRM] {len(self._prm_nodes)} nodes')

        # ---- Async compute pool (double-buffer: render never blocks) ----
        _compute_pool   = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ker_compute')
        _compute_future = None
        _last_fc        = None
        _last_tick: float = time.monotonic()

        # ---- Interactive loop ----------------------------------------
        while True:
            if not self._running:
                break
            _now = time.monotonic()
            dt   = min(_now - _last_tick, 0.1)   # cap at 100 ms to handle pauses
            _last_tick = _now

            ex_ = self.draggable_point_evader.x()
            ey_ = self.draggable_point_evader.y()
            ox_ = self.draggable_point_observer.x()
            oy_ = self.draggable_point_observer.y()
            act = self.draggable_point_corner

            # Project auto-observer onto roadmap for computation
            if self.auto_evader and self._roadmap_pursuer and self._roadmap_obs_pos is not None:
                ao_x, ao_y   = self._roadmap_obs_pos
                ox_comp, oy_comp = ao_x, ao_y
            elif self.auto_evader and self._auto_observer_pos is not None:
                ao_x, ao_y = self._auto_observer_pos
                ao_pt = Point(ao_x, ao_y)
                best_proj, best_d = None, float('inf')
                for _pl in data.path_lines:
                    _proj = _pl.interpolate(_pl.project(ao_pt))
                    _d = ao_pt.distance(_proj)
                    if _d < best_d:
                        best_d, best_proj = _d, _proj
                ox_comp = best_proj.x if best_proj else ao_x
                oy_comp = best_proj.y if best_proj else ao_y
            else:
                ao_x, ao_y       = float(ox_), float(oy_)
                ox_comp, oy_comp = float(ox_), float(oy_)

            # ---- Polygon + patrol paths --------------------------------
            # Subtle interior fill then outline
            self._d_filled_polygon(data.x, data.y, QColor(18, 45, 110, 45))
            self._d_polygon(data.x, data.y, C_POLYGON, 3)
            self._d_vertex_labels(data.poly)
            for p in data.path_lines:
                self._d_glow_line(p.coords[0][0], p.coords[0][1],
                                  p.coords[1][0], p.coords[1][1], C_PATH, width=3)

            # ---- PRM overlay (toggle with [G]) ------------------------
            if self._show_prm and self._prm_nodes:
                C_PRM_EDGE = QColor(80, 180, 255, 55)
                C_PRM_NODE = QColor(80, 180, 255, 130)
                for i in range(len(self._prm_nodes)):
                    for j in self._prm_adj[i]:
                        if j > i:
                            x0, y0 = self._prm_nodes[i]
                            x1, y1 = self._prm_nodes[j]
                            self._d_line(x0, y0, x1, y1, C_PRM_EDGE, width=1)
                for (nx, ny) in self._prm_nodes:
                    self._d_dot(nx, ny, 2, C_PRM_NODE)

            # ---- Intersection points for active corner -----------------
            for pt in list(data.intersection_points[act]):
                self._d_dot(pt.x, pt.y, 5, C_INTER_PT)

            # ---- Evader ------------------------------------------------
            er = 14 if self.dragging_evader else POINT_RADIUS + 2
            self._d_ring(ex_, ey_, er + 3, QColor(200, 0, 0, 80), width=6)
            self._d_dot(ex_, ey_, er, C_EVADER)

            # ---- Observer ----------------------------------------------
            or_ = 14 if self.dragging_observer else POINT_RADIUS + 2
            self._d_ring(ox_, oy_, or_ + 3, QColor(0, 150, 50, 80), width=6)
            self._d_dot(ox_, oy_, or_, C_OBSERVER)

            # ---- Active corner -----------------------------------------
            cr = 14 if self.dragging_corner else POINT_RADIUS + 2
            self._d_ring(data.poly[act].x(), data.poly[act].y(),
                         cr + 3, QColor(0, 180, 200, 80), width=6)
            self._d_dot(data.poly[act].x(), data.poly[act].y(), cr, C_CORNER)

            # ---- KER frame computation (async double-buffer) ------------
            # If the background worker finished, harvest the result.
            if _compute_future is not None and _compute_future.done():
                try:
                    _last_fc = _compute_future.result()
                except Exception:
                    pass
                _compute_future = None
            # Submit a new computation only when the worker is idle.
            if _compute_future is None:
                _compute_future = _compute_pool.submit(
                    ker_pipeline.compute_frame,
                    float(ex_), float(ey_), ox_comp, oy_comp, data)
            # First frame: block once so we always have a valid fc.
            if _last_fc is None:
                _last_fc = _compute_future.result()
                _compute_future = None
            fc = _last_fc

            # ---- Draw optimal edge -------------------------------------
            v2x, v2y = data.vertices[fc.opt_edge_v1].x, data.vertices[fc.opt_edge_v1].y
            v3x, v3y = data.vertices[fc.opt_edge_v2].x, data.vertices[fc.opt_edge_v2].y
            self._d_glow_line(v2x, v2y, v3x, v3y, C_OPT_EDGE, width=8)
            self._d_dot(v2x, v2y, 10, C_OPT_EDGE)
            self._d_dot(v3x, v3y, 10, C_OPT_EDGE)

            # ---- Guard position ----------------------------------------
            guard = fc.guard
            self._d_ring(guard.x, guard.y, 16, QColor(255, 50, 255, 100), width=8)
            self._d_diamond(guard.x, guard.y, 12, C_GUARD)

            # ---- Observer → guard shortest path on graph ---------------
            if len(fc.obs_to_guard_path) >= 2:
                C_SP_LINE = QColor(0, 220, 255, 200)
                for _si in range(len(fc.obs_to_guard_path) - 1):
                    _ax, _ay = fc.obs_to_guard_path[_si]
                    _bx, _by = fc.obs_to_guard_path[_si + 1]
                    self._d_glow_line(_ax, _ay, _bx, _by, C_SP_LINE, width=3)

            # ---- HUD overlay -------------------------------------------
            alpha_lines = '  '.join(f'\u03B1{i}={a:.2f}' for i, a in enumerate(fc.alphas))
            self._d_text(-490, -460, f'\u03B1 opt = {fc.opt_alpha:.4f}', size=14)
            self._d_text(-490, -490, alpha_lines, size=11)
            if self.auto_evader and self._roadmap_pursuer and self._roadmap_obs_pos is not None:
                self._d_text(-490, 470, f'Observer roadmap ({ao_x:.0f},{ao_y:.0f})', size=11)
                self._d_text(-490, 490, f'Guard ({guard.x:.0f},{guard.y:.0f})', size=10)
            elif self.auto_evader and self._auto_observer_pos is not None:
                self._d_text(-490, 470, f'Observer free ({ao_x:.0f},{ao_y:.0f})', size=11)
                self._d_text(-490, 490,
                             f'Obs roadmap ({ox_comp:.0f},{oy_comp:.0f})  Guard ({guard.x:.0f},{guard.y:.0f})',
                             size=10)
            else:
                self._d_text(-490,  470, f'Observer ({ox_:.0f}, {oy_:.0f})', size=11)
                self._d_text(-490,  490, f'Guard    ({guard.x:.0f}, {guard.y:.0f})', size=11)

            # Legend (top-right)
            lx, ly, ls = 270, 480, 12
            self._d_dot(lx,      ly,       8, C_EVADER);   self._d_text(lx+15, ly-5,   'Evader',          ls)
            self._d_dot(lx,      ly - 28,  8, C_OBSERVER); self._d_text(lx+15, ly-33,  'Observer',        ls)
            self._d_dot(lx,      ly - 56,  8, C_CORNER);   self._d_text(lx+15, ly-61,  'Corner',          ls)
            self._d_diamond(lx,  ly - 84,  8, C_GUARD);    self._d_text(lx+15, ly-89,  'Guard opt',       ls)
            self._d_glow_line(lx-8, ly-108, lx+8, ly-108, QColor(0, 220, 255, 200), width=3)
            self._d_text(lx+15, ly-113, 'Obs→Guard path', ls)
            C_AUTO_OBS  = QColor(255, 220, 0)
            C_SKEL_LEG  = QColor(80, 80, 180, 200)
            C_PATH_SKEL = QColor(255, 120, 0, 220)
            self._d_dot(lx,     ly - 136, 8, C_AUTO_OBS);  self._d_text(lx+15, ly-141, 'Auto observer',   ls)
            self._d_glow_line(lx-8, ly-160, lx+8, ly-160, C_SKEL_LEG, width=1)
            self._d_text(lx+15, ly-165, 'Skeleton', ls)
            self._d_glow_line(lx-8, ly-184, lx+8, ly-184, C_PATH_SKEL, width=2)
            self._d_text(lx+15, ly-189, 'Evader path', ls)
            self._d_ring(lx, ly - 208, 6, QColor(255, 180, 0, 200), width=2)
            self._d_text(lx+15, ly-213, 'Evader dest', ls)

            # ---- Voronoi skeleton (faint) -------------------------------
            C_SKEL_DRAW = QColor(80, 80, 120, 120)
            for (sx1, sy1), (sx2, sy2) in self._skel_edges:
                self._d_glow_line(sx1, sy1, sx2, sy2, C_SKEL_DRAW, width=1)

            # ---- Auto-evader mode --------------------------------------
            if self.auto_evader:
                self._d_text(-490, -430, '[A] AUTO EVADER ON', size=11)
                pursuer_label = 'ROADMAP' if self._roadmap_pursuer else 'FREE'
                self._d_text(-490, -410, f'[P] pursuer: {pursuer_label}', size=11)
                if self._roadmap_pursuer:
                    self._d_text(-350, -410,
                                 f'[S] strategy: {self._pursuer_strategy}', size=11)

                # Step evader along skeleton path
                speed = self._evader_speed * dt
                while speed > 0 and self._skel_path:
                    if self._skel_seg_idx >= len(self._skel_path) - 1:
                        # Snap to evader's actual position so the new path
                        # genuinely starts from where the evader currently is.
                        ev_x = float(self.draggable_point_evader.x())
                        ev_y = float(self.draggable_point_evader.y())
                        cur_node = _skeleton_nearest_node(self._skel_nodes, ev_x, ev_y)
                        if self._queued_dest is not None:
                            dst = self._queued_dest
                            self._queued_dest = None
                        else:
                            dst = _skeleton_pick_destination(self._skel_nodes, self._skel_adj, cur_node)
                        new_path = _skeleton_path(self._skel_nodes, self._skel_adj, cur_node, dst)
                        # Fallback: queued dest was trivial (same node) — auto-pick instead
                        if not new_path or len(new_path) <= 1:
                            dst = _skeleton_pick_destination(self._skel_nodes, self._skel_adj, cur_node)
                            new_path = _skeleton_path(self._skel_nodes, self._skel_adj, cur_node, dst)
                        if new_path and len(new_path) > 1:
                            self._skel_path     = new_path
                            self._skel_seg_idx  = 0
                            self._skel_path_pos = 0.0
                        else:
                            # Clear so the outer guard can re-initialise next frame
                            self._skel_path    = []
                            self._skel_seg_idx = 0
                            break

                    i0 = self._skel_path[self._skel_seg_idx]
                    i1 = self._skel_path[self._skel_seg_idx + 1]
                    x0, y0   = self._skel_nodes[i0]
                    x1_, y1_ = self._skel_nodes[i1]
                    seg_len   = math.hypot(x1_ - x0, y1_ - y0)
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
                C_PATH_SKEL_DRAW = QColor(255, 120, 0, 200)
                for si in range(len(self._skel_path) - 1):
                    n0 = self._skel_path[si]
                    n1 = self._skel_path[si + 1]
                    p0 = self._skel_nodes[n0]
                    p1 = self._skel_nodes[n1]
                    self._d_glow_line(p0[0], p0[1], p1[0], p1[1], C_PATH_SKEL_DRAW, width=2)
                if self._skel_path:
                    dst_n   = self._skel_path[-1]
                    dx_, dy_ = self._skel_nodes[dst_n]
                    self._d_ring(dx_, dy_, 12, QColor(255, 180, 0, 180), width=2)
                if self._queued_dest is not None:
                    qx, qy = self._skel_nodes[self._queued_dest]
                    self._d_ring(qx, qy, 16, QColor(0, 230, 180, 220), width=2)
                    self._d_ring(qx, qy,  9, QColor(0, 230, 180, 120), width=1)
                    self._d_text(int(qx) + 14, int(qy), 'queued', size=9)

                # Step roadmap pursuer
                if self._roadmap_pursuer and self._roadmap_obs_pos is not None:
                    if self._pursuer_strategy == 'minmax-alpha':
                        target = guard
                    else:
                        if self._strategy_obj is None or \
                                self._strategy_obj.name != self._pursuer_strategy:
                            from benchmark.pursuers import PURSUER_CLASSES
                            ex_ = float(self.draggable_point_evader.x())
                            ey_ = float(self.draggable_point_evader.y())
                            self._strategy_obj = PURSUER_CLASSES[
                                self._pursuer_strategy](
                                    data, (ex_, ey_), self._pursuer_speed)
                        ex_ = float(self.draggable_point_evader.x())
                        ey_ = float(self.draggable_point_evader.y())
                        tx_, ty_ = self._strategy_obj.target(
                            (ex_, ey_), fc.path_lengths,
                            tuple(self._roadmap_obs_pos))
                        target = Point(tx_, ty_)
                        self._d_ring(tx_, ty_, 10, QColor(255, 120, 255, 200),
                                     width=2)
                    self._step_roadmap_pursuer(target, data, dt)

                # Step free (autonomous) observer via PRM
                elif self._auto_observer_pos is not None:
                    pspeed = self._pursuer_speed * dt
                    gx, gy = guard.x, guard.y
                    px, py = self._auto_observer_pos

                    if self._prm_nodes:
                        # Replan when guard moves >15 px or path is exhausted
                        guard_moved = (
                            self._prm_last_guard is None
                            or math.hypot(gx - self._prm_last_guard[0],
                                          gy - self._prm_last_guard[1]) > 15
                            or not self._prm_path
                            or self._prm_seg_idx >= len(self._prm_path) - 1
                        )
                        if guard_moved:
                            src = _skeleton_nearest_node(self._prm_nodes, px, py)
                            dst = _skeleton_nearest_node(self._prm_nodes, gx, gy)
                            self._prm_path      = _skeleton_path(
                                self._prm_nodes, self._prm_adj, src, dst)
                            self._prm_seg_idx   = 0
                            self._prm_last_guard = [gx, gy]

                        # Waypoint following — always move from actual (px,py) toward
                        # the next node so replanning never causes a position jump.
                        rem = pspeed
                        while (rem > 0 and self._prm_path
                               and self._prm_seg_idx < len(self._prm_path) - 1):
                            wx, wy = self._prm_nodes[
                                self._prm_path[self._prm_seg_idx + 1]]
                            dx, dy = wx - px, wy - py
                            dist   = math.hypot(dx, dy)
                            if dist < 1e-6 or rem >= dist:
                                rem -= dist
                                px, py = wx, wy
                                self._prm_seg_idx += 1
                            else:
                                px += (dx / dist) * rem
                                py += (dy / dist) * rem
                                rem = 0
                        if math.hypot(gx - px, gy - py) <= pspeed:
                            px, py = gx, gy
                    else:
                        # PRM not yet built — straight-line fallback
                        ddx, ddy = gx - px, gy - py
                        dist_to_guard = math.hypot(ddx, ddy)
                        if dist_to_guard > pspeed:
                            px += (ddx / dist_to_guard) * pspeed
                            py += (ddy / dist_to_guard) * pspeed
                        else:
                            px, py = gx, gy

                    self._auto_observer_pos = [px, py]

                    # Visibility polygon overlay
                    try:
                        _vp = vis.Visibility_Polygon(
                            vis.Point(px, py), data.env, EPSILON)
                        _vx = [_vp[i].x() for i in range(_vp.n())]
                        _vy = [_vp[i].y() for i in range(_vp.n())]
                        self._d_filled_polygon(_vx, _vy, QColor(255, 220, 0, 35))
                        self._d_polygon(_vx, _vy, QColor(255, 220, 0, 100), width=2)
                    except Exception:
                        pass

                    C_AUTO_OBS = QColor(255, 220, 0)
                    self._d_ring(px, py, POINT_RADIUS + 4, QColor(255, 220, 0, 80), width=5)
                    self._d_dot(px, py, POINT_RADIUS + 1, C_AUTO_OBS)
                    self.draw([Op.dotted_line, px, py, ox_comp, oy_comp, 1,
                               QColor(255, 220, 0, 120)])
                    self._d_ring(ox_comp, oy_comp, 5, QColor(255, 220, 0, 180), width=2)
                    self._d_glow_line(ox_comp, oy_comp, guard.x, guard.y,
                                      QColor(255, 220, 0, 160), width=1)
            else:
                self._d_text(-490, -430, '[A] auto evader off', size=11)
                pursuer_label = 'ROADMAP' if self._roadmap_pursuer else 'FREE'
                self._d_text(-490, -410, f'[P] pursuer: {pursuer_label}', size=11)
            prm_label = 'ON' if self._show_prm else 'OFF'
            self._d_text(-490, -390, f'[G] PRM graph: {prm_label}', size=11)
            gating_label = 'ON' if self._gating_enabled else 'OFF'
            self._d_text(-490, -370, f'[V] vis-gating: {gating_label}', size=11)

            self.execute()
            time.sleep(0.016)   # cap sim-thread at ~60 fps; frees CPU for compute worker

    # ------------------------------------------------------------------
    # Roadmap pursuer step — all self._roadmap_* state mutations
    # ------------------------------------------------------------------
    def _step_roadmap_pursuer(self, guard, data, dt: float = 0.016):
        pspeed = self._pursuer_speed * dt
        gx, gy = guard.x, guard.y
        _px_start, _py_start = self._roadmap_obs_pos
        self._dbg_frame_count += 1

        self._rm_ctrl.pos = list(self._roadmap_obs_pos)
        px, py = self._rm_ctrl.step(
            data.graph, (gx, gy), pspeed,
            log=lambda m: _dbg(f'[PURSUER][frame={self._dbg_frame_count}] {m}'))
        self._roadmap_obs_pos = [px, py]

        # Oscillation detector
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
                         f'base_idx={self._rm_ctrl.base_idx}/{len(self._rm_ctrl.base_path)} '
                         f'base_path={self._rm_ctrl.base_path}')
                else:
                    self._dbg_dir_flips = 0
            self._dbg_prev_move_vec = _cur_vec
        self._dbg_prev_pursuer_pos = [_px_start, _py_start]

        # Sync green dot position
        self.draggable_point_observer = QPoint(int(px), int(py))

        # Visibility polygon overlay for roadmap pursuer
        try:
            _vp = vis.Visibility_Polygon(vis.Point(px, py), data.env, EPSILON)
            _vx = [_vp[i].x() for i in range(_vp.n())]
            _vy = [_vp[i].y() for i in range(_vp.n())]
            self._d_filled_polygon(_vx, _vy, QColor(255, 220, 0, 35))
            self._d_polygon(_vx, _vy, QColor(255, 220, 0, 100), width=2)
        except Exception:
            pass

        # Draw committed base-node path
        C_RM_PATH  = QColor(255, 220, 0, 160)
        _remaining_nodes = self._rm_ctrl.base_path[self._rm_ctrl.base_idx:]
        _draw_pts  = [(px, py)] + _remaining_nodes + [(gx, gy)]
        for si in range(len(_draw_pts) - 1):
            ax, ay = _draw_pts[si]
            bx, by = _draw_pts[si + 1]
            self._d_glow_line(ax, ay, bx, by, C_RM_PATH, width=2)

        for _ni, _bn in enumerate(_remaining_nodes):
            self._d_dot(_bn[0], _bn[1], 5, QColor(255, 220, 0, 200))
            self._d_text(int(_bn[0]) + 7, int(_bn[1]) + 7, str(_ni + 1), size=9)

        _node_strs = ([f'p({px:.0f},{py:.0f})'] +
                      [f'{_ni+1}({_bn[0]:.0f},{_bn[1]:.0f})'
                       for _ni, _bn in enumerate(_remaining_nodes)] +
                      [f'g({gx:.0f},{gy:.0f})'])
        _path_str = ' → '.join(_node_strs)
        self._d_text(-490, 450, _path_str, size=9)
        _dbg(f'[PURSUER][frame={self._dbg_frame_count}] PATH {_path_str}')
