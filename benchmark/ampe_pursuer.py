"""Baseline pursuers driven by the minimax value function of Antonio, Becerra
and Murrieta-Cid, "Approximate Methods for Visibility-Based Pursuit-Evasion"
(solver in baselines/ampe, tables written by ampe_solve.py).

The solver samples positions for both players and computes, for every pair
(pursuer sample p, evader sample e), the value V(p, e) of the game in which the
evader tries to break the line of sight: V > 0 if the pursuer keeps the evader
in sight, and V = -(distance the pursuer needs to see the evader again) once
sight is lost. In their recursion the evader moves and the pursuer answers with
the neighbor of largest value.

Execution in our simulator (the same rule for both variants): at every frame
the state is snapped to the nearest samples, the pursuer takes as its goal the
neighbor of its sample (or the sample itself) with the largest value against
the CURRENT evader sample, which is the pursuer's side of their recursion, and
moves toward it with its speed budget. The goal is recomputed every frame.

  ampe-roadmap : pursuer samples on our patrol roadmap, motion by the same
                 stable-node controller as every other roadmap strategy.
  ampe-free    : pursuer samples anywhere in the polygon (their setting),
                 motion along the geodesic shortest path.
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.prepared import prep

from cache import poly_fingerprint
from geometry import suppress_output
from pursuer_motion import StableNodeController

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_HERE, 'resources', 'ampe')
_TABLES = {}


class _Table:
    def __init__(self, prefix):
        with open(prefix + '.meta') as f:
            self.meta = json.load(f)
        sp, se = self.meta['Np'], self.meta['Ne']
        self.pp = np.fromfile(prefix + '.pp', dtype=np.float64).reshape(sp, 2)
        self.pe = np.fromfile(prefix + '.pe', dtype=np.float64).reshape(se, 2)
        self.V = np.memmap(prefix + '.V', dtype=np.float32, mode='r', shape=(sp, se))
        self.off = np.fromfile(prefix + '.nbo', dtype=np.int32)
        self.idx = np.fromfile(prefix + '.nbi', dtype=np.int32)
        self.tp = cKDTree(self.pp)
        self.te = cKDTree(self.pe)

    def neighbors(self, p):
        return self.idx[self.off[p]:self.off[p + 1]]


def _table(data, mode, ratio):
    key = (id(data), mode, round(ratio, 3))
    t = _TABLES.get(key)
    if t is None:
        prefix = os.path.join(_OUT, f'{poly_fingerprint(data.poly)[:16]}_{mode}_r{ratio:.3f}')
        if not os.path.exists(prefix + '.meta'):
            raise FileNotFoundError(
                f'no minimax table for this polygon, mode {mode}, ratio {ratio:.3f}: '
                f'run ampe_solve.py --modes {mode} --ratios {ratio}')
        t = _TABLES[key] = _Table(prefix)
    return t


class _AmpeBase:
    MODE = None
    K = 8                  # nearest samples examined when snapping a position

    def __init__(self, data, evader_start, speed, s_e=1.0):
        self.data = data
        self.speed = speed
        self.t = _table(data, self.MODE, speed / s_e)
        self._room = prep(data.shapely_env.buffer(0.05))
        e = self._snap_evader(evader_start)
        # spawn at the best pursuer sample for the evader's start, as the other
        # strategies spawn at their own first target
        p0 = int(np.argmax(self.t.V[:, e]))
        self._start(tuple(self.t.pp[p0]))

    def _sees(self, a, b):
        return self._room.covers(LineString([a, b]))

    def _snap(self, tree, pts, pos):
        k = min(self.K, len(pts))
        _, ids = tree.query(pos, k=k)
        ids = np.atleast_1d(ids)
        for i in ids:
            if self._sees(pos, tuple(pts[i])):
                return int(i)
        return int(ids[0])

    def _snap_evader(self, pos):
        return self._snap(self.t.te, self.t.pe, pos)

    def goal(self, evader_pos):
        e = self._snap_evader(evader_pos)
        p = self._snap_pursuer(tuple(self.pos))
        cand = np.concatenate(([p], self.t.neighbors(p)))
        vals = np.asarray(self.t.V[cand, e], dtype=np.float64)
        best = cand[vals >= vals.max() - 1e-9]
        if len(best) > 1:                              # ties: the nearest one
            d = np.hypot(self.t.pp[best, 0] - self.pos[0], self.t.pp[best, 1] - self.pos[1])
            return tuple(self.t.pp[best[int(np.argmin(d))]])
        return tuple(self.t.pp[best[0]])

    def target(self, evader_pos, path_lengths=None, cur_pos=None):
        return self.goal(evader_pos)


class AmpeRoadmapPursuer(_AmpeBase):
    """Their value function with the pursuer samples on our roadmap."""
    name = 'ampe-roadmap'
    MODE = 'roadmap'

    def _start(self, pos):
        self.ctrl = StableNodeController(pos)

    def _snap_pursuer(self, pos):
        return int(self.t.tp.query(pos)[1])

    def step(self, evader_pos, path_lengths):
        return self.ctrl.step(self.data.graph, self.goal(evader_pos), self.speed)

    @property
    def pos(self):
        return self.ctrl.pos


class AmpeFreePursuer(_AmpeBase):
    """Their setting: the pursuer moves anywhere in the polygon."""
    name = 'ampe-free'
    MODE = 'free'

    def _start(self, pos):
        self.pos = [float(pos[0]), float(pos[1])]

    def _snap_pursuer(self, pos):
        return self._snap(self.t.tp, self.t.pp, pos)

    def step(self, evader_pos, path_lengths):
        g = self.goal(evader_pos)
        px, py = self.pos
        if self._sees((px, py), g):
            way = [g]
        else:
            with suppress_output():
                sp = self.data.geodesic.shortest_path((px, py), g)
            way = [(q.x, q.y) for q in sp[1:]] or [g]
        rem = self.speed
        for wx, wy in way:
            d = float(np.hypot(wx - px, wy - py))
            if d <= rem:
                px, py, rem = wx, wy, rem - d
            else:
                px, py, rem = px + (wx - px) / d * rem, py + (wy - py) / d * rem, 0.0
                break
        self.pos = [px, py]
        return px, py
