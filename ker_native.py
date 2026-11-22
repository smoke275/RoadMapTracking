"""Loader for the native per-frame core (native/kercore.cpp).

ker_pipeline uses it for compute_path_lengths, compute_optimal_guard and
_opt_offset when config.NATIVE_CORE is on (or the environment variable
KER_NATIVE=1) and native/libkercore.so has been built with native/build.sh.
Without the library the Python implementation is used.
"""
import ctypes
import math
import os
import weakref

import numpy as np

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native', 'libkercore.so')
_lib = None
_handles = {}          # id(SimulationData) -> _Core


def enabled() -> bool:
    env = os.environ.get('KER_NATIVE')
    if env is not None:
        return env not in ('0', '')
    import config
    return bool(getattr(config, 'NATIVE_CORE', False))


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_LIB_PATH)
        dp, ip = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        lib.ker_create.restype = ctypes.c_void_p
        lib.ker_create.argtypes = [ctypes.c_int, dp, ctypes.c_int, ip, ctypes.c_int, dp,
                                   ctypes.c_int, ip, ip, dp]
        lib.ker_destroy.argtypes = [ctypes.c_void_p]
        lib.ker_path_lengths.restype = ctypes.c_int
        lib.ker_path_lengths.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, dp]
        lib.ker_optimal_guard.restype = None
        lib.ker_optimal_guard.argtypes = [ctypes.c_void_p, dp, ctypes.c_char_p, ip, ip, dp, dp]
        lib.ker_edge_offset.restype = ctypes.c_double
        lib.ker_edge_offset.argtypes = [ctypes.c_void_p, dp, ctypes.c_char_p,
                                        ctypes.c_int, ctypes.c_int]
        lib.ker_shortest_path.restype = ctypes.c_int
        lib.ker_shortest_path.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
                                          ctypes.c_double, ctypes.c_double, dp, ctypes.c_int]
        _lib = lib
    return _lib


def _dp(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _ip(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def available() -> bool:
    """The native core is switched on and its library is built."""
    return enabled() and os.path.exists(_LIB_PATH)


class _Core:
    def __init__(self, data):
        lib = _load()
        self.corners = list(data.corners)
        self.index = {c: i for i, c in enumerate(self.corners)}
        nc = len(self.corners)
        xy = np.ascontiguousarray([(p.x(), p.y()) for p in data.poly], dtype=np.float64)
        C = np.ascontiguousarray(self.corners, dtype=np.intc)
        D = np.ascontiguousarray([[v[c] for c in self.corners] for v in data.vectors_org],
                                 dtype=np.float64).reshape(len(data.vectors_org), nc)
        e1 = np.ascontiguousarray([int(e[0]) for e in data.total_edges], dtype=np.intc)
        e2 = np.ascontiguousarray([int(e[1]) for e in data.total_edges], dtype=np.intc)
        el = np.ascontiguousarray([data.vertices[int(a)].distance(data.vertices[int(b)])
                                   for a, b in zip(e1, e2)], dtype=np.float64)
        self._h = lib.ker_create(len(xy), _dp(xy), nc, _ip(C), len(D), _dp(D),
                                 len(e1), _ip(e1), _ip(e2), _dp(el))
        self._lib = lib
        self._L = np.empty(nc, dtype=np.float64)
        self._Lp = _dp(self._L)
        self._masks = {}
        self._v1, self._v2 = ctypes.c_int(), ctypes.c_int()
        self._a, self._x = ctypes.c_double(), ctypes.c_double()

    def close(self):
        if self._h:
            self._lib.ker_destroy(self._h)
            self._h = None

    def path_lengths(self, ex, ey, floor):
        """dict corner -> geodesic distance, or None if the point reaches no corner."""
        if self._lib.ker_path_lengths(self._h, ex, ey, self._Lp):
            return None
        return {c: (v if v > floor else floor) for c, v in zip(self.corners, self._L.tolist())}

    def _args(self, path_lengths, corners):
        L = self._L
        for c, i in self.index.items():
            L[i] = path_lengths.get(c, math.inf)
        if corners is None:
            return None
        key = tuple(corners)
        mask = self._masks.get(key)
        if mask is None:
            sel = set(key)
            mask = self._masks[key] = bytes(1 if c in sel else 0 for c in self.corners)
        return mask

    def optimal_guard(self, path_lengths, corners=None):
        mask = self._args(path_lengths, corners)
        self._lib.ker_optimal_guard(self._h, self._Lp, mask, ctypes.byref(self._v1),
                                    ctypes.byref(self._v2), ctypes.byref(self._a),
                                    ctypes.byref(self._x))
        return self._v1.value, self._v2.value, self._a.value

    def edge_offset(self, path_lengths, v1, v2, corners=None):
        mask = self._args(path_lengths, corners)
        return self._lib.ker_edge_offset(self._h, self._Lp, mask, int(v1), int(v2))


def core(data):
    """The native core for this SimulationData, or None when it is off or
    the library is missing."""
    if not enabled():
        return None
    key = id(data)
    c = _handles.get(key)
    if c is None:
        if not os.path.exists(_LIB_PATH):
            return None
        c = _handles[key] = _Core(data)
        weakref.finalize(data, lambda k=key: (_handles.pop(k).close() if k in _handles else None))
    return c


class _Geo:
    """Shortest paths in a polygon (no roadmap data)."""

    def __init__(self, poly):
        lib = _load()
        xy = np.ascontiguousarray([(p.x(), p.y()) for p in poly], dtype=np.float64)
        self._h = lib.ker_create(len(xy), _dp(xy), 0, None, 0, None, 0, None, None, None)
        self._lib = lib
        self._cap = len(xy) + 2
        self._buf = np.empty(2 * self._cap, dtype=np.float64)
        self._bp = _dp(self._buf)

    def close(self):
        if self._h:
            self._lib.ker_destroy(self._h)
            self._h = None

    def shortest_path(self, a, b):
        """List of (x, y) from a to b, or None when there is no path."""
        k = self._lib.ker_shortest_path(self._h, float(a[0]), float(a[1]),
                                        float(b[0]), float(b[1]), self._bp, self._cap)
        if k < 2:
            return None
        v = self._buf[:2 * k].tolist()
        return list(zip(v[0::2], v[1::2]))


_geo_handles = {}      # id(Geodesic) -> _Geo


def geo(owner, poly):
    """The native shortest-path object for a Geodesic, or None when off."""
    if not enabled():
        return None
    key = id(owner)
    g = _geo_handles.get(key)
    if g is None:
        if not os.path.exists(_LIB_PATH):
            return None
        g = _geo_handles[key] = _Geo(poly)
        weakref.finalize(owner, lambda k=key: (_geo_handles.pop(k).close()
                                               if k in _geo_handles else None))
    return g
