"""Shared constants and file-path configuration for the KER simulation."""
import os

EPSILON      = 1e-7
BOUNDARY_X   = 500
BOUNDARY_Y   = 500
POINT_RADIUS = 8
WINDOW_SIZE  = 750

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME   = os.path.join(_SCRIPT_DIR, 'resources', 'sites_poly9.csv')
CACHE_FILE  = os.path.join(_SCRIPT_DIR, 'resources', 'ker_cache.pkl')
GEO_FILE    = os.path.join(_SCRIPT_DIR, 'resources', 'ker_geodesic.graph')

# ---- Patrol roadmap source --------------------------------------------------
# 'ker'      : geodesics between the ILP-selected KER guards (the paper's roadmap)
# 'watchman' : the shortest watchman route of the polygon (watchman_route.py),
#              a single closed cycle that sees every point of the polygon.
# Everything downstream (entry nodes, distance tables, g*, controller, s*) is
# identical; only the embedded graph differs. run_max_speed.py and
# run_benchmark.py take --roadmap to override this per run.
ROADMAP_MODE = 'ker'

# ---- Guard selection ---------------------------------------------------------
# 'witness'     : minimum subset of the KER candidates that puts a guard in every
#                 association region A(c) and sees the whole polygon (Algorithm 1
#                 of the paper: set-cover ILP with witness points added on demand).
# 'witness-minspeed' : among the minimum guard sets, the one with the smallest
#                 critical speed ratio found by select_guard_set.py (stored in
#                 resources/guard_sets.json; falls back to 'witness' if absent).
# 'contraction' : the earlier selection (boundary contraction, guard regions,
#                 corner rows only). Kept to reproduce results before 2026-11-21.
GUARD_SELECTION = 'witness-minspeed'
# Minimum guard sets sampled per polygon by select_guard_set.py.
GUARD_SAMPLES = 100
# Unseen components smaller than this fraction of the polygon area are treated
# as floating-point slivers by the witness loop.
COVERAGE_ETA = 2e-4

# ---- Native core ---------------------------------------------------------------
# Use native/kercore.cpp (built by native/build.sh) for the geodesic distances
# to the corners and the min-max position on the roadmap. The Python code is
# used when this is off or the library is missing. KER_NATIVE=0/1 overrides.
NATIVE_CORE = True

# ---- Multi-pursuer (see corner_groups.py / multi_window.py) ----------------
# Number of pursuers. 1 runs the original single-pursuer demo; >1 partitions
# the reflex corners into that many groups (clustered on how often two
# corners' escapes can be watched from the same roadmap spot) and gives each
# group its own pursuer.
NUM_PURSUERS = 1
# Evader-grid resolution per axis used to average the corner-pair overlap
# affinity (see corner_groups.corner_affinity). Coarser = faster startup.
GROUP_AFFINITY_GRID = 10
# Sampling step (world units) along each escape path when tracing visibility.
GROUP_TRACE_STEP = 5.0
GROUPS_FILE = os.path.join(_SCRIPT_DIR, 'resources', 'corner_groups.pkl')
# Number of evaders for multi_evader.py (each corner is scored against the
# evader nearest to it, so any k pursuers / m evaders combination works).
NUM_EVADERS = 2
