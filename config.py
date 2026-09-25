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

# ---- Multi-pursuer (see corner_groups.py / multi_window.py) ----------------
# Number of pursuers. 1 runs the original single-pursuer demo; >1 partitions
# the reflex corners into that many groups (clustered on how often two
# corners' escapes can be watched from the same roadmap spot) and gives each
# group its own pursuer.
NUM_PURSUERS = 2
# Evader-grid resolution per axis used to average the corner-pair overlap
# affinity (see corner_groups.corner_affinity). Coarser = faster startup.
GROUP_AFFINITY_GRID = 10
# Sampling step (world units) along each escape path when tracing visibility.
GROUP_TRACE_STEP = 5.0
GROUPS_FILE = os.path.join(_SCRIPT_DIR, 'resources', 'corner_groups.pkl')
# Number of evaders for multi_evader.py (each corner is scored against the
# evader nearest to it, so any k pursuers / m evaders combination works).
NUM_EVADERS = 2
