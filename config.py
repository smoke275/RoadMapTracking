"""Shared constants and file-path configuration for the KER simulation."""
import os

EPSILON      = 1e-7
BOUNDARY_X   = 500
BOUNDARY_Y   = 500
POINT_RADIUS = 8
WINDOW_SIZE  = 750

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME   = os.path.join(_SCRIPT_DIR, 'resources', 'sites_poly3.csv')
CACHE_FILE  = os.path.join(_SCRIPT_DIR, 'resources', 'ker_cache.pkl')
GEO_FILE    = os.path.join(_SCRIPT_DIR, 'resources', 'ker_geodesic.graph')
