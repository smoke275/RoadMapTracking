"""Polygon fingerprinting and disk cache for expensive KER pipeline results."""
import hashlib
import os
import pickle

import pyvisgraph as vg
from scipy.sparse.csgraph import csgraph_from_dense

from config import CACHE_FILE, GEO_FILE
from graph import Geodesic


def poly_fingerprint(poly) -> str:
    """SHA-256 of cleaned polygon vertex coords rounded to 2 dp."""
    coords = tuple((round(p.x(), 2), round(p.y(), 2)) for p in poly)
    return hashlib.sha256(str(coords).encode()).hexdigest()


def save_cache(fingerprint: str, payload: dict, geodesic: Geodesic):
    """Pickle payload and save geodesic graph to disk."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'fingerprint': fingerprint, 'data': payload}, f, protocol=4)
        geodesic.graph.save(GEO_FILE)
        print('[CACHE] Saved.')
    except Exception as e:
        print(f'[CACHE] Save failed: {e}')


def load_cache(fingerprint: str):
    """Return (payload dict, Geodesic) if cache is valid, else (None, None)."""
    try:
        if not os.path.exists(CACHE_FILE) or not os.path.exists(GEO_FILE):
            return None, None
        with open(CACHE_FILE, 'rb') as f:
            blob = pickle.load(f)
        if blob.get('fingerprint') != fingerprint:
            print('[CACHE] Polygon changed — recomputing.')
            return None, None
        geo = vg.VisGraph()
        geo.load(GEO_FILE)
        print('[CACHE] Loaded.')
        return blob['data'], geo
    except Exception as e:
        print(f'[CACHE] Load failed: {e}')
        return None, None
