# RoadMapTracking

KER-based pursuit-evasion simulation with roadmap-guided pursuer motion.

A pursuer navigates a polygonal environment using a precomputed patrol path derived from **Kernel Edge Rays (KER)** — visibility-theoretic guard points placed at the kernels of star-shaped subregions induced by the polygon's corners. An evader moves freely inside the polygon; the pursuer tracks it along a geodesic roadmap.

---

## Architecture

The simulation is split into 9 modules:

| Module | Responsibility |
|---|---|
| `config.py` | Shared constants and resource paths |
| `geometry.py` | Low-level math helpers (slopes, interpolation, distances) |
| `graph.py` | `Geodesic` class, visibility graph, Dijkstra |
| `skeleton.py` | Voronoi skeleton construction and autonomous evader navigation |
| `cache.py` | Polygon fingerprinting (SHA-256) and disk cache |
| `ker_pipeline.py` | One-time KER pipeline and per-frame pure computation |
| `draw_polygon.py` | Pygame-based polygon drawing tool (pre-simulation) |
| `window.py` | PyQt5 window, rendering, and interactive loop |
| `main.py` | Entry point |

---

## Requirements

```
python >= 3.10
PyQt5
pygame
visilibity
pyvisgraph
shapely
pyclipper
scipy
seaborn
bidict
numpy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

`visilibity` builds from source on install and needs a C++ toolchain and `swig` available.

---

## Docker

A `Dockerfile` and `docker-run.sh` are included so you don't need to install the toolchain locally. The repo is bind-mounted into the container rather than copied in, so edits on the host are picked up immediately without rebuilding.

```bash
./docker-run.sh
```

This builds the `roadmaptracking` image on first run, then drops you into a shell in the `roadmaptracking-dev` container with the repo mounted at `/app`. Running it again attaches to that same container (starting it back up if it was stopped) instead of creating a new one. From the shell:

```bash
python main.py
```

Other flags:

```bash
./docker-run.sh --build   # force a rebuild of the image
./docker-run.sh --rm      # stop and remove the dev container
```

The container is launched with `--net=host` and the host's `/tmp/.X11-unix` mounted in, so PyQt5/pygame windows render on the host's X display — this setup targets Linux with a running X server.

GPU-accelerated GLX is disabled in favor of software rendering (`LIBGL_ALWAYS_SOFTWARE=1`, `QT_XCB_GL_INTEGRATION=none`), since the container can't use a host GPU that's bound to a proprietary driver (e.g. NVIDIA). This app only does 2D drawing, so there's no performance cost.

---

## Usage

```bash
python main.py
```

Draw a polygon with the pygame tool (SPACE to close, BACKSPACE to undo, ESC when done), then the simulation window opens.

### Force recompute (bypass cache)

```bash
python main.py --recompute
# or
python main.py -r
```

The pipeline caches results keyed by a SHA-256 fingerprint of the polygon. Use `-r` after changing the polygon or algorithm parameters to recompute from scratch.

---

## Controls

| Input | Action |
|---|---|
| Drag **red** dot | Move evader anywhere inside the polygon |
| Drag **green** dot | Slide observer/pursuer along the patrol path |
| Drag **cyan** dot | Cycle active corner |
| `A` | Toggle autonomous evader (Voronoi skeleton navigation) |
| `P` | Toggle roadmap pursuer |
| `Esc` | Quit |

---

## KER Pipeline

1. **Corner detection** — finds reflex and convex corners of the polygon
2. **KER boundary contraction** — contracts each star-shaped subregion to locate its kernel
3. **KER points** — one guard candidate per subregion kernel
4. **Set cover** — selects the minimum subset of KER points whose visibility polygons cover all corners
   - Exact solution via `scipy.optimize.milp` (ILP) with domination pre-processing
   - Falls back to greedy $\ln|C|$-approximation if MILP is unavailable or infeasible
5. **Coverage sanity check** — verifies the union of guard visibility polygons covers ≥ 99.9% of the environment area
6. **Patrol path** — geodesic shortest paths between guard positions form a cyclic patrol route
7. **Voronoi skeleton** — used for autonomous evader motion planning

---

## Polygon files

Polygon presets are in `resources/sites_poly{1-9}.csv`. Edit `config.py` (`FILE_NAME`) to switch polygons.
