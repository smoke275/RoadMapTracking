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

### Skip the polygon-drawing tool

```bash
python main.py --skip-draw
# or
python main.py -s
```

Loads the polygon straight from `resources/` (`config.py`'s `FILE_NAME`) instead of opening the interactive pygame drawing tool — useful once you already have a polygon saved and just want to re-run the simulation.

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

---

## Line-of-sight trace tool

```bash
python los_trace.py --skip-draw          # current polygon
python los_trace.py --skip-draw --step 2 # finer sampling along the escape path
```

No pursuer. Drag the red dot to place the evader, click a reflex corner (or press `C` to cycle) to pick the escape target. The tool walks the evader along its geodesic to that corner and draws, in green, the part of the patrol roadmap that stays visible to the evader for the **whole** trip. In a simple polygon that set is exactly the intersection of the visible roadmap at the path's vertices (start, each reflex vertex it bends around, and the corner), so it is computed from those few viewpoints, which are memoised per polygon (`VisCache`). `SPACE` replays the walk showing the visibility polygon and the visible roadmap at each sampled instant; the samples are for display only. Pure computation lives in `roadmap_los.py`.

---

## Multiple pursuers

Set `NUM_PURSUERS = k` in `config.py` and run `python main.py` as usual; with `k > 1` the k-pursuer window (`multi_window.py`) opens instead of the single-pursuer demo.

**Method.** The minimax guard optimisation is separable by corner, so partitioning the reflex corners into `k` groups turns the problem into `k` independent single-pursuer problems on the shared roadmap. The partition is chosen by an exact mixed-integer program (`partition_ilp.py`, solved with scipy's HiGHS): it assigns corners to pursuers and, for every evader grid position, a roadmap position to each pursuer, minimising the team critical speed ratio and then, at that bound, maximising the fraction of escapes the responsible pursuer keeps in line of sight from where it stands. Relaxing the speed bound traces the speed/sight Pareto front. A clustering heuristic (`corner_groups.py`: co-visibility affinity from the trace tool above + average-linkage clustering + local search) is kept as a comparison point and as the fallback for instances too large to solve exactly.

```bash
python run_groups.py --polygons poly9 --k 1 2 3                 # exact ILP, stores the partition
python run_groups.py --polygons poly9 --k 2 3 --relax 0.1 0.25  # extra Pareto points
python run_groups.py --method heuristic --refine                # clustering + local search
python run_groups.py --method heuristic --raw --show-affinity   # raw clustering, affinity table
```

Stored partitions live in `resources/corner_groups.pkl`; the GUI prefers the ILP partition, then a refined heuristic one, then raw clustering computed on the spot.

Window controls: drag the red dot (evader), `A` auto-evader, `P` freeze/unfreeze pursuers, `V` visibility polygons, `R` respawn pursuers at their guards, `Esc` quit. The HUD shows *team α\** (what `k` zero-transit pursuers would achieve for this evader position) and *achieved α* (max over corners of the best pursuer's actual distance ratio).

## Multiple evaders

```bash
python multi_evader.py --skip-draw -k 3 -m 3 --auto
```

`k` pursuers and `m` evaders (defaults `NUM_PURSUERS` / `NUM_EVADERS` in `config.py`). Pursuers guard corners, not evaders: each reflex corner is scored against the evader that can reach it first, `L_c = min_j d_geo(e_j, c)`, and every pursuer runs the unchanged single-evader optimiser on its corner group, so no run-time assignment is needed and any `k`/`m` combination works. Drag any red dot to move that evader, `A` sets them all wandering, `N` toggles the links showing which evader defines each corner. The HUD reports each evader's own worst corner alpha and whether any pursuer currently sees it.

