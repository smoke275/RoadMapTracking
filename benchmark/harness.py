"""Headless trial runner: no Qt window, no pygame, pure computation."""
import csv
import math
import multiprocessing
import os
import random
import statistics
import time
from dataclasses import dataclass, field

import visilibity as vis

import ker_pipeline
from geometry import get_random_point_in_polygon
from ker_pipeline import compute_path_lengths

from benchmark.evaders import AdversarialEvader, SkeletonEvader
from benchmark.metrics import TrialRecorder, alphas_at
from benchmark.pursuers import PURSUER_CLASSES

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES = os.path.join(_SCRIPT_DIR, 'resources')

EVADER_CLASSES = {'skeleton': SkeletonEvader, 'adversarial': AdversarialEvader}


def load_poly(name: str) -> list:
    """Load a polygon: 'current' (config.FILE_NAME — the polygon the drawing
    tool saves to), a short preset name ('poly2'), or a CSV path."""
    if name == 'current':
        from config import FILE_NAME
        path = FILE_NAME
    elif os.path.isfile(name):
        path = name
    else:
        path = os.path.join(RESOURCES, f'sites_{name}.csv')
    points = []
    with open(path) as f:
        for row in csv.reader(f):
            points.append((float(row[0]), float(row[1])))
    if len(points) < 3:
        raise ValueError(f'{path}: polygon has fewer than 3 points')
    return [vis.Point(p[0], p[1]) for p in points]


@dataclass
class TrialResult:
    environment: str
    strategy: str
    evader: str
    per_seed: list = field(default_factory=list)   # list of metric dicts

    def aggregate(self) -> dict:
        """mean and std (population of seeds) for each metric."""
        out = {}
        for key in self.per_seed[0]:
            vals = [m[key] for m in self.per_seed if math.isfinite(m[key])]
            if not vals:
                out[key] = (float('nan'), float('nan'))
            else:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                out[key] = (mean, std)
        return out


def _make_evader(model: str, data, start, speed):
    if model == 'skeleton':
        return SkeletonEvader(data.skel_nodes, data.skel_adj,
                              data.shapely_env, start, speed)
    if model == 'adversarial':
        return AdversarialEvader(data, start, speed)
    raise ValueError(f'unknown evader model: {model}')


def run_trial(data, seed: int, strategy: str = 'minmax-alpha',
              evader_model: str = 'skeleton', frames: int = 600,
              s_p: float = 0.8, s_e: float = 1.0) -> dict:
    """One simulation trial. Speeds are world units per frame (paper setup)."""
    random.seed(seed)

    ev_start = get_random_point_in_polygon(data.shapely_env)
    evader = _make_evader(evader_model, data, (ev_start.x, ev_start.y), s_e)
    # Pursuer spawns at its own strategy's initial target (patrol assumed
    # converged before the trial), avoiding spawn-location bias in alpha_max.
    pursuer = PURSUER_CLASSES[strategy](data, (ev_start.x, ev_start.y), s_p)
    recorder = TrialRecorder()

    for _ in range(frames):
        ex, ey = evader.pos
        path_lengths = compute_path_lengths(ex, ey, data)
        pursuer.step((ex, ey), path_lengths)

        # Metrics with a consistent snapshot: pursuer just moved, evader
        # still at the position path_lengths was computed for.
        recorder.record(pursuer.pos, (ex, ey), path_lengths, data)

        p_alphas = alphas_at(pursuer.pos, path_lengths, data)
        evader.step(dt=1.0, pursuer_alphas=p_alphas)

    return recorder.metrics()


def run_config(polygon: str, strategy: str = 'minmax-alpha',
               evader_model: str = 'skeleton',
               n_seeds: int = 5, frames: int = 600,
               s_p: float = 0.8, s_e: float = 1.0,
               data=None) -> TrialResult:
    """Run n_seeds independent trials of one (polygon, strategy, evader) config.
    Pass a prebuilt `data` to share one pipeline build across configs."""
    if data is None:
        poly = load_poly(polygon)
        data = ker_pipeline.build(poly, renderer=None, force_recompute=True)

    result = TrialResult(environment=polygon, strategy=strategy,
                         evader=evader_model)
    for seed in range(n_seeds):
        print(f'  [{polygon} | {strategy} | {evader_model}] '
              f'seed {seed + 1}/{n_seeds} ...', flush=True)
        result.per_seed.append(
            run_trial(data, seed=seed, strategy=strategy,
                      evader_model=evader_model, frames=frames,
                      s_p=s_p, s_e=s_e))
    return result


# ---------------------------------------------------------------------------
# Parallel runner — each trial is ~1-2 minutes of real computation, so
# hundreds/thousands of seeds are only practical spread across CPU cores.
# ---------------------------------------------------------------------------
_worker_data = None   # set via Pool(initializer=...); inherited by forked
                       # children via copy-on-write, never pickled — this is
                       # what makes sharing an unpicklable SimulationData
                       # (wraps SWIG vis.Environment / pyvisgraph objects)
                       # across processes possible at all.


def _init_worker(data):
    global _worker_data
    _worker_data = data


def _run_one_seed(args):
    seed, strategy, evader_model, frames, s_p, s_e = args
    return seed, run_trial(_worker_data, seed=seed, strategy=strategy,
                           evader_model=evader_model, frames=frames,
                           s_p=s_p, s_e=s_e)


def run_config_parallel(polygon: str, strategy: str = 'minmax-alpha',
                        evader_model: str = 'skeleton',
                        n_seeds: int = 1000, frames: int = 600,
                        s_p: float = 0.8, s_e: float = 1.0,
                        n_workers: int = None, data=None,
                        progress_every: int = 25) -> TrialResult:
    """Same as run_config, but distributes seeds across a process pool using
    the 'fork' start method — requires Linux/macOS, not Windows. The
    pipeline build happens once here (avoiding N concurrent writers to the
    on-disk cache file) and is inherited by workers via COW fork, not
    pickled, since SimulationData holds unpicklable C-extension objects."""
    if data is None:
        poly = load_poly(polygon)
        data = ker_pipeline.build(poly, renderer=None, force_recompute=True)

    n_workers = n_workers or multiprocessing.cpu_count()
    ctx = multiprocessing.get_context('fork')
    jobs = [(seed, strategy, evader_model, frames, s_p, s_e)
            for seed in range(n_seeds)]

    per_seed = [None] * n_seeds
    t0 = time.time()
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(data,)) as pool:
        done = 0
        for seed, metrics in pool.imap_unordered(_run_one_seed, jobs):
            per_seed[seed] = metrics
            done += 1
            if done % progress_every == 0 or done == n_seeds:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n_seeds - done) / rate if rate > 0 else float('inf')
                print(f'  [{polygon} | {strategy} | {evader_model}] '
                      f'{done}/{n_seeds} trials  '
                      f'({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, '
                      f'{n_workers} workers)', flush=True)

    result = TrialResult(environment=polygon, strategy=strategy,
                         evader=evader_model, per_seed=per_seed)
    return result
