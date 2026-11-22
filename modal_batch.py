"""
Run stages of run_witness_batch.sh on Modal, one container per stage.

Each stage is the same command the local batch runs. Stages run in parallel,
because every container has its own copy of the repo and so its own pipeline
cache. Result tables, raw CSVs and logs are written to the Modal volume
`roadmap-results` and fetched with the `fetch` entrypoint.

    modal run --detach modal_batch.py::launch --stages test
    modal run --detach modal_batch.py::launch --stages sweeps,poly8,poly2,poly1
    modal run modal_batch.py::fetch --run-id <id printed by launch>

Stage groups: test (poly3 tracking, to compare with the local result),
polyN (tracking + kernel baseline for that polygon), sweeps (single-pursuer
and team speed sweeps on poly9), all.

Package versions are pinned to those of the local `roadmaptracking` Docker
image so that the integer programs pick the same guard sets.
"""
import datetime
import os
import subprocess

import modal

CORES = 16                # physical cores per stage (also the worker count)
SEEDS, FRAMES = 50, 600
ALL_STRATS = 'minmax-alpha minmax-alpha-vis geo-follow greedy-los tsp-patrol'
SPEEDS = ['0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0']

image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('build-essential', 'swig', 'libgl1', 'libglib2.0-0', 'libsm6',
                 'libice6', 'libxext6', 'libxrender1', 'libxi6',
                 'libxkbcommon-x11-0', 'libxcb-cursor0', 'libxcb-xinerama0',
                 'libxcb-icccm4', 'libxcb-image0', 'libxcb-keysyms1',
                 'libxcb-render-util0', 'libxcb-shape0', 'libxcb-randr0',
                 'libxcb-xfixes0', 'libxcb1', 'libx11-xcb1', 'libdbus-1-3',
                 'libfontconfig1', 'libfreetype6')
    .pip_install('numpy==2.4.6', 'scipy==1.17.1', 'shapely==2.1.2',
                 'pyclipper==1.4.0', 'pyvisgraph==0.2.1', 'tqdm==4.70.0',
                 'VisiLibity==1.0.10', 'PyQt5==5.15.11', 'PyQt5-Qt5==5.15.19',
                 'PyQt5_sip==12.19.0', 'pygame==2.6.1', 'matplotlib==3.11.1',
                 'seaborn==0.13.2', 'bidict==0.23.1')
    .add_local_dir('.', '/app', ignore=[
        '.git', 'video', 'ros2_sim', 'RoadmapBasedTracking', 'benchmark_results',
        '**/__pycache__', '*.pdf', '*.log', 'resources/data*', 'resources/optimal_data',
        'resources/ker_cache.pkl', 'resources/ker_geodesic.graph', 'resources/output.txt',
        'run_witness_batch*.sh',
        'resources/*.png', 'resources/dog_breeds*', 'resources/buffer_output.txt'])
)

app = modal.App('roadmap-witness-batch', image=image)
results = modal.Volume.from_name('roadmap-results', create_if_missing=True)


def _stages(groups):
    """{stage name: argv} for the requested groups."""
    w = ['--seeds', str(SEEDS), '--frames', str(FRAMES), '--workers', str(CORES)]
    out = {}

    def bench(p):
        out[f'bench_{p}'] = (['python', 'run_benchmark.py', '--polygons', p, '--strategies']
                             + ALL_STRATS.split() + ['--evaders', 'skeleton', 'adversarial'] + w)
        out[f'bench_kernel_{p}'] = (['python', 'run_benchmark.py', '--polygons', p,
                                     '--strategies', 'kernel-control', '--evaders', 'skeleton'] + w)

    for g in groups:
        if g == 'test':
            bench('poly3')
        elif g.startswith('poly'):
            bench(g)
        elif g == 'sweeps':
            for sp in SPEEDS:
                out[f'sweep_sp{sp}'] = (['python', 'run_benchmark.py', '--polygons', 'poly9',
                                         '--strategies', 'minmax-alpha', 'minmax-alpha-vis',
                                         '--evaders', 'skeleton', 'adversarial'] + w + ['--sp', sp])
                out[f'team_sweep_sp{sp}'] = (['python', 'run_multi_evader.py', '--polygons', 'poly9',
                                              '--k', '1', '2', '3', '--m', '1', '--configs', '50']
                                             + w + ['--sp', sp])
        elif g == 'all':
            return _stages(['poly%d' % i for i in range(1, 10)] + ['sweeps'])
        else:
            raise ValueError(f'unknown stage group {g!r}')
    return out


@app.function(cpu=CORES, memory=CORES * 512, timeout=10 * 3600, volumes={'/results': results})
def run_stage(run_id: str, name: str, argv: list) -> dict:
    import shutil
    import time
    env = dict(os.environ, PYTHONPATH='/app', PYTHONUNBUFFERED='1', QT_QPA_PLATFORM='offscreen')
    os.makedirs('/app/benchmark_results', exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(argv, cwd='/app', env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    out_dir = f'/results/{run_id}/{name}'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/{name}.log', 'w') as f:
        f.write(proc.stdout)
    files = []
    for fn in sorted(os.listdir('/app/benchmark_results')):
        src = os.path.join('/app/benchmark_results', fn)
        if os.path.isfile(src):
            shutil.copy(src, out_dir)
            files.append(fn)
    results.commit()
    return {'name': name, 'exit': proc.returncode, 'minutes': round((time.time() - t0) / 60, 1),
            'files': files, 'tail': proc.stdout[-600:]}


@app.local_entrypoint()
def launch(stages: str = 'test'):
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    todo = _stages([s for s in stages.split(',') if s])
    print(f'RUN_ID {run_id}: {len(todo)} stages on {CORES} cores each: {", ".join(todo)}')
    calls = [run_stage.spawn(run_id, name, argv) for name, argv in todo.items()]
    for c in calls:
        r = c.get()
        status = 'DONE' if r['exit'] == 0 else 'FAIL'
        print(f'{status} {r["name"]}  {r["minutes"]} min  files={r["files"]}')
        if r['exit'] != 0:
            print(r['tail'])
    print(f'RUN_ID {run_id} finished. Fetch with: modal run modal_batch.py::fetch --run-id {run_id}')


@app.local_entrypoint()
def fetch(run_id: str):
    dest = os.path.join('benchmark_results', f'modal_{run_id}')
    os.makedirs(dest, exist_ok=True)
    n = 0
    for entry in results.iterdir(f'/{run_id}', recursive=True):
        if entry.type == modal.volume.FileEntryType.FILE:
            local = os.path.join(dest, os.path.relpath(entry.path, run_id))
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, 'wb') as f:
                for chunk in results.read_file(entry.path):
                    f.write(chunk)
            n += 1
    print(f'{n} files written to {dest}')
