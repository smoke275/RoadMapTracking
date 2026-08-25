"""
Measure the actual KER+ILP guard-placement topology (paper Table 2's
"Proposed" column: |S|, |E|, Length, %Area) using the guard cover the
pipeline already computes, and compare it against the paper's claimed
values.

Unlike run_benchmark.py, this has no --seeds/--frames — a guard cover is a
deterministic, one-time property of the pipeline, not something with
trial-to-trial variance.

Run:  python run_topology.py                          # the current polygon
      python run_topology.py poly2 poly4 poly8 poly9   # specific presets
      python run_topology.py my_env.csv                # a specific file
"""
import argparse

from benchmark.topology import format_topology_table, measure_topology


def main():
    parser = argparse.ArgumentParser(description='KER+ILP guard-cover topology report')
    parser.add_argument('polygons', nargs='*', default=['current'],
                        help="Polygons to measure: 'current' (default), preset "
                             'names (polyN), or CSV paths')
    args = parser.parse_args()

    results = []
    for p in args.polygons:
        print(f'[{p}] measuring guard cover ...', flush=True)
        results.append(measure_topology(p))

    print()
    print(format_topology_table(results))


if __name__ == '__main__':
    main()
