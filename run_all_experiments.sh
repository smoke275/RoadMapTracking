#!/usr/bin/env bash
# Run every experiment the paper reports, for every polygon, one after another
# (the pipeline cache is single-slot, so runs must not overlap).
#
# Usage (inside the roadmaptracking image, repo at /app):
#   ./run_all_experiments.sh                      # all sites_poly*.csv
#   POLYS="poly5 poly6" ./run_all_experiments.sh  # a subset
#   SEEDS=50 WORKERS=22 ./run_all_experiments.sh
#   MULTI_POLYS="poly9 poly3" MULTI_SEEDS=50 ./run_all_experiments.sh
#
# Stages (each writes to benchmark_results/, logs to benchmark_results/logs/):
#   1  environment stats (Table I)             run_environment_stats.py
#   2  single-pursuer speed bound (Table VI)   run_max_speed.py
#   3  tracking benchmark (Tables III, IV)     run_benchmark.py
#   4  multi-pursuer partitions (Table VII)    run_groups.py         (MULTI_POLYS only)
#   5  k pursuers x m evaders (Table VIII)     run_multi_evader.py   (MULTI_POLYS only;
#      its m = 1 rows are the multi-pursuer single-evader trials)
# A stage that fails is logged and the script moves on.
set -u
cd "$(dirname "$0")"
SEEDS=${SEEDS:-50}
FRAMES=${FRAMES:-600}
WORKERS=${WORKERS:-22}
MULTI_POLYS=${MULTI_POLYS:-poly9}   # team experiments (stages 4-5) run only on these
MULTI_SEEDS=${MULTI_SEEDS:-50}
POLYS=${POLYS:-$(ls resources/sites_poly*.csv | sed -E 's/.*sites_(poly[0-9]+)\.csv/\1/' | sort -V | tr '\n' ' ')}
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=benchmark_results/logs/$STAMP
mkdir -p "$LOGDIR"
STATUS=$LOGDIR/status.txt
echo "polygons: $POLYS  seeds=$SEEDS frames=$FRAMES workers=$WORKERS" | tee "$STATUS"

run() {   # run <name> <cmd...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a "$STATUS"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "$(date '+%F %T') DONE  $name" | tee -a "$STATUS"
  else
    echo "$(date '+%F %T') FAIL  $name (see $LOGDIR/$name.log)" | tee -a "$STATUS"
  fi
}

run env_stats   python run_environment_stats.py --polygons $POLYS
run max_speed   python run_max_speed.py --polygons $POLYS
for p in $POLYS; do
  run bench_$p        python run_benchmark.py --polygons $p \
                        --strategies minmax-alpha geo-follow tsp-patrol \
                        --evaders skeleton adversarial \
                        --seeds $SEEDS --frames $FRAMES --workers $WORKERS
  run bench_kernel_$p python run_benchmark.py --polygons $p \
                        --strategies kernel-control --evaders skeleton \
                        --seeds $SEEDS --frames $FRAMES --workers $WORKERS
done
for p in $MULTI_POLYS; do
  run groups_$p       python run_groups.py --polygons $p --k 1 2 3 --relax 0.1
done
for p in $MULTI_POLYS; do
  run multi_evader_$p python run_multi_evader.py --polygons $p --k 1 2 3 --m 1 2 3 \
                        --configs 300 --seeds $MULTI_SEEDS --frames $FRAMES --workers $WORKERS
done
echo "$(date '+%F %T') ALL STAGES FINISHED" | tee -a "$STATUS"
