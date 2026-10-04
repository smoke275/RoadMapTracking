#!/usr/bin/env bash
# Review-revision batch (2026-10-03): single-pursuer tables with the tau
# escape metric and the two new strategies (minmax-alpha-vis, greedy-los),
# speed sweeps across s* for k = 1 and for teams, and the team-table
# refresh with the corrected-V(e,c) partitions.
# Stages are sequential: the pipeline cache is single-slot.
set -u
cd "$(dirname "$0")"
SEEDS=${SEEDS:-50}
FRAMES=${FRAMES:-600}
WORKERS=${WORKERS:-22}
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=benchmark_results/logs/review_$STAMP
mkdir -p "$LOGDIR"
STATUS=$LOGDIR/status.txt
echo "seeds=$SEEDS frames=$FRAMES workers=$WORKERS" | tee "$STATUS"

run() {   # run <name> <cmd...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a "$STATUS"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "$(date '+%F %T') DONE  $name" | tee -a "$STATUS"
  else
    echo "$(date '+%F %T') FAIL  $name (see $LOGDIR/$name.log)" | tee -a "$STATUS"
  fi
}

ALL_STRATS="minmax-alpha minmax-alpha-vis geo-follow greedy-los tsp-patrol"

# 1) flagship polygon first: full strategy set, both evaders
run bench_poly9 python run_benchmark.py --polygons poly9 \
    --strategies $ALL_STRATS --evaders skeleton adversarial \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS
run bench_kernel_poly9 python run_benchmark.py --polygons poly9 \
    --strategies kernel-control --evaders skeleton \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS

# 2) k = 1 speed sweep across s* with the escape metric
for sp in 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
  run sweep_sp$sp python run_benchmark.py --polygons poly9 \
      --strategies minmax-alpha minmax-alpha-vis --evaders skeleton adversarial \
      --seeds $SEEDS --frames $FRAMES --workers $WORKERS --sp $sp
done

# 3) team speed sweep (k = 1,2,3 pursuers vs one evader)
for sp in 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
  run team_sweep_sp$sp python run_multi_evader.py --polygons poly9 \
      --k 1 2 3 --m 1 --configs 50 \
      --seeds $SEEDS --frames $FRAMES --workers $WORKERS --sp $sp
done

# 4) team table refresh (corrected-V partitions + escape metric)
run multi_evader_poly9 python run_multi_evader.py --polygons poly9 \
    --k 1 2 3 --m 1 2 3 --configs 300 \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS

# 5) remaining polygons: full strategy set, both evaders
for p in poly1 poly2 poly3 poly4 poly5 poly6 poly7 poly8; do
  run bench_$p python run_benchmark.py --polygons $p \
      --strategies $ALL_STRATS --evaders skeleton adversarial \
      --seeds $SEEDS --frames $FRAMES --workers $WORKERS
  run bench_kernel_$p python run_benchmark.py --polygons $p \
      --strategies kernel-control --evaders skeleton \
      --seeds $SEEDS --frames $FRAMES --workers $WORKERS
done

echo "$(date '+%F %T') ALL STAGES FINISHED" | tee -a "$STATUS"
