#!/usr/bin/env bash
# Second pass (evader waypoints seeded per trial, see skeleton.seed_evader_rng).
# Tracking and multi-evader stages only; the speed sweeps run on Modal
# (modal_batch.py) and the tables without trials are kept from the first pass.
# Rerun of every simulation result in the paper with the witness-point guard
# selection (config.GUARD_SELECTION = 'witness', Algorithm 1 of Section IV).
# Same stages and settings as run_review_batch.sh plus the environment table,
# the speed bound, the polygon descriptors, and the team partitions.
# Stages are sequential: the pipeline cache is single-slot.
set -u
cd "$(dirname "$0")"
SEEDS=${SEEDS:-50}
FRAMES=${FRAMES:-600}
WORKERS=${WORKERS:-22}
POLYS="poly1 poly2 poly3 poly4 poly5 poly6 poly7 poly8 poly9"
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=benchmark_results/logs/witness2_$STAMP
mkdir -p "$LOGDIR"
STATUS=$LOGDIR/status.txt
echo "guard selection: witness  seeds=$SEEDS frames=$FRAMES workers=$WORKERS" | tee "$STATUS"

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

# 0) poly3 first: it is the check against the Modal run
run bench_poly3 python run_benchmark.py --polygons poly3 \
    --strategies $ALL_STRATS --evaders skeleton adversarial \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS
run bench_kernel_poly3 python run_benchmark.py --polygons poly3 \
    --strategies kernel-control --evaders skeleton \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS

# 3) flagship polygon: full strategy set, both evaders
run bench_poly9 python run_benchmark.py --polygons poly9 \
    --strategies $ALL_STRATS --evaders skeleton adversarial \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS
run bench_kernel_poly9 python run_benchmark.py --polygons poly9 \
    --strategies kernel-control --evaders skeleton \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS

# 4) k pursuers against m evaders on poly9
run multi_evader_poly9 python run_multi_evader.py --polygons poly9 \
    --k 1 2 3 --m 1 2 3 --configs 300 \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS

# 5) remaining polygons: full strategy set, both evaders
for p in poly7 poly4 poly6 poly5 poly1 poly2 poly8; do
  run bench_$p python run_benchmark.py --polygons $p \
      --strategies $ALL_STRATS --evaders skeleton adversarial \
      --seeds $SEEDS --frames $FRAMES --workers $WORKERS
  run bench_kernel_$p python run_benchmark.py --polygons $p \
      --strategies kernel-control --evaders skeleton \
      --seeds $SEEDS --frames $FRAMES --workers $WORKERS
done

echo "$(date '+%F %T') ALL STAGES FINISHED" | tee -a "$STATUS"
