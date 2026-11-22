#!/usr/bin/env bash
# Experiments of the paper that run_minspeed_batch.sh does not cover:
#   - speed sweep against the escaping evader on poly9 (middle panel of the
#     speed-sweep figure, four strategies)
#   - the watchman route as the roadmap on poly9, and Watchman-Patrol on the
#     KER roadmap (Discussion)
# Run after run_minspeed_batch.sh has finished (the pipeline cache is
# single-slot). Resumable in the same way.
set -u
cd "$(dirname "$0")"
SEEDS=${SEEDS:-50}
FRAMES=${FRAMES:-600}
WORKERS=${WORKERS:-22}
PREFIX=${PREFIX:-native}_extra
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=benchmark_results/logs/${PREFIX}_$STAMP
mkdir -p "$LOGDIR"
STATUS=$LOGDIR/status.txt
echo "extra stages  seeds=$SEEDS frames=$FRAMES workers=$WORKERS" | tee "$STATUS"

run() {   # run <name> <cmd...>
  local name=$1; shift
  if grep -qs "DONE  $name\$" benchmark_results/logs/${PREFIX}_*/status.txt; then
    echo "$(date '+%F %T') SKIP  $name (already done)" | tee -a "$STATUS"; return
  fi
  echo "$(date '+%F %T') START $name" | tee -a "$STATUS"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "$(date '+%F %T') DONE  $name" | tee -a "$STATUS"
  else
    echo "$(date '+%F %T') FAIL  $name (see $LOGDIR/$name.log)" | tee -a "$STATUS"
  fi
}

for sp in 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
  run esc_sweep_sp$sp python run_benchmark.py --polygons poly9 \
      --strategies minmax-alpha minmax-alpha-vis geo-follow greedy-los --evaders escaping \
      --seeds $SEEDS --frames $FRAMES --workers $WORKERS --sp $sp
done

WM_STRATS="minmax-alpha minmax-alpha-vis geo-follow greedy-los tsp-patrol watchman-patrol"
run watchman_poly9 python run_benchmark.py --polygons poly9 --roadmap watchman \
    --strategies $WM_STRATS --evaders skeleton adversarial \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS
run watchman_patrol_ker_poly9 python run_benchmark.py --polygons poly9 --roadmap ker \
    --strategies watchman-patrol --evaders skeleton adversarial \
    --seeds $SEEDS --frames $FRAMES --workers $WORKERS

echo "$(date '+%F %T') ALL STAGES FINISHED" | tee -a "$STATUS"
