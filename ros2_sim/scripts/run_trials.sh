#!/usr/bin/env bash
# Batch runner for the Gazebo case study: for each strategy and seed, run
# one headless, unthrottled closed-loop trial and collect the metrics CSV
# under ros2_sim/logs/trials/. Runs sequentially (host networking means
# concurrent containers would need distinct IGN_PARTITION/ROS_DOMAIN_ID;
# unthrottled trials are fast enough that sequential is simpler and safe).
#
# Usage (from the repo root on the host):
#   ros2_sim/scripts/run_trials.sh                      # defaults
#   STRATEGIES="alpha-guard" SEEDS="1 2 3" ros2_sim/scripts/run_trials.sh
set -u
cd "$(dirname "$0")/../.."     # repo root
STRATEGIES=${STRATEGIES:-"alpha-guard naive-dijkstra geo-follow"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
DURATION=${DURATION:-180}      # sim seconds per trial
OUTDIR=ros2_sim/logs/trials
mkdir -p "$OUTDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
STATUS=$OUTDIR/${STAMP}_status.txt
echo "strategies: $STRATEGIES  seeds: $SEEDS  duration: ${DURATION}s" | tee "$STATUS"

for strat in $STRATEGIES; do
  for seed in $SEEDS; do
    csv=$OUTDIR/${STAMP}_${strat}_seed${seed}.csv
    echo "$(date '+%F %T') START $strat seed=$seed" | tee -a "$STATUS"
    timeout 900 docker run --rm --net=host --user "$(id -u):$(id -g)" \
      -e HOME=/tmp -e IGN_PARTITION=trial_${strat}_${seed} -e ROS_DOMAIN_ID=42 \
      -v "$PWD":/app ros2_sim bash -c "
        source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash &&
        timeout 850 ros2 launch ros2_sim sim.launch.py headless:=true rviz:=false \
          world:=poly9_world_fast strategy:=$strat evader_seed:=$seed \
          run_duration:=$DURATION log_csv:=/app/$csv" \
      > "$OUTDIR/${STAMP}_${strat}_seed${seed}.log" 2>&1
    rows=$(wc -l < "$csv" 2>/dev/null || echo 0)
    echo "$(date '+%F %T') DONE  $strat seed=$seed rows=$rows" | tee -a "$STATUS"
  done
done
echo "$(date '+%F %T') ALL TRIALS FINISHED" | tee -a "$STATUS"
