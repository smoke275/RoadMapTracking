#!/usr/bin/env bash
# Build the whole video: 2D segments (four containers in parallel), cards,
# two Gazebo takes (needs the desktop: keep the Gazebo window unobscured
# while it records), then assemble. ~15 min end to end.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=video/out
mkdir -p $OUT/frames
DOCK="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/app -w /app -e PYTHONPATH=/app roadmaptracking"
if [ "${SKIP_2D:-}" != 1 ]; then
  for seg in problem roadmap race team; do
    $DOCK python video/render2d.py $seg --out /app/$OUT/frames > $OUT/render_$seg.log 2>&1 &
  done
  python3 video/cards.py $OUT > $OUT/cards.log 2>&1 &
  wait
  grep -hE "^\[|done in" $OUT/render_*.log
fi
if [ "${SKIP_GZ:-}" != 1 ]; then
  python3 video/gz_record.py $OUT/gz_alpha --strategy alpha-guard --seconds 70 --shots full
  python3 video/gz_record.py $OUT/gz_geo --strategy geo-follow --seconds 40 --shots drone
fi
python3 video/assemble.py $OUT
