#!/usr/bin/env bash
# Stop the all-polygon run once tracking is done, then run the team
# experiments on poly9 only with 50 seeds.
S=benchmark_results/logs/20260927_231315/status.txt
until grep -qE 'START groups_|ALL STAGES FINISHED' "$S"; do
  docker ps --format '{{.Names}}' | grep -qx roadmap-experiments || break
  sleep 20
done
docker stop roadmap-experiments >/dev/null 2>&1
echo "$(date '+%F %T') STOP  all-polygon run after tracking (team stages: poly9 only)" >> "$S"
docker rm -f roadmap-team >/dev/null 2>&1
docker run -d --name roadmap-team --user 1000:1000 -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
  -v "/home/smokemsi/Documents/Repository/RoadmapTracking":/app -w /app -e PYTHONPATH=/app roadmaptracking bash -c '
  S=20260927_231315; S=benchmark_results/logs/$S/status.txt
  echo "$(date "+%F %T") START groups_poly9" >> $S
  python run_groups.py --polygons poly9 --k 1 2 3 --relax 0.1 > benchmark_results/logs/'"20260927_231315"'/groups_poly9.log 2>&1 \
    && echo "$(date "+%F %T") DONE  groups_poly9" >> $S || echo "$(date "+%F %T") FAIL  groups_poly9" >> $S
  echo "$(date "+%F %T") START multi_evader_poly9" >> $S
  python run_multi_evader.py --polygons poly9 --k 1 2 3 --m 1 2 3 --configs 300 --seeds 50 --frames 600 --workers 22 > benchmark_results/logs/'"20260927_231315"'/multi_evader_poly9.log 2>&1 \
    && echo "$(date "+%F %T") DONE  multi_evader_poly9" >> $S || echo "$(date "+%F %T") FAIL  multi_evader_poly9" >> $S
  echo "$(date "+%F %T") ALL STAGES FINISHED" >> $S'
