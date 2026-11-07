# Video storyboard — "From Optimal Placement to Team Tracking on Patrol Roadmaps in Polygons"

Target: 3:00, 1920×1080, 30 fps, H.264, no audio (captions burned in; a
narration script per segment is below for an optional voiceover).
Every number quoted on screen is taken from `RoadmapBasedTracking/sections/07_experiments.tex`.

| # | Segment | Source | Length |
|---|---------|--------|--------|
| 0 | Title card | `cards.py` | 0:06 |
| 1 | The problem: shadows of reflex corners | `render2d.py problem` | 0:20 |
| 2 | Building the patrol roadmap (KER → ILP cover → geodesic graph) | `render2d.py roadmap` | 0:22 |
| 3 | Online placement and tracking, proposed vs. Geo-Follow, same seed | `render2d.py race` | 0:30 |
| 4 | Results: nine polygons, five baselines, critical speed, speed sweep | `cards.py` (animated charts) | 0:24 |
| 5 | Teams: MILP corner partition, 3 pursuers vs. 3 evaders | `render2d.py team` | 0:25 |
| 6 | Closed loop in ROS 2 / Gazebo: hexacopter over poly9 | `gz_record.py` + `assemble.py` | 0:45 |
| 7 | Summary card | `cards.py` | 0:08 |

## Captions and narration

**0 Title.** Title, authors, Iowa State University. Three-line outline.

**1 Problem (poly9, escaping evader, pursuer held still).**
- "A pursuer confined to a patrol roadmap tracks an evader that moves freely inside a known polygon."
- "The evader can break line of sight only by crossing into the shadow of a reflex corner."
- "Occupying that corner's association region before the evader arrives is necessary to prevent the escape."

**2 Roadmap.**
- "Reflex corners of poly9: 17." (corners pop)
- "Keil's kernel candidates: 1,477 points." (candidate pool fades in)
- "An exact 0/1 set cover keeps 7, so every association region holds a patrol node." (guards ring, pool fades)
- "Geodesic patrol roadmap between the guards." (edges draw)
- "Nine benchmark polygons: 360–2,784 candidates → 5–13 guards; 95.6–100% of the area seen."

**3 Race (proposed Min-Max +Vis vs. Geo-Follow, skeleton evader, s_p/s_e = 0.8).**
- "Each corner is a timing race: pursuer roadmap distance to the association region over evader distance to the corner (α)."
- "Per frame, the optimal guard g* is a weighted network 1-center, solved exactly by an upper envelope."
- "A stable-node controller turns placement into tracking; an ε-slack layer spends spare timing margin on sight."
- Live strip: worst-corner α, %LOS, escapes for both panels.
- "Lowest ᾱ on all nine environments, up to 42% below Geo-Follow; +Vis adds 0.7–7.7 points of %LOS at no timing cost."

**4 Results cards.**
- Card A: ᾱ and %LOS, mean over the nine polygons, six strategies (Table I).
- Card B: critical speed ratio s* per polygon: 1.42–2.71×; "no single pursuer at or below the evader's speed can win every corner race".
- Card C: speed sweep on poly9: %LOS 81% → 96% and escapes cut threefold (skeleton); with margin (s_p/s_e ≥ 1.4) the fewest escapes of all methods, 0.56 per trial.

**5 Team (poly9, k = 3, m = 3).**
- "A mixed-integer program partitions the corners: team speed ratio 1.682 (one pursuer) → 1.134 (k = 2) → 1.004 (k = 3)."
- "Nearest-evader rule: each corner races the evader closest to it."
- "Three pursuers keep 89.1% of three evaders in view; one pursuer keeps 48.2%."

**6 Gazebo.**
- "The planner runs unmodified as a 5 Hz ROS 2 node. poly9 at 0.05 m per unit, 3 m walls; a hexacopter with per-rotor thrust dynamics tracks the roadmap state overhead."
- "Five 180 s flights per strategy: acceleration RMS 1.9–2.0 m/s² against a 2.5 m/s² limit, about six direction reversals per minute."
- "Physical line of sight 62–63%, within ten points of the kinematic planner."
- "Geo-Follow under the same dynamics: 40 reversals per minute and 10 m/s² RMS; naive per-tick replanning drops line of sight to 50%."

**7 Summary.** Three lines: exact placement on a patrol roadmap; critical speed ratio and team partition; validated in simulation and closed-loop flight. Repo URL.

## Build

```bash
video/make_video.sh            # everything: 2D frames (docker), cards, Gazebo take, assemble
```
Intermediate files go to `video/out/` (git-ignored except the final mp4).

### Notes on regenerating

- `render2d.py` runs in the `roadmaptracking` image (needs visilibity, shapely,
  PIL, matplotlib fonts); one container per segment, 1 to 12 minutes each at
  1920×1080 with 2× supersampling. The race segment uses seed 21, chosen with
  `render2d.py race --dry --seed N` (the exact render loop without drawing):
  Min-Max +Vis holds 100% line of sight at a mean worst-corner ratio of 1.07,
  Geo-Follow 70% at 2.23 with one escape. The renderer seeds both `random`
  and `skeleton._rng`; the latter is the module-level generator that picks
  the skeleton evader's destinations and is otherwise seeded from OS entropy,
  which is why trials were not reproducible across processes before (and why
  `benchmark/harness.py`, which seeds only `random`, does not pair the
  evader's waypoint sequence across strategies). The problem segment searches
  seeds for an escape between 4 and 11 s.
- `gz_record.py` needs the desktop: it raises the Gazebo window and records it
  with ffmpeg x11grab, so nothing may cover that window while it records
  (70 s + 40 s). The GUI layout comes from `video/gui.config` through the
  launch file's `gui_config` argument; the camera is driven with
  `/gui/move_to/pose` and `/gui/follow`. If Gazebo hangs at startup (the
  spawners keep printing "Requesting list of world names"), rerun.
- The Gazebo takes use the walking Spot as the evader (`evader_type:=spot`).
  The paper's case-study text describes a differential-drive ground robot;
  either update that sentence or record with `--evader pioneer`.
- `assemble.py` burns captions into the Gazebo cuts with ffmpeg drawtext,
  concatenates all segments and does a two-pass encode to `--target-mb`
  (default 90 MB; RA-L/T-RO multimedia attachments are capped at 100 MB).
