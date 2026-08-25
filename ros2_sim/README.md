# poly9 case study — ROS 2 / Gazebo / RViz2

A ground robot (evader) and a drone (pursuer) in a scaled, dilated
real-world replica of `poly9` from the Python KER pursuit-evasion
simulation. Case-study visualization only — not the source of any results
in the paper; the actual experiments run in the Python simulation
(`ker_pipeline.py` / `benchmark/`).

Everything below has been **actually run and verified** against a real
pulled `osrf/ros:humble-desktop-full` image and a real Gazebo server —
not just checked for syntax. See "Verification status" at the bottom for
exactly what was tested and how.

## What's here

- `scripts/generate_world.py` — generates `worlds/poly9_world.sdf` (walls
  extruded along the polygon boundary) and `worlds/scene_info.json` (scale,
  room size, connectivity check, spawn points) directly from this repo's
  existing polygon-loading/cleaning tools. Run from the **main**
  `roadmaptracking` Docker image (has shapely), not this one — it doesn't
  need ROS/Gazebo:
  ```bash
  docker run --rm -v "$(pwd)":/app -w /app roadmaptracking \
    python ros2_sim/scripts/generate_world.py poly9 \
    --scale 0.03 --clearance 0.35 --wall-height 1.2 --wall-thickness 0.1 \
    --out ros2_sim/worlds/poly9_world.sdf
  ```
- `worlds/poly9_world.sdf` — the generated world (39 walls + ground plane).
  Already built at **0.03 m/unit → 27.0m × 24.0m room**. This scale was
  chosen deliberately: poly9's narrowest passage is 32.09 sim units; at
  0.03 m/unit that's 0.96m against a 0.70m requirement (2×0.35m ground-robot
  clearance) — about 37% safety margin. Smaller scales (≤0.02 m/unit) make
  the walkable interior **disconnect** into separate islands once dilated
  for real robot clearance — verified directly, not assumed.
- `models/x4_uav/` — the **pursuer**: a detailed quadrotor mesh (**6 rotors**
  — it's a hexacopter, not a quadcopter, despite the name — plus LEDs and
  spotlights) downloaded from Gazebo Fuel
  (`OpenRobotics/models/X4 UAV Config 1`). Vendored locally (mesh URIs
  rewritten from Fuel URLs to relative paths, so it doesn't need network
  access at simulation time). Flies on **real per-rotor thrust dynamics** —
  6× `MulticopterMotorModel` (one per rotor, each with its own thrust/
  moment/time constants) plus `MulticopterVelocityControl` (a body-frame
  velocity/attitude controller that computes motor commands via a control
  allocation matrix), not a kinematic teleport. Parameters copied verbatim
  from Fortress's own shipped `multicopter_velocity_control.sdf` example,
  which uses this exact model — not estimated.
- `models/pioneer3at/` — the **evader**: a Pioneer 3AT mesh downloaded from
  Gazebo Fuel (`OpenRobotics/models/Pioneer 3AT`, CC-BY 3.0, Dereck
  Wonnacott). Its original `SkidSteerDrivePlugin` is a **Gazebo Classic**
  plugin and was removed (see `model.config`) — replaced with
  `VelocityControl` (kinematic — no wheel torque, friction, or momentum),
  plus `OdometryPublisher`/`PosePublisher` like `x4_uav`. Deliberately
  **not** given the drone's realistic dynamics: the case study only
  evaluates the pursuer's tracking performance, so the evader doesn't need
  to be physically realistic — it just needs to move. (An earlier version
  of this model used `DiffDrive` with real wheel-torque dynamics; reverted
  once the evader's own physical accuracy turned out not to matter for
  anything being measured.)
- `models/simple_drone/`, `models/simple_ground_robot/` — the original
  hand-built box/cylinder placeholders, kept as a lightweight fallback
  (no longer spawned by default — see `launch/sim.launch.py`).
- `ros2_sim/pursuit_controller.py` — direct-pursuit stand-in controller:
  flies the drone to hover above the evader at a fixed cruise altitude
  above the walls. **Not** the KER/min-max-α controller from the paper —
  porting that (roadmap graph, visibility queries) into ROS 2 is future
  work. The subscription hooks for both robots' poses are already there;
  only `_compute_target_velocity()` needs replacing for a faithful port.
  Publishes its current target (an orange sphere at the evader's actual
  position, cruise altitude) as a `visualization_msgs/Marker` on
  `/viz/algorithm_targets`, so RViz can show "actual vs. intended" for the
  pursuer.
- `ros2_sim/evader_wanderer.py` — simple random-walk placeholder driving
  the evader (nothing else moves it) via `VelocityControl`'s body-frame
  forward-speed + turn-rate interface (kinematic — see the model note
  above). Not a port of the paper's actual Skeleton/Adversarial Evader
  models — same future-work note as the controller above. Subscribes to a
  chassis contact sensor (`/evader/wall_contact`, from the
  `<sensor type="contact">` block in `models/pioneer3at/model.sdf`) and
  reacts to real wall hits: on contact with anything named `wall_*` (the
  sensor is always in contact with `ground_plane` too, which is filtered
  out), it reverses briefly and picks a new heading roughly opposite its
  current one (with jitter, so repeated hits don't keep aiming into the
  same corner), overriding the normal periodic random-heading timer until
  the recovery window ends. Also publishes its current target heading,
  projected a short distance ahead, as a cyan sphere on the same
  `/viz/algorithm_targets` topic. Takes a `seed` parameter (default 42, set
  via the launch file's `evader_seed` argument) so a wander trajectory can
  be replayed exactly — used for the A/B tuning and wall-crossing
  measurements below.
- `scripts/pursuer_wall_stats.py` — measures how much the drone's XY
  position strays outside the room's actual polygon footprint while it's
  cruising above wall height — i.e. cutting across a wall/notch it could
  only get away with because it's flying, not walking. Subscribes to
  `/pursuer/odom` live and checks each sample against
  `scene_info.json`'s `polygon_vertices_m` with shapely. Run against an
  already-launched sim: `python3 ros2_sim/scripts/pursuer_wall_stats.py
  --duration 60`. See "Verification status" for the actual measured
  numbers.
- `launch/sim.launch.py` — brings up Gazebo with the world, spawns both
  robots (via `-name` override, so topic names stay `simple_drone`/
  `simple_ground_robot` regardless of which model file backs them) at the
  coordinates in `scene_info.json`, starts the ROS-Gazebo bridge (now
  including the evader's contact-sensor topic), RViz2, and both controller
  nodes. Takes `headless:=true` (server-only Gazebo, no GUI, for
  testing/CI) and `evader_seed:=<int>` (default 42) arguments.
- `rviz/sim.rviz` — RViz2 config: grid, an Odometry trail for each robot
  (evader in red, pursuer in blue) with `Fixed Frame: world` matching each
  robot's `odom_frame` (set explicitly in both model SDFs — see
  "Verification status" below for why), and a Marker display for
  `/viz/algorithm_targets`. The earlier config's `RobotModel` display and
  `TF` display were removed — nothing in this package publishes
  `/robot_description` (the models are plain SDF, not URDF) or bridges
  `/tf`, so both displays only rendered empty/broken panels.

## Build and run

This Dockerfile is separate from the main repo's — it's a full ROS 2
Humble desktop image (`osrf/ros:humble-desktop-full`, confirmed to bundle
**Ignition Gazebo 6 "Fortress"**, RViz2, and `ros_gz` already) plus
`shapely` for `generate_world.py`. Different and much heavier toolchain
than the Python simulation's image.

```bash
cd ros2_sim
docker build -t ros2_sim .
xhost +local:docker
docker run --rm -it \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --net=host \
  ros2_sim
# inside the container:
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 launch ros2_sim sim.launch.py
```

Headless sanity-check (no GUI, confirms the whole stack is wired up):
```bash
ros2 launch ros2_sim sim.launch.py rviz:=false headless:=true
```

Adjust the X11/GPU flags to whatever's worked for you before with Gazebo in
Docker on this machine — Gazebo's rendering load is much heavier than the
Python sim's pygame/PyQt5 windows, so you may want GPU passthrough
(`--gpus all` + an NVIDIA-enabled image) depending on your setup.

## Regenerating for a different polygon or scale

```bash
docker run --rm -v "$(pwd)":/app -w /app roadmaptracking \
  python ros2_sim/scripts/generate_world.py poly2 --scale 0.04 \
  --out ros2_sim/worlds/poly2_world.sdf
```

The script always prints (and saves to `scene_info.json`) the narrowest
passage width and whether the requested clearance actually fits — check
that output before assuming a new polygon/scale combination is navigable.

## Verification status

This has been checked well beyond structural syntax — pulled the real base
image and ran the real simulator:

- **Confirmed the Gazebo generation directly**: `osrf/ros:humble-desktop-full`
  ships Ignition Gazebo 6 "Fortress" (binary `ign`, not `gz`). Every plugin
  filename/class-name/message-type-string used here (`ignition-gazebo-*-system`
  filenames, mixed `gz::sim::systems::*` / `ignition::gazebo::systems::*`
  class namespaces, `ignition.msgs.*` bridge type strings) was read off this
  version's own shipped example worlds and compiled binaries, not assumed
  from generic Gazebo docs — which would have been wrong, since Fortress
  shipped mid-rename from "Ignition" to "Gazebo" and its own examples are
  internally inconsistent about which namespace a given plugin uses.
- **Found and fixed a real architecture bug**: TurtleBot3's ROS 2 Humble
  packages spawn via Gazebo *Classic* (confirmed by reading
  `turtlebot3_gazebo`'s own launch file — `package='gazebo_ros'`), which is
  a different, incompatible simulator from Fortress. Originally scaffolded
  with TurtleBot3; replaced with a custom model, then with the vendored
  Pioneer 3AT (whose own original plugin was *also* Classic-only, and was
  likewise stripped and replaced).
- **Found and fixed a real packaging bug**: `ament_python` packages need a
  `setup.cfg` redirecting `install-scripts` to `install/<pkg>/lib/<pkg>/` —
  without it (this package's original state, hand-written without
  `ros2 pkg create`), `ros2 launch`'s `Node` action fails immediately with
  `libexec directory ... does not exist`. Caught by actually running the
  launch file, not by inspection.
- **Full end-to-end launch actually run** (`ros2 launch ros2_sim
  sim.launch.py rviz:=false headless:=true`, in a detached container):
  Gazebo server starts, both real Fuel models spawn cleanly, all 5
  ROS-Gazebo bridge topic mappings create with zero errors, both Python
  controller nodes start. **Live odometry confirms actual behavior**: the
  evader (Pioneer 3AT) moves under its wander controller, and the pursuer
  (X4 UAV) tracks the evader's live X/Y position while holding cruise
  altitude — the real pursuit loop, not just clean logs. Checked the full
  log output for `warn|error|missing|fail` — nothing, including for the
  vendored mesh files (confirms the Fuel-URL-to-relative-path mesh URI
  rewrite actually resolves correctly).
- **Upgraded from kinematic to real dynamics, re-verified the same way**:
  both robots originally used `VelocityControl` (teleports the body at a
  commanded velocity, ignoring mass/inertia/friction entirely). Replaced
  with `DiffDrive` (evader — real wheel torque through Pioneer 3AT's own 4
  wheel joints + ground friction) and 6×`MulticopterMotorModel` +
  `MulticopterVelocityControl` (pursuer — real per-rotor thrust, using
  X4 UAV's own 6 rotor joints). Confirmed `MulticopterVelocityControl`
  expects **body-frame** velocity commands from Fortress's own header
  comment on `multicopter_velocity_control.sdf` — `pursuit_controller.py`
  now extracts the drone's yaw from odometry and rotates the world-frame
  pursuit direction into body frame before publishing; `evader_wanderer.py`
  now commands `DiffDrive`'s actual interface (forward speed + turn rate
  toward a target heading) instead of world-frame x/y. **Since reverted for
  the evader specifically** (see the next two bullets) — the pursuer keeps
  its real dynamics, since that's the one thing this case study actually
  evaluates. Re-ran the full
  headless launch after this change: all dynamics plugins load with zero
  errors, and live odometry over multiple samples shows real accelerating
  motion (e.g. the evader traveling several meters under wheel-driven
  motion, the pursuer's altitude holding within 3cm of the 2.7m target
  while translating to follow) rather than instantaneous teleporting.
- **Found and fixed the RViz2 rendering being broken, and added wall-hit
  detection**: the Odometry displays had `Fixed Frame: world` while
  `OdometryPublisher` defaults to `<model_name>/odom` (confirmed against
  Fortress's own `worlds/vehicle.sdf`) — with no `/tf` bridged, RViz had no
  way to resolve that frame mismatch, so the arrows/trails silently never
  drew. Fixed by setting `<odom_frame>world</odom_frame>` explicitly in
  both model SDFs (tf2 treats same-named frames as an identity transform
  with no broadcaster needed). Also removed the RViz config's `RobotModel`
  and `TF` displays, which were rendering nothing (see above). Separately,
  added a `<sensor type="contact">` to the evader's chassis link plus the
  world-level `Contact` system plugin (`generate_world.py` now emits it) —
  empirically confirmed the resulting topic name/type via `ign topic -l`/
  `-i` on a live run rather than assuming it (`ignition.msgs.Contacts` on
  `/world/poly9_world/model/simple_ground_robot/link/chassis/sensor/
  chassis_contact/contact`), bridged it to `/evader/wall_contact`, and
  confirmed by direct observation that the chassis is *always* touching
  `ground_plane` (it's resting on it) — so the turn-around logic filters
  for `wall_*` contacts specifically, not "any contact." Re-ran the full
  headless launch: zero errors, `/viz/algorithm_targets` publishing both
  target markers, and over a 40s sample the evader hit a wall and
  correctly reversed/turned around 4 times while otherwise wandering
  normally (traveled ~8m in the 15s between two contact events, i.e. not
  stuck oscillating against a wall).
- **Tuned the drone's acceleration limit, with a measured before/after**:
  the direct-pursuit controller was overshooting past the evader
  repeatedly rather than converging — `MulticopterVelocityControl`'s
  `maximumLinearAcceleration`, copied verbatim from the shipped example,
  capped horizontal acceleration at 1 m/s², too little to decelerate/
  redirect when the evader turned, so the drone kept flying past it. Fixed
  the evader's own randomness confounding the comparison first (added the
  `seed` parameter above), then A/B measured mean pursuer-evader separation
  over 20s on two different seeds: 1 m/s² gave 3.67m/4.73m, 2.5 m/s² gave
  3.17m/2.75m (chosen), 4 m/s² gave 3.88m (worse — too much authority
  overshoots harder with the same gain). A separate attempt to also raise
  `velocityGain`/`attitudeGain` measured worse still (underdamped the
  attitude loop) and was reverted; only the acceleration limit changed.
- **Measured how much the drone flies outside the room's actual
  footprint** (`scripts/pursuer_wall_stats.py`, live `/pursuer/odom`
  sampling against the polygon in `scene_info.json`): across 4 independent
  runs on different evader seeds (42, 7, 99, 123 — 40-60s each, ~180s /
  ~9000 samples total), **0.0% of samples were outside the room
  footprint** — the drone never flew over a wall. This isn't because the
  room is easy: a separate static check (300 random pairs of interior
  points, straight lines between them) found **63.9% of arbitrary
  transits through this polygon exit its footprint**, by up to 2.85m —
  poly9 is genuinely non-convex with several sharp notches. The reason the
  live number is 0% is that the direct-pursuit controller keeps the drone
  tightly tracking the evader's continuously-updating position (mean
  separation ~2-3m per the tuning above) rather than ever needing to
  bridge a large jump — its actual moves are short and local, and short
  local moves rarely cross a notch even in a polygon this jagged. Caveat
  for the paper: this measures the *current* tuned tracking behavior over
  ~180s of typical wander, not a proof that crossings can't happen — a
  longer run, a harder-to-track evader, or degraded tracking (e.g. control
  lag, a dropped command) could still produce one, since the geometry
  underneath is clearly capable of it.
- **Not verified** (no display in this environment): RViz2 actually
  rendering pixels on screen — the frame-mismatch bug above is fixed and
  every topic it depends on is confirmed publishing correctly, but nothing
  in this sandbox can open a window to look at the result.
