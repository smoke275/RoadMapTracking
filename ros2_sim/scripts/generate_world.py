"""
Generate a Gazebo SDF world from a saved polygon (poly1-9, or a CSV path),
scaled to real-world meters, with walls extruded along the polygon boundary.

Reuses this repo's existing polygon-loading/cleaning tooling directly rather
than re-parsing CSVs — the world is guaranteed to match whatever the Python
simulation actually used.

Also prints (and saves to scene_info.json) a connectivity check: whether a
ground robot with the given clearance can actually traverse the space at
this scale, and where it's safe to spawn the evader/pursuer.

Run (from repo root, so the existing benchmark/geometry modules resolve):
    python ros2_sim/scripts/generate_world.py poly9 \
        --scale 0.03 --clearance 0.35 --wall-height 1.2 --wall-thickness 0.1 \
        --out ros2_sim/worlds/poly9_world.sdf
"""
import argparse
import json
import math
import os
import sys

# Repo root must be on sys.path to reach benchmark/geometry — this script
# lives at ros2_sim/scripts/, two levels below root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

from benchmark.harness import load_poly
from geometry import clean_polygon


def _narrowest_passage(shp: ShapelyPolygon) -> float:
    """Binary search for the largest inward buffer that keeps the interior
    a single connected Polygon — i.e. half the narrowest passage width."""
    lo, hi = 0.0, min(shp.bounds[2] - shp.bounds[0], shp.bounds[3] - shp.bounds[1])
    for _ in range(30):
        mid = (lo + hi) / 2
        d = shp.buffer(-mid)
        if d.geom_type == 'Polygon' and not d.is_empty:
            lo = mid
        else:
            hi = mid
    return lo


def _wall_model_sdf(name, x1, y1, x2, y2, height, thickness):
    length = math.hypot(x2 - x1, y2 - y1)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    yaw = math.atan2(y2 - y1, x2 - x1)
    return f"""    <model name='{name}'>
      <static>true</static>
      <pose>{mx:.4f} {my:.4f} {height / 2:.4f} 0 0 {yaw:.6f}</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{length:.4f} {thickness:.4f} {height:.4f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{length:.4f} {thickness:.4f} {height:.4f}</size></box></geometry>
          <material>
            <ambient>0.7 0.7 0.72 1</ambient>
            <diffuse>0.75 0.75 0.77 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""


def generate(polygon_name: str, scale: float, clearance: float,
            wall_height: float, wall_thickness: float, out_path: str):
    poly = clean_polygon(load_poly(polygon_name))
    pts_units = [(p.x(), p.y()) for p in poly]
    shp_units = ShapelyPolygon(pts_units)

    max_clearance_units = _narrowest_passage(shp_units)
    max_clearance_m = max_clearance_units * scale
    passage_m = 2 * max_clearance_units * scale
    connected = clearance <= max_clearance_units * scale

    pts_m = [(x * scale, y * scale) for x, y in pts_units]
    shp_m = ShapelyPolygon(pts_m)
    minx, miny, maxx, maxy = shp_m.bounds

    # Spawn points: evader at a point comfortably inside the dilated
    # (clearance-eroded) region so it starts with real clearance from every
    # wall; pursuer (drone) spawns above the same point at cruising altitude.
    dilated_m = shp_m.buffer(-clearance)
    if dilated_m.is_empty:
        spawn = shp_m.centroid
    else:
        rep = dilated_m.representative_point()
        spawn = rep
    evader_spawn = (spawn.x, spawn.y, 0.05)
    drone_cruise_alt = wall_height + 1.5
    drone_spawn = (spawn.x, spawn.y, drone_cruise_alt)

    walls_xml = []
    n = len(pts_m)
    for i in range(n):
        x1, y1 = pts_m[i]
        x2, y2 = pts_m[(i + 1) % n]
        if math.hypot(x2 - x1, y2 - y1) < 1e-6:
            continue
        walls_xml.append(_wall_model_sdf(f'wall_{i}', x1, y1, x2, y2,
                                         wall_height, wall_thickness))

    sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <world name='{polygon_name}_world'>
    <physics name='default_physics' type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <!-- Plugin filenames verified against the actual installed Gazebo
         (Ignition Fortress / Gazebo 6, shipped in osrf/ros:humble-desktop-full)
         by inspecting its own shipped example worlds — this version predates
         the gz-sim-* filename convention used by later Gazebo releases. -->
    <plugin filename='ignition-gazebo-physics-system' name='gz::sim::systems::Physics'/>
    <plugin filename='ignition-gazebo-scene-broadcaster-system' name='gz::sim::systems::SceneBroadcaster'/>
    <plugin filename='ignition-gazebo-user-commands-system' name='gz::sim::systems::UserCommands'/>
    <!-- World-level Contact system required for any model's <sensor type="contact">
         to actually produce data (confirmed against Fortress's own shipped
         worlds/contact_sensor.sdf) — needed for the evader's wall-bump sensor. -->
    <plugin filename='ignition-gazebo-contact-system' name='gz::sim::systems::Contact'/>

    <light type='directional' name='sun'>
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name='ground_plane'>
      <static>true</static>
      <link name='link'>
        <collision name='collision'>
          <geometry><plane><normal>0 0 1</normal><size>{(maxx - minx) + 10:.2f} {(maxy - miny) + 10:.2f}</size></plane></geometry>
        </collision>
        <visual name='visual'>
          <geometry><plane><normal>0 0 1</normal><size>{(maxx - minx) + 10:.2f} {(maxy - miny) + 10:.2f}</size></plane></geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Walls extruded from {polygon_name} ({n} vertices), scale={scale} m/unit -->
{''.join(walls_xml)}
  </world>
</sdf>
"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(sdf)

    scene_info = {
        'polygon': polygon_name,
        'scale_m_per_unit': scale,
        'wall_height_m': wall_height,
        'wall_thickness_m': wall_thickness,
        'room_bounds_m': {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy},
        'room_size_m': {'width': maxx - minx, 'height': maxy - miny},
        'clearance_requested_m': clearance,
        'narrowest_passage_m': passage_m,
        'max_safe_clearance_m': max_clearance_m,
        'clearance_feasible': connected,
        'evader_spawn_m': {'x': evader_spawn[0], 'y': evader_spawn[1], 'z': evader_spawn[2]},
        'drone_spawn_m': {'x': drone_spawn[0], 'y': drone_spawn[1], 'z': drone_spawn[2]},
        'drone_cruise_altitude_m': drone_cruise_alt,
        # Room footprint in meters, same points the walls were extruded
        # from — lets a stats script check whether the drone's XY position
        # stays within the walkable interior (i.e. isn't flying out over
        # where a wall/exterior is, just because it's above wall height).
        'polygon_vertices_m': pts_m,
    }
    info_path = os.path.join(os.path.dirname(out_path), 'scene_info.json')
    with open(info_path, 'w') as f:
        json.dump(scene_info, f, indent=2)

    print(f'[generate_world] {n} walls written to {out_path}')
    print(f'[generate_world] room size: {maxx - minx:.1f}m x {maxy - miny:.1f}m')
    print(f'[generate_world] narrowest passage: {passage_m:.2f}m '
         f'(max safe single-side clearance: {max_clearance_m:.3f}m)')
    if not connected:
        print(f'[generate_world] WARNING: requested clearance {clearance}m exceeds '
             f'max safe clearance {max_clearance_m:.3f}m at this scale — the '
             f'walkable space will be DISCONNECTED for a robot this size. '
             f'Increase --scale or decrease --clearance.')
    else:
        margin = max_clearance_m - clearance
        print(f'[generate_world] OK: {clearance}m clearance fits with '
             f'{margin:.3f}m margin to spare')
    print(f'[generate_world] evader spawn: ({evader_spawn[0]:.2f}, {evader_spawn[1]:.2f})')
    print(f'[generate_world] drone spawn: ({drone_spawn[0]:.2f}, {drone_spawn[1]:.2f}, '
         f'{drone_spawn[2]:.2f})')
    print(f'[generate_world] scene info written to {info_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate a Gazebo SDF world from a saved polygon')
    parser.add_argument('polygon', nargs='?', default='poly9',
                        help="Polygon name (polyN) or CSV path (default: poly9)")
    parser.add_argument('--scale', type=float, default=0.03,
                        help='Meters per simulation unit (default: 0.03)')
    parser.add_argument('--clearance', type=float, default=0.35,
                        help='Ground-robot clearance from walls, in meters (default: 0.35)')
    parser.add_argument('--wall-height', type=float, default=1.2,
                        help='Wall height in meters (default: 1.2 — the drone cruises above this)')
    parser.add_argument('--wall-thickness', type=float, default=0.1,
                        help='Wall thickness in meters (default: 0.1)')
    parser.add_argument('--out', default='ros2_sim/worlds/poly9_world.sdf',
                        help='Output SDF file path')
    args = parser.parse_args()
    generate(args.polygon, args.scale, args.clearance, args.wall_height,
             args.wall_thickness, args.out)


if __name__ == '__main__':
    main()
