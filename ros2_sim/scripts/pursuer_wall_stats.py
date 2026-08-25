"""
Quantifies how much the pursuer (drone) flies outside the room's actual
footprint — i.e. over/beyond where a wall is — rather than staying above
the walkable interior. The drone cruises above wall height so it never
physically collides when it does this, but the direct-pursuit stand-in
controller (pursuit_controller.py) flies straight toward the evader's raw
XY position with no awareness of the room's (non-convex) shape, so on a
polygon like poly9 it can and does cut across notches/reflex corners that a
ground-constrained agent (or the paper's actual geodesic-routed KER
controller) could not. This script measures that gap directly from live
odometry rather than assuming it.

Run against a live sim (from inside the ros2_sim container, sim already
launched separately):
    python3 ros2_sim/scripts/pursuer_wall_stats.py --duration 60
"""
import argparse
import json
import math
import os
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from shapely.geometry import Point, Polygon


class WallStatsSampler(Node):
    def __init__(self, polygon: Polygon, sample_hz: float):
        super().__init__('pursuer_wall_stats')
        self._polygon = polygon
        self._boundary = polygon.exterior
        self.samples = []  # (t, x, y, outside: bool, depth_m: float)
        self._t0 = None
        self.create_subscription(Odometry, '/pursuer/odom', self._on_odom, 10)
        self.create_timer(1.0 / sample_hz, self._noop)  # keeps the executor spinning at a steady rate

    def _noop(self):
        pass

    def _on_odom(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None:
            self._t0 = now
        p = msg.pose.pose.position
        pt = Point(p.x, p.y)
        outside = not self._polygon.contains(pt)
        depth = self._boundary.distance(pt) if outside else 0.0
        self.samples.append((now - self._t0, p.x, p.y, outside, depth))


def _report(samples, duration_requested: float):
    if not samples:
        print('[pursuer_wall_stats] no /pursuer/odom samples received — is the sim running?')
        return

    n = len(samples)
    outside = [s for s in samples if s[3]]
    frac_outside = len(outside) / n

    # Count contiguous outside runs as separate "wall-crossing events"
    # rather than counting every sample while outside.
    events = 0
    prev_outside = False
    for s in samples:
        if s[3] and not prev_outside:
            events += 1
        prev_outside = s[3]

    depths = [s[4] for s in outside]
    span = samples[-1][0] - samples[0][0]

    print(f'[pursuer_wall_stats] {n} samples over {span:.1f}s (requested {duration_requested:.0f}s)')
    print(f'[pursuer_wall_stats] time spent outside the room footprint: '
         f'{frac_outside * 100:.1f}% of samples ({len(outside)}/{n})')
    print(f'[pursuer_wall_stats] distinct wall-crossing events (contiguous outside runs): {events}')
    if depths:
        print(f'[pursuer_wall_stats] crossing depth when outside — '
             f'mean: {sum(depths) / len(depths):.2f}m, max: {max(depths):.2f}m')
    else:
        print('[pursuer_wall_stats] crossing depth: n/a (never went outside)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=60.0, help='Seconds to sample (default: 60)')
    parser.add_argument('--scene-info', default=None,
                        help='Path to scene_info.json (default: auto-detect from install or worlds/)')
    args = parser.parse_args()

    scene_info_path = args.scene_info
    if scene_info_path is None:
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'worlds', 'scene_info.json'),
            '/ros2_ws/install/ros2_sim/share/ros2_sim/worlds/scene_info.json',
        ]
        scene_info_path = next((c for c in candidates if os.path.exists(c)), None)
    if scene_info_path is None or not os.path.exists(scene_info_path):
        print('[pursuer_wall_stats] could not find scene_info.json — pass --scene-info explicitly',
             file=sys.stderr)
        sys.exit(1)

    with open(scene_info_path) as f:
        scene = json.load(f)
    if 'polygon_vertices_m' not in scene:
        print('[pursuer_wall_stats] scene_info.json has no polygon_vertices_m — '
             're-run generate_world.py to regenerate it', file=sys.stderr)
        sys.exit(1)
    polygon = Polygon(scene['polygon_vertices_m'])

    rclpy.init()
    node = WallStatsSampler(polygon, sample_hz=20.0)
    print(f'[pursuer_wall_stats] sampling /pursuer/odom for {args.duration:.0f}s against '
         f'{scene["polygon"]}\'s {len(scene["polygon_vertices_m"])}-vertex footprint...')

    import time
    t_start = time.time()
    try:
        while time.time() - t_start < args.duration:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        _report(node.samples, args.duration)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
