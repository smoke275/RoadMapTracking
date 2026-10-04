"""
Skeleton evader for the Gazebo case study: the Python simulation's actual
SkeletonEvader (benchmark/evaders.py — wanders the polygon's Voronoi
skeleton, avoiding recently visited destinations) runs as a virtual point
in sim units, and the real ground robot tracks it with a heading/speed
controller. The virtual point HOLDS whenever the robot falls more than
`hold_dist` metres behind, so robot and model stay coupled.

Replaces the random-heading placeholder (evader_wanderer.py). Needs the
main repo bind-mounted at `repo_path` (default /app); runs on sim time.
"""
import math
import os
import random
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SkeletonEvaderNode(Node):
    def __init__(self):
        super().__init__('skeleton_evader')
        self.declare_parameter('repo_path', '/app')
        self.declare_parameter('polygon', 'poly9')
        self.declare_parameter('seed', 42)
        self.declare_parameter('evader_speed_mps', 1.0)
        self.declare_parameter('hold_dist', 1.2)
        self.declare_parameter('max_turn_rate', 1.8)      # rad/s
        self.declare_parameter('kp_turn', 2.0)

        repo = self.get_parameter('repo_path').value
        sys.path.insert(0, repo)
        import json
        import ker_pipeline                                # noqa: E402
        from benchmark.harness import load_poly            # noqa: E402
        from benchmark.evaders import SkeletonEvader       # noqa: E402

        with open(os.path.join(repo, 'ros2_sim', 'worlds',
                               'scene_info.json')) as f:
            self._scale = json.load(f)['scale_m_per_unit']

        poly = self.get_parameter('polygon').value
        self.get_logger().info(f'building pipeline for {poly} ...')
        t0 = time.time()
        data = ker_pipeline.build(load_poly(poly), renderer=None)
        random.seed(int(self.get_parameter('seed').value))

        self._robot_m = None
        self._robot_yaw = None
        self._virtual = None       # created on first odom, at the robot's pos
        self._data = data
        self._SkeletonEvader = SkeletonEvader
        self._last_t = None
        self._t0 = None
        self.get_logger().info(f'skeleton ready in {time.time() - t0:.1f}s '
                               f'({len(data.skel_nodes)} nodes)')

        self.create_subscription(Odometry, '/evader/odom', self._on_odom, 10)
        self._cmd_pub = self.create_publisher(Twist, '/evader/cmd_vel', 10)
        self._marker_pub = self.create_publisher(Marker, '/viz/evader_target', 10)
        self.create_timer(1.0 / 20.0, self._tick)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self._robot_m = (p.x, p.y)
        self._robot_yaw = _yaw_from_quaternion(msg.pose.pose.orientation)

    def _now_s(self) -> float:
        t = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None and t > 0:
            self._t0 = t
        return t - (self._t0 or t)

    def _tick(self):
        if self._robot_m is None or self._robot_yaw is None:
            return
        t = self._now_s()
        dt = 0.0 if self._last_t is None else max(0.0, min(t - self._last_t, 0.5))
        self._last_t = t
        s = self._scale

        if self._virtual is None:
            start_u = (self._robot_m[0] / s, self._robot_m[1] / s)
            # speed in units/s; step() is called with dt in seconds
            speed_u = self.get_parameter('evader_speed_mps').value / s
            self._virtual = self._SkeletonEvader(
                self._data.skel_nodes, self._data.skel_adj,
                self._data.shapely_env, start_u, speed_u, avoid_recent=3)
            self.get_logger().info(f'virtual evader initialised at {start_u}')

        vx_m = (self._virtual.pos[0] * s, self._virtual.pos[1] * s)
        lag = math.hypot(self._robot_m[0] - vx_m[0], self._robot_m[1] - vx_m[1])
        if lag <= self.get_parameter('hold_dist').value and dt > 0:
            self._virtual.step(dt)
            vx_m = (self._virtual.pos[0] * s, self._virtual.pos[1] * s)

        # drive the robot toward the virtual point: heading P-control
        dx, dy = vx_m[0] - self._robot_m[0], vx_m[1] - self._robot_m[1]
        dist = math.hypot(dx, dy)
        cmd = Twist()
        if dist > 0.05:
            desired = math.atan2(dy, dx)
            err = (desired - self._robot_yaw + math.pi) % (2 * math.pi) - math.pi
            max_turn = self.get_parameter('max_turn_rate').value
            cmd.angular.z = max(-max_turn, min(max_turn,
                                self.get_parameter('kp_turn').value * err))
            speed = self.get_parameter('evader_speed_mps').value
            cmd.linear.x = speed * max(0.0, math.cos(err)) * min(1.0, dist / 0.5)
        self._cmd_pub.publish(cmd)

        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'skeleton_evader'
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (*vx_m, 0.2)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.9, 0.1, 0.9)
        self._marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = SkeletonEvaderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
