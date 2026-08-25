"""
Case-study pursuit controller: flies the drone (pursuer) to hover above the
ground robot (evader), maintaining a fixed cruise altitude above the walls.

Drives simple_drone's MulticopterVelocityControl plugin, which takes
BODY-FRAME linear velocity + yaw rate (confirmed from Fortress's own
worlds/multicopter_velocity_control.sdf header comment: "command linear
velocity ... in the body frame of the vehicle") — real per-rotor thrust
dynamics, not a world-frame velocity teleport. This needs the drone's own
current yaw (from odometry) to rotate the desired world-frame direction
into its body frame; the earlier VelocityControl-based version didn't need
this since it teleported the body directly in world-frame x/y.

This is a direct-pursuit stand-in for the full KER / min-max-alpha
controller from the Python simulation (ker_pipeline.py / pursuer_motion.py)
— porting that pipeline's roadmap/visibility-graph machinery into ROS 2 is
future work, out of scope for this initial case-study scaffold. The hook is
here: both poses are already subscribed, so a real port only needs to
replace _compute_target_velocity below.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PursuitController(Node):
    def __init__(self):
        super().__init__('pursuit_controller')
        self.declare_parameter('cruise_altitude', 2.7)
        self.declare_parameter('max_horizontal_speed', 1.5)   # m/s, body-frame
        self.declare_parameter('max_vertical_speed', 0.5)     # m/s
        self.declare_parameter('kp_horizontal', 0.8)
        self.declare_parameter('kp_vertical', 0.8)

        self._evader_pos = None    # (x, y) from /evader/odom
        self._drone_pos = None     # (x, y, z) from /pursuer/odom
        self._drone_yaw = None     # from /pursuer/odom orientation

        self.create_subscription(Odometry, '/evader/odom', self._on_evader, 10)
        self.create_subscription(Odometry, '/pursuer/odom', self._on_drone, 10)
        self._cmd_pub = self.create_publisher(Twist, '/pursuer/cmd_vel', 10)
        self._marker_pub = self.create_publisher(Marker, '/viz/algorithm_targets', 10)

        self.create_timer(1.0 / 30.0, self._step)
        self.get_logger().info('pursuit_controller started — drone will hover '
                               'above the evader at fixed cruise altitude, '
                               'commanding MulticopterVelocityControl in body frame')

    def _on_evader(self, msg: Odometry):
        p = msg.pose.pose.position
        self._evader_pos = (p.x, p.y)

    def _on_drone(self, msg: Odometry):
        p = msg.pose.pose.position
        self._drone_pos = (p.x, p.y, p.z)
        self._drone_yaw = _yaw_from_quaternion(msg.pose.pose.orientation)

    def _step(self):
        if self._evader_pos is None or self._drone_pos is None or self._drone_yaw is None:
            return
        cmd = self._compute_target_velocity()
        self._cmd_pub.publish(cmd)
        self._publish_target_marker()

    def _publish_target_marker(self):
        """The controller's current target: hover directly above the
        evader's actual position at cruise altitude — this IS the whole
        "algorithm" for the direct-pursuit stand-in, so this marker is
        exactly what _compute_target_velocity is steering toward."""
        ex, ey = self._evader_pos
        cruise_z = self.get_parameter('cruise_altitude').value

        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'pursuer_target'
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = ex
        m.pose.position.y = ey
        m.pose.position.z = cruise_z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.35
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.65, 0.0, 0.9)
        self._marker_pub.publish(m)

    def _compute_target_velocity(self) -> Twist:
        """Direct pursuit: move toward the evader, hold cruise altitude.
        Replace this with the ported KER/min-max-alpha logic for a
        faithful reproduction of the Python simulation's controller.

        MulticopterVelocityControl expects the linear velocity in the
        drone's BODY frame, so the world-frame direction to the evader is
        rotated by -yaw before publishing. angular.z (yaw rate) is left at
        0 — the drone holds its current heading rather than actively
        facing the evader; this is a deliberate simplification, not an
        oversight."""
        ex, ey = self._evader_pos
        dx, dy, dz = self._drone_pos
        yaw = self._drone_yaw
        cruise_z = self.get_parameter('cruise_altitude').value
        max_h = self.get_parameter('max_horizontal_speed').value
        max_v = self.get_parameter('max_vertical_speed').value
        kp_h = self.get_parameter('kp_horizontal').value
        kp_v = self.get_parameter('kp_vertical').value

        wx_err, wy_err = ex - dx, ey - dy
        dist = math.hypot(wx_err, wy_err)
        vx_world, vy_world = 0.0, 0.0
        if dist > 1e-3:
            speed = min(max_h, kp_h * dist)
            vx_world = wx_err / dist * speed
            vy_world = wy_err / dist * speed

        # World -> body frame: rotate by -yaw.
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        vx_body = cos_y * vx_world + sin_y * vy_world
        vy_body = -sin_y * vx_world + cos_y * vy_world

        z_err = cruise_z - dz
        vz = max(-max_v, min(max_v, kp_v * z_err))

        cmd = Twist()
        cmd.linear.x = vx_body
        cmd.linear.y = vy_body
        cmd.linear.z = vz
        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = PursuitController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
