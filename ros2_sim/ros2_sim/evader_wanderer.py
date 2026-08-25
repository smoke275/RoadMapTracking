"""
Case-study evader motion: simple randomized wandering (pick a random
heading, drive that way for a few seconds, repeat), so the ground robot
visibly moves for the pursuer to track. Backs off and turns around when its
chassis contact sensor detects it has hit a wall.

Drives simple_ground_robot's VelocityControl plugin — kinematic, no wheel
torque/friction/momentum. This is deliberate: the evader's own motion isn't
what the case study evaluates (only the pursuer's tracking performance is),
so it doesn't need — and shouldn't be slowed down by — realistic wheel
dynamics. VelocityControl still takes forward speed (linear.x) + turn rate
(angular.z) in the model's BODY frame (confirmed empirically against
Fortress's own shipped velocity_control.sdf demo — see the comment on the
plugin block in model.sdf), so this needs the robot's own current heading
(from odometry) to compute a turn-rate command toward the target heading,
same as it would for a real DiffDrive-driven robot.

This is a placeholder, not a port of the Python simulation's actual
Skeleton/Adversarial Evader models (skeleton.py / benchmark/evaders.py),
which need the polygon's Voronoi skeleton and geodesic graph — porting
those into ROS 2 is future work, same scope note as
pursuit_controller.py's direct-pursuit placeholder. Correspondingly, the
"target" marker this node publishes is only the wander placeholder's
current target heading projected a fixed distance ahead — not a real
skeleton destination.
"""
import math
import random

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts
from rclpy.node import Node
from visualization_msgs.msg import Marker


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class EvaderWanderer(Node):
    def __init__(self):
        super().__init__('evader_wanderer')
        self.declare_parameter('forward_speed', 0.4)    # m/s
        self.declare_parameter('reverse_speed', 0.25)    # m/s, while recovering from a wall hit
        self.declare_parameter('max_turn_rate', 0.8)     # rad/s
        self.declare_parameter('heading_period', 4.0)    # seconds between new headings
        self.declare_parameter('turn_gain', 1.5)         # proportional heading control
        self.declare_parameter('recovery_duration', 1.5)  # seconds to back off + turn after a wall hit
        self.declare_parameter('target_lookahead', 1.5)  # m, for the "intended heading" marker
        self.declare_parameter('seed', 42)  # fixes the wander trajectory for reproducible A/B tuning runs

        seed = self.get_parameter('seed').value
        if seed is not None:
            random.seed(seed)

        self._pos = None                                  # (x, y) from /evader/odom
        self._yaw = None
        self._target_heading = random.uniform(-math.pi, math.pi)
        self._recovering_until = None                      # ROS time (s), or None

        self.create_subscription(Odometry, '/evader/odom', self._on_odom, 10)
        self.create_subscription(Contacts, '/evader/wall_contact', self._on_contact, 10)
        self._pub = self.create_publisher(Twist, '/evader/cmd_vel', 10)
        self._marker_pub = self.create_publisher(Marker, '/viz/algorithm_targets', 10)

        period = self.get_parameter('heading_period').value
        self.create_timer(period, self._new_heading)
        self.create_timer(1.0 / 20.0, self._step)
        self.get_logger().info('evader_wanderer started — random-walk placeholder driving '
                               'DiffDrive (forward speed + turn rate), backs off and turns '
                               'around on wall contact')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self._pos = (p.x, p.y)
        self._yaw = _yaw_from_quaternion(msg.pose.pose.orientation)

    def _on_contact(self, msg: Contacts):
        now = self._now()
        if self._recovering_until is not None and now < self._recovering_until:
            return  # already backing off from a hit, ignore further contacts
        if self._yaw is None:
            return
        hit_wall = any(
            c.collision1.name.startswith('wall_') or c.collision2.name.startswith('wall_')
            for c in msg.contacts
        )
        if not hit_wall:
            return  # only the ground-plane contact that's always present
        # Turn roughly around, away from the wall, with jitter so repeated
        # hits don't keep aiming back into the same corner.
        new_heading = self._yaw + math.pi + random.uniform(-0.6, 0.6)
        self._target_heading = math.atan2(math.sin(new_heading), math.cos(new_heading))
        self._recovering_until = now + self.get_parameter('recovery_duration').value
        self.get_logger().info('evader_wanderer: wall contact detected, backing off and turning around')

    def _new_heading(self):
        now = self._now()
        if self._recovering_until is not None and now < self._recovering_until:
            return  # don't override the escape heading mid-recovery
        self._target_heading = random.uniform(-math.pi, math.pi)

    def _step(self):
        if self._yaw is None or self._pos is None:
            return  # no odometry yet
        speed = self.get_parameter('forward_speed').value
        reverse_speed = self.get_parameter('reverse_speed').value
        max_turn = self.get_parameter('max_turn_rate').value
        kp = self.get_parameter('turn_gain').value

        # Shortest-path heading error, wrapped to [-pi, pi].
        err = math.atan2(math.sin(self._target_heading - self._yaw),
                         math.cos(self._target_heading - self._yaw))
        turn_rate = max(-max_turn, min(max_turn, kp * err))

        now = self._now()
        recovering = self._recovering_until is not None and now < self._recovering_until
        if self._recovering_until is not None and not recovering:
            self._recovering_until = None

        cmd = Twist()
        cmd.linear.x = -reverse_speed if recovering else speed
        cmd.angular.z = turn_rate
        self._pub.publish(cmd)
        self._publish_target_marker()

    def _publish_target_marker(self):
        lookahead = self.get_parameter('target_lookahead').value
        x, y = self._pos
        tx = x + lookahead * math.cos(self._target_heading)
        ty = y + lookahead * math.sin(self._target_heading)

        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'evader_target'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = tx
        m.pose.position.y = ty
        m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 0.9, 0.9, 0.9)
        self._marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = EvaderWanderer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
