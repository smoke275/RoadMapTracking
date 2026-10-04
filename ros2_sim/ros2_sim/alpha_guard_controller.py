"""
The ACTUAL Min-Max alpha-Guard pipeline from the paper, closed-loop in
Gazebo — replaces the direct-pursuit stand-in (pursuit_controller.py).

Architecture: the paper's planner runs as a *virtual pursuer* in simulation
units — the exact code path of the Python benchmark (ker_pipeline
compute_path_lengths / compute_optimal_guard + pursuer_motion
StableNodeController) — and the drone tracks the virtual pursuer's position
(scaled to metres, at cruise altitude) with the same body-frame velocity law
as the stand-in controller. The loop is genuinely closed: whenever the
drone's tracking error exceeds `hold_dist`, the virtual pursuer HOLDS, so
the algorithm never outruns the physical vehicle, and the evader input is
the real ground robot's odometry.

Strategies (parameter `strategy`):
  alpha-guard    g* of the min-max alpha placement, followed with the
                 stable-node controller (the proposed method).
  alpha-vis      the paper's Min-Max +Vis slack variant (epsilon = 0.1),
                 reusing benchmark.pursuers.MinMaxAlphaVisPursuer.target.
  naive-dijkstra same g*, but the virtual pursuer replans a fresh Dijkstra
                 path every tick and walks it with no stable-node
                 filtering — the chattering failure mode of Section VI.
  geo-follow     stable-node controller toward the roadmap vertex
                 geodesically closest to the evader (baseline).

Per planner tick a CSV row is logged (sim time, evader, drone, virtual
pursuer, g*, worst-corner alpha, 2-D line of sight for both drone and
virtual positions, tracking lag) for the paper's plots.

Needs the main repo bind-mounted at `repo_path` (default /app) with its
pipeline cache present; runs on sim time (set use_sim_time).
"""
import csv
import math
import os
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class AlphaGuardController(Node):
    def __init__(self):
        super().__init__('alpha_guard_controller')
        self.declare_parameter('repo_path', '/app')
        self.declare_parameter('polygon', 'poly9')
        self.declare_parameter('strategy', 'alpha-guard')
        # Default: cruise ABOVE the walls and track the virtual pursuer
        # directly (no collision surface at altitude). Corridor flight
        # below wall height ('corridor') follows the virtual's trail
        # instead; it works but needs careful gain/clearance margins.
        self.declare_parameter('flight_mode', 'above')   # 'above' | 'corridor'
        self.declare_parameter('cruise_altitude', 4.0)
        # Speed/gain sized against the airframe's 2.5 m/s^2 acceleration
        # limit: approach speed must fall below sqrt(a*r) before the carrot
        # pop radius, or the drone settles into a circular limit cycle of
        # radius v^2/a around the target instead of reaching it.
        self.declare_parameter('max_horizontal_speed', 1.5)   # m/s, body frame
        self.declare_parameter('max_vertical_speed', 0.6)
        self.declare_parameter('kp_horizontal', 1.0)
        # damping on the drone's measured velocity: the airframe's inner
        # velocity loop is slow, so velocity-command P-position control is
        # an underdamped oscillator without this term (observed: sustained
        # +/-2 m oscillation straight through a static target)
        self.declare_parameter('kd_horizontal', 0.8)
        self.declare_parameter('kp_vertical', 0.8)
        self.declare_parameter('pursuer_speed_mps', 0.8)      # virtual pursuer
        # hold_dist must sit well above the P-tracker's steady-state error
        # (pursuer_speed/kp ≈ 0.53 m) plus corner transients, or the virtual
        # pursuer is held almost permanently.
        self.declare_parameter('hold_dist', 2.0)              # m: virtual holds
        self.declare_parameter('planner_rate', 5.0)           # Hz, sim time
        self.declare_parameter('log_csv', '')
        self.declare_parameter('run_duration', 0.0)           # sim s; 0 = run forever
        self.declare_parameter('command_frame', 'body')       # 'body' | 'world'

        repo = self.get_parameter('repo_path').value
        sys.path.insert(0, repo)
        # Heavy imports only after the repo is on the path.
        import json
        import ker_pipeline                                    # noqa: E402
        from benchmark.harness import load_poly                # noqa: E402
        from shapely.geometry import LineString, Point         # noqa: E402

        self._LineString = LineString
        self._Point = Point

        info_path = os.path.join(repo, 'ros2_sim', 'worlds', 'scene_info.json')
        with open(info_path) as f:
            self._scale = json.load(f)['scale_m_per_unit']

        poly = self.get_parameter('polygon').value
        self.get_logger().info(f'building KER pipeline for {poly} ...')
        t0 = time.time()
        self._data = ker_pipeline.build(load_poly(poly), renderer=None)
        self._kp = ker_pipeline
        from pursuer_motion import StableNodeController        # noqa: E402
        from graph import dijkstra                             # noqa: E402
        from benchmark.metrics import alphas_at                # noqa: E402
        self._snc = StableNodeController()
        self._dijkstra = dijkstra
        self._alphas_at = alphas_at
        self._visp = None
        if self.get_parameter('strategy').value == 'alpha-vis':
            from benchmark.pursuers import MinMaxAlphaVisPursuer  # noqa: E402
            start = (self._data.vertices[0].x, self._data.vertices[0].y)
            self._visp = MinMaxAlphaVisPursuer(self._data, start, 0.0)
        self.get_logger().info(
            f'pipeline ready in {time.time() - t0:.1f}s '
            f'({len(self._data.corners)} corners, scale {self._scale} m/unit)')

        from collections import deque
        self._trail = deque()           # virtual pursuer's path, metres
        self._last_carrot = None
        self._drone_vel_m = (0.0, 0.0)  # EMA of measured velocity, world
        self._last_odom_t = None
        self._verts = [(v.x, v.y) for v in self._data.vertices]
        self._evader_m = None
        self._drone_m = None
        self._drone_yaw = None
        self._virtual_u = None          # virtual pursuer, sim units
        self._gstar_u = None
        self._last_tick = None
        self._t0 = None
        self._done = False

        strategy = self.get_parameter('strategy').value
        log_csv = self.get_parameter('log_csv').value
        if not log_csv:
            stamp = time.strftime('%Y%m%d_%H%M%S')
            log_csv = os.path.join(repo, 'ros2_sim', 'logs',
                                   f'{stamp}_{strategy}.csv')
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        self._csv_f = open(log_csv, 'w', newline='')
        self._csv = csv.writer(self._csv_f)
        # high-frequency log at the (strategy-independent) control tick:
        # drone state + commanded velocity, for fair accel/jerk comparisons
        self._hf_f = open(log_csv.replace('.csv', '_hf.csv'), 'w', newline='')
        self._hf = csv.writer(self._hf_f)
        self._hf.writerow(['t', 'x', 'y', 'z', 'cmd_vx', 'cmd_vy'])
        self._csv.writerow(['t', 'ev_x', 'ev_y', 'dr_x', 'dr_y', 'dr_z',
                            'virt_x', 'virt_y', 'g_x', 'g_y', 'alpha_max',
                            'los_drone', 'los_virt', 'lag_m', 'held',
                            'trail_n', 'car_x', 'car_y', 'yaw'])
        self.get_logger().info(f'strategy={strategy}  logging to {log_csv}')

        self.create_subscription(Odometry, '/evader/odom', self._on_evader, 10)
        self.create_subscription(Odometry, '/pursuer/odom', self._on_drone, 10)
        self._cmd_pub = self.create_publisher(Twist, '/pursuer/cmd_vel', 10)
        self._enable_pub = self.create_publisher(Bool, '/simple_drone/enable', 10)
        self._marker_pub = self.create_publisher(Marker, '/viz/algorithm_targets', 10)
        self.create_timer(1.0, lambda: self._enable_pub.publish(Bool(data=True)))

        rate = self.get_parameter('planner_rate').value
        self.create_timer(1.0 / rate, self._plan_tick)
        self.create_timer(1.0 / 30.0, self._control_tick)
        self.create_timer(1.0, self._publish_roadmap_marker)

    # ------------------------------------------------------------------
    def _on_evader(self, msg: Odometry):
        p = msg.pose.pose.position
        self._evader_m = (p.x, p.y)

    def _on_drone(self, msg: Odometry):
        p = msg.pose.pose.position
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._drone_m is not None and self._last_odom_t is not None:
            dt = now - self._last_odom_t
            if 1e-4 < dt < 0.5:
                vx = (p.x - self._drone_m[0]) / dt
                vy = (p.y - self._drone_m[1]) / dt
                ax, ay = self._drone_vel_m
                self._drone_vel_m = (0.7 * ax + 0.3 * vx, 0.7 * ay + 0.3 * vy)
        self._last_odom_t = now
        self._drone_m = (p.x, p.y, p.z)
        self._drone_yaw = _yaw_from_quaternion(msg.pose.pose.orientation)

    def _now_s(self) -> float:
        t = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None and t > 0:
            self._t0 = t
        return t - (self._t0 or t)

    # ------------------------------------------------------------------
    def _los_u(self, a_u, b_u) -> bool:
        try:
            return self._data.shapely_env.covers(
                self._LineString([a_u, b_u]))
        except Exception:
            return False

    def _geo_follow_target(self, ev_u):
        ex, ey = ev_u
        cands = sorted(self._verts,
                       key=lambda v: math.hypot(v[0] - ex, v[1] - ey))[:15]

        def geo_d(v):
            try:
                return self._data.geodesic.get_distance((ex, ey), v)
            except Exception:
                return math.hypot(v[0] - ex, v[1] - ey)

        return min(cands, key=geo_d)

    def _plan_tick(self):
        if self._done or self._evader_m is None or self._drone_m is None:
            return
        t = self._now_s()
        dt = 0.0 if self._last_tick is None else max(0.0, min(t - self._last_tick, 0.5))
        self._last_tick = t

        s = self._scale
        ev_u = (self._evader_m[0] / s, self._evader_m[1] / s)
        dr_u = (self._drone_m[0] / s, self._drone_m[1] / s)

        if self._virtual_u is None:
            # start the virtual pursuer at the nearest roadmap node the
            # drone can reach in a straight line (a merely-nearest node
            # could sit across a wall)
            nodes = sorted(self._data.graph.keys(),
                           key=lambda n: math.hypot(n[0] - dr_u[0],
                                                    n[1] - dr_u[1]))
            start = next((n for n in nodes[:50] if self._los_u(dr_u, n)),
                         nodes[0])
            self._snc.reset(start)
            self._virtual_u = tuple(start)
            self.get_logger().info(f'virtual pursuer initialised at {start}')

        pl = self._kp.compute_path_lengths(ev_u[0], ev_u[1], self._data)
        strategy = self.get_parameter('strategy').value
        if strategy == 'geo-follow':
            target = self._geo_follow_target(ev_u)
            alpha = float('nan')
        elif strategy == 'alpha-vis':
            target = self._visp.target(ev_u, pl)
            alpha = float('nan')
        else:
            v1, v2, alpha = self._kp.compute_optimal_guard(pl, self._data)
            from geometry import interpolate_point
            g = interpolate_point(self._data.vertices[v1], self._data.vertices[v2],
                                  self._kp._opt_offset(pl, v1, v2, self._data))
            target = (g.x, g.y)
        self._gstar_u = target

        # Above the walls, lag is the straight-line distance to the
        # virtual pursuer; in corridor mode it is the backlog along the
        # virtual's trail (straight lines could cut through walls).
        if self.get_parameter('flight_mode').value == 'above':
            lag_m = math.hypot(self._drone_m[0] - self._virtual_u[0] * s,
                               self._drone_m[1] - self._virtual_u[1] * s)
        else:
            lag_m = self._trail_backlog_m()
        held = lag_m > self.get_parameter('hold_dist').value
        prev_u = self._virtual_u
        if not held and dt > 0:
            budget = self.get_parameter('pursuer_speed_mps').value / s * dt
            if strategy == 'naive-dijkstra':
                self._virtual_u = self._naive_step(target, budget)
            else:
                self._virtual_u = tuple(
                    self._snc.step(self._data.graph, target, budget))
            if math.hypot(self._virtual_u[0] - prev_u[0],
                          self._virtual_u[1] - prev_u[1]) > 1e-4:
                self._trail.append((self._virtual_u[0] * s,
                                    self._virtual_u[1] * s))

        alphas = self._alphas_at(self._virtual_u, pl, self._data)
        finite = [a for a in alphas.values() if math.isfinite(a)]
        alpha_max = max(finite) if finite else float('nan')

        self._csv.writerow([
            f'{t:.3f}',
            f'{self._evader_m[0]:.3f}', f'{self._evader_m[1]:.3f}',
            f'{self._drone_m[0]:.3f}', f'{self._drone_m[1]:.3f}',
            f'{self._drone_m[2]:.3f}',
            f'{self._virtual_u[0] * s:.3f}', f'{self._virtual_u[1] * s:.3f}',
            f'{target[0] * s:.3f}', f'{target[1] * s:.3f}',
            f'{alpha_max:.4f}',
            int(self._los_u(dr_u, ev_u)), int(self._los_u(self._virtual_u, ev_u)),
            f'{lag_m:.3f}', int(held), len(self._trail),
            f'{self._last_carrot[0]:.3f}' if self._last_carrot else '',
            f'{self._last_carrot[1]:.3f}' if self._last_carrot else '',
            f'{self._drone_yaw:.3f}' if self._drone_yaw is not None else ''])

        dur = self.get_parameter('run_duration').value
        if dur > 0 and t >= dur:
            self._done = True
            self._csv_f.flush()
            self._csv_f.close()
            self._hf_f.flush()
            self._hf_f.close()
            self.get_logger().info(f'RUN COMPLETE at sim t={t:.1f}s')
            self._cmd_pub.publish(Twist())      # stop the drone
            # exit the process so the launch (OnProcessExit -> Shutdown)
            # tears the whole trial down instead of idling until timeout.
            # SystemExit propagates out of rclpy.spin(); rclpy.shutdown()
            # here would deadlock waiting for this very callback.
            raise SystemExit(0)

    def _trail_backlog_m(self) -> float:
        """Distance the drone still has to fly along the virtual pursuer's
        trail: drone -> first trail point -> ... -> virtual position."""
        if self._drone_m is None:
            return 0.0
        pts = list(self._trail)
        if self._virtual_u is not None:
            s = self._scale
            pts.append((self._virtual_u[0] * s, self._virtual_u[1] * s))
        if not pts:
            return 0.0
        total = math.hypot(pts[0][0] - self._drone_m[0],
                           pts[0][1] - self._drone_m[1])
        for a, b in zip(pts, pts[1:]):
            total += math.hypot(b[0] - a[0], b[1] - a[1])
        return total

    def _naive_step(self, target, budget):
        """Per-tick fresh Dijkstra with no stable-node filtering: walk the
        raw waypoint list, including the behind-node artifact — the failure
        mode of Section VI, for the chattering comparison."""
        px, py = self._virtual_u
        _, wp = self._dijkstra(self._data.graph, (px, py), self._Point(*target))
        pos = [px, py]
        rem = budget
        for node in list(wp[1:]) + [target]:
            dx, dy = node[0] - pos[0], node[1] - pos[1]
            d = math.hypot(dx, dy)
            if d < 1e-9:
                continue
            if d <= rem:
                pos = [node[0], node[1]]
                rem -= d
            else:
                pos = [pos[0] + dx / d * rem, pos[1] + dy / d * rem]
                break
        return tuple(pos)

    # ------------------------------------------------------------------
    def _control_tick(self):
        if (self._done or self._virtual_u is None or self._drone_m is None
                or self._drone_yaw is None):
            return
        s = self._scale
        dx, dy, dz = self._drone_m
        yaw = self._drone_yaw
        cruise_z = self.get_parameter('cruise_altitude').value
        max_h = self.get_parameter('max_horizontal_speed').value
        max_v = self.get_parameter('max_vertical_speed').value
        kp_h = self.get_parameter('kp_horizontal').value
        kp_v = self.get_parameter('kp_vertical').value

        if self.get_parameter('flight_mode').value == 'above':
            # above the walls: track the virtual pursuer directly
            tx, ty = self._virtual_u[0] * s, self._virtual_u[1] * s
            self._trail.clear()
        else:
            # corridor mode: drain the trail by progress plus radius pop
            while len(self._trail) >= 2:
                d0 = math.hypot(self._trail[0][0] - dx, self._trail[0][1] - dy)
                d1 = math.hypot(self._trail[1][0] - dx, self._trail[1][1] - dy)
                if d1 <= d0 or d0 < 0.4:
                    self._trail.popleft()
                else:
                    break
            if len(self._trail) == 1 and math.hypot(self._trail[0][0] - dx,
                                                    self._trail[0][1] - dy) < 0.4:
                self._trail.popleft()
            if self._trail:
                tx, ty = self._trail[0]
            else:
                tx, ty = self._virtual_u[0] * s, self._virtual_u[1] * s
        self._last_carrot = (tx, ty)

        # PD toward the carrot: P on position error, D on measured velocity
        # (the damping the sluggish inner velocity loop lacks), capped.
        kd_h = self.get_parameter('kd_horizontal').value
        wx_err, wy_err = tx - dx, ty - dy
        mvx, mvy = self._drone_vel_m
        vx_world = kp_h * wx_err - kd_h * mvx
        vy_world = kp_h * wy_err - kd_h * mvy
        v = math.hypot(vx_world, vy_world)
        if v > max_h:
            vx_world *= max_h / v
            vy_world *= max_h / v
        cmd = Twist()
        if self.get_parameter('command_frame').value == 'world':
            cmd.linear.x, cmd.linear.y = vx_world, vy_world
        else:
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            cmd.linear.x = cos_y * vx_world + sin_y * vy_world
            cmd.linear.y = -sin_y * vx_world + cos_y * vy_world
        cmd.linear.z = max(-max_v, min(max_v, kp_v * (cruise_z - dz)))
        self._cmd_pub.publish(cmd)
        self._hf.writerow([f'{self._now_s():.3f}', f'{dx:.3f}', f'{dy:.3f}',
                           f'{dz:.3f}', f'{vx_world:.3f}', f'{vy_world:.3f}'])
        self._publish_point_marker(2, (tx, ty), (0.1, 0.5, 1.0))   # virtual pursuer
        if self._gstar_u is not None:
            self._publish_point_marker(3, (self._gstar_u[0] * s,
                                           self._gstar_u[1] * s), (1.0, 0.1, 0.8))

    def _publish_point_marker(self, mid, xy, rgb):
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'alpha_guard'
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y = xy
        m.pose.position.z = self.get_parameter('cruise_altitude').value
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 0.9)
        self._marker_pub.publish(m)

    def _publish_roadmap_marker(self):
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'alpha_guard'
        m.id = 1
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = (0.3, 0.9, 0.3, 0.7)
        from geometry_msgs.msg import Point as GPoint
        s = self._scale
        for seg in self._data.path_lines:
            (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
            for x, y in ((x1, y1), (x2, y2)):
                p = GPoint()
                p.x, p.y, p.z = x * s, y * s, 0.15
                m.points.append(p)
        self._marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = AlphaGuardController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
