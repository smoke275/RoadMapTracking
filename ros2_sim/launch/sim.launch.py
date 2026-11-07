"""
Launch the poly9 case-study world: simple_ground_robot (evader, ground) +
simple_drone (pursuer, aerial) in Gazebo, bridged to ROS 2, visualized in
RViz2, with the pursuit_controller node driving the drone.

Uses a custom simple_ground_robot model rather than TurtleBot3: TurtleBot3's
ROS 2 Humble packages spawn via Gazebo Classic (gazebo_ros/spawn_entity.py,
confirmed by reading turtlebot3_gazebo's own launch file) — a different,
incompatible simulator from the Ignition Fortress world/drone used here.
simple_ground_robot uses the same VelocityControl plugin pattern as
simple_drone so the whole scene stays on one consistent Gazebo generation.

Spawn coordinates are read from worlds/scene_info.json (written by
scripts/generate_world.py) rather than hardcoded, so they always match
whatever world was last generated.

evader_type:=pioneer (default) spawns the Pioneer 3AT (kinematic
VelocityControl, wall-bump contact sensor). evader_type:=spot spawns a
Boston Dynamics Spot with walking legs: the evader node's body-frame
cmd_vel goes to the CHAMP quadruped controller (vendored in spot_stack/,
copied from CodeICRA25/icra25_ros2), which turns it into leg joint
trajectories executed by Gazebo's JointTrajectoryController. The evader
nodes are unchanged — both bodies take the same forward-speed + yaw-rate
Twist on /evader/cmd_vel and report on /evader/odom.

Usage:
    ros2 launch ros2_sim sim.launch.py
    ros2 launch ros2_sim sim.launch.py rviz:=false   # skip RViz2
    ros2 launch ros2_sim sim.launch.py evader_type:=spot
"""
import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent,
                            IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _build(context, *args, **kwargs):
    pkg_share = get_package_share_directory('ros2_sim')
    world_name = 'poly9_world'  # SDF <world name=...> (same in both variants)
    world_file = LaunchConfiguration('world').perform(context)
    world_path = os.path.join(pkg_share, 'worlds', f'{world_file}.sdf')
    models_path = os.path.join(pkg_share, 'models')
    scene_info_path = os.path.join(pkg_share, 'worlds', 'scene_info.json')

    with open(scene_info_path) as f:
        scene = json.load(f)
    ev = scene['evader_spawn_m']
    dr = scene['drone_spawn_m']

    # simple_drone lives in worlds/../models, not the default gz resource
    # path, so extend it (comma-separated on Linux).
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    os.environ['GZ_SIM_RESOURCE_PATH'] = (
        models_path if not existing else f'{models_path}:{existing}')

    evader_type = LaunchConfiguration('evader_type').perform(context)
    if evader_type not in ('pioneer', 'spot'):
        raise RuntimeError(f"evader_type must be 'pioneer' or 'spot', got {evader_type!r}")
    spot = evader_type == 'spot'
    if spot:
        # The spot_evader SDF references meshes as
        # package://spot_description/..., which Gazebo resolves against the
        # share *parent* of that package.
        spot_share = get_package_share_directory('spot_description')
        os.environ['GZ_SIM_RESOURCE_PATH'] += ':' + os.path.dirname(spot_share)
    os.environ['IGN_GAZEBO_RESOURCE_PATH'] = os.environ['GZ_SIM_RESOURCE_PATH']

    headless = LaunchConfiguration('headless').perform(context) == 'true'
    gz_args = f'-r -s {world_path}' if headless else f'-r {world_path}'
    gui_config = LaunchConfiguration('gui_config').perform(context)
    if gui_config and not headless:
        # e.g. /app/video/gui.config: a bare 3D view for video capture
        gz_args += f' --gui-config {gui_config}'
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                        'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    # Real Fuel-downloaded models (see models/x4_uav, models/pioneer3at) —
    # -name overrides whatever entity name is baked into each model.sdf, so
    # every topic name / controller / bridge mapping downstream stays
    # unchanged (simple_drone, simple_ground_robot) regardless of which
    # model file is actually spawned.
    spawn_drone = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=[
            '-name', 'simple_drone',
            '-file', os.path.join(models_path, 'x4_uav', 'model.sdf'),
            '-x', str(dr['x']), '-y', str(dr['y']), '-z', str(dr['z']),
        ],
    )

    if spot:
        # Spot's plugin topics are baked into the SDF as
        # /model/spot_evader/* (see scripts/adapt_spot_model.py); -name must
        # match so the spawned entity owns them. z=0.62 stands it on its legs
        # (CHAMP nominal_height 0.48 plus foot clearance), as in icra25_ros2.
        spawn_ground_robot = Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=[
                '-name', 'spot_evader',
                '-file', os.path.join(models_path, 'spot_evader', 'model.sdf'),
                '-x', str(ev['x']), '-y', str(ev['y']), '-z', '0.62',
            ],
        )
    else:
        spawn_ground_robot = Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=[
                '-name', 'simple_ground_robot',
                '-file', os.path.join(models_path, 'pioneer3at', 'model.sdf'),
                '-x', str(ev['x']), '-y', str(ev['y']), '-z', str(ev['z']),
            ],
        )

    if spot:
        # cmd_vel stays a ROS topic (CHAMP's input); only CHAMP's resulting
        # joint trajectories cross into Gazebo. No contact sensor on Spot.
        evader_bridge = [
            '/model/spot_evader/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            '/model/spot_evader/joint_trajectory@trajectory_msgs/msg/JointTrajectory]ignition.msgs.JointTrajectory',
        ]
        evader_remaps = [('/model/spot_evader/odometry', '/evader/odom')]
    else:
        evader_bridge = [
            '/model/simple_ground_robot/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/model/simple_ground_robot/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            # Wall-bump sensor on the evader's chassis (see the <sensor type="contact">
            # block in models/pioneer3at/model.sdf) — topic name/type confirmed
            # empirically (`ign topic -l` / `ign topic -i`) against the running sim,
            # not assumed from docs.
            f'/world/{world_name}/model/simple_ground_robot/link/chassis/sensor/'
            'chassis_contact/contact@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts',
        ]
        evader_remaps = [
            ('/model/simple_ground_robot/odometry', '/evader/odom'),
            ('/model/simple_ground_robot/cmd_vel', '/evader/cmd_vel'),
            (f'/world/{world_name}/model/simple_ground_robot/link/chassis/sensor/'
             'chassis_contact/contact', '/evader/wall_contact'),
        ]

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            # MulticopterVelocityControl's own topic (robotNamespace +
            # commandSubTopic, set in x4_uav/model.sdf) — NOT
            # /model/simple_drone/cmd_vel, which was only ever valid for
            # the plain VelocityControl plugin this model no longer uses.
            '/simple_drone/gazebo/command/twist@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            # MulticopterVelocityControl ignores twist commands until a
            # Boolean true arrives on <robotNamespace>/<enableSubTopic> —
            # without this the drone never lifts off.
            '/simple_drone/enable@std_msgs/msg/Bool]ignition.msgs.Boolean',
            '/model/simple_drone/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            '/model/simple_drone/pose@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
            *evader_bridge,
        ],
        remappings=[
            ('/simple_drone/gazebo/command/twist', '/pursuer/cmd_vel'),
            ('/model/simple_drone/odometry', '/pursuer/odom'),
            *evader_remaps,
        ],
    )

    champ = None
    if spot:
        # CHAMP: body-frame cmd_vel on /evader/cmd_vel -> leg joint
        # trajectories on /model/spot_evader/joint_trajectory (bridged above).
        # Node + parameters copied from CodeICRA25/icra25_ros2's launch file.
        from launch_ros.parameter_descriptions import ParameterFile
        from launch.substitutions import PathJoinSubstitution
        champ_cfg = get_package_share_directory('champ_config')
        champ = Node(
            package='champ_base', executable='quadruped_controller_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'gazebo': True,
                'publish_joint_states': False,
                'publish_foot_contacts': False,
                'publish_joint_control': True,
                'joint_controller_topic': '/model/spot_evader/joint_trajectory',
                'urdf': os.path.join(spot_share, 'models', 'spot', 'model.urdf'),
            },
                ParameterFile(PathJoinSubstitution([champ_cfg, 'config/links/links.yaml']), allow_substs=True),
                ParameterFile(PathJoinSubstitution([champ_cfg, 'config/joints/joints.yaml']), allow_substs=True),
                ParameterFile(PathJoinSubstitution([champ_cfg, 'config/gait/gait.yaml']), allow_substs=True),
            ],
            remappings=[('cmd_vel/smooth', '/evader/cmd_vel')])

    # Walking Spot is slower and turns more gently than the kinematic
    # Pioneer: CHAMP's gait caps at 1.5 m/s / 1.0 rad/s, and ~0.6 m/s is
    # what it actually walks cleanly (icra25_ros2's default).
    evader_speed = LaunchConfiguration('evader_speed').perform(context)
    evader_speed = float(evader_speed) if evader_speed else (0.6 if spot else 1.0)
    max_turn_rate = 1.0 if spot else 1.8

    rviz = Node(
        package='rviz2', executable='rviz2', output='screen',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'sim.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    mode = LaunchConfiguration('controller').perform(context)
    base = [gz_sim, spawn_drone, spawn_ground_robot, bridge, rviz]
    if champ is not None:
        base.append(champ)
    if mode == 'none':
        return base
    use_alpha = mode == 'alpha'
    if use_alpha:
        controller = Node(
            package='ros2_sim', executable='alpha_guard_controller',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'strategy': LaunchConfiguration('strategy').perform(context),
                'cruise_altitude': float(LaunchConfiguration('cruise_altitude').perform(context)),
                'run_duration': float(LaunchConfiguration('run_duration').perform(context)),
                'log_csv': LaunchConfiguration('log_csv').perform(context),
            }],
        )
        evader = Node(
            package='ros2_sim', executable='skeleton_evader', output='screen',
            parameters=[{
                'use_sim_time': True,
                'seed': int(LaunchConfiguration('evader_seed').perform(context)),
                'evader_speed_mps': evader_speed,
                'max_turn_rate': max_turn_rate,
            }],
        )
    else:
        controller = Node(
            package='ros2_sim', executable='pursuit_controller', output='screen',
            parameters=[{'cruise_altitude': scene['drone_cruise_altitude_m']}],
        )
        evader = Node(
            package='ros2_sim', executable='evader_wanderer', output='screen',
            parameters=[{'seed': int(LaunchConfiguration('evader_seed').perform(context))}],
        )

    # When the controller finishes its run_duration it exits; tear the
    # whole trial down with it so batch runs don't idle until timeout.
    on_done = RegisterEventHandler(OnProcessExit(
        target_action=controller,
        on_exit=[EmitEvent(event=Shutdown(reason='run complete'))]))

    return base + [controller, evader, on_done]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Launch RViz2 alongside the sim'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run Gazebo server-only, no GUI (for testing/CI)'),
        DeclareLaunchArgument('gui_config', default_value='',
                              description='Gazebo GUI config file (empty = '
                                          'Fortress default layout)'),
        DeclareLaunchArgument('evader_type', default_value='pioneer',
                              description="'pioneer' = Pioneer 3AT (kinematic); "
                                          "'spot' = Boston Dynamics Spot walked "
                                          "by CHAMP (spot_stack/)"),
        DeclareLaunchArgument('evader_speed', default_value='',
                              description='Evader speed in m/s (skeleton evader '
                                          'only); empty = 1.0 for pioneer, 0.6 '
                                          'for spot'),
        DeclareLaunchArgument('evader_seed', default_value='42',
                              description='RNG seed for the evader wander pattern (reproducible A/B runs)'),
        DeclareLaunchArgument('controller', default_value='alpha',
                              description="'alpha' = real Min-Max alpha-Guard "
                                          "planner + skeleton evader; 'legacy' "
                                          "= old direct-pursuit stand-in"),
        DeclareLaunchArgument('strategy', default_value='alpha-guard',
                              description='alpha-guard | naive-dijkstra | geo-follow'),
        DeclareLaunchArgument('world', default_value='poly9_world',
                              description='World file (no .sdf): poly9_world '
                                          'or poly9_world_fast (unthrottled)'),
        DeclareLaunchArgument('cruise_altitude', default_value='4.0',
                              description='Drone cruise altitude in metres '
                                          '(below the 3 m walls: real occlusion)'),
        DeclareLaunchArgument('run_duration', default_value='0.0',
                              description='Stop logging after this many sim '
                                          'seconds (0 = run forever)'),
        DeclareLaunchArgument('log_csv', default_value='',
                              description='Metrics CSV path (empty = auto '
                                          'under /app/ros2_sim/logs/)'),
        OpaqueFunction(function=_build),
    ])
