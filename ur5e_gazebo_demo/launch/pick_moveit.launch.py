import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    pkg_gz   = get_package_share_directory('ur5e_gazebo_demo')
    robotiq_share  = get_package_share_directory('robotiq_description')
    robotiq_parent = os.path.dirname(robotiq_share)

    world = os.path.join(pkg_gz, 'worlds', 'camera_world.sdf')
    urdf  = os.path.join(pkg_gz, 'urdf', 'ur5e.urdf.xacro')

    moveit_config = (
        MoveItConfigsBuilder('ur5e', package_name='ur5e_moveit_config')
        .to_moveit_configs()
    )

    robot_description = ParameterValue(
        Command(['xacro ', urdf, ' ur_type:=ur5e name:=ur5e prefix:="" '
                 'sim_gazebo:=true']),
        value_type=str
    )

    # ── 环境变量 ────────────────────────────────────────────
    numpy_compat = SetEnvironmentVariable('NPY_COMPAT_OVERRIDE', '1')
    gz_plugin  = SetEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH', '/opt/ros/jazzy/lib')
    gz_res     = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        [os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
         ':', robotiq_parent,
         ':', pkg_gz])

    # ── Gazebo ──────────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world, '--render-engine', 'ogre2'],
        output='screen')

    # ── Robot State Publisher ────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen')

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'ur5e'],
        output='screen')

    # ── 控制器 ───────────────────────────────────────────────
    jsb = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen')

    arm_ctrl = Node(
        package='controller_manager', executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        output='screen')

    gripper_ctrl = Node(
        package='controller_manager', executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen')

    # ── MoveIt2 move_group（延迟 5 秒等 Gazebo 就绪）───────────
    move_group = TimerAction(
        period=5.0,
        actions=[Node(
            package='moveit_ros_move_group',
            executable='move_group',
            parameters=[
                moveit_config.to_dict(),
                {'use_sim_time': True},
            ],
            output='screen',
        )]
    )

    # ── Clock bridge ─────────────────────────────────────────
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'],
        output='screen',
    )

    # ── Camera bridge (Gazebo → ROS2) ────────────────────────
    camera_bridge = TimerAction(
        period=8.0,
        actions=[Node(
            package='ros_gz_image',
            executable='image_bridge',
            arguments=[
                '/top_camera/image',
                '/camera/image_raw',
            ],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )]
    )

    # ── Vision pipeline (starts after camera bridge is up) ──────────
    camera_tf = TimerAction(
        period=10.0,
        actions=[Node(
            package='ur5e_gazebo_demo',
            executable='camera_tf_publisher',
            parameters=[{'use_sim_time': True}],
            output='screen',
        )]
    )

    aruco_detect = TimerAction(
        period=11.0,
        actions=[Node(
            package='ur5e_gazebo_demo',
            executable='aruco_detect',
            parameters=[{'use_sim_time': False}],
            output='screen',
        )]
    )

    object_pose_estimate = TimerAction(
        period=12.0,
        actions=[Node(
            package='ur5e_gazebo_demo',
            executable='object_pose_estimate',
            parameters=[{'use_sim_time': True}],
            output='screen',
        )]
    )

    return LaunchDescription([
        numpy_compat,
        gz_plugin,
        gz_res,
        gazebo,
        rsp,
        spawn_robot,
        jsb,
        arm_ctrl,
        gripper_ctrl,
        clock_bridge,
        move_group,
        camera_bridge,
        camera_tf,
        aruco_detect,
        object_pose_estimate,
    ])
