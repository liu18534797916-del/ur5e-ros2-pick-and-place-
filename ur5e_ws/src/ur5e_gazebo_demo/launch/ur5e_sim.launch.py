import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg_path = get_package_share_directory('ur5e_gazebo_demo')
    robotiq_share = get_package_share_directory('robotiq_description')
    robotiq_parent = os.path.dirname(robotiq_share)

    world = os.path.join(pkg_path, 'worlds', 'camera_world.sdf')
    urdf = os.path.join(pkg_path, 'urdf', 'ur5e.urdf.xacro')

    robot_description = ParameterValue(
        Command([
            'xacro ', urdf,
            ' ur_type:=ur5e',
            ' name:=ur5e',
            ' prefix:=""'
        ]),
        value_type=str
    )

    gz_plugin_path = SetEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH',
        '/opt/ros/jazzy/lib'
    )

    gz_resource_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        [os.environ.get('GZ_SIM_RESOURCE_PATH', ''), ':', robotiq_parent]
    )

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world, '--render-engine', 'ogre2'],
        output='screen'
    )

    # 将 Gazebo 仿真时钟桥接到 ROS 2 /clock topic
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen'
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'ur5e'],
        output='screen'
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    return LaunchDescription([
        gz_plugin_path,
        gz_resource_path,
        gazebo,
        clock_bridge,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner,
    ])
