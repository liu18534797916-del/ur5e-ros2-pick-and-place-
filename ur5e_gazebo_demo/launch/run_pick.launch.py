from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    pick_node = TimerAction(
        period=3.0,
        actions=[Node(
            package='ur5e_gazebo_demo',
            executable='pick_and_place_moveit',
            parameters=[{'use_sim_time': True}],
            output='screen',
        )]
    )

    return LaunchDescription([
        pick_node,
    ])
