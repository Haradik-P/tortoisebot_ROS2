from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            name='use_sim_time', default_value='True',
            description='Use simulation clock (set False on the real robot)'),
        Node(
            package='tortoisebot_control',
            executable='obstacle_avoidance',
            name='obstacle_avoidance',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'stop_distance': 0.4,
                'clear_distance': 0.5,
                'forward_speed': 0.12,
                'turn_speed': 0.6,
                'front_angle_deg': 30.0,
                'side_angle_deg': 90.0,
            }],
        ),
    ])
