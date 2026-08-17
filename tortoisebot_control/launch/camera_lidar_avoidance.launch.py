from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    image_topic = LaunchConfiguration('image_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            name='use_sim_time', default_value='True',
            description='Use simulation clock (set False on the real robot)'),
        DeclareLaunchArgument(
            name='image_topic', default_value='/camera/image_raw',
            description='Camera image topic (check: ros2 topic list | grep -i image)'),
        Node(
            package='tortoisebot_control',
            executable='camera_lidar_avoidance',
            name='camera_lidar_avoidance',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'image_topic': image_topic,
                'stop_distance': 0.4,
                'clear_distance': 0.5,
                'forward_speed': 0.12,
                'turn_speed': 0.6,
                'front_angle_deg': 30.0,
                'side_angle_deg': 90.0,
                'edge_density_threshold': 0.12,
                'scan_timeout': 1.0,
            }],
        ),
    ])
