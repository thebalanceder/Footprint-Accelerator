from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    baseline_arg = DeclareLaunchArgument(
        'baseline',
        default_value='0.54',  # Default baseline in meters (KITTI stereo baseline)
        description='Stereo baseline distance in meters'
    )

    # Configure the node with parameters and topic remapping
    soft2_node = Node(
        package='soft2_vo',
        executable='soft2_node',
        name='soft2_vo_node',
        parameters=[{'baseline': LaunchConfiguration('baseline')}],
        remappings=[
            ('/camera/left/image_raw', '/image_0'),  # Remap to KITTI left camera topic
            ('/camera/right/image_raw', '/image_1'),  # Remap to KITTI right camera topic
            ('/odom', '/soft2_vo/odom')  # Optional: Namespace odometry topic
        ]
    )

    # Return the launch description
    return LaunchDescription([
        baseline_arg,
        soft2_node
    ])
