# The composed pipeline for the S4.2 executor/IPC study:
# frames_player (rate mode, clock stamps) -> tbd tracker -> draw, all in ONE
# container process.
#
#   ros2 launch ctrk_ros pipeline.launch.py \
#     container:=component_container_isolated intra_process:=true \
#     images:=$PWD/data/mot/MOT16-04/img1/%06d.jpg \
#     model:=$PWD/models/cache/yolov8n_640.onnx rate_hz:=30.0
#
# The tracker is a lifecycle component — activate it once the container is up:
#   ros2 lifecycle set /ctrk_tbd_node configure
#   ros2 lifecycle set /ctrk_tbd_node activate
# Pipeline latency: ros2 topic delay /image_annotated (draw keeps the stamp).
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument('container', default_value='component_container',
                              description='component_container | _mt | _isolated'),
        DeclareLaunchArgument('intra_process', default_value='true'),
        DeclareLaunchArgument('images',
                              default_value='data/mot/MOT16-04/img1/%06d.jpg'),
        DeclareLaunchArgument('model',
                              default_value='models/cache/yolov8n_640.onnx'),
        DeclareLaunchArgument('rate_hz', default_value='30.0'),
        DeclareLaunchArgument('threads', default_value='4'),
        DeclareLaunchArgument('detect_every', default_value='1'),
        DeclareLaunchArgument('gmc', default_value='off'),
        DeclareLaunchArgument('bench', default_value='true'),
        DeclareLaunchArgument('loop', default_value='true'),
    ]

    intra_process = ParameterValue(LaunchConfiguration('intra_process'), value_type=bool)
    extra = [{'use_intra_process_comms': intra_process}]

    player = ComposableNode(
        package='ctrk_ros', plugin='ctrk_ros::FramesPlayerNode', name='ctrk_frames_player',
        parameters=[{
            'path': LaunchConfiguration('images'),
            'mode': 'rate',
            'rate_hz': ParameterValue(LaunchConfiguration('rate_hz'), value_type=float),
            'stamp_source': 'clock',
            'loop': ParameterValue(LaunchConfiguration('loop'), value_type=bool),
        }],
        extra_arguments=extra)

    tracker = ComposableNode(
        package='ctrk_ros', plugin='ctrk_ros::TbdNode', name='ctrk_tbd_node',
        parameters=[{
            'detector.model_path': LaunchConfiguration('model'),
            'detector.engine.intra_op_threads':
                ParameterValue(LaunchConfiguration('threads'), value_type=int),
            'detect_interval':
                ParameterValue(LaunchConfiguration('detect_every'), value_type=int),
            'gmc': LaunchConfiguration('gmc'),
            'bench': ParameterValue(LaunchConfiguration('bench'), value_type=bool),
        }],
        extra_arguments=extra)

    draw = ComposableNode(
        package='ctrk_ros', plugin='ctrk_ros::DrawNode', name='ctrk_draw_node',
        extra_arguments=extra)

    container = ComposableNodeContainer(
        name='ctrk_pipeline',
        namespace='',
        package='rclcpp_components',
        executable=LaunchConfiguration('container'),
        composable_node_descriptions=[player, tracker, draw],
        output='screen')

    return LaunchDescription(args + [container])
