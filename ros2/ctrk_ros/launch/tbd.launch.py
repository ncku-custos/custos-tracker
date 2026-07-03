# Standalone TBD tracker: one lifecycle node, auto configure -> activate.
# ros2 launch ctrk_ros tbd.launch.py params_file:=<yaml>
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
from launch.events import matches_action
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution(
            [FindPackageShare('ctrk_ros'), 'params', 'tbd_default.yaml']),
        description='Parameter YAML for the tracker node')

    node = LifecycleNode(
        package='ctrk_ros',
        executable='ctrk_tbd_node',
        name='ctrk_tbd_node',
        namespace='',
        parameters=[LaunchConfiguration('params_file')])

    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(node),
        transition_id=Transition.TRANSITION_CONFIGURE))
    activate_when_inactive = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=node,
        start_state='configuring',
        goal_state='inactive',
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(node),
            transition_id=Transition.TRANSITION_ACTIVATE))]))

    return LaunchDescription([params_file, node, activate_when_inactive, configure])
