#!/usr/bin/env python3
"""
Static transform publisher: world → top_camera_link.

Publishes the camera pose as a static TF so downstream nodes can use tf2
to transform detections from camera frame to world frame.

Camera physical pose (from camera_world.sdf):
  Position: (0.72, 0.0, 2.8)
  Orientation: pitch = π/2 (1.5707 rad), roll = 0, yaw = 0
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class CameraTfPublisher(Node):

    def __init__(self):
        super().__init__('camera_tf_publisher')

        self.tf_broadcaster = StaticTransformBroadcaster(self)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'top_camera_link'

        # Camera position in world (from camera_world.sdf)
        t.transform.translation.x = 0.72
        t.transform.translation.y = 0.0
        t.transform.translation.z = 2.8

        # Camera orientation: pitch = π/2 (looking straight down at table)
        # rpy = (0, π/2, 0) → quaternion
        import math
        from tf2_ros import TransformBroadcaster
        cy = math.cos(1.5707 / 2)
        sy = math.sin(1.5707 / 2)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = sy  # sin(pitch/2)
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = cy  # cos(pitch/2)

        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(
            f'Published static TF: world → top_camera_link '
            f'(pos=[0.72, 0.0, 2.8], rpy=[0, 1.5707, 0])')

        # Keep alive with a slow timer so the broadcast persists
        self.timer = self.create_timer(5.0, self._republish)

    def _republish(self):
        """Periodic republish (latching for late subscribers)."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'top_camera_link'
        t.transform.translation.x = 0.72
        t.transform.translation.y = 0.0
        t.transform.translation.z = 2.8
        import math
        cy = math.cos(1.5707 / 2)
        sy = math.sin(1.5707 / 2)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = sy
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = cy
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = CameraTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
