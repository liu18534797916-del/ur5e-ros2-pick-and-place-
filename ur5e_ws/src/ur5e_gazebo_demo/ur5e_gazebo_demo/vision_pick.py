#!/usr/bin/env python3

import time
import cv2
import numpy as np

import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class VisionPick(Node):

    def __init__(self):

        super().__init__('vision_pick')

        self.bridge = CvBridge()

        self.image = None

        self.subscription = self.create_subscription(
            Image,
            '/world/camera_world/model/top_camera/link/camera_link/sensor/camera_sensor/image',
            self.image_callback,
            10
        )

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        time.sleep(2)

    def image_callback(self, msg):

        self.image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

    def move_arm(self, positions, sec=3):

        traj = JointTrajectory()

        traj.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        point = JointTrajectoryPoint()

        point.positions = positions
        point.time_from_start.sec = sec

        traj.points.append(point)

        self.arm_pub.publish(traj)

        time.sleep(sec + 1)

    def detect_red_object(self):

        self.get_logger().info('WAITING IMAGE...')

        while self.image is None:
            rclpy.spin_once(self)

        frame = self.image.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 120, 70])
        upper1 = np.array([10, 255, 255])

        lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = mask1 + mask2

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:

            self.get_logger().error('NO RED OBJECT')

            return None

        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)

        cx = x + w // 2
        cy = y + h // 2

        self.get_logger().info(
            f'OBJECT PIXEL: {cx}, {cy}'
        )

        return cx, cy

    def pixel_to_robot(self, cx, cy):

        x = 0.65
        y = -(cx - 320) / 1000.0

        return x, y

    def run(self):

        result = self.detect_red_object()

        if result is None:
            return

        obj_x, obj_y = self.pixel_to_robot(*result)

        self.get_logger().info(
            f'ROBOT TARGET: {obj_x:.2f}, {obj_y:.2f}'
        )

        self.move_arm([
            -0.35,
            -1.75,
            2.0,
            -1.8,
            -1.57,
            0.0
        ])

        time.sleep(2)

        self.move_arm([
            1.2,
            -1.3,
            1.8,
            -2.0,
            -1.57,
            0.0
        ])

        self.get_logger().info('DONE')


def main(args=None):

    rclpy.init(args=args)

    node = VisionPick()

    node.run()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()