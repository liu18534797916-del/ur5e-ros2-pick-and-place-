#!/usr/bin/env python3
"""
Object Pose Estimate — intermediate node implementing ObjectPoseEstimate contract.

Subscribes: /aruco_detections (vision_msgs/Detection3DArray in camera frame)
Publishes:
  /object_pose        — geometry_msgs/PoseStamped (latest, world frame)
  /object_pose_{0,1,2} — geometry_msgs/PoseStamped per marker (world frame)
  /object_pre_grasp_{0,1,2} — geometry_msgs/PoseStamped (grasp approach pose)
  TF: world → object_{0,1,2} — dynamic transforms

Transforms camera-frame detections to world frame and applies exponential
moving average filtering for stable poses. Compares against Gazebo ground
truth for validation.
"""

import math
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TransformStamped
from vision_msgs.msg import Detection3DArray
from tf2_ros import TransformBroadcaster


# ── Camera pose in world ──────────────────────────────────────────────
CAM_POS = np.array([0.72, 0.0, 2.8])

# Camera optical → world rotation
R_CAM_TO_WORLD = np.array([
    [0, -1,  0],
    [-1, 0,  0],
    [0,  0, -1]
], dtype=np.float64)

# ── Grasp geometry (from pick_and_place_moveit.py) ────────────────────
FINGER_DEPTH = 0.11
BOX_HALF_X = 0.03
GRASP_Z_OFF = 0.04  # tool0 Z offset above box center
PRE_DIST = 0.12      # pre-grasp standoff distance behind box

# ── Filter ────────────────────────────────────────────────────────────
EMA_ALPHA = 0.7   # exponential moving average (higher = faster response)
STALE_TIMEOUT = 2.0  # seconds before a marker is considered lost

# ── Gazebo ground truth ───────────────────────────────────────────────
ID_TO_BOX = {0: 'box_1', 1: 'box_2', 2: 'box_3'}


def quat_from_rpy(roll, pitch, yaw):
    """rpy → quaternion (x, y, z, w)."""
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


class ObjectPoseEstimate(Node):
    """Transform camera-frame detections → world-frame object poses."""

    def __init__(self):
        super().__init__('object_pose_estimate')

        self.declare_parameter('compare_ground_truth', True)
        self.compare_gt = (
            self.get_parameter('compare_ground_truth').get_parameter_value().bool_value)

        # Subscription
        self.sub = self.create_subscription(
            Detection3DArray, '/aruco_detections', self._det_cb, 10)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        self.pose_pubs = {}
        self.pre_grasp_pubs = {}
        for mid in [0, 1, 2]:
            self.pose_pubs[mid] = self.create_publisher(
                PoseStamped, f'/object_pose_{mid}', 10)
            self.pre_grasp_pubs[mid] = self.create_publisher(
                PoseStamped, f'/object_pre_grasp_{mid}', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # State: per-marker filtered position
        self._filtered = {}     # {marker_id: np.array([x,y,z])}
        self._last_seen = {}    # {marker_id: timestamp(float)}
        self._gt_errors = []
        self._last_gt_query = 0.0

        self.get_logger().info(
            'ObjectPoseEstimate ready '
            f'(ema_alpha={EMA_ALPHA}, gt_compare={self.compare_gt})')

    # ── Detection callback ─────────────────────────────────────────────
    def _det_cb(self, msg: Detection3DArray):
        now = time.time()
        stamp = msg.header.stamp

        seen_ids = set()

        for det in msg.detections:
            if not det.id.startswith('marker_'):
                continue
            try:
                marker_id = int(det.id.split('_')[1])
            except (ValueError, IndexError):
                continue

            if marker_id not in [0, 1, 2]:
                continue

            seen_ids.add(marker_id)

            # Extract camera-frame pose (BoundingBox3D.center is geometry_msgs/Pose)
            cam_pos = det.bbox.center.position
            t_cam = np.array([
                cam_pos.x,
                cam_pos.y,
                cam_pos.z])

            # Transform to world frame
            t_world = R_CAM_TO_WORLD @ t_cam + CAM_POS

            # EMA filter
            if marker_id in self._filtered:
                self._filtered[marker_id] = (
                    EMA_ALPHA * t_world + (1 - EMA_ALPHA) * self._filtered[marker_id])
            else:
                self._filtered[marker_id] = t_world

            self._last_seen[marker_id] = now

        # ── Publish filtered poses ───────────────────────────────────
        latest_pose = None
        for mid in [0, 1, 2]:
            if mid not in self._filtered:
                continue
            # Check staleness
            if now - self._last_seen.get(mid, 0) > STALE_TIMEOUT:
                continue

            pos = self._filtered[mid]
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = 'world'
            pose.pose.position.x = float(pos[0])
            pose.pose.position.y = float(pos[1])
            pose.pose.position.z = float(pos[2])
            # Marker flat on table
            pose.pose.orientation.w = 1.0

            # Per-ID topic
            self.pose_pubs[mid].publish(pose)

            # Pre-grasp pose: offset behind box + above
            pre_grasp = PoseStamped()
            pre_grasp.header.stamp = stamp
            pre_grasp.header.frame_id = 'world'
            # Approach from -X direction (gripper comes from camera side)
            pre_grasp.pose.position.x = float(pos[0]
                - BOX_HALF_X - FINGER_DEPTH - PRE_DIST)
            pre_grasp.pose.position.y = float(pos[1])
            pre_grasp.pose.position.z = float(pos[2] + GRASP_Z_OFF + 0.06)
            pre_grasp.pose.orientation.w = 1.0
            self.pre_grasp_pubs[mid].publish(pre_grasp)

            # Track most recent
            if latest_pose is None or self._last_seen[mid] > self._last_seen.get(
                    latest_pose[0], 0):
                latest_pose = (mid, pose)

            # ── Publish TF ────────────────────────────────────────────
            tfs = TransformStamped()
            tfs.header.stamp = stamp
            tfs.header.frame_id = 'world'
            tfs.child_frame_id = f'object_{mid}'
            tfs.transform.translation.x = float(pos[0])
            tfs.transform.translation.y = float(pos[1])
            tfs.transform.translation.z = float(pos[2])
            tfs.transform.rotation.x = 0.0
            tfs.transform.rotation.y = 0.0
            tfs.transform.rotation.z = 0.0
            tfs.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(tfs)

        # Publish aggregate /object_pose
        if latest_pose is not None:
            self.pose_pub.publish(latest_pose[1])

        # ── Ground truth comparison ───────────────────────────────────
        if self.compare_gt and seen_ids:
            if now - self._last_gt_query > 5.0:
                self._last_gt_query = now
                self._compare_ground_truth(seen_ids)

    # ── Ground truth comparison ───────────────────────────────────────
    def _compare_ground_truth(self, marker_ids):
        errors = []
        for marker_id in marker_ids:
            if marker_id not in ID_TO_BOX or marker_id not in self._filtered:
                continue
            box_name = ID_TO_BOX[marker_id]
            try:
                r = subprocess.run(
                    ['gz', 'model', '-m', box_name, '--pose'],
                    capture_output=True, text=True, timeout=3)
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        nums = [float(x) for x in line[1:-1].split()]
                        if len(nums) >= 3:
                            gt = np.array(nums[:3])
                            est = self._filtered[marker_id]
                            err_xy = float(np.hypot(est[0] - gt[0],
                                                     est[1] - gt[1]))
                            err_z = float(abs(est[2] - gt[2]))
                            errors.append(err_xy)
                            self.get_logger().info(
                                f'  GT object_{marker_id}: '
                                f'est=({est[0]:.4f},{est[1]:.4f},{est[2]:.4f}) '
                                f'gt=({gt[0]:.4f},{gt[1]:.4f},{gt[2]:.4f}) '
                                f'err_xy={err_xy*100:.1f}cm err_z={err_z*100:.1f}cm')
                        break
            except Exception as e:
                self.get_logger().warn(f'GT query failed for {box_name}: {e}')

        if errors:
            self._gt_errors.extend(errors)
            rms = math.sqrt(sum(e * e for e in self._gt_errors)
                            / len(self._gt_errors))
            mae = sum(self._gt_errors) / len(self._gt_errors)
            self.get_logger().info(
                f'  GT stats (world-frame): rms_xy={rms*100:.1f}cm '
                f'mae_xy={mae*100:.1f}cm n={len(self._gt_errors)}')


def main():
    rclpy.init()
    node = ObjectPoseEstimate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
