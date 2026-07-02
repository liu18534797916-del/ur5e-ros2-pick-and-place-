#!/usr/bin/env python3
"""
视觉检测节点：从 Gazebo 相机抓图 → 检测盒子顶部的白色标记方块。

相机位姿：(0.72, 0, 2.8) 垂直向下
输出：/aruco_detections (Detection3DArray) → object_pose_estimate

检测策略：
  1. ArUco 检测（标准 4x4 字典）— 标记是白色方块，通常检测不到
  2. Blob 回退 — 多级百分位阈值 + 轮廓提取 → 找 3 个最亮方块
"""

import math
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from vision_msgs.msg import (Detection3D, Detection3DArray,
                              ObjectHypothesisWithPose, BoundingBox3D)
from visualization_msgs.msg import Marker, MarkerArray

os.environ['NPY_COMPAT_OVERRIDE'] = '1'

# ── 相机参数 ────────────────────────────────────────────────────────────
CAM_FX, CAM_FY = 554.3, 554.3
CAM_CX, CAM_CY = 320.0, 240.0
CAMERA_MATRIX = np.array([[CAM_FX, 0, CAM_CX],
                          [0, CAM_FY, CAM_CY],
                          [0, 0, 1]], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)
CAM_POS = np.array([0.72, 0.0, 2.8])

# 旋转矩阵：相机系 → 世界系
# 相机垂直向下：X_cam→世界-Y, Y_cam→世界-X, Z_cam→世界-Z
R_CAM_TO_WORLD = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64)

# ── 标记几何 ────────────────────────────────────────────────────────────
MARKER_SIZE = 0.045       # 标记物理尺寸 (m)
HALF_M = MARKER_SIZE / 2
MARKER_OBJ_PTS = np.array(
    [[-HALF_M, -HALF_M, 0], [HALF_M, -HALF_M, 0],
     [HALF_M, HALF_M, 0], [-HALF_M, HALF_M, 0]], dtype=np.float64)

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()

# ── 深度参数 ────────────────────────────────────────────────────────────
D_MARKER = 1.7735   # 相机到标记平面 (2.8 - 0.925 - 0.1015)
D_OBJECT = 1.875    # 相机到盒子中心 (2.8 - 0.925)
OBJECT_Z = 0.925    # 盒子中心 Z（固定）

DIAG_DIR = Path('/tmp/aruco_diagnostics')
DIAG_SAVE_EVERY = 10
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


def rvec_to_quaternion(rvec):
    """旋转向量 → 四元数 (x, y, z, w)。"""
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    axis = rvec.flatten() / angle
    half = angle / 2.0
    s = math.sin(half)
    return (float(axis[0] * s), float(axis[1] * s),
            float(axis[2] * s), float(math.cos(half)))


class ArucoDetect(Node):

    def __init__(self):
        super().__init__('aruco_detect')
        self.declare_parameter('capture_rate', 10.0)
        self.capture_rate = (
            self.get_parameter('capture_rate').get_parameter_value().double_value)

        self.detector = cv2.aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

        self.det_pub   = self.create_publisher(Detection3DArray, '/aruco_detections', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/aruco_markers', 10)
        self.diag_pub  = self.create_publisher(RosImage, '/aruco_diagnostic', 10)

        self._frame_count = 0

        # 订阅 ROS2 相机桥（优先），不可用时回退到直接 Gazebo 抓图
        self._latest_img = None
        self.sub = self.create_subscription(
            RosImage, '/camera/image_raw', self._img_cb, 10)

        period = 1.0 / max(self.capture_rate, 1.0)
        self.timer = self.create_timer(period, self._capture_and_detect)

        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f'ArUco detector ready (cam=({CAM_POS[0]:.2f},{CAM_POS[1]:.2f},{CAM_POS[2]:.2f}), '
            f'D_object={D_OBJECT:.2f}m, rate={self.capture_rate}Hz)')

    # ── 图像回调 ────────────────────────────────────────────────────────
    def _img_cb(self, msg: RosImage):
        """存储 ROS2 桥发来的最新图像。"""
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            enc = msg.encoding
            if enc in ('rgb8', 'rgba8'):
                self._latest_img = data.reshape((msg.height, msg.width, -1))
            elif enc in ('bgr8', 'bgra8'):
                img = data.reshape((msg.height, msg.width, -1))
                self._latest_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif enc == 'mono8':
                self._latest_img = cv2.cvtColor(
                    data.reshape((msg.height, msg.width)), cv2.COLOR_GRAY2RGB)
            else:
                self._latest_img = data.reshape((msg.height, msg.width, 3))
        except Exception:
            pass

    # ── 主检测循环 ──────────────────────────────────────────────────────
    def _capture_and_detect(self):
        self._frame_count += 1

        # 优先 ROS2 桥图像，回退到直接 Gazebo 抓图
        rgb = self._latest_img
        if rgb is not None:
            rgb = rgb.copy()
        else:
            rgb = self._capture_gz_image()

        if rgb is None:
            if self._frame_count <= 3 or self._frame_count % 30 == 1:
                self.get_logger().warn(f'Frame #{self._frame_count}: no image')
            return

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)

        if self._frame_count <= 5 or self._frame_count % 30 == 1:
            self.get_logger().info(
                f'Frame #{self._frame_count}: ArUco={0 if ids is None else len(ids)}, '
                f'img_mean={gray.mean():.0f} max={gray.max()}')

        stamp = self.get_clock().now().to_msg()
        det_array = Detection3DArray()
        det_array.header.stamp = stamp
        det_array.header.frame_id = 'top_camera_link'
        marker_array = MarkerArray()
        annotated = rgb.copy()

        # ArUco 未命中 → blob 回退
        use_blob = (ids is None or len(ids) == 0)
        blob_detections = self._detect_blobs_color(rgb) if use_blob else []
        use_blob = len(blob_detections) > 0

        if use_blob:
            self._process_blob_detections(
                blob_detections, det_array, marker_array, annotated, stamp)
        elif ids is not None and len(ids) > 0:
            self._process_aruco_detections(
                ids, corners, det_array, marker_array, annotated, stamp)

        self.det_pub.publish(det_array)
        self.marker_pub.publish(marker_array)

        # 保存诊断图像
        if self._frame_count % DIAG_SAVE_EVERY == 0:
            diag = RosImage()
            diag.header.stamp = stamp
            diag.header.frame_id = 'top_camera_link'
            diag.height, diag.width = annotated.shape[:2]
            diag.encoding = 'rgb8'
            diag.step = annotated.shape[1] * 3
            diag.data = annotated.tobytes()
            self.diag_pub.publish(diag)
            cv2.imwrite(str(DIAG_DIR / f'frame_{self._frame_count:05d}.png'),
                       cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # ── ArUco 检测处理 ──────────────────────────────────────────────────
    def _process_aruco_detections(self, ids, corners, det_array, marker_array,
                                   annotated, stamp):
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            cx_c = int(corners[i][0][:, 0].mean())
            cy_c = int(corners[i][0][:, 1].mean())
            wx, wy = self._pixel_to_world(cx_c, cy_c)

            label = f'ID:{marker_id} ({wx:.3f},{wy:.3f})'
            cv2.putText(annotated, label, (cx_c + 15, cy_c),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[marker_id % 3], 1)
            cv2.circle(annotated, (cx_c, cy_c), 4, COLORS[marker_id % 3], -1)

            self._build_detection(det_array, marker_array, marker_id,
                                  wx, wy, stamp, 'aruco_marker', 1.0)

    # ── Blob 检测处理 ───────────────────────────────────────────────────
    def _process_blob_detections(self, blobs, det_array, marker_array,
                                  annotated, stamp):
        for mid, cu, cv_px, box_pts, wx, wy in blobs:
            box_pts = np.array(box_pts, dtype=np.float64)
            if box_pts.ndim == 3:
                box_pts = box_pts.reshape(4, 2)
            center = box_pts.mean(axis=0)
            angles = np.arctan2(box_pts[:, 1] - center[1],
                               box_pts[:, 0] - center[0])
            ordered = box_pts[np.argsort(angles)]
            img_pts = np.ascontiguousarray(ordered, dtype=np.float64)
            obj_pts = np.ascontiguousarray(MARKER_OBJ_PTS, dtype=np.float64)

            success, rvec, _tvec = cv2.solvePnP(
                obj_pts, img_pts, CAMERA_MATRIX, DIST_COEFFS,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not success:
                continue

            cv2.drawContours(annotated, [img_pts.astype(np.int32)], 0,
                            COLORS[mid], 2)
            cv2.drawFrameAxes(annotated, CAMERA_MATRIX, DIST_COEFFS,
                             rvec, _tvec, MARKER_SIZE * 1.5)
            label = f'ID:{mid} blob ({wx:.3f},{wy:.3f})'
            cv2.putText(annotated, label, (int(cu) + 15, int(cv_px)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[mid], 1)

            self._build_detection(det_array, marker_array, mid,
                                  wx, wy, stamp, 'blob_marker', 0.8)

    # ── 通用：构建 Detection3D + Marker ─────────────────────────────────
    def _build_detection(self, det_array, marker_array, marker_id,
                          wx, wy, stamp, class_id, score):
        """像素→世界坐标，填充 Detection3DArray 和 MarkerArray。"""
        # 世界坐标（Z 固定为盒子中心）
        t_world = np.array([wx, wy, OBJECT_Z])
        # 相机系坐标
        dw = t_world - CAM_POS
        t_cam = np.array([-dw[1], -dw[0], -dw[2]], dtype=np.float64)

        det = Detection3D()
        det.header.stamp = stamp
        det.header.frame_id = 'top_camera_link'
        det.id = f'marker_{marker_id}'
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = class_id
        hyp.hypothesis.score = score
        det.results.append(hyp)
        bbox = BoundingBox3D()
        bbox.size.x = float(MARKER_SIZE)
        bbox.size.y = float(MARKER_SIZE)
        bbox.size.z = 0.001
        bbox.center.position.x = float(t_cam[0])
        bbox.center.position.y = float(t_cam[1])
        bbox.center.position.z = float(t_cam[2])
        bbox.center.orientation.w = 1.0
        det.bbox = bbox
        det_array.detections.append(det)

        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = 'world'
        m.ns = 'aruco_marker'
        m.id = int(marker_id)
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(wx)
        m.pose.position.y = float(wy)
        m.pose.position.z = float(OBJECT_Z)
        m.pose.orientation.w = 1.0
        m.scale.x = MARKER_SIZE
        m.scale.y = MARKER_SIZE
        m.scale.z = 0.002
        c = COLORS[marker_id % 3]
        m.color.r = float(c[2]) / 255.0
        m.color.g = float(c[1]) / 255.0
        m.color.b = float(c[0]) / 255.0
        m.color.a = 0.8
        marker_array.markers.append(m)

    # ── Blob 检测 ───────────────────────────────────────────────────────
    def _detect_blobs_color(self, rgb):
        """检测盒子顶部白色方块。

        Ogre2 渲染的"白色"约 180-200（不是 255）。
        用多级百分位阈值（97→94→90→85）找最亮区域。
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blobs = []

        for pct in [97, 94, 90, 85]:
            thresh_val = np.percentile(gray, pct)
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            conts, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in conts:
                a = cv2.contourArea(cnt)
                if a < 30 or a > 500:      # 标记面积 ~100-250 px²
                    continue
                M = cv2.moments(cnt)
                if M['m00'] <= 0:
                    continue
                cu = int(M['m10'] / M['m00'])
                cv_px = int(M['m01'] / M['m00'])
                wx, wy = self._pixel_to_world(cu, cv_px)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                blobs.append((a, cu, cv_px, box, wx, wy))

            if len(blobs) >= 3:
                break

        if self._frame_count <= 5 or len(blobs) == 0:
            self.get_logger().info(
                f'  Blob: {len(blobs)} found '
                f'(mean={gray.mean():.0f} max={gray.max()} '
                f'p97={np.percentile(gray, 97):.0f} '
                f'p90={np.percentile(gray, 90):.0f})')

        # 面积降序取 top-3，世界 Y 排序分配 ID
        blobs.sort(key=lambda b: -b[0])
        blobs = blobs[:3]
        blobs.sort(key=lambda b: b[5])

        return [(i, cu, cv_px, box, wx, wy)
                for i, (area, cu, cv_px, box, wx, wy) in enumerate(blobs)]

    # ── 像素 → 世界坐标 ─────────────────────────────────────────────────
    def _pixel_to_world(self, cx, cy):
        """针孔相机逆映射：像素 (cx,cy) → 世界 XY（用 D_MARKER 深度）。"""
        xn = (cx - CAM_CX) / CAM_FX
        yn = (cy - CAM_CY) / CAM_FY
        wx = float(CAM_POS[0] + yn * D_MARKER)
        wy = float(-xn * D_MARKER)
        return wx, wy

    # ── Gazebo 直接抓图 ─────────────────────────────────────────────────
    def _capture_gz_image(self):
        """从 Gazebo 话题 /top_camera/image 直接抓取一帧。

        gz topic -e 输出 protobuf 文本格式，3.6MB/帧，
        含 data 字段（八进制转义的 uint8 数组）。
        """
        try:
            r = subprocess.run(
                ['gz', 'topic', '-e', '-t', '/top_camera/image', '-n', '1'],
                capture_output=True, timeout=3)
            if r.returncode != 0 or len(r.stdout) < 1000:
                return None

            text = r.stdout.decode('utf-8', errors='replace')
            idx = text.find('data: "')
            if idx < 0:
                return None
            start = idx + len('data: "')
            end = text.find('"\n', start)
            if end < 0:
                end = len(text)
            data_str = text[start:end]

            raw = bytearray()
            i = 0
            while i < len(data_str):
                if (data_str[i:i+2] == '\\\\' and i+5 <= len(data_str)
                        and data_str[i+2:i+5].isdigit()):
                    raw.append(int(data_str[i+2:i+5], 8))
                    i += 5
                elif (data_str[i] == '\\' and i+3 < len(data_str)
                      and data_str[i+1:i+4].isdigit()):
                    raw.append(int(data_str[i+1:i+4], 8))
                    i += 4
                else:
                    raw.append(ord(data_str[i]))
                    i += 1

            expected = 640 * 480 * 3
            if len(raw) < expected:
                return None
            return np.frombuffer(bytes(raw[:expected]), dtype=np.uint8).reshape((480, 640, 3))
        except Exception:
            return None


def main():
    rclpy.init()
    try:
        rclpy.spin(ArucoDetect())
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
