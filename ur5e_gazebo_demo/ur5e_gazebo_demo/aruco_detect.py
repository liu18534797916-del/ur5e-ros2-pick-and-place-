#!/usr/bin/env python3
"""
ArUco marker detection node — OpenCV 4.x API.

Captures images directly from Gazebo via gz topic -e (no ROS2 bridge needed).
Publishes:
  /aruco_detections   — vision_msgs/Detection3DArray (camera frame)
  /aruco_markers      — visualization_msgs/MarkerArray (world frame, RViz)
  /aruco_diagnostic   — sensor_msgs/Image (annotated debug image)

Saves diagnostic images to /tmp/aruco_diagnostics/.
Ground truth comparison against Gazebo via "gz model --pose".
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
from sensor_msgs.msg import Image as RosImage, JointState
from vision_msgs.msg import (Detection3D, Detection3DArray,
                              ObjectHypothesisWithPose, BoundingBox3D)
from visualization_msgs.msg import Marker, MarkerArray

# Fix NumPy 2.x compatibility
os.environ['NPY_COMPAT_OVERRIDE'] = '1'

# Fix numpy 2.x compatibility with ikpy/sympy
import sympy.core.numbers as scn
_orig_cvt = scn._convert_numpy_types
def _patched_cvt(a, **kw):
    if isinstance(a, np.floating):
        from sympy.core.numbers import Float
        return Float(float(a), precision=np.finfo(a).nmant + 1)
    if isinstance(a, np.complexfloating):
        return scn._sympy_converter[complex](complex(a))
    if isinstance(a, np.integer):
        from sympy.core.sympify import sympify
        return sympify(int(a), **kw)
    return _orig_cvt(a, **kw)
scn._convert_numpy_types = _patched_cvt

from ikpy.chain import Chain
from ament_index_python.packages import get_package_share_directory

# ── Camera intrinsics ─────────────────────────────────────────────────
CAM_FX = 554.3
CAM_FY = 554.3
CAM_CX = 320.0
CAM_CY = 240.0
CAMERA_MATRIX = np.array([[CAM_FX, 0, CAM_CX],
                          [0, CAM_FY, CAM_CY],
                          [0, 0, 1]], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)
CAM_POS = np.array([0.72, 0.0, 2.8])
R_CAM_TO_WORLD = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64)

# ── Marker geometry ───────────────────────────────────────────────────
MARKER_SIZE = 0.045
HALF_M = MARKER_SIZE / 2
MARKER_OBJ_PTS = np.array(
    [[-HALF_M, -HALF_M, 0], [HALF_M, -HALF_M, 0],
     [HALF_M, HALF_M, 0], [-HALF_M, HALF_M, 0]], dtype=np.float64)

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
ID_TO_BOX = {0: 'box_1', 1: 'box_2', 2: 'box_3'}

DIAG_DIR = Path('/tmp/aruco_diagnostics')
DIAG_SAVE_EVERY = 10
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

# ── Arm-centric depth model (Plan A) ───────────────────────────────────
# Object plane: box center Z = 0.925 (table 0.825 + box half-height 0.10)
# Camera Z = 2.80 → optical depth D_OBJECT = 2.80 - 0.925 = 1.875
# Marker sits on box top at Z = 0.925 + 0.1015 = 1.0265
# Pixel→world XY computed at object plane; world Z output = 0.925 (box center)
# Arm-centric depth = FK(tool0_z) - OBJECT_Z (logged for validation)
D_OBJECT = 1.875
OBJECT_Z = 0.925
MARKER_TO_OBJECT_Z_OFFSET = 0.1015

# ── Arm FK setup (for arm-centric depth metric) ────────────────────────
BASE_Z = 0.80
JOINT_NAMES_FK = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint',      'wrist_2_joint',       'wrist_3_joint',
]
URDF_PATH = os.path.join(get_package_share_directory('ur5e_gazebo_demo'),
                         'urdf', 'ur5e_full.urdf')


def rvec_to_quaternion(rvec):
    """Rotation vector → quaternion (x, y, z, w) as pure Python floats."""
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

        self.declare_parameter('compare_ground_truth', True)
        self.compare_gt = (
            self.get_parameter('compare_ground_truth')
            .get_parameter_value().bool_value)
        self.declare_parameter('capture_rate', 10.0)
        self.capture_rate = (
            self.get_parameter('capture_rate')
            .get_parameter_value().double_value)

        self.detector = cv2.aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

        # Publishers
        self.det_pub = self.create_publisher(
            Detection3DArray, '/aruco_detections', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/aruco_markers', 10)
        self.diag_pub = self.create_publisher(
            RosImage, '/aruco_diagnostic', 10)

        # State
        self._frame_count = 0
        self._gt_errors = []
        self._last_gt_query = 0.0

        # Subscription to camera images (ROS2 bridged topic)
        self._latest_img = None
        self.sub = self.create_subscription(
            RosImage, '/camera/image_raw', self._img_cb, 10)

        # Process timer — runs at capture_rate Hz
        period = 1.0 / max(self.capture_rate, 1.0)
        self.timer = self.create_timer(period, self._capture_and_detect)

        # ── Arm FK for arm-centric depth metric ─────────────────────
        try:
            self._chain = Chain.from_urdf_file(URDF_PATH)
        except Exception as e:
            self.get_logger().warn(
                f'Failed to load URDF for FK: {e}. Arm-centric depth disabled.')
            self._chain = None

        self._current_joints = None
        self._joint_sub = self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10)

        # Diagnostic housekeeping
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        all_files = sorted(DIAG_DIR.glob('frame_*.png'))
        if len(all_files) > 100:
            for f in all_files[:-100]:
                f.unlink()

        self.get_logger().info(
            f'ArUco detector ready (dict=DICT_4X4_50, marker={MARKER_SIZE}m, '
            f'cam=({CAM_POS[0]:.2f},{CAM_POS[1]:.2f},{CAM_POS[2]:.2f}), '
            f'D_object={D_OBJECT:.4f}, capture_rate={self.capture_rate}Hz, '
            f'gt_compare={self.compare_gt})')

    # ── Main capture + detection loop ──────────────────────────────────
    def _img_cb(self, msg: RosImage):
        """Store latest camera image (no cv_bridge)."""
        try:
            h, w = msg.height, msg.width
            data = np.frombuffer(msg.data, dtype=np.uint8)
            enc = msg.encoding
            if enc in ('rgb8', 'rgba8'):
                self._latest_img = data.reshape((h, w, -1))
            elif enc in ('bgr8', 'bgra8'):
                img = data.reshape((h, w, -1))
                self._latest_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif enc == 'mono8':
                self._latest_img = cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2RGB)
            else:
                self._latest_img = data.reshape((h, w, 3))
        except Exception:
            pass

    # ── Arm FK callbacks ────────────────────────────────────────────────
    def _js_cb(self, msg: JointState):
        """Store latest joint positions for FK-based arm-centric depth."""
        m = dict(zip(msg.name, msg.position))
        if all(n in m for n in JOINT_NAMES_FK):
            self._current_joints = [m[n] for n in JOINT_NAMES_FK]

    def _compute_arm_centric_depth(self):
        """Compute arm-centric depth: tool0_world_z - OBJECT_Z.

        Uses IKPy FK from current joint state. Returns depth in meters,
        or None if joint state or chain unavailable.
        When arm is at home (standing), tool0_world_z ≈ 1.88-1.93 m,
        giving arm_centric_depth ≈ 0.96-1.01 m.
        """
        if self._chain is None or self._current_joints is None:
            return None
        try:
            full_joints = [0.0, 0.0] + list(self._current_joints) + [0.0]
            fk = self._chain.forward_kinematics(full_joints)
            tool0_base_z = float(fk[2, 3])
            tool0_world_z = tool0_base_z + BASE_Z
            return tool0_world_z - OBJECT_Z
        except Exception:
            return None

    def _capture_and_detect(self):
        self._frame_count += 1
        # Try ROS2 topic first, fall back to direct Gazebo capture
        rgb = self._latest_img
        if rgb is not None:
            rgb = rgb.copy()
        else:
            rgb = self._capture_gz_image()
        if rgb is None:
            if self._frame_count % 30 == 1:
                print(f'[ARUCO] Frame #{self._frame_count}: no image', flush=True)
            return

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if self._frame_count % 30 == 1:
            aruco_n = len(ids) if ids is not None else 0
            self.get_logger().info(
                f'Frame #{self._frame_count}: ArUco={aruco_n}, checking blobs...')
            arm_depth = self._compute_arm_centric_depth()
            if arm_depth is not None:
                self.get_logger().info(
                    f'  Arm-centric depth: {arm_depth:.3f}m '
                    f'(optical D_object={D_OBJECT:.3f}m)')

        stamp = self.get_clock().now().to_msg()
        det_array = Detection3DArray()
        det_array.header.stamp = stamp
        det_array.header.frame_id = 'top_camera_link'

        marker_array = MarkerArray()
        annotated = rgb.copy()

        # Fallback: if ArUco fails, use blob/color detection
        use_blob_fallback = (ids is None or len(ids) == 0)
        blob_detections = []
        if use_blob_fallback:
            blob_detections = self._detect_blobs_color(rgb)
            use_blob_fallback = len(blob_detections) > 0

        if self._frame_count % 30 == 1:
            self.get_logger().info(
                f'Frame #{self._frame_count}: ArUco={0 if ids is None else len(ids)}, '
                f'blobs={len(blob_detections)}, fallback={use_blob_fallback}')

        if not use_blob_fallback and ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                success, rvec, tvec = cv2.solvePnP(
                    MARKER_OBJ_PTS, corners[i][0],
                    CAMERA_MATRIX, DIST_COEFFS,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not success:
                    continue

                t_cam = tvec.flatten()
                # Plan A: shift from marker plane (Z=1.0265) to object plane (Z=0.925)
                t_cam[2] += MARKER_TO_OBJECT_Z_OFFSET
                t_world = R_CAM_TO_WORLD @ t_cam + CAM_POS
                # t_world[2] ≈ 0.925 (box center), not 1.0265 (marker top)

                cv2.drawFrameAxes(annotated, CAMERA_MATRIX, DIST_COEFFS,
                                  rvec, tvec, MARKER_SIZE * 1.5)

                cx_c = int(corners[i][0][:, 0].mean())
                cy_c = int(corners[i][0][:, 1].mean())
                label = f'ID:{marker_id} ({t_world[0]:.3f},{t_world[1]:.3f})'
                cv2.putText(annotated, label, (cx_c + 15, cy_c),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            COLORS[marker_id % 3], 1)

                # Detection3D in camera frame
                det = Detection3D()
                det.header.stamp = stamp
                det.header.frame_id = 'top_camera_link'
                det.id = f'marker_{marker_id}'
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = 'aruco_marker'
                hyp.hypothesis.score = 1.0
                det.results.append(hyp)
                bbox = BoundingBox3D()
                bbox.size.x = float(MARKER_SIZE)
                bbox.size.y = float(MARKER_SIZE)
                bbox.size.z = 0.001
                bbox.center.position.x = float(t_cam[0])
                bbox.center.position.y = float(t_cam[1])
                bbox.center.position.z = float(t_cam[2])
                qx_c, qy_c, qz_c, qw_c = rvec_to_quaternion(rvec)
                bbox.center.orientation.x = qx_c
                bbox.center.orientation.y = qy_c
                bbox.center.orientation.z = qz_c
                bbox.center.orientation.w = qw_c
                det.bbox = bbox
                det_array.detections.append(det)

                # RViz marker (world frame)
                R_marker_cam, _ = cv2.Rodrigues(rvec)
                R_marker_world = R_CAM_TO_WORLD @ R_marker_cam
                r_world, _ = cv2.Rodrigues(R_marker_world)
                qx, qy, qz, qw = rvec_to_quaternion(r_world)
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = 'world'
                m.ns = 'aruco_marker'
                m.id = int(marker_id)
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = float(t_world[0])
                m.pose.position.y = float(t_world[1])
                m.pose.position.z = float(t_world[2])
                m.pose.orientation.x = qx
                m.pose.orientation.y = qy
                m.pose.orientation.z = qz
                m.pose.orientation.w = qw
                m.scale.x = MARKER_SIZE
                m.scale.y = MARKER_SIZE
                m.scale.z = 0.002
                c = COLORS[marker_id % 3]
                m.color.r = float(c[2]) / 255.0
                m.color.g = float(c[1]) / 255.0
                m.color.b = float(c[0]) / 255.0
                m.color.a = 0.8
                marker_array.markers.append(m)

        elif use_blob_fallback:
            # ── Blob/color fallback processing ──────────────────────
            for mid, cu, cv_px, box_pts, wx, wy in blob_detections:
                # Ensure 4 corner points in consistent order for solvePnP
                # cv2.boxPoints returns (4,2) or (4,1,2), normalize to (4,2)
                box_pts = np.array(box_pts, dtype=np.float64)
                if box_pts.ndim == 3:
                    box_pts = box_pts.reshape(4, 2)
                # Sort by angle around centroid for CCW order
                center = box_pts.mean(axis=0)
                angles = np.arctan2(box_pts[:, 1] - center[1],
                                    box_pts[:, 0] - center[0])
                ordered = box_pts[np.argsort(angles)]
                # imagePoints: (4, 2) float64 contiguous
                img_pts = np.ascontiguousarray(ordered, dtype=np.float64)
                # objectPoints: (4, 3) — must be 3D, Z=0 for planar marker
                obj_pts = np.ascontiguousarray(MARKER_OBJ_PTS, dtype=np.float64)

                # solvePnP with blob corners
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts,
                    CAMERA_MATRIX, DIST_COEFFS,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not success:
                    continue

                # Compute correct camera-frame position from pixel geometry
                # (solvePnP tvec unreliable with blob corners)
                # Camera model: u = -FX*Yw/D + CX, v = FY*(Xw-0.72)/D + CY
                # Inverse: Yw = -(u-CX)*D/FX, Xw = 0.72 + (v-CY)*D/FY
                # Plan A: use D_OBJECT = 1.875 (object plane), world_z = OBJECT_Z = 0.925
                world_z = float(CAM_POS[2] - D_OBJECT)  # = 0.925 (box center)
                t_world = np.array([wx, wy, world_z])
                # Camera-frame position: R_world_to_cam @ (world - cam_pos)
                # R_world_to_cam = R_cam_to_world^T = [[0,-1,0],[1,0,0],[0,0,-1]]
                dw = t_world - CAM_POS  # [wx-0.72, wy-0, world_z-2.8]
                t_cam_correct = np.array([-dw[1], dw[0], -dw[2]], dtype=np.float64)
                # t_cam_correct = [-wy, wx-0.72, D_OBJECT]

                # Draw blob contour + axes
                cv2.drawContours(annotated, [box_pts.astype(np.int32)], 0,
                                 COLORS[mid], 2)
                cv2.drawFrameAxes(annotated, CAMERA_MATRIX, DIST_COEFFS,
                                  rvec, tvec, MARKER_SIZE * 1.5)
                label = f'ID:{mid} blob ({t_world[0]:.3f},{t_world[1]:.3f})'
                cv2.putText(annotated, label, (int(cu) + 15, int(cv_px)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[mid], 1)

                # Build Detection3D
                det = Detection3D()
                det.header.stamp = stamp
                det.header.frame_id = 'top_camera_link'
                det.id = f'marker_{mid}'
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = 'blob_marker'
                hyp.hypothesis.score = 0.8
                det.results.append(hyp)
                bbox = BoundingBox3D()
                bbox.size.x = float(MARKER_SIZE)
                bbox.size.y = float(MARKER_SIZE)
                bbox.size.z = 0.001
                bbox.center.position.x = float(t_cam_correct[0])
                bbox.center.position.y = float(t_cam_correct[1])
                bbox.center.position.z = float(t_cam_correct[2])
                qx_c, qy_c, qz_c, qw_c = rvec_to_quaternion(rvec)
                bbox.center.orientation.x = qx_c
                bbox.center.orientation.y = qy_c
                bbox.center.orientation.z = qz_c
                bbox.center.orientation.w = qw_c
                det.bbox = bbox
                det_array.detections.append(det)

                # RViz marker
                R_marker_cam, _ = cv2.Rodrigues(rvec)
                R_marker_world = R_CAM_TO_WORLD @ R_marker_cam
                r_world, _ = cv2.Rodrigues(R_marker_world)
                qx, qy, qz, qw = rvec_to_quaternion(r_world)
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = 'world'
                m.ns = 'aruco_marker'
                m.id = int(mid)
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = float(t_world[0])
                m.pose.position.y = float(t_world[1])
                m.pose.position.z = float(t_world[2])
                m.pose.orientation.x = qx
                m.pose.orientation.y = qy
                m.pose.orientation.z = qz
                m.pose.orientation.w = qw
                m.scale.x = MARKER_SIZE
                m.scale.y = MARKER_SIZE
                m.scale.z = 0.002
                c = COLORS[mid % 3]
                m.color.r = float(c[2]) / 255.0
                m.color.g = float(c[1]) / 255.0
                m.color.b = float(c[0]) / 255.0
                m.color.a = 0.8
                marker_array.markers.append(m)

        self.det_pub.publish(det_array)
        self.marker_pub.publish(marker_array)

        # Diagnostic image
        if self._frame_count % DIAG_SAVE_EVERY == 0:
            diag = RosImage()
            diag.header.stamp = stamp
            diag.header.frame_id = 'top_camera_link'
            diag.height = annotated.shape[0]
            diag.width = annotated.shape[1]
            diag.encoding = 'rgb8'
            diag.is_bigendian = False
            diag.step = annotated.shape[1] * 3
            diag.data = annotated.tobytes()
            self.diag_pub.publish(diag)

            fname = DIAG_DIR / f'frame_{self._frame_count:05d}.png'
            cv2.imwrite(str(fname), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        # GT comparison
        if self.compare_gt and ids is not None and len(ids) > 0:
            now_t = time.time()
            if now_t - self._last_gt_query > 5.0:
                self._last_gt_query = now_t
                self._compare_gt(ids.flatten())

    # ── Blob + bright-square detection ────────────────────────────────
    def _detect_blobs_color(self, rgb):
        """Detect bright square markers on box tops.

        Gazebo Ogre2 renders markers as bright white/gray squares (~14px).
        Strategy: find the 3 brightest square-like blobs in the left half
        of the image (blue zone), sort by Y coordinate (near→far), and
        assign marker IDs in order.
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        left_gray = gray[:, :320]

        def _pixel_to_world(cx, cy):
            xn = (cx - CAM_CX) / CAM_FX
            yn = (cy - CAM_CY) / CAM_FY
            wx = CAM_POS[0] + yn * D_OBJECT
            wy = -xn * D_OBJECT
            return float(wx), float(wy)

        # Find all bright blobs in left half
        lth = np.percentile(left_gray, 92)
        _, thresh = cv2.threshold(left_gray, lth, 255, cv2.THRESH_BINARY)
        conts, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for cnt in conts:
            a = cv2.contourArea(cnt)
            # Marker area: ~80-250 px at this distance for 0.045m²
            if a < 80 or a > 300:
                continue
            M = cv2.moments(cnt)
            if M['m00'] <= 0:
                continue
            cu = int(M['m10'] / M['m00'])
            cv_px = int(M['m01'] / M['m00'])
            wx, wy = _pixel_to_world(cu, cv_px)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            blobs.append((a, cu, cv_px, box, wx, wy))

        # Sort by area descending, take top 3
        blobs.sort(key=lambda b: -b[0])
        blobs = blobs[:3]

        # Sort by world Y (distance from camera center = how far into blue zone)
        # to assign marker IDs consistently: nearest first → marker 0
        blobs.sort(key=lambda b: b[5])  # sort by wy

        detections = []
        for i, (area, cu, cv_px, box, wx, wy) in enumerate(blobs):
            detections.append((i, cu, cv_px, box, wx, wy))
            self.get_logger().info(
                f'  Marker[{i}]: pixel=({cu},{cv_px}) area={area:.0f} '
                f'→ world=({wx:.3f},{wy:.3f})')

        return detections

    # ── Capture image from Gazebo ─────────────────────────────────────
    def _capture_gz_image(self):
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
                if (data_str[i:i + 2] == '\\\\' and i + 5 <= len(data_str)
                        and data_str[i + 2:i + 5].isdigit()):
                    raw.append(int(data_str[i + 2:i + 5], 8))
                    i += 5
                elif (data_str[i] == '\\' and i + 3 < len(data_str)
                        and data_str[i + 1:i + 4].isdigit()):
                    raw.append(int(data_str[i + 1:i + 4], 8))
                    i += 4
                else:
                    raw.append(ord(data_str[i]))
                    i += 1

            expected = 640 * 480 * 3
            if len(raw) < expected:
                return None
            raw = bytes(raw[:expected])
            return np.frombuffer(raw, dtype=np.uint8).reshape((480, 640, 3))
        except Exception:
            return None

    # ── Ground truth comparison ───────────────────────────────────────
    def _compare_gt(self, ids_arr):
        errors = []
        for marker_id in ids_arr:
            if marker_id not in ID_TO_BOX:
                continue
            try:
                r = subprocess.run(
                    ['gz', 'model', '-m', ID_TO_BOX[marker_id], '--pose'],
                    capture_output=True, text=True, timeout=3)
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        nums = [float(x) for x in line[1:-1].split()]
                        if len(nums) >= 3:
                            gt = np.array(nums[:3])
                            # Need world-frame estimate from most recent detection
                            self.get_logger().info(
                                f'  GT box_{marker_id+1}: '
                                f'gt=({gt[0]:.4f},{gt[1]:.4f},{gt[2]:.4f})')
                        break
            except Exception:
                pass


def main():
    rclpy.init()
    node = ArucoDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
