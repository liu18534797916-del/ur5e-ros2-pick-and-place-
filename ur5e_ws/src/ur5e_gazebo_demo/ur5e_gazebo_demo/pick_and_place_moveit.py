#!/usr/bin/env python3
"""
Pick and Place — IKPy 逆运动学 + 硬编码位置 + 侧向水平前伸抓取。

策略：先下降至盒子后方指定高度，再水平前伸抓取，避免顶向下碰倒盒子。

桌子 2.5m × 2.0m，前方 +X 用 Y=0 分左右：
  Y>0 → 蓝色区域 ← 抓取
  Y<0 → 红色区域 ← 放置
"""

import os
import subprocess
import time
import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ═══════════════════════════════════════════════════════════════════════════
# Fix sympy for NumPy 2.x compatibility BEFORE importing ikpy
# ═══════════════════════════════════════════════════════════════════════════
import sympy.core.numbers as scn

def _fixed_convert_numpy_types(a, **sympify_args):
    import numpy as _np
    if not isinstance(a, _np.floating):
        if _np.iscomplex(a):
            return scn._sympy_converter[complex](a.item())
        else:
            from sympy import sympify
            return sympify(a.item(), **sympify_args)
    else:
        from sympy.core.numbers import Float
        return Float(float(a), precision=53)

scn._convert_numpy_types = _fixed_convert_numpy_types

import ikpy.chain

# ═══════════════════════════════════════════════════════════════════════════
# Load IKPy chain from URDF
# ═══════════════════════════════════════════════════════════════════════════
from ament_index_python.packages import get_package_share_directory
_URDF_PATH = os.path.join(
    get_package_share_directory('ur5e_gazebo_demo'), 'urdf', 'ur5e_full.urdf')
_CHAIN = ikpy.chain.Chain.from_urdf_file(_URDF_PATH, base_elements=['world'])

# Active revolute joint indices in the 10-link chain:
# [0]=world(fixed) [1]=base_joint(fixed) [2]=inertia(fixed)
# [3]=shoulder_pan [4]=shoulder_lift [5]=elbow
# [6]=wrist_1 [7]=wrist_2 [8]=wrist_3 [9]=ft_frame(fixed)
_IDX_REVOLUTE = [3, 4, 5, 6, 7, 8]

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint',      'wrist_2_joint',       'wrist_3_joint',
]
HOME_JOINTS = [0.0, -1.5707, 0.0, -1.5707, 0.0, 0.0]

# ═══════════════════════════════════════════════════════════════════════════
# IK helper: world position → joint angles (6-DOF)
# ═══════════════════════════════════════════════════════════════════════════

# Target end-effector Z axis in world frame.
# ft_frame Z = world -X → tool0 Z = world -X → Robotiq fingers = world +X (FORWARD).
_TARGET_Z_AXIS = [-1.0, 0.0, 0.0]

def _joints_to_chain(q_6dof):
    """Convert 6-DOF joint list to 10-element chain vector."""
    full = [0.0] * 10
    for idx, val in zip(_IDX_REVOLUTE, q_6dof):
        full[idx] = val
    return full

def _chain_to_joints(full):
    """Extract 6-DOF joints from 10-element chain vector."""
    return [full[i] for i in _IDX_REVOLUTE]

def ik_solve(world_x, world_y, world_z, seed_6dof):
    """Compute IK for a world-frame position with tool Z constrained.

    Uses orientation_mode='Z' to align ft_frame Z to world -X (backward).
    Since Robotiq fingers point opposite to ft_frame Z, this makes the
    fingers point FORWARD (world +X).

    Returns 6 joint angles in [-π, π].
    """
    target_pos = [world_x, world_y, world_z]
    seed_full = _joints_to_chain(seed_6dof)
    result_full = _CHAIN.inverse_kinematics(
        target_position=target_pos,
        target_orientation=_TARGET_Z_AXIS,
        initial_position=seed_full,
        orientation_mode='Z')
    result_6dof = _chain_to_joints(result_full)
    # Wrap to [-π, π]
    return [(np.pi + q) % (2 * np.pi) - np.pi for q in result_6dof]

def fk_position(joints_6dof):
    """Forward kinematics: joint angles → world-frame ft_frame position."""
    full = _joints_to_chain(joints_6dof)
    T = _CHAIN.forward_kinematics(full)
    return np.array([T[0, 3], T[1, 3], T[2, 3]])

def fk_axis(joints_6dof):
    """FK: joint angles → ft_frame Z-axis in world frame."""
    full = _joints_to_chain(joints_6dof)
    T = _CHAIN.forward_kinematics(full)
    return np.array([T[0, 2], T[1, 2], T[2, 2]])

# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════

BASE_Z = 0.80
PRE_DIST    = 0.16   # tool0 到手指末端距离 + 安全余量，让手指在盒子正上方
GRASP_Z_OFF = 0.05   # tool0 高于盒子中心的高度，让手指末端下降到盒子中部
LIFT_Z      = 0.20   # 抓取后提起高度
ABOVE_Z     = 0.20   # 预抓取 tool0 高于盒子中心高度

FORWARD_PRE_DIST = 0.15  # 前向推进预抓取距离：停在盒子前方15cm
BOX_WIDTH = 0.06           # 盒子宽度 (6cm)
# 夹爪开合映射：joint=0.0 → 85mm全开, joint=0.8 → 0mm闭合
# 保留 BOX_WIDTH 间隙：joint ≈ 0.8 * (1 - 60/85) ≈ 0.24
GRIP_CLOSE_PRIMARY  = 0.24
GRIP_CLOSE_FALLBACK = 0.20
LIFT_VERIFY_DZ      = 0.05

# ═══════════════════════════════════════════════════════════════════════════
# PickAndPlace Node
# ═══════════════════════════════════════════════════════════════════════════

class PickAndPlace(Node):

    def __init__(self):
        super().__init__('pick_and_place_moveit')

        self._current_joints = HOME_JOINTS[:]
        self._detected_poses = {}
        self._attached_box = None
        self._attach_offset = None

        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

        self.arm_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # Wait for joint states
        deadline = time.time() + 30.0
        while self._current_joints is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.3)
        if self._current_joints is None:
            self._current_joints = HOME_JOINTS[:]

        self.get_logger().info(
            'Ready. joints: ' + str([round(v, 2) for v in self._current_joints]))

    # ── Callbacks ──────────────────────────────────────────────────────
    def _js_cb(self, msg: JointState):
        m = dict(zip(msg.name, msg.position))
        if all(n in m for n in JOINT_NAMES):
            self._current_joints = [m[n] for n in JOINT_NAMES]

    def _spin(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    @property
    def current_joints(self):
        return list(self._current_joints) if self._current_joints else HOME_JOINTS[:]

    # ── Joint motion ──────────────────────────────────────────────────
    def direct_joints(self, positions, secs=5, steps=20):
        """平滑多 waypoint 关节轨迹插值。"""
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0

        start = np.array(self.current_joints, dtype=np.float64)
        end = np.array(positions, dtype=np.float64)

        dt = secs / max(steps - 1, 1)
        for k in range(steps):
            alpha = k / max(steps - 1, 1)
            interp = start + alpha * (end - start)
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in interp]
            t = k * dt
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            msg.points.append(pt)

        self.arm_pub.publish(msg)
        self._spin(secs + 2.0)

    def gripper(self, pos, secs=2):
        msg = JointTrajectory()
        msg.joint_names = ['robotiq_85_left_knuckle_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(pos)]
        pt.time_from_start.sec = int(secs)
        pt.time_from_start.nanosec = int((secs - int(secs)) * 1e9)
        msg.points.append(pt)
        self.gripper_pub.publish(msg)
        self._spin(secs + 0.3)

    # ── Pose-to-joints (IKPy) ─────────────────────────────────────────
    def _move_to_world_pose(self, wx, wy, wz, label='', secs=5):
        """IK solve → move with smooth interpolation."""
        seed = self.current_joints
        try:
            target_joints = ik_solve(wx, wy, wz, seed)
        except Exception as e:
            self.get_logger().error(f'[{label}] IK failed: {e}')
            return False

        self.get_logger().info(
            f'[{label}] moving to world ({wx:.3f}, {wy:.3f}, {wz:.3f})')
        self.direct_joints(target_joints, secs=secs)
        self._current_joints = target_joints
        return True

    def _move_line(self, wx1, wy1, wz1, wx2, wy2, wz2, label='', secs=6, steps=15):
        """直线插值运动：task space 采样 → IK 求解 → 发送所有 waypoints。

        使用 15步/6秒 慢速推进，避免碰倒盒子。
        wrist_3 每步锁定，防止绕工具 Z 轴旋转。
        """
        seed = self.current_joints
        self.get_logger().info(
            f'[{label}] line: ({wx1:.3f},{wy1:.3f},{wz1:.3f}) → '
            f'({wx2:.3f},{wy2:.3f},{wz2:.3f})')

        # 记住起始 wrist_3，全程锁定
        fixed_wrist3 = seed[5]  # JOINT_NAMES[5] = 'wrist_3_joint'

        all_joints = [seed]
        for k in range(steps):
            alpha = (k + 1) / steps
            wx = wx1 + alpha * (wx2 - wx1)
            wy = wy1 + alpha * (wy2 - wy1)
            wz = wz1 + alpha * (wz2 - wz1)
            try:
                seed = ik_solve(wx, wy, wz, seed)
                seed[5] = fixed_wrist3  # 锁定 wrist_3，防止旋转
                all_joints.append(seed)
            except Exception as e:
                self.get_logger().error(f'[{label}] IK line-step {k} failed: {e}')
                return False

        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        total_dt = secs / max(steps, 1)
        for k, joints in enumerate(all_joints[1:], start=1):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in joints]
            t = k * total_dt
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            msg.points.append(pt)

        self.arm_pub.publish(msg)
        self._spin(secs + 2.0)
        self._current_joints = all_joints[-1]
        return True

    # ── Gazebo 工具 ──────────────────────────────────────────────────
    def _get_gz_pose(self, model, link=None):
        args = ['gz', 'model', '-m', model, '--pose']
        if link:
            args = ['gz', 'model', '-m', model, '-l', link, '--pose']
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=2)
            all_poses = []
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    try:
                        nums = [float(x) for x in line[1:-1].split()]
                        if len(nums) == 3:
                            all_poses.append(nums)
                    except ValueError:
                        continue
            if not all_poses:
                return None
            if link and len(all_poses) >= 2:
                return all_poses[-2]
            if len(all_poses) >= 2:
                return all_poses[-2]
            return all_poses[-1] if all_poses else None
        except Exception:
            pass
        return None

    # ── 运动学附着/脱离 ─────────────────────────────────────────────
    def _attach_box_kinematic(self, box_name):
        """记录盒子相对夹爪的偏移，开始跟随。"""
        box = self._get_gz_pose(box_name)
        arm = self._get_gz_pose('ur5e', 'wrist_3_link')
        if not box or not arm:
            self.get_logger().warn('Cannot compute attach offset')
            return False
        self._attach_offset = [box[i] - arm[i] for i in range(3)]
        self._attached_box = box_name
        self.get_logger().info(
            f'Attached {box_name}: offset={[round(v, 3) for v in self._attach_offset]}')
        return True

    def _detach_box_kinematic(self, box_name):
        """停止跟随，释放盒子。"""
        self._attached_box = None
        self._attach_offset = None
        self.get_logger().info(f'Detached {box_name}')

    def _sync_box(self):
        """将盒子位姿同步到夹爪当前位置。"""
        if not self._attached_box or not self._attach_offset:
            return
        arm = self._get_gz_pose('ur5e', 'wrist_3_link')
        if not arm:
            return
        bx, by, bz = [arm[i] + self._attach_offset[i] for i in range(3)]
        subprocess.run(['gz', 'service', '-s', '/world/camera_world/set_pose',
                       '--reqtype', 'gz.msgs.Pose',
                       '--reptype', 'gz.msgs.Boolean', '--timeout', '2000',
                       '-r', f'name: "{self._attached_box}" '
                             f'position: {{x: {bx:.6f} y: {by:.6f} z: {bz:.6f}}} '
                             f'orientation: {{x: 0.0 y: 0.0 z: 0.0 w: 1.0}}'],
                       capture_output=True, timeout=3)

    def _spin_sync(self, secs, sync_interval=0.2):
        """等待期间持续同步盒子位置。"""
        end = time.time() + secs
        while time.time() < end:
            self._sync_box()
            rclpy.spin_once(self, timeout_sec=min(sync_interval, end - time.time()))

    # ── 抓取验证 ──────────────────────────────────────────────────────
    def _verify_grasp(self, box_name, orig_pos):
        """检查箱子是否还在原位（未被推走/碰倒）。"""
        for attempt in range(3):
            self._spin(0.3)
            box = self._get_gz_pose(box_name, None)
            if box:
                dist = np.hypot(box[0] - orig_pos[0], box[1] - orig_pos[1])
                dz = box[2] - orig_pos[2]
                self.get_logger().info(
                    f'Grasp pre-check #{attempt+1}: dist={dist:.3f}m dz={dz:.3f}m')
                if dz < -0.05:
                    self.get_logger().warn(f'❌ {box_name} knocked over')
                    return False
                if dist > 0.03:
                    self.get_logger().warn(f'❌ {box_name} pushed away (dist={dist:.3f}m)')
                    return False
                self.get_logger().info(f'✅ {box_name} in position')
                return True
            time.sleep(0.3)
        return True

    def _verify_lift(self, box_name, orig_z):
        """抬升后验证箱子是否离开桌面。"""
        self._spin(0.5)
        box = self._get_gz_pose(box_name, None)
        if box:
            dz = box[2] - orig_z
            self.get_logger().info(f'Lift verify: box z={box[2]:.3f} dz={dz:.3f}m')
            if dz > LIFT_VERIFY_DZ:
                self.get_logger().info(f'✅ {box_name} lift confirmed!')
                return True
            self.get_logger().warn(f'❌ {box_name} did not rise')
        return False

    # ═══════════════════════════════════════════════════════════════════
    # 单次抓取放置 — 前向水平推进方式
    #   夹爪水平朝前，从 -X 方向向前推进，避免垂直下降碰倒盒子。
    #   预抓取 → 笛卡尔直线推进 → 闭合 → 提拉 → 搬运 → 放置
    # ═══════════════════════════════════════════════════════════════════
    def _pick_forward(self, box_name, pick_x, pick_y, pick_z,
                      place_x, place_y, place_z):
        """
        前向水平推进抓取：
          Step 1: 移到盒子前方 FORWARD_PRE_DIST 处
          Step 2: 笛卡尔直线推进到抓取位置
          Step 3: 静止等待 → 闭合夹爪 → 提拉验证
          Step 4: 搬运到放置点
        """
        self.get_logger().info(
            f'=== {box_name} [FORWARD]: pick ({pick_x:.2f},{pick_y:.2f}) → '
            f'place ({place_x:.2f},{place_y:.2f}) ===')

        # tool0 位置计算：手指朝 +X，tool0 在手指后方
        tool0_x_grasp = pick_x - PRE_DIST
        tool0_x_pre   = tool0_x_grasp - FORWARD_PRE_DIST
        tool_z = pick_z + GRASP_Z_OFF
        lift_z = pick_z + LIFT_Z

        # ── Step 1: 预抓取 — 移到盒子前方 ──────────────────────────
        self.get_logger().info(
            f'--- {box_name} pre-grasp: forward approach '
            f'({tool0_x_pre:.3f}, {pick_y:.3f}, {tool_z:.3f}) ---')
        if not self._move_to_world_pose(tool0_x_pre, pick_y, tool_z,
                                        label=f'{box_name} pre-grasp', secs=5):
            return False

        # ── Step 2: 笛卡尔直线水平推进 ─────────────────────────────
        self.get_logger().info(
            f'--- {box_name} grasp-in: line '
            f'({tool0_x_pre:.3f},{pick_y:.3f},{tool_z:.3f}) → '
            f'({tool0_x_grasp:.3f},{pick_y:.3f},{tool_z:.3f}) ---')
        if not self._move_line(tool0_x_pre, pick_y, tool_z,
                               tool0_x_grasp, pick_y, tool_z,
                               label=f'{box_name} grasp-in', secs=6, steps=15):
            self._move_to_world_pose(tool0_x_pre, pick_y, tool_z,
                                     label=f'{box_name} abort-pre', secs=4)
            return False

        self._spin(2.0)  # 等机械臂完全静止

        # ── Step 3: 夹爪闭合 + 提拉验证 ────────────────────────────
        GRIP_POSITIONS = [GRIP_CLOSE_PRIMARY, GRIP_CLOSE_FALLBACK]
        grasp_success = False
        for grip_attempt in range(3):
            grip_pos = GRIP_POSITIONS[min(grip_attempt, len(GRIP_POSITIONS) - 1)]
            self.get_logger().info(
                f'--- {box_name} close gripper to {grip_pos} '
                f'(attempt {grip_attempt + 1}/3) ---')
            self.gripper(grip_pos, secs=2.0)
            self._spin(0.8)

            if self._verify_grasp(box_name, [pick_x, pick_y, pick_z]):
                self.get_logger().info(f'--- {box_name} ATTACH ---')
                self._attach_box_kinematic(box_name)
                self.get_logger().info(f'--- {box_name} lift ---')
                self._move_to_world_pose(tool0_x_grasp, pick_y, lift_z,
                                         label=f'{box_name} lift', secs=3)
                self._sync_box()
                self.get_logger().info(f'✅ {box_name} grasped and attached!')
                grasp_success = True
                break
            else:
                self.get_logger().warn(f'{box_name} box disturbed — retrying')
                self.gripper(0.0, secs=2.0)
                self._move_to_world_pose(tool0_x_pre, pick_y, tool_z,
                                         label=f'{box_name} retry-pre', secs=4)
                self._move_line(tool0_x_pre, pick_y, tool_z,
                                tool0_x_grasp, pick_y, tool_z,
                                label=f'{box_name} retry-grasp', secs=6, steps=15)

        if not grasp_success:
            self.get_logger().error(f'❌ {box_name} FORWARD GRASP FAILED after 3 attempts')
            self.gripper(0.0, secs=2.0)
            self._move_to_world_pose(tool0_x_pre, pick_y, tool_z,
                                     label=f'{box_name} final-pre', secs=4)
            return False

        # ── 回到 home 再搬运到放置点 ──────────────────────────────
        self._move_to_world_pose(0.0, 0.0, BASE_Z + 0.5,
                                 label=f'{box_name} home-mid', secs=5)
        self._sync_box()

        self.get_logger().info(f'--- {box_name} transport to place ---')
        place_tool0_x = place_x - PRE_DIST
        place_tool0_x_pre = place_tool0_x - FORWARD_PRE_DIST
        self._move_to_world_pose(place_tool0_x_pre, place_y, lift_z,
                                 label=f'{box_name} place-pre', secs=4)
        self._sync_box()

        self._move_line(place_tool0_x_pre, place_y, lift_z,
                        place_tool0_x, place_y, lift_z,
                        label=f'{box_name} place-in', secs=6, steps=15)

        self._spin_sync(2.0)  # 静止后同步
        self.get_logger().info(f'--- {box_name} DETACH ---')
        self._detach_box_kinematic(box_name)
        self.get_logger().info(f'--- {box_name} gripper open ---')
        self.gripper(0.0, secs=2.0)
        self.get_logger().info(f'📦 {box_name} placed at ({place_x:.2f}, {place_y:.2f})')

        # 退出
        self._move_line(place_tool0_x, place_y, lift_z,
                        place_tool0_x_pre, place_y, lift_z,
                        label=f'{box_name} retreat', secs=6, steps=15)

        self.get_logger().info(f'=== {box_name} DONE ===')
        return True

    # ═══════════════════════════════════════════════════════════════════
    # 单次抓取放置 — 夹爪在盒子正上方，垂直下降抓取
    # ═══════════════════════════════════════════════════════════════════
    def _pick_and_place_one(self, box_name, pick_x, pick_y, pick_z,
                            place_x, place_y, place_z):
        """
        夹爪水平朝前，从盒子正上方垂直下降抓取：
          Step 1: 移到盒子正上方 -> (pick_x, pick_y, pick_z + ABOVE_Z)
          Step 2: 垂直下降 ->       (pick_x, pick_y, pick_z + GRASP_Z_OFF)
          Step 3: 夹爪闭合 + 提起
        """
        self.get_logger().info(
            f'=== {box_name}: pick ({pick_x:.2f},{pick_y:.2f}) → '
            f'place ({place_x:.2f},{place_y:.2f}) ===')

        # tool0 往后偏移手指长度，让夹爪末端处于盒子正上方
        tool0_x = pick_x - PRE_DIST
        above_z = pick_z + ABOVE_Z     # 预抓取高处 Z
        grasp_z = pick_z + GRASP_Z_OFF  # 抓取 Z: 与盒子同高（水平抓取）
        lift_z  = pick_z + LIFT_Z       # 提起后 Z

        # ── Step 1: 夹爪末端移到盒子正上方 ────────────────────────────
        self.get_logger().info(
            f'--- {box_name} Step1: above ({tool0_x:.3f}, {pick_y:.3f}, {above_z:.3f}) ---')
        if not self._move_to_world_pose(tool0_x, pick_y, above_z,
                                        label=f'{box_name} step1-above', secs=5):
            return False

        # ── Step 2: 垂直下降，夹爪包住盒子 ────────────────────────────
        self.get_logger().info(
            f'--- {box_name} Step2: descend ({tool0_x:.3f}, {pick_y:.3f}, {grasp_z:.3f}) ---')
        if not self._move_to_world_pose(tool0_x, pick_y, grasp_z,
                                        label=f'{box_name} step2-descend', secs=4):
            self._move_to_world_pose(tool0_x, pick_y, above_z,
                                     label=f'{box_name} abort-above', secs=4)
            return False

        self._spin(2.0)  # 等机械臂完全静止再闭合夹爪

        # ── Step 3: 夹爪闭合 + 抓取验证（带重试）─────────────────
        GRIP_POSITIONS = [GRIP_CLOSE_PRIMARY, GRIP_CLOSE_FALLBACK]
        grasp_success = False
        for grip_attempt in range(3):
            grip_pos = GRIP_POSITIONS[min(grip_attempt, len(GRIP_POSITIONS) - 1)]
            self.get_logger().info(
                f'--- {box_name} Step3: close gripper to {grip_pos} '
                f'(attempt {grip_attempt + 1}/3) ---')
            self.gripper(grip_pos, secs=2.0)
            self._spin(0.8)

            if self._verify_grasp(box_name, [pick_x, pick_y, pick_z]):
                self.get_logger().info(f'--- {box_name} lift ---')
                self._move_to_world_pose(tool0_x, pick_y, lift_z,
                                         label=f'{box_name} lift', secs=3)

                if self._verify_lift(box_name, pick_z):
                    self.get_logger().info(f'✅ {box_name} lifted!')
                    grasp_success = True
                    break
                else:
                    self.get_logger().warn(f'{box_name} lift FAILED — retrying')
                    self.gripper(0.0, secs=2.0)
                    self._move_to_world_pose(tool0_x, pick_y, above_z,
                                             label=f'{box_name} retry-above', secs=3)
                    self._move_to_world_pose(tool0_x, pick_y, grasp_z,
                                             label=f'{box_name} retry-descend', secs=3)
            else:
                self.get_logger().warn(f'{box_name} box disturbed — retrying')
                self.gripper(0.0, secs=2.0)
                self._move_to_world_pose(tool0_x, pick_y, above_z,
                                         label=f'{box_name} retry-above', secs=4)
                self._move_to_world_pose(tool0_x, pick_y, grasp_z,
                                         label=f'{box_name} retry-descend', secs=3)

        if not grasp_success:
            self.get_logger().error(f'❌ {box_name} GRASP FAILED after 3 attempts')
            self.gripper(0.0, secs=2.0)
            self._move_to_world_pose(tool0_x, pick_y, above_z,
                                     label=f'{box_name} final-above', secs=4)
            return False

        self.get_logger().info(f'>>> {box_name} physics grip active <<<')
        self._spin(0.3)

        # ── 搬运到红区 ──────────────────────────────────────────
        self.get_logger().info(f'--- {box_name} transport to red zone ---')
        place_tool0_x = place_x - PRE_DIST
        self._move_to_world_pose(place_tool0_x, place_y, lift_z,
                                 label=f'{box_name} to-red-above', secs=4)

        self.get_logger().info(f'--- {box_name} place at red zone ---')
        place_grasp_z = place_z + GRASP_Z_OFF
        self._move_to_world_pose(place_tool0_x, place_y, place_grasp_z,
                                 label=f'{box_name} place-descend', secs=3)

        # 松开夹爪（等机械臂完全静止）
        self.get_logger().info(f'--- {box_name} gripper open ---')
        self._spin(2.0)
        self.gripper(0.0, secs=2)
        self.get_logger().info(f'📦 {box_name} placed at red zone ({place_x:.2f}, {place_y:.2f})')

        # 升高离开
        self.get_logger().info(f'--- {box_name} retreat ---')
        self._move_to_world_pose(place_tool0_x, place_y, lift_z,
                                 label=f'{box_name} retreat', secs=3)

        self.get_logger().info(f'=== {box_name} DONE ===')
        return True

    # ═══════════════════════════════════════════════════════════════════
    # 主流程
    # ═══════════════════════════════════════════════════════════════════
    def run(self):
        self.get_logger().info('=== PICK AND PLACE — IKPy + Side-Approach Strategy ===')

        # 初始化
        self.gripper(0.0)
        self.get_logger().info('--- home ---')
        self.direct_joints(HOME_JOINTS, secs=5)

        # 硬编码位置 (SDF 定义)
        box_name = 'box_1'
        blue_pick = (0.55, 0.18, 0.925)    # 蓝区 box_1
        red_place = (0.979, -0.38, 0.925)  # 红区

        self.get_logger().info(
            f'Hardcoded positions: pick={blue_pick} place={red_place}')
        self._spin(0.5)

        # 使用前向笛卡尔直线推进方式
        # 如需垂直下降方式，改为 _pick_and_place_one
        self._pick_forward(box_name,
                           blue_pick[0], blue_pick[1], blue_pick[2],
                           red_place[0], red_place[1], red_place[2])

        self.get_logger().info('--- home-final ---')
        self.direct_joints(HOME_JOINTS, secs=5)
        self.get_logger().info('=== DONE ===')


def main():
    rclpy.init()
    node = PickAndPlace()
    node.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
