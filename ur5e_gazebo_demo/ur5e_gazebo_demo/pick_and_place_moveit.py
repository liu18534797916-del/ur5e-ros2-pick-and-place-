#!/usr/bin/env python3
"""
Pick and Place — IKPy 逆运动学 + 前向水平推进抓取。

桌子 2.5m × 2.0m，前方 +X：
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
# Fix sympy for NumPy 2.x before importing ikpy
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
# IKPy chain
# ═══════════════════════════════════════════════════════════════════════════
from ament_index_python.packages import get_package_share_directory

_CHAIN = ikpy.chain.Chain.from_urdf_file(
    os.path.join(get_package_share_directory('ur5e_gazebo_demo'),
                 'urdf', 'ur5e_full.urdf'),
    base_elements=['world'])

# Chain indices: [0]=world [1]=base_joint [2]=inertia
#                [3]=shoulder_pan [4]=shoulder_lift [5]=elbow
#                [6]=wrist_1 [7]=wrist_2 [8]=wrist_3 [9]=ft_frame
_IDX_REVOLUTE = [3, 4, 5, 6, 7, 8]

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint',      'wrist_2_joint',       'wrist_3_joint',
]
HOME_JOINTS = [0.0, -1.5707, 0.0, -1.5707, 0.0, 0.0]

# End-effector Z axis in world frame: ft_frame Z = world -X → fingers = world +X
_TARGET_Z_AXIS = [-1.0, 0.0, 0.0]

# ═══════════════════════════════════════════════════════════════════════════
# IK 工具函数
# ═══════════════════════════════════════════════════════════════════════════
def _to_chain(q_6dof):
    full = [0.0] * 10
    for idx, val in zip(_IDX_REVOLUTE, q_6dof):
        full[idx] = val
    return full

def _from_chain(full):
    return [full[i] for i in _IDX_REVOLUTE]

def ik_solve(wx, wy, wz, seed_6dof):
    """IK: 世界坐标 → 6-DOF 关节角，wrist_3 锁定保持水平。"""
    seed_full = _to_chain(seed_6dof)
    result_full = _CHAIN.inverse_kinematics(
        target_position=[wx, wy, wz],
        target_orientation=_TARGET_Z_AXIS,
        initial_position=seed_full,
        orientation_mode='Z')
    result_6dof = _from_chain(result_full)
    result_6dof[5] = seed_6dof[5]
    return [(np.pi + q) % (2 * np.pi) - np.pi for q in result_6dof]

# ═══════════════════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════════════════
PRE_DIST         = 0.15   # tool0 → 手指末端距离（减小3cm，tool0更深入盒子）
FORWARD_PRE_DIST = 0.23   # 前方预抓取距离
GRASP_Z_OFF      = 0.05   # tool0 高于盒子中心高度
LIFT_Z           = 0.20   # 抓取后提起高度
SAFE_Z           = 0.45   # 搬运安全高度（高于盒顶）
SAFE_APPROACH_Z  = 0.50   # 避障过渡高度

BOX_W         = 0.06      # 盒子宽度 (m)
# Robotiq 85 行程 0.085m，手指偏移 ≈ 0.0425m/侧
# 闭合到手指间距 = BOX_W → joint = 0.8 * (1 - BOX_W/2 / 0.0425) ≈ 0.235
GRIP_PRIMARY  = 0.25      # 夹爪闭合（主，预留盒子宽度）
GRIP_FALLBACK = 0.22      # 夹爪闭合（备用，稍松）

GRASP_TOL_XY  = 0.06      # 盒子位移容差 (m)
GRASP_TOL_DZ  = -0.12     # 盒子下落容差 (m)
ARRIVAL_TOL   = 0.05      # 关节到位容差 (rad)
ARRIVAL_TIMEOUT = 10.0    # 到位超时 (s)

# ═══════════════════════════════════════════════════════════════════════════
# PickAndPlace Node
# ═══════════════════════════════════════════════════════════════════════════
class PickAndPlace(Node):

    def __init__(self):
        super().__init__('pick_and_place_moveit')

        self._current_joints = HOME_JOINTS[:]
        self._attached_box = None
        self._attach_offset = None

        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # 等 joint_states
        deadline = time.time() + 30.0
        while self._current_joints is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.3)
        if self._current_joints is None:
            self._current_joints = HOME_JOINTS[:]

    # ── 回调 ──────────────────────────────────────────────────────────
    def _js_cb(self, msg: JointState):
        m = dict(zip(msg.name, msg.position))
        if all(n in m for n in JOINT_NAMES):
            self._current_joints = [m[n] for n in JOINT_NAMES]

    def _spin(self, secs):
        """等待指定秒数，搬运期间同步盒子位置。"""
        end = time.time() + secs
        last_sync = time.time()
        while time.time() < end:
            if self._attached_box and time.time() - last_sync > 0.15:
                self._sync_box()
                last_sync = time.time()
            rclpy.spin_once(self, timeout_sec=0.05)

    @property
    def current_joints(self):
        return (list(self._current_joints) if self._current_joints
                else HOME_JOINTS[:])

    # ── 运动控制 ──────────────────────────────────────────────────────
    def _wait_arrival(self, target_positions, timeout=ARRIVAL_TIMEOUT,
                      tolerance=ARRIVAL_TOL):
        """等待关节到达目标，超时打 warning。搬运期间持续同步盒子。"""
        target = np.array(target_positions, dtype=np.float64)
        deadline = time.time() + timeout
        last_sync = time.time()
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._attached_box and time.time() - last_sync > 0.15:
                self._sync_box()
                last_sync = time.time()
            if np.max(np.abs(np.array(self.current_joints) - target)) < tolerance:
                return True
        err = np.max(np.abs(np.array(self.current_joints) - target))
        self.get_logger().warn(f'Arrival timeout: max_err={err:.4f}')
        return False

    def _publish_trajectory(self, waypoints, secs):
        """构建并发送关节轨迹，等执行完再验证到位。返回是否到位。"""
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        n = len(waypoints)
        for k, joints in enumerate(waypoints):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in joints]
            t = secs * k / max(n - 1, 1)
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            msg.points.append(pt)
        self.arm_pub.publish(msg)
        self._spin(secs)
        return self._wait_arrival(waypoints[-1])

    def direct_joints(self, positions, secs=5, steps=20):
        """关节空间平滑插值运动。"""
        start = np.array(self.current_joints)
        end = np.array(positions, dtype=np.float64)
        waypoints = [start + (k / max(steps - 1, 1)) * (end - start)
                     for k in range(steps)]
        self._publish_trajectory(waypoints, secs)

    def gripper(self, pos, secs=2):
        msg = JointTrajectory()
        msg.joint_names = ['robotiq_85_left_knuckle_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(pos)]
        t = secs
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
        msg.points.append(pt)
        self.gripper_pub.publish(msg)
        self._spin(secs + 0.3)

    # ── IK 运动 ──────────────────────────────────────────────────────
    def _move_to(self, wx, wy, wz, label='', secs=5):
        """IK → 关节空间运动。"""
        try:
            joints = ik_solve(wx, wy, wz, self.current_joints)
        except Exception as e:
            self.get_logger().error(f'[{label}] IK failed: {e}')
            return False
        self.get_logger().info(
            f'[{label}] → ({wx:.3f}, {wy:.3f}, {wz:.3f})')
        self.direct_joints(joints, secs=secs)
        return True

    def _move_line(self, x1, y1, z1, x2, y2, z2, label='', secs=6, steps=15):
        """笛卡尔直线插值（wrist_3 锁定）。"""
        seed = self.current_joints
        fixed_wrist3 = seed[5]
        self.get_logger().info(
            f'[{label}] line ({x1:.3f},{y1:.3f},{z1:.3f}) → '
            f'({x2:.3f},{y2:.3f},{z2:.3f})')

        waypoints = [seed]
        for k in range(steps):
            alpha = (k + 1) / steps
            try:
                j = ik_solve(x1 + alpha * (x2 - x1),
                             y1 + alpha * (y2 - y1),
                             z1 + alpha * (z2 - z1), seed)
                j[5] = fixed_wrist3
                waypoints.append(j)
                seed = j
            except Exception as e:
                self.get_logger().error(f'[{label}] IK step {k}: {e}')
                return False

        self._publish_trajectory(waypoints, secs)
        return True

    # ── Gazebo 工具 ──────────────────────────────────────────────────
    def _get_gz_pose(self, model, link=None):
        """查询模型/连杆在 Gazebo 中的位姿。"""
        args = ['gz', 'model', '-m', model, '--pose']
        if link:
            args = ['gz', 'model', '-m', model, '-l', link, '--pose']
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=3)
            poses = []
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    nums = [float(x) for x in line[1:-1].split()]
                    if len(nums) == 3:
                        poses.append(nums)
            # 有 link 时取倒数第2个（连杆位姿），否则最后1个（模型位姿）
            if link and len(poses) >= 2:
                return poses[-2]
            return poses[-1] if poses else None
        except Exception:
            return None

    def _attach(self, box_name):
        box = self._get_gz_pose(box_name)
        arm = self._get_gz_pose('ur5e', 'wrist_3_link')
        if not box or not arm:
            return False
        self._attach_offset = [box[i] - arm[i] for i in range(3)]
        self._attached_box = box_name
        self.get_logger().info(
            f'Attached {box_name}: offset={[round(v, 3) for v in self._attach_offset]}')
        return True

    def _detach(self, box_name):
        self._attached_box = None
        self._attach_offset = None
        self.get_logger().info(f'Detached {box_name}')

    def _delete_box(self, box_name):
        try:
            r = subprocess.run(
                ['gz', 'service', '-s', '/world/camera_world/remove',
                 '--reqtype', 'gz.msgs.Entity',
                 '--reptype', 'gz.msgs.Boolean', '--timeout', '3000',
                 '-r', f'name: "{box_name}" type: MODEL'],
                capture_output=True, text=True, timeout=5)
            self.get_logger().info(f'Deleted {box_name}: {r.stdout.strip()}')
            return True
        except Exception as e:
            self.get_logger().warn(f'Delete {box_name} failed: {e}')
            return False

    def _sync_box(self):
        if not self._attached_box or not self._attach_offset:
            return
        arm = self._get_gz_pose('ur5e', 'wrist_3_link')
        if not arm:
            return
        bx, by, bz = [arm[i] + self._attach_offset[i] for i in range(3)]
        subprocess.run(
            ['gz', 'service', '-s', '/world/camera_world/set_pose',
             '--reqtype', 'gz.msgs.Pose',
             '--reptype', 'gz.msgs.Boolean', '--timeout', '2000',
             '-r', f'name: "{self._attached_box}" '
                   f'position: {{x: {bx:.6f} y: {by:.6f} z: {bz:.6f}}} '
                   f'orientation: {{x: 0.0 y: 0.0 z: 0.0 w: 1.0}}'],
            capture_output=True, timeout=3)

    # ── 抓取验证 ──────────────────────────────────────────────────────
    def _check_box_stable(self, box_name, orig):
        """检查盒子是否在原位（未被推走/碰倒）。"""
        for i in range(3):
            self._spin(0.3)
            pose = self._get_gz_pose(box_name)
            if not pose:
                continue
            dist = np.hypot(pose[0] - orig[0], pose[1] - orig[1])
            dz = pose[2] - orig[2]
            self.get_logger().info(
                f'  check #{i+1}: dist={dist:.3f}m dz={dz:.3f}m')
            if dz < GRASP_TOL_DZ:
                self.get_logger().warn(f'❌ {box_name} knocked over')
                return False
            if dist > GRASP_TOL_XY:
                self.get_logger().warn(f'❌ {box_name} pushed away '
                                       f'(dist={dist:.3f}m)')
                return False
            return True
        return True  # 3次都读到 None，可能已被 delete，认为 OK

    # ── 抓取重试循环 ────────────────────────────────────────────────
    def _try_grasp(self, box_name, pick_x, pick_y, pick_z,
                   grasp_x, grasp_y, grasp_z, pre_x, pre_y, pre_z):
        """闭合夹爪 + 附着盒子。最多 2 次重试。返回是否成功。"""
        grips = [GRIP_PRIMARY, GRIP_FALLBACK]
        for attempt in range(2):
            gp = grips[min(attempt, len(grips) - 1)]
            self.get_logger().info(
                f'  grip {gp} (attempt {attempt+1}/2)')
            self.gripper(gp, secs=2.0)
            self._spin(0.5)

            # 直接用附着抓取，不做严格位置检查
            if self._attach(box_name):
                self._move_to(grasp_x, grasp_y, pick_z + LIFT_Z,
                             label=f'{box_name} lift', secs=3)
                self._sync_box()
                self.get_logger().info(f'✅ {box_name} grasped!')
                return True

            self.get_logger().warn(f'{box_name} attach failed — retrying')
            self.gripper(0.0, secs=2.0)

        return False

    # ── 搬运到放置点 ────────────────────────────────────────────────
    def _transport_to_place(self, box_name, from_x, from_y, from_z,
                            place_x, place_y, place_z):
        """从抓取位姿搬运到放置点（搬运全程盒子附着跟随）。"""
        place_tool = place_x - PRE_DIST
        place_pre  = place_tool - FORWARD_PRE_DIST
        safe_z = from_z + SAFE_Z

        self._move_line(from_x, from_y, from_z,
                       from_x - FORWARD_PRE_DIST, from_y, from_z,
                       label=f'{box_name} retreat', secs=4, steps=10)
        self._sync_box()

        self._move_to(from_x - FORWARD_PRE_DIST, from_y, safe_z,
                     label=f'{box_name} rise', secs=4)
        self._sync_box()

        self._move_to(place_pre, place_y, safe_z,
                     label=f'{box_name} to-place', secs=5)
        self._sync_box()

        self._move_line(place_pre, place_y, safe_z,
                       place_tool, place_y, safe_z,
                       label=f'{box_name} place-in', secs=6, steps=15)
        self._sync_box()

    # ═══════════════════════════════════════════════════════════════════
    # 单次前向抓取 + 放置
    # ═══════════════════════════════════════════════════════════════════
    def _pick_forward(self, box_name, pick_x, pick_y, pick_z,
                      place_x, place_y, place_z):
        self.get_logger().info(
            f'=== {box_name} pick=({pick_x:.2f},{pick_y:.2f}) → '
            f'place=({place_x:.2f},{place_y:.2f}) ===')

        # 计算关键位姿
        grasp_x = pick_x - PRE_DIST
        pre_x   = grasp_x - FORWARD_PRE_DIST
        tool_z  = pick_z + GRASP_Z_OFF

        # Step 1: 移到盒子前方预抓取位姿
        if not self._move_to(pre_x, pick_y, tool_z,
                            label=f'{box_name} pre-grasp', secs=5):
            return False

        # Step 2: 直线推进
        if not self._move_line(pre_x, pick_y, tool_z,
                               grasp_x, pick_y, tool_z,
                               label=f'{box_name} grasp', secs=6, steps=15):
            self._move_to(pre_x, pick_y, tool_z,
                         label=f'{box_name} abort', secs=4)
            return False

        self._spin(2.0)

        # Step 3: 抓取 + 重试
        if not self._try_grasp(box_name, pick_x, pick_y, pick_z,
                               grasp_x, pick_y, tool_z,
                               pre_x, pick_y, tool_z):
            self.get_logger().error(f'❌ {box_name} GRASP FAILED')
            self.gripper(0.0, secs=2.0)
            self._move_to(pre_x, pick_y, tool_z,
                         label=f'{box_name} abort', secs=4)
            return False

        # Step 4: 搬运 + 放置
        self._transport_to_place(box_name, grasp_x, pick_y, tool_z,
                                place_x, place_y, pick_z)

        self._spin(2.0)
        self._detach(box_name)
        self.gripper(0.0, secs=2.0)
        self._delete_box(box_name)
        self.get_logger().info(f'📦 {box_name} placed at '
                               f'({place_x:.2f}, {place_y:.2f})')

        # 退出放置位
        self._move_line(place_x - PRE_DIST, place_y, pick_z + SAFE_Z,
                       place_x - PRE_DIST - FORWARD_PRE_DIST, place_y,
                       pick_z + SAFE_Z,
                       label=f'{box_name} exit', secs=4, steps=10)
        self._move_to(place_x - PRE_DIST - FORWARD_PRE_DIST, place_y,
                     pick_z + LIFT_Z,
                     label=f'{box_name} done', secs=4)
        return True

    # ═══════════════════════════════════════════════════════════════════
    # 主流程
    # ═══════════════════════════════════════════════════════════════════
    def run(self):
        self.get_logger().info('=== PICK AND PLACE ===')
        self.gripper(0.0)
        self.direct_joints(HOME_JOINTS, secs=5)

        boxes = [
            ('box_1', 0.672, 0.18, 0.925),
            ('box_2', 0.831, 0.38, 0.925),
            ('box_3', 0.66, 0.58, 0.925),
        ]
        red = (0.755, -0.38, 0.925)

        # 从近到远排序
        boxes.sort(key=lambda b: b[1]**2 + b[2]**2)
        self.get_logger().info(f'Pick order: {[b[0] for b in boxes]}')

        for name, px, py, pz in boxes:
            self._spin(0.5)
            self._pick_forward(name, px, py, pz, red[0], red[1], red[2])

        self.get_logger().info('--- home ---')
        self.direct_joints(HOME_JOINTS, secs=5)
        self.get_logger().info('=== DONE ===')


def main():
    rclpy.init()
    PickAndPlace().run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
