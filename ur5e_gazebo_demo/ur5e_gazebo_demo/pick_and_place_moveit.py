#!/usr/bin/env python3
"""
UR5e Pick-and-Place — 视觉引导抓取 + 放置。

工作流程：
  相机(2.8m高垂直向下) → aruco_detect(像素→世界XY)
  → object_pose_estimate(相机系→世界系) → /object_pose_{0,1,2}
  → pick_and_place(订阅 XY + 固定 Z=0.925)

桌子 2.5m×2.0m：
  Y>0 → 蓝色区域（抓取）    Y<0 → 红色区域（放置）

运动方式：
  IKPy 逆运动学求解关节角 → 关节插值 / 笛卡尔直线插值
  盒子跟随：ROS2 Timer 50ms 将盒子传送到 tool0(TCP)+offset
"""
import os
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── NumPy 2.x + SymPy 兼容 ─────────────────────────────────────────────
import sympy.core.numbers as scn

def _fixed_convert_numpy_types(a, **sympify_args):
    import numpy as _np
    if not isinstance(a, _np.floating):
        if _np.iscomplex(a):
            return scn._sympy_converter[complex](a.item())
        from sympy import sympify
        return sympify(a.item(), **sympify_args)
    from sympy.core.numbers import Float
    return Float(float(a), precision=53)

scn._convert_numpy_types = _fixed_convert_numpy_types

import ikpy.chain
from ament_index_python.packages import get_package_share_directory

# ── IKPy 运动学链 ──────────────────────────────────────────────────────
_CHAIN = ikpy.chain.Chain.from_urdf_file(
    os.path.join(get_package_share_directory('ur5e_gazebo_demo'),
                 'urdf', 'ur5e_full.urdf'),
    base_elements=['world'])

# Chain: [0]world [1]base_joint [2]inertia [3]shoulder_pan [4]shoulder_lift
#        [5]elbow  [6]wrist_1  [7]wrist_2  [8]wrist_3  [9]ft_frame
_IDX_REVOLUTE = [3, 4, 5, 6, 7, 8]

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint',      'wrist_2_joint',       'wrist_3_joint',
]
HOME_JOINTS = [0.0, -1.5707, 0.0, -1.5707, 0.0, 0.0]
_TARGET_Z_AXIS = [-1.0, 0.0, 0.0]  # 末端 Z 轴指向世界 +X


def _to_chain(q_6dof):
    full = [0.0] * 10
    for idx, val in zip(_IDX_REVOLUTE, q_6dof):
        full[idx] = val
    return full

def _from_chain(full):
    return [full[i] for i in _IDX_REVOLUTE]

def ik_solve(wx, wy, wz, seed_6dof):
    """IK：世界坐标 → 6-DOF 关节角（wrist_3 锁定保持水平）。"""
    seed_full = _to_chain(seed_6dof)
    result_full = _CHAIN.inverse_kinematics(
        target_position=[wx, wy, wz],
        target_orientation=_TARGET_Z_AXIS,
        initial_position=seed_full,
        orientation_mode='Z')
    result_6dof = _from_chain(result_full)
    result_6dof[5] = seed_6dof[5]
    return [(np.pi + q) % (2 * np.pi) - np.pi for q in result_6dof]


# ── 抓取几何参数 ────────────────────────────────────────────────────────
# 坐标系：世界 +X 前方（夹爪朝向），+Y 左侧，+Z 上方
PRE_DIST         = 0.15   # tool0 到手指末端距离
FORWARD_PRE_DIST = 0.23   # 笛卡尔推进抓取距离
GRASP_Z_OFF      = 0.05   # tool0 高于盒子中心
LIFT_Z           = 0.20   # 抓取后提起
SAFE_Z           = 0.45   # 搬运安全高度

GRIP_PRIMARY  = 0.25      # 夹爪闭合（主）
GRIP_FALLBACK = 0.22      # 夹爪闭合（备用）
FIXED_Z       = 0.925     # 盒子 Z（视觉只提供 XY）
VISION_TIMEOUT = 5.0      # 等待视觉超时

FALLBACK_BOXES = [
    ('box_1', 0.672, 0.18, FIXED_Z),
    ('box_2', 0.831, 0.38, FIXED_Z),
    ('box_3', 0.66,  0.58, FIXED_Z),
]

ARRIVAL_TOL     = 0.05    # 关节到位容差 (rad)
ARRIVAL_TIMEOUT = 10.0    # 到位超时 (s)


class PickAndPlace(Node):
    """视觉引导抓取放置节点。"""

    def __init__(self):
        super().__init__('pick_and_place_moveit')

        self._current_joints = HOME_JOINTS[:]
        self._attached_box = None
        self._attach_offset = None
        self._vision_xy = {0: None, 1: None, 2: None}

        # 关节状态 + 控制指令
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # Timer 驱动盒子跟随 tool0(TCP)：50Hz 传送
        self._sync_timer = self.create_timer(0.05, self._sync_box)

        # 订阅视觉位姿（只取 XY）
        for mid in [0, 1, 2]:
            self.create_subscription(
                PoseStamped, f'/object_pose_{mid}',
                lambda msg, m=mid: self._pose_cb(m, msg), 10)

        # 等待 joint_states 初始值
        deadline = time.time() + 30.0
        while self._current_joints is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.3)
        if self._current_joints is None:
            self._current_joints = HOME_JOINTS[:]

    # ═══════════════════════════════════════════════════════════════════
    # 回调
    # ═══════════════════════════════════════════════════════════════════
    def _js_cb(self, msg: JointState):
        m = dict(zip(msg.name, msg.position))
        if all(n in m for n in JOINT_NAMES):
            self._current_joints = [m[n] for n in JOINT_NAMES]

    def _pose_cb(self, marker_id, msg: PoseStamped):
        self._vision_xy[marker_id] = (msg.pose.position.x, msg.pose.position.y)

    @property
    def current_joints(self):
        return list(self._current_joints) if self._current_joints else HOME_JOINTS[:]

    # ═══════════════════════════════════════════════════════════════════
    # 视觉融合：订阅位姿 vs 硬编码回退
    # ═══════════════════════════════════════════════════════════════════
    def _get_vision_boxes(self):
        """轮询视觉位姿，超时用硬编码回退。返回 [(name, x, y, z), ...]"""
        deadline = time.time() + VISION_TIMEOUT
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if all(v is not None for v in self._vision_xy.values()):
                break

        self.get_logger().info('─── 视觉 vs 硬编码 坐标对比 ───')
        boxes = []
        for mid in [0, 1, 2]:
            xy = self._vision_xy[mid]
            hx, hy = FALLBACK_BOXES[mid][1], FALLBACK_BOXES[mid][2]
            if xy is not None:
                bx, by = xy
                self.get_logger().info(
                    f'box_{mid+1}: 视觉=({bx:.3f}, {by:.3f})  '
                    f'硬编码=({hx:.3f}, {hy:.3f})  '
                    f'ΔXY={np.hypot(bx-hx, by-hy)*100:.1f}cm')
            else:
                bx, by = hx, hy
                self.get_logger().warn(
                    f'box_{mid+1}: 视觉超时，使用硬编码=({bx:.3f}, {by:.3f})')
            boxes.append((f'box_{mid+1}', bx, by, FIXED_Z))
        return boxes

    # ═══════════════════════════════════════════════════════════════════
    # 运动控制底层
    # ═══════════════════════════════════════════════════════════════════
    def _spin(self, secs):
        """等待 secs 秒（处理 ROS2 回调和 Timer）。"""
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_arrival(self, target, timeout=ARRIVAL_TIMEOUT, tol=ARRIVAL_TOL):
        """等待关节到达目标位置。"""
        target = np.array(target, dtype=np.float64)
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if np.max(np.abs(np.array(self.current_joints) - target)) < tol:
                return True
        err = np.max(np.abs(np.array(self.current_joints) - target))
        self.get_logger().warn(f'Arrival timeout: max_err={err:.4f}')
        return False

    def _publish_trajectory(self, waypoints, secs):
        """发送关节轨迹并等待执行完成。"""
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
        """控制夹爪开合。pos=0 张开，pos=0.25 闭合。"""
        msg = JointTrajectory()
        msg.joint_names = ['robotiq_85_left_knuckle_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(pos)]
        pt.time_from_start.sec = secs
        pt.time_from_start.nanosec = 0
        msg.points.append(pt)
        self.gripper_pub.publish(msg)
        self._spin(secs + 0.3)

    # ═══════════════════════════════════════════════════════════════════
    # IK 运动：点到点 / 直线
    # ═══════════════════════════════════════════════════════════════════
    def _move_to(self, wx, wy, wz, label='', secs=5):
        """IK 求解 → 关节空间运动到世界坐标 (wx, wy, wz)。"""
        try:
            joints = ik_solve(wx, wy, wz, self.current_joints)
        except Exception as e:
            self.get_logger().error(f'[{label}] IK failed: {e}')
            return False
        self.get_logger().info(f'[{label}] → ({wx:.3f}, {wy:.3f}, {wz:.3f})')
        self.direct_joints(joints, secs=secs)
        return True

    def _move_line(self, x1, y1, z1, x2, y2, z2, label='', secs=6, steps=15):
        """笛卡尔直线插值（wrist_3 锁定保持水平）。"""
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

    # ═══════════════════════════════════════════════════════════════════
    # Gazebo 交互：位姿查询、盒子附着、删除
    # ═══════════════════════════════════════════════════════════════════
    def _get_gz_pose(self, model, link=None):
        """查询 Gazebo 模型/连杆位姿。返回 [x, y, z] 或 None。"""
        args = ['gz', 'model', '-m', model]
        if link:
            args += ['-l', link]
        args += ['--pose']
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=3)
            poses = []
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    nums = [float(x) for x in line[1:-1].split()]
                    if len(nums) == 3:
                        poses.append(nums)
            if link and len(poses) >= 2:
                return poses[-2]   # 连杆位姿（倒数第2个）
            return poses[-1] if poses else None
        except Exception:
            return None

    def _attach(self, box_name):
        """附着盒子：记录 box 相对 tool0(TCP) 的偏移，Timer 自动同步。"""
        box = self._get_gz_pose(box_name)
        arm = self._get_gz_pose('ur5e', 'tool0')
        if not box or not arm:
            self.get_logger().error(f'Attach {box_name}: cannot get poses')
            return False
        self._attach_offset = [box[i] - arm[i] for i in range(3)]
        self._attached_box = box_name
        self.get_logger().info(
            f'Attached {box_name} → tool0 (TCP) '
            f'offset={[round(v, 3) for v in self._attach_offset]}')
        return True

    def _detach(self, box_name):
        """解除附着。"""
        self._attached_box = None
        self._attach_offset = None
        self.get_logger().info(f'Detached {box_name} from tool0')

    def _delete_box(self, box_name):
        """从 Gazebo 中删除盒子模型。"""
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
        """Timer 回调(50Hz)：传送盒子到 tool0(TCP) + offset。"""
        if not self._attached_box or not self._attach_offset:
            return
        arm = self._get_gz_pose('ur5e', 'tool0')
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

    # ═══════════════════════════════════════════════════════════════════
    # 抓取与搬运
    # ═══════════════════════════════════════════════════════════════════
    def _try_grasp(self, box_name, pick_x, pick_y, pick_z,
                   grasp_x, grasp_y, grasp_z, pre_x, pre_y, pre_z):
        """闭合夹爪 + 附着盒子。2 级力度重试。"""
        grips = [GRIP_PRIMARY, GRIP_FALLBACK]
        for attempt in range(2):
            gp = grips[attempt]
            self.get_logger().info(f'  grip {gp} (attempt {attempt+1}/2)')
            self.gripper(gp, secs=2.0)
            self._spin(0.5)

            if self._attach(box_name):
                self._move_to(grasp_x, grasp_y, pick_z + LIFT_Z,
                             label=f'{box_name} lift', secs=3)
                self.get_logger().info(f'✅ {box_name} grasped!')
                return True

            self.get_logger().warn(f'{box_name} attach failed — retrying')
            self.gripper(0.0, secs=2.0)
        return False

    def _transport_to_place(self, box_name, from_x, from_y, from_z,
                            place_x, place_y, place_z):
        """搬运盒子到放置点（Timer 自动同步盒子位置）。

        动作序列：后退 → 抬升 → 平移 → 推进
        """
        place_tool = place_x - PRE_DIST
        place_pre  = place_tool - FORWARD_PRE_DIST
        safe_z = from_z + SAFE_Z

        self._move_line(from_x, from_y, from_z,
                       from_x - FORWARD_PRE_DIST, from_y, from_z,
                       label=f'{box_name} retreat', secs=4, steps=10)
        self._move_to(from_x - FORWARD_PRE_DIST, from_y, safe_z,
                     label=f'{box_name} rise', secs=4)
        self._move_to(place_pre, place_y, safe_z,
                     label=f'{box_name} to-place', secs=5)
        self._move_line(place_pre, place_y, safe_z,
                       place_tool, place_y, safe_z,
                       label=f'{box_name} place-in', secs=6, steps=15)

    # ═══════════════════════════════════════════════════════════════════
    # 单次抓取 + 放置流程
    # ═══════════════════════════════════════════════════════════════════
    def _pick_forward(self, box_name, pick_x, pick_y, pick_z,
                      place_x, place_y, place_z):
        """前向抓取单个盒子。

        抓取方向：从 +X 方向（前方）水平推进夹爪到盒子位置。
        关键位姿：
          pick    — 盒子中心
          grasp   — tool0 在盒子中心 (pick_x - PRE_DIST)
          pre     — 预抓取位, grasp 前方 FORWARD_PRE_DIST
        """
        self.get_logger().info(
            f'=== {box_name} pick=({pick_x:.2f},{pick_y:.2f}) → '
            f'place=({place_x:.2f},{place_y:.2f}) ===')

        grasp_x = pick_x - PRE_DIST
        pre_x   = grasp_x - FORWARD_PRE_DIST
        tool_z  = pick_z + GRASP_Z_OFF

        # 1. 预抓取位姿
        if not self._move_to(pre_x, pick_y, tool_z,
                            label=f'{box_name} pre-grasp', secs=5):
            return False
        # 2. 直线推进
        if not self._move_line(pre_x, pick_y, tool_z,
                               grasp_x, pick_y, tool_z,
                               label=f'{box_name} grasp', secs=6, steps=15):
            self._move_to(pre_x, pick_y, tool_z,
                         label=f'{box_name} abort', secs=4)
            return False
        self._spin(2.0)
        # 3. 抓取
        if not self._try_grasp(box_name, pick_x, pick_y, pick_z,
                               grasp_x, pick_y, tool_z, pre_x, pick_y, tool_z):
            self.get_logger().error(f'❌ {box_name} GRASP FAILED')
            self.gripper(0.0, secs=2.0)
            self._move_to(pre_x, pick_y, tool_z,
                         label=f'{box_name} abort', secs=4)
            return False
        # 4. 搬运 + 放置
        self._transport_to_place(box_name, grasp_x, pick_y, tool_z,
                                place_x, place_y, pick_z)
        self._spin(2.0)
        self._detach(box_name)
        self.gripper(0.0, secs=2.0)
        self._delete_box(box_name)
        self.get_logger().info(
            f'📦 {box_name} placed at ({place_x:.2f}, {place_y:.2f})')
        # 5. 退出放置区
        self._move_line(place_x - PRE_DIST, place_y, pick_z + SAFE_Z,
                       place_x - PRE_DIST - FORWARD_PRE_DIST, place_y,
                       pick_z + SAFE_Z,
                       label=f'{box_name} exit', secs=4, steps=10)
        self._move_to(place_x - PRE_DIST - FORWARD_PRE_DIST, place_y,
                     pick_z + LIFT_Z, label=f'{box_name} done', secs=4)
        return True

    # ═══════════════════════════════════════════════════════════════════
    # 主流程
    # ═══════════════════════════════════════════════════════════════════
    def run(self):
        self.get_logger().info('=== PICK AND PLACE (VISION) ===')
        self.gripper(0.0)
        self.direct_joints(HOME_JOINTS, secs=5)

        boxes = self._get_vision_boxes()
        red = (0.755, -0.38, FIXED_Z)

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
