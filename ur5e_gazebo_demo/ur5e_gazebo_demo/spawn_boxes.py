#!/usr/bin/env python3
"""在蓝色区域内随机生成 3 个盒子位置并 spawn 到 Gazebo。"""

import json
import os
import random
import subprocess
import sys
import time


# ═══════════════════════════════════════════════════════════════════════════
# 蓝色区域参数 (与 camera_world.sdf 中 blue_zone 一致)
# ═══════════════════════════════════════════════════════════════════════════
BLUE_CENTER_X = 0.746
BLUE_CENTER_Y = 0.38
BLUE_HALF_X  = 0.40   # 0.80m / 2
BLUE_HALF_Y  = 0.35   # 0.70m / 2
BOX_HALF     = 0.03   # 盒子半宽，边距
BOX_Z        = 0.925  # 桌面 + 半个盒子高度

MIN_DIST      = 0.15  # 两两最小中心距离 (0.06 盒子宽 + 0.09 安全间距)
MAX_ATTEMPTS  = 200   # 随机生成最大重试次数

BOX_NAMES = ['box_1', 'box_2', 'box_3']
OUTPUT_FILE = '/tmp/box_positions.json'


def random_position():
    """在蓝色区域内生成随机 (x, y)。"""
    x_min = BLUE_CENTER_X - BLUE_HALF_X + BOX_HALF
    x_max = BLUE_CENTER_X + BLUE_HALF_X - BOX_HALF
    y_min = BLUE_CENTER_Y - BLUE_HALF_Y + BOX_HALF
    y_max = BLUE_CENTER_Y + BLUE_HALF_Y - BOX_HALF
    return random.uniform(x_min, x_max), random.uniform(y_min, y_max)


def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def generate_positions():
    """生成 3 个不重叠的位置。"""
    for _ in range(MAX_ATTEMPTS):
        positions = []
        for _ in range(3):
            for __ in range(MAX_ATTEMPTS):
                x, y = random_position()
                if all(distance((x, y), p) >= MIN_DIST for p in positions):
                    positions.append((x, y))
                    break
        if len(positions) == 3:
            return positions
    raise RuntimeError(f'Failed to generate 3 non-overlapping positions '
                       f'after {MAX_ATTEMPTS} attempts')


def get_model_path(box_name):
    """获取盒子 SDF 模型文件路径。"""
    from ament_index_python.packages import get_package_share_directory
    pkg_share = get_package_share_directory('ur5e_gazebo_demo')
    return os.path.join(pkg_share, 'models', box_name, 'model.sdf')


def list_models():
    """列出 Gazebo 中所有模型名。"""
    try:
        r = subprocess.run(['gz', 'model', '--list'], capture_output=True,
                          text=True, timeout=5)
        models = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith('- '):
                models.append(line[2:].strip())
        return models
    except Exception:
        return []


def spawn_box(box_name, x, y, z, retries=3):
    """通过 ros2 run ros_gz_sim create 直接 spawn 盒子。"""
    model_file = get_model_path(box_name)
    if not os.path.isfile(model_file):
        print(f'[ERROR] Model file not found: {model_file}', file=sys.stderr)
        return False

    cmd = [
        'ros2', 'run', 'ros_gz_sim', 'create',
        '-world', 'camera_world',
        '-file', model_file,
        '-name', box_name,
        '-allow_renaming', 'true',
        '-x', str(x), '-y', str(y), '-z', str(z),
    ]

    for attempt in range(1, retries + 1):
        # 先检查是不是已经存在了
        if box_name in list_models():
            print(f'  ✓ {box_name} already in world (skip spawn)')
            return True

        print(f'  Spawning {box_name} at ({x:.3f}, {y:.3f}, {z:.3f}) '
              f'(attempt {attempt}/{retries})')
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            # ros_gz_sim create 成功时 stdout 输出 "Entity creation successful"
            if 'successful' in stdout.lower() or result.returncode == 0:
                print(f'  ✓ {box_name} created')
                # 额外验证
                if box_name in list_models():
                    return True
                print(f'  ⚠ {box_name} spawned but not in model list yet, '
                      f'retrying...')
            else:
                print(f'  ⚠ {box_name} create: '
                      f'stdout={stdout} stderr={stderr}')
        except subprocess.TimeoutExpired:
            print(f'  ⚠ {box_name} spawn timed out (attempt {attempt})')
        except FileNotFoundError:
            print(f'  ✗ ros2 not found — is the ROS2 environment sourced?',
                  file=sys.stderr)
            return False

        time.sleep(1.5)

    print(f'  ✗ {box_name} spawn FAILED after {retries} attempts',
          file=sys.stderr)
    return False


def main():
    print('=== Spawn Boxes — Random Positions in Blue Zone ===')

    # 1. 生成随机位置
    positions = generate_positions()
    print(f'Generated {len(positions)} positions:')
    for i, (x, y) in enumerate(positions):
        print(f'  {BOX_NAMES[i]}: ({x:.3f}, {y:.3f}, {BOX_Z:.3f})')
        for j in range(i):
            d = distance(positions[i], positions[j])
            print(f'    ↳ dist to {BOX_NAMES[j]}: {d:.3f}m')

    # 2. 逐个 spawn 盒子
    box_data = {}
    for i, box_name in enumerate(BOX_NAMES):
        x, y = positions[i]
        success = spawn_box(box_name, x, y, BOX_Z)
        if not success:
            print(f'FATAL: Failed to spawn {box_name}', file=sys.stderr)
            sys.exit(1)
        box_data[box_name] = {'x': x, 'y': y, 'z': BOX_Z}
        time.sleep(0.5)

    # 3. 最终验证
    all_models = list_models()
    spawned = [bn for bn in BOX_NAMES if bn in all_models]
    missing = [bn for bn in BOX_NAMES if bn not in all_models]
    print(f'Model list check: {len(spawned)}/3 boxes in Gazebo')
    print(f'  ✅ Present: {spawned}')
    if missing:
        print(f'  ❌ Missing: {missing}', file=sys.stderr)
        sys.exit(1)

    # 4. 保存位置
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(box_data, f, indent=2)
    print(f'Positions saved to {OUTPUT_FILE}')
    print('=== Spawn Complete ===')


if __name__ == '__main__':
    main()
