#!/usr/bin/env python3
"""
Camera calibration: use random box positions, record pixel→world mapping,
derive depth estimation formula.
"""
import subprocess, time, sys
import numpy as np
import cv2

CAM_FX, CAM_FY = 554.3, 554.3
CAM_CX, CAM_CY = 320.0, 240.0
CAM_X, CAM_Y, CAM_Z = 0.72, 0.0, 2.8
MARKER_SIZE = 0.045


def get_box_pose(box_name):
    """Get actual box position from Gazebo."""
    r = subprocess.run(['gz', 'model', '-m', box_name, '--pose'],
                       capture_output=True, text=True, timeout=3)
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            nums = [float(x) for x in line[1:-1].split()]
            if len(nums) >= 3:
                return nums[:3]
    return None


def capture_and_detect():
    """Capture frame, detect marker, return (cx, cy, area)."""
    r = subprocess.run(['gz', 'topic', '-e', '-t', '/top_camera/image', '-n', '1'],
                       capture_output=True, timeout=5)
    text = r.stdout.decode('utf-8', errors='replace')
    idx = text.find('data: "')
    if idx < 0: return None
    start = idx + len('data: "')
    end = text.rfind('"')
    data_str = text[start:end]
    raw = bytearray()
    i = 0
    while i < len(data_str):
        if data_str[i:i+2] == '\\\\' and i+5 <= len(data_str) and data_str[i+2:i+5].isdigit():
            raw.append(int(data_str[i+2:i+5], 8)); i += 5
        elif data_str[i] == '\\' and i+3 < len(data_str) and data_str[i+1:i+4].isdigit():
            raw.append(int(data_str[i+1:i+4], 8)); i += 4
        else: raw.append(ord(data_str[i])); i += 1
    raw = bytes(raw)
    if len(raw) < 640*480*3: return None
    img = np.frombuffer(raw[:640*480*3], dtype=np.uint8).reshape((480, 640, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for th in [140, 160, 180, 200]:
        _, t = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        c, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in c:
            a = cv2.contourArea(cnt)
            if 15 < a < 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                return (x + w/2, y + h/2, a)
    return None


def main():
    print("Moving boxes to random positions and recording data...")
    print(f"{'True X':>8} {'True Y':>8} {'True Z':>8} | "
          f"{'px X':>8} {'px Y':>8} {'Area':>8} | "
          f"{'est X':>8} {'est Y':>8} {'est Z':>8} {'err Z':>8}")
    print("-" * 100)
    data = []
    for trial in range(30):
        # Move box to new random position
        import random
        x = random.uniform(0.52, 0.88)
        y = random.uniform(0.13, 0.58)
        z = 0.925
        req = (f'name: "box_1" position '
               f'{{x: {x:.6f} y: {y:.6f} z: {z:.6f}}} '
               f'orientation {{w: 1.0}}')
        subprocess.run(['gz', 'service', '-s', '/world/camera_world/set_pose',
                        '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                        '--timeout', '5000', '--req', req],
                       capture_output=True, timeout=8)
        time.sleep(0.3)

        true_pos = get_box_pose('box_1')
        px = capture_and_detect()
        if true_pos is None or px is None:
            continue
        tx, ty, tz = true_pos
        cx, cy, area = px

        # Current estimation
        est_dist = CAM_FX * MARKER_SIZE / np.sqrt(area)
        est_x = CAM_X - (cx - CAM_CX) / CAM_FX * est_dist
        est_y = (cy - CAM_CY) / CAM_FY * est_dist
        est_z = CAM_Z - est_dist
        err_z = est_z - tz

        print(f"{tx:8.3f} {ty:8.3f} {tz:8.3f} | "
              f"{cx:8.1f} {cy:8.1f} {area:8.0f} | "
              f"{est_x:8.3f} {est_y:8.3f} {est_z:8.3f} {err_z:8.4f}")

        data.append({'true': (tx, ty, tz), 'pixel': (cx, cy, area),
                     'est': (est_x, est_y, est_z)})

        if len(data) >= 15:
            break

    if len(data) < 3:
        print("Not enough data!")
        return

    print(f"\n=== Collected {len(data)} data points ===")

    # Analyze Z estimation
    print("\n--- Z analysis ---")
    for d in data:
        tz = d['true'][2]
        area = d['pixel'][2]
        est_dist = CAM_FX * MARKER_SIZE / np.sqrt(area)
        actual_dist = CAM_Z - tz
        print(f"  area={area:6.0f}  est_dist={est_dist:.3f}  actual_dist={actual_dist:.3f}  "
              f"ratio={actual_dist/est_dist:.4f}")

    # Fit: actual_dist = k * FX * M / sqrt(area) = k * est_dist
    ratios = [(CAM_Z - d['true'][2]) / (CAM_FX * MARKER_SIZE / np.sqrt(d['pixel'][2]))
              for d in data]
    k_dist = np.mean(ratios)
    print(f"\n  Correction factor k_dist = {k_dist:.4f} (std={np.std(ratios):.4f})")

    # Fit X: true_x = a * est_x + b
    X = np.array([[d['est'][0], 1] for d in data])
    y_x = np.array([d['true'][0] for d in data])
    coeffs_x = np.linalg.lstsq(X, y_x, rcond=None)[0]

    # Fit Y: true_y = a * est_y + b
    Y = np.array([[d['est'][1], 1] for d in data])
    y_y = np.array([d['true'][1] for d in data])
    coeffs_y = np.linalg.lstsq(Y, y_y, rcond=None)[0]

    print(f"\n=== Calibrated Formula ===")
    print(f"  dist_corr = {k_dist:.4f}")
    print(f"  d = {k_dist:.4f} * {CAM_FX:.1f} * {MARKER_SIZE} / sqrt(area)")
    print(f"  wx = {coeffs_x[0]:.4f} * est_x + {coeffs_x[1]:.4f}")
    print(f"  wy = {coeffs_y[0]:.4f} * est_y + {coeffs_y[1]:.4f}")
    print(f"  wz = CAM_Z - d")
    print(f"where est_x = CAM_X - (cx-CX)/FX * d, est_y = (cy-CY)/FY * d")

    # Evaluate
    err_x = [d['true'][0] - (coeffs_x[0]*d['est'][0] + coeffs_x[1]) for d in data]
    err_y = [d['true'][1] - (coeffs_y[0]*d['est'][1] + coeffs_y[1]) for d in data]
    err_z = [d['true'][2] - (CAM_Z - k_dist * CAM_FX * MARKER_SIZE / np.sqrt(d['pixel'][2]))
             for d in data]
    print(f"\n  Corrected errors: X={np.mean(err_x):.4f}±{np.std(err_x):.4f} "
          f"Y={np.mean(err_y):.4f}±{np.std(err_y):.4f} "
          f"Z={np.mean(err_z):.4f}±{np.std(err_z):.4f}")


if __name__ == '__main__':
    main()
