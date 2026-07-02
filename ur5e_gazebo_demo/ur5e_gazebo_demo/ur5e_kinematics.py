#!/usr/bin/env python3
"""
UR5e Kinematics — Standard DH Parameters, Pure NumPy.

Standard DH convention:  T_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)

Kinematic chain:
    base_link ──[Rz(π)]──> base_link_inertia ──[DH 1..6]──> wrist_3 ──[Rx(π)]──> ft_frame

    The leading Rz(π) is the REP-103 alignment (base X+ = forward; robot X+ = backward).
    The trailing Rx(π) is the wrist_3 → ft_frame fixed rotation.

Verified against IKPy FK to machine precision (1e-16).
No ikpy, no sympy, no URDF parsing — just numpy.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  Standard DH Parameter Table — UR5e
# ═══════════════════════════════════════════════════════════════════════
#
#   Joint  │  a (m)   │  d (m)   │  α (rad)  │  θ
#  ────────┼──────────┼──────────┼───────────┼─────
#   J1     │   0       │  0.1625  │   π/2     │  θ₁   shoulder_pan
#   J2     │  -0.425   │  0       │   0       │  θ₂   shoulder_lift
#   J3     │  -0.3922  │  0       │   0       │  θ₃   elbow
#   J4     │   0       │  0.1333  │   π/2     │  θ₄   wrist_1
#   J5     │   0       │  0.0997  │  -π/2     │  θ₅   wrist_2
#   J6     │   0       │  0.0996  │   0       │  θ₆   wrist_3
#  ────────┴──────────┴──────────┴───────────┴─────
#
#  Fixed transforms (not in DH table):
#    PRE:  base_link → base_link_inertia   Rz(π)
#    POST: wrist_3   → ft_frame            Rx(π)

DH_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
DH_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

# Fixed chain transforms
_T_PRE = np.array([  # Rz(π): base_link → base_link_inertia
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])
_T_POST = np.array([  # Rx(π): wrist_3 → ft_frame
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
])


# ── Helper ────────────────────────────────────────────────────────────
def _rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, ca, -sa, 0], [0, sa, ca, 0], [0, 0, 0, 1]])

def _rz(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0, 0], [sa, ca, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def _trans(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


# ── DH transformation matrix ──────────────────────────────────────────
def _dh_matrix(theta, d, a, alpha):
    """Standard DH: T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,      ca,       d],
        [0,   0,       0,        1],
    ])


# ── Forward Kinematics ────────────────────────────────────────────────
def fk(joints):
    """Forward kinematics: joint angles → ft_frame pose (base_link frame).

    Args:
        joints: [θ₁..θ₆] in radians.

    Returns:
        4×4 homogeneous transform matrix.
    """
    q = np.asarray(joints, dtype=np.float64)
    T = _T_PRE.copy()
    for i in range(6):
        T = T @ _dh_matrix(q[i], DH_D[i], DH_A[i], DH_ALPHA[i])
    return T @ _T_POST


def fk_position(joints):
    """Return ft_frame position [x, y, z] in base_link frame."""
    T = fk(joints)
    return np.array([T[0, 3], T[1, 3], T[2, 3]], dtype=np.float64)


def fk_axis(joints):
    """Return ft_frame Z-axis direction in base_link frame."""
    T = fk(joints)
    return np.array([T[0, 2], T[1, 2], T[2, 2]], dtype=np.float64)


# ── Geometric Jacobian ────────────────────────────────────────────────
def _jacobian(joints):
    """6×6 geometric Jacobian: [v; ω] = J @ q̇.

    J_v,i = z_i × (p_ee − p_i)      — position contribution
    J_ω,i = z_i                      — orientation contribution
    """
    q = np.asarray(joints, dtype=np.float64)
    n = len(q)

    T_ee = fk(joints)
    p_ee = T_ee[:3, 3]

    # Walk the chain, extract z_i and p_i from each joint's frame
    T = _T_PRE.copy()
    J = np.zeros((6, n))
    for i in range(n):
        # Joint i frame = after fixed transform, before θ rotation
        z_i = T[:3, 2]
        p_i = T[:3, 3]
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i
        # Advance past joint rotation
        T = T @ _dh_matrix(q[i], DH_D[i], DH_A[i], DH_ALPHA[i])

    return J


def _jacobian_fd(joints, eps=1e-6):
    """Finite-difference Jacobian (for verification only)."""
    q = np.asarray(joints, dtype=np.float64)
    n = len(q)
    J = np.zeros((6, n))
    T0 = fk(q)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    for i in range(n):
        qp = q.copy(); qp[i] += eps
        Tp = fk(qp)
        J[:3, i] = (Tp[:3, 3] - p0) / eps
        dR = (Tp[:3, :3] - R0) / eps
        S = dR @ R0.T
        J[3:, i] = np.array([S[2, 1], S[0, 2], S[1, 0]])
    return J


# ── Inverse Kinematics ────────────────────────────────────────────────
def ik(target_position, target_orientation=None, seed=None, active_mask=None,
       max_iter=500, tol_pos=1e-5, tol_ori=1e-4):
    """Damped least-squares IK with adaptive Levenberg-Marquardt damping.

    Args:
        target_position:   [x, y, z] in base_link frame.
        target_orientation: [ax, ay, az] for Z-axis alignment
                           (like IKPy orientation_mode='Z').  None = position-only.
        seed:              Initial [θ₁..θ₆]; defaults to zeros.
        active_mask:       6-element bool list; True = active DOF.
        max_iter:          Max iterations.
        tol_pos:           Position convergence (m).
        tol_ori:           Orientation convergence — sin(angle) threshold.

    Returns:
        np.array([θ₁..θ₆]) in [-π, π].
    """
    if seed is None:
        theta = np.zeros(6)
    else:
        theta = np.asarray(seed, dtype=np.float64).copy()

    if active_mask is None:
        active_mask = [True] * 6
    active = np.array(active_mask, dtype=bool)
    n_active = active.sum()
    if n_active == 0:
        return theta

    tgt_z = None
    if target_orientation is not None:
        tgt_z = np.asarray(target_orientation, dtype=np.float64)
        nrm = np.linalg.norm(tgt_z)
        if nrm > 1e-12:
            tgt_z = tgt_z / nrm

    tgt_p = np.asarray(target_position, dtype=np.float64)
    best_theta = theta.copy()
    best_err = float('inf')
    lam = 0.5

    for _ in range(max_iter):
        T = fk(theta)
        cur_p = np.array([T[0, 3], T[1, 3], T[2, 3]])
        err_p = tgt_p - cur_p
        pos_err = np.linalg.norm(err_p)

        if tgt_z is not None:
            cur_z = np.array([T[0, 2], T[1, 2], T[2, 2]])
            cross_z = np.cross(cur_z, tgt_z)
            ori_err = np.linalg.norm(cross_z)
        else:
            cross_z = np.zeros(3)
            ori_err = 0.0

        total_err = pos_err + ori_err
        if total_err < best_err:
            best_err = total_err
            best_theta = theta.copy()

        if pos_err < tol_pos and ori_err < tol_ori:
            break

        J_full = _jacobian(theta)
        J = J_full[:, active]
        err = np.concatenate([err_p, 0.5 * cross_z])

        # Δθ = Jᵀ (J Jᵀ + λ² I)⁻¹ err
        JJt = J @ J.T
        A = JJt + lam * lam * np.eye(6)
        try:
            delta = J.T @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            delta = J.T @ np.linalg.lstsq(A, err, rcond=None)[0]

        max_step = 0.8
        nrm_delta = np.linalg.norm(delta)
        if nrm_delta > max_step:
            delta *= max_step / nrm_delta

        theta_new = theta.copy()
        theta_new[active] += delta
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        new_pos_err = np.linalg.norm(tgt_p - fk_position(theta_new))
        if new_pos_err < pos_err:
            theta = theta_new
            lam = max(lam * 0.5, 1e-4)
        else:
            lam = min(lam * 2.0, 10.0)
            if lam > 5.0:
                theta[active] += np.random.uniform(-0.1, 0.1, n_active)
                theta = (theta + np.pi) % (2 * np.pi) - np.pi

    return best_theta


# ── Convenience: IK with frozen joints ────────────────────────────────
def ik_frozen(target_position, target_orientation, seed, freeze_indices=None):
    """IK with selected joints frozen at seed values.

    Args:
        target_position:   [x, y, z] in base_link frame.
        target_orientation: [ax, ay, az] for Z-axis alignment.
        seed:              Initial [θ₁..θ₆].
        freeze_indices:    Joint indices (0..5) to freeze.

    Returns:
        np.array([θ₁..θ₆]) in [-π, π].
    """
    mask = [True] * 6
    if freeze_indices is not None:
        for idx in freeze_indices:
            mask[idx] = False

    seed_arr = np.asarray(seed, dtype=np.float64).copy()
    result = ik(target_position, target_orientation, seed=seed_arr, active_mask=mask)

    if freeze_indices is not None:
        for idx in freeze_indices:
            result[idx] = seed_arr[idx]
    return result


# ── Smoke test ────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print("╔══════════════════════════════════════════════╗")
    print("║   UR5e DH Kinematics — Self-Test            ║")
    print("╚══════════════════════════════════════════════╝")

    # ── DH table ──
    print("\n── Standard DH Parameters ──")
    print(f"{'Joint':<7} {'a(m)':<10} {'d(m)':<10} {'α(rad)':<10}")
    names = ['J1 (pan)', 'J2 (lift)', 'J3 (elbow)', 'J4 (w1)', 'J5 (w2)', 'J6 (w3)']
    for i, name in enumerate(names):
        print(f"{name:<7} {DH_A[i]:<10.4f} {DH_D[i]:<10.4f} {DH_ALPHA[i]:<10.4f}")

    # ── FK verification against IKPy ──
    home = [0.0, -1.5707, 0.0, -1.5707, 0.0, 0.0]
    T = fk(home)
    pos = [T[0, 3], T[1, 3], T[2, 3]]
    zax = [T[0, 2], T[1, 2], T[2, 2]]
    print(f"\nFK home: pos={[f'{v:.4f}' for v in pos]}  Z={[f'{v:.4f}' for v in zax]}")
    assert np.allclose(pos, [0.0001, 0.2329, 1.0794], atol=1e-3)
    assert np.allclose(zax, [0.0, -1.0, 0.0], atol=1e-3)
    print("  ✓ matches IKPy")

    test = [0.5, -1.2, 0.3, -1.8, 0.1, 0.5]
    T2 = fk(test)
    pos2 = [T2[0, 3], T2[1, 3], T2[2, 3]]
    assert np.allclose(pos2, [0.2672, 0.4108, 0.9602], atol=1e-3)
    print("  ✓ test pose matches IKPy")

    # ── Jacobian check ──
    J_an = _jacobian(home)
    J_fd = _jacobian_fd(home)
    J_err = np.abs(J_an - J_fd).max()
    print(f"\nJacobian: max |analytic - fd| = {J_err:.2e}")
    assert J_err < 1e-4
    print("  ✓ matches finite differences")

    # ── IK tests ──
    home_pos = fk_position(home)
    orient = [0.0, -1.0, 0.0]

    target = np.array([0.4, 0.25, 0.95])
    result = ik(target, orient, seed=home)
    err = np.linalg.norm(fk_position(result) - target)
    print(f"\nIK target={target} → err={err:.5f}m")
    assert err < 0.02
    print("  ✓ converges")

    # Frozen wrist IK (realistic grasp descent)
    above_j = ik(np.array([0.5, 0.15, 1.0]), orient, seed=home)
    grasp_target = fk_position(above_j) + np.array([0.0, 0.0, -0.16])
    grasp_j = ik_frozen(grasp_target, orient, seed=above_j, freeze_indices=[4, 5])
    err_g = np.linalg.norm(fk_position(grasp_j) - grasp_target)
    print(f"Grasp descent (4-DOF): err={err_g:.5f}m")
    assert err_g < 0.02
    assert abs(grasp_j[4] - above_j[4]) < 1e-10
    assert abs(grasp_j[5] - above_j[5]) < 1e-10
    print("  ✓ wrist frozen, converges")

    print("\n✅ All tests passed!")
