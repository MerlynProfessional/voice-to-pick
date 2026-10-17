"""Inverse kinematics: find 7 joint angles that put the gripper site at a target."""
import mujoco
import numpy as np
 
# Gripper pointing straight down. Columns = the gripper's x, y, z axes in the world.
DOWN = np.array([[1.0,  0.0,  0.0],
                 [0.0, -1.0,  0.0],
                 [0.0,  0.0, -1.0]])
 
def orientation_error(R_now, R_goal):
    """Small rotation (as a 3-vector) that turns R_now towards R_goal."""
    return 0.5 * sum(np.cross(R_now[:, i], R_goal[:, i]) for i in range(3))
 
def solve_ik(m, d, target_pos, target_rot=DOWN, site="gripper",
             iters=200, tol=1e-3, damping=1e-2, step=0.5):
    """Return (7 joint angles, final error in m). Uses a COPY of the data,
    so the real robot does not move while we search."""
    d_ik = mujoco.MjData(m)
    d_ik.qpos[:] = d.qpos                      # start from the robot's current pose
    sid = m.site(site).id
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
 
    for _ in range(iters):
        mujoco.mj_kinematics(m, d_ik)          # forward kinematics: where is it all?
        mujoco.mj_comPos(m, d_ik)              # needed before asking for a Jacobian
        pos_err = target_pos - d_ik.site_xpos[sid]
        rot_err = orientation_error(d_ik.site_xmat[sid].reshape(3, 3), target_rot)
        err = np.concatenate([pos_err, rot_err])             # 6 numbers
        if np.linalg.norm(pos_err) < tol and np.linalg.norm(rot_err) < 0.01:
            break
        mujoco.mj_jacSite(m, d_ik, jacp, jacr, sid)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])   # 6 x 7 Jacobian, arm joints only
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)  # damped LSQ
        d_ik.qpos[:7] = np.clip(d_ik.qpos[:7] + step * dq, lo, hi)  # obey joint limits
    return d_ik.qpos[:7].copy(), float(np.linalg.norm(pos_err))
