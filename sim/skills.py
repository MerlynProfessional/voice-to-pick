"""Robot skills built on IK: move_to, open/close gripper, pick, place."""
import time
import mujoco
import numpy as np
from ik import solve_ik
 
OPEN, CLOSED = 255.0, 0.0         # actuator8 range: 255 = fingers open (8 cm), 0 = closed
HOVER = 0.12                      # approach height above an object (m)
 
class Robot:
    def __init__(self, m, d, viewer=None):
        self.m, self.d, self.viewer = m, d, viewer
        self.grip = OPEN
 
    def step(self, n=1):
        """Advance physics n steps (2 ms each). If a viewer is open, draw it."""
        for _ in range(n):
            self.d.ctrl[7] = self.grip
            mujoco.mj_step(self.m, self.d)
            if self.viewer and self.viewer.is_running() and self.d.time % 0.02 < 0.002:
                self.viewer.sync()              # redraw ~50 times per simulated second
                time.sleep(0.02)                # so it plays at roughly real speed
 
    def move_joints(self, q_goal, duration=1.5):
        """Glide smoothly from the current joint angles to q_goal."""
        q_start = self.d.qpos[:7].copy()
        n = int(duration / self.m.opt.timestep)
        for i in range(1, n + 1):
            s = i / n
            s = 3 * s**2 - 2 * s**3            # smoothstep: slow start, slow stop
            self.d.ctrl[:7] = q_start + s * (q_goal - q_start)
            self.step()
        self.step(150)                          # 0.3 s to settle
    def move_to(self, xyz, duration=1.5):
        q, err = solve_ik(self.m, self.d, np.asarray(xyz, dtype=float))
        if err > 0.005:
            raise RuntimeError(f"IK could not reach {xyz}: error {err*1000:.1f} mm")
        self.move_joints(q, duration)
 
    def gripper(self, value, wait=0.6):
        self.grip = value
        self.step(int(wait / self.m.opt.timestep))
 
    def gripper_pos(self):
        return self.d.site("gripper").xpos.copy()
 
    def pick(self, xyz):
        x, y, z = xyz
        self.gripper(OPEN, 0.3)
        self.move_to([x, y, z + HOVER])         # 1. pre-grasp above the object
        self.move_to([x, y, z], duration=1.0)   # 2. straight down
        self.gripper(CLOSED)                    # 3. squeeze and wait for friction to build
        self.move_to([x, y, z + HOVER])         # 4. lift
 
    def place(self, xyz):
        x, y, z = xyz
        self.move_to([x, y, z + HOVER])         # 5. carry above the target
        self.move_to([x, y, z], duration=1.0)
        self.gripper(OPEN, 0.5)                 # 6. release
        self.move_to([x, y, z + HOVER], duration=1.0)

