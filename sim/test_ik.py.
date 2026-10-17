"""Check IK: can we reach a few targets with the gripper pointing down?"""
from pathlib import Path
import mujoco
import numpy as np
from ik import solve_ik
 
XML = Path(__file__).parent / "franka" / "pick_scene.xml"
m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
mujoco.mj_forward(m, d)
print("gripper now at", np.round(d.site("gripper").xpos, 3))
 
for target in [(0.5, -0.15, 0.22), (0.6, -0.05, 0.32), (0.55, 0.30, 0.35), (1.2, 0.0, 0.2)]:
    q, err = solve_ik(m, d, np.array(target))
    status = "OK" if err < 0.005 else "UNREACHABLE"
    print(f"target {target}: error {err*1000:6.1f} mm  {status}  q = {np.round(q, 2)}")
