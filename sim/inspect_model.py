"""Print what is inside the robot scene: sizes, joints, motors, bodies."""
from pathlib import Path
import mujoco
 
XML = Path(__file__).parent / "franka" / "pick_scene.xml"
m = mujoco.MjModel.from_xml_path(str(XML))
 
print(f"nq (position numbers) = {m.nq}")
print(f"nv (velocity numbers) = {m.nv}")
print(f"nu (motors/actuators) = {m.nu}")
print(f"timestep              = {m.opt.timestep} s")
 
print("\nJOINTS  (name, type, where its numbers start in qpos)")
types = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
for j in range(m.njnt):
    kind = types[m.jnt_type[j]]
    print(f"  {j:2d} {m.joint(j).name:15s} {kind:6s} qpos[{m.jnt_qposadr[j]}]")
 
print("\nACTUATORS  (index, name, allowed control range)")
for a in range(m.nu):
    lo, hi = m.actuator_ctrlrange[a]
    print(f"  {a} {m.actuator(a).name:10s} ctrl in [{lo:7.3f}, {hi:7.3f}]")
 
print("\nBODIES:", [m.body(b).name for b in range(m.nbody)])
print("SITES: ", [m.site(s).name for s in range(m.nsite)])
print("CAMERAS:", [m.camera(c).name for c in range(m.ncam)])
