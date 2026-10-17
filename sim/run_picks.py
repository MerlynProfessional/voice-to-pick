"""Scripted pick-and-place test: 10 trials, random cube positions, count successes.
Run:  python sim/run_picks.py          (fast, no window)
      python sim/run_picks.py --view   (watch it in the 3D viewer)"""
import sys
from pathlib import Path
import mujoco
import numpy as np
from skills import Robot
 
XML = Path(__file__).parent / "franka" / "pick_scene.xml"
BIN = np.array([0.55, 0.30])                       # bin centre (x, y) from pick_scene.xml
TRIALS = 10
 
def cube_in_bin(d, name):
    p = d.body(name).xpos
    return bool(np.all(np.abs(p[:2] - BIN) < 0.08) and p[2] < 0.30)
 
def run(m, d, viewer=None):
    rng = np.random.default_rng(seed=0)    # fixed seed: same "random" cubes each run
    wins = 0
    for t in range(TRIALS):
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        adr = m.joint("red_cube").qposadr[0]   # where the cube's 7 numbers start
        d.qpos[adr:adr + 3] = [rng.uniform(0.40, 0.65), rng.uniform(-0.30, 0.10), 0.22]
        mujoco.mj_forward(m, d)
        robot = Robot(m, d, viewer)
        robot.step(250)                            # let the cube settle (0.5 s)
        cube = d.body("red_cube").xpos.copy()
        try:
            robot.pick(cube)
            lifted = d.body("red_cube").xpos[2] > 0.27
            robot.place([BIN[0], BIN[1], 0.30])
            ok = lifted and cube_in_bin(d, "red_cube")
        except RuntimeError as e:
            print("  ", e); ok = False
        wins += ok
        result = "SUCCESS" if ok else "FAIL"
        print(f"trial {t+1:2d}: cube at {np.round(cube[:2], 3)} -> {result}")
    print(f"\nSuccess: {wins}/{TRIALS}")
 
if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    if "--view" in sys.argv:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(m, d) as v:   # the window shows THIS m and d
            run(m, d, v)
    else:
        run(m, d)
