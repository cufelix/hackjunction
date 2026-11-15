import sys, time
sys.path.insert(0, "/data/work/oliver/pan/tools")
import run_pan
from aaltonomous.sim.scene_client import SceneClient
from aaltonomous.scene.scene import Scene
sc = SceneClient("127.0.0.1")
lay = run_pan.layout_for(Scene.load(), "small_left")
pass
C = {"m": ([0.30, -0.40, 0.30], [0.32, 0.05, 0.02]), "n": ([0.05, -0.45, 0.40], [0.34, 0.05, 0.02]),
     "o": ([0.70, -0.35, 0.35], [0.30, 0.03, 0.02]), "p": ([0.15, -0.30, 0.25], [0.34, 0.06, 0.02])}
for k, (eye, tgt) in C.items():
    sc.call(cmd="showcam_pose", eye=eye, tgt=tgt)
    time.sleep(2.0)
    print(k, sc.call(cmd="showcam_pose", snap=f"/data/work/oliver/pan/cam_{k}.jpg"))
