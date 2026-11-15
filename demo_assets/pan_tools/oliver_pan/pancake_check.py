import sys, time
sys.path.insert(0, "/data/work/oliver/pan/tools")
import run_pan
from aaltonomous.sim.scene_client import SceneClient
from aaltonomous.scene.scene import Scene
sc = SceneClient("127.0.0.1")
lay = run_pan.layout_for(Scene.load(), "small_left")
print([ (o["color"], o["shape"]) for o in lay])
print(sc.call(cmd="reset", layout=lay).get("ok"))
time.sleep(4)
for o in sc.call(cmd="state")["objects"]:
    if o["pos"][0] > 0: print(o["name"], [round(v, 3) for v in o["pos"]])
sc.call(cmd="showcam_pose", eye=[0.55, -0.25, 0.30], tgt=[0.32, 0.07, 0.02]); time.sleep(2)
sc.call(cmd="showcam_pose", snap="/data/work/oliver/pan/pancake.jpg")
