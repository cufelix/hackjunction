import sys, numpy as np
from aaltonomous.scene.scene import Scene
from aaltonomous.baselines.classical.config import hand_postures, home_arm, load_config
from aaltonomous.baselines.classical.planner import Planner
from aaltonomous.baselines.classical.perception import Detection
SC = Scene.load()
def det(shape, xy, yaw=0.0):
    d = Detection("red", np.array([xy[0], xy[1], SC.counter_z + SC.size]), 100, 100, "depth", yaw)
    d.shape, d.shape_conf, d.shape_path = shape, 0.9, "metric"
    d.apex_yaw, d.apex_conf = (1.0, 0.9) if shape=="triangle" else (None, 0.0)
    return d
for d_shift in (0.0, 0.015, 0.02, 0.03):
  for gap in (0.015, 0.04, 0.08):
    cfg = load_config(overlay="ab_insert"); cfg["insert"]["palm_shift_m"]=d_shift; cfg["insert"]["release_gap_m"]=min(gap,0.012) if gap<0.02 else gap; cfg["insert"]["release_gap_max_m"]=gap+0.01
    pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg)); pl.insert_active=True
    arm0, hand0 = home_arm(cfg), np.zeros(14)
    res=[]
    for shape in ("triangle","cube","cylinder"):
        oks=0; why=set()
        for xy in ((0.30,0.15),(0.33,0.22),(0.28,0.28),(0.36,0.12)):
            p = pl.plan_insert(det(shape, xy, 0.2), arm0, hand0)
            oks += bool(p.ok); why.add(p.reason[:70])
        res.append((shape, oks, sorted(why)[:2]))
    print(f"shift={d_shift} gap={gap}", res, flush=True)
