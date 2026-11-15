import numpy as np
from aaltonomous.scene.scene import Scene
from aaltonomous.baselines.classical.config import hand_postures, home_arm, load_config
from aaltonomous.baselines.classical.planner import Planner
SC = Scene.load()
cfg = load_config(overlay="ab_insert")
pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
arm0 = home_arm(cfg)
base = {k: v.copy() for k, v in pl.kin.offset.items()}
print("offset", base)
def maxx(side, y, z, d, topdown, tilt_ok=None):
    pl.kin.offset = {"left": base["left"] + np.array([d,0,0]), "right": base["right"] + np.array([d,0,0])}
    lo, hi = 0.40, 0.65
    best = None
    for x in np.arange(0.46, 0.64, 0.01):
        q, ok, err, tilt = pl.kin.solve(side, np.r_[x, y, z], arm0, top_down=topdown)
        if err <= 0.006 and (not topdown or True):
            best = (x, np.degrees(tilt))
    return best
# strategies: palm offset shift d (edge grasp: object centre d ahead of stock grasp point)
for side, ys in (("left", (0.19, 0.26, 0.33)), ("right", (0.19, 0.26, 0.33))):
  for d in (0.0, 0.02, 0.03):
    for topdown in (True, False):
      for z in (0.10, 0.15, 0.20):
        r = [maxx(side, y, z, d, topdown) for y in ys]
        print(side, f"d={d:.2f}", "topdown" if topdown else "free(tilt any)", f"z={z:.2f}", " ".join(f"y{y}:{'%.2f@%.0fdeg'%(b[0],b[1]) if b else 'none'}" for y, b in zip(ys, r)))
