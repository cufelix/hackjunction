import numpy as np
from aaltonomous.scene.scene import Scene
from aaltonomous.baselines.classical.config import hand_postures, home_arm, load_config
from aaltonomous.baselines.classical.planner import Planner
from aaltonomous.baselines.classical.insert_shapes import hole_targets
SC = Scene.load(); cfg = load_config(overlay="ab_insert")
pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg)); arm0=home_arm(cfg)
g=cfg["grasp"]; H=pl.height; zc=SC.counter_z; wall=pl._wall(SC.rim_wall("left"))
zt=zc+wall+SC.cfg["lid"]["thickness"]
print("zc",zc,"H",H,"wall",wall,"z_top",zt,"z_carry",zc+max(g["carry_height"],wall+H+0.03),"hfrac",g["height_frac"])
base=pl.kin.offset["left"].copy()
T=hole_targets(SC,z_lid_top=zt)
for d in (0,0.015,0.03,0.045):
    pl.kin.offset["left"]=base+np.array([d,0,0])
    for sh,t in T.items():
        out=[]
        for dz in (0.0,0.02,0.04,0.06,0.08,0.10):
            z=zt+0.006+g["height_frac"]*H+dz
            q,ok,err,tilt=pl.kin.solve("left",np.r_[t["xy"],z],arm0,top_down=True,extra_seeds=[arm0[:7]])
            out.append(f"+{dz*100:.0f}cm:{err*1000:.0f}mm/{np.degrees(tilt):.0f}d")
        print(f"d={d} {sh} xy={np.round(t['xy'],3)} z0={zt+0.006+g['height_frac']*H:.3f}", " ".join(out), flush=True)
