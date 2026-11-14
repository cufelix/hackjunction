"""Reachability of each arm over the official spawn region x 3 bins: fraction of object positions with a full plan
(reach + keepouts + dry run) per arm and bin, and the planned pick duration. Offline (pinocchio only).

    PYTHONPATH=$REPO python tools/arm_reach.py [--overlay finals_r21] [--n 6]
"""
import argparse
import numpy as np
from aaltonomous.baselines.classical.config import load_config, hand_postures, home_arm
from aaltonomous.baselines.classical.perception import Detection
from aaltonomous.baselines.classical.planner import Planner
from aaltonomous.scene.scene import Scene


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--overlay", default="finals_r21"); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    cfg = load_config(profile="sim", overlay=a.overlay)
    SC = Scene.load(); post = hand_postures(cfg); home = home_arm(cfg)
    pl = Planner(cfg, SC, post, home)
    L = SC.cfg["layout"]
    xs = np.linspace(*L["region_x"], a.n); ys = np.linspace(*L["region_y"], a.n)
    pts = SC.mat_to_pelvis(np.array([[x, y, 0.0] for x in xs for y in ys]))[:, :2]
    H = pl.height
    for bin_name, colour in (("left", "red"), ("center", "yellow"), ("right", "blue")):
        for side in ("left", "right"):
            ok, dur = 0, []
            for p in pts:
                det = Detection(colour, np.r_[p, SC.counter_z + H], 0, 0, "depth")
                plan = pl._plan_side_hints(det, side, bin_name, home.copy(), post["open"].copy())
                if plan.ok:
                    ok += 1; dur.append(plan.duration)
            print(f"bin={bin_name:6s} arm={side:5s} feasible {ok}/{len(pts)} = {100*ok/len(pts):.0f}%  "
                  f"mean pick duration {np.mean(dur) if dur else float('nan'):.1f}s")


if __name__ == "__main__":
    main()
