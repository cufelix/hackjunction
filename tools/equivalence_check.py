#!/usr/bin/env python3
"""Flags-off equivalence: digest of everything the classical policy does in mock episodes.

Run the SAME script against two trees (it only uses APIs that exist at the starter baseline 36daf5b):

    PYTHONPATH=<baseline tree>    python tools/equivalence_check.py > base.txt
    PYTHONPATH=<integration tree> python tools/equivalence_check.py > integ.txt
    diff base.txt integ.txt && echo IDENTICAL

Per episode it hashes every returned action (arm/hand targets as float64 bytes + done), the policy's event notes and
the end reason / outcomes; plus a perception digest over random renders (with and without depth). Non-threaded, so
the result is deterministic. Exit 0 always; the diff is the verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np


def episode_digest(seed, time_limit, overrides=None):
    from aaltonomous.baselines.classical.mock_world import make_mock_env
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    env = make_mock_env(seed=seed)
    pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=time_limit, overrides=overrides)
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"], "time_limit": time_limit})
    h, n = hashlib.sha256(), 0
    while True:
        a = pol.act(obs)
        h.update(np.ascontiguousarray(a["arm_q_target"], np.float64).tobytes())
        h.update(np.ascontiguousarray(a["hand_q_target"], np.float64).tobytes())
        h.update(b"D" if a.get("done") else b"-")
        n += 1
        obs, info = env.step(a)
        if info["episode_over"] or n > 20000:
            break
    for e in pol.events:
        h.update(repr(e).encode())
    if pol.brain is not None:   # stderr: the digest itself must not change
        print(f"# brain seed={seed} tl={time_limit:g} stats={pol.brain.stats} overrides={len(pol.brain.trace)}",
              file=sys.stderr)
    out = env.backend.outcomes(pol.mapping)
    return (f"episode seed={seed} tl={time_limit:g} acts={n} end={info['end_reason']} "
            f"clips={info['safety_summary']['clipped_steps']} outcomes={out} plans={len(pol.plans)} "
            f"sha={h.hexdigest()[:16]}")


def perception_digest(n):
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.mock_world import Renderer
    from aaltonomous.baselines.classical.perception import Perception
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    sc, cal = Scene.load(), Calibration.load()
    p, r = Perception(load_config(), sc, cal), Renderer(sc, cal)
    L = sc.cfg["layout"]
    h = hashlib.sha256()
    rng = np.random.default_rng(12345)
    for _ in range(n):
        objs, used = [], []
        for color in rng.permutation(["red", "yellow", "blue"]):
            while True:
                x, y = rng.uniform(*L["region_x"]), rng.uniform(*L["region_y"])
                if all(np.hypot(x - a, y - b) >= 0.09 for a, b in used):
                    break
            used.append((x, y))
            objs.append(dict(color=str(color), shape="cube", yaw=float(rng.uniform(-0.6, 0.6)), size=sc.size,
                             height=sc.object_height(),
                             pos=sc.mat_to_pelvis([x, y, sc.object_height() / 2])))
        rgb, depth = r.render(objs, rng, depth_noise_mm=2.0, hole_frac=0.05)
        for dep in (depth, np.zeros_like(depth)):
            dets, rej = p.detect(rgb, dep, debug=True)
            for d in dets:
                h.update(d.color.encode() + np.asarray(d.top, np.float64).tobytes() + d.method.encode())
            for d, why in rej:
                h.update(b"R" + why.encode())
    return f"perception renders={n} sha={h.hexdigest()[:16]}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--renders", type=int, default=20)
    ap.add_argument("--brain", choices=["stub", "llamacpp"], default=None,
                    help="run with brain.enabled + brain.backend=<x> (+debug_wait); the digest must equal the flags-off one")
    a = ap.parse_args(argv)
    ov = {"brain.enabled": True, "brain.backend": a.brain, "brain.debug_wait": True} if a.brain else None
    print(perception_digest(a.renders), flush=True)
    for seed in range(a.seeds):
        for tl in (120.0, 25.0):  # 25 s exercises the 'not enough time' end
            print(episode_digest(seed, tl, ov), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
