#!/usr/bin/env python3
"""Synthetic policy log from the mock sim, for tools/held_offline_eval.py (run from the worktree root).

  PYTHONPATH=. $PY tools/held_mock_log.py --out DIR [--episodes 3] [--tau 0.15] [--mode stall|finger_sum|softgrip]

Writes DIR/policy_server.log (same line format as policy_server.py: `reset task=.. seed=..` + the policy's own
`[grasp_diag]` / `[held_abort]` / `[soft_baseline]` lines) and DIR/eval.json (`{"trials": [...]}` with per-object
outcomes from the mock scene). `--tau` is the mock's first-order hand lag (the only soft knob the mock has). The mock
has NO finger stall on an object, so an object close and an empty close look the same except for the lag: this log
checks the plumbing and the lag-normalised softgrip, not real held/empty separation.
"""
import argparse
import json
import os
import time

from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.baselines.classical.policy import ClassicalPolicy

SOFT_ON = {"grasp.soft_squeeze": True, "grasp.soft_hand": True, "held.soft_detector": True,
           "grasp.held_detector": "finger_sum"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--mode", choices=("stall", "finger_sum", "softgrip"), default="finger_sum")
    ap.add_argument("--time-limit", type=float, default=60.0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ov = {"policy.grasp_diag_log": True}
    if a.mode == "finger_sum":
        ov["grasp.held_detector"] = "finger_sum"
    elif a.mode == "softgrip":
        ov.update(SOFT_ON)
    lines, trials = [], []

    def log(msg):
        lines.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')},000 INFO {msg}")

    for seed in range(a.episodes):
        env = make_mock_env(seed=seed, tau=a.tau)
        obs = env.reset()
        pol = ClassicalPolicy(config=load_config(overlay="none", overrides=ov), safety=env.filter.cfg,
                              threaded=False, time_limit=a.time_limit, log=log)
        log(f"reset task=sort_fixed seed={seed} instruction={obs['instruction']!r}")
        pol.reset({"instruction": obs["instruction"], "seed": seed})
        info = {}
        while True:
            obs, info = env.step(pol.act(obs))
            if info["episode_over"]:
                break
        outc = env.backend.outcomes(pol.mapping)
        trials.append({"seed": seed, "end_reason": info.get("end_reason"),
                       "outcomes": [{"color": o["color"], "outcome": r}
                                    for o, r in zip(env.backend.objects, outc)]})
        log(f"[mock] seed={seed} outcomes={outc}")
    with open(os.path.join(a.out, "policy_server.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "eval.json"), "w") as fh:
        json.dump({"trials": trials}, fh, indent=1, default=str)
    print(f"wrote {a.out}/policy_server.log and eval.json ({len(trials)} episodes, mode={a.mode}, tau={a.tau})")


if __name__ == "__main__":
    main()
