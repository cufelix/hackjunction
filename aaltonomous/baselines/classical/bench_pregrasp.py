"""Real-wall-clock timing of the pre-grasp pause (grasp.pregrasp_relocalise). Run:
    python -m aaltonomous.baselines.classical.bench_pregrasp [reps]

Threaded policy (safety_shell, as in the finals), real clock: during the pause every hold chunk is slept for its
duration and the planning / re-detection thread really runs meanwhile, so the numbers include the planner. Reported
per arm and per offset: wall seconds from the first paused tick to the start of the pick (median over reps), and the
time the worker spent in perception.relocalise + planning.
"""
import statistics
import sys
import time
from unittest import mock

import numpy as np

from .config import load_config
from .mock_world import make_mock_env
from .policy import ClassicalPolicy
from .test_pregrasp import BLUE, EMPTY, HZ, ON, RED, SC, YELLOW, H, _frame

CASES = ((0.0, 0.0), (0.02, 0.0), (0.0, 0.04), (0.0, 0.06))   # (dx, dy) of the first-look error, m


def one(scene, off, overlay=ON, overrides=None):
    """-> dict(wall, work, kind) for one pick; wall = first paused tick -> pregrasp_done start (real seconds)."""
    env = make_mock_env(seed=0)
    base = env.reset()
    ov = dict(overrides or {})
    pol = ClassicalPolicy(config=load_config(overlay=overlay, overrides=ov), safety=env.filter.cfg, log=None)
    pol.reset({"instruction": "", "time_limit": 120})
    true_p = SC.mat_to_pelvis([scene[1][0], scene[1][1], H / 2])
    first = _frame(scene[0], true_p + np.array([off[0], off[1], 0.0]))
    truth = _frame(scene[0], true_p)
    t_pause, t_done, work = None, None, {}
    begin = pol._start

    def rec(plan, t, **kw):
        nonlocal t_done
        if kw.get("pregrasp_done") and t_done is None:
            t_done = time.perf_counter()
        return begin(plan, t, **kw)
    pol._start = rec
    run = pol._pg_run

    def timed(pg):
        t0 = time.perf_counter()
        try:
            return run(pg)
        finally:
            work["s"] = time.perf_counter() - t0
    pol._pg_run = timed
    t = 0.0
    with mock.patch.object(pol.perception, "_fk_arm_mask", return_value=EMPTY):
        for _ in range(20000):
            job = getattr(pol, "_job", None)
            if pol._pg is None and job is not None and job.get("result") is None:
                time.sleep(0.002)                      # first look still planning: not part of the pause
                continue
            paused = pol._pg is not None and pol._pg["t0"] is not None      # at the pre-grasp pose, holding
            t_call = time.perf_counter()
            rgb, dep = truth if pol._pg is not None else first
            a_q = base["arm_q"] if pol.arm_cmd is None else pol.arm_cmd
            h_q = base["hand_q"] if pol.hand_cmd is None else pol.hand_cmd
            res = pol.act(dict(base, head_rgb=rgb, head_depth=dep, t=t, arm_q=a_q, hand_q=h_q))
            n = len(res["arm_q_target"])
            if t_pause is None and pol._pg is not None and pol._pg["t0"] is not None:
                t_pause = t_call                       # the call that reached the pre-grasp pose
                paused = True
            if paused:
                time.sleep(n / HZ)                     # the hold chunk really takes its time
            t += n / HZ
            if t_done is not None:
                break
    kind = next((m for _, m in pol.events if "[pregrasp]" in m), "")
    return {"wall": None if t_done is None or t_pause is None else t_done - t_pause, "work": work.get("s"),
            "note": kind}


def table(reps=5, overlay=ON, overrides=None):
    rows = []
    for arm, scene in (("left (yellow)", YELLOW), ("right (red)", RED)):
        for off in CASES:
            r = [one(scene, off, overlay, overrides) for _ in range(reps)]
            w = [x["wall"] for x in r if x["wall"] is not None]
            k = [x["work"] for x in r if x["work"] is not None]
            rows.append((arm, 1000 * float(np.hypot(*off)), statistics.median(w) if w else float("nan"),
                         max(w) if w else float("nan"), statistics.median(k) if k else float("nan")))
    return rows


def main():
    """python -m ...bench_pregrasp [reps] [--load]: --load keeps every CPU core busy (the sim / other processes share
    the machine in a run), so the planner thread competes for CPU like it does there."""
    import os
    import subprocess
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reps = int(args[0]) if args else 5
    burn = []
    if "--load" in sys.argv:
        burn = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(os.cpu_count() or 4)]
    try:
        print("arm            shift_mm  wall_med_s  wall_max_s  worker_med_s")
        for arm, mm, wm, wx, kw in table(reps):
            print(f"{arm:<14} {mm:8.0f}  {wm:10.3f}  {wx:10.3f}  {kw:12.3f}")
    finally:
        for b in burn:
            b.kill()


if __name__ == "__main__":
    main()
