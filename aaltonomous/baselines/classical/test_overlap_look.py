"""motion.overlap_look (Oliver J8): the look runs during the return home; the result is used on arrival.

Run: $PY -m pytest -q aaltonomous/baselines/classical/test_overlap_look.py -s   (-s shows the time-per-object table)

The mock runs unpaced, so the look thread (nice 10) would starve and the sim clock would run ahead of it. While a
look is pending the episode loop therefore sleeps one tick period per tick, like the 30 Hz robot loop does.
Both arms use the same pacing.
"""
import time
import unittest
from types import SimpleNamespace

import numpy as np

from .config import load_config
from .mock_world import make_mock_env
from .policy import HZ, ClassicalPolicy


def pending(pol):
    """A look (normal or overlap) is running in its thread."""
    for j in (pol._job, pol._overlap_job):
        if j is not None and j.get("result") is None:
            return True
    return False


def run_episode(overlap, seed, time_limit=120.0, record=False):
    cfg = load_config(overlay="rc1")
    cfg["motion"]["overlap_look"] = overlap
    env = make_mock_env(seed=seed)
    pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg, time_limit=time_limit, threaded=True)
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"], "time_limit": time_limit})
    seen = {"plan_from": False, "q0_calls": [], "start": None}
    if record:   # record every start pose the planner receives while the overlap job is running
        q0 = pol._q0

        def recording_q0():
            out = q0()
            if pol._plan_from is not None:
                seen["q0_calls"].append(out[0].copy())
            return out
        pol._q0 = recording_q0
    info, ticks = {}, 0
    while True:
        act = pol.act(obs)
        if pending(pol):
            time.sleep(len(act["arm_q_target"]) / HZ)
        if pol._plan_from is not None:
            seen["plan_from"] = True
            if seen["start"] is None:   # overlap just started: remember the queue end and the arm position
                seen["start"] = (pol._plan_from[0].copy(), pol.arm_cmd.copy())
        obs, info = env.step(act)
        ticks += 1
        if info["episode_over"] or ticks > 40000:
            break
    out = env.backend.outcomes(pol.mapping)
    notes = [m for _, m in pol.events]
    return {"t": float(obs["t"]), "end": info.get("end_reason"), "placed": out.count("correct"),
            "wrong": out.count("wrong") + out.count("off"), "outcomes": out, "notes": notes,
            "seen": seen, "policy": pol}


class TestOverlapLook(unittest.TestCase):
    def test_flag_off_never_plans_from_the_queue_end(self):
        cfg = load_config(overlay="rc1")
        self.assertFalse(cfg["motion"]["overlap_look"])
        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg, time_limit=120.0, threaded=True)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"], "time_limit": 120.0})
        for _ in range(40000):
            act = pol.act(obs)
            if pending(pol):
                time.sleep(len(act["arm_q_target"]) / HZ)
            self.assertIsNone(pol._plan_from)
            self.assertIsNone(pol._overlap_job)
            obs, info = env.step(act)
            if info["episode_over"]:
                break
        self.assertTrue(any("execute" in m for _, m in pol.events))
        self.assertFalse(any("overlap" in m for _, m in pol.events))

    def test_ab_overlap_overlay_flips_only_the_flag(self):
        from .config import effective_flags
        on = {k for k, _, is_on, _, _ in effective_flags(load_config(overlay="ab_overlap")) if is_on}
        base = {k for k, _, is_on, _, _ in effective_flags(load_config(overlay="rc1")) if is_on}
        self.assertEqual(on - base, {"motion.overlap_look"})
        self.assertEqual(base - on, set())
        self.assertTrue(load_config(overlay="ab_overlap")["motion"]["overlap_look"])

    def test_overlap_plans_from_the_queue_end_pose(self):
        r = run_episode(True, seed=0, record=True)
        seen = r["seen"]
        self.assertTrue(seen["plan_from"], "overlap look never started")
        self.assertTrue(seen["q0_calls"], "no planning call read the start pose during the overlap job")
        end_arm, arm_at_start = seen["start"]
        self.assertGreater(np.abs(end_arm - arm_at_start).max(), 1e-6,
                           "arm was already at the queue end: the test cannot tell the two poses apart")
        for arm in seen["q0_calls"]:
            np.testing.assert_allclose(arm, end_arm, atol=1e-12)
        self.assertNotIn("look/plan error", " ".join(r["notes"]))

    def test_freshness_rule(self):
        cfg = load_config(overlay="rc1")
        cfg["motion"]["overlap_look"] = True
        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg, time_limit=120.0, threaded=True)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"], "time_limit": 120.0})
        pol.arm_cmd, pol.hand_cmd = np.zeros(14), np.zeros(14)
        used = []
        pol._after_look = lambda res, t: used.append(res) or "used"

        def plan(xy):
            return ("plan", SimpleNamespace(waypoints={"grasp": np.array([xy[0], xy[1], 0.03])}))

        def drain(result, xy_attempted=(0.27, 0.02), t=5.0):
            used.clear()
            pol._overlap_job = {"obs": {}, "result": result, "xy": np.asarray(xy_attempted, float), "t0": None}
            pol._plan_from = (np.zeros(14), np.zeros(14))
            out = pol._overlap_drain(obs, t)
            return out, list(used), pol._overlap_job, pol._plan_from

        # fresh: a different object 10 cm away is used at once
        out, got, job, pf = drain(plan((0.27, 0.12)))
        self.assertEqual(out, "used")
        self.assertEqual(len(got), 1)
        self.assertIsNone(job)
        self.assertIsNone(pf)
        # stale: the target is within 3 cm of the object we just attempted -> discarded, normal look
        out, got, job, pf = drain(plan((0.28, 0.03)))
        self.assertIsNone(out)
        self.assertEqual(got, [])
        self.assertIsNone(job)
        self.assertIsNone(pf)
        # error: discarded
        out, got, _, _ = drain(("error", "boom"))
        self.assertIsNone(out)
        self.assertEqual(got, [])
        # not ready: hold, then after 1.0 s of sim time give up and fall back to the normal look
        pol._overlap_job = {"obs": {}, "result": None, "xy": np.array([0.27, 0.02]), "t0": None}
        pol._plan_from = (np.zeros(14), np.zeros(14))
        self.assertIn("arm_q_target", pol._overlap_drain(obs, 5.0))
        self.assertIsNotNone(pol._overlap_job)
        self.assertIn("arm_q_target", pol._overlap_drain(obs, 5.9))
        self.assertIsNone(pol._overlap_drain(obs, 6.1))
        self.assertIsNone(pol._overlap_job)
        self.assertIsNone(pol._plan_from)

    def test_verify_placement_disables_overlap(self):
        cfg = load_config(overlay="rc1")
        cfg["motion"]["overlap_look"] = True
        cfg["policy"]["verify_placement"] = True
        pol = ClassicalPolicy(config=cfg, log=None, threaded=True)
        self.assertFalse(pol._overlap_ok())

    def test_q0_is_identity_when_flag_off(self):
        pol = ClassicalPolicy(config=load_config(overlay="rc1"), log=None)
        arm, hand = np.arange(14.0), np.arange(14.0) + 1
        pol.arm_cmd, pol.hand_cmd = arm, hand
        q_arm, q_hand = pol._q0()
        self.assertIs(q_arm, arm)
        self.assertIs(q_hand, hand)

    def test_mock_episodes_flag_on_vs_off(self):
        rows = []
        for seed in (0, 1, 2):
            off = run_episode(False, seed)
            on = run_episode(True, seed)
            self.assertEqual(off["wrong"], 0, f"off seed {seed}: {off['outcomes']}")
            self.assertEqual(on["wrong"], 0, f"on seed {seed}: {on['outcomes']}")
            self.assertGreaterEqual(on["placed"], off["placed"], f"seed {seed}")
            self.assertFalse(any("look/plan error" in m for m in on["notes"]))
            rows.append((seed, off, on))
        print()
        print(f"{'seed':>4} {'off placed':>10} {'on placed':>9} {'off s/obj':>9} {'on s/obj':>8} {'saving s/obj':>12}")
        tot = {"off": [0.0, 0], "on": [0.0, 0]}
        for seed, off, on in rows:
            s_off, s_on = off["t"] / max(off["placed"], 1), on["t"] / max(on["placed"], 1)
            print(f"{seed:>4} {off['placed']:>10} {on['placed']:>9} {s_off:>9.2f} {s_on:>8.2f} {s_off - s_on:>12.2f}")
            for k, r in (("off", off), ("on", on)):
                tot[k][0] += r["t"]
                tot[k][1] += r["placed"]
            discards = sum("discarded" in m for m in on["notes"])
            print(f"     on: overlap used {sum('overlap look used' in m for m in on['notes'])}, discarded {discards}")
        s_off, s_on = tot["off"][0] / tot["off"][1], tot["on"][0] / tot["on"][1]
        print(f"all seeds: off {s_off:.2f} s/obj, on {s_on:.2f} s/obj, saving {s_off - s_on:.2f} s/obj")
        self.assertLess(s_on, s_off)


if __name__ == "__main__":
    unittest.main()
