"""grasp.adaptive_close / motion.conditional_settle (fasthold). Run from the repo root:
    PYTHONPATH=. python -m unittest aaltonomous.baselines.classical.test_fasthold -v
Unit checks of the timing rules + mock-world episodes: ab_fasthold (cand_r17 + both flags) vs cand_r17, seeds 1-3.
"""
import copy
import functools
import unittest

import numpy as np

from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .mock_world import make_mock_env
from .perception import Detection
from .planner import Planner, close_timing
from .policy import HZ, ClassicalPolicy

SEEDS = (1, 2, 3)


def make_policy(overlay, overrides=None, log=None):
    cfg = load_config(overlay=overlay, overrides=overrides)
    env = make_mock_env(seed=1)
    pol = ClassicalPolicy(config=cfg, log=log or (lambda m: None), safety=env.filter.cfg, time_limit=120.0)
    return pol


@functools.lru_cache(maxsize=None)
def episode(overlay, seed):
    """Full mock episode with the classical policy. Returns plain numbers + the policy's own log lines."""
    from aaltonomous.interface.tasks import FIXED_MAPPING
    cfg = load_config(overlay=overlay)
    env = make_mock_env(seed=seed)
    lines = []
    pol = ClassicalPolicy(config=cfg, log=lines.append, safety=env.filter.cfg, time_limit=120.0)
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"], "time_limit": 120.0})
    attach, prev, info = 0, dict(env.backend.held), {}
    for _ in range(20000):
        obs, info = env.step(pol.act(obs))
        for side in ("left", "right"):
            if env.backend.held[side] is not None and prev[side] is None:
                attach += 1                       # the mock attached an object to a closing hand (a held grasp)
        prev = dict(env.backend.held)
        if info["episode_over"]:
            break
    outcomes = env.backend.outcomes(FIXED_MAPPING)
    return {"t": float(info["t"]), "n_obj": len(env.backend.objects), "attach": attach,
            "correct": sum(o == "correct" for o in outcomes), "clips": info["safety_summary"]["clipped_steps"],
            "lines": tuple(lines)}


class TestTiming(unittest.TestCase):
    def test_close_timing_off_is_the_stock_hold(self):
        self.assertEqual(close_timing(load_config(overlay="cand_r17")), (False, 1.2, 1.2))

    def test_close_timing_adaptive_is_min_0_4_timeout_1_2(self):
        self.assertEqual(close_timing(load_config(overlay="ab_fasthold")), (True, 0.4, 1.2))

    def test_planned_close_hold_is_the_timeout_in_both(self):
        """The plan always holds up to the timeout; the stall only ends it early (policy._lift_on_stall)."""
        from aaltonomous.scene.scene import Scene
        sc = Scene.load()
        L = sc.cfg["layout"]
        d = Detection("yellow", sc.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]),
                                                  sc.object_height()]), 0, 0, "depth")
        for ov in ("cand_r17", "ab_fasthold"):
            cfg = load_config(overlay=ov)
            pl = Planner(cfg, sc, hand_postures(cfg), home_arm(cfg))
            p = pl.plan(d, pl.home.copy(), pl.post["open"].copy())
            self.assertTrue(p.ok, p.reason)
            a, b = p.close_hold
            self.assertEqual(b - a, int(round(1.2 / pl.dt)), ov)

    def test_stall_lift_respects_min_and_timeout(self):
        pol = make_policy("ab_fasthold")
        n, hs, he = 60, 10, 46                  # hold = ticks [10, 46) = 1.2 s
        min_ticks = int(round(0.4 * HZ))

        def run(hand_moves):
            pol.queue_a = np.zeros((n, 14))
            pol.queue_h = np.zeros((n, 14))
            pol._exec = None
            pol._stall = {"n": n, "hold": (hs, he), "sl": ArmKinematics.sl("left"), "min": min_ticks, "rad": 0.01,
                          "need": 3, "last": None, "still": 0, "t": 0.0}
            cut_at = None
            for k in range(n):
                q = np.full(14, 0.05 * k if hand_moves else 0.0)     # 0.05 rad per tick > stall_rad 0.01
                pol._lift_on_stall({"hand_q": q, "t": k / HZ})
                if pol._stall is None:
                    cut_at = k
                    break
                pol.queue_a, pol.queue_h = pol.queue_a[1:], pol.queue_h[1:]   # one tick popped per act()
            return cut_at, len(pol.queue_a)

        cut, _ = run(hand_moves=False)          # fingers stopped: lift early, but never before the 0.4 s minimum
        self.assertIsNotNone(cut)
        self.assertGreaterEqual(cut - hs, min_ticks - 1)
        self.assertLess(cut, he)
        cut, _ = run(hand_moves=True)           # fingers never stall: the watcher ends only at the planned end of hold
        self.assertGreaterEqual(cut, he)

    def test_adaptive_settle_bounds_and_flag_off(self):
        pol = make_policy("ab_fasthold")
        mn, mx = 0.1, 0.3
        self.assertEqual(pol._settle_s(), mx)                       # no history: not known to be still
        still = [(i / 30.0, np.zeros(14)) for i in range(3)]
        pol._arm_hist = still
        self.assertEqual(pol._settle_s(), mn)
        pol._arm_hist = [(i / 30.0, np.full(14, 0.1 * i)) for i in range(3)]   # moving at 3 rad/s
        self.assertEqual(pol._settle_s(), mx)
        rng = np.random.default_rng(0)
        for _ in range(50):                                          # never outside [min, max]
            pol._arm_hist = [(i / 30.0, rng.normal(size=14) * 0.01) for i in range(3)]
            self.assertIn(pol._settle_s(), (mn, mx))
        off = make_policy("cand_r17")
        self.assertEqual(off._settle_s(), float(off.cfg["motion"]["settle_s"]))
        self.assertEqual(off._arm_hist, [])                         # nothing recorded when the flag is off

    def test_bad_timing_params_fail_at_load(self):
        bad = [{"motion.settle_min_s": 0.5}, {"motion.settle_max_s": 0.0}, {"motion.settle_vel_eps": -1.0},
               {"motion.settle_min_s": float("nan")}, {"motion.settle_window_samples": 1},
               {"grasp.adaptive_close_hold_s": 1.5}, {"grasp.lift_on_stall_timeout_s": 0.0}]
        for overrides in bad:
            with self.assertRaises(ValueError, msg=str(overrides)):
                load_config(overlay="cand_r17", overrides=overrides)

    def test_flags_off_are_stock_values(self):
        off, on = load_config(overlay="cand_r17"), load_config(overlay="ab_fasthold")
        self.assertFalse(off["grasp"]["adaptive_close"])
        self.assertFalse(off["motion"]["conditional_settle"])
        same = copy.deepcopy(on)
        same["grasp"]["adaptive_close"] = False
        same["motion"]["conditional_settle"] = False
        same["overlay"] = off["overlay"]
        self.assertEqual(same, off)


class TestMockWorld(unittest.TestCase):
    def test_three_seeds_time_saved_and_no_held_regression(self):
        rows = []
        for s in SEEDS:
            off, on = episode("cand_r17", s), episode("ab_fasthold", s)
            rows.append((s, off, on))
            self.assertEqual(on["clips"], 0)
            self.assertEqual(on["n_obj"], off["n_obj"], s)
            self.assertEqual(on["attach"], off["attach"], f"held seed {s}")          # held grasps: no regression
            self.assertEqual(on["correct"], off["correct"], f"placed seed {s}")
        saved = [(off["t"] - on["t"]) / off["n_obj"] for _, off, on in rows]
        print("\n[fasthold] seed  objects  held(off/on)  placed(off/on)  t_off  t_on  s_saved/object")
        for (s, off, on), sv in zip(rows, saved):
            print(f"[fasthold] {s:>4}  {off['n_obj']:>7}  {off['attach']}/{on['attach']:<10}  "
                  f"{off['correct']}/{on['correct']:<12}  {off['t']:5.1f}  {on['t']:5.1f}  {sv:.2f}")
        print(f"[fasthold] mean seconds saved per object: {np.mean(saved):.2f}")
        self.assertGreater(float(np.mean(saved)), 0.0)

    def test_stall_lifts_in_episodes_stay_in_bounds(self):
        import re
        lifts = [ln for s in SEEDS for ln in episode("ab_fasthold", s)["lines"] if "[lift_on_stall]" in ln]
        self.assertTrue(lifts)
        for ln in lifts:
            m = re.search(r"lift after ([\d.]+) s of hold.*skipped ([\d.]+) s", ln)
            self.assertIsNotNone(m, ln)
            after, skipped = float(m.group(1)), float(m.group(2))
            self.assertGreaterEqual(after, 0.4 - 1e-6, ln)          # never before the adaptive minimum
            self.assertLessEqual(after + skipped, 1.2 + 1e-6, ln)   # never past the timeout


if __name__ == "__main__":
    unittest.main()
