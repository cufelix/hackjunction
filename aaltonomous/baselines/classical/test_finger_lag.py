"""Dex3 finger lag (300-380 ms): R44 close hold >= 0.8 s + release wait >= 0.5 s, R49 release confirmation.
Mock world with a 350 ms first-order finger lag (RenderMockBackend hand_tau). Virtual time, short episodes.
    python -m unittest aaltonomous.baselines.classical.test_finger_lag -v
"""
import sys
import time
import unittest
from pathlib import Path

import numpy as np

from .config import hand_postures, load_config
from .mock_world import make_mock_env
from .policy import HZ, ClassicalPolicy

FINALS = ("finals_r18", "finals_r19", "finals_r19b", "finals_r19c", "finals_r20", "finals_r21", "finals_real")
OVERLAY, TAU, TIME_LIMIT = "finals_r21", 0.35, 45.0
# the mock never blocks the fingers, so finger_sum reads every close as empty and aborts the carry: use the stall detector
STALL = {"grasp.held_detector": "stall",
         # integ-all: depth-less mock, nominal counter: pin the rc7 real-table prior / height phase / async vote / near-bin hedge
         "scene.counter_z_prior": None, "calib.plane.apply_height": False, "bin.vote.enabled": False,
         "perception.color_gate.near_margin_m": 0.0, "motion.output_caps": None}   # M4 limiter lags the plan by a few ticks
OFF = {"grasp.release_wait_s": None, "grasp.release_confirm": False}


def run(seed=0, overlay=OVERLAY, overrides=None, tau=TAU, sticky=None, time_limit=TIME_LIMIT):
    """One episode. Returns dict(ev=[(kind, t_or_msg)], starts, cmds=[(arm, hand)] per tick, pol)."""
    overrides = {**STALL, **(overrides or {})}
    env = make_mock_env(seed=seed, hand_tau=tau)
    ev, cmds, starts = [], [], []
    pol = ClassicalPolicy(config=load_config(overlay=overlay, overrides=overrides),
                          safety=env.filter.cfg, time_limit=time_limit, log=lambda m: ev.append(("log", str(m))))
    b = env.backend
    adv = b.advance
    state = {"engaged": False, "seen_rel": False, "done": False}
    post = hand_postures(pol.cfg)

    def advance(dt):
        prev = b.hand_q.copy()
        adv(dt)
        cmds.append((b.arm_cmd.copy(), b.hand_cmd.copy()))
        if sticky and not state["done"]:   # the object stays wedged in the fingers until a second, different open
            side = 7 if b.held["right"] is not None else 0
            held = b.held["left"] is not None or b.held["right"] is not None
            sl = slice(side, side + 7)
            rel = np.abs(b.hand_cmd[sl] - post["release"][sl]).max() < 1e-6
            if not state["engaged"] and held and b._closure("left" if side == 0 else "right") < 0.95 and not rel:
                state["engaged"] = True
            if state["engaged"]:
                state["seen_rel"] |= rel
                if state["seen_rel"] and not rel and sticky != "forever":
                    state["done"] = True           # the wiggle: the object lets go
                else:
                    b.hand_q = prev                # fingers pinned
    b.advance = advance
    begin = pol._start

    def start(plan, t):
        starts.append((float(t), plan, len(cmds)))
        ev.append(("start", float(t)))
        return begin(plan, t)
    pol._start = start
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"], "time_limit": time_limit})
    t0, n = time.monotonic(), 0
    while n < 20000 and time.monotonic() - t0 < 120:
        job = getattr(pol, "_job", None)
        if job is not None and job.get("result") is None:
            time.sleep(0.002)
            continue
        obs, info = env.step(pol.act(obs))
        n += 1
        if info["episode_over"]:
            break
    return {"ev": ev, "starts": starts, "cmds": cmds, "pol": pol, "n": n, "info": info}


def per_pick_s(r):
    t = [s[0] for s in r["starts"]]
    return float(np.mean(np.diff(t))) if len(t) > 1 else float("nan")


class TestFingerLag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lag = run()

    def test_close_hold_at_least_0p8_s_in_every_finals_overlay(self):    # R44a
        for ov in FINALS:
            cfg = load_config(overlay=ov)
            self.assertGreaterEqual(cfg["motion"]["close_hold_s"], 0.8, ov)
            self.assertEqual(cfg["grasp"]["release_wait_s"], 0.5, ov)
            self.assertTrue(cfg["grasp"]["release_confirm"], ov)
        self.assertTrue(self.lag["starts"], "no pick started")
        for _, plan, _ in self.lag["starts"]:
            ph = {n: (s, e) for n, s, e in plan.phases}
            self.assertGreaterEqual((ph["close"][1] - ph["close"][0]) / HZ, 0.8 - 1e-9)   # close ramp + hold before the lift
            self.assertEqual(ph["close"][1], ph["lift"][0])

    def test_retract_starts_at_least_0p5_s_after_the_open_command(self):   # R44b, on the commands actually sent
        n_checked = 0
        for _, plan, n0 in self.lag["starts"]:
            ph = {n: (s, e) for n, s, e in plan.phases}
            if "open" not in ph or plan.side not in ("left", "right"):
                continue
            self.assertGreaterEqual((ph["retreat"][0] - ph["open"][0]) / HZ, 0.5 - 1e-9)   # plan
            cmds = self.lag["cmds"]
            if n0 + ph["retreat"][0] + 90 >= len(cmds):
                continue
            a = np.array([c[0] for c in cmds[n0:n0 + ph["retreat"][0] + 90]])
            open_t = ph["open"][0]
            moved = np.where(np.abs(np.diff(a[open_t:], axis=0)).max(axis=1) > 1e-9)[0]
            self.assertTrue(len(moved))
            self.assertGreaterEqual(moved[0] / HZ, 0.5 - 2.0 / HZ)   # first arm motion after the open command
            n_checked += 1
        self.assertGreater(n_checked, 0)

    def test_release_confirmed_without_reopen_when_the_fingers_open(self):
        logs = [m for k, m in self.lag["ev"] if k == "log" and "[release_confirm]" in m]
        self.assertTrue(logs)
        self.assertFalse([m for m in logs if "re-open" in m or "stop" in m], logs)

    def test_sticky_object_one_reopen_and_no_pick_until_empty(self):       # R49
        r = run(sticky="once", time_limit=60.0)
        ev = r["ev"]
        idx = [i for i, (k, m) in enumerate(ev) if k == "log" and "re-open once" in m]
        self.assertEqual(len(idx), 1, [m for k, m in ev if k == "log" and "release_confirm" in m])
        done = [i for i, (k, m) in enumerate(ev) if k == "log" and "hand empty" in m and "(re-opened)" in m]
        self.assertEqual(len(done), 1)
        self.assertGreater(done[0], idx[0])
        self.assertFalse([i for i, (k, _) in enumerate(ev) if k == "start" and idx[0] < i < done[0]])
        self.assertTrue([i for i, (k, _) in enumerate(ev) if k == "start" and i > done[0]], "no pick after the empty hand")

    def test_sticky_forever_goes_home_and_stops_picking(self):
        r = run(sticky="forever", time_limit=60.0)
        ev = r["ev"]
        self.assertEqual(len([1 for k, m in ev if k == "log" and "re-open once" in m]), 1)
        stop = [i for i, (k, m) in enumerate(ev) if k == "log" and "-> home, stop" in m]
        self.assertEqual(len(stop), 1)
        self.assertFalse([i for i, (k, _) in enumerate(ev) if k == "start" and i > stop[0]])

    def test_flags_off_is_identical(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
        from equivalence_check import episode_digest
        for seed in (0, 1):
            self.assertEqual(episode_digest(seed, 60.0), episode_digest(seed, 60.0, dict(OFF)))


if __name__ == "__main__":
    unittest.main()
