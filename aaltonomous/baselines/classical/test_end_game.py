"""End-game rules (decision D2: T-20 s no new pick, T-8 s release over the bin or retract) on mock_world episodes
with a shortened time limit, for the three flag overlays (submission, rc1, cand_r18).

Two variants per overlay:
  shipped  the overlay exactly as it is (policy.object_time_cap is false in all three: the rules are inert)
  forced   the same overlay + policy.object_time_cap = true (checks the logic the overlay would switch on)

Deadline placement. The stock admission check (_has_time) only starts a plan that ends before the limit, so the
deadline cannot land inside a plan by itself. Scenarios a-d therefore shorten the limit once the first plan has
started: policy.time_limit and env.time_limit are both set to now + (ticks to the target phase)/HZ, so the DEADLINE
lands in the chosen phase (approach, grasp close, carry, retreat). T-8 is then whatever the policy is doing 8 s
before that: often already past when the limit is shortened, in which case the rule must act at once. Scenario e
starts a 25 s deadline at the first look with objects on the counter.
Once the real deadline is reached the sim runs on for POST_S seconds (env.time_limit extended, harness only) so the
replacement motion can finish; the end-state check uses the state AT the deadline.

T-8 is checked only when the running plan would be cut by the deadline (its end is after T): a plan that finishes
before T needs no action. Failing tests here are real policy gaps, not test errors: their messages start with the check
name and the timeline. Extra cases: f (deadline as the hand opens; T-8 falls in approach, so the rule acts at once)
and g (deadline learned while carrying near the bin, 1 s before T: the release-over-bin branch).
Run: PYTHONPATH=. python -m unittest aaltonomous.baselines.classical.test_end_game -v
"""
import unittest

import numpy as np

from . import decision as dec
from .config import load_config
from .kinematics import ArmKinematics
from .mock_world import make_mock_env
from .policy import HZ, ClassicalPolicy

OVERLAYS = ("submission", "rc1", "cand_r18")
VARIANTS = {"shipped": {}, "forced": {"policy.object_time_cap": True}}
# scenario -> (phase the DEADLINE lands in, fraction through that phase)
SCENARIOS = {
    "a_approach": ("to_pregrasp", 0.5),
    "b_grasp_close": ("close", 0.5),
    "c_carry_mid": ("carry", 0.5),
    "d_right_after_place": ("retreat", 0.1),
    "e_25s_left": ("start", 25.0),
    "f_release_over_bin": ("open", 0.0),   # deadline as the hand opens over the bin: T-8 is in the carry
    "g_release_now": ("carry_now", 0.9),   # deadline learned in the carry near the bin (T-8 already past): release
}
T0 = 200.0                          # generous deadline until injection: the first plan is admitted as in a real run
POST_S = 12.0                       # sim seconds after the real deadline, so the replacement motion can finish
NO_NEW_S = dec.DEFAULTS["no_new_s"]
RELEASE_S = dec.DEFAULTS["release_s"]
NOT_NEEDED = (None, "open", "retreat", "home")   # phases where the T-8 rule has nothing to do

_CACHE = {}


def _last_target(pol):
    """Last commanded (arm, hand) target: the end of the queue, or the current command when the queue is empty."""
    q, h = pol.queue_a, pol.queue_h
    arm = q[-1] if q is not None and len(q) else pol.arm_cmd
    hand = h[-1] if h is not None and len(h) else pol.hand_cmd
    return np.asarray(arm, float), np.asarray(hand, float)


class Episode:
    """One mock episode driven tick by tick; records what the policy did (its events, targets and state)."""

    def __init__(self, scn, overlay, variant, seed=0):
        self.scn, self.overlay, self.variant = scn, overlay, variant
        self.phase, self.frac = SCENARIOS[scn]
        self.cfg = load_config(overlay=overlay, overrides=VARIANTS[variant])
        self.env = make_mock_env(seed=seed, cfg=self.cfg)
        self.lines = []
        self.pol = ClassicalPolicy(config=self.cfg, safety=self.env.filter.cfg, time_limit=T0,
                                   log=self.lines.append, threaded=False, require_confirm=False)
        self.starts, self.cur_plan, self.cur_start, self.T = [], None, None, T0
        self.inj, self.inj_done, self.remaining_at_inj = None, False, None
        self.at_t8, self.at_deadline, self.final = None, None, None
        self.exc, self.safety = None, []
        self.nonfinite, self.worst_arm, self.worst_hand = 0, 0.0, 0.0
        orig_start = self.pol._start

        def start(plan, t):   # record every pick the policy starts, with the deadline it was started under
            self.starts.append({"t": round(float(t), 2), "T": self.pol.time_limit, "color": plan.color,
                                "side": plan.side, "end": round(float(t) + len(plan.arm) / HZ, 2)})
            self.cur_plan, self.cur_start = plan, float(t)
            return orig_start(plan, t)
        self.pol._start = start

    # ------------------------------------------------------------------ helpers
    def _phase(self):
        plan, q = self.cur_plan, self.pol.queue_a
        if plan is None or q is None:
            return None
        return dec.phase_at(plan.phases, len(plan.arm) - len(q))

    def _plan_end(self):
        if self.cur_plan is None or self.cur_start is None:
            return None
        return self.cur_start + len(self.cur_plan.arm) / HZ

    def _over_bin(self):
        plan = self.cur_plan
        return bool(plan is not None and plan.target_bin in self.env.scene.bin_names and self.pol._over_target_bin(plan))

    def _fired(self):
        return bool(self.pol._exec is not None and self.pol._exec.get("t8"))

    def _on_counter(self):
        sc = self.env.scene
        boxes = [sc.bin_inner_box(b) for b in sc.bin_names]
        n = 0
        for ob in self.env.backend.objects:
            pm = sc.pelvis_to_mat(ob["pos"])
            inside = any(np.all(pm[:2] >= lo[:2]) and np.all(pm[:2] <= hi[:2]) for lo, hi in boxes)
            n += 0 if inside else 1
        return n

    def _set_deadline(self, T):
        self.T = T
        self.pol.time_limit = T
        self.env.time_limit = T

    def _inject(self, t):
        if self.inj_done:
            return
        if self.phase == "start":   # e: 25 s left at the first look with objects on the counter
            if self.pol.state != "look" or self._on_counter() == 0:
                return
            self.inj_done, self.remaining_at_inj = True, self._on_counter()
            self.inj = {"t": round(t, 2), "phase": "look", "T": round(t + 25.0, 2)}
            self._set_deadline(t + 25.0)
            return
        plan, q = self.cur_plan, self.pol.queue_a
        if plan is None or q is None or not len(q):
            return
        name = self.phase.replace("_now", "")
        spans = {n: (s, e) for n, s, e in plan.phases}
        s, e = spans[name]
        c = len(plan.arm) - len(q)                  # current tick of the plan
        if self.phase.endswith("_now"):             # learned during the phase: the limit is 1 s away
            if not (s + self.frac * (e - s) <= c < e):
                return
            self.inj_done = True
            self.inj = {"t": round(t, 2), "phase": name, "tick": c, "mode": "now", "T": round(t + 1.0, 2)}
            self._set_deadline(t + 1.0)
            return
        self.inj_done = True
        k = s + self.frac * (e - s)                  # target tick inside the phase
        if k <= c:
            self.inj = {"t": round(t, 2), "phase": self.phase, "skipped": True, "tick": c, "target_tick": k}
            return
        T = t + (k - c) / HZ                        # the deadline lands in the target phase
        self.inj = {"t": round(t, 2), "phase": self.phase, "tick": c, "target_tick": round(k, 1), "T": round(T, 2)}
        self._set_deadline(T)

    def _snapshot(self, t):
        pol = self.pol
        held = [k for k, v in self.env.backend.held.items() if v is not None]
        arm_end, _ = _last_target(pol)
        retracting = self._fired() and bool(np.allclose(arm_end[self._sl()], pol.home[self._sl()], atol=1e-6))
        return {"t": round(t, 2), "state": pol.state, "phase": self._phase(), "fired": self._fired(),
                "held": held, "retracting": bool(retracting), "T": self.T}

    def _sl(self):
        return ArmKinematics.sl(self.cur_plan.side if self.cur_plan is not None else "left")

    def summary(self):
        ev = [m for m in self.pol.events][-8:]
        return (f"overlay={self.overlay} variant={self.variant} scenario={self.scn} inj={self.inj} "
                f"remaining_at_inj={self.remaining_at_inj} at_T-8={self.at_t8} at_deadline={self.at_deadline} "
                f"final={self.final} starts={self.starts} last_events={ev} exc={self.exc}")

    # ------------------------------------------------------------------ run
    def run(self):
        env, pol = self.env, self.pol
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        self._set_deadline(T0)
        self.prev_arm = np.asarray(obs["arm_q"], float)
        self.prev_hand = np.asarray(obs["hand_q"], float)
        arm_b = float(self.cfg["motion"]["arm_max_vel"]) / HZ
        hand_b = float(self.cfg["motion"]["hand_max_vel"]) / HZ
        self.bounds = (arm_b, hand_b)
        max_ticks = int((T0 + POST_S + 60.0) * HZ)
        try:
            for _ in range(max_ticks):
                t = float(obs["t"])
                self._inject(t)
                first8 = self.inj_done and self.at_t8 is None and self.T is not None and t >= self.T - RELEASE_S
                ph8 = self._phase() if first8 else None
                act = pol.act(obs)
                if first8:
                    self.at_t8 = {"t": round(t, 2), "T": self.T, "phase": ph8, "fired": self._fired(),
                                  "over_bin": self._over_bin() if ph8 == "carry" else None,
                                  "plan_end": self._plan_end()}
                self._check_targets(act, arm_b, hand_b)
                obs, info = env.step(act)
                if info["safety_mode"] in ("aborted", "safe_stop") or info["end_reason"] in ("aborted", "safe_stop"):
                    self.safety.append((round(info["t"], 2), info["safety_mode"], info["end_reason"]))
                if info["episode_over"]:
                    if info["end_reason"] == "time_limit" and self.at_deadline is None:
                        self.at_deadline = self._snapshot(info["t"])
                        env.time_limit = info["t"] + POST_S      # harness only: let the replacement motion finish
                        env.episode_over = False
                        continue
                    break
            else:
                self.exc = "tick budget exhausted"
        except Exception as e:  # any exception is a failed check, with the message kept
            self.exc = f"{type(e).__name__}: {e}"
        if self.at_deadline is None:
            self.at_deadline = self._snapshot(float(obs["t"]))
        sl = self._sl()
        self.final = {"held": [k for k, v in env.backend.held.items() if v is not None],
                      "hand_open": bool(np.allclose(pol.hand_cmd[sl], pol.post["open"][sl], atol=1e-6)),
                      "arm_home": bool(np.allclose(pol.arm_cmd[sl], pol.home[sl], atol=1e-6))}

    def _check_targets(self, act, arm_b, hand_b):
        a = np.asarray(act["arm_q_target"], float)
        h = np.asarray(act["hand_q_target"], float)
        if not (np.isfinite(a).all() and np.isfinite(h).all()):
            self.nonfinite += 1
            return
        da = np.abs(np.diff(np.vstack([self.prev_arm, a]), axis=0)).max()
        dh = np.abs(np.diff(np.vstack([self.prev_hand, h]), axis=0)).max()
        self.worst_arm = max(self.worst_arm, float(da) / arm_b)
        self.worst_hand = max(self.worst_hand, float(dh) / hand_b)
        self.prev_arm, self.prev_hand = a[-1].copy(), h[-1].copy()


def episode(scn, overlay, variant):
    key = (scn, overlay, variant)
    if key not in _CACHE:
        ep = Episode(scn, overlay, variant)
        ep.run()
        _CACHE[key] = ep
    return _CACHE[key]


# ---------------------------------------------------------------------------------------------- checks (ok, message)
def _c_scenario_reached(e):
    ok = e.inj_done and not (e.inj or {}).get("skipped")
    if e.phase == "start":
        ok = ok and (e.remaining_at_inj or 0) >= 1
    return ok, f"scenario not reached (deadline never placed / no objects left for e): {e.summary()}"


def _c_no_exception(e):
    ok = e.exc is None and not e.pol._dec_warned
    return ok, f"exception={e.exc} swallowed_decision_errors={sorted(e.pol._dec_warned)}: {e.summary()}"


def _c_no_safety_stop(e):
    return not e.safety, f"safety stop / abort: {e.safety}: {e.summary()}"


def _c_no_new_pick_after_t20(e):
    bad = [s for s in e.starts if s["t"] >= s["T"] - NO_NEW_S]
    return not bad, (f"pick(s) started inside the last {NO_NEW_S:.0f} s (t, deadline in force at start): "
                     f"{[(s['t'], s['T'], s['color']) for s in bad]}: {e.summary()}")


def _c_commands_finite_and_rate(e):
    ok = e.nonfinite == 0 and e.worst_arm <= 1.0 + 1e-9 and e.worst_hand <= 1.0 + 1e-9
    return ok, (f"nonfinite_chunks={e.nonfinite} worst_arm_step/(arm_max_vel*dt)={e.worst_arm:.3f} "
                f"worst_hand_step/(hand_max_vel*dt)={e.worst_hand:.3f}: {e.summary()}")


def _c_t8_release_or_retract(e):
    a = e.at_t8
    if a is None:   # the episode ended before T-8: no plan was running then
        return True, ""
    ph = a["phase"]
    cut = a["plan_end"] is not None and a["plan_end"] > a["T"]     # the deadline would cut the running plan
    ok = (not cut) or ph in NOT_NEEDED or a["fired"]
    return ok, (f"at T-8 (t={a['t']}) the running plan was in phase {ph!r}, would end at {a['plan_end']} after the "
                f"deadline {a['T']:.2f}, and the policy did not release/retract (fired={a['fired']}): {e.summary()}")


def _c_end_state_released_or_retracted(e):
    d = e.at_deadline or {}
    ok_dl = not d.get("held") or d.get("retracting")
    f = e.final or {}
    ok_fin = not f.get("held") and f.get("hand_open") and f.get("arm_home")
    return ok_dl and ok_fin, (f"at deadline={d} final(after replacement)={f}: {e.summary()}")


SAFETY_CHECKS = {
    "no_exception": _c_no_exception,
    "no_safety_stop": _c_no_safety_stop,
    "commands_finite_and_rate": _c_commands_finite_and_rate,
}
RULE_CHECKS = {   # T-20 / T-8 rules: only meaningful with policy.object_time_cap = true
    "scenario_reached": _c_scenario_reached,
    "no_new_pick_after_T20": _c_no_new_pick_after_t20,
    "t8_release_or_retract": _c_t8_release_or_retract,
    "end_state_released_or_retracted": _c_end_state_released_or_retracted,
}
CHECKS = {**SAFETY_CHECKS, **RULE_CHECKS}


def _make_case(scn, overlay, variant):
    checks = CHECKS if variant == "forced" else SAFETY_CHECKS   # shipped: safety checks only (gap 1)

    def make(check):
        fn = checks[check]

        def test(self):
            ok, msg = fn(episode(scn, overlay, variant))
            self.assertTrue(ok, f"[{check}] {msg}")
        return test
    attrs = {f"test_{c}": make(c) for c in checks}
    name = f"EndGame_{scn}__{overlay}__{variant}"
    return type(name, (unittest.TestCase,), attrs)


# Runtime budget (< 60 s): rule checks on every overlay (forced); shipped safety checks on submission only.
VARIANT_OVERLAYS = {"forced": OVERLAYS, "shipped": ("submission",)}


def _register():
    for scn in SCENARIOS:
        for var, ovs in VARIANT_OVERLAYS.items():
            for ov in ovs:
                cls = _make_case(scn, ov, var)
                globals()[cls.__name__] = cls


def _register_flag_test():
    """Shipped flag values: a later flip of policy.object_time_cap shows up here (see gap 1 in the report)."""
    def test_shipped_object_time_cap_off(self):
        for ov in OVERLAYS:
            cfg = load_config(overlay=ov)
            self.assertIs(cfg["policy"]["object_time_cap"], False,
                          f"[{ov}] policy.object_time_cap is {cfg['policy']['object_time_cap']!r}: the T-20/T-8 rules "
                          f"are inert while False (gap 1); the forced-variant tests cover the logic")
    cls = type("EndGame_ShippedFlags", (unittest.TestCase,), {"test_shipped_object_time_cap_off": test_shipped_object_time_cap_off})
    globals()[cls.__name__] = cls


_register()
_register_flag_test()
