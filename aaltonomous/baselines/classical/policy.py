"""ClassicalPolicy: obs -> action wrapper around perception + planner (rung 1, sort_fixed).

    pol = ClassicalPolicy()                   # config.yaml, scene.yaml, calibration.yaml
    pol.reset({"instruction": obs["instruction"]})
    while True:
        action = pol.act(obs)                 # {"arm_q_target": (k,14), "hand_q_target": (k,14), "done": bool}
        obs, info = env.step(action)

Behaviour: go home -> settle -> look (head RGB-D) -> plan one object -> execute -> back home -> look again ...
until nothing sortable is left (then `done`). While it is planning (background thread when `threaded=True`) or
waiting for an operator confirm, act() returns 1-tick HOLD actions (the last target), so the env's watchdog
never fires. Targets are absolute, planned at <= motion.arm_max_vel and pre-checked against a SafetyFilter.

Operator confirm (real robot): ClassicalPolicy(require_confirm=True) asks before the first move to home and before
each object; the runner calls confirm() / skip() / stop() while `pol.waiting` is True (`pol.pending_text` says what).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.tasks import FIXED_MAPPING
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene, resolve_bin

from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .perception import Perception
from .planner import Planner

HZ = 30.0


def _lower_thread_priority(nice=10):
    """Linux: raise the nice value of the calling thread only (setpriority on its native thread id)."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except (AttributeError, OSError):
        pass


class ClassicalPolicy:
    def __init__(self, config=None, scene=None, calibration=None, safety: SafetyConfig | None = None,
                 require_confirm=False, threaded=False, time_limit=120.0, log=print, overrides=None):
        self.cfg = config if isinstance(config, dict) else load_config(config, overrides)
        self.scene = scene if isinstance(scene, Scene) else Scene.load(scene)
        self.cal = calibration if isinstance(calibration, Calibration) else Calibration.load(calibration)
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)
        self.home = home_arm(self.cfg, self.spec)
        self.kin = ArmKinematics(self.cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=1.0 / HZ)
        self.perception = Perception(self.cfg, self.scene, self.cal)
        self.planner = Planner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec)
        self.require_confirm = require_confirm
        self.threaded = threaded
        if threaded:  # pure-Python planning holds the GIL: hand it back to act() sooner (default 5 ms)
            sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
        self.time_limit = time_limit
        self.log = log or (lambda *a: None)
        self.reset()

    # ------------------------------------------------------------------ API
    def _instruction_mapping(self, info):
        """Instruction text -> mapping for `scene.resolve_bin`, or None to fall back to FIXED.

        Off unless `instruction.resolve` is true in config.yaml, so the reference measurement is
        unchanged by default. NEVER raises: reset() is under a 1.0 s watchdog (safety.py:69) and an
        exception here would end the episode as a BAD_END, which is worse than any wrong mapping.
        The resolver is pure stdlib and measured at ~0.13 ms per sentence.
        """
        try:                                        # the WHOLE body, not just the resolve call
            if not (self.cfg.get("instruction") or {}).get("resolve", False):
                return None
            text = (info or {}).get("instruction") or ""
            if not text:
                return None
            from .instruction import resolve
            return resolve(text) or None
        except Exception as e:                      # a parse bug must not cost the whole episode
            self.log(f"instruction resolve failed ({e}), using FIXED_MAPPING")
            return None

    def reset(self, info=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = info or {}
        self.mapping = info.get("mapping") or self._instruction_mapping(info) or dict(FIXED_MAPPING)
        self.planner.mapping = self.mapping
        self.time_limit = info.get("time_limit", self.time_limit)
        self.state = "init"
        self.queue_a = self.queue_h = None
        self.arm_cmd = self.hand_cmd = None
        self.attempts = []          # (position, count) of objects we tried to pick
        self.failed = []            # positions whose plans failed
        self.pending = None         # Plan waiting for the operator
        self.pending_text = ""      # what the operator is asked to confirm
        self.waiting = False
        self.events = []
        self.plans = []
        self._job = None
        self._decision = None
        self.last_detections = []
        self.mapping_conf = dict(info.get("mapping_conf") or {})   # colour -> confidence (default 1.0), triage tie-break
        self.too_long = []          # (position, plan duration) of feasible plans that did not fit the time left
        self.n_started = 0          # executed plans started this episode, whatever object they were for
        self._summary_logged = False
        self.perception.reset_calibration()
        self.calibration = None
        self._calibration_done = False
        self.last_bin_detections = []
        self.used_drop_slots = {name: set() for name in self.scene.bin_names}
        self.reserved_drop_slots = {name: set() for name in self.scene.bin_names}
        self.verification_pending = None
        self.last_verification = None
        self.retry_colors = set()
        self.retry_targets = []
        self.verify_relooks_remaining = 0
        self.scheduled_corrections = []

    def confirm(self):
        self._decision = "go"

    def skip(self):
        self._decision = "skip"

    def stop(self):
        self._decision = "stop"

    def act(self, obs):
        if self.arm_cmd is None:  # first call: start from the measured pose
            # Clamp into the URDF limits: real Dex3 fingers at rest read slightly past 0 (2026-11-11), and holding
            # that measured pose would be clipped every tick -> safety abort before the first move.
            sp = self.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        t = float(obs.get("t", 0.0))
        if not self._calibration_done and (self.cfg["perception"].get("self_calibration") or {}).get("enabled", False):
            started = time.perf_counter()
            try:
                cal = self.perception.calibrate_bins(obs.get("head_depth"))
            except Exception as exc:  # calibration is optional; never let it terminate the control loop
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                cal = self.perception.calibration_failure(
                    f"{type(exc).__name__}: {exc}", elapsed_ms=elapsed_ms)
            if cal.valid_frame:
                self._calibration_done = True
                self.calibration = cal
                off = cal.xy_offset * 1000.0
                state = "accepted" if cal.success else "fallback"
                self._note(t, f"head calibration {state}: xy=({off[0]:+.1f},{off[1]:+.1f}) mm, "
                              f"confidence={cal.confidence:.2f}, residual={cal.residual_m * 1000.0:.1f} mm, "
                              f"points={cal.n_points}, time={cal.elapsed_ms:.1f} ms ({cal.reason})")
        if self.queue_a is not None and len(self.queue_a):
            return self._pop()
        if self.state == "init":
            a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, self.home, self.post["open"])
            self._home_move = (a, h)
            dq = np.abs(self.home - self.arm_cmd).max()
            self.pending_text = f"move both arms to home_arm_q (max joint change {dq:.2f} rad, {len(a) / HZ:.1f} s)"
            if self.require_confirm:
                self.state, self.waiting, self.pending = "await_home", True, None
                self._note(t, f"waiting for confirm: {self.pending_text}")
                return self._hold()
            self.state = "home"
        if self.state == "await_home":
            d, self._decision = self._decision, None
            if d in ("skip", "stop"):
                self.waiting, self.state = False, "done"
                self._note(t, "operator stop before the first move")
                return self._hold(done=True)
            if d != "go":
                return self._hold()
            self.waiting, self.state = False, "home"
        if self.state == "home":
            a, h = self._home_move
            self._enqueue(a, h)
            self.state = "settle"
            self._note(t, f"to home ({len(a) / HZ:.1f} s)")
            return self._pop() if len(a) else self._hold()
        if self.state == "settle":
            n = int(round(self.cfg["motion"]["settle_s"] * HZ))
            self._enqueue(np.repeat(self.arm_cmd[None], n, 0), np.repeat(self.hand_cmd[None], n, 0))
            self.state = "look"
            return self._pop()
        if self.state == "look":
            cap = self.cfg["policy"].get("max_total_attempts")
            if cap is not None and self.n_started >= int(cap):   # knocked objects reappear as 'new': hard stop
                self._note(t, f"attempt cap reached ({self.n_started} >= max_total_attempts {int(cap)}): done")
                self.state = "done"
                return self._hold(done=True)
            if self.threaded:
                if self._job is None:
                    self._job = {"obs": obs, "result": None}
                    threading.Thread(target=self._look_job, args=(self._job,), daemon=True).start()
                if self._job["result"] is None:
                    return self._hold()
                res, self._job = self._job["result"], None
            else:
                res = self._look(obs)
            return self._after_look(res, t)
        if self.state == "await":
            d, self._decision = self._decision, None
            if d == "go":
                self.waiting = False
                if self._verify_on() and not self._has_time(self.pending, t):
                    self.state = "done"
                    self._note(t, f"confirmation came too late for {self.pending.color}")
                    return self._hold(done=True)
                return self._start(self.pending, t)
            if d == "skip":
                self.waiting = False
                self.failed.append(self.pending.waypoints["grasp"][:2])
                self._note(t, f"operator skipped {self.pending.color}")
                self.state = "look"
                return self._hold()
            if d == "stop":
                self.waiting = False
                self.state = "done"
                self._note(t, "operator stop")
            return self._hold(done=self.state == "done")
        if self.state == "executing":  # queue drained -> back home: look again after settling
            self.state = "settle"
            return self.act(obs)
        return self._hold(done=True)

    # ------------------------------------------------------------------ look + plan
    def _look_job(self, job):
        _lower_thread_priority()  # planning must not starve act() (the control loop's thread)
        try:
            job["result"] = self._look(job["obs"])
        except Exception as e:  # never kill the control loop
            job["result"] = ("error", repr(e))

    def _look(self, obs):
        verify = self._verify_on()
        if not verify:  # flag off: the stock look (no bin review, no corrective picks, no relooks)
            dets = self.perception.detect(obs["head_rgb"], obs["head_depth"])
            self.last_detections = dets
            return self._choose(dets, obs, verify=False)
        dets, rejected = self.perception.detect(obs["head_rgb"], obs["head_depth"], debug=True, include_bins=True)
        bin_dets = [(d, why.split(":", 1)[1]) for d, why in rejected if why.startswith("in_bin:")]
        self.last_detections = dets
        self.last_bin_detections = bin_dets
        corrections = self._review_bins(bin_dets)

        # A known wrong-bin object is the largest score swing, so repair it before
        # touching a new counter object.  The fresh detection is planned directly;
        # stale corrective targets are never retained between camera frames.
        self.scheduled_corrections = corrections
        for d, source_bin, target_bin in corrections:
            for slot in self._free_drop_slots(target_bin):
                p = self.planner.plan(d, self.arm_cmd, self.hand_cmd, bin_name=target_bin, relocate=False,
                                      drop_slot=slot, source_bin=source_bin)
                self.plans.append(p)
                if p.ok:
                    return ("plan", p, len(dets))
                self.log(f"[classical] cannot correct {d.color} from {source_bin} to slot {slot}: {p.reason}")
            p = self.planner.plan_to_counter(d, self.arm_cmd, self.hand_cmd, source_bin,
                                             obstacles=[o.top[:2] for o in dets])
            self.plans.append(p)
            if p.ok:
                return ("plan", p, len(dets))
            self.log(f"[classical] cannot remove {d.color} from wrong {source_bin} bin: {p.reason}")
        if corrections:
            return ("uncorrectable", corrections, len(dets))

        fresh_retry = self.last_verification is not None and self.last_verification[0] == "retry"
        pending_retry = self.last_verification[1] if fresh_retry else None
        attributable = False
        if pending_retry is not None:
            source = pending_retry.get("source_xy")
            attributable = any(d.color == pending_retry["color"] and
                               (source is None or np.linalg.norm(d.top[:2] - source) <
                                2 * self.cfg["policy"]["same_object_radius"])
                               for d in dets)
            if attributable:
                self.verify_relooks_remaining = 0
            elif self.verify_relooks_remaining > 0:
                self.verify_relooks_remaining -= 1
                return ("relook", None, len(dets))
        return self._choose(dets, obs, verify=True)

    def _choose(self, dets, obs, verify):
        """Pick the next counter object and plan it (stock order; #3 time triage and #6 drop slots when on)."""
        cands = []
        for d in dets:
            xy = d.top[:2]
            if any(np.linalg.norm(xy - f) < self.cfg["policy"]["same_object_radius"] for f in self.failed):
                continue
            n = sum(c for p, c in self.attempts if np.linalg.norm(xy - p) < self.cfg["policy"]["same_object_radius"])
            if n >= self.cfg["policy"]["max_attempts"]:
                continue
            cands.append(d)
        if self.cfg["policy"]["order"] == "nearest_bin":
            def base(d):
                b = resolve_bin(self.mapping, d.color, getattr(d, "shape", None))
                return np.inf if b is None else np.linalg.norm(d.top[:2] - self.planner.bin_xy(b))
        else:
            def base(d):
                return -d.n_px

        def retry_prefix(d):  # #6: objects whose release was not verified go first (nearest to their source)
            if not verify:
                return ()
            retry = [np.linalg.norm(d.top[:2] - p) for color, p in self.retry_targets if color == d.color]
            retry_key = min(retry) if retry else (0.0 if d.color in self.retry_colors else np.inf)
            return (not np.isfinite(retry_key), retry_key)

        pol = self.cfg["policy"]
        triage = bool(pol.get("time_triage", False))
        if triage and pol.get("triage_order", "duration") == "priority":
            eps = float(pol.get("triage_tie_m", 0.02))   # candidates with ~equal cheap estimate: cheaper shape first
            cands.sort(key=lambda d: retry_prefix(d) + (round(base(d) / eps), self._priority(d)))
        elif verify:
            cands.sort(key=lambda d: retry_prefix(d) + (base(d),))
        else:
            cands.sort(key=base)
        t = float(obs.get("t", 0.0))
        left = self.time_limit - t - self._time_reserve(corrective=False)
        n_long, shortest = 0, None
        for d in cands:
            if triage:  # a plan that did not fit before cannot fit now (time only runs out; plans start from home)
                rad = pol["same_object_radius"]
                if any(np.linalg.norm(d.top[:2] - p) < rad and dur > left for p, dur in self.too_long):
                    n_long += 1
                    continue
            others = [o.top[:2] for o in dets if o is not d]
            if verify:  # #6: explicit target bin + a free, verified drop slot
                target_bin = resolve_bin(self.mapping, d.color, getattr(d, "shape", None))
                if target_bin is None:
                    continue
                slot = self._next_drop_slot(target_bin)
                if slot is None:
                    self.log(f"[classical] no free drop slot in {target_bin}; leave {d.color} on counter")
                    continue
                p = self.planner.plan(d, self.arm_cmd, self.hand_cmd, bin_name=target_bin, obstacles=others,
                                      drop_slot=slot)
            else:
                p = self.planner.plan(d, self.arm_cmd, self.hand_cmd, obstacles=others)
            self.plans.append(p)
            if p.ok:
                if triage and p.duration > left:  # feasible but too slow for the time left: try the next candidate
                    self.too_long.append((d.top[:2], p.duration))
                    n_long += 1
                    shortest = p if shortest is None or p.duration < shortest.duration else shortest
                    self.log(f"[classical] {d.color} takes {p.duration:.1f} s > {left:.1f} s left: trying the next candidate")
                    continue
                return ("plan", p, len(dets))
            self.failed.append(d.top[:2])
            self.log(f"[classical] skip {d}: {p.reason}")
        if verify and self.verify_relooks_remaining > 0:
            self.verify_relooks_remaining -= 1
            return ("relook", None, len(dets))
        if n_long:
            return ("notime", shortest, len(dets))
        return ("none", None, len(dets))

    def _priority(self, d):
        """Triage tie-break tuple (lower = first): shape cost (cube < cylinder < triangle; `shape` attribute of the
        detection, unknown = 0), then -mapping confidence of the colour (default 1.0)."""
        cost = self.cfg["policy"].get("shape_cost", {})
        return (float(cost.get(getattr(d, "shape", None), 0.0)), -float(self.mapping_conf.get(d.color, 1.0)))

    def _score_bin(self, det):
        """Estimated scoring bin from a top-face detection, or None for a stack/rim/counter object."""
        centre = np.asarray(det.top, float).copy()
        centre[2] -= 0.5 * self.perception.height
        return self.scene.bin_of(self.scene.pelvis_to_mat(centre))

    def _nearest_slot(self, bin_name, xy):
        slots = self.planner.drop_slots(bin_name)
        return int(np.argmin([np.linalg.norm(np.asarray(xy) - p) for p in slots]))

    def _next_drop_slot(self, bin_name):
        return next(iter(self._free_drop_slots(bin_name)), None)

    def _free_drop_slots(self, bin_name):
        blocked = self.used_drop_slots[bin_name] | self.reserved_drop_slots[bin_name]
        return [i for i in range(len(self.planner.drop_slots(bin_name))) if i not in blocked]

    def _review_bins(self, bin_dets):
        """Update slot occupancy, verify the last release, and return corrective picks.

        A coloured component on a bin footprint but with its estimated centre
        above the rim is deliberately treated as unscored and scheduled for a
        re-pick, just like an object in the wrong bin.
        """
        scored = []
        corrections = []
        for d, footprint_bin in bin_dets:
            target_bin = resolve_bin(self.mapping, d.color)
            score_bin = self._score_bin(d)
            scored.append((d, footprint_bin, score_bin))
            if score_bin is not None and score_bin == target_bin:
                self.used_drop_slots[score_bin].add(self._nearest_slot(score_bin, d.top[:2]))
            elif target_bin is not None:
                corrections.append((d, footprint_bin, target_bin))

        pending = self.verification_pending
        self.last_verification = None
        if pending is not None:
            target = pending["target_bin"]
            slot = pending["drop_slot"]
            slot_xy = self.planner.drop_slots(target)[slot]
            radius = float(self.cfg["policy"].get("drop_verify_radius", 0.04))
            found = any(d.color == pending["color"] and score_bin == target
                        and np.linalg.norm(d.top[:2] - slot_xy) <= radius
                        for d, _, score_bin in scored)
            self.reserved_drop_slots[target].discard(slot)
            if found:
                self.used_drop_slots[target].add(slot)
                self.retry_colors.discard(pending["color"])
                self.verify_relooks_remaining = 0
                matching = [i for i, (color, _) in enumerate(self.retry_targets) if color == pending["color"]]
                if matching:
                    self.retry_targets.pop(matching[0])
                self.last_verification = ("ok", pending)
            else:
                # Keep the attempted slot unavailable: if the side-bin view was
                # occluded, a successful object may still be sitting there.
                self.reserved_drop_slots[target].add(slot)
                self.retry_colors.add(pending["color"])
                if pending.get("source_xy") is not None:
                    self.retry_targets.append((pending["color"], np.asarray(pending["source_xy"], float)))
                self.verify_relooks_remaining = int(self.cfg["policy"].get("verify_relooks", 1))
                self.last_verification = ("retry", pending)
            self.verification_pending = None
        return corrections

    def _after_look(self, res, t):
        if self.last_verification is not None:
            status, pending = self.last_verification
            if status == "ok":
                self._note(t, f"verified {pending['color']} in {pending['target_bin']} slot {pending['drop_slot']}")
            else:
                self._note(t, f"release of {pending['color']} not verified in {pending['target_bin']}; retry")
            self.last_verification = None
        kind = res[0]
        if kind == "error":
            self._note(t, f"look/plan error {res[1]}: stopping")
            self.state = "done"
            return self._hold(done=True)
        if kind == "relook":
            self._note(t, "placement unresolved; take one more settled verification frame")
            self.state = "settle"
            return self._hold()
        if kind == "uncorrectable":
            self._note(t, "known wrong-bin object has no safe corrective or cleanup plan; stopping")
            self.state = "done"
            return self._hold(done=True)
        if kind == "none":
            self._note(t, f"nothing left to sort ({res[2]} detections)")
            self.state = "done"
            return self._hold(done=True)
        if kind == "notime":
            self._note(t, f"no candidate fits the time left (shortest {res[1].duration:.0f} s)"
                          if res[1] is not None else "no candidate fits the time left")
            self.state = "done"
            return self._hold(done=True)
        plan = res[1]
        if not self._has_time(plan, t):
            self._note(t, f"not enough time for {plan.color} ({plan.duration:.0f} s)")
            self.state = "done"
            return self._hold(done=True)
        if self.require_confirm:
            self.pending, self.waiting, self.state = plan, True, "await"
            self.pending_text = plan.summary()
            self._note(t, f"waiting for confirm: {plan.summary()}")
            return self._hold()
        return self._start(plan, t)

    def _verify_on(self):
        return bool(self.cfg["policy"].get("verify_placement", False))

    def _time_reserve(self, corrective=False):
        """Seconds a plan must leave before time_limit: #3 time_margin_s (default 0), plus with #6 on the settle
        time and, for a new (non-corrective) placement, correction_reserve_s. Flags off: 0 = stock check."""
        pol = self.cfg["policy"]
        reserve = float(pol.get("time_margin_s", 0.0))
        if self._verify_on():
            reserve += float(self.cfg["motion"]["settle_s"])
            if not corrective:
                reserve += float(pol.get("correction_reserve_s", 0.0))
        return reserve

    def _has_time(self, plan, t):
        return t + plan.duration + self._time_reserve(plan.corrective) <= self.time_limit

    def _start(self, plan, t):
        xy = plan.waypoints["grasp"][:2]
        self.attempts.append((xy, 1))
        self.n_started += 1
        self._enqueue(plan.arm, plan.hand)
        if plan.target_bin in self.scene.bin_names and plan.drop_slot is not None:
            self.reserved_drop_slots[plan.target_bin].add(plan.drop_slot)
            self.verification_pending = {"color": plan.color, "target_bin": plan.target_bin,
                                         "drop_slot": plan.drop_slot, "corrective": plan.corrective,
                                         "source_xy": plan.waypoints["grasp"][:2].copy()}
        self.state = "executing"
        self._note(t, f"execute {plan.summary()}")
        return self._pop()

    # ------------------------------------------------------------------ action plumbing
    def _enqueue(self, a, h):
        self.queue_a, self.queue_h = np.asarray(a, float), np.asarray(h, float)

    def _pop(self):
        k = int(self.cfg["motion"]["chunk"])
        a, h = self.queue_a[:k], self.queue_h[:k]
        self.queue_a, self.queue_h = self.queue_a[k:], self.queue_h[k:]
        if not len(a):
            return self._hold()
        self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
        return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}

    def _hold(self, done=False):
        """Hold the last target. While waiting (planning / operator) the hold is `motion.hold_ticks` ticks long, so a
        remote runner needs one request per hold chunk instead of one per tick: the command rate stays at 30 Hz
        even when the planner thread and other processes (sims) load the CPU."""
        k = 1 if done else max(1, int(self.cfg["motion"].get("hold_ticks", 1)))
        if done and not self._summary_logged and self.cfg["policy"].get("episode_summary_log", False):
            # passive end-of-episode summary (log only, once)
            self._summary_logged = True
            self.log(f"[classical] episode end: attempts={len(self.attempts)} failed={len(self.failed)} "
                     f"too_long={len(getattr(self, 'too_long', []))} started={self.n_started}")
        return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0), "hand_q_target": np.repeat(self.hand_cmd[None], k, 0),
                "done": bool(done)}

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[classical t={t:6.2f}] {msg}")
