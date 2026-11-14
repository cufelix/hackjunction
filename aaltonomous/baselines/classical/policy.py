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
import re
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
from .planner import Plan, Planner, lookup_bin

HZ = 30.0
_HOLE = re.compile(r"\bholes?\b")   # policy.insert_guard: the rung-3 sentence names a hole AND the lid
_LID = re.compile(r"\blids?\b")


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
        self.perception = Perception(self.cfg, self.scene, self.cal, log=log or (lambda *a: None))
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
            if (self.cfg.get("instruction") or {}).get("normalize", False):   # typo/synonym normaliser (Felix)
                from .normalize import normalize
                text = normalize(text)
            return resolve(text) or None
        except Exception as e:                      # a parse bug must not cost the whole episode
            self.log(f"instruction resolve failed ({e}), using FIXED_MAPPING")
            return None

    def _insert_guard_mapping(self, info):
        """policy.insert_guard: rung 3 (insert_shapes) -> {"default": "abstain"}, i.e. pick nothing; else None.

        The lid sits on the left bin, so FIXED_MAPPING (what the resolver falls back to for the hole sentence) releases
        every red object onto the lid: -1 each (forced). The holes are past the arm's reach, so 0 is the best safe
        score. Detected from the task id (policy_server passes it) or the sentence (needs both 'hole' and 'lid').
        NEVER raises (reset() runs under the 1 s watchdog)."""
        try:
            if not (self.cfg.get("policy") or {}).get("insert_guard", False):
                return None
            task = (info or {}).get("task")
            text = str((info or {}).get("instruction") or "").lower()
            if task == "insert_shapes" or (_HOLE.search(text) and _LID.search(text)):
                self.log("[insert_guard] insert_shapes: abstain on every object (no release over the lid)")
                return {"default": "abstain"}
            return None
        except Exception as e:
            self.log(f"insert_guard failed ({e})")
            return None

    def reset(self, info=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = info or {}
        self.mapping = (info.get("mapping") or self._insert_guard_mapping(info) or self._instruction_mapping(info)
                        or dict(FIXED_MAPPING))
        self.planner.mapping = self.mapping
        self.time_limit = info.get("time_limit", self.time_limit)
        self.state = "init"
        self.queue_a = self.queue_h = None
        self.arm_cmd = self.hand_cmd = None
        self.attempts = []          # (position, count) of objects we tried to pick
        self.attempt_tracks = []    # retry-vary: {color, latest xy, cumulative count} stable pushed-object tracks
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
        self.retry_allow = []       # (position, extra attempts) granted by the retry pass; replaces max_attempts there
        self.n_corrections = 0      # #6 corrective plans started (policy.max_corrections_per_episode)
        self.slot_used = {name: set() for name in self.scene.bin_names}   # grasp.drop_slots standalone
        self.slot_pending = None    # (bin, slot, grasp xy, colour) of the last standalone-slot placement
        self._gdiag = None          # policy.grasp_diag_log: the pick being watched
        self._bin_misses = []       # (colour, xy) already logged as [bin_lookup] miss this episode
        self._stall = None          # motion.lift_on_stall: the close hold being watched
        self.retry_pass_used = False
        self.vary_hist = []         # policy.retry_vary_grasp: {color, xy, yaw, posture, side, held} per started pick
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
        if self._gdiag is not None:
            self._grasp_diag(obs)
        if self._stall is not None:
            self._lift_on_stall(obs)
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
            return self._choose_retry(dets, obs, verify=False)
        dets, rejected = self.perception.detect(obs["head_rgb"], obs["head_depth"], debug=True, include_bins=True)
        bin_dets = [(d, why.split(":", 1)[1]) for d, why in rejected if why.startswith("in_bin:")]
        self.last_detections = dets
        self.last_bin_detections = bin_dets
        corrections = self._review_bins(bin_dets)

        # A known wrong-bin object is the largest score swing, so repair it before
        # touching a new counter object.  The fresh detection is planned directly;
        # stale corrective targets are never retained between camera frames.
        self.scheduled_corrections = corrections
        cap = self.cfg["policy"].get("max_corrections_per_episode")
        if cap is None:  # stock #6: corrections first; none plannable -> stop
            if corrections:
                res = self._plan_corrections(corrections, dets)
                return res if res is not None else ("uncorrectable", corrections, len(dets))
        # max_corrections_per_episode set (Felix review): a correction never blocks a new pick while a pickable
        # object is on the counter -> counter picks first, corrections (within the cap) only when nothing else is
        # left; an unplannable correction no longer ends the episode.

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
        res = self._choose_retry(dets, obs, verify=True)
        if cap is not None and corrections and res[0] in ("none", "notime") and self.n_corrections < int(cap):
            alt = self._plan_corrections(corrections, dets)
            if alt is not None:
                return alt
        return res

    def _plan_corrections(self, corrections, dets):
        """First plannable corrective pick: wrong/unscored bin -> a target-bin slot, else -> counter. None if none."""
        for d, source_bin, target_bin in corrections:
            n = self._attempt_count(d.top[:2], d.color)
            limit = self._attempt_limit(d.top[:2])
            varied = self._max_attempts_vary()
            if n >= (max(limit, varied) if varied is not None else limit):
                self.log(f"[classical] correction attempt limit reached for {d.color} in {source_bin}")
                continue
            for slot in self._free_drop_slots(target_bin) or self._overflow_slots(target_bin):
                p = self._plan_pick(d, dict(bin_name=target_bin, relocate=False,
                                            drop_slot=slot, source_bin=source_bin))
                self.plans.append(p)
                if p.ok:
                    return ("plan", p, len(dets))
                self.log(f"[classical] cannot correct {d.color} from {source_bin} to slot {slot}: {p.reason}")
            def cleanup(det, arm, hand, **_):
                return self.planner.plan_to_counter(det, arm, hand, source_bin,
                                                    obstacles=[o.top[:2] for o in dets])
            p = self._plan_pick(d, {}, plan_fn=cleanup)
            self.plans.append(p)
            if p.ok:
                return ("plan", p, len(dets))
            self.log(f"[classical] cannot remove {d.color} from wrong {source_bin} bin: {p.reason}")
        return None

    def _overflow_slots(self, bin_name):
        """grasp.drop_slot_overflow: counter (stock #6) = no slot; stack = every slot again (drop onto the pile)."""
        if self.cfg["grasp"].get("drop_slot_overflow", "counter") != "stack":
            return []
        return list(range(len(self.planner.drop_slots(bin_name))))

    def _choose_retry(self, dets, obs, verify):
        """_choose + #3 retry pass (Oliver 40fb22a, policy.retry_pass, default off)."""
        if not verify and self._slots_on():
            self._settle_slot(dets)
        res = self._choose(dets, obs, verify)
        pol = self.cfg["policy"]
        if res[0] == "none" and dets and pol.get("retry_pass", False) and not self.retry_pass_used:
            # Retry pass (once per episode): objects excluded only by max_attempts get retry_extra_attempts more
            # tries. `failed` (IK failure / operator skip) stays excluded. The time check is the normal one on the
            # real plan duration: triage skips misfits, without triage _after_look refuses a plan that does not fit.
            rad = self._attempt_radius()
            excluded = [(d.color, d.top[:2]) for d in dets if self._attempt_excluded(d.top[:2], d.color)]
            excl = [xy for _, xy in excluded]
            if excl:
                self.retry_pass_used = True
                extra = int(pol.get("retry_extra_attempts", 2))
                self.attempts = [(p, c) for p, c in self.attempts
                                 if not any(np.linalg.norm(xy - p) < rad for xy in excl)]
                self.attempt_tracks = [tr for tr in self.attempt_tracks
                                       if not any(tr["color"] == color and np.linalg.norm(xy - tr["xy"]) < rad
                                                  for color, xy in excluded)]
                self.retry_allow += [(xy, extra) for xy in excl]
                self._note(float(obs.get("t", 0.0)), f"retry pass: {len(excl)} object(s) get {extra} more attempts")
                res = self._choose(dets, obs, verify)
        return res

    def _bin_of(self, d):
        """lookup_bin for a counter candidate; a miss (None / abstain) is logged once per object per episode."""
        shape = getattr(d, "shape", None)
        b = lookup_bin(self.cfg, self.mapping, d.color, shape)
        if b is None:
            rad = self.cfg["policy"]["same_object_radius"]
            xy = np.asarray(d.top[:2], float)
            if not any(c == d.color and np.linalg.norm(xy - p) < rad for c, p in self._bin_misses):
                self._bin_misses.append((d.color, xy))
                self.log(f"[bin_lookup] miss color={d.color} shape={shape}")
        return b

    def _slots_on(self):
        """grasp.drop_slots (standalone, default off). With verify_placement on, #6 manages the slots itself."""
        return bool(self.cfg["grasp"].get("drop_slots", False)) and not self._verify_on()

    def _settle_slot(self, dets):
        """Standalone slots: the slot of the last placement is freed again when the object is still on the counter
        where we tried to grasp it (the hand closed on nothing), so failed picks do not use up the slots."""
        if self.slot_pending is None:
            return
        bin_name, slot, xy, color = self.slot_pending
        self.slot_pending = None
        rad = 2 * self.cfg["policy"]["same_object_radius"]
        if any(d.color == color and np.linalg.norm(d.top[:2] - xy) < rad for d in dets):
            self.slot_used[bin_name].discard(slot)

    def _attempt_excluded(self, xy, color=None):
        """True when the object at xy is skipped for its attempt count only (not in `failed`)."""
        rad = self._attempt_radius()
        if any(np.linalg.norm(xy - f) < rad for f in self.failed):
            return False
        return self._attempt_count(xy, color) >= self._attempt_limit(xy)

    def _attempt_radius(self):
        """Association radius for executed attempts. Retry-vary uses its wider pushed-object track."""
        pol = self.cfg["policy"]
        return float(pol.get("retry_assoc_radius", 0.06) if self._vary_on() else pol["same_object_radius"])

    def _attempt_count(self, xy, color=None):
        rad = self._attempt_radius()
        if self._vary_on():
            tracks = [tr for tr in self.attempt_tracks if color is None or tr["color"] == color]
            near = [tr for tr in tracks if np.linalg.norm(xy - tr["xy"]) < rad]
            return 0 if not near else min(near, key=lambda tr: np.linalg.norm(xy - tr["xy"]))["count"]
        return sum(c for p, c in self.attempts if np.linalg.norm(xy - p) < rad)

    def _track_attempt(self, color, xy):
        """Advance the nearest same-colour retry track, preserving count through successive <radius pushes."""
        if not self._vary_on():
            return
        xy = np.asarray(xy, float).copy()
        same = [tr for tr in self.attempt_tracks if tr["color"] == color]
        near = [tr for tr in same if np.linalg.norm(xy - tr["xy"]) < self._attempt_radius()]
        if near:
            tr = min(near, key=lambda item: np.linalg.norm(xy - item["xy"]))
            tr["xy"], tr["count"] = xy, tr["count"] + 1
        else:
            self.attempt_tracks.append({"color": color, "xy": xy, "count": 1})

    def _attempt_limit(self, xy):
        """max_attempts, or the retry pass's extra attempts for an object it re-admitted."""
        rad = self._attempt_radius()
        for p, extra in self.retry_allow:
            if np.linalg.norm(xy - p) < rad:
                return extra
        return self.cfg["policy"]["max_attempts"]

    def _choose(self, dets, obs, verify):
        """Pick the next counter object and plan it (stock order; #3 time triage and #6 drop slots when on)."""
        cands, extra = [], []
        mav = self._max_attempts_vary()
        for d in dets:
            xy = d.top[:2]
            if any(np.linalg.norm(xy - f) < self.cfg["policy"]["same_object_radius"] for f in self.failed):
                continue
            n = self._attempt_count(xy, d.color)
            if n >= self._attempt_limit(xy):
                if mav is not None and n < mav:
                    extra.append(d)   # policy.max_attempts_vary: one more varied try, after every other candidate
                continue
            cands.append(d)
        if self.cfg["policy"]["order"] == "nearest_bin":
            def base(d):
                b = lookup_bin(self.cfg, self.mapping, d.color, getattr(d, "shape", None))
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
        if extra:
            extra.sort(key=base)
            cands += extra
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
            target_bin = self._bin_of(d)
            if target_bin is None:
                continue  # unmapped / abstain: 0 points on the counter, never a -1 guess
            if verify:  # #6: explicit target bin + a free, verified drop slot
                slot = self._next_drop_slot(target_bin)
                if slot is None:
                    slot = next(iter(self._overflow_slots(target_bin)), None)
                if slot is None:
                    self.log(f"[classical] no free drop slot in {target_bin}; leave {d.color} on counter")
                    continue
                p = self._plan_pick(d, dict(bin_name=target_bin, obstacles=others, drop_slot=slot))
            elif self._slots_on():  # grasp.drop_slots standalone: each successive object in a bin -> next slot
                free = [i for i in range(len(self.planner.drop_slots(target_bin)))
                        if i not in self.slot_used[target_bin]]
                slot = free[0] if free else next(iter(self._overflow_slots(target_bin)), None)
                if slot is None:
                    self.log(f"[classical] drop slots of {target_bin} used up; leave {d.color} on counter")
                    continue
                p = self._plan_pick(d, dict(bin_name=target_bin, obstacles=others, drop_slot=slot))
            else:
                p = self._plan_pick(d, dict(obstacles=others))
            self.plans.append(p)
            if p.ok and any(d is e for e in extra) and not self._has_time(p, t):   # max_attempts_vary: only with time to spare
                self.log(f"[retry_vary] no time for an extra try at {d.color} ({p.duration:.1f} s)")
                continue
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

    # ------------------------------------------------------------------ policy.retry_vary_grasp
    def _vary_on(self):
        return bool(self.cfg["policy"].get("retry_vary_grasp", False))

    def _max_attempts_vary(self):
        """policy.max_attempts_vary (needs retry_vary_grasp): attempt limit for extra, varied tries; None = off."""
        mav = self.cfg["policy"].get("max_attempts_vary")
        return int(mav) if (mav is not None and self._vary_on()) else None

    def _posture_of(self, plan):
        return plan.posture[0] if plan.posture else self.cfg["grasp"].get("close_posture", "closed")

    def _hist_for(self, d):
        """Earlier picks of this object that did not hold it (held no / unknown): same colour, within
        retry_assoc_radius of the earlier grasp point (a missed object is often pushed a few cm)."""
        r = float(self.cfg["policy"].get("retry_assoc_radius", 0.06))
        xy = np.asarray(d.top[:2], float)
        return [h for h in self.vary_hist if h["held"] is not True and h["color"] == d.color
                and np.linalg.norm(xy - h["xy"]) < r]

    def _same_grasp(self, key, h):
        pol = self.cfg["policy"]
        xy, yaw, posture = key
        if posture != h["posture"] or np.linalg.norm(np.asarray(xy) - h["xy"]) >= float(pol.get("retry_same_xy_m", 0.01)):
            return False
        if yaw is None or h["yaw"] is None:
            return True
        dy = abs((yaw - h["yaw"] + np.pi) % (2 * np.pi) - np.pi)
        return dy < np.radians(float(pol.get("retry_same_yaw_deg", 20.0)))

    def _grasp_key(self, plan):
        fh = getattr(self.planner, "finger_heading", None)
        yaw = fh(plan) if fh is not None else None
        return (np.asarray(plan.waypoints["grasp"][:2], float).copy(), yaw, self._posture_of(plan))

    def _variants(self, d, hist):
        """Grasp changes to try, best first: {"posture": name | None, "yaw": rad | None} (None = the stock choice).
        Posture v1 <-> demo for a triangle / label None (demo = triangles 18/25 vs pinch_v1 3/9, but cubes 7/25 with
        demo: a demo grasp on a non-triangle goes back to the configured posture first); the finger heading
        +-retry_yaw_deg[label] from the last try (the miss direction relative to the pinch changes)."""
        pol, g = self.cfg["policy"], self.cfg["grasp"]
        prev = hist[-1]
        label = getattr(d, "shape", None)
        demo, v1 = "closed_demo", g.get("close_posture", "closed")
        p0 = prev["posture"]
        alt = v1 if p0 == demo else demo
        degs = pol.get("retry_yaw_deg") or {}
        dy = np.radians(float(degs.get(label if label is not None else "none", degs.get("default", 90.0))))
        yaws = []
        if prev["yaw"] is not None:
            ref = np.radians(-25.0 if prev["side"] == "left" else 25.0)   # natural finger heading of that arm
            ys = [prev["yaw"] + dy, prev["yaw"] - dy]
            ys.sort(key=lambda a: abs((a - ref + np.pi) % (2 * np.pi) - np.pi))
            yaws = [(a + np.pi) % (2 * np.pi) - np.pi for a in ys]
        if label == "triangle":
            out = ([{"posture": demo, "yaw": a} for a in yaws] + [{"posture": v1, "yaw": None}] if p0 == demo else
                   [{"posture": demo, "yaw": None}] + [{"posture": demo, "yaw": a} for a in yaws])
        elif label is None:
            out = ([{"posture": alt, "yaw": None}] + [{"posture": p0, "yaw": a} for a in yaws]
                   + [{"posture": alt, "yaw": a} for a in yaws])
        else:   # cube / cylinder: never the demo posture (cubes 7/25 with it)
            out = ([{"posture": v1, "yaw": None}] if p0 == demo else []) + [{"posture": v1, "yaw": a} for a in yaws]
        return out[: int(pol.get("retry_max_variants", 3))]

    def _plan_pick(self, d, kw, plan_fn=None):
        """planner.plan, or with policy.retry_vary_grasp and an earlier miss on this object: the first feasible grasp
        variant whose (point, finger heading, posture) differs from every earlier try; the stock plan only if it
        differs too. Flag off: exactly planner.plan(d, arm, hand, **kw)."""
        def plan(relocate=None):
            args = dict(kw)
            if relocate is not None and plan_fn is None:
                args["relocate"] = relocate
            fn = self.planner.plan if plan_fn is None else plan_fn
            return fn(d, self.arm_cmd, self.hand_cmd, **args)

        if not self._vary_on():
            return plan()
        hist = self._hist_for(d)
        if not hist:
            p = plan()
            if p.ok:
                p.grasp_key = self._grasp_key(p)
            return p
        for v in self._variants(d, hist):
            self.planner.vary = v
            try:
                p = plan(relocate=False)
            finally:
                self.planner.vary = None
            if not p.ok:
                self.log(f"[retry_vary] {d.color} variant {self._vary_txt(v)} (before {self._key_txt((hist[-1]['xy'], hist[-1]['yaw'], hist[-1]['posture']))}) infeasible: {p.reason}")
                continue
            key = self._grasp_key(p)
            if any(self._same_grasp(key, h) for h in hist):
                self.log(f"[retry_vary] {d.color} variant {self._vary_txt(v)} repeats a tried grasp")
                continue
            p.grasp_key = key
            self.log(f"[retry_vary] {d.color} label={getattr(d, 'shape', None)} try {len(hist) + 1}: "
                     f"{self._vary_txt(v)} -> {self._key_txt(key)} (before: "
                     + ", ".join(self._key_txt((h["xy"], h["yaw"], h["posture"])) for h in hist) + ")")
            return p
        p = plan()
        if p.ok:
            p.grasp_key = self._grasp_key(p)
            if any(self._same_grasp(p.grasp_key, h) for h in hist):
                return Plan(False, color=d.color, reason="retry_vary: every grasp repeats a tried (point, yaw, posture)")
        return p

    @staticmethod
    def _vary_txt(v):
        yaw = "stock" if v["yaw"] is None else f"{np.degrees(v['yaw']):.0f}"
        return f"posture={v['posture']} yaw={yaw}"

    @staticmethod
    def _key_txt(k):
        xy, yaw, posture = k
        y = "n/a" if yaw is None else f"{np.degrees(yaw):.0f}"
        return f"({xy[0]:+.3f},{xy[1]:+.3f}) yaw={y} {posture}"

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
        yellow_unknown = bool(self.cfg["policy"].get("verify_yellow_unknown", False))
        for d, footprint_bin in bin_dets:
            target_bin = lookup_bin(self.cfg, self.mapping, d.color, getattr(d, "shape", None))
            score_bin = self._score_bin(d)
            scored.append((d, footprint_bin, score_bin))
            if score_bin is not None and score_bin == target_bin:
                self.used_drop_slots[score_bin].add(self._nearest_slot(score_bin, d.top[:2]))
            elif yellow_unknown and d.color == "yellow":
                continue  # Felix review: a yellow in-bin blob is "unknown", never a mismatch -> no correction
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
            elif yellow_unknown and pending["color"] == "yellow":
                self.reserved_drop_slots[target].add(slot)   # unknown: keep the slot blocked, no retry / relook
                self.last_verification = ("unknown", pending)
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
            elif status == "unknown":
                self._note(t, f"release of {pending['color']} in {pending['target_bin']} unknown (not retried)")
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
        self._track_attempt(plan.color, xy)
        self.n_started += 1
        if plan.corrective:
            self.n_corrections += 1
        if plan.posture is not None and self.cfg["policy"].get("grasp_diag_log", False):
            nm, label, conf, path = plan.posture[:4]
            gate = ""
            if len(plan.posture) > 4:   # grasp.posture_apex_gate decision (only logged when the flag is on)
                ok, ac, ang = plan.posture[4]
                gate = (f" apex_gate={'pass' if ok else 'fallback'} apex_conf={ac:.2f} "
                        f"max_angle={'None' if ang is None else f'{ang:.1f}'}")
            self.log(f"[grasp_posture] t={t:.1f} {plan.color} {plan.side} label={label} conf={conf:.2f} path={path} "
                     f"posture={'pinch_demo' if nm == 'closed_demo' else 'pinch_v1'} ({nm})" + gate)
        ls = self.cfg["motion"].get("lift_on_stall") or {}
        self._stall = None
        if ls.get("enabled", False) and plan.close_hold is not None and plan.side in ("left", "right"):
            self._stall = {"n": len(plan.arm), "hold": plan.close_hold, "sl": ArmKinematics.sl(plan.side),
                           "min": int(round(float(self.cfg["motion"]["close_hold_s"]) * HZ)),
                           "rad": float(ls.get("stall_rad", 0.01)), "need": int(ls.get("settle_ticks", 3)),
                           "last": None, "still": 0, "t": t}
        if self.cfg["policy"].get("grasp_diag_log", False) and plan.side in ("left", "right"):
            ph = {name: (s_, e) for name, s_, e in (plan.phases or [])}
            if "close" in ph and "lift" in ph:
                self._gdiag = {"plan": plan, "n": len(plan.arm), "close_end": ph["close"][1],
                               "lift_end": ph["lift"][1], "t": t, "samples": {}}
        if self._vary_on() and plan.side in ("left", "right"):
            key = plan.grasp_key or (np.asarray(xy, float).copy(), None, self._posture_of(plan))
            h = {"color": plan.color, "xy": np.asarray(key[0], float), "yaw": key[1], "posture": key[2],
                 "side": plan.side, "held": None}
            self.vary_hist.append(h)
        self._enqueue(plan.arm, plan.hand)
        if self._slots_on() and plan.target_bin in self.scene.bin_names and plan.drop_slot is not None:
            self.slot_used[plan.target_bin].add(plan.drop_slot)
            self.slot_pending = (plan.target_bin, plan.drop_slot, plan.waypoints["grasp"][:2].copy(), plan.color)
        elif plan.target_bin in self.scene.bin_names and plan.drop_slot is not None:
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

    def _grasp_diag(self, obs):
        """policy.grasp_diag_log (passive, never changes a command): once per pick, measured hand_q vs commanded
        hand_cmd vs the `closed` posture at the end of the close hold and at the end of the lift. held = the active
        hand still lags its command by >= 0.010 rad (fingers stalled on an object); no = they reached the target
        (closed on nothing)."""
        g = self._gdiag
        q = obs.get("hand_q")
        left = 0 if self.queue_a is None else len(self.queue_a)
        done = g["n"] - left
        if q is not None:
            for key, at in (("close", g["close_end"]), ("lift", g["lift_end"])):
                if key not in g["samples"] and done >= at:
                    g["samples"][key] = (done, np.asarray(q, float).copy(), np.asarray(self.hand_cmd, float).copy())
        if "lift" not in g["samples"] and left > 0:
            return
        self._gdiag = None
        plan = g["plan"]
        sl = ArmKinematics.sl(plan.side)
        closed = self.post[(plan.posture[0] if plan.posture else self.cfg["grasp"].get("close_posture", "closed"))][sl]
        parts = []
        held = None
        for key in ("close", "lift"):
            if key not in g["samples"]:
                parts.append(f"{key}=n/a")
                continue
            tick, hq, hc = g["samples"][key]
            err = np.abs(hc[sl] - hq[sl])
            held = bool(err.max() >= 0.010)
            parts.append(f"{key}@{tick}: track_err_max={err.max():.3f} stall_vs_closed="
                         f"{np.round(closed - hq[sl], 3).tolist()} cmd_vs_closed={np.round(closed - hc[sl], 3).tolist()}")
        self.log(f"[grasp_diag] t={g['t']:.1f} {plan.color} {plan.side} target={plan.target_bin} "
                 f"held={'n/a' if held is None else ('yes' if held else 'no')} " + " ".join(parts))

    def _in_hold(self):
        g = self._stall
        done = g["n"] - (0 if self.queue_a is None else len(self.queue_a))
        return g["hold"][0] <= done < g["hold"][1]

    def _lift_on_stall(self, obs):
        """motion.lift_on_stall: inside the close hold, skip the rest of the hold (-> lift) once the active hand's
        hand_q moved < stall_rad for settle_ticks consecutive ticks and at least close_hold_s has passed; else the hold
        runs to timeout_s (planned length). Only drops hold ticks (same arm + hand target), never adds motion."""
        g = self._stall
        left = 0 if self.queue_a is None else len(self.queue_a)
        done = g["n"] - left
        hs, he = g["hold"]
        if done >= he or left == 0:
            self._stall = None
            return
        q = obs.get("hand_q")
        if done < hs or q is None:
            return
        q = np.asarray(q, float)[g["sl"]]
        if g["last"] is not None:
            g["still"] = g["still"] + 1 if np.abs(q - g["last"]).max() < g["rad"] else 0
        g["last"] = q
        if done - hs >= g["min"] and g["still"] >= g["need"]:
            cut = he - done
            self.queue_a, self.queue_h = self.queue_a[cut:], self.queue_h[cut:]
            self.log(f"[lift_on_stall] t={g['t']:.1f} lift after {(done - hs) / HZ:.2f} s of hold (stall), "
                     f"skipped {cut / HZ:.2f} s")
            self._stall = None

    def _pop(self):
        k = int(self.cfg["motion"]["chunk"])
        if self._stall is not None and self._in_hold():
            k = 1  # lift_on_stall: one tick per request inside the close hold, so every tick's hand_q is seen
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
                     f"too_long={len(getattr(self, 'too_long', []))} started={self.n_started} "
                     f"retry_pass_used={int(self.retry_pass_used)}")
        return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0), "hand_q_target": np.repeat(self.hand_cmd[None], k, 0),
                "done": bool(done)}

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[classical t={t:6.2f}] {msg}")
