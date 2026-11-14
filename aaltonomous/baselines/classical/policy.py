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
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .perception import Perception
from .planner import Planner, lookup_bin, lookup_detection_bin
from .brain import BrainLayer, PlanLayer, build_world, instruction_needs_plan
from .brain.find import FindClient, extract_queries, match_found

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
        brain_cfg = self.cfg.get("brain") or {}
        # Whole-plan response validation includes IK dry runs. Keep that work,
        # along with ordinary classical planning, off the 30 Hz act thread.
        self.threaded = bool(threaded or (brain_cfg.get("enabled") and brain_cfg.get("planner")))
        if self.threaded:  # pure-Python planning holds the GIL: hand it back to act() sooner (default 5 ms)
            sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
        self.time_limit = time_limit
        self.log = log or (lambda *a: None)
        bc = self.cfg.get("brain") or {}
        self.brain = BrainLayer(bc, log=self.log) if bc.get("enabled", False) else None   # FLAG brain.enabled
        self.plan_brain = (PlanLayer(bc, backend=self.brain.backend, log=self.log)
                           if self.brain is not None and bc.get("planner", False) else None)  # FLAG brain.planner
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
        given = info.get("mapping")
        guarded = None if given else self._insert_guard_mapping(info)
        resolved = None if given or guarded else self._instruction_mapping(info)
        self.mapping = given or guarded or resolved or dict(FIXED_MAPPING)
        self.planner.mapping = self.mapping
        self._task, self._instruction = info.get("task"), info.get("instruction")
        self._brain_looks = 0
        self._brain_finds = 0
        self._brain_find_fn = getattr(self, "_brain_find_fn", None)
        self._brain_queries = (extract_queries(self._instruction)
                               if self.brain is not None and (self.cfg.get("brain") or {}).get("find", False) else [])
        self._brain_mapping, self._brain_mapping_source = self._resolver_mapping(info, guarded)
        self._brain_abstain = self._resolver_abstains(self._brain_mapping)
        self._brain_reg, self._brain_n, self._last_pick = [], 0, None
        if self.brain is not None:
            self.brain.reset()
        if self.plan_brain is not None:
            self.plan_brain.reset()
        self._planner_needed = (self.plan_brain is not None and
                                instruction_needs_plan(self._instruction, self._brain_abstain))
        self._planner_place_cache = {}
        self._planner_pending = None
        self._planner_execution = None
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
        self.retry_allow = []       # (position, extra attempts) granted by the retry pass; replaces max_attempts there
        self.n_corrections = 0      # #6 corrective plans started (policy.max_corrections_per_episode)
        self.slot_used = {name: set() for name in self.scene.bin_names}   # grasp.drop_slots standalone
        self.slot_pending = None    # (bin, slot, grasp xy, colour) of the last standalone-slot placement
        self._gdiag = None          # policy.grasp_diag_log: the pick being watched
        self._bin_misses = []       # (colour, xy) already logged as [bin_lookup] miss this episode
        self._stall = None          # motion.lift_on_stall: the close hold being watched
        self.retry_pass_used = False
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
        if self.state == "planner_wait":
            # Poll the asynchronous initial plan/re-plan while returning normal HOLD chunks at 30 Hz.
            dets, verify = self._planner_pending
            res = self._choose(dets, obs, verify)
            if res[0] == "planner_pending":
                return self._hold()
            self._planner_pending = None
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
            self._planner_observe(dets, obs)
            return self._choose_retry(dets, obs, verify=False)
        dets, rejected = self.perception.detect(obs["head_rgb"], obs["head_depth"], debug=True, include_bins=True)
        bin_dets = [(d, why.split(":", 1)[1]) for d, why in rejected if why.startswith("in_bin:")]
        self.last_detections = dets
        self.last_bin_detections = bin_dets
        self._planner_observe(dets, obs)
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
            for slot in self._free_drop_slots(target_bin) or self._overflow_slots(target_bin):
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
            rad = pol["same_object_radius"]
            excl = [d.top[:2] for d in dets if self._attempt_excluded(d.top[:2])]
            if excl:
                self.retry_pass_used = True
                extra = int(pol.get("retry_extra_attempts", 2))
                self.attempts = [(p, c) for p, c in self.attempts
                                 if not any(np.linalg.norm(xy - p) < rad for xy in excl)]
                self.retry_allow += [(xy, extra) for xy in excl]
                self._note(float(obs.get("t", 0.0)), f"retry pass: {len(excl)} object(s) get {extra} more attempts")
                res = self._choose(dets, obs, verify)
        return res

    def _bin_of(self, d):
        """lookup_bin for a counter candidate; a miss (None / abstain) is logged once per object per episode."""
        shape = getattr(d, "shape", None)
        b = lookup_detection_bin(self.cfg, self.mapping, d)
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

    def _attempt_excluded(self, xy):
        """True when the object at xy is skipped for its attempt count only (not in `failed`)."""
        rad = self.cfg["policy"]["same_object_radius"]
        if any(np.linalg.norm(xy - f) < rad for f in self.failed):
            return False
        n = sum(c for p, c in self.attempts if np.linalg.norm(xy - p) < rad)
        return n >= self._attempt_limit(xy)

    def _attempt_limit(self, xy):
        """max_attempts, or the retry pass's extra attempts for an object it re-admitted."""
        rad = self.cfg["policy"]["same_object_radius"]
        for p, extra in self.retry_allow:
            if np.linalg.norm(xy - p) < rad:
                return extra
        return self.cfg["policy"]["max_attempts"]

    def _plan_variant(self, variant, *a, **kw):
        """planner.plan, with the brain's retry grasp variant applied to the config for this one call only.
        same / None = unchanged; closed_tight = that posture; demo = the by-shape demo posture (triangles)."""
        g = self.cfg["grasp"]
        saved = {k: g.get(k) for k in ("close_posture", "close_posture_by_shape")}
        try:
            if variant == "closed_tight":
                g["close_posture"] = "closed_tight"
            elif variant == "demo":
                g["close_posture_by_shape"] = True
            return self.planner.plan(*a, **kw)
        finally:
            for k, v in saved.items():
                g[k] = v

    def _ordered_cands(self, dets, verify):
        """Counter candidates in the order _choose tries them (attempt / failed filter + sort). Pure: also used to
        predict the next decision state for the brain prefetch."""
        cands = []
        for d in dets:
            xy = d.top[:2]
            if any(np.linalg.norm(xy - f) < self.cfg["policy"]["same_object_radius"] for f in self.failed):
                continue
            n = sum(c for p, c in self.attempts if np.linalg.norm(xy - p) < self.cfg["policy"]["same_object_radius"])
            if n >= self._attempt_limit(xy):
                continue
            cands.append(d)
        if self.cfg["policy"]["order"] == "nearest_bin":
            def base(d):
                b = lookup_detection_bin(self.cfg, self.mapping, d)
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
        return cands

    def _choose(self, dets, obs, verify):
        """Pick the next counter object and plan it (stock order; #3 time triage and #6 drop slots when on)."""
        cands = self._ordered_cands(dets, verify)
        pol = self.cfg["policy"]
        triage = bool(pol.get("time_triage", False))
        t = float(obs.get("t", 0.0))
        left = self.time_limit - t - self._time_reserve(corrective=False)
        n_long, shortest = 0, None
        forced = None   # (det, bin, variant, table_xy) from the brain/planner
        planner_forced = False
        if self._planner_needed:
            sel = self._planner_select(cands, dets, obs)
            if (sel is None and self.plan_brain.steps is None and
                    self.plan_brain.req.status == "pending"):
                return ("planner_pending", dets, verify)
            if (sel is None and self.plan_brain.replans and self.plan_brain.steps is None):
                return ("none", None, len(dets))       # failed replan: fail closed, do not improvise
            if sel is not None:
                kind, d0, b0, var, table_xy = sel
                if kind == "look":
                    return ("relook", None, len(dets))
                if kind == "done":
                    return ("none", None, len(dets))
                if kind == "stop":
                    return ("none", None, len(dets))
                if kind in ("pick", "place"):
                    cands = [d0] + [d for d in cands if d is not d0]
                    forced, planner_forced = (d0, b0, var, table_xy), True
        elif self.brain is not None and self.plan_brain is None:
            sel = self._brain_select(cands, dets, obs)
            if sel is not None:
                kind, d0, b0, var = sel
                if kind == "look":
                    return ("relook", None, len(dets))
                if kind == "done":
                    return ("none", None, len(dets))
                if kind == "skip":
                    self.failed.append(d0.top[:2])          # the brain gave this object up: no more tries
                    cands = [d for d in cands if d is not d0]
                elif kind in ("pick", "retry"):
                    cands = [d0] + [d for d in cands if d is not d0]
                    forced = (d0, b0, var, None)
        for d in cands:
            if triage:  # a plan that did not fit before cannot fit now (time only runs out; plans start from home)
                rad = pol["same_object_radius"]
                if any(np.linalg.norm(d.top[:2] - p) < rad and dur > left for p, dur in self.too_long):
                    n_long += 1
                    continue
            others = [o.top[:2] for o in dets if o is not d]
            is_forced = forced is not None and d is forced[0]
            target_bin = forced[1] if is_forced else self._bin_of(d)
            var = forced[2] if forced is not None and d is forced[0] else None
            table_xy = forced[3] if is_forced else None
            if target_bin is None and table_xy is None:
                continue  # unmapped / abstain: 0 points on the counter, never a -1 guess
            if table_xy is not None:
                ids_now = self._brain_ids(dets)
                dmap_now = {ids_now[i]: obj for i, obj in enumerate(dets)}
                oid_now = next((oid for oid, obj in dmap_now.items() if obj is d), None)
                ok_place, why_place, _ = self._planner_place_at(dmap_now, dets, oid_now, table_xy)
                p = self._planner_place_cache.pop(
                    (id(d), round(float(table_xy[0]), 4), round(float(table_xy[1]), 4)), None)
                if not ok_place or p is None:
                    from .planner import Plan
                    p = Plan(False, color=d.color, target_bin="counter", reason=why_place or "place_at dry-run failed")
            elif verify:  # #6: explicit target bin + a free, verified drop slot
                slot = self._next_drop_slot(target_bin)
                if slot is None:
                    slot = next(iter(self._overflow_slots(target_bin)), None)
                if slot is None:
                    self.log(f"[classical] no free drop slot in {target_bin}; leave {d.color} on counter")
                    continue
                p = self._plan_variant(var, d, self.arm_cmd, self.hand_cmd, bin_name=target_bin, obstacles=others,
                                      drop_slot=slot)
            elif self._slots_on():  # grasp.drop_slots standalone: each successive object in a bin -> next slot
                free = [i for i in range(len(self.planner.drop_slots(target_bin)))
                        if i not in self.slot_used[target_bin]]
                slot = free[0] if free else next(iter(self._overflow_slots(target_bin)), None)
                if slot is None:
                    self.log(f"[classical] drop slots of {target_bin} used up; leave {d.color} on counter")
                    continue
                p = self._plan_variant(var, d, self.arm_cmd, self.hand_cmd, bin_name=target_bin, obstacles=others,
                                      drop_slot=slot)
            elif is_forced:   # brain override: the classical planner still checks reach
                p = self._plan_variant(var, d, self.arm_cmd, self.hand_cmd, bin_name=target_bin, obstacles=others)
            else:
                p = self._plan_variant(var, d, self.arm_cmd, self.hand_cmd, obstacles=others)
            self.plans.append(p)
            if p.ok:
                if triage and p.duration > left:  # feasible but too slow for the time left: try the next candidate
                    if planner_forced and is_forced:
                        self._planner_failed(dets, f"plan duration {p.duration:.1f}s exceeds time left", obs)
                        return ("planner_pending", dets, verify) if self.plan_brain.req.status == "pending" \
                            else ("none", None, len(dets))
                    self.too_long.append((d.top[:2], p.duration))
                    n_long += 1
                    shortest = p if shortest is None or p.duration < shortest.duration else shortest
                    self.log(f"[classical] {d.color} takes {p.duration:.1f} s > {left:.1f} s left: trying the next candidate")
                    continue
                if planner_forced and is_forced:
                    call = self.plan_brain.next_call()
                    self._planner_execution = {"call": call, "source_xy": np.asarray(d.top[:2], float).copy(),
                                               "color": d.color,
                                               "table_xy": None if table_xy is None else np.asarray(table_xy, float).copy()}
                return ("plan", p, len(dets))
            self.failed.append(d.top[:2])
            self.log(f"[classical] skip {d}: {p.reason}")
            if planner_forced and is_forced:
                self._planner_failed(dets, p.reason, obs)
                return ("planner_pending", dets, verify) if self.plan_brain.req.status == "pending" \
                    else ("none", None, len(dets))
        if verify and self.verify_relooks_remaining > 0:
            self.verify_relooks_remaining -= 1
            return ("relook", None, len(dets))
        if n_long:
            return ("notime", shortest, len(dets))
        return ("none", None, len(dets))

    # ---------------------------------------------------------- whole-plan brain
    def _planner_select(self, cands, dets, obs):
        """Return one validated plan primitive, or None so classical behaviour continues unchanged."""
        try:
            ids = self._brain_ids(dets)
            dmap = {ids[i]: d for i, d in enumerate(dets)}
            world = self._brain_world(dets, ids, None)
            for o in world["objects"]:
                d = dmap[o["id"]]
                o["size"] = round(float(getattr(d, "generic_width", None) or self.scene.size), 4)
            world["table_region"] = self._planner_table_region()
            world["find_enabled"] = bool(self.cfg["brain"].get("find", False))
            classical_has_work = any(self._lookup(d) is not None for d in cands)

            if not self.plan_brain.requested:
                self.plan_brain.request(world)        # first scene: async episode prefetch, never blocks act()
            if self.cfg["brain"].get("debug_wait", False):
                self.plan_brain.wait(float(self.cfg["brain"].get("plan_timeout_s", 2.0)))

            def reach(oid, bin_name):
                p = self.planner.plan(dmap[oid], self.arm_cmd, self.hand_cmd, bin_name=bin_name, relocate=False,
                                      obstacles=[d.top[:2] for d in dets if d is not dmap[oid]])
                return p.ok, p.reason

            def place(oid, xy, virtual_xy=None):
                return self._planner_place_at(dmap, dets, oid, xy, virtual_xy)

            steps = self.plan_brain.take(
                world, reach_fn=reach, place_fn=place,
                find_enabled=bool(self.cfg["brain"].get("find", False)),
                classical_has_work=classical_has_work,
            )
            if steps is None:
                return None
            while True:
                call = self.plan_brain.next_call()
                for skipped in self.plan_brain.skipped_ids:
                    dskip = dmap.get(skipped)
                    if dskip is not None and not any(np.linalg.norm(dskip.top[:2] - f) <
                                                     self.cfg["policy"]["same_object_radius"] for f in self.failed):
                        self.failed.append(np.asarray(dskip.top[:2], float).copy())
                if call is None:
                    return ("stop", None, None, None, None) if self.plan_brain.adopted else None
                tool, a = call["tool"], call["args"]
                if tool == "skip":
                    d = dmap.get(a["obj_id"])
                    if d is not None:
                        self.failed.append(np.asarray(d.top[:2], float).copy())
                    self.plan_brain.mark("ok", "left on table")
                    continue
                if tool == "find":
                    try:
                        found = [ids[i] for i in self._brain_find(a["description"], obs, dets)]
                        self.log(f"[planner] find({a['description']!r}) -> {found}")
                        self.plan_brain.mark("ok", f"found={found}")
                    except Exception as e:
                        self.plan_brain.mark("failed", repr(e))
                        self.plan_brain.replan(world, call, repr(e))
                        return None
                    continue
                if tool == "wait":
                    self.plan_brain.mark("ok")
                    return ("look", None, None, None, None)
                if tool == "done":
                    if not self.plan_brain.can_done(classical_has_work):
                        self.plan_brain.mark("rejected", "no-early-done guard")
                        continue
                    self.plan_brain.mark("ok")
                    return ("done", None, None, None, None)
                d = dmap.get(a["obj_id"])
                if d is None:                       # scene changed after validation: skip, never improvise
                    self.plan_brain.mark("invalid", "object no longer in scene")
                    continue
                if tool == "pick_place":
                    return ("pick", d, a["bin"], None, None)
                return ("place", d, None, None, np.asarray(a["table_xy"], float))
        except Exception as e:
            self.log(f"[planner] select failed ({e!r})")
            return (("stop", None, None, None, None)
                    if self.plan_brain is not None and self.plan_brain.adopted else None)

    def _planner_table_region(self):
        L = self.scene.cfg["layout"]
        corners = self.scene.mat_to_pelvis(np.array([[x, y, 0.0] for x in L["region_x"] for y in L["region_y"]]))
        lo, hi = corners[:, :2].min(0), corners[:, :2].max(0)
        return {"x": [round(float(lo[0]), 4), round(float(hi[0]), 4)],
                "y": [round(float(lo[1]), 4), round(float(hi[1]), 4)],
                "required_margin": 0.02, "clearance": 0.06}

    def _planner_place_at(self, dmap, dets, oid, xy, virtual_xy=None):
        """Geometry gate plus the classical motion/IK/SafetyFilter dry run for place_at."""
        xy = np.asarray(xy, float)
        reg = self._planner_table_region()
        lo = np.array([reg["x"][0], reg["y"][0]])
        hi = np.array([reg["x"][1], reg["y"][1]])
        margin = float(np.min(np.r_[xy - lo, hi - xy]))
        if margin < 0.02:
            return False, "outside reachable table region or within its 2 cm IK margin", margin
        # Distance to each bin's outer footprint, not merely its centre.
        outer = np.asarray(self.scene.cfg["bins"]["outer_size"], float) / 2
        for name, centre in self.scene.cfg["bins"]["centers"].items():
            c = np.asarray(centre, float)
            corners = self.scene.mat_to_pelvis(np.array([[c[0] + sx * outer[0], c[1] + sy * outer[1], 0.0]
                                                         for sx in (-1, 1) for sy in (-1, 1)]))[:, :2]
            blo, bhi = corners.min(0), corners.max(0)
            delta = np.maximum(np.maximum(blo - xy, 0.0), xy - bhi)
            if np.linalg.norm(delta) < 0.06:
                return False, f"less than 6 cm from {name} bin", margin
        positions = virtual_xy or {k: v.top[:2] for k, v in dmap.items()}
        for other_id, other_xy in positions.items():
            if other_id != oid and len(other_xy) == 2 and np.linalg.norm(xy - np.asarray(other_xy, float)) < 0.06:
                return False, f"less than 6 cm from object {other_id}", margin
        d = dmap[oid]
        obstacles = [o.top[:2] for o in dets if o is not d]
        p = self.planner.plan_place_at(d, self.arm_cmd, self.hand_cmd, xy, obstacles=obstacles)
        key = (id(d), round(float(xy[0]), 4), round(float(xy[1]), 4))
        if p.ok:
            self._planner_place_cache[key] = p
        return p.ok, p.reason, margin

    def _planner_track_move(self, color, source_xy, table_xy):
        """Keep a plan object's id stable when place_at moves it farther than the normal 5 cm tracker gate."""
        old = np.asarray(source_xy, float)
        candidates = [(np.linalg.norm(old - pxy), row) for row in self._brain_reg
                      for oid, col, pxy in [row] if col == color]
        if candidates:
            min(candidates, key=lambda x: x[0])[1][2] = np.asarray(table_xy, float).copy()

    def _planner_failed(self, dets, reason, obs):
        call = self.plan_brain.next_call()
        self.plan_brain.mark("failed", reason)
        try:
            ids = self._brain_ids(dets)
            world = self._brain_world(dets, ids, None)
            for o in world["objects"]:
                d = dets[ids.index(o["id"])]
                o["size"] = round(float(getattr(d, "generic_width", None) or self.scene.size), 4)
            world["table_region"] = self._planner_table_region()
            world["find_enabled"] = bool(self.cfg["brain"].get("find", False))
            self.plan_brain.replan(world, call, reason)   # one async retry, 0.5 s deadline
        except Exception as e:
            self.log(f"[planner] replan failed ({e!r})")

    def _planner_observe(self, dets, obs):
        """Confirm the prior primitive from the first settled post-motion look, then advance/replan."""
        ex = self._planner_execution
        if ex is None or self.plan_brain is None:
            return
        self._planner_execution = None
        rad = 2 * self.cfg["policy"]["same_object_radius"]
        source = ex["source_xy"]
        at_source = any(d.color == ex["color"] and np.linalg.norm(np.asarray(d.top[:2]) - source) < rad for d in dets)
        target = ex["table_xy"]
        if target is None:
            ok = not at_source                    # bin placement: object left the counter source
        else:
            ok = any(d.color == ex["color"] and np.linalg.norm(np.asarray(d.top[:2]) - target) < rad for d in dets)
        if ok:
            if target is not None:
                self._planner_track_move(ex["color"], source, target)
            self.plan_brain.mark("ok", "settled look confirmed")
        else:
            self._planner_failed(dets, "settled look did not confirm primitive", obs)

    # ------------------------------------------------------------------ brain (policy.brain)
    def _resolver_mapping(self, info, guarded=None):
        """Mapping opinions that may bind the brain: explicit reset mapping, the fixed task, or keys actually parsed
        by the resolver. The classical fallback remains unchanged but is not presented as an opinion for open vocab."""
        if self.brain is None:
            return {}, "disabled"
        try:
            if (info or {}).get("mapping"):
                return dict(info["mapping"]), "explicit"
            if guarded:
                return dict(guarded), "insert_guard"
            if (info or {}).get("task") == "sort_fixed":
                return dict(self.mapping), "fixed"
            if not (self.cfg.get("instruction") or {}).get("resolve", False):
                return {}, "unresolved"
            from .instruction import resolve
            parsed = resolve(str((info or {}).get("instruction") or ""), fallback={}) or {}
            return dict(parsed), "resolver" if parsed else "unresolved"
        except Exception:
            return {}, "unresolved"

    @staticmethod
    def _resolver_abstains(mapping):
        return not bool(mapping)

    def _brain_ids(self, dets):
        """Stable object ids across looks: a detection takes the id of the nearest same-colour registry entry
        (<= 5 cm, one-to-one), else a new one. The registry is per episode."""
        used, out = set(), []
        for d in dets:
            xy = np.asarray(d.top[:2], float)
            best, bd = None, 0.05
            for j, (oid, col, pxy) in enumerate(self._brain_reg):
                if j not in used and col == d.color and np.linalg.norm(xy - pxy) <= bd:
                    best, bd = j, np.linalg.norm(xy - pxy)
            if best is None:
                self._brain_reg.append([f"o{self._brain_n}", d.color, xy])
                self._brain_n += 1
                best = len(self._brain_reg) - 1
            else:
                self._brain_reg[best][2] = xy
            used.add(best)
            out.append(self._brain_reg[best][0])
        return out

    def _brain_world(self, dets, ids, lp):
        bc = self.cfg["brain"]
        calibrated = self.calibration is not None and self.calibration.success
        bins = [(n, self.planner.bin_xy(n), "calibrated" if calibrated else "scene") for n in self.scene.bin_names]
        reach = (lambda d, b: self.planner.plan(d, self.arm_cmd, self.hand_cmd, bin_name=b, relocate=False).ok) \
            if bc.get("world_reach", False) else None
        opinions = self._brain_mapping
        known_bin = lambda d: lookup_bin(self.cfg, opinions, d.color, getattr(d, "shape", None))
        w = build_world(dets, bins, self._instruction, self._task, opinions, known_bin, reach,
                           lambda d: self.perception._near_bin(d.top), ids,
                           {"id": lp["oid"], "color": lp["color"]} if lp else None)
        if self._find_on():
            w["queries"] = list(self._brain_queries)
        return w

    def _find_on(self):
        """brain.find (default off): the model may call find(query); at most brain.max_finds per episode."""
        bc = self.cfg.get("brain") or {}
        return bool(bc.get("find", False)) and bool(self._brain_queries) \
            and self._brain_finds < int(bc.get("max_finds", 2))

    def _brain_find(self, query, obs, dets):
        """Run the find tool for one phrase: image points -> ids of OUR detections (nearest uv)."""
        if self._brain_find_fn is None:
            bc = self.cfg["brain"]
            self._brain_find_fn = FindClient(bc.get("find_path", "/data/brain/find/find_tool.py"), bc.get("find_url"),
                                             float(bc.get("find_timeout_s", 1.5)))
        pts = self._brain_find_fn(obs["head_rgb"], query)
        return [d_i for d_i in match_found(pts, dets)]

    def _brain_points(self, lp, dets, cands, ids):
        """The decision points of the next cycle, with the classical call for each: [(point, state, classical, resolver_abstains)].
        lp = the pick just executed ({"xy","color","oid"}) or None. Shared by the real decision and the prefetch."""
        bc = self.cfg["brain"]
        idof = {id(d): ids[i] for i, d in enumerate(dets)}
        can_look = self._brain_looks < int(bc.get("max_looks", 2))
        first = next(((d, self._lookup(d)) for d in cands if self._lookup(d) is not None), None)
        if lp is not None and lp.get("oid") is not None:
            rad = 2 * self.cfg["policy"]["same_object_radius"]
            again = next((d for d in cands if d.color == lp["color"]
                          and np.linalg.norm(np.asarray(d.top[:2]) - lp["xy"]) < rad), None)
            if again is not None and first is not None and first[0] is again:   # re-detected where we picked: failed
                return [("after_fail", {"obj_id": idof[id(again)], "can_look": can_look, "allow_done": False},
                         {"tool": "retry", "args": {"obj_id": idof[id(again)], "variant": "same"}}, False)]
            pts = [("after_place", {"obj_id": lp["oid"], "can_look": can_look, "allow_done": False},
                    {"tool": "verify", "args": {"obj_id": lp["oid"]}} if self._verify_on()
                    else {"tool": "look", "args": {}}, False)]
        else:
            pts = []
        classical = ({"tool": "pick_place", "args": {"obj_id": idof[id(first[0])], "bin": first[1]}}
                     if first else {"tool": "done", "args": {}})
        st = {"candidates": [idof[id(d)] for d in cands], "can_look": can_look, "allow_done": first is None}
        if self._find_on():
            st["queries"] = list(self._brain_queries)
        pts.append(("after_look", st, classical,
                    self._brain_abstain))
        return pts

    def _brain_prefetch(self, plan):
        """At pick start: ask for the decisions of the NEXT cycle now (while the arm moves). Two predicted
        outcomes: placed (object gone) and failed (object still there). Used only if the state hash matches."""
        try:
            if self.brain is None or plan.corrective:
                return
            xy = np.asarray(plan.waypoints["grasp"][:2], float)
            lp = self._brain_pick(xy, plan.color)
            self._last_pick = lp                # always: the next look asks after_place / after_fail about it
            dets = list(self.last_detections)
            if not self.cfg["brain"].get("prefetch", True) or lp["oid"] is None or not dets:
                return
            rad = 2 * self.cfg["policy"]["same_object_radius"]
            ver = self._verify_on()
            for gone in (True, False):          # placed / failed
                ds = [d for d in dets if not (gone and d.color == plan.color
                                              and np.linalg.norm(np.asarray(d.top[:2]) - xy) < rad)]
                ids = self._brain_ids(ds)
                cands = self._ordered_cands(ds, ver)
                world = self._brain_world(ds, ids, lp)
                for point, state, cl, ab in self._brain_points(lp, ds, cands, ids):
                    self.brain.prefetch(point, world, state, cl, None, ab)
        except Exception as e:
            self.log(f"[brain] prefetch failed ({e!r})")

    def _brain_pick(self, xy, color):
        best, bd = None, 0.1
        for oid, col, pxy in self._brain_reg:
            if col == color and np.linalg.norm(np.asarray(xy) - pxy) <= bd:
                best, bd = oid, np.linalg.norm(np.asarray(xy) - pxy)
        return {"xy": np.asarray(xy, float).copy(), "color": color, "oid": best}

    def _brain_select(self, cands, dets, obs=None):
        """Brain decision points at a look (after_place / after_fail / after_look), each answered by the backend
        (index + confidence), validated and EV-gated. -> (kind, det, bin, variant) with kind pick | retry | skip |
        look | done, or None (= the classical planner decides). Never raises, never waits: the answer comes
        asynchronously (often prefetched); until then / on timeout / invalid / low confidence: None."""
        try:
            if self.mapping.get("default") == "abstain" and len(self.mapping) == 1:
                return None                     # insert_guard: pick nothing, no brain override
            bc = self.cfg["brain"]
            ids = self._brain_ids(dets)
            dmap = {ids[i]: d for i, d in enumerate(dets)}
            lp, self._last_pick = self._last_pick, None
            world = self._brain_world(dets, ids, lp)
            image = self._brain_image(obs) if bc.get("image", False) else None

            def reach_check(oid, b):
                p = self.planner.plan(dmap[oid], self.arm_cmd, self.hand_cmd, bin_name=b, relocate=False)
                return p.ok, p.reason

            for point, state, classical, ab in self._brain_points(lp, dets, cands, ids):
                self.brain.request(point, world, state, classical, image, ab)
                if bc.get("debug_wait", False):
                    self.brain.wait(float(bc.get("timeout_s", 2.0)))
                call = self.brain.take(reach_check)
                if call is not None and call["tool"] == "find":      # information only: ground the phrase, ask again
                    self._brain_finds += 1
                    q = call["args"]["query"]
                    found = [ids[i] for i in self._brain_find(q, obs, dets)]
                    self.log(f"[brain] find({q!r}) -> {found}")
                    w2 = {**world, "found": {q: found}, "queries": []}
                    self.brain.request(point, w2, {**state, "queries": []}, classical, image, ab)
                    if bc.get("debug_wait", False):
                        self.brain.wait(float(bc.get("timeout_s", 2.0)))
                    call = self.brain.take(reach_check)
                if call is None or call == classical:
                    if point == "after_fail":
                        return None
                    continue
                tool, a = call["tool"], call["args"]
                if point == "after_place":
                    if tool == "verify":                         # classical (verify off) just looks; verify = one more settled look
                        self._brain_looks += 1
                        return ("look", None, None, None)
                    if tool == "done":
                        return ("done", None, None, None)
                    continue
                if tool == "pick_place" and a["obj_id"] in dmap:
                    return ("pick", dmap[a["obj_id"]], a["bin"], None)
                if tool == "retry" and a["obj_id"] in dmap and self._lookup(dmap[a["obj_id"]]) is not None:
                    return ("retry", dmap[a["obj_id"]], self._lookup(dmap[a["obj_id"]]), a["variant"])
                if tool == "skip" and a["obj_id"] in dmap:
                    return ("skip", dmap[a["obj_id"]], None, None)
                if tool == "look":
                    self._brain_looks += 1
                    return ("look", None, None, None)
                if tool == "done":
                    return ("done", None, None, None)
            return None
        except Exception as e:
            self.log(f"[brain] select failed ({e!r}), classical decides")
            return None

    def _brain_image(self, obs):
        try:
            import base64
            import cv2
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(obs["head_rgb"]), cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf.tobytes()).decode() if ok else None
        except Exception:
            return None

    def _lookup(self, d):
        return lookup_detection_bin(self.cfg, self.mapping, d)

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
            target_bin = lookup_detection_bin(self.cfg, self.mapping, d)
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
        if kind == "planner_pending":
            self._planner_pending = (res[1], bool(res[2]))
            self.state = "planner_wait"
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
        if self.brain is not None and self.plan_brain is None:
            self._brain_prefetch(plan)
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
