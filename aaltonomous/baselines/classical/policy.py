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

import contextlib
import copy
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

from . import decision as dec
from .bin_vote import vote_bins
from .config import hand_postures, home_arm, load_config
from .held import HeldDetector
from .fk_xcheck import FkHandXCheck, FkXCheckResult
from .kinematics import ArmKinematics
from .perception import Perception
from .planner import Plan, Planner, lookup_bin, lookup_detection_bin, ramp
from .safety_shell import reserve_pair, reserve_plan
from .brain import BrainLayer, build_world
from .brain.find import FindClient, extract_queries, match_found

HZ = 30.0
_HOLE = re.compile(r"\bholes?\b")   # policy.insert_guard: the rung-3 sentence names a hole AND the lid
_LID = re.compile(r"\blids?\b")
_TRUNC_CUT = ".;!?\n"   # instruction.max_chars: sentence / clause boundaries, then whitespace
_WS = " \t\r\n"


def _lower_thread_priority(nice=10):
    """Linux: raise the nice value of the calling thread only (setpriority on its native thread id)."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except (AttributeError, OSError):
        pass


class ClassicalPolicy:
    @staticmethod
    def _safety_hand_limits(spec, safety):
        """SafetyFilter._shrink for the hand, kept read-only and Pinocchio-free."""
        margin = np.minimum(safety.hand_pos_margin, 0.25 * (spec.hand_upper - spec.hand_lower))
        return spec.hand_lower + margin, spec.hand_upper - margin

    def __init__(self, config=None, scene=None, calibration=None, safety: SafetyConfig | None = None,
                 require_confirm=False, threaded=False, time_limit=120.0, log=print, overrides=None):
        self.cfg = config if isinstance(config, dict) else load_config(config, overrides)
        self.held_detector = HeldDetector(self.cfg["grasp"].get("held_detector", "stall"),
                                          self.cfg["grasp"].get("held_threshold_rad", 0.045))
        self.scene = scene if isinstance(scene, Scene) else Scene.load(scene)
        self.cal = calibration if isinstance(calibration, Calibration) else Calibration.load(calibration)
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)
        self.home = home_arm(self.cfg, self.spec)
        self.kin = ArmKinematics(self.cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=1.0 / HZ)
        # Mirror SafetyFilter's hand position clamp without constructing or
        # mutating a filter. The detector must compare hand_q with the command
        # that the runner actually accepts at the URDF-margin stage.
        self._held_hand_lo, self._held_hand_hi = self._safety_hand_limits(self.spec, self.safety)
        self.perception = Perception(self.cfg, self.scene, self.cal, log=log or (lambda *a: None))
        # perception.hand_calibration (default OFF): head-camera pose from the arms in depth. The arm model + URDF load
        # here (~0.5 s, outside any episode); a load failure only disables the feature.
        self.hand_calib = None
        hcfg = self.cfg["perception"].get("hand_calibration") or {}
        if hcfg.get("enabled", False):
            try:
                from .hand_calib import HandCalibrator
                self.hand_calib = HandCalibrator(hcfg)
                _ = self.hand_calib.surface
                pc = self.cfg["perception"]
                others = [k for k, on in (("xy_offset", any(pc.get("xy_offset") or [])), ("y_affine", pc.get("y_affine")),
                                         ("self_calibration", (pc.get("self_calibration") or {}).get("enabled")))
                          if on]
                if others:     # all correct the same head-camera error: stacking them double-corrects
                    (log or print)(f"WARNING hand_calibration is ON together with {others}: double-correction risk")
            except Exception as e:
                self.hand_calib = None
                (log or print)(f"hand calibration disabled: {type(e).__name__}: {e}")
        self.planner = Planner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec)
        # calib.fk_xcheck (FLAG, default OFF, log-only): RGB head-pose cross-check vs the Dex3 hand FK. Its own
        # ArmKinematics: the worker thread must never touch the planner's pinocchio data.
        self.fk_xcheck_enabled = bool((self.cfg.get("calib") or {}).get("fk_xcheck", False))
        if self.fk_xcheck_enabled:
            self._fk_xcheck_kin = ArmKinematics(self.cfg, self.spec)
            self._fk_xcheck = FkHandXCheck(self.cfg["calib"]["fk_xcheck_config"], self.perception)
        self.perception.bin_shift_src = self.planner   # perception.color_gate.live_bins reads planner.bin_shift (bin.vote)
        self.perception.bin_vote_on = bool(((self.cfg.get("bin") or {}).get("vote") or {}).get("enabled"))   # bin footprints follow the shift
        self.require_confirm = require_confirm
        self._shell = bool(self.cfg["policy"].get("safety_shell", False))
        if self._shell:   # policy.safety_shell: all heavy work off the 30 Hz loop (look/plan thread), plans within the reserve
            threaded = True
            self._shell_init()
        self.threaded = threaded
        if threaded:  # pure-Python planning holds the GIL: hand it back to act() sooner (default 5 ms)
            sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
        self.time_limit = time_limit
        self.log = log or (lambda *a: None)
        bc = self.cfg.get("brain") or {}
        self.brain = BrainLayer(bc, log=self.log) if bc.get("enabled", False) else None   # FLAG brain.enabled
        self.reset()

    # ------------------------------------------------------------------ API
    def _cap_text(self, text):
        """instruction.max_chars: cut a long instruction at the last boundary within the cap (logged). Unchanged when off."""
        mc = int((self.cfg.get("instruction") or {}).get("max_chars", 0) or 0)
        if mc <= 0 or not isinstance(text, str) or len(text) <= mc:
            return text
        head = text[:mc]
        cut = max(head.rfind(c) for c in _TRUNC_CUT) + 1
        if cut <= 0:
            cut = max(head.rfind(c) for c in _WS) + 1 or mc
        self.log(f"[instruction] truncated {len(text)}->{cut} chars")
        return text[:cut]

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
            text = self._cap_text((info or {}).get("instruction") or "")
            if not text:
                return None
            from .instruction import resolve
            if (self.cfg.get("instruction") or {}).get("normalize", False):   # typo/synonym normaliser (Felix)
                from .normalize import normalize
                text = normalize(text)
            if (self.cfg.get("instruction") or {}).get("strict", False):   # never guess: abstain (Felix strict)
                from .normalize import strict_mapping
                return strict_mapping(text, resolve(text, fallback={}))
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
        self.time_limit = info.get("time_limit", self.time_limit)
        self._resolve_time_alias()
        self.state = "init"
        self.queue_a = self.queue_h = None
        self.arm_cmd = self.hand_cmd = None
        self.attempts = []          # (position, count) of objects we tried to pick
        self.attempt_tracks = []    # retry-vary: {color, latest xy, cumulative count} stable pushed-object tracks
        self.object_time_tracks = []  # EV budget: {color, latest xy, cumulative planned motion seconds}
        self.budget_abandoned = []  # (color, xy) objects whose hard per-object budget was exhausted
        self.gate_skipped = []      # (color, xy) objects deliberately left on the counter by place_gate
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
        self.held_detector.reset()
        self._bin_misses = []       # (colour, xy) already logged as [bin_lookup] miss this episode
        self._stall = None          # motion.lift_on_stall: the close hold being watched
        self.retry_pass_used = False
        self.vary_hist = []         # policy.retry_vary_grasp: {color, xy, yaw, posture, side, held} per started pick
        self._summary_logged = False
        self._emitted = None        # safety.nan_guard: (arm, hand) last row of the last action returned this episode
        # Bump the episode token under the lock: a stale worker from the previous episode then fails its token check
        # and can never write into this one. act() never waits on the worker (nominal extrinsics until it lands).
        lock = self.__dict__.setdefault("_pitch_lock", threading.Lock())
        with lock:
            self._pitch_token = getattr(self, "_pitch_token", 0) + 1
            self.perception.reset_calibration()
            self.pitch_calibration = None
            self._pitch_calibration_done = False   # = the worker was started this episode
        self._pitch_thread = None
        self.planner.bin_shift, self.planner.drop_extra, self.planner.bin_conf = {}, {}, {}   # bin.vote: nominal until voted
        self._bv_frames, self._bv_done = [], False
        self.calibration = None
        self._calibration_done = False
        self.hand_calibration = None    # HeadCalibResult of this episode (perception.hand_calibration)
        self._hc_done = self.hand_calib is None
        self._hc_tries = 0
        self._hc_prev_q = None
        self._hc_job = None
        self._hc_still = 0
        self.last_bin_detections = []
        self.used_drop_slots = {name: set() for name in self.scene.bin_names}
        self.reserved_drop_slots = {name: set() for name in self.scene.bin_names}
        self.verification_pending = None
        self.last_verification = None
        self.retry_colors = set()
        self.retry_targets = []
        self.verify_relooks_remaining = 0
        self.scheduled_corrections = []
        self._joint_offset_t0 = None                      # debug.log_joint_offset: start of the first-second window
        self._last_policy_arm_cmd = self._last_policy_hand_cmd = None
        self._last_policy_cmd_steady = False
        # decision layer (policy.confidence_order / object_time_cap / speed_tiers / confidence_tiers); inert when off
        self.tiers = dec.SpeedTiers(self._dcfg())   # D3 state
        self.tier = None            # D4: CONFIDENT | DEGRADED | PRIOR-ONLY once the hold is done
        self.C = None               # D4: calibration confidence
        self._calib = None          # D4: the running hold {"t0", "frames", "depth", "job"}
        self._stab = []             # recent frames [(colour, xy)] for the temporal-stability term
        self._stab_last = None
        self._pend_out = None       # D3: the pick whose outcome the next look judges
        self._exec = None           # D2 T-8 s: the plan being executed {"plan", "n", "cut", "t8"}
        self._cap_logged = set()
        self._dec_warned = set()    # F7: decision helpers that already fell back to stock this episode
        self._prior_side = set()    # F1: side bins already attempted in PRIOR-ONLY
        if self.fk_xcheck_enabled:
            self._fk_xcheck_generation = getattr(self, "_fk_xcheck_generation", 0) + 1
            self._fk_xcheck_thread = getattr(self, "_fk_xcheck_thread", None)
            self._fk_xcheck_pending = None
            self._fk_xcheck_last_t = -np.inf
            self._fk_xcheck_last_log = None
            self.fk_xcheck_result = None

    # ------------------------------------------------------------------ calib.fk_xcheck (B2, log-only)
    def _online_pitch_deg(self):
        """Pitch the online pitch calibration already applies to the pose in use (None = not active)."""
        pc = self.pitch_calibration
        return None if pc is None or not pc.success else float(pc.pitch_deg)

    def _poll_fk_xcheck(self, obs, t):
        """Publish a finished estimate and launch a throttled worker; act() never waits for the search."""
        thread = self._fk_xcheck_thread
        if thread is not None and not thread.is_alive():
            self._fk_xcheck_thread = None
            result, self._fk_xcheck_pending = self._fk_xcheck_pending, None
            if result is not None:
                self.fk_xcheck_result = result
                signature = (result.visible, round(result.pitch_deg, 1), round(result.yaw_deg, 1), result.reason)
                if signature != self._fk_xcheck_last_log:
                    agreement = (" standalone" if result.online_pitch_deg is None else
                                 f" online={result.online_pitch_deg:+.2f}deg delta={result.disagreement_deg:.2f}deg")
                    self.log(f"[fk_xcheck] visible={result.visible} residual pitch={result.pitch_deg:+.2f}deg "
                             f"yaw={result.yaw_deg:+.2f}deg confidence={result.confidence:.2f} "
                             f"points={result.n_points} error={result.error_px:.1f}px time={result.elapsed_ms:.1f}ms"
                             f"{agreement} ({result.reason})")
                    self._fk_xcheck_last_log = signature
        if self._fk_xcheck_thread is not None:
            return
        period = float(self.cfg["calib"]["fk_xcheck_config"].get("period_s", 0.5))
        rgb, depth = obs.get("head_rgb"), obs.get("head_depth")
        arm_q, hand_q = obs.get("arm_q"), obs.get("hand_q")
        if t - self._fk_xcheck_last_t < period or rgb is None or arm_q is None or hand_q is None:
            return
        self._fk_xcheck_last_t = t
        rgb = np.asarray(rgb).copy()
        depth = None if depth is None else np.asarray(depth).copy()
        arm_q, hand_q = np.asarray(arm_q, float).copy(), np.asarray(hand_q, float).copy()
        online_pitch, generation = self._online_pitch_deg(), self._fk_xcheck_generation

        def work():
            try:
                points = np.vstack([self._fk_xcheck_kin.hand_points(side, arm_q, hand_q)
                                    for side in ("left", "right")])
                out = self._fk_xcheck.estimate(rgb, points, depth, online_pitch)
            except Exception as exc:
                out = FkXCheckResult(False, 0.0, 0.0, 0.0, 0, np.inf, 0.0,
                                     f"worker error: {type(exc).__name__}: {exc}")
            if generation == self._fk_xcheck_generation:
                self._fk_xcheck_pending = out

        self._fk_xcheck_thread = threading.Thread(target=work, name="fk-xcheck", daemon=True)
        self._fk_xcheck_thread.start()

    def _start_pitch_worker(self, rgb, depth):
        """Run the camera-pitch fit ONCE per episode in a daemon thread, started on the first valid frame."""
        self._pitch_calibration_done = True
        token, rgb, depth = self._pitch_token, np.array(rgb), np.array(depth)

        def work():
            started = time.perf_counter()
            try:
                out = self.perception.compute_pitch(rgb, depth)
            except Exception as exc:  # optional calibration must never hurt the episode
                out = self.perception.make_pitch_failure(f"{type(exc).__name__}: {exc}",
                                                         (time.perf_counter() - started) * 1000.0)
            with self._pitch_lock:
                if token != self._pitch_token:
                    return
                self.perception.commit_pitch(out)   # atomic swap; every tick before this used nominal extrinsics
                self.pitch_calibration = out
            self._log_pitch(out)

        self._pitch_thread = threading.Thread(target=work, name="pitch_cal", daemon=True)
        self._pitch_thread.start()

    def _log_pitch(self, pitch_cal):
        if pitch_cal.mode == "table":
            state = "accepted" if pitch_cal.success else "rejected"
            self.log(f"[pitch_cal] mode=table {state}: dpitch={pitch_cal.pitch_deg:+.2f} deg, "
                     f"droll={pitch_cal.roll_deg:+.2f} deg, height={pitch_cal.camera_height_m:.3f} m, "
                     f"inliers={pitch_cal.n_inliers}/{pitch_cal.n_points}, "
                     f"rms={pitch_cal.residual_m * 1000.0:.1f} mm, "
                     f"time={pitch_cal.elapsed_ms:.1f} ms ({pitch_cal.reason})")
        else:
            state = "accepted" if pitch_cal.success else "fallback"
            self.log(f"[pitch_cal] {state}: dtheta={pitch_cal.pitch_deg:+.2f} deg, "
                     f"residual={pitch_cal.residual_m * 1000.0:.1f} mm, "
                     f"points={pitch_cal.n_points}, time={pitch_cal.elapsed_ms:.1f} ms "
                     f"({pitch_cal.reason})")

    def _hand_calibrate(self, obs, t):
        """perception.hand_calibration, ASYNC: on a still frame (encoders vs depth in sync) the 30-300 ms ICP runs on a
        worker thread over copies of depth/arm_q/hand_q; act() keeps returning its normal action and only polls the
        job (a few us). A finished success -> perception uses a corrected copy until the next reset; a failure retries
        on a later still frame (max_tries); a job older than job_timeout_s is abandoned -> nominal, logged. Never raises."""
        job = self._hc_job
        if job is not None:
            if job["done"].is_set():
                self._hc_job = None
                self._hand_calib_apply(job["res"], t)
            elif time.perf_counter() - job["t0"] > float(self.hand_calib.p.get("job_timeout_s", 1.5)):
                self._hc_job = None
                self._hc_done = True
                self._note(t, f"hand calibration fallback: solve timed out (> {self.hand_calib.p.get('job_timeout_s', 1.5)} s)")
            return
        try:
            p = self.hand_calib.p
            q = np.array(obs["arm_q"], float)
            hq = None if obs.get("hand_q") is None else np.array(obs["hand_q"], float)
            cur = q if hq is None else np.r_[q, hq]
            prev, self._hc_prev_q = self._hc_prev_q, cur
            if prev is None or prev.shape != cur.shape or np.abs(cur - prev).max() > p["still_rad"]:
                self._hc_still = 0
            else:
                self._hc_still += 1
            # still_ticks consecutive still ticks (arms AND hands): the depth lags the encoders by 50-100 ms
            if self._hc_still < int(p.get("still_ticks", 5)) or obs.get("head_depth") is None:
                return
            self._hc_tries += 1
            from .hand_calib import corrected_calibration
            start = corrected_calibration(self.perception.cal, self.perception.cam, self.perception.current_T())
            depth = np.array(obs["head_depth"])
        except Exception as e:
            self._hc_done = True
            self._note(t, f"hand calibration fallback: {type(e).__name__}: {e}")
            return
        job = {"done": threading.Event(), "res": None, "t0": time.perf_counter()}
        self._hc_job = job
        self._hc_still = 0
        cam, solver = self.perception.cam, self.hand_calib
        lock = bool(self.cfg["perception"].get("online_pitch_cal"))   # A owns pitch/roll: B1 only yaw + xyz

        def work():
            try:
                job["res"] = solver.solve(start, cam, depth, q, hq, lock_tilt=lock)
            except Exception as e:   # solve() never raises; belt and braces
                job["res"] = e
            finally:
                job["done"].set()

        threading.Thread(target=work, daemon=True, name="hand-calib").start()

    def _hand_calib_apply(self, res, t):
        if isinstance(res, Exception):
            self._hc_done = True
            self._note(t, f"hand calibration fallback: {type(res).__name__}: {res}")
            return
        if res.success:
            self.perception.set_camera_pose(res.T)
        elif self._hc_tries < int(self.hand_calib.p["max_tries"]):
            return
        self._hc_done = True
        self.hand_calibration = res
        self._note(t, f"hand calibration {'accepted' if res.success else 'fallback'} (try {self._hc_tries}): "
                      f"{res.summary()}")

    def confirm(self):
        self._decision = "go"

    def skip(self):
        self._decision = "skip"

    def stop(self):
        self._decision = "stop"

    def act(self, obs):
        ng = bool((self.cfg.get("safety") or {}).get("nan_guard", False))
        if ng:
            obs, hold = self._nan_guard(obs)
            if hold is not None:
                return hold
        if self.fk_xcheck_enabled:   # before both paths: the safety shell's act() is the same entry point
            self._poll_fk_xcheck(obs, float(obs.get("t", 0.0)))
        res = self._act_shell(obs) if self._shell else self._act(obs)
        if ng:
            a, h = np.asarray(res["arm_q_target"], float), np.asarray(res["hand_q_target"], float)
            if len(a) and len(h):
                self._emitted = (a[-1].copy(), h[-1].copy())
        return res

    # ------------------------------------------------------------------ safety.nan_guard
    _nan_t = -1e9

    def _nan_log(self, msg):
        now = time.monotonic()
        if now - self._nan_t >= 1.0:   # rate limit: at most one line per second
            self._nan_t = now
            self.log(f"[nan_guard] {msg}")

    @staticmethod
    def _fallback_pose(q, home):
        """Per joint: the measured value where it is finite, else the home value of that joint (self.home =
        config home_arm_q, hand: self.post['open'] from config hand.open). A missing or wrong-shape reading gives the
        whole home pose. Never zeros: a zero target is a large move for the arms."""
        home = np.asarray(home, float)
        try:
            v = np.asarray(q, float).reshape(-1)
        except (TypeError, ValueError):
            return home.copy()
        if v.shape != home.shape:
            return home.copy()
        return np.where(np.isfinite(v), v, home)

    def _nan_guard(self, obs):
        """Non-finite arm_q / hand_q: skip the planner this tick and hold the last emitted action (before any action this
        episode: measured-or-home pose, see _fallback_pose). Non-finite head_depth pixels -> 0 = missing depth: perception keeps only
        depth_range_mm (config.yaml, 150..2000 mm) so 0 is invalid there (perception.py:369, :336). All-finite obs: returned
        as is. Returns (obs, hold_action or None)."""
        if not isinstance(obs, dict):
            return obs, None
        bad_q = False
        for k in ("arm_q", "hand_q"):
            v = obs.get(k)
            if v is not None:
                try:
                    bad_q = bad_q or not bool(np.all(np.isfinite(np.asarray(v, float))))
                except (TypeError, ValueError):
                    pass   # malformed: _act's own handling is unchanged
        dep = obs.get("head_depth")
        if isinstance(dep, np.ndarray) and dep.dtype.kind == "f" and not np.isfinite(dep).all():
            obs = {**obs, "head_depth": np.where(np.isfinite(dep), dep, dep.dtype.type(0))}
            self._nan_log(f"head_depth: {int((~np.isfinite(dep)).sum())} non-finite px -> 0 (missing depth)")
        if not bad_q:
            return obs, None
        if self._emitted is not None:
            a, h = self._emitted
            what = "last emitted action"
        else:   # nothing emitted yet: per joint the measured value where finite, else home (never zeros)
            a = self._fallback_pose(obs.get("arm_q"), self.home)
            h = self._fallback_pose(obs.get("hand_q"), self.post["open"])
            what = "measured-or-home pose (none emitted yet)"
        self._nan_log(f"non-finite arm_q/hand_q: planner skipped, holding {what}")
        k = max(1, int(self.cfg["motion"].get("hold_ticks", 1)))
        return obs, {"arm_q_target": np.repeat(np.asarray(a, float)[None], k, 0),
                     "hand_q_target": np.repeat(np.asarray(h, float)[None], k, 0), "done": False}

    # ------------------------------------------------------------------ policy.safety_shell
    def _shell_init(self):
        sf, res = self.safety, float(self.cfg["policy"].get("safety_shell_vel_reserve", 0.0))
        lim = (sf.dt, sf.arm_max_vel, sf.hand_max_vel, res)
        self._shell_budget = float(sf.hold_after_s)   # beyond this the filter calls the policy stale (stop_after_s: latched stop)
        self._shell_stats = {"miss": 0, "exception": 0, "overrun": 0}
        self._shell_last = None                         # (arm (14,), hand (14,)) last row actually handed to the runner
        self._jobs = {}
        if res <= 0.0:   # default: no time-stretching, no planner wrappers
            return
        pl = self.planner
        for name in ("plan", "plan_to_counter"):        # every plan, whoever asks (_look runs in the thread), is within the reserve
            fn = getattr(pl, name)
            setattr(pl, name, lambda d, a, h, *args, _fn=fn, **kw: reserve_plan(_fn(d, a, h, *args, **kw), a, h, *lim))
        go = pl.go_to
        pl.go_to = lambda af, hf, at, ht: reserve_pair(*go(af, hf, at, ht), af, hf, *lim)

    def _bg(self, key, fn, *args):
        """Run fn(*args) once in a low-priority thread; None while running, then ("ok", value) / ("err", exc)."""
        job = self._jobs.get(key)
        if job is None:
            job = self._jobs[key] = {"result": None}

            def run():
                _lower_thread_priority()
                try:
                    job["result"] = ("ok", fn(*args))
                except Exception as e:
                    job["result"] = ("err", e)
            threading.Thread(target=run, daemon=True).start()
        if job["result"] is None:
            return None
        return self._jobs.pop(key)["result"]

    def _shell_hold(self, obs):
        if self._shell_last is not None:
            a, h = self._shell_last
        elif self.arm_cmd is not None:
            a, h = self.arm_cmd, self.hand_cmd
        else:
            # nothing commanded yet: measured where finite, else home (same rule as _nan_guard; never zeros)
            a = self._fallback_pose(obs.get("arm_q"), self.home)
            h = self._fallback_pose(obs.get("hand_q"), self.post["open"])
        k = max(1, int(self.cfg["motion"].get("hold_ticks", 1)))
        return {"arm_q_target": np.repeat(np.asarray(a, float)[None], k, 0),
                "hand_q_target": np.repeat(np.asarray(h, float)[None], k, 0), "done": False}

    def _act_shell(self, obs):
        """act() that never raises and never returns an untested target late: exception or compute over budget ->
        hold the last commanded target (a chunk already popped for the discarded answer is put back in front of the queue)."""
        st = self._shell_stats
        self._popped = None
        t0 = time.perf_counter()
        why = None
        try:
            res = self._act(obs)
            if time.perf_counter() - t0 > self._shell_budget:
                why, st["overrun"] = "overrun", st["overrun"] + 1
        except Exception as exc:
            res, why = None, f"exception {type(exc).__name__}: {exc}"
            st["exception"] += 1
        if why is None:
            self._shell_last = (np.array(res["arm_q_target"][-1], float), np.array(res["hand_q_target"][-1], float))
            return res
        if self._popped is not None:       # undo the pop: the runner never executes it
            a, h = self._popped
            self.queue_a = a if self.queue_a is None else np.vstack([a, self.queue_a])
            self.queue_h = h if self.queue_h is None else np.vstack([h, self.queue_h])
            if self._shell_last is not None:
                self.arm_cmd, self.hand_cmd = self._shell_last[0].copy(), self._shell_last[1].copy()
        st["miss"] += 1
        if st["miss"] % 10 == 1:           # once per 10 misses
            self.log(f"[safety_shell] hold-last ({why}); misses={st['miss']} overruns={st['overrun']} "
                     f"exceptions={st['exception']} budget={self._shell_budget:.2f}s")
        return self._shell_hold(obs)

    def _held_command(self):
        """Policy's last hand target, with the safety filter's position clamp."""
        if self.held_detector.mode == "finger_sum":
            return np.clip(np.asarray(self.hand_cmd, float), self._held_hand_lo, self._held_hand_hi)
        return self.hand_cmd

    def _act(self, obs):
        first = self.arm_cmd is None
        if first:  # first call: start from the measured pose
            # Clamp into the URDF limits: real Dex3 fingers at rest read slightly past 0 (2026-11-11), and holding
            # that measured pose would be clipped every tick -> safety abort before the first move.
            sp = self.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        t = float(obs.get("t", 0.0))
        if first:
            self._ep_t0 = t
        if self.held_detector.mode == "finger_sum":
            if first:
                # The first command is synthesized from this same measurement,
                # so it carries no tracking-bias information. Start the window
                # now and sample subsequent actual command/measurement pairs.
                self.held_detector.start(t)
            elif obs.get("hand_q") is not None:
                self.held_detector.observe(t, self._held_command(), obs["hand_q"])
        self._joint_offset_diag(obs, t)
        if (not self._pitch_calibration_done and self.cfg["perception"].get("online_pitch_cal", False)
                and self.perception.pitch_frame_valid(obs.get("head_rgb"), obs.get("head_depth"))):
            self._start_pitch_worker(obs.get("head_rgb"), obs.get("head_depth"))
        if not self._calibration_done and (self.cfg["perception"].get("self_calibration") or {}).get("enabled", False):
            started = time.perf_counter()
            try:
                if self._shell:   # heavy (depth fit): thread, hold meanwhile
                    r = self._bg("cal", self.perception.calibrate_bins, obs.get("head_depth"))
                    if r is None:
                        return self._hold()
                    if r[0] == "err":
                        raise r[1]
                    cal = r[1]
                else:
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
        if not self._hc_done:
            self._hand_calibrate(obs, t)
        if self._gdiag is not None:
            self._grasp_diag(obs)
        if self._stall is not None:
            self._lift_on_stall(obs)
        if self._exec is not None and self.queue_a is not None and len(self.queue_a):
            try:
                r = self._endgame(t)      # policy.object_time_cap: T-8 s release over the bin / retract (inert when off)
            except Exception as e:
                self._dec_fail("_endgame", e)
                r = None
            if r is not None:
                return r
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
        if self.state == "look" and not self._bv_done and (self.cfg.get("bin") or {}).get("vote", {}).get("enabled"):
            # bin.vote: first n_frames looks at home, before any pick; then freeze the median, never again
            v = self.cfg["bin"]["vote"]
            self._bv_frames.append({"head_rgb": obs.get("head_rgb"), "head_depth": obs.get("head_depth")})
            if len(self._bv_frames) < int(v["n_frames"]):
                return self._hold()
            self._bv_done = True
            shift, extra, conf = vote_bins(self.perception, self.planner, self._bv_frames, self.cfg, log=lambda m: self._note(t, m))
            self.planner.bin_shift, self.planner.drop_extra, self.planner.bin_conf = shift, extra, conf
            self._bv_frames = []
        if self.state == "look" and self._await_pitch_cal(t):
            return self._hold()
        if self.state == "look":
            if self._dflag("confidence_tiers") and self.tier is None:
                try:
                    r = self._calib_step(obs, t)   # first seconds: hold still, measure C, pick the tier
                except Exception as e:
                    self._dec_fail("_calib_step", e)
                    self.tier, r = dec.CONFIDENT, None
                if r is not None:
                    return r
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
            return self._act(obs)
        return self._hold(done=True)

    PITCH_WAIT_S = 1.0   # perception.online_pitch_cal: the first look waits at most this long (sim time) for the commit

    def _await_pitch_cal(self, t):
        """online_pitch_cal ON: the first look must not run on the nominal extrinsics before the fit committed (it would
        see 0 detections and end the episode). True = hold this tick (max PITCH_WAIT_S from the episode start). Threaded:
        never blocks. Not threaded: bounded join of the fit worker. Flag off: always False."""
        if not self.cfg["perception"].get("online_pitch_cal", False) or self.pitch_calibration is not None:
            return False
        if t - getattr(self, "_ep_t0", t) >= self.PITCH_WAIT_S:
            return False
        th = self._pitch_thread
        if not self.threaded and th is not None:
            th.join(timeout=2.0)
            return self.pitch_calibration is None and th.is_alive()
        return True

    # ------------------------------------------------------------------ look + plan
    def _look_job(self, job):
        _lower_thread_priority()  # planning must not starve act() (the control loop's thread)
        try:
            job["result"] = self._look(job["obs"])
        except Exception as e:  # never kill the control loop
            job["result"] = ("error", repr(e))

    def _detect_kw(self):
        """Only with perception.colour_kmeans: tell detect() whether the arm is out of view (parked at home)."""
        if not self.cfg["perception"].get("colour_kmeans", False):
            return {}
        return {"arm_clear": bool(np.abs(self.arm_cmd - self.home).max() < 0.05)}

    def _fk_kw(self, obs):
        """Only with perception.color_gate.fk_mask: the measured arm / hand pose for detect()'s arm mask."""
        if not (self.cfg["perception"].get("color_gate") or {}).get("fk_mask", False) or obs.get("arm_q") is None:
            return {}
        return {"robot_q": (np.asarray(obs["arm_q"], float),
                            None if obs.get("hand_q") is None else np.asarray(obs["hand_q"], float))}

    def _look(self, obs):
        verify = self._verify_on()
        if not verify:  # flag off: the stock look (no bin review, no corrective picks, no relooks)
            dets = self.perception.detect(obs["head_rgb"], obs["head_depth"], **self._detect_kw(), **self._fk_kw(obs))
            self.last_detections = dets
            return self._choose_retry(dets, obs, verify=False)
        dets, rejected = self.perception.detect(obs["head_rgb"], obs["head_depth"], debug=True, include_bins=True,
                                                   **self._detect_kw(), **self._fk_kw(obs))
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
            budget_blocked = False
            if self._vary_on():   # no place gate: the object is already in a bin
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
                if p.ok and not self._object_time_allows(d, p.duration):
                    budget_blocked = True
                    break
                if p.ok:
                    return ("plan", p, len(dets))
                self.log(f"[classical] cannot correct {d.color} from {source_bin} to slot {slot}: {p.reason}")
            if budget_blocked:
                continue
            def cleanup(det, arm, hand, planner=None, **_):
                return (planner or self.planner).plan_to_counter(det, arm, hand, source_bin,
                                                    obstacles=[o.top[:2] for o in dets])
            p = self._plan_pick(d, {}, plan_fn=cleanup)
            self.plans.append(p)
            if p.ok and not self._object_time_allows(d, p.duration):
                continue
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

    def _attempt_excluded(self, xy, color=None):
        """True when the object at xy is skipped for its attempt count only (not in `failed`)."""
        rad = self._attempt_radius()
        if any(np.linalg.norm(xy - f) < rad for f in self.failed):
            return False
        return self._attempt_count(xy, color) >= self._attempt_limit(xy)

    def _tracked_here(self, tracks, color, xy):
        """Nearest same-colour tracked item inside the existing attempt association radius."""
        xy = np.asarray(xy, float)
        near = [item for item in tracks if item[0] == color and np.linalg.norm(xy - item[1]) < self._attempt_radius()]
        return None if not near else min(near, key=lambda item: np.linalg.norm(xy - item[1]))

    GATE_MAX_SKIPS = 3

    def _bin_confidence(self, d):
        """Confidence in the destination: instruction mapping and bin geometry."""
        mapping = float(self.mapping_conf.get(d.color, 1.0))
        # Only a successful calibration is evidence; failed/fallback/rejected/none means nominal bins (safe, 1.0).
        cal = self.calibration
        calibration = float(cal.confidence) if cal is not None and getattr(cal, "success", False) is True else 1.0
        return float(np.clip(min(mapping, calibration), 0.0, 1.0))

    def _place_gate_allows(self, d, target_bin, remember=True):
        """EV placement decision for a counter object.  Off is an unconditional pass."""
        pol = self.cfg["policy"]
        if not pol.get("place_gate", False):
            return True
        color_conf = float(getattr(d, "color_conf", 1.0))
        bin_conf = self._bin_confidence(d)
        color_min = float(pol.get("place_gate_colour_min", 0.6))
        bin_min = float(pol.get("place_gate_bin_min", 0.5))
        ok = np.isfinite(color_conf) and color_conf >= color_min and np.isfinite(bin_conf) and bin_conf >= bin_min
        if not ok and remember:
            xy = np.asarray(d.top[:2], float).copy()
            tr = self._tracked_here(self.gate_skipped, d.color, xy)
            if tr is None:
                tr = [d.color, xy, 0]
                self.gate_skipped.append(tr)
            else:
                tr[1] = xy
            tr[2] += 1  # a skip is re-evaluated on later looks; _choose stops offering it after 3 skips
            reason = "colour" if not np.isfinite(color_conf) or color_conf < color_min else "bin"
            self.log(f"[gate] skip {d.color} target={target_bin} reason={reason} "
                     f"colour_conf={color_conf:.3f} min={color_min:.3f} "
                     f"bin_conf={bin_conf:.3f} min={bin_min:.3f}")
        return bool(ok)

    def _object_time_spent(self, color, xy):
        tr = self._tracked_here([(item["color"], item["xy"], item) for item in self.object_time_tracks], color, xy)
        return 0.0 if tr is None else float(tr[2]["spent_s"])

    def _object_time_allows(self, d, duration, remember=True):
        """Whether another plan fits the object's cumulative hard motion budget.

        Retired: policy.object_time_budget_s is an alias of the decision cap (_resolve_time_alias), so this timer
        never counts (budget None) and the cap is the only mechanism.
        """
        budget = self._budget_timer_s
        if budget is None:
            return True
        budget = float(budget)
        spent = self._object_time_spent(d.color, d.top[:2])
        ok = spent + float(duration) <= budget + 1e-9
        if not ok and remember:
            xy = np.asarray(d.top[:2], float).copy()
            if self._tracked_here(self.budget_abandoned, d.color, xy) is None:
                self.budget_abandoned.append((d.color, xy))
                self.log(f"[time_budget] abandon {d.color}: spent={spent:.1f}s + plan={float(duration):.1f}s "
                         f"> budget={budget:.1f}s")
        return bool(ok)

    def _record_object_time(self, color, xy, duration):
        if self._budget_timer_s is None:
            return
        xy = np.asarray(xy, float).copy()
        tr = self._tracked_here([(item["color"], item["xy"], item) for item in self.object_time_tracks], color, xy)
        if tr is None:
            self.object_time_tracks.append({"color": color, "xy": xy, "spent_s": float(duration)})
        else:
            tr[2]["xy"] = xy
            tr[2]["spent_s"] += float(duration)

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

    def _ordered_cands(self, dets, obs, verify, decision=True):
        """(cands, extra): counter candidates in the order _choose tries them (attempt / failed filter + sort, extra =
        max_attempts_vary tries, already appended to cands). decision=False skips the decision layer (D1/D4 filter and
        ranking): the brain prefetch predicts the next state with it, without touching decision state."""
        cands, extra = [], []
        mav = self._max_attempts_vary()
        pol = self.cfg["policy"]
        for d in dets:
            xy = d.top[:2]
            if any(np.linalg.norm(xy - f) < self.cfg["policy"]["same_object_radius"] for f in self.failed):
                continue
            if pol.get("place_gate", False):
                gs = self._tracked_here(self.gate_skipped, d.color, xy)
                if gs is not None and gs[2] >= self.GATE_MAX_SKIPS:
                    continue
            if self._budget_timer_s is not None and self._tracked_here(
                    self.budget_abandoned, d.color, xy) is not None:
                continue
            n = self._attempt_count(xy, d.color)
            if n >= self._attempt_limit(xy):
                if mav is not None and n < mav:
                    extra.append(d)   # policy.max_attempts_vary: one more varied try, after every other candidate
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

        triage = bool(pol.get("time_triage", False))
        if decision and self._decision_on():
            try:
                cands, extra = self._decision_filter(cands, obs), self._decision_filter(extra, obs)
            except Exception as e:
                self._dec_fail("_decision_filter", e)
        if decision and self._dflag("confidence_order"):
            try:
                cands = self._conf_rank(cands, dets, obs, retry_prefix)
            except Exception as e:
                self._dec_fail("_conf_rank", e)
                cands.sort(key=base)
        elif triage and pol.get("triage_order", "duration") == "priority":
            eps = float(pol.get("triage_tie_m", 0.02))   # candidates with ~equal cheap estimate: cheaper shape first
            cands.sort(key=lambda d: retry_prefix(d) + (round(base(d) / eps), self._priority(d)))
        elif verify:
            cands.sort(key=lambda d: retry_prefix(d) + (base(d),))
        else:
            cands.sort(key=base)
        if decision and self.tier == dec.DEGRADED and not self._dflag("confidence_order"):   # D4: centre bin first
            cands.sort(key=lambda d: not self._is_centre(d))
        if self._isolated_on():   # decision.isolated_first: touching candidates go after the isolated ones (stable)
            try:
                tiered = self.tier in (dec.DEGRADED, dec.PRIOR_ONLY)   # the tier order wins: reorder within its groups
                cands = dec.defer_touching_within(cands, dets, self._dcfg()["isolated_min_m"],
                                                  self._is_centre if tiered else (lambda d: 0))
            except Exception as e:
                self._dec_fail("isolated_first", e)
        if extra:
            extra.sort(key=base)
            cands += extra
        return cands, extra

    def _choose(self, dets, obs, verify):
        """Pick the next counter object and plan it (stock order; #3 time triage and #6 drop slots when on)."""
        pol = self.cfg["policy"]
        t = float(obs.get("t", 0.0))
        try:
            stop = self._decision_stop(dets, t)
        except Exception as e:
            self._dec_fail("_decision_stop", e)
            stop = None
        if stop is not None:
            return stop
        cands, extra = self._ordered_cands(dets, obs, verify)
        triage = bool(pol.get("time_triage", False))
        forced = None   # (det, bin, variant) from the brain (brain.enabled)
        if self.brain is not None:
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
                    forced = (d0, b0, var)
        left = self.time_limit - t - self._time_reserve(corrective=False)
        n_long, shortest = 0, None
        cap_s, over_cap = None, []
        if self._dflag("object_time_cap"):   # D2: per-object cap (remaining - 15 s) / objects left, 15..35 s
            cap_s = dec.object_cap(self.time_limit - t, len(cands) - len(extra) or len(cands), self._dcfg())
        for d in cands:
            if triage:  # a plan that did not fit before cannot fit now (time only runs out; plans start from home)
                rad = pol["same_object_radius"]
                if any(np.linalg.norm(d.top[:2] - p) < rad and dur > left for p, dur in self.too_long):
                    n_long += 1
                    continue
            others = [o.top[:2] for o in dets if o is not d]
            mine = forced is not None and d is forced[0]
            target_bin = forced[1] if mine else self._bin_of(d)
            pfn = self._variant_plan_fn(forced[2]) if mine else None
            if target_bin is None:
                continue  # unmapped / abstain: 0 points on the counter, never a -1 guess
            if not self._place_gate_allows(d, target_bin):
                continue
            if verify:  # #6: explicit target bin + a free, verified drop slot
                slot = self._next_drop_slot(target_bin)
                if slot is None:
                    slot = next(iter(self._overflow_slots(target_bin)), None)
                if slot is None:
                    self.log(f"[classical] no free drop slot in {target_bin}; leave {d.color} on counter")
                    continue
                p = self._plan_pick(d, dict(bin_name=target_bin, obstacles=others, drop_slot=slot), pfn)
            elif self._slots_on():  # grasp.drop_slots standalone: each successive object in a bin -> next slot
                free = [i for i in range(len(self.planner.drop_slots(target_bin)))
                        if i not in self.slot_used[target_bin]]
                slot = free[0] if free else next(iter(self._overflow_slots(target_bin)), None)
                if slot is None:
                    self.log(f"[classical] drop slots of {target_bin} used up; leave {d.color} on counter")
                    continue
                p = self._plan_pick(d, dict(bin_name=target_bin, obstacles=others, drop_slot=slot), pfn)
            elif mine:   # brain override: the classical planner still checks reach
                p = self._plan_pick(d, dict(bin_name=target_bin, obstacles=others), pfn)
            else:
                p = self._plan_pick(d, dict(obstacles=others))
            self.plans.append(p)
            if p.ok and not self._object_time_allows(d, p.duration):
                continue
            if p.ok and any(d is e for e in extra) and not self._has_time(p, t):   # max_attempts_vary: only with time to spare
                self.log(f"[retry_vary] no time for an extra try at {d.color} ({p.duration:.1f} s)")
                continue
            if p.ok and cap_s is not None and p.duration > cap_s:
                over_cap.append(p)   # too slow for its share of the time: try the next object, this one is the fallback
                self.log(f"[time_cap] {d.color} plan {p.duration:.1f} s > cap {cap_s:.1f} s: trying the next candidate")
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
        if over_cap:   # nothing within the cap: the quickest over-cap plan beats leaving points on the counter
            best = min(over_cap, key=lambda q: q.duration)
            if not triage or best.duration <= left:
                self.log(f"[time_cap] no candidate within {cap_s:.1f} s: taking the quickest ({best.duration:.1f} s)")
                return ("plan", best, len(dets))
        if verify and self.verify_relooks_remaining > 0:
            self.verify_relooks_remaining -= 1
            return ("relook", None, len(dets))
        if n_long:
            return ("notime", shortest, len(dets))
        return ("none", None, len(dets))

    # ------------------------------------------------------------------ decision layer D1-D4 (decision.py)
    def _resolve_time_alias(self):
        """policy.object_time_budget_s is a DEPRECATED alias of the decision layer's object time cap (one timer only).

        budget set, cap off  -> the cap mechanism runs with cap_min_s = cap_max_s = budget (self._cap_alias_s);
        both set             -> the cap wins, one warning that the budget is ignored;
        only the cap set     -> unchanged.
        The old cumulative timer (_object_time_allows / _record_object_time) never runs: self._budget_timer_s is None.
        """
        pol = self.cfg["policy"]
        budget = pol.get("object_time_budget_s")
        self._cap_alias_s = None
        self._budget_timer_s = None   # the cumulative per-object timer is retired in favour of the cap
        if budget is None:
            return
        if pol.get("object_time_cap", False):
            self.log(f"[time_cap] policy.object_time_budget_s={float(budget):g} ignored: "
                     "deprecated alias, policy.object_time_cap is set")
        else:
            self._cap_alias_s = float(budget)

    def _dflag(self, name):
        if name == "object_time_cap" and self._cap_alias_s is not None:
            return True
        return bool(self.cfg["policy"].get(name, False))

    def _dec_fail(self, name, exc):
        """F7: a decision helper raised: the caller falls back to stock behaviour; one warning per episode and helper."""
        if name not in self._dec_warned:
            self._dec_warned.add(name)
            self.log(f"[decision] {name} failed ({exc!r}), stock behaviour")

    def _decision_on(self):
        return any(self._dflag(f) for f in ("confidence_order", "object_time_cap", "speed_tiers", "confidence_tiers"))

    def _isolated_on(self):
        return bool((self.cfg.get("decision") or {}).get("isolated_first", False))

    def _dcfg(self):
        d = dict(dec.DEFAULTS)
        d.update(self.cfg.get("decision") or {})
        if self._cap_alias_s is not None:   # deprecated policy.object_time_budget_s: a fixed per-object cap
            d["cap_min_s"] = d["cap_max_s"] = self._cap_alias_s
        return d

    def _centre_name(self):
        names = list(self.scene.bin_names)
        return min(names, key=lambda n: abs(float(self.planner.bin_xy(n)[1])))

    def _is_centre(self, d):
        b = lookup_detection_bin(self.cfg, self.mapping, d)
        return b is not None and b == self._centre_name()

    def _push_frame(self, dets):
        """Temporal-stability history: one frame per look / hold frame (never twice for the same detection list)."""
        if self._stab_last is dets:
            return
        self._stab_last = dets
        self._stab.append([(d.color, np.asarray(d.top[:2], float).copy()) for d in dets])
        del self._stab[:-int(self._dcfg()["stab_frames"])]

    def _stability(self, d):
        """Fraction of the recent frames that saw this object (same colour within same_object_radius); None < 2 frames."""
        if len(self._stab) < 2:
            return None
        rad = float(self.cfg["policy"]["same_object_radius"])
        xy = np.asarray(d.top[:2], float)
        seen = sum(any(c == d.color and np.linalg.norm(xy - p) < rad for c, p in fr) for fr in self._stab)
        return seen / len(self._stab)

    def _hue_margin(self, d, obs):
        """Hue margin (deg) of the detection's blob centre from its colour's HSV range edge; None if unknown."""
        try:
            rgb = obs.get("head_rgb")
            ranges = (getattr(self.perception, "hsv", None) or {}).get(d.color)
            u, v = d.uv
            if rgb is None or not ranges or (u == 0.0 and v == 0.0):
                return None
            import cv2
            h, w = rgb.shape[:2]
            u0, v0 = int(round(u)), int(round(v))
            win = np.ascontiguousarray(rgb[max(v0 - 3, 0):min(v0 + 4, h), max(u0 - 3, 0):min(u0 + 4, w)])
            if not win.size:
                return None
            hue = float(np.median(cv2.cvtColor(win, cv2.COLOR_RGB2HSV)[..., 0]))
            return dec.hue_margin_deg(ranges, hue)
        except Exception:
            return None

    def _conf_info(self, d, obs):
        """D1: p_colour, p_bin, p_grasp, p = product, gate = p_colour * p_bin, est time, target bin / centre flag."""
        dc = self._dcfg()
        b = lookup_detection_bin(self.cfg, self.mapping, d)
        if b is None:
            return {"p": 0.0, "gate": 0.0, "t": 99.0, "bin": None, "centre": False, "pc": 0.0, "pb": 0.0, "pg": 0.0}
        bxy = np.asarray(self.planner.bin_xy(b), float)
        centre = b == self._centre_name()
        pc = dec.p_colour(d.n_px, self._hue_margin(d, obs), self._stability(d), dc)
        pb = dec.p_bin(centre, self.C, dc)
        cov = (d.n_depth / d.n_px) if d.n_px else 0.0
        pg = dec.p_grasp(getattr(d, "shape", None), d.top[:2], float(bxy[1]), cov, dc)
        return {"p": pc * pb * pg, "gate": pc * pb, "t": dec.est_time(d.top[:2], bxy, dc), "bin": b, "centre": centre,
                "pc": pc, "pb": pb, "pg": pg}

    def _conf_rank(self, cands, dets, obs, retry_prefix):
        """policy.confidence_order: 2*p/est_time, shape ties cube < cylinder < prism, never the hardest first. Called on
        every look, so the order is re-ranked after each placement (fresh detections, fresh stability)."""
        self._push_frame(dets)
        items = []
        for d in cands:
            i = self._conf_info(d, obs)
            items.append({"d": d, "i": i, "p": i["p"], "t": i["t"], "shape": getattr(d, "shape", None),
                          "centre_first": self.tier == dec.DEGRADED and i["centre"]})
        out = dec.rank(items, self._dcfg())
        out = sorted(out, key=lambda it: retry_prefix(it["d"]))   # #6 retry prefix first (stable; () when #6 is off)
        if out:
            self.log("[confidence_order] " + " > ".join(
                f"{it['d'].color}/{it['shape']} p={it['p']:.2f} (c={it['i']['pc']:.2f} b={it['i']['pb']:.2f} "
                f"g={it['i']['pg']:.2f}) t={it['t']:.0f}s" for it in out))
        return [it["d"] for it in out]

    def _observe_outcome(self, dets):
        """D3: judge the last pick at the next look: the object (same colour, within retry_assoc_radius of the grasp
        point) is still on the counter = failure, gone = clean success."""
        po, self._pend_out = self._pend_out, None
        if po is None:
            return
        rad = float(self.cfg["policy"].get("retry_assoc_radius", 0.06))
        near = [d for d in dets if d.color == po["color"] and np.linalg.norm(np.asarray(d.top[:2]) - po["xy"]) <= 0.03]
        if near and self.attempt_tracks:   # retry-vary on: the survivor must belong to this attempt's track
            same = [tr for tr in self.attempt_tracks if tr["color"] == po["color"]
                    and np.linalg.norm(tr["xy"] - po["xy"]) < rad]
            tr = min(same, key=lambda t_: np.linalg.norm(t_["xy"] - po["xy"])) if same else None
            near = [d for d in near if tr is not None and np.linalg.norm(np.asarray(d.top[:2]) - tr["xy"]) <= 0.03]
        still = bool(near)
        self.tiers.observe(clean=not still)
        self.log(f"[speed_tier] {po['color']} {'still on the counter: failure' if still else 'gone: clean'} -> "
                 f"scale {self.tiers.scale():.2f}{' centre bin only' if self.tiers.centre_only() else ''}")

    def _decision_stop(self, dets, t):
        """Reasons not to start any new pick right now (("none", ...) result) or None."""
        if self._dflag("speed_tiers"):
            self._observe_outcome(dets)
        if self._dflag("object_time_cap") and dec.no_new_object(t, self.time_limit, self._dcfg()):
            if "t20" not in self._cap_logged:
                self._cap_logged.add("t20")
                self.log(f"[time_cap] t={t:.1f}: inside the last {self._dcfg()['no_new_s']:.0f} s, no new object")
            return ("none", None, len(dets))
        return None

    def _decision_filter(self, cands, obs):
        """Tier restrictions on the candidates: DEGRADED p_colour*p_bin >= 0.85; PRIOR-ONLY or a failed pick (speed tiers):
        centre-bin objects only. Unmapped objects are left to the stock skip."""
        dc = self._dcfg()
        prior = self._dflag("confidence_tiers") and self.tier == dec.PRIOR_ONLY
        centre_only = prior or (self._dflag("speed_tiers") and self.tiers.centre_only())
        degraded = self._dflag("confidence_tiers") and self.tier == dec.DEGRADED
        if not (centre_only or degraded):
            return cands
        infos = [self._conf_info(d, obs) for d in cands]
        # F1: the centre-only rule holds only while a centre-bin candidate exists (else every candidate passes: no deadlock)
        has_centre = any(i["bin"] is not None and i["centre"] for i in infos)
        centre_only = centre_only and has_centre
        out = []
        for d, i in zip(cands, infos):
            if i["bin"] is None:
                out.append(d)
            elif centre_only and not i["centre"]:
                continue
            elif prior and not has_centre and not i["centre"] and i["bin"] in self._prior_side:
                continue   # PRIOR-ONLY side fallback: at most one attempt per side bin
            elif degraded and i["gate"] < float(dc["degraded_min_p"]):
                self.log(f"[tier] DEGRADED: skip {d.color} (p={i['gate']:.2f} < {dc['degraded_min_p']})")
                continue
            else:
                out.append(d)
        return out

    def _speed_scale(self):
        """Arm speed scale for the next plan: speed tiers (0.7 / 0.85-0.9 / 0.6), DEGRADED and PRIOR-ONLY 0.6. Never > 1."""
        s = 1.0
        if self._dflag("speed_tiers"):
            s = self.tiers.scale()
        if self._dflag("confidence_tiers") and self.tier in (dec.DEGRADED, dec.PRIOR_ONLY):
            s = min(s, float(self._dcfg()["degraded_scale"]))
        return min(s, 1.0)

    @contextlib.contextmanager
    def _decision_motion(self):
        """Yields the planner to plan with: with a speed scale / first-object clearance, a shallow copy whose cfg has
        its own motion / grasp dicts (arm_max_vel / arm_max_acc x scale, grasp.approach_extra_m + clearance), so the
        shared cfg is never mutated while a background plan runs. The stock planner when nothing changes."""
        try:
            s = self._speed_scale()
            extra = self.tiers.clearance() if self._dflag("speed_tiers") else 0.0
            if s >= 1.0 and not extra:
                pl = self.planner
            else:
                pl = copy.copy(self.planner)
                cfg = dict(self.planner.cfg)
                mo, g = dict(cfg["motion"]), dict(cfg["grasp"])
                mo["arm_max_vel"], mo["arm_max_acc"] = mo["arm_max_vel"] * s, mo["arm_max_acc"] * s
                g["approach_extra_m"] = g.get("approach_extra_m", 0.0) + extra
                cfg["motion"], cfg["grasp"] = mo, g
                pl.cfg = cfg
        except Exception as e:
            self._dec_fail("_decision_motion", e)
            pl = self.planner
        yield pl

    # ---- D4: first seconds
    def _pitch_cal_value(self):
        try:
            pc = getattr(self.perception, "pitch_cal", None)
            if pc is None:
                return None
            return float(getattr(pc, "confidence", pc))
        except Exception:
            return None

    def _bin_agree(self):
        """Depth bin-wall agreement with the prior (perception.self_calibration); None when that is not enabled."""
        sc = self.cfg["perception"].get("self_calibration") or {}
        if not sc.get("enabled", False):
            return None
        cal = self.calibration
        if cal is None:
            return 0.0
        if cal.success:
            off = float(np.linalg.norm(cal.xy_offset)) / max(float(sc.get("max_offset_m", 0.03)), 1e-6)
            return float(cal.confidence) * (1.0 - 0.5 * min(1.0, off))
        return 0.5 * float(cal.confidence)

    def _calib_frame(self, obs, c):
        try:
            depth = obs.get("head_depth")
            if depth is not None and np.ndim(depth) == 2:
                lo, hi = self.cfg["perception"]["depth_range_mm"]
                c["depth"].append(float(((depth >= lo) & (depth <= hi)).mean()))
            dets = self.perception.detect(obs.get("head_rgb"), depth, **self._fk_kw(obs))
            self.last_detections = dets
            self._push_frame(dets)
        except Exception as e:   # the hold must never kill the control loop
            self.log(f"[tier] hold frame failed ({e!r})")

    def _calib_step(self, obs, t):
        """policy.confidence_tiers: hold still `hold_s` seconds, take `hold_frames` detect frames, then C and the tier.
        Returns a hold action while measuring, None once the tier is set (the normal look follows)."""
        d = self._dcfg()
        c = self._calib
        if c is None:
            c = self._calib = {"t0": t, "n": 0, "depth": [], "job": None}
        n_total, hold_s = max(1, int(d["hold_frames"])), float(d["hold_s"])
        el = t - c["t0"]
        if c["n"] < n_total and el >= c["n"] * hold_s / n_total:
            if self.threaded:
                j = c["job"]
                if j is None:
                    j = c["job"] = {"done": False}

                    def run(job=j, o=obs):
                        _lower_thread_priority()
                        self._calib_frame(o, c)
                        job["done"] = True
                    threading.Thread(target=run, daemon=True).start()
                if not j["done"]:
                    return self._hold()
                c["job"] = None
            else:
                self._calib_frame(obs, c)
            c["n"] += 1
        if el < hold_s:
            return self._hold()
        frac = float(np.mean(c["depth"])) if c["depth"] else None
        pitch, agree = self._pitch_cal_value(), self._bin_agree()
        self.C = dec.calib_confidence(pitch, agree, frac, d)
        self.tier = dec.tier_of(self.C, d)
        fmt = lambda x: "n/a" if x is None else f"{x:.2f}"
        msg = (f"[tier] {self.tier} C={fmt(self.C)} pitch={fmt(pitch)} bin={fmt(agree)} depth={fmt(frac)} "
               f"frames={c['n']} hold={el:.1f}s")
        self.events.append((round(t, 2), msg))
        self.log(msg)
        return None

    # ---- D2: T-8 s
    def _hand_xyz(self, plan):
        return self.kin.grasp_pose(plan.side, self.arm_cmd)[0]

    def _over_target_bin(self, plan):
        if plan.target_bin not in self.scene.bin_names:
            return False
        half = np.asarray(self.scene.cfg["bins"]["outer_size"], float) / 2 - 0.02
        return bool(np.all(np.abs(np.asarray(self._hand_xyz(plan), float)[:2] - self.planner.bin_xy(plan.target_bin)) <= half))

    def _endgame(self, t):
        """policy.object_time_cap, T-8 s with a plan running: carrying over the target bin -> open the hand where it is,
        then home; anywhere else before the release -> abandon the plan and go home (hand opens on the way).
        Once per plan; returns the first action of the replacement motion, or None (nothing to change)."""
        ex = self._exec
        if ex["t8"] or not self._dflag("object_time_cap") or not dec.endgame(t, self.time_limit, self._dcfg()):
            return None
        ex["t8"] = True
        plan = ex["plan"]
        tick = ex["n"] - len(self.queue_a)   # lift_on_stall slices queue_a, so cuts are already in
        phase = dec.phase_at(plan.phases, tick)
        over = phase == "carry" and self._over_target_bin(plan)
        what = dec.endgame_action(phase, over)
        self.log(f"[time_cap] t={t:.1f} T-{self._dcfg()['release_s']:.0f}: plan phase={phase} over_bin={over} -> {what}")
        if what is None:
            return None
        sl = ArmKinematics.sl(plan.side)
        home = self.arm_cmd.copy()
        home[sl] = self.home[sl]
        if what == "release":
            rel = self.hand_cmd.copy()
            rel[sl] = self.post[self.cfg["grasp"].get("release_posture", "preshape")][sl]
            h1 = ramp(self.hand_cmd, rel, self.cfg["motion"]["hand_max_vel"], 1.0 / HZ)
            h1 = np.vstack([h1, np.repeat(rel[None], int(round(float(self.cfg["motion"]["open_hold_s"]) * HZ)), 0)])
            a1 = np.repeat(self.arm_cmd[None], len(h1), 0)
            a2, h2 = self.planner.go_to(self.arm_cmd, rel, home, self.post["open"])
            a, h = np.vstack([a1, a2]), np.vstack([h1, h2])
        else:
            a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, home, self.post["open"])
        self._stall = self._gdiag = None
        self._enqueue(a, h)
        return self._pop()

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

    def _first_object_planner(self, pl):
        """policy.first_object_speed_scale: every plan made before the first pick starts (so the first started pick)
        gets its arm phases x scale. A shallow copy carries the scale, so the shared planner is never mutated while a
        background look plans. Flag off, or a pick already started: the planner passed in, unchanged."""
        scale = self.cfg["policy"].get("first_object_speed_scale")
        if scale is None or self.n_started != 0:
            return pl
        pl = copy.copy(pl)
        pl.object_speed_scale = float(scale)
        return pl

    def _plan_pick(self, d, kw, plan_fn=None):
        """planner.plan, or with policy.retry_vary_grasp and an earlier miss on this object: the first feasible grasp
        variant whose (point, finger heading, posture) differs from every earlier try; the stock plan only if it
        differs too. Flag off: exactly planner.plan(d, arm, hand, **kw)."""
        def plan(relocate=None):
            args = dict(kw)
            if relocate is not None and plan_fn is None:
                args["relocate"] = relocate
            with self._decision_motion() as pl:   # D3/D4 speed scale + first-object clearance (stock planner when off)
                pl = self._first_object_planner(pl)
                if plan_fn is None:
                    return pl.plan(d, self.arm_cmd, self.hand_cmd, **args)
                if pl is not self.planner:
                    args["planner"] = pl
                return plan_fn(d, self.arm_cmd, self.hand_cmd, **args)

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
            parsed = resolve(str(self._cap_text((info or {}).get("instruction") or "")), fallback={}) or {}
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
                cands, _ = self._ordered_cands(ds, {}, ver, decision=False)
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

    def _variant_plan_fn(self, variant):
        """The brain's retry grasp variant as a _plan_pick plan_fn, or None (same / None = the normal plan).
        closed_tight = that posture; demo = the by-shape demo posture. Plans on a planner COPY with its own grasp dict
        (like _decision_motion), so the shared cfg is never mutated while a background plan runs."""
        if variant not in ("closed_tight", "demo"):
            return None

        def fn(d, arm, hand, planner=None, **kw):
            pl = copy.copy(planner or self.planner)
            cfg = dict(pl.cfg)
            g = dict(cfg["grasp"])
            if variant == "closed_tight":
                g["close_posture"] = "closed_tight"
            else:
                g["close_posture_by_shape"] = True
            cfg["grasp"] = g
            pl.cfg = cfg
            return pl.plan(d, arm, hand, **kw)
        return fn

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
        self._record_object_time(plan.color, xy, plan.duration)
        self.n_started += 1
        if self.brain is not None:
            self._brain_prefetch(plan)
        if self._dflag("speed_tiers") and not plan.corrective:   # D3: the next look judges this pick
            self._pend_out = {"color": plan.color, "xy": np.asarray(xy, float).copy()}
            self.log(f"[speed_tier] t={t:.1f} {plan.color}: scale {self._speed_scale():.2f} approach +"
                     f"{100 * self.tiers.clearance():.1f} cm (clean={self.tiers.n_clean})")
        if self._dflag("confidence_tiers") and self.tier == dec.PRIOR_ONLY and plan.target_bin in self.scene.bin_names \
                and plan.target_bin != self._centre_name():
            self._prior_side.add(plan.target_bin)
            self.log(f"[tier] PRIOR-ONLY: side-bin attempt {plan.target_bin} ({plan.color}), max 1 per side bin")
        if self._dflag("object_time_cap") and plan.phases and plan.side in ("left", "right"):
            self._exec = {"plan": plan, "n": len(plan.arm), "cut": 0, "t8": False}
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
        if (self.cfg["policy"].get("grasp_diag_log", False) or self.held_detector.mode == "finger_sum") \
                and plan.side in ("left", "right"):
            ph = {name: (s_, e) for name, s_, e in (plan.phases or [])}
            if "close" in ph and "lift" in ph:
                check = self.held_detector.evaluation_tick(*ph["close"], HZ)
                self._gdiag = {"plan": plan, "n": len(plan.arm), "close_end": check,
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

    def _held_abort_due(self, g, hq, hc):
        """grasp.held_detector=finger_sum only: the close sample is clearly empty. Unsure (no bias reference, missing
        or non-finite joint data) is never empty, so the placement runs as before."""
        det = self.held_detector
        if not det.ready or hq.shape != (14,) or hc.shape != (14,):
            return False
        if not (np.all(np.isfinite(hq)) and np.all(np.isfinite(hc))):
            return False
        return not det.held(g["plan"].side, hc, hq)[0]

    def _abort_empty_grasp(self, g):
        """Skip the carry of an empty grasp: open and return home. The pick counts as a miss (attempt counted at
        start, vary_hist held=False), so the next look sees the object still on the table and retry-vary runs.
        Standalone-slot bookkeeping is left to _settle_slot, which frees the slot when the object remains."""
        plan = g["plan"]
        self._gdiag = None
        self._stall = self._exec = None      # per-plan watchers: their indices no longer match the queue
        if self._vary_on() and self.vary_hist:   # the entry _start appended for this plan
            self.vary_hist[-1]["held"] = False
        self.log(f"[held_abort] t={g['t']:.1f} {plan.color} {plan.side} target={plan.target_bin} "
                 f"close sample empty: carry skipped, object remains")
        home = self.arm_cmd.copy()
        home[ArmKinematics.sl(plan.side)] = self.home[ArmKinematics.sl(plan.side)]
        a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, home, self.post["open"])
        self._enqueue(a, h)

    def _joint_offset_diag(self, obs, t):
        """debug.log_joint_offset (passive): cmd-minus-measured over the first second of steady holds, from the policy's
        own previous action only. A trajectory chunk is not steady state, so it is skipped; a hold is compared on the
        next act() call, after the runner has applied it. Never changes a command."""
        if not (self.cfg.get("debug") or {}).get("log_joint_offset", False):
            return
        if not self._last_policy_cmd_steady:
            return
        if self._joint_offset_t0 is None:  # the startup home trajectory must not consume the window
            self._joint_offset_t0 = t
        if t - self._joint_offset_t0 > 1.05:
            return
        aq, hq = obs.get("arm_q"), obs.get("hand_q")
        if aq is None or hq is None or self._last_policy_arm_cmd is None or self._last_policy_hand_cmd is None:
            return
        da = self._last_policy_arm_cmd - np.asarray(aq, float)
        dh = self._last_policy_hand_cmd - np.asarray(hq, float)
        self.log(f"[joint_offset] t={t:.2f} cmd-minus-measured arm={np.round(da, 4).tolist()} "
                 f"hand={np.round(dh, 4).tolist()}")

    def _remember_policy_action(self, a, h, done, steady=False):
        """Build the action dict; keep its last command for debug.log_joint_offset only (flag off: nothing retained)."""
        out = {"arm_q_target": np.asarray(a, float).copy(), "hand_q_target": np.asarray(h, float).copy(),
               "done": bool(done)}
        if (self.cfg.get("debug") or {}).get("log_joint_offset", False):
            self._last_policy_arm_cmd = out["arm_q_target"][-1].copy()
            self._last_policy_hand_cmd = out["hand_q_target"][-1].copy()
            self._last_policy_cmd_steady = bool(steady)
        return out

    def _grasp_diag(self, obs):
        """policy.grasp_diag_log (passive, never changes a command): once per pick, measured hand_q vs commanded
        hand_cmd vs the `closed` posture at the detector's close-check tick and at the end of the lift."""
        g = self._gdiag
        q = obs.get("hand_q")
        left = 0 if self.queue_a is None else len(self.queue_a)
        done = g["n"] - left
        if q is not None:
            cmd = self._held_command()
            for key, at in (("close", g["close_end"]), ("lift", g["lift_end"])):
                if key not in g["samples"] and done >= at:
                    g["samples"][key] = (done, np.asarray(q, float).copy(), np.asarray(cmd, float).copy())
                    if key == "close" and self._held_abort_due(g, np.asarray(q, float), np.asarray(cmd, float)):
                        self._abort_empty_grasp(g)   # queue replaced; _act pops it on this same tick
                        return
        need = ("close", "lift") if self.held_detector.mode == "finger_sum" else ("lift",)
        if any(key not in g["samples"] for key in need) and left > 0:
            return
        self._gdiag = None
        plan = g["plan"]
        sl = ArmKinematics.sl(plan.side)
        closed = self.post[(plan.posture[0] if plan.posture else self.cfg["grasp"].get("close_posture", "closed"))][sl]
        parts = []
        held = None
        close_held = None
        for key in ("close", "lift"):
            if key not in g["samples"]:
                parts.append(f"{key}=n/a")
                continue
            tick, hq, hc = g["samples"][key]
            held, scores = self.held_detector.held(plan.side, hc, hq)
            if key == "close":
                close_held = held
            if self.held_detector.mode == "stall":
                # Preserve the legacy diagnostic bytes when the new flag is off.
                parts.append(f"{key}@{tick}: track_err_max={scores['track_err_max']:.3f} stall_vs_closed="
                             f"{np.round(closed - hq[sl], 3).tolist()} cmd_vs_closed={np.round(closed - hc[sl], 3).tolist()}")
            else:
                parts.append(f"{key}@{tick}: finger_sum={scores['finger_sum']:.3f} "
                             f"index_sum={scores['index']:.3f} middle_sum={scores['middle']:.3f} "
                             f"threshold={self.held_detector.threshold_rad:.3f} sample_held={'yes' if held else 'no'} "
                             f"stall_vs_closed="
                             f"{np.round(closed - hq[sl], 3).tolist()} cmd_vs_closed={np.round(closed - hc[sl], 3).tolist()}")
        if self.held_detector.mode == "finger_sum" and close_held is not None:
            held = close_held       # headline is the post-wait close sample; lift stays separately labelled above
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
            if self._exec is not None:
                self._exec["cut"] += cut
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
        self._popped = (a, h) if self._shell else None
        return self._remember_policy_action(a, h, False)

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
        return self._remember_policy_action(np.repeat(self.arm_cmd[None], k, 0),
                                            np.repeat(self.hand_cmd[None], k, 0), done, steady=True)

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[classical t={t:6.2f}] {msg}")
