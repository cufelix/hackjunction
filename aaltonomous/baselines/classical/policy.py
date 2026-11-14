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
import dataclasses
from dataclasses import dataclass
import os
import re
import sys
import threading
import time
from types import MappingProxyType

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.tasks import FIXED_MAPPING
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from . import decision as dec
from . import landing
from .bin_vote import vote_bins
from .config import equalize_index_middle, hand_postures, home_arm, load_config, soft_detector_active, soft_hand_active, soft_overclose
from .held import HeldDetector
from .fk_xcheck import FkHandXCheck, FkXCheckResult
from .kinematics import ArmKinematics
from .perception import Perception
from .planner import MotionOutputLimiter, Plan, Planner, lookup_bin, lookup_detection_bin, ramp, staggered_release_ramp
from .safety_shell import reserve_pair, reserve_plan
from .brain import BrainLayer, build_world
from .brain.find import FindClient, extract_queries, match_found

HZ = 30.0
_HOLE = re.compile(r"\bholes?\b")   # policy.insert_guard: the rung-3 sentence names a hole AND the lid
_LID = re.compile(r"\blids?\b")
_TRUNC_CUT = ".;!?\n"   # instruction.max_chars: sentence / clause boundaries, then whitespace
_WS = " \t\r\n"


@dataclass(frozen=True)
class _BinVoteResult:
    """Thread result with no mutable arrays/dicts; only the matching episode may publish it."""
    generation: int
    shift: tuple
    extra: tuple
    confidence: tuple
    messages: tuple

    @classmethod
    def freeze(cls, generation, raw, messages):
        shift, extra, confidence = raw
        vecs = tuple((str(k), tuple(float(x) for x in np.asarray(v, float))) for k, v in sorted(shift.items()))
        return cls(int(generation), vecs,
                   tuple((str(k), float(v)) for k, v in sorted(extra.items())),
                   tuple((str(k), float(v)) for k, v in sorted(confidence.items())),
                   tuple(str(m) for m in messages))

    def mutable_copies(self):
        return ({k: np.asarray(v, float) for k, v in self.shift}, dict(self.extra), dict(self.confidence))


@dataclass(frozen=True)
class _LookResult:
    generation: int
    outcome: tuple
    state: tuple
    perception_state: tuple
    counter_z_runtime: object
    logs: tuple


def _lower_thread_priority(nice=10):
    """Linux: raise the nice value of the calling thread only (setpriority on its native thread id)."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except (AttributeError, OSError):
        pass


class ClassicalPolicy:
    _LOOK_COMMIT_FIELDS = (
        "last_detections", "last_bin_detections", "scheduled_corrections", "plans", "retry_pass_used",
        "attempts", "attempt_tracks", "retry_allow", "slot_pending", "slot_used", "_bin_misses", "gate_skipped",
        "budget_abandoned", "failed", "too_long", "verification_pending", "last_verification",
        "verify_relooks_remaining", "retry_colors", "retry_targets", "used_drop_slots", "reserved_drop_slots",
        "_skill_pending", "_pend_out", "_stab", "_stab_last", "_dec_warned", "_cap_logged", "_prior_side",
        "_brain_reg", "_brain_n", "_last_pick", "_brain_looks", "_brain_finds", "tiers", "tier", "C",
    )
    _LOOK_PERCEPTION_FIELDS = ("_ah_est", "_ah_tries", "hsv", "_colour_kmeans_done", "_colour_kmeans_tries",
                               "colour_kmeans_info", "color_gate_degraded", "fk_mask_px")
    _LOOK_DEEP_FIELDS = {"attempt_tracks", "gate_skipped", "slot_used", "verification_pending", "last_verification",
                         "used_drop_slots", "reserved_drop_slots", "_skill_pending", "_pend_out", "tiers"}
    @staticmethod
    def _safety_hand_limits(spec, safety):
        """SafetyFilter._shrink for the hand, kept read-only and Pinocchio-free."""
        margin = np.minimum(safety.hand_pos_margin, 0.25 * (spec.hand_upper - spec.hand_lower))
        return spec.hand_lower + margin, spec.hand_upper - margin

    def __init__(self, config=None, scene=None, calibration=None, safety: SafetyConfig | None = None,
                 require_confirm=False, threaded=False, time_limit=120.0, log=print, overrides=None,
                 profile=None, overlay=None):
        self.cfg = config if isinstance(config, dict) else load_config(
            config, overrides, profile=profile, overlay=overlay)
        self.held_detector = HeldDetector(self.cfg["grasp"].get("held_detector", "stall"),
                                          self.cfg["grasp"].get("held_threshold_rad", 0.045))
        self.scene = scene if isinstance(scene, Scene) else Scene.load(scene)
        self.cal = calibration if isinstance(calibration, Calibration) else Calibration.load(calibration)
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)
        self._mirror_im = bool(self.cfg["hand"].get("mirror_index_middle", False))
        self.home = home_arm(self.cfg, self.spec)
        self._soft_det = soft_detector_active(self.cfg)    # held.soft_detector on a soft hand (default OFF)
        if self._soft_det:
            h, g = self.cfg["held"], self.cfg["grasp"]
            self.held_detector.configure_soft(
                self.post["preshape"], h["soft_margin_rad"], h["soft_lift_margin_rad"], h["soft_prior_ratio"],
                float(g["soft_lag_s"]) + float(g["soft_settle_s"]) + float(h["soft_read_after_s"]))
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
        # calib.hand_icp_primary: self-contained fitter + generation-guarded executor. Construction (URDF/FK asset load)
        # is outside the episode watchdog. The policy only polls it and emits holds until the result is ready.
        self.primary_calibration = None
        ccfg = self.cfg.get("calib") or {}
        self.primary_calibration_enabled = bool(ccfg.get("hand_icp_primary", False))
        if ccfg.get("hand_icp_primary", False) or ccfg.get("counter_after_hand", False):
            try:
                from .calibration_fit import AsyncCalibrationPhase, CounterHeightTracker, PrimaryCalibrationFitter
                primary = PrimaryCalibrationFitter(ccfg.get("primary_fit") or {},
                                                   crosscheck=ccfg.get("refit_per_look", False))
                if ccfg.get("hand_icp_primary", False):
                    self.primary_calibration = AsyncCalibrationPhase(primary, ccfg.get("primary_fit") or {})
                if ccfg.get("counter_after_hand", False):
                    self.perception.counter_tracker = CounterHeightTracker(
                        primary, ccfg.get("counter_fit") or {}, self.scene.nominal_counter_z)
            except Exception as exc:
                self.primary_calibration = None
                (log or print)(f"primary calibration disabled: {type(exc).__name__}: {exc}")
        self.planner = Planner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec)
        mo = self.cfg["motion"]
        self.output_limiter = (MotionOutputLimiter(
            mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"], mo["hand_max_acc"], 1.0 / HZ)
            if mo.get("output_caps") is not None else None)
        # calib.fk_xcheck (FLAG, default OFF, log-only): RGB head-pose cross-check vs the Dex3 hand FK. Its own
        # ArmKinematics: the worker thread must never touch the planner's pinocchio data.
        self.fk_xcheck_enabled = bool((self.cfg.get("calib") or {}).get("fk_xcheck", False))
        if self.fk_xcheck_enabled:
            self._fk_xcheck_kin = ArmKinematics(self.cfg, self.spec)
            self._fk_xcheck = FkHandXCheck(self.cfg["calib"]["fk_xcheck_config"], self.perception)
        self.perception.bin_shift_src = self.planner   # perception.color_gate.live_bins reads planner.bin_shift (bin.vote)
        self.perception.geometry_runtime = bool((self.cfg.get("bin") or {}).get("geometry_runtime"))   # perception follows the shift without bin.vote/live_bins gates
        self.perception.bin_vote_on = bool(((self.cfg.get("bin") or {}).get("vote") or {}).get("enabled"))   # bin footprints follow the shift
        self.require_confirm = require_confirm
        self._jobs = {}   # bounded background work; reset swaps the dict so stale jobs cannot publish into a new episode
        self._shell = bool(self.cfg["policy"].get("safety_shell", False))
        bv = (self.cfg.get("bin") or {}).get("vote") or {}
        self._async_bin_vote = bool(bv.get("enabled", False) and bv.get("async_extract", False))
        # A detector deadline can only be non-blocking when the look runs away from act().
        if self._async_bin_vote or self.cfg["perception"].get("look_timeout_s") is not None:
            threaded = True
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
        fm = (self.cfg.get("policy") or {}).get("fixed_mapping")   # FLAG: explicit finals scene metadata
        self.fixed_mapping = dict(fm) if fm else dict(FIXED_MAPPING)   # used only by an identified sort_fixed reset
        self.skill_memory = None   # skills.memory (FLAG, default OFF): learned order of the retry-vary arms
        if (self.cfg.get("skills") or {}).get("memory", False):
            try:
                from .skill_memory import SkillMemory
                self.skill_memory = SkillMemory(path=self.cfg["skills"].get("memory_path"), log=self.log)
            except Exception as exc:  # learning is optional and must never prevent policy construction
                self.log(f"[skills] unavailable ({type(exc).__name__}: {exc}); using stock retry order")
        self.reset()

    # ------------------------------------------------------------------ API
    @staticmethod
    def _perception_snapshot(perception):
        """Stable episode state captured before a deadline-controlled detector starts."""
        state = {}
        for name in ("cal", "pitch_calibration", "calibration", "_ah_tries", "_colour_kmeans_done",
                     "_colour_kmeans_tries", "colour_kmeans_info", "color_gate_degraded"):
            if hasattr(perception, name):
                state[name] = copy.deepcopy(getattr(perception, name))
        for name in ("pitch_T", "xy_offset", "_ah_est", "hsv"):
            if hasattr(perception, name):
                state[name] = copy.deepcopy(getattr(perception, name))
        scene = getattr(perception, "scene", None)
        if scene is not None:
            state["counter_z_runtime"] = scene.counter_z_runtime
        return state

    def _replace_stale_perception(self, state=None):
        """Give an unfinished detector sole ownership of its mutable perception state."""
        scene = copy.copy(self.scene)
        self.scene = scene
        self.planner.scene = scene
        self.perception = Perception(self.cfg, scene, self.cal, log=self.log)
        state = state or {}
        for name, value in state.items():
            if name != "counter_z_runtime" and hasattr(self.perception, name):
                setattr(self.perception, name, value)
        if "counter_z_runtime" in state:
            scene.counter_z_runtime = state["counter_z_runtime"]
        self.perception.bin_shift_src = self.planner
        self.perception.geometry_runtime = bool((self.cfg.get("bin") or {}).get("geometry_runtime"))
        self.perception.bin_vote_on = bool(((self.cfg.get("bin") or {}).get("vote") or {}).get("enabled"))
        if self.fk_xcheck_enabled:
            self._fk_xcheck.perception = self.perception

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
        """Instruction text -> mapping for `scene.resolve_bin`, or a truthy fail-closed abstain.

        Off unless `instruction.resolve` is true in config.yaml, so the reference measurement is
        unchanged by default. NEVER raises: reset() is under a 1.0 s watchdog (safety.py:69) and an
        exception here would end the episode as a BAD_END, which is worse than any wrong mapping.
        The resolver is pure stdlib and measured at ~0.13 ms per sentence.
        """
        strict = bool((self.cfg.get("instruction") or {}).get("strict", False))
        try:                                        # the WHOLE body, not just the resolve call
            instruction_cfg = self.cfg.get("instruction") or {}
            strict = instruction_cfg.get("strict", False)
            if not instruction_cfg.get("resolve", False):
                return None
            text = self._cap_text((info or {}).get("instruction") or "")
            if not str(text).strip():               # fixed mapping is reserved for an identified sort_fixed task
                return {"default": None}
            from .instruction import resolve
            if (self.cfg.get("instruction") or {}).get("venue_colour_aliases", False):   # T10: venue colour words
                from .normalize import venue_colour_aliases
                text = venue_colour_aliases(text)
            if (self.cfg.get("instruction") or {}).get("multilingual", False):   # FI/CZ + 10 languages (Timi)
                from .normalize import canonicalize_multilingual
                text = canonicalize_multilingual(text)
            if (self.cfg.get("instruction") or {}).get("normalize", False):   # typo/synonym normaliser (Felix)
                from .normalize import normalize
                text = normalize(text)
            if strict:                              # never guess: abstain (Felix strict)
                from .normalize import strict_mapping
                if (self.cfg.get("instruction") or {}).get("structured", False):
                    from .normalize import structured_mapping
                    parsed = structured_mapping(text)
                    if parsed is not None:          # recognised structures fail closed on an incomplete parse
                        return parsed
                return strict_mapping(text, resolve(text, fallback={}))
            return resolve(text, fallback={}) or {"default": None}
        except Exception as e:                      # a parse bug must not cost the whole episode
            self.log(f"instruction resolve failed ({e}), abstaining")
            return {"default": None}

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

    def _log_fixed_conflict(self, info, used):
        """Logging only: sort_fixed ignores the instruction sentence. Say so when the sentence names a different mapping.
        NEVER raises and never changes `used` (the mapping the episode runs with)."""
        try:
            parsed = self._instruction_mapping(info) or {}
            said = {k: v for k, v in parsed.items() if k != "default" and v}
            if said and any(used.get(k) != v for k, v in said.items()):
                self.log(f"[fixed_mapping] conflict: instruction says {said}, using venue {used}")
        except Exception as e:
            self.log(f"[fixed_mapping] conflict check failed ({e})")

    def _episode_mapping(self, info):
        """Return (mapping, insert_guard_mapping) for one reset without touching motion/perception state."""
        # Episode mapping precedence is explicit and task-scoped. A reset-supplied mapping is authoritative; otherwise
        # only sort_fixed may consume the configured venue scene mapping. Language blank/errors stay on the counter.
        given = info.get("mapping")
        if not given and info.get("task") == "sort_fixed":
            given = dict(self.fixed_mapping)
            self._log_fixed_conflict(info, given)
        guarded = None if given else self._insert_guard_mapping(info)
        resolved = None if given or guarded else self._instruction_mapping(info)
        legacy = dict(FIXED_MAPPING) if not (self.cfg.get("instruction") or {}).get("resolve", False) else {"default": None}
        return (dict(given) if given else guarded or resolved or legacy), guarded

    def reset(self, info=None):
        generation_workers = bool((self.cfg.get("policy") or {}).get("generation_workers", False))
        if generation_workers:
            # Invalidate first.  No lock/join: reset is constant-time with respect
            # to stale work, which can only publish to its detached job object.
            self._look_generation = getattr(self, "_look_generation", 0) + 1
            current = getattr(self, "_job", None)
            if current is not None and current.get("result") is None and "deadline_lock" in current:
                with current["deadline_lock"]:
                    current["cancelled"] = True
                self._stale_detect_jobs = [*getattr(self, "_stale_detect_jobs", []), current]
            self._job = None
        if isinstance(info, str):
            info = {"instruction": info}
        info = info or {}
        if self.skill_memory is not None:
            self.skill_memory.reseed(info.get("seed", 0))   # seed: reproducible skill sampling
        self.mapping, guarded = self._episode_mapping(info)
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
        self._novel_brain_candidates = []
        self._novel_brain_bin = None
        if self.brain is not None:
            self.brain.reset()
        self.time_limit = info.get("time_limit", self.time_limit)
        self._resolve_time_alias()
        calib = self.cfg.get("calib") or {}
        preflight = bool(self.cfg["perception"].get("online_pitch_cal", False)
                         or calib.get("plane", {}).get("apply_height", False)
                         or calib.get("hand_icp_primary", False))
        self.state = "calibrate" if preflight else "init"
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
        stale_detect_jobs = [job for job in getattr(self, "_stale_detect_jobs", []) if job["result"] is None]
        active_detect_job = getattr(self, "_job", None)
        if active_detect_job is not None and active_detect_job.get("result") is None \
                and "deadline_lock" in active_detect_job:
            with active_detect_job["deadline_lock"]:
                active_detect_job["cancelled"] = True
            stale_detect_jobs.append(active_detect_job)
        if not generation_workers:
            self._job = None
        # Preserve unfinished calls across episode reset. The next detector uses an isolated copy until these exit.
        self._stale_detect_jobs = stale_detect_jobs
        owner = next((job for job in stale_detect_jobs if job.get("detector") is self.perception), None)
        if owner is not None:
            # reset_calibration mutates perception and scene state. Do not touch either while a cancelled native call
            # still owns them; move the new episode to fresh state instead.
            self._replace_stale_perception(owner.get("perception_state"))
        self._clear_looks = 0
        self._clear_since = None
        self._target_cooldowns = []     # [{color, xy, until}], planner.per_target_guard only
        self._latest_t = 0.0            # newest episode clock seen by act(); planning workers use it for cooldown start
        # A stale generation owns its detached snapshot and may finish in the
        # background.  The new episode gets a fresh worker slot immediately.
        self._look_thread = None
        look_lock = self.__dict__.setdefault("_look_lock", threading.Lock())
        with look_lock:
            self._look_generation = getattr(self, "_look_generation", 0) + 1
            self._bv_job = None
        self._jobs = {}
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
        self._held_cal_phase = None
        self._held_cal_done = not (self.held_detector.mode == "finger_sum" and
                                   bool((self.cfg["grasp"].get("held_calibration") or {}).get("enabled", False)))
        self._softcal = None        # held.soft_detector: the episode-start empty close being watched
        self._softcal_done = False
        self._bin_misses = []       # (colour, xy) already logged as [bin_lookup] miss this episode
        self._stall = None          # motion.lift_on_stall: the close hold being watched
        self.retry_pass_used = False
        self.vary_hist = []         # policy.retry_vary_grasp: {color, xy, yaw, posture, side, held} per started pick
        self._summary_logged = False
        self._emitted = None        # safety.nan_guard: (arm, hand) last row of the last action returned this episode
        self._encoder_hold = None    # M2: last complete finite encoder sample; never synthesize an unobserved joint
        self._encoder_latched = False
        if self._shell:
            # A calibration hold must start from this episode's encoders, never a target cached by the shell in the
            # previous episode.  _act() installs the measured pose before the first _hold().
            self._shell_last = None
        if self.output_limiter is not None:
            self.output_limiter.reset()
        self._partial_relooks = 0   # partial_ok_near_hand: bounded retract/re-observe attempts
        self._final_relook_used = False
        self._relook_fresh = 0
        self._relook_last_t = None
        self._relook_reason = None
        # Bump the episode token under the lock: a stale worker from the previous episode then fails its token check
        # and can never write into this one. act() never waits on the worker (nominal extrinsics until it lands).
        lock = self.__dict__.setdefault("_pitch_lock", threading.Lock())
        with lock:
            self._pitch_token = getattr(self, "_pitch_token", 0) + 1
            self.perception.reset_calibration()
            self.pitch_calibration = None
            self._pitch_calibration_done = False   # = the worker was started this episode
        self._pitch_thread = None
        self._calib_phase_t0 = None
        self._height_phase_t0 = None
        self.planner.bin_shift, self.planner.drop_extra, self.planner.bin_conf = {}, {}, {}   # bin.vote: nominal until voted
        self._bv_frames, self._bv_done = [], False
        self.calibration = None
        self._calibration_done = False
        self._empty_recovery = None
        self.hand_calibration = None    # HeadCalibResult of this episode (perception.hand_calibration)
        self._hc_done = self.hand_calib is None
        self._hc_tries = 0
        self._hc_prev_q = None
        self._hc_job = None
        self._hc_still = 0
        self._primary_done = not self.primary_calibration_enabled
        self._primary_result = None
        self._primary_counter_tries = 0
        self._last_motion_time = None
        if self.primary_calibration is not None:
            self.primary_calibration.reset()
        self.last_bin_detections = []
        self.used_drop_slots = {name: set() for name in self.scene.bin_names}
        self.reserved_drop_slots = {name: set() for name in self.scene.bin_names}
        self.verification_pending = None
        self.last_verification = None
        self.retry_colors = set()
        self.retry_targets = []
        self.verify_relooks_remaining = 0
        self.scheduled_corrections = []
        self._landing = None        # policy.verify_landing: the release whose outcome the next look classifies
        self._occ_prev = None       # verify_landing: {bin: occupancy} of the previous look
        self.landing_events = []    # verify_landing: (kind, colour, bin) per classified release
        self.landing_regrants = 0
        self.landing_refunds = 0
        self.bounce_count = {}
        self._landing_excl_logged = set()
        self._skill_pending = None  # skills.memory: the started pick whose outcome is not settled yet
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
        self._rel = None            # grasp.release_confirm: the plan's release watcher
        self._rel_fail = False      # release_confirm: the hand never read empty -> stop picking
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
        token = self._pitch_token

        def work():
            started = time.perf_counter()
            try:
                out = self.perception.compute_pitch(np.array(rgb, copy=True), np.array(depth, copy=True))
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

    def _pitch_timeout(self, t):
        """Freeze nominal extrinsics and invalidate a late worker before any motion is allowed."""
        with self._pitch_lock:
            if self.pitch_calibration is not None:
                return
            self._pitch_token += 1
            out = self.perception.make_pitch_failure("pre-motion calibration timeout",
                                                     max(0.0, t - self._calib_phase_t0) * 1000.0)
            self.perception.commit_pitch(out)
            self.pitch_calibration = out
        self._log_pitch(out)

    def _preflight_calibration(self, obs, t):
        """Hold measured joints while pitch then N-gated table height calibrate; never performs a fit in act()."""
        if self._calib_phase_t0 is None:
            self._calib_phase_t0 = t
            self._note(t, "pre-motion calibration: holding measured joints")

        # handcal: the hand/forearm ICP fit (calib.hand_icp_primary) selects the full extrinsic first, off-thread;
        # the pitch-only fit is then skipped and the height below is measured in the corrected frame.
        if self._primary_calibration_step(obs, t):
            return self._hold()
        calib = self.cfg.get("calib") or {}
        pitch_on = (bool(self.cfg["perception"].get("online_pitch_cal", False))
                    and not calib.get("hand_icp_primary", False))
        if pitch_on and self.pitch_calibration is None:
            if (not self._pitch_calibration_done
                    and self.perception.pitch_frame_valid(obs.get("head_rgb"), obs.get("head_depth"))):
                self._start_pitch_worker(obs.get("head_rgb"), obs.get("head_depth"))
            if t - self._calib_phase_t0 >= self.PITCH_WAIT_S:
                self._pitch_timeout(t)
            else:
                return self._hold()

        plane = (self.cfg.get("calib") or {}).get("plane") or {}
        if plane.get("apply_height", False) and not self.perception._ah_done and not calib.get("counter_after_hand", False):
            if self._height_phase_t0 is None:
                self._height_phase_t0 = t
            timeout = float(plane.get("timeout_s", 3.0))
            if t - self._height_phase_t0 >= timeout:
                self.perception._ah_done, self.perception._ah_committed = True, None
                self.planner.commit_counter_height(None)
                self._jobs.pop("counter_height", None)
                self._note(t, f"runtime height fallback: pre-motion timeout ({timeout:.1f} s); nominal keepout")
            else:
                job = self._jobs.get("counter_height")
                if job is None:
                    depth = obs.get("head_depth")
                    frame = self.perception.frame_transform()
                    r = self._bg("counter_height", self._measure_counter_height, depth, frame)
                else:
                    r = self._bg("counter_height", self._measure_counter_height, None, None)
                if r is None:
                    return self._hold()
                if r[0] == "err":
                    from .perception import CounterHeightSample
                    sample = CounterHeightSample(np.nan, np.nan, 0, f"{type(r[1]).__name__}: {r[1]}")
                else:
                    sample = r[1]
                done, height = self.perception.accept_counter_height_sample(sample, commit_scene=False)
                if done:
                    self.planner.commit_counter_height(height)
                    state = "nominal fallback" if height is None else f"{height:.4f} m"
                    self._note(t, f"runtime height committed once: {state}; planner keepout={self.planner.counter_keepout_z:.4f} m")
                else:
                    return self._hold()

        self.state = "init"
        self._note(t, f"pre-motion calibration complete: transform v{self.perception.frame_transform().version}")
        return self._hold()

    def _measure_counter_height(self, depth, frame):
        """Worker-only copy + fit; copying a full depth frame never runs in act()."""
        depth = None if depth is None else np.array(depth, copy=True)
        return self.perception.measure_counter_height(depth, frame)

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

    def _calibration_frame(self, obs):
        """Metadata consumed only by the opt-in stale-frame guards."""
        aq = np.asarray(obs.get("arm_q"), float)
        hq = np.asarray(obs.get("hand_q"), float)
        stamps = [float(obs.get("t", 0.0))]
        stamps.extend(float(obs[k]) for k in ("head_timestamp", "joint_timestamp") if obs.get(k) is not None)
        return {"timestamp": min(stamps),
                "joints": np.r_[aq, hq], "last_motion_time": self._last_motion_time}

    def _primary_calibration_step(self, obs, t):
        """Return True while the pre-motion phase owns the state machine and act() must emit only a hold."""
        if self._primary_done:
            return False
        if self.primary_calibration is None:
            prior = self.scene.nominal_counter_z + float(
                self.cfg["calib"]["primary_fit"].get("fallback_counter_dz_m", 0.15))
            self.perception.set_camera_pose(self.perception.current_T())
            self.scene.counter_z_runtime = prior
            self._primary_done = True
            self._note(t, f"[primary_calib] unavailable: nominal extrinsic, counter_z prior={prior:.3f}")
            return False
        if self._primary_result is None:
            try:
                update = self.primary_calibration.advance(
                    self.perception.cal, self.perception.cam, obs.get("head_depth"), obs.get("arm_q"),
                    obs.get("hand_q"), self._calibration_frame(obs)["timestamp"], self.scene.nominal_counter_z,
                    self._last_motion_time)
            except Exception as exc:
                update = None
                self._note(t, f"[primary_calib] worker adapter failed: {type(exc).__name__}: {exc}")
            if update is None or (not update.hold and update.result is None):
                # The worker itself failed/timed out. Preserve the explicit final fallback.
                self.perception.set_camera_pose(self.perception.current_T())
                self.scene.counter_z_runtime = self.scene.nominal_counter_z + float(
                    self.cfg["calib"]["primary_fit"].get("fallback_counter_dz_m", 0.15))
                self._primary_done = True
                self._note(t, f"[primary_calib] nominal fallback ({'unknown' if update is None else update.reason})")
                return False
            if update.result is None:
                return True
            result = update.result
            self._primary_result = result
            self.perception.set_camera_pose(result.T)
            for name, passed, detail in result.checks:
                self._note(t, f"[primary_calib/check] {name}: {'PASS' if passed else 'FAIL'} {detail}")
            self._note(t, f"[primary_calib] {result.summary()}")
            if result.source == "nominal":
                self.scene.counter_z_runtime = result.counter_z
                self._primary_done = True
                return False
        if self.cfg["calib"].get("counter_after_hand", False):
            self._primary_counter_tries += 1
            update = self.perception.update_counter_height(obs.get("head_depth"), self._calibration_frame(obs))
            if update is not None and update.committed:
                self._primary_done = True
                return False
            limit = int((self.cfg["calib"].get("counter_fit") or {}).get("max_tries", 12))
            if self._primary_counter_tries >= limit:
                prior = self.scene.nominal_counter_z + float(
                    self.cfg["calib"]["primary_fit"].get("fallback_counter_dz_m", 0.15))
                self.scene.counter_z_runtime = prior
                self._primary_done = True
                self._note(t, f"[counter_after_hand] fallback after {limit} tries: counter_z prior={prior:.3f}")
                return False
            return True
        self._primary_done = True
        return False

    def confirm(self):
        self._decision = "go"

    def skip(self):
        self._decision = "skip"

    def stop(self):
        self._decision = "stop"

    def act(self, obs):
        res = self._act_entry(obs)
        if self._mirror_im:   # hand.mirror_index_middle (R42a): last stop before the runner, every return path
            h = np.asarray(res["hand_q_target"], float)
            res = dict(res, hand_q_target=np.stack([self._im_emit(r) for r in h]) if h.ndim == 2 else self._im_emit(h))
        return res

    def _im_emit(self, r):
        """Posture-aware index/middle repair at emission: a hand flexed past the open/closed midpoint keeps the more
        closed value (closed_demo reaches its full -1.41, no under-grip on soft real fingers); otherwise the more open."""
        f = np.r_[3:7, 10:14]
        mid = (np.abs(self.post["open"][f]).mean() + np.abs(self.post["closed"][f]).mean()) / 2
        return equalize_index_middle(r, closed=bool(np.abs(r[f]).mean() > mid))

    def _act_entry(self, obs):
        ng = bool((self.cfg.get("safety") or {}).get("nan_guard", False))
        hold = None
        if ng:
            obs, hold = self._nan_guard(obs)
        if hold is None:
            if self.fk_xcheck_enabled:   # before both paths: the safety shell's act() is the same entry point
                self._poll_fk_xcheck(obs, float(obs.get("t", 0.0)))
            res = self._act_shell(obs) if self._shell else self._act(obs)
            if not getattr(self, "_last_action_steady", True):   # handcal stale-frame guard: last commanded motion
                self._last_motion_time = float(obs.get("t", 0.0))
        else:
            res = hold
        if self.output_limiter is not None:
            res = self._limit_motion_output(res, obs)
        if ng:
            a, h = np.asarray(res["arm_q_target"], float), np.asarray(res["hand_q_target"], float)
            if len(a) and len(h):
                self._emitted = (a[-1].copy(), h[-1].copy())
        return res

    def _limit_motion_output(self, action, obs):
        """M4 final boundary: cap every route, including holds, home, endgame and empty-grasp queue replacement."""
        a, h = np.asarray(action["arm_q_target"], float), np.asarray(action["hand_q_target"], float)

        def start(key, target):
            try:
                q = np.asarray(obs.get(key), float).reshape(-1)
            except (AttributeError, TypeError, ValueError):
                q = np.asarray(target, float)
            return q if q.shape == (14,) and np.all(np.isfinite(q)) else np.asarray(target, float)

        a, h = self.output_limiter.limit(
            a, h, start("arm_q", a[0]), start("hand_q", h[0]), float(obs.get("t", 0.0)))
        out = {**action, "arm_q_target": a, "hand_q_target": h}
        if len(a):
            self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
            if self._shell:
                self._shell_last = (self.arm_cmd.copy(), self.hand_cmd.copy())
            if (self.cfg.get("debug") or {}).get("log_joint_offset", False):
                self._last_policy_arm_cmd, self._last_policy_hand_cmd = self.arm_cmd.copy(), self.hand_cmd.copy()
        return out

    # ------------------------------------------------------------------ safety.nan_guard
    _nan_t = -1e9

    def _nan_log(self, msg):
        now = time.monotonic()
        if now - self._nan_t >= 1.0:   # rate limit: at most one line per second
            self._nan_t = now
            self.log(f"[nan_guard] {msg}")

    @staticmethod
    def _encoder_vector(q):
        """Return an exact 14-vector or None.  Never fills an unobserved joint."""
        try:
            v = np.asarray(q, float).reshape(-1)
        except (TypeError, ValueError):
            return None
        return v if v.shape == (14,) else None

    def _complete_encoder(self, obs):
        a, h = self._encoder_vector(obs.get("arm_q")), self._encoder_vector(obs.get("hand_q"))
        if a is None or h is None or not (np.all(np.isfinite(a)) and np.all(np.isfinite(h))):
            return None
        return a.copy(), h.copy()

    def _nan_guard(self, obs):
        """Non-finite arm_q / hand_q: latch the planner until a complete finite encoder frame. Hold the last emitted
        or complete measured pose; before either exists return no action rather than filling an unobserved joint.
        Non-finite head_depth pixels -> 0 = missing depth: perception keeps only
        depth_range_mm (config.yaml, 150..2000 mm) so 0 is invalid there (perception.py:369, :336). All-finite obs: returned
        as is. Returns (obs, hold_action or None)."""
        if not isinstance(obs, dict):
            return obs, None
        complete = self._complete_encoder(obs)
        if complete is not None:
            self._encoder_hold = complete
            self._encoder_latched = False
        bad_q = complete is None
        for k in ("arm_q", "hand_q"):
            v = obs.get(k)
            if v is not None:
                try:
                    bad_q = bad_q or not bool(np.all(np.isfinite(np.asarray(v, float))))
                except (TypeError, ValueError):
                    bad_q = True
        dep = obs.get("head_depth")
        if isinstance(dep, np.ndarray) and dep.dtype.kind == "f" and not np.isfinite(dep).all():
            obs = {**obs, "head_depth": np.where(np.isfinite(dep), dep, dep.dtype.type(0))}
            self._nan_log(f"head_depth: {int((~np.isfinite(dep)).sum())} non-finite px -> 0 (missing depth)")
        if not bad_q:
            return obs, None
        self._encoder_latched = True
        if self._emitted is not None:
            a, h = self._emitted
            what = "last emitted action"
        elif self._encoder_hold is not None:
            a, h = self._encoder_hold
            what = "last complete encoder sample"
        else:
            # The runner treats a zero-row chunk as no new action and holds the
            # SafetyFilter command established from the measured reset pose.
            # This neither invents an absolute joint target nor records a
            # nonfinite safety clip while encoder trust is still unavailable.
            self._nan_log("non-finite arm_q/hand_q: no-motion latch, returning no action "
                          "until the first complete encoder sample")
            empty = np.empty((0, 14), float)
            return obs, {"arm_q_target": empty.copy(), "hand_q_target": empty.copy(), "done": False}
        self._nan_log(f"non-finite arm_q/hand_q: no-motion latch, holding {what}")
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
        elif self._encoder_hold is not None:
            a, h = self._encoder_hold
        else:
            # The shell can be enabled without nan_guard, so capture a complete
            # finite first observation here as an exact (non-invented) hold.
            complete = self._complete_encoder(obs) if isinstance(obs, dict) else None
            if complete is not None:
                a, h = complete
            else:
                # With any unobserved joint, return no target and let the runner
                # retain its measured reset command without a safety clip.
                empty = np.empty((0, 14), float)
                return {"arm_q_target": empty.copy(), "hand_q_target": empty.copy(), "done": False}
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

    def _held_calibration_tick(self, obs, t):
        """M3 empty-close calibration, entirely in the normal non-blocking action state machine."""
        if self._held_cal_done or self.held_detector.mode != "finger_sum":
            return None
        p = self.cfg["grasp"].get("held_calibration") or {}
        if not p.get("enabled", False) or self.state not in ("look", "held_calibration"):
            return None
        if self._held_cal_phase is None:
            # settle sets state=look as soon as it enqueues the hold.  Do not
            # replace that queue: calibration starts only after every settle
            # target has actually been returned to the runner.
            if self.queue_a is not None and len(self.queue_a):
                return None
            self.state = "held_calibration"
            self._held_cal_phase = "sample"
            self.held_detector.require_empty_close_calibration(
                t, steady_samples=p.get("steady_samples", 4),
                max_velocity_rad_s=p.get("max_velocity_rad_s", 0.08),
                hold_after_settle_s=p.get("hold_after_settle_s", 0.4),
                timeout_s=p.get("timeout_s", 5.0))
            posture = self.cfg["grasp"].get("close_posture", "closed")
            target = self.post[posture]
            h = ramp(self.hand_cmd, target, self.cfg["motion"]["hand_max_vel"], 1.0 / HZ)
            n = max(1, int(np.ceil(float(p.get("hold_after_settle_s", 0.4)) * HZ)))
            h = np.vstack([h, np.repeat(target[None], n, 0)])
            self._enqueue(np.repeat(self.arm_cmd[None], len(h), 0), h)
            self._note(t, "held calibration: empty close at home")
            return self._pop()
        if self._held_cal_phase == "sample":
            status = self.held_detector.sample_empty_close(t, self._held_command(), obs.get("hand_q"))
            if self.queue_a is not None and len(self.queue_a):
                return None
            if status == "pending":
                return self._hold()
            target = self.post["open"]
            h = ramp(self.hand_cmd, target, self.cfg["motion"]["hand_max_vel"], 1.0 / HZ)
            n = max(1, int(np.ceil(float(self.cfg["motion"].get("open_hold_s", 0.4)) * HZ)))
            h = np.vstack([h, np.repeat(target[None], n, 0)])
            self._enqueue(np.repeat(self.arm_cmd[None], len(h), 0), h)
            self._held_cal_phase = "reopen"
            state = "accepted" if status == "ready" else "invalid; early abort disabled"
            self._note(t, f"held calibration {state}: {self.held_detector.calibration_reason}")
            return self._pop()
        if self._held_cal_phase == "reopen" and not (self.queue_a is not None and len(self.queue_a)):
            self._held_cal_phase = None
            self._held_cal_done = True
            self.state = "look"
        return None

    def _act(self, obs):
        first = self.arm_cmd is None
        if first:  # first call: start from the measured pose
            # Clamp into the URDF limits: real Dex3 fingers at rest read slightly past 0 (2026-11-11), and holding
            # that measured pose would be clipped every tick -> safety abort before the first move.
            sp = self.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        t = float(obs.get("t", 0.0))
        self._latest_t = t
        if first:
            self._ep_t0 = t
        held_cal_on = bool((self.cfg["grasp"].get("held_calibration") or {}).get("enabled", False))
        if self.held_detector.mode == "finger_sum" and not held_cal_on:
            if first:
                # The first command is synthesized from this same measurement,
                # so it carries no tracking-bias information. Start the window
                # now and sample subsequent actual command/measurement pairs.
                self.held_detector.start(t)
            elif obs.get("hand_q") is not None:
                self.held_detector.observe(t, self._held_command(), obs["hand_q"])
        self._joint_offset_diag(obs, t)
        if self.state == "calibrate":
            return self._preflight_calibration(obs, t)
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
        if self._softcal is not None:
            self._soft_baseline_step(obs)
        if self._empty_recovery is not None:
            return self._empty_recovery_step(t)
        if self._stall is not None:
            self._lift_on_stall(obs)
        held_cal_action = self._held_calibration_tick(obs, t)
        if held_cal_action is not None:
            return held_cal_action
        if self._rel is not None:
            r = self._release_confirm(obs, t)
            if r is not None:
                return r
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
        if self.state == "relook_settle":
            n = int(round(self.cfg["motion"]["settle_s"] * HZ))
            self._enqueue(np.repeat(self.arm_cmd[None], n, 0), np.repeat(self.hand_cmd[None], n, 0))
            self.state = "relook_fresh"
            return self._pop()
        if self.state == "relook_fresh":
            # A request may repeat an observation while a command chunk is consumed. Count only increasing episode time.
            if self._relook_last_t is None or t > self._relook_last_t:
                self._relook_fresh += 1
                self._relook_last_t = t
            need = int(self.cfg["perception"].get("relook_fresh_frames", 2))
            if self._relook_fresh < need:
                return self._hold()
            self._note(t, f"{self._relook_reason}: {self._relook_fresh} fresh frames; re-detect")
            self.state = "look"
        if self.state == "init":
            if (self.cfg.get("calib") or {}).get("plane", {}).get("apply_height", False):
                if "startup_route" not in self._jobs:
                    r = self._bg("startup_route", self.planner.safe_home_route,
                                 self.arm_cmd.copy(), self.hand_cmd.copy(), ("left", "right"), None)
                else:
                    r = self._bg("startup_route", self.planner.safe_home_route, None, None)
                if r is None:
                    return self._hold()
                if r[0] == "err":
                    self.state = "staging_hold"
                    self._note(t, f"height-safe staging unproved: {type(r[1]).__name__}: {r[1]}; holding")
                    return self._hold()
                route = r[1]
                if not route.ok:
                    self.state = "staging_hold"
                    self._note(t, f"height-safe staging unproved: {route.reason}; holding")
                    return self._hold()
                a, h = route.arm, route.hand
            else:
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
            if self._soft_det and not self._softcal_done:
                self._soft_baseline_start(t)   # one empty close per hand at home, queued: act() never blocks
            return self._pop()
        if self.state == "look" and self._await_pitch_cal(t):
            # H3/H4: no bin-vote frame (or any look) is captured while the first pitch commit is pending.
            return self._hold()
        vote_cfg = (self.cfg.get("bin") or {}).get("vote", {})
        if self.state == "look" and not self._bv_done and vote_cfg.get("enabled"):
            # bin.vote: first n_frames looks at home, before any pick; then freeze the median, never again
            if self._async_bin_vote:
                if self._async_bin_vote_step(obs, t):
                    return self._hold()
            else:
                frame = {"head_rgb": obs.get("head_rgb"), "head_depth": obs.get("head_depth"),
                         "transform": self.perception.frame_transform()}
                if vote_cfg.get("frame_transform_snapshot", False):
                    frame["transform_snapshot"] = self.perception.projection_snapshot()
                self._bv_frames.append(frame)
                if len(self._bv_frames) < int(vote_cfg["n_frames"]):
                    return self._hold()
                self._bv_done = True
                shift, extra, conf = vote_bins(self.perception, self.planner, self._bv_frames, self.cfg,
                                                log=lambda m: self._note(t, m))
                self.planner.bin_shift, self.planner.drop_extra, self.planner.bin_conf = shift, extra, conf
                self._bv_frames = []
        if self.state == "look":
            if self._dflag("confidence_tiers") and self.tier is None:
                try:
                    r = self._calib_step(obs, t)   # first seconds: hold still, measure C, pick the tier
                except Exception as e:
                    self._dec_fail("_calib_step", e)
                    self.tier, r = dec.CONFIDENT, None
                if r is not None:
                    return r
            robust_look = (self.cfg["policy"].get("done_confirm", False)
                           or self.cfg["perception"].get("look_timeout_s") is not None
                           or (self.cfg.get("planner") or {}).get("per_target_guard", False))
            if robust_look and t >= self.time_limit:
                self._note(t, "episode time limit reached")
                self.state = "done"
                return self._hold(done=True)
            cap = self.cfg["policy"].get("max_total_attempts")
            if cap is not None and self.n_started >= int(cap):   # knocked objects reappear as 'new': hard stop
                self._note(t, f"attempt cap reached ({self.n_started} >= max_total_attempts {int(cap)}): done")
                self.state = "done"
                return self._hold(done=True)
            if self.threaded:
                self._stale_detect_jobs = [job for job in self._stale_detect_jobs if job["result"] is None]
                max_inflight = int(self.cfg["perception"].get("detect_max_inflight", 2))
                if self.cfg["perception"].get("look_timeout_s") is not None \
                        and len(self._stale_detect_jobs) >= max_inflight:
                    return self._hold()
                if self._job is None:
                    if self.cfg["policy"].get("generation_workers", False):
                        # A detached stale worker owns its snapshot.  Do not start
                        # another planner user until it exits; act() only polls.
                        # With look_timeout_s a wedged thread is bounded by detect_max_inflight (checked above).
                        if self._look_thread is not None and self._look_thread.is_alive() \
                                and self.cfg["perception"].get("look_timeout_s") is None:
                            return self._hold()
                        generation = self._look_generation
                        worker, logs = self._look_snapshot()
                        job_obs = {k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else copy.deepcopy(v))
                                   for k, v in obs.items()}
                        self._job = {"generation": generation, "obs": job_obs, "result": None}
                        if self.cfg["perception"].get("look_timeout_s") is not None:
                            # The snapshot isolates perception/scene, so a timed-out call needs no perception replacement.
                            self._job.update(started_wall=time.monotonic(), phase="detect", cancelled=False,
                                             deadline_lock=threading.Lock())
                        self._look_thread = threading.Thread(target=self._look_job,
                                                             args=(self._job, worker, logs), daemon=True)
                        self._look_thread.start()
                    else:
                        self._job = {"obs": obs, "result": None, "detector": self.perception}
                        if self.cfg["perception"].get("look_timeout_s") is not None:
                            self._job.update(started_wall=time.monotonic(), phase="detect", cancelled=False,
                                             deadline_lock=threading.Lock(),
                                             perception_state=self._perception_snapshot(self.perception))
                        threading.Thread(target=self._look_job, args=(self._job,), daemon=True).start()
                if self._job["result"] is None:
                    deadline = self.cfg["perception"].get("look_timeout_s")
                    job = self._job
                    timed_out = False
                    if deadline is not None and time.monotonic() - job["started_wall"] >= float(deadline):
                        with job["deadline_lock"]:
                            if job["phase"] == "detect":
                                job["cancelled"] = timed_out = True
                    if timed_out:
                        self._job = None
                        self._stale_detect_jobs.append(job)
                        if job.get("detector") is self.perception:
                            # Retire perception + scene together. The retry then updates the same fresh scene used by
                            # planning, while the unkillable native call keeps exclusive ownership of the old pair.
                            self._replace_stale_perception(job.get("perception_state"))
                        res = ("error", f"detector deadline exceeded ({float(deadline):g} s)")
                        return self._after_look(res, t)
                    return self._hold()
                res, self._job = self._job["result"], None
                if self.cfg["policy"].get("generation_workers", False):
                    if not isinstance(res, _LookResult) or res.generation != self._look_generation:
                        return self._hold()
                    self._commit_look_result(res)
                    res = res.outcome
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
        if self.state == "executing" and self._rel_fail:   # release_confirm: hand not empty after the re-open
            self.state = "done"
            self._note(t, "release not confirmed: stop picking")
            return self._hold(done=True)
        if self.state == "executing":  # queue drained -> back home: look again after settling
            self.state = "settle"
            return self._act(obs)
        if self.state == "staging_hold":
            return self._hold()
        if self.state == "empty_hold":
            return self._hold()
        return self._hold(done=True)

    PITCH_WAIT_S = 1.0   # perception.online_pitch_cal: pre-motion hold waits at most this long (observation time)

    def _await_pitch_cal(self, t):
        """online_pitch_cal ON: the first look must not run on the nominal extrinsics before the fit committed (it would
        see 0 detections and end the episode). Timeout is measured from calibration-phase entry, never episode reset.
        This poll never joins a worker, so act() cannot block. Flag off: always False."""
        if ((self.cfg.get("calib") or {}).get("hand_icp_primary", False)   # handcal: the hand fit owns the pose
                or not self.cfg["perception"].get("online_pitch_cal", False) or self.pitch_calibration is not None):
            return False
        entered = self._calib_phase_t0 if self._calib_phase_t0 is not None else t
        if t - entered >= self.PITCH_WAIT_S:
            return False
        return True

    # ------------------------------------------------------------------ look + plan
    def _look_snapshot(self):
        """Fast copy-on-launch transaction for one look/plan worker."""
        worker = copy.copy(self)
        for name in self._LOOK_COMMIT_FIELDS:
            if hasattr(self, name):
                value = getattr(self, name)
                if name in self._LOOK_DEEP_FIELDS:
                    value = copy.deepcopy(value)
                elif isinstance(value, (list, dict, set)):
                    value = value.copy()
                setattr(worker, name, value)
        worker.planner = copy.copy(self.planner)
        worker.planner.__dict__ = self.planner.__dict__.copy()
        # Pinocchio Model is read-only after construction, but Data and the
        # temporary IK hints/offsets are mutated by FK/IK.  Give every worker
        # its own transaction-local state so stale and current generations may
        # finish concurrently after a reset.
        worker.kin = copy.copy(self.kin)
        worker.kin.data = worker.kin.model.createData()
        worker.kin.offset = {side: np.asarray(value, float).copy() for side, value in self.kin.offset.items()}
        worker.kin.hint = np.asarray(self.kin.hint, float).copy()
        worker.planner.kin = worker.kin
        for name in ("mapping", "bin_shift", "drop_extra", "bin_conf"):
            setattr(worker.planner, name, copy.deepcopy(getattr(self.planner, name)))
        # Scene configuration is read-only; only counter_z_runtime changes per
        # look, so a shallow object copy isolates that write without copying the
        # full scene/config synchronously in act().
        worker.scene = copy.copy(self.scene)
        worker.planner.scene = worker.scene
        worker.perception = copy.copy(self.perception)
        worker.perception.__dict__ = self.perception.__dict__.copy()
        worker.perception._arm_surface = None  # lazily rebuilt inside this worker; its Pinocchio Data is mutable
        for name in self._LOOK_PERCEPTION_FIELDS:
            if hasattr(self.perception, name):
                setattr(worker.perception, name, copy.deepcopy(getattr(self.perception, name)))
        worker.perception.scene = worker.scene
        worker.perception.bin_shift_src = worker.planner
        logs = []
        worker.log = logs.append
        worker.perception.log = logs.append
        return worker, logs

    def _commit_look_result(self, result):
        for name, value in result.state:
            setattr(self, name, value)
        for name, value in result.perception_state:
            setattr(self.perception, name, value)
        self.scene.counter_z_runtime = result.counter_z_runtime
        for message in result.logs:
            self.log(message)

    def _look_job(self, job, worker=None, logs=None):
        _lower_thread_priority()  # planning must not starve act() (the control loop's thread)
        if job.get("kind") == "bin_vote":
            self._bin_vote_look_job(job)
            return
        try:
            if worker is None:
                if self.cfg["perception"].get("look_timeout_s") is None:
                    job["result"] = self._look(job["obs"])
                else:
                    job["result"] = self._look(job["obs"], deadline_job=job, detector=job["detector"])
                return
            outcome = worker._look(job["obs"], deadline_job=job) if "deadline_lock" in job else worker._look(job["obs"])
            state = tuple((name, getattr(worker, name)) for name in self._LOOK_COMMIT_FIELDS if hasattr(worker, name))
            perception_state = tuple((name, getattr(worker.perception, name))
                                     for name in self._LOOK_PERCEPTION_FIELDS if hasattr(worker.perception, name))
            if job["generation"] == self._look_generation:
                job["result"] = _LookResult(job["generation"], outcome, state, perception_state,
                                             worker.scene.counter_z_runtime, tuple(logs))
        except Exception as e:  # never kill the control loop
            if worker is None:
                job["result"] = ("error", repr(e))
            else:
                if job["generation"] == self._look_generation:
                    job["result"] = _LookResult(job["generation"], ("error", repr(e)), (), (),
                                                 worker.scene.counter_z_runtime, tuple(logs))

    def _copy_bin_vote_frame(self, obs):
        """Copy camera buffers before returning to the runner; the worker never observes reused frame storage."""
        frame = {}
        for key in ("head_rgb", "head_depth"):
            value = obs.get(key)
            if value is None:
                frame[key] = None
            else:
                value = np.asarray(value).copy()
                value.setflags(write=False)
                frame[key] = value
        frame["transform"] = self.perception.frame_transform()
        if self.cfg["bin"]["vote"].get("frame_transform_snapshot", False):
            frame["transform_snapshot"] = self.perception.projection_snapshot()
        return MappingProxyType(frame)

    def _async_bin_vote_step(self, obs, t):
        """Collect/start/poll only. Return True while act() must hold; extraction runs in the look worker."""
        job = self._bv_job
        if job is not None:
            if not job["done"].is_set():
                return True
            result = job["result"]
            if result is None or result.generation != self._look_generation:
                return True
            self._bv_job = None
            self._bv_done = True
            self.planner.bin_shift, self.planner.drop_extra, self.planner.bin_conf = result.mutable_copies()
            for message in result.messages:
                self._note(t, message)
            return False

        self._bv_frames.append(self._copy_bin_vote_frame(obs))
        if len(self._bv_frames) < int(self.cfg["bin"]["vote"]["n_frames"]):
            return True
        job = {"kind": "bin_vote", "generation": self._look_generation,
               "frames": tuple(self._bv_frames), "result": None, "done": threading.Event()}
        self._bv_frames = []
        with self._look_lock:
            if job["generation"] != self._look_generation:
                return True
            self._bv_job = job
        threading.Thread(target=self._look_job, args=(job,), name="bin-vote", daemon=True).start()
        return True

    def _bin_vote_look_job(self, job):
        messages = []
        try:
            raw = vote_bins(self.perception, self.planner, job["frames"], self.cfg, log=messages.append)
            result = _BinVoteResult.freeze(job["generation"], raw, messages)
        except Exception as exc:  # vote_bins is defensive; keep the worker boundary defensive too
            names = self.scene.bin_names
            v = self.cfg["bin"]["vote"]
            raw = ({}, {n: float(v["low_conf_extra_m"]) for n in names}, {n: 0.5 for n in names})
            messages.append(f"[bin.vote] worker failed ({type(exc).__name__}: {exc}); prior positions")
            result = _BinVoteResult.freeze(job["generation"], raw, messages)
        try:
            with self._look_lock:
                if job["generation"] == self._look_generation and self._bv_job is job:
                    job["result"] = result
        finally:
            job["done"].set()

    def _detect_kw(self):
        """Optional context for a settled policy look."""
        out = {}
        if self.cfg["perception"].get("colour_kmeans", False):
            out["arm_clear"] = bool(np.abs(self.arm_cmd - self.home).max() < 0.05)
        if self.cfg["perception"].get("still_vote", False):
            out["still_frame"] = True
        return out

    def _fk_kw(self, obs):
        """Measured pose for an enabled colour-gate, depth-recovery or novel-object FK mask."""
        need_fk = ((self.cfg["perception"].get("color_gate") or {}).get("fk_mask", False) or
                   self.cfg["perception"].get("depth_blob_recover", False) or
                   self.cfg["perception"].get("novel_objects", False))
        if not need_fk or obs.get("arm_q") is None:
            return {}
        return {"robot_q": (np.asarray(obs["arm_q"], float),
                            None if obs.get("hand_q") is None else np.asarray(obs["hand_q"], float))}

    def _counter_kw(self, obs):
        if not self.cfg.get("calib", {}).get("counter_after_hand", False):
            return {}
        return {"calibration_frame": self._calibration_frame(obs)}

    @staticmethod
    def _detector_finished(deadline_job):
        if deadline_job is None:
            return False
        with deadline_job["deadline_lock"]:
            deadline_job["phase"] = "plan"
            return bool(deadline_job.get("cancelled"))

    def _look(self, obs, deadline_job=None, detector=None):
        detector = self.perception if detector is None else detector
        verify = self._verify_on()
        if not verify:  # flag off: the stock look (no bin review, no corrective picks, no relooks)
            dets = detector.detect(obs["head_rgb"], obs["head_depth"], **self._detect_kw(), **self._fk_kw(obs),
                                   **self._counter_kw(obs))
            if self._detector_finished(deadline_job):
                return ("error", "late detector result discarded")
            if dets is None:
                return ("still_vote_pending",)
            if (self.cfg.get("instruction") or {}).get("novel_objects", False):
                self._match_novel(dets)
            self.last_detections = dets
            self._settle_skill_attempt(dets, float(obs.get("t", 0.0)))
            if self._landing_on():   # policy.verify_landing: classify the last release, never target an in-bin object
                relook, dets = self._landing_review(obs, dets)
                if relook is not None:
                    return relook
            return self._choose_retry(dets, obs, verify=False)
        dets, rejected = detector.detect(obs["head_rgb"], obs["head_depth"], debug=True, include_bins=True,
                                         **self._detect_kw(), **self._fk_kw(obs), **self._counter_kw(obs))
        if self._detector_finished(deadline_job):
            return ("error", "late detector result discarded")
        if dets is None:
            return ("still_vote_pending",)
        if (self.cfg.get("instruction") or {}).get("novel_objects", False):
            self._match_novel(dets)
        bin_dets = [(d, why.split(":", 1)[1]) for d, why in rejected if why.startswith("in_bin:")]
        self.last_detections = dets
        self.last_bin_detections = bin_dets
        t = float(obs.get("t", 0.0))
        if self.verification_pending is None:   # a pending release is settled by _review_bins instead
            self._settle_skill_attempt(dets, t)
        corrections = self._review_bins(bin_dets, t)

        # A known wrong-bin object is the largest score swing, so repair it before
        # touching a new counter object.  The fresh detection is planned directly;
        # stale corrective targets are never retained between camera frames.
        self.scheduled_corrections = corrections
        cap = self.cfg["policy"].get("max_corrections_per_episode")
        guarded_corrections = False
        if cap is None:  # stock #6: corrections first; none plannable -> stop
            if corrections:
                res = self._plan_corrections(corrections, dets, t)
                if res is not None:
                    return res
                if not self._target_guard_on():
                    return ("uncorrectable", corrections, len(dets))
                guarded_corrections = True
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
        if guarded_corrections and res[0] == "none":
            return ("relook", None, len(dets))
        if cap is not None and corrections and res[0] in ("none", "notime") and self.n_corrections < int(cap):
            alt = self._plan_corrections(corrections, dets, t)
            if alt is not None:
                return alt
            if self._target_guard_on() and res[0] == "none":
                return ("relook", None, len(dets))
        return res

    def _match_novel(self, dets):
        """Attach a bin only to one deterministic visible novel match.

        Unknown nouns expose a closed visible-novel list to the already
        bounded asynchronous brain.  With no brain, or on timeout, every such
        object keeps ``novel_target_bin=None`` and is therefore unpickable.
        """
        self._novel_brain_candidates, self._novel_brain_bin = [], None
        novels = [d for d in dets if getattr(d, "novel", False)]
        for d in novels:
            d.novel_target_bin, d.novel_matched = None, False
        if not novels or not (self.cfg.get("instruction") or {}).get("novel_objects", False):
            return
        from .novel_objects import match_instruction
        match = match_instruction(self._instruction, novels)
        if match.matched is not None:
            match.matched.novel_target_bin = match.target_bin
            match.matched.novel_matched = True
            self.log(f"[novel] matched {match.matched.color} {match.matched.shape} -> {match.target_bin}")
        elif match.brain_candidates and self.brain is not None:
            self._novel_brain_candidates = list(match.brain_candidates)
            self._novel_brain_bin = match.target_bin
            self.log(f"[novel] unresolved noun {match.unresolved_noun!r}; offered "
                     f"{len(match.brain_candidates)} visible novel candidates to brain")

    def _plan_corrections(self, corrections, dets, t=0.0):
        """First plannable corrective pick: wrong/unscored bin -> a target-bin slot, else -> counter. None if none."""
        for d, source_bin, target_bin in corrections:
            if getattr(d, "tentative", False):
                continue  # partial_ok_near_hand: never grasp an FK-boundary fragment directly
            if self._target_guarded(d, t):
                continue
            budget_blocked = False
            target_blocked = False
            if self._vary_on():   # no place gate: the object is already in a bin
                n = self._attempt_count(d.top[:2], d.color)
                limit = self._attempt_limit(d.top[:2])
                varied = self._max_attempts_vary()
                if n >= (max(limit, varied) if varied is not None else limit):
                    self.log(f"[classical] correction attempt limit reached for {d.color} in {source_bin}")
                    continue
            for slot in self._free_drop_slots(target_bin) or self._overflow_slots(target_bin):
                try:
                    p = self._plan_pick(d, dict(bin_name=target_bin, relocate=False,
                                                drop_slot=slot, source_bin=source_bin))
                except Exception as exc:
                    if not self._target_guard_on():
                        raise
                    self._cool_target(d, t, f"{type(exc).__name__}: {exc}")
                    target_blocked = True
                    break
                self.plans.append(p)
                if p.ok and not self._object_time_allows(d, p.duration):
                    budget_blocked = True
                    break
                if p.ok:
                    return ("plan", p, len(dets))
                if self._target_guard_on():
                    self._cool_target(d, t, p.reason or "infeasible")
                    target_blocked = True
                    break
                self.log(f"[classical] cannot correct {d.color} from {source_bin} to slot {slot}: {p.reason}")
            if budget_blocked or target_blocked:
                continue
            def cleanup(det, arm, hand, planner=None, **_):
                return (planner or self.planner).plan_to_counter(det, arm, hand, source_bin,
                                                    obstacles=[o.top[:2] for o in dets])
            try:
                p = self._plan_pick(d, {}, plan_fn=cleanup)
            except Exception as exc:
                if not self._target_guard_on():
                    raise
                self._cool_target(d, t, f"{type(exc).__name__}: {exc}")
                continue
            self.plans.append(p)
            if p.ok and not self._object_time_allows(d, p.duration):
                continue
            if p.ok:
                return ("plan", p, len(dets))
            if self._target_guard_on():
                self._cool_target(d, t, p.reason or "infeasible")
            else:
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
        choose_dets = dets
        tentative = []
        if self.cfg["perception"].get("partial_ok_near_hand", False):
            tentative = [d for d in dets if getattr(d, "tentative", False)]
            choose_dets = [d for d in dets if not getattr(d, "tentative", False)]
        res = self._choose(choose_dets, obs, verify)
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
                res = self._choose(choose_dets, obs, verify)
        max_relooks = int((self.cfg["perception"].get("partial_ok_near_hand_params") or {}).get("max_relooks", 2))
        if res[0] == "none" and tentative and self._partial_relooks < max_relooks:
            return ("partial", None, len(dets))
        if res[0] == "none" and choose_dets and self._landing_on():
            res = self._landing_regrant(res, choose_dets, obs)
        return res

    # ------------------------------------------------------------------ policy.verify_landing
    def _landing_on(self):
        """verify_landing is the observe-only verifier; the old verify_placement (#6) keeps its own machinery."""
        return bool(self.cfg["policy"].get("verify_landing", False)) and not self._verify_on()

    def _landing_review(self, obs, dets):
        """-> (relook result | None, dets). Never raises: any error keeps the stock detections."""
        try:
            return self._landing_review_inner(obs, dets)
        except Exception as exc:
            self.log(f"[landing] disabled for this look: {type(exc).__name__}: {exc}")
            return None, dets

    def _landing_review_inner(self, obs, dets):
        pol = self.cfg["policy"]
        t = float(obs.get("t", 0.0))
        occ = landing.bin_occupancy(self.perception, obs.get("head_depth"))
        pend = self._landing
        if pend is not None:
            min_delta = float(pol.get("landing_min_delta", 0.5))
            kind, info = landing.classify_landing(pend, self._occ_prev, occ, dets, min_delta)
            if kind in ("lost", "unknown"):
                cb = self._landing_colour_bin(obs, pend["color"])
                if cb is not None:
                    kind, info = landing.classify_landing(pend, self._occ_prev, occ, dets, min_delta, colour_bin=cb)
            if kind in ("lost", "unknown") and pend.get("relooks", int(pol.get("landing_relooks", 1))) > 0:
                pend["relooks"] = pend.get("relooks", int(pol.get("landing_relooks", 1))) - 1
                return ("relook", None, len(dets)), dets   # a bounce may still roll: one more settled frame
            self._landing = None
            self.landing_events.append((kind, pend["color"], pend["bin"]))
            dtxt = {k: round(v, 2) for k, v in info["delta"].items()}
            self.log(f"[landing] t={t:.1f} {pend['color']}->{pend['bin']}: {kind} delta={dtxt}")
            if kind == "wrong_bin":
                self.log(f"[landing] {pend['color']} landed in {info['bin']} (wrong bin, -1): logged, not corrected")
            elif kind == "bounce":
                dets = self._landing_requeue(pend, info["det"], dets)
        self._occ_prev = occ
        margin = float(pol.get("landing_in_bin_margin_m", 0.01))
        keep = []
        for d in dets:
            b = landing.inside_bin(self.perception, d.top[:2], margin)
            if b is None:
                keep.append(d)
            elif (d.color, b) not in self._landing_excl_logged:
                self._landing_excl_logged.add((d.color, b))
                self.log(f"[landing] {d.color} inside the {b} bin footprint: never a pick target")
        self.last_detections = keep
        return None, keep

    def _landing_colour_bin(self, obs, color):
        """Bin name of a same-colour blob inside a bin footprint (colour fallback when depth is inconclusive)."""
        try:
            _, rejected = self.perception.detect(obs["head_rgb"], obs["head_depth"], debug=True, include_bins=True,
                                                 **self._detect_kw(), **self._fk_kw(obs))
        except Exception:
            return None
        for d, why in rejected:
            if d is not None and d.color == color and why.startswith("in_bin:"):
                return why.split(":", 1)[1]
        return None

    def _landing_requeue(self, pend, det, dets):
        """A bounced object is a normal counter target again: refund the attempt, forget the old give-ups, jitter the grasp."""
        pol = self.cfg["policy"]
        src = np.asarray(pend["src_xy"], float)
        if self.landing_refunds < int(pol.get("landing_max_refunds", 4)):
            self.landing_refunds += 1
            rad = self._attempt_radius()
            for i in range(len(self.attempts) - 1, -1, -1):
                if np.linalg.norm(np.asarray(self.attempts[i][0], float) - src) < rad:
                    self.attempts.pop(i)
                    break
            near = [tr for tr in self.attempt_tracks
                    if tr["color"] == pend["color"] and np.linalg.norm(src - tr["xy"]) < rad]
            if near:
                tr = min(near, key=lambda item: np.linalg.norm(src - item["xy"]))
                tr["count"] = max(0, tr["count"] - 1)
        rad = float(pol["same_object_radius"])
        self.failed = [f for f in self.failed if np.linalg.norm(det.top[:2] - f) >= rad
                       and np.linalg.norm(src - f) >= rad]
        n = self.bounce_count.get(pend["color"], 0)
        self.bounce_count[pend["color"]] = n + 1
        jit = landing.jitter(n, float(pol.get("landing_jitter_m", 0.01)), det.top[:2])
        moved = dataclasses.replace(det, top=det.top + np.r_[jit, 0.0])
        self.log(f"[landing] {pend['color']} bounced back to the counter at ({det.top[0]:+.3f},{det.top[1]:+.3f}): "
                 f"re-queued, jitter ({jit[0] * 100:+.1f},{jit[1] * 100:+.1f}) cm")
        return [moved if d is det else d for d in dets]

    def _landing_regrant(self, res, dets, obs):
        """Nothing pickable but objects were skipped for max_attempts only: with a full pick+place cycle of time left
        they get one fresh attempt (vary_hist keeps the retry from repeating a grasp). At most landing_regrants_max."""
        pol = self.cfg["policy"]
        t = float(obs.get("t", 0.0))
        if (self.landing_regrants >= int(pol.get("landing_regrants_max", 2))
                or self.time_limit - t < float(pol.get("landing_regrant_min_s", 14.0))):
            return res
        rad = self._attempt_radius()
        excluded = [(d.color, d.top[:2]) for d in dets if self._attempt_excluded(d.top[:2], d.color)]
        if not excluded:
            return res
        self.landing_regrants += 1
        self.attempts = [(p, c) for p, c in self.attempts
                         if not any(np.linalg.norm(xy - p) < rad for _, xy in excluded)]
        self.attempt_tracks = [tr for tr in self.attempt_tracks
                               if not any(tr["color"] == color and np.linalg.norm(xy - tr["xy"]) < rad
                                          for color, xy in excluded)]
        self.retry_allow += [(xy, 1) for _, xy in excluded]
        self._note(t, f"landing regrant {self.landing_regrants}: {len(excluded)} object(s) get one more attempt")
        return self._choose(dets, obs, False)

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
        return self._assoc_radius() if self._vary_on() else float(self.cfg["policy"]["same_object_radius"])

    def _assoc_radius(self):
        """Same-colour detection-to-target association radius across camera frames."""
        perception = self.cfg.get("perception") or {}
        # The policy fallback keeps small isolated unit-test configs compatible; production config owns the value.
        return float(perception.get("assoc_radius_m", self.cfg.get("policy", {}).get("retry_assoc_radius", 0.06)))

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
            if self._target_guarded(d, float(obs.get("t", 0.0))):
                continue
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
        if decision and pol.get("score_order", False):
            try:
                cands = self._score_order(cands, obs)
            except Exception as e:
                self._dec_fail("score_order", e)
        if extra:
            extra.sort(key=base)
            cands += extra
        return cands, extra

    # ------------------------------------------------------------------ policy.score_order (EV per second)
    def _so_estimate(self, d, t):
        """(p_success, p_wrong, t_cost_s, arm, bin) for one counter candidate. Pure arithmetic, no IK."""
        pol = self.cfg["policy"]
        tab = pol.get("score_p_success", {}) or {}
        shape = getattr(d, "shape", None) or "unknown"
        ps = float(tab.get(shape, tab.get("unknown", 0.5)))
        sc = float(getattr(d, "shape_conf", 0.0) or 0.0)
        if getattr(d, "shape", None) and 0.0 < sc < 1.0:
            ps *= 0.8 + 0.2 * sc                        # unsure shape -> unsure grasp posture
        xy = np.asarray(d.top[:2], float)
        arm_y = float(pol.get("score_arm_y_m", 0.2))
        side = "left" if xy[1] >= 0 else "right"        # pelvis frame: x forward, y left
        anchor = np.array([0.0, arm_y if side == "left" else -arm_y])
        reach = float(np.linalg.norm(xy - anchor))
        r_max = float(pol.get("score_reach_m", 0.55))
        if reach > r_max:                               # past the comfortable reach: success drops off linearly
            ps *= max(0.1, 1.0 - (reach - r_max) / 0.15)
        if self._skills_on():                           # skill memory: shrink the prior towards the observed rate
            try:
                st = self.skill_memory._stats[self._skill_key(d, side)]["stock"]
                n = float(st["n"])
                if n > 0:
                    ps = (ps * 2.0 + float(st["reward"])) / (2.0 + n)
            except Exception:
                pass
        b = lookup_detection_bin(self.cfg, self.mapping, d)
        if b is None:                                   # abstain class: never a guess
            return float(np.clip(ps, 0, 1)), 1.0, 1.0, side, None
        conf = float(np.clip(getattr(d, "color_conf", 1.0), 0.0, 1.0)) * self._bin_confidence(d)
        if getattr(d, "shape", None) is None and self.cfg.get("policy", {}).get("bin_lookup") == "resolve_bin":
            conf *= 0.9                                 # shape-dependent mapping with an unknown shape
        pw = float(np.clip(1.0 - conf, 0.0, 1.0))
        bxy = np.asarray(self.planner.bin_xy(b), float)
        path = 2.0 * reach + float(np.linalg.norm(xy - bxy))   # home -> object -> bin, back
        tc = float(pol.get("score_overhead_s", 8.0)) + path / float(pol.get("score_speed_mps", 0.12))
        return float(np.clip(ps, 0, 1)), pw, max(tc, 1.0), side, b

    def _score_order(self, cands, obs):
        """EV per second: ps * (2 (1 - pw) - pw) / t_cost. EV <= 0 is never picked; a pick must finish before
        time_limit - reserve - score_safety_s; in the last score_endgame_s only the quickest certain object."""
        pol = self.cfg["policy"]
        t = float(obs.get("t", 0.0))
        left = self.time_limit - t - self._time_reserve(corrective=False) - float(pol.get("score_safety_s", 3.0))
        rows = []
        for d in cands:
            ps, pw, tc, side, b = self._so_estimate(d, t)
            ev = ps * (2.0 * (1.0 - pw) - pw)
            rows.append((ev / tc, ev, ps, pw, tc, side, b, d))
        rows.sort(key=lambda r: -r[0])
        out, notes = [], []
        for r in rows:
            why = "ok"
            if r[1] <= 0.0:
                why = "ev<=0"
            elif r[4] > left:
                why = "no time"
            notes.append((r, why))
            if why == "ok":
                out.append(r)
        if out and left < float(pol.get("score_endgame_s", 25.0)):
            certain = [r for r in out if r[3] <= float(pol.get("score_certain_pw", 0.1))]
            out = [min(certain, key=lambda r: r[4])] if certain else []
        for r, why in notes:
            self.log(f"[score_order] t={t:.1f} {r[7].color:<6} {str(getattr(r[7], 'shape', None)):<8} arm={r[5]:<5} "
                     f"bin={r[6]} ps={r[2]:.2f} pw={r[3]:.2f} t={r[4]:.1f}s EV={r[1]:+.2f} EV/s={r[0]:+.3f} {why}")
        if out:
            self.log(f"[score_order] t={t:.1f} -> {out[0][7].color} ({out[0][5]} arm, EV/s={out[0][0]:+.3f})")
        return [r[7] for r in out]

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
            brain_only_novel = bool(self._novel_brain_candidates)
            offered = [d for d in self._novel_brain_candidates if any(d is x for x in dets)]
            sel = self._brain_select(offered, offered, obs) if brain_only_novel else self._brain_select(cands, dets, obs)
            if sel is not None:
                kind, d0, b0, var = sel
                if brain_only_novel:
                    valid = (kind == "pick" and any(d0 is d for d in offered)
                             and getattr(d0, "novel", False) and b0 == self._novel_brain_bin
                             and getattr(d0, "graspable", False))
                    if valid:
                        d0.novel_target_bin, d0.novel_matched = b0, True
                    else:
                        # The novel hook may neither stop the known task nor
                        # select a known/ungraspable object or another bin.
                        sel = None
                if sel is None:
                    kind = None
                if kind is None:
                    pass
                elif kind == "look":
                    return ("relook", None, len(dets))
                elif kind == "done":
                    return ("none", None, len(dets))
                elif kind == "skip":
                    self.failed.append(d0.top[:2])          # the brain gave this object up: no more tries
                    cands = [d for d in cands if d is not d0]
                elif kind in ("pick", "retry"):
                    cands = [d0] + [d for d in cands if d is not d0]
                    forced = (d0, b0, var)
        left = self.time_limit - t - self._time_reserve(corrective=False)
        n_long, shortest = 0, None
        cap_s, over_cap = None, []
        guarded_failure = False
        if self._dflag("object_time_cap"):   # D2: per-object cap (remaining - 15 s) / objects left, 15..35 s
            cap_s = dec.object_cap(self.time_limit - t, len(cands) - len(extra) or len(cands), self._dcfg())
        for d in cands:
            if triage:  # a plan that did not fit before cannot fit now (time only runs out; plans start from home)
                rad = pol["same_object_radius"]
                if any(np.linalg.norm(d.top[:2] - p) < rad and dur > left for p, dur in self.too_long):
                    n_long += 1
                    continue
            others = [o.top[:2] for o in dets if o is not d]
            common = {"obstacles": others}
            if pol.get("novel_obstacles", False):
                clearance = []
                for o in dets:
                    if o is d or not getattr(o, "novel", False):
                        continue
                    dims = np.asarray(getattr(o, "footprint_dims", [0.05, 0.05]), float)
                    radius = 0.5 * float(np.max(dims))
                    top_z = self.scene.counter_z + float(getattr(o, "height", o.top[2] - self.scene.counter_z))
                    centre = np.asarray(getattr(o, "obstacle_center", o.top[:2]), float)
                    clearance.append([centre[0], centre[1], radius, top_z])
                common["clearance_obstacles"] = clearance
            mine = forced is not None and d is forced[0]
            target_bin = forced[1] if mine else self._bin_of(d)
            pfn = self._variant_plan_fn(forced[2]) if mine else None
            if target_bin is None:
                continue  # unmapped / abstain: 0 points on the counter, never a -1 guess
            if not self._place_gate_allows(d, target_bin):
                continue
            try:
                if verify:  # #6: explicit target bin + a free, verified drop slot
                    slot = self._next_drop_slot(target_bin)
                    if slot is None:
                        slot = next(iter(self._overflow_slots(target_bin)), None)
                    if slot is None:
                        self.log(f"[classical] no free drop slot in {target_bin}; leave {d.color} on counter")
                        continue
                    p = self._plan_pick(d, dict(bin_name=target_bin, drop_slot=slot, **common), pfn)
                elif self._slots_on():  # grasp.drop_slots standalone: each successive object in a bin -> next slot
                    free = [i for i in range(len(self.planner.drop_slots(target_bin)))
                            if i not in self.slot_used[target_bin]]
                    slot = free[0] if free else next(iter(self._overflow_slots(target_bin)), None)
                    if slot is None:
                        self.log(f"[classical] drop slots of {target_bin} used up; leave {d.color} on counter")
                        continue
                    p = self._plan_pick(d, dict(bin_name=target_bin, drop_slot=slot, **common), pfn)
                elif mine:   # brain override: the classical planner still checks reach
                    p = self._plan_pick(d, dict(bin_name=target_bin, **common), pfn)
                else:
                    p = self._plan_pick(d, common)
            except Exception as exc:
                if not self._target_guard_on():
                    raise
                self._cool_target(d, t, f"{type(exc).__name__}: {exc}")
                guarded_failure = True
                continue
            self.plans.append(p)
            if p.ok and not self._object_time_allows(d, p.duration):
                continue
            if p.ok and pol.get("score_order", False) and \
                    t + p.duration + float(pol.get("score_safety_s", 3.0)) + self._time_reserve(p.corrective) > self.time_limit:
                n_long += 1                      # score_order: do not start a pick that cannot finish in time
                shortest = p if shortest is None or p.duration < shortest.duration else shortest
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
            if self._target_guard_on():
                self._cool_target(d, t, p.reason or "infeasible")
                guarded_failure = True
            else:
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
        cooling = self._target_guard_on() and any(self._target_guarded(d, t) for d in dets)
        if guarded_failure or cooling:
            return ("relook", None, len(dets))
        return ("none", None, len(dets))

    def _target_guard_on(self):
        return bool((self.cfg.get("planner") or {}).get("per_target_guard", False))

    def _target_guarded(self, d, t):
        if not self._target_guard_on():
            return False
        self._target_cooldowns = [x for x in self._target_cooldowns if x["until"] > t]
        radius = float(self.cfg["policy"]["same_object_radius"])
        xy = np.asarray(d.top[:2], float)
        return any(x["color"] == d.color and np.linalg.norm(x["xy"] - xy) < radius
                   for x in self._target_cooldowns)

    def _cool_target(self, d, t, reason):
        cooldown = float(self.cfg["planner"]["per_target_cooldown_s"])
        t = max(float(t), float(self._latest_t))
        xy = np.asarray(d.top[:2], float).copy()
        self._target_cooldowns = [x for x in self._target_cooldowns
                                  if not (x["color"] == d.color and np.linalg.norm(x["xy"] - xy) <
                                          float(self.cfg["policy"]["same_object_radius"]))]
        self._target_cooldowns.append({"color": d.color, "xy": xy, "until": t + cooldown})
        self.log(f"[plan_guard] {d.color} target cooled for {cooldown:g} s: {reason}")

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
        """D3: judge the last pick at the next look; the attempt track uses perception.assoc_radius_m."""
        po, self._pend_out = self._pend_out, None
        if po is None:
            return
        rad = self._assoc_radius()
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
            mo = self.cfg["motion"]
            if self.cfg["grasp"].get("staggered_release", False):
                h1, _ = staggered_release_ramp(
                    self.hand_cmd, rel, plan.side, mo["hand_max_vel"], 1.0 / HZ,
                    float(self.cfg["grasp"].get("staggered_release_delay_s", 0.30)))
            else:
                amax = mo.get("hand_max_acc") if mo.get("output_caps") is not None else None
                h1 = ramp(self.hand_cmd, rel, mo["hand_max_vel"], 1.0 / HZ, amax=amax)
            h1 = np.vstack([h1, np.repeat(rel[None], int(round(float(self.cfg["motion"]["open_hold_s"]) * HZ)), 0)])
            a1 = np.repeat(self.arm_cmd[None], len(h1), 0)
            a2, h2 = self.planner.go_to(self.arm_cmd, rel, home, self.post["open"])
            a, h = np.vstack([a1, a2]), np.vstack([h1, h2])
        else:
            a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, home, self.post["open"])
        self._stall = self._gdiag = self._rel = None
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
        perception.assoc_radius_m of the earlier grasp point (a missed object is often pushed a few cm)."""
        r = self._assoc_radius()
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

    def _jittered_retry_detection(self, d):
        """Shallow-copy a retry detection with deterministic +x, -x, +y, -y grasp-point jitter."""
        magnitude = float(self.cfg["policy"].get("retry_jitter_m", 0.0))
        if magnitude <= 0.0:
            return d
        attempt = self._attempt_count(np.asarray(d.top[:2], float), getattr(d, "color", None))
        if attempt <= 0:
            return d
        pattern = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        shifted = copy.copy(d)
        shifted.top = np.asarray(d.top, float).copy()
        shifted.top[:2] += magnitude * np.asarray(pattern[(attempt - 1) % len(pattern)], float)
        return shifted

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
        no_yaw = label == "cylinder" and bool(g.get("cylinder_no_yaw_vary", False))   # rotationally symmetric
        if prev["yaw"] is not None and not no_yaw:
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
        plan_detection = self._jittered_retry_detection(d)

        def plan(relocate=None):
            args = dict(kw)
            if relocate is not None and plan_fn is None:
                args["relocate"] = relocate
            with self._decision_motion() as pl:   # D3/D4 speed scale + first-object clearance (stock planner when off)
                pl = self._first_object_planner(pl)
                if plan_fn is None:
                    return pl.plan(plan_detection, self.arm_cmd, self.hand_cmd, **args)
                if pl is not self.planner:
                    args["planner"] = pl
                return plan_fn(plan_detection, self.arm_cmd, self.hand_cmd, **args)

        if not self._vary_on():
            p = plan()
            return self._tag_skill_plan(p, d, "stock") if self._skills_on() else p
        hist = self._hist_for(d)
        if not hist:
            p = plan()
            if p.ok:
                p.grasp_key = self._grasp_key(p)
            if not self._skills_on() or not p.ok:
                return self._tag_skill_plan(p, d, "stock")
            return self._skill_first_try(d, p, plan)
        for arm, v in self._ordered_skill_variants(d, hist, self._variants(d, hist)):
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
            self._tag_skill_plan(p, d, arm)
            self.log(f"[retry_vary] {d.color} label={getattr(d, 'shape', None)} try {len(hist) + 1}: "
                     f"{self._vary_txt(v)} -> {self._key_txt(key)} (before: "
                     + ", ".join(self._key_txt((h["xy"], h["yaw"], h["posture"])) for h in hist) + ")")
            return p
        p = plan()
        if p.ok:
            p.grasp_key = self._grasp_key(p)
            if any(self._same_grasp(p.grasp_key, h) for h in hist):
                return Plan(False, color=d.color, reason="retry_vary: every grasp repeats a tried (point, yaw, posture)")
        return self._tag_skill_plan(p, d, "stock")

    # ---- skills.memory (FLAG, default OFF). Only ORDERS the existing retry-vary motions and their grasp parameters;
    # it never adds a motion, never changes the bin, and a variant must keep the stock plan's destination.
    def _skills_on(self):
        return getattr(self, "skill_memory", None) is not None

    @staticmethod
    def _skill_key(d, side):
        return (getattr(d, "shape", None) or "unknown", getattr(d, "color", "unknown"), side or "unknown")

    @staticmethod
    def _skill_arm(v, prev):
        """Name an existing `_variants` motion without creating a new posture or heading."""
        alt = v["posture"] is not None and v["posture"] != prev["posture"]
        if v["yaw"] is None:
            return "alternate_posture" if alt else "stock"
        delta = (v["yaw"] - prev["yaw"] + np.pi) % (2 * np.pi) - np.pi
        suffix = "positive" if delta >= 0 else "negative"
        return ("alternate_posture_yaw_" if alt else "yaw_") + suffix

    def _ordered_skill_variants(self, d, hist, variants):
        """Reorder only motions already returned by `_variants`; flag off = the original order, same tuples."""
        if not self._skills_on():
            return [(None, v) for v in variants]
        prev = hist[-1]
        labelled = [(self._skill_arm(v, prev), v) for v in variants]
        chosen = self.skill_memory.choose(self._skill_key(d, prev["side"]), [arm for arm, _ in labelled])
        rank = {arm: i for i, arm in enumerate(chosen)}
        return sorted(labelled, key=lambda item: rank.get(item[0], len(rank)))

    def _skill_first_try(self, d, p, plan):
        """First try at an object: the stock plan, unless memory ranks an existing variant above 'stock' AND that
        variant is a plan to the same destination (a counter relocation stays stock: relocate=False never adds one)."""
        key = p.grasp_key
        prev = {"color": d.color, "xy": np.asarray(key[0], float), "yaw": key[1], "posture": key[2],
                "side": p.side, "held": False}
        variants = [(self._skill_arm(v, prev), v) for v in self._variants(d, [prev])]
        chosen = self.skill_memory.choose(self._skill_key(d, p.side), ["stock"] + [arm for arm, _ in variants])
        by_arm = dict(variants)
        for arm in chosen:
            if arm == "stock":
                break
            v = by_arm.get(arm)
            if v is None:
                continue
            self.planner.vary = v
            try:
                candidate = plan(relocate=False)
            finally:
                self.planner.vary = None
            if candidate.ok and candidate.target_bin == p.target_bin:
                candidate.grasp_key = self._grasp_key(candidate)
                return self._tag_skill_plan(candidate, d, arm)
        return self._tag_skill_plan(p, d, "stock")

    def _tag_skill_plan(self, plan, d, arm):
        if self._skills_on() and plan.ok:
            plan.skill_arm = arm
            plan.skill_key = self._skill_key(d, plan.side)
        return plan

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

    def _review_bins(self, bin_dets, t=None):
        """Update slot occupancy, verify the last release, and return corrective picks.

        A coloured component on a bin footprint but with its estimated centre
        above the rim is deliberately treated as unscored and scheduled for a
        re-pick, just like an object in the wrong bin.
        """
        scored = []
        corrections = []
        yellow_unknown = bool(self.cfg["policy"].get("verify_yellow_unknown", False))
        for d, footprint_bin in bin_dets:
            if getattr(d, "tentative", False):
                continue  # re-observe with retracted arms; never score or correct this fragment
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
                self._finish_skill_attempt("placed", t, held=True)
            elif yellow_unknown and pending["color"] == "yellow":
                self.reserved_drop_slots[target].add(slot)   # unknown: keep the slot blocked, no retry / relook
                self.last_verification = ("unknown", pending)
                self._finish_skill_attempt("held", t, held=True)
            else:
                # Keep the attempted slot unavailable: if the side-bin view was
                # occluded, a successful object may still be sitting there.
                self.reserved_drop_slots[target].add(slot)
                self.retry_colors.add(pending["color"])
                if pending.get("source_xy") is not None:
                    self.retry_targets.append((pending["color"], np.asarray(pending["source_xy"], float)))
                self.verify_relooks_remaining = int(self.cfg["policy"].get("verify_relooks", 1))
                self.last_verification = ("retry", pending)
                self._finish_skill_attempt("dropped", t, held=False)
            self.verification_pending = None
        return corrections

    def _after_look(self, res, t):
        if res[0] == "still_vote_pending":
            return self._hold()
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
        confirm = bool(self.cfg["policy"].get("done_confirm", False))
        retry_failed = confirm or self.cfg["perception"].get("look_timeout_s") is not None
        if (confirm or retry_failed or self._target_guard_on()) and t >= self.time_limit:
            self._clear_looks, self._clear_since = 0, None
            self._note(t, "episode time limit reached")
            self.state = "done"
            return self._hold(done=True)
        if confirm and kind != "none":
            self._clear_looks, self._clear_since = 0, None
        if kind == "error":
            if retry_failed:
                self._note(t, f"look failed ({res[1]}); retrying while time remains")
                self.state = "settle"
                return self._hold()
            self._note(t, f"look/plan error {res[1]}: stopping")
            self.state = "done"
            return self._hold(done=True)
        if kind == "relook":
            self._note(t, "placement unresolved; take one more settled verification frame")
            self.state = "settle"
            return self._hold()
        if kind == "partial":
            self._partial_relooks += 1
            return self._begin_retracted_relook(t, "tentative FK-boundary candidate",
                                                f"partial relook {self._partial_relooks}")
        if kind == "uncorrectable":
            if confirm:   # a visible known-bin object is never a reason to finish: keep looking until time runs out
                self._note(t, "known wrong-bin object has no safe corrective or cleanup plan yet; retrying")
                self.state = "settle"
                return self._hold()
            self._note(t, "known wrong-bin object has no safe corrective or cleanup plan; stopping")
            self.state = "done"
            return self._hold(done=True)
        if kind == "none":
            if self.cfg["perception"].get("relook_before_done", False) and not self._final_relook_used:
                self._final_relook_used = True
                return self._begin_retracted_relook(t, "nothing-left guard", "final relaxed relook")
            if confirm:
                n_dets = int(res[2])
                if n_dets:
                    self._clear_looks, self._clear_since = 0, None
                    self._note(t, f"{n_dets} detection(s), but no plan; retrying")
                    self.state = "settle"
                    return self._hold()
                if self._clear_since is None:
                    self._clear_since = float(t)
                self._clear_looks += 1
                need = int(self.cfg["policy"]["done_confirm_looks"])
                span = float(t) - self._clear_since
                min_span = float(self.cfg["policy"]["done_confirm_span_s"])
                if self._clear_looks < need or span < min_span:
                    self._note(t, f"clear look {self._clear_looks}/{need}, span {span:.1f}/{min_span:g} s; confirming")
                    self.state = "settle"
                    return self._hold()
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

    def _begin_retracted_relook(self, t, reason, note):
        """Queue a normal home/open trajectory, then let `_act` settle and count fresh frames without sleeping."""
        a, h = self.planner.go_to(self.arm_cmd, self.hand_cmd, self.home, self.post["open"])
        self._enqueue(a, h)
        self.state = "relook_settle"
        self._relook_fresh = 0
        self._relook_last_t = None
        self._relook_reason = reason
        self._note(t, f"{note}: retract arms to home before re-detection")
        return self._pop()

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
        if self._landing_on() and not plan.corrective and plan.target_bin in self.scene.bin_names:
            self._landing = {"color": plan.color, "bin": plan.target_bin, "src_xy": np.asarray(xy, float).copy(),
                             "pre_xy": [d.top[:2].copy() for d in self.last_detections if d.color == plan.color],
                             "t": float(t)}
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
        self._rel = None
        if self.cfg["grasp"].get("release_confirm", False) and plan.side in ("left", "right"):
            ph = {name: e for name, _, e in (plan.phases or [])}
            if "open" in ph:
                self._rel = {"plan": plan, "n": len(plan.arm), "open_end": ph["open"], "phase": "wait", "reopened": False}
        self._stall = None
        if ls.get("enabled", False) and plan.close_hold is not None and plan.side in ("left", "right"):
            self._stall = {"n": len(plan.arm), "hold": plan.close_hold, "sl": ArmKinematics.sl(plan.side),
                           "min": int(round(float(self.cfg["motion"]["close_hold_s"]) * HZ)),
                           "rad": float(ls.get("stall_rad", 0.01)), "need": int(ls.get("settle_ticks", 3)),
                           "last": None, "still": 0, "t": t}
        skill_side = self._skills_on() and plan.side in ("left", "right")
        diag_on = self.cfg["policy"].get("grasp_diag_log", False) or self.held_detector.mode == "finger_sum"
        if (diag_on or skill_side) and plan.side in ("left", "right"):
            ph = {name: (s_, e) for name, s_, e in (plan.phases or [])}
            lift_key = "lift_check" if "lift_check" in ph else "lift"   # grasp.soft_squeeze: re-read after the first 2 cm
            if "close" in ph and lift_key in ph:
                check = self.held_detector.evaluation_tick(
                    *ph["close"], HZ, hold_start=plan.close_hold[0] if plan.close_hold else None)
                # skill_only: watched for skills.memory alone -> no empty-grasp abort, no [grasp_diag] line
                self._gdiag = {"plan": plan, "n": len(plan.arm), "close_end": check,
                               "lift_end": ph[lift_key][1], "t": t, "samples": {}, "skill_only": not diag_on}
                if lift_key == "lift_check" and self.held_detector.soft is not None:
                    self._gdiag["lift_check"] = True
                if bool((self.cfg.get("held") or {}).get("read_through_carry", False)) \
                        and self.held_detector.mode == "finger_sum" and "carry" in ph:
                    self._gdiag["carry"] = ph["carry"]
        hist = None
        if self._vary_on() and plan.side in ("left", "right"):
            key = plan.grasp_key or (np.asarray(xy, float).copy(), None, self._posture_of(plan))
            h = {"color": plan.color, "xy": np.asarray(key[0], float), "yaw": key[1], "posture": key[2],
                 "side": plan.side, "held": None}
            self.vary_hist.append(h)
            hist = h
        if skill_side:
            self._skill_pending = {"key": getattr(plan, "skill_key", ("unknown", plan.color, plan.side)),
                                   "arm": getattr(plan, "skill_arm", "stock"), "color": plan.color,
                                   "xy": np.asarray(xy, float).copy(), "held": None, "hist": hist,
                                   "started": float(t), "duration_s": float(plan.duration)}
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

    def _thr(self, key):
        m = self._soft_margin(key)
        return self.held_detector.threshold_rad if m is None else m

    def _soft_margin(self, key):
        """Soft detector margin for a read: the close read uses soft.margin, the lift_check / carry reads the (smaller)
        lift margin. None when the soft detector is off (held() then ignores it)."""
        soft = self.held_detector.soft
        if soft is None:
            return None
        return soft["margin"] if key == "close" else soft["lift_margin"]

    def _soft_baseline_start(self, t):
        """held.soft_detector: once per episode, at home with both hands open, close both hands on nothing with the
        grasp command (over-closed, same ramp and hold as a pick), read how far the soft fingers lag, reopen. The
        ratio error/travel is the empty-hand expectation the pick reads are compared against."""
        self._softcal_done = True
        g = self.cfg["grasp"]
        base = self.post[g.get("close_posture", "closed")]
        cls = soft_overclose(self.cfg, self.spec, base, self.post["preshape"]) if soft_hand_active(self.cfg) else base
        hv, dt = float(self.cfg["motion"]["hand_max_vel"]), 1.0 / HZ
        hold = max(float(self.cfg["motion"]["close_hold_s"]),
                   float(g["soft_lag_s"]) + float(g["soft_settle_s"]) + float(self.cfg["held"]["soft_read_after_s"]))
        up = ramp(self.hand_cmd, cls, hv, dt)
        hold_n = int(round(hold / dt))
        down = ramp(cls, self.post["open"], hv, dt)
        h = np.vstack([up, np.repeat(np.asarray(cls, float)[None], hold_n, 0), down])
        a = np.repeat(self.arm_cmd[None], len(h), 0)
        if self.queue_a is not None and len(self.queue_a):
            self.queue_a, self.queue_h = np.vstack([self.queue_a, a]), np.vstack([self.queue_h, h])
        else:
            self._enqueue(a, h)
        base_n = len(self.queue_a) - len(a)
        self._softcal = {"n": len(self.queue_a), "read": base_n + len(up) + hold_n, "t": t, "samples": []}
        self._note(t, f"soft baseline: empty close of both hands ({len(h) / HZ:.1f} s)")

    def _soft_baseline_step(self, obs):
        c = self._softcal
        q = obs.get("hand_q")
        left = 0 if self.queue_a is None else len(self.queue_a)
        done = c["n"] - left
        if q is not None and c["read"] - 12 <= done <= c["read"]:   # the last ~0.4 s of the empty hold
            c["samples"].append((self._held_command().copy(), np.asarray(q, float).copy()))
        if done > c["read"] or left == 0:
            self._softcal = None
            ok = {}
            for side in ("left", "right"):
                good = [(cm, m) for cm, m in c["samples"] if np.all(np.isfinite(m)) and m.shape == (14,)]
                ok[side] = self.held_detector.soft_calibrate(side, good)
            ratios = {s_: self.held_detector.soft_ratio(s_) for s_ in ("left", "right")}
            self.log(f"[soft_baseline] n={len(c['samples'])} ok={ok} empty-close error/travel "
                     f"left={ {k: round(v, 3) for k, v in ratios['left'].items()} } "
                     f"right={ {k: round(v, 3) for k, v in ratios['right'].items()} }")

    def _held_abort_due(self, g, hq, hc, margin=None):
        """grasp.held_detector=finger_sum only: the close sample is clearly empty. Unsure (no bias reference, missing
        or non-finite joint data) is never empty, so the placement runs as before."""
        det = self.held_detector
        if not det.ready or hq.shape != (14,) or hc.shape != (14,):
            return False
        if not (np.all(np.isfinite(hq)) and np.all(np.isfinite(hc))):
            return False
        if det.soft is not None:
            return not det.held(g["plan"].side, hc, hq, margin)[0]
        return not det.held(g["plan"].side, hc, hq)[0]

    def _abort_empty_grasp(self, g, sample="close"):
        """Cancel carry and request a proved open -> vertical lift -> home recovery in a worker."""
        plan = g["plan"]
        self._gdiag = None
        self._rel = None
        self._stall = self._exec = None      # per-plan watchers: their indices no longer match the queue
        if self._vary_on() and self.vary_hist:   # the entry _start appended for this plan
            self.vary_hist[-1]["held"] = False
        self.log(f"[held_abort] t={g['t']:.1f} {plan.color} {plan.side} target={plan.target_bin} "
                 f"{sample} sample empty: carry cancelled, proving vertical escape")
        self.queue_a = np.empty((0, 14)); self.queue_h = np.empty((0, 14))
        self._jobs.pop("empty_recovery", None)
        self._empty_recovery = {"arm": self.arm_cmd.copy(), "hand": self.hand_cmd.copy(), "side": plan.side,
                                "source_bin": plan.source_bin, "t": g["t"]}

    def _empty_recovery_step(self, t):
        """Poll only; route generation/IK/dry-run stays off the control loop."""
        rec = self._empty_recovery
        if "empty_recovery" not in self._jobs:
            r = self._bg("empty_recovery", self.planner.safe_home_route, rec["arm"], rec["hand"],
                         (rec["side"],), rec["source_bin"])
        else:
            r = self._bg("empty_recovery", self.planner.safe_home_route, None, None)
        if r is None:
            return self._hold()
        if r[0] == "err" or not r[1].ok:
            reason = f"{type(r[1]).__name__}: {r[1]}" if r[0] == "err" else r[1].reason
            self._empty_recovery = None
            self.state = "empty_hold"
            self._note(t, f"empty-grasp recovery unproved: {reason}; holding")
            return self._hold()
        route = r[1]
        self._empty_recovery = None
        self._enqueue(route.arm, route.hand)
        self.state = "executing"
        self._note(t, "empty-grasp recovery: open -> vertical lift -> checked home")
        return self._pop() if len(route.arm) else self._hold()

    def _release_confirm(self, obs, t):
        """grasp.release_confirm: at the end of the open hold stop the plan (retreat + home stashed) and poll the finger
        state for release_confirm_s. Not empty: wiggle open -> release once and poll again; still not empty: go home
        and stop picking (state done). Returns an action while it owns the tick, else None."""
        r = self._rel
        side, hq = r["plan"].side, obs.get("hand_q")
        if r["phase"] == "wait":
            if self.queue_a is None or self.queue_h is None or r["n"] - len(self.queue_a) < r["open_end"]:
                return None
            r.update(phase="check", rest=(self.queue_a, self.queue_h), t0=t)
            self.queue_a = self.queue_h = None
            self._exec = None     # ponytail: the T-8 s watcher is dropped once a release is being confirmed
        if r["phase"] == "reopen":
            if self.queue_a is not None and len(self.queue_a):
                return None       # the wiggle is still being popped
            r.update(phase="check", t0=t)
        cmd = self._held_command()
        ok = hq is not None and np.shape(hq) == (14,) and np.all(np.isfinite(hq)) and self.held_detector.empty(side, cmd, hq)
        if ok:
            self._rel = None
            self.queue_a, self.queue_h = r["rest"]
            self.log(f"[release_confirm] t={t:.1f} {r['plan'].color} {side}: hand empty after {t - r['t0']:.2f} s"
                     f"{' (re-opened)' if r['reopened'] else ''}")
            return None
        if t - r["t0"] < float(self.cfg["grasp"].get("release_confirm_s", 1.5)):
            return self._hold()
        arm = self.arm_cmd
        if not r["reopened"]:
            r["reopened"], r["phase"] = True, "reopen"
            sl = ArmKinematics.sl(side)   # only this hand wiggles; the other keeps its posture
            hv, h0 = self.cfg["motion"]["hand_max_vel"], self.hand_cmd.copy()
            wide, rel = h0.copy(), h0.copy()
            wide[sl] = self.post["open"][sl]
            rel[sl] = self.post[self.cfg["grasp"].get("release_posture", "preshape")][sl]
            wait = int(round(float(self.cfg["grasp"].get("release_wait_s") or 0.5) * HZ))
            h = np.vstack([ramp(h0, wide, hv, 1.0 / HZ), ramp(wide, rel, hv, 1.0 / HZ), np.repeat(rel[None], wait, 0)])
            self._enqueue(np.repeat(arm[None], len(h), 0), h)
            self.log(f"[release_confirm] t={t:.1f} {r['plan'].color} {side}: hand still not empty -> re-open once")
            return self._pop()
        self._rel, self._rel_fail = None, True
        home = arm.copy()
        home[ArmKinematics.sl(side)] = self.home[ArmKinematics.sl(side)]
        a, h = self.planner.go_to(arm, self.hand_cmd, home, self.post["open"])
        self._enqueue(a, h)
        self.log(f"[release_confirm] t={t:.1f} {r['plan'].color} {side}: still not empty after the re-open -> home, stop")
        return self._pop()

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
        self._last_action_steady = bool(steady)
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
        read_through = "carry" in g and self.held_detector.mode == "finger_sum"
        if q is not None:
            cmd = self._held_command()
            for key, at in (("close", g["close_end"]), ("lift", g["lift_end"])):
                if key not in g["samples"] and done >= at:
                    g["samples"][key] = (done, np.asarray(q, float).copy(), np.asarray(cmd, float).copy())
                    if key == "close" and not g.get("skill_only") \
                            and self._held_abort_due(g, np.asarray(q, float), np.asarray(cmd, float)):
                        self._abort_empty_grasp(g)   # queue replaced; _act pops it on this same tick
                        self._log_aborted_diag(g)
                        return
                    if key == "lift" and (read_through or g.get("lift_check")) and not g.get("skill_only") \
                            and self._held_abort_due(g, np.asarray(q, float), np.asarray(cmd, float),
                                                     self._soft_margin("lift")):
                        self._abort_empty_grasp(g, "post-lift")
                        self._log_aborted_diag(g)
                        return
            if read_through:
                carry_start, carry_end = g["carry"]
                if carry_start < done <= carry_end:
                    carry = g["samples"].setdefault("carry", [])
                    if not carry or carry[-1][0] != done:
                        carry.append((done, np.asarray(q, float).copy(), np.asarray(cmd, float).copy()))
        if read_through and left > 0 and done < g["carry"][1]:
            return
        need = ("close", "lift") if self.held_detector.mode == "finger_sum" else ("lift",)
        if any(key not in g["samples"] for key in need) and left > 0:
            return
        self._gdiag = None
        plan = g["plan"]
        held, parts = self._diag_body(g)
        pend = self._skill_pending if self._skills_on() else None   # flag off: the attribute is never read
        if pend is not None and pend["color"] == plan.color:
            pend["held"] = None if held is None else bool(held)
            if pend["hist"] is not None:
                pend["hist"]["held"] = pend["held"]
        if not g.get("skill_only"):
            self.log(f"[grasp_diag] t={g['t']:.1f} {plan.color} {plan.side} target={plan.target_bin} "
                     f"held={'n/a' if held is None else ('yes' if held else 'no')} " + " ".join(parts))

    def _diag_body(self, g):
        """Formats the [grasp_diag] fields for a watched pick: (headline held, field parts). Shared by the normal
        line and the aborted line; reads only, so it never changes a command or the detector state."""
        plan = g["plan"]
        read_through = "carry" in g and self.held_detector.mode == "finger_sum"
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
            if self.held_detector.soft is not None:
                held, scores = self.held_detector.held(plan.side, hc, hq, self._soft_margin(key))
            else:
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
                             f"threshold={self._thr(key):.3f} sample_held={'yes' if held else 'no'} "
                             f"stall_vs_closed="
                             f"{np.round(closed - hq[sl], 3).tolist()} cmd_vs_closed={np.round(closed - hc[sl], 3).tolist()}")
        if self.held_detector.mode == "finger_sum" and close_held is not None:
            held = close_held       # headline is the post-wait close sample; lift stays separately labelled above
        if read_through:
            post = ([g["samples"]["lift"]] if "lift" in g["samples"] else []) + list(g["samples"].get("carry", []))
            if post:
                held, scores = self.held_detector.held_max(plan.side, [(hc, hq) for _, hq, hc in post],
                                                           self._soft_margin("lift"))
                parts.append(f"carry_max@n={len(post)}: finger_sum={scores['finger_sum']:.3f} "
                             f"threshold={self._thr('lift'):.3f} "
                             f"sample_held={'yes' if held else 'no'}")
        return held, parts

    def _log_aborted_diag(self, g):
        """policy.grasp_diag_log: an aborted pick still gets its [grasp_diag] line (same formatter, aborted=1).
        Logging only; called after the abort has queued its recovery and never raises into the control path."""
        if g.get("skill_only"):
            return
        try:
            held, parts = self._diag_body(g)
            plan = g["plan"]
            self.log(f"[grasp_diag] t={g['t']:.1f} {plan.color} {plan.side} target={plan.target_bin} "
                     f"held={'n/a' if held is None else ('yes' if held else 'no')} aborted=1 " + " ".join(parts))
        except Exception as exc:   # diagnostics only
            self.log(f"[grasp_diag] aborted=1 format_error={type(exc).__name__}")

    def _settle_skill_attempt(self, dets, t):
        """The object still on the counter = that try failed; gone = it was held (or the release is judged later)."""
        if not self._skills_on() or self._skill_pending is None:
            return
        pending = self._skill_pending
        radius = self._assoc_radius()
        still = any(d.color == pending["color"] and np.linalg.norm(d.top[:2] - pending["xy"]) < radius for d in dets)
        if still:
            outcome = "dropped" if pending["held"] is True else "miss"
            self._finish_skill_attempt(outcome, t, held=False)
        else:
            self._finish_skill_attempt("held" if pending["held"] is not False else "miss", t,
                                       held=pending["held"] is not False)

    def _finish_skill_attempt(self, outcome, t=None, held=None):
        if not self._skills_on() or self._skill_pending is None:
            return
        pending = self._skill_pending
        if pending["hist"] is not None and held is not None:
            pending["hist"]["held"] = bool(held)
        duration = pending["duration_s"]
        if t is not None:
            duration = max(duration, max(0.0, float(t) - pending["started"]))
        self.skill_memory.record(pending["key"], pending["arm"], outcome, duration)
        self.log(f"[skills] key={pending['key']} arm={pending['arm']} outcome={outcome}")
        self._skill_pending = None

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
        if self._rel is not None and self._rel["phase"] == "wait":   # stop exactly at the end of the open hold
            k = max(1, min(k, self._rel["open_end"] - (self._rel["n"] - len(self.queue_a))))
        if self._stall is not None and self._in_hold():
            k = 1  # lift_on_stall: one tick per request inside the close hold, so every tick's hand_q is seen
        if self._gdiag is not None and "carry" in self._gdiag:
            done = self._gdiag["n"] - len(self.queue_a)
            # Stop exactly at lift end so the post-lift measurement is read before any carry target is emitted.
            for boundary in (self._gdiag["lift_end"], self._gdiag["carry"][1]):
                if done < boundary < done + k:
                    k = boundary - done
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
