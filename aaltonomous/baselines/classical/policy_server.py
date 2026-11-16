"""Reference submission: the classical rung-1 sorter (ClassicalPolicy) behind the policy-server protocol.

    python -m aaltonomous.baselines.classical.policy_server            # :8000, what the Docker image runs
    docker build -f aaltonomous/docker/baseline.Dockerfile -t aalto/baseline-classical:v1 .

Same planner/perception as run.py / sim_eval.py (no forked logic). It runs threaded: while it looks and plans
(IK, ~0.1-few s) act() returns 1-tick holds at once, so it never blocks the runner (command rate stays ~30 Hz);
motion comes back in `motion.chunk`-tick chunks. It returns "done" when nothing sortable is left.
Uses only head_rgb, head_depth, arm_q, hand_q, t from obs; camera intrinsics/extrinsics come from the
calibration.yaml baked into the image (spec §11), so rebuild after a calibration change.
Rung 1 only: the colour -> bin mapping is scene.yaml's, whatever the instruction says.
Env: AALTO_TIME_LIMIT (s, default 120; must match the runner's --time-limit), AALTO_ARM_VEL (rad/s override),
AALTO_PROFILE (explicit opt-in; default `none` = robot base values, `sim` is only for non-finals simulation).
"""
from __future__ import annotations

import copy
import logging
import os
import time

import numpy as np

from aaltonomous.deploy.policy_server import Policy, serve

from .policy import ClassicalPolicy
from .profile_detect import detect_profile

log = logging.getLogger("aaltonomous.baseline")


class ClassicalServerPolicy(Policy):
    name = "classical-rung1"

    def __init__(self, time_limit=None, arm_vel=None, profile=None, overlay=None, **kw):
        tl = float(time_limit or os.environ.get("AALTO_TIME_LIMIT", 120.0))
        # This server is the real-robot entry point. Simulation remains available only by an explicit argument/env.
        explicit = profile is not None or bool(os.environ.get("AALTO_PROFILE"))
        effective_profile = profile if profile is not None else os.environ.get("AALTO_PROFILE", "none")
        effective_profile = effective_profile or "none"
        ov = {}
        av = arm_vel or os.environ.get("AALTO_ARM_VEL")
        if av:
            ov["motion.arm_max_vel"] = float(av)
        self._build_args = (tl, ov, overlay, kw)
        self._auto_buf, self._pending_reset, self.probe_cfg = None, None, None
        if not explicit:
            from .config import _get, load_config
            probe = load_config(profile="none", overlay=overlay)   # flags only: no profile applied yet
            force = probe["profile_auto"].get("force")
            if force in ("sim", "none"):   # PROFILE_MODE image: no detection, the decision is made at build time
                extra = None
                if force == "sim":
                    extra = dict(probe["profile_auto"]["sim_overrides"] or {})
                    for k in extra:
                        _get(probe, k)
                    log.info("[profile] forced=sim, sim_overrides applied: %s", sorted(extra))
                else:
                    log.info("[profile] forced=none (real robot)")
                self.pol = None
                self._build(force, refuse_finals_sim=False, extra_overrides=extra)
                return
            if probe["profile_auto"]["auto_detect"]:
                for k in (probe["profile_auto"]["sim_overrides"] or {}):
                    _get(probe, k)   # KeyError at startup for a typo, not mid-episode
                self.probe_cfg = probe
                self._auto_frames = max(1, int(probe["profile_auto"]["auto_frames"]))
                self._auto_buf = []
                self.pol = None
                self.action_horizon = int(probe["motion"]["chunk"])
                log.info("[profile] auto-detect pending: profile chosen from the first %d frames", self._auto_frames)
                return
        self._build(effective_profile, refuse_finals_sim=True)

    def _build(self, profile, refuse_finals_sim, extra_overrides=None):
        tl, ov, overlay, kw = self._build_args
        self.pol = ClassicalPolicy(threaded=True, require_confirm=False, time_limit=tl,
                                   overrides={**(ov or {}), **(extra_overrides or {})},
                                   profile=profile, overlay=overlay, log=lambda m: log.info(m), **kw)
        cfg = self.pol.cfg
        selected_overlay = str(cfg.get("overlay", "none"))
        if refuse_finals_sim and cfg.get("profile") == "sim" and (
                selected_overlay == "submission" or selected_overlay.startswith("finals")):
            raise RuntimeError(f"refusing finals overlay {selected_overlay!r} under profile 'sim'; use profile='none'")
        self.config_snapshot = copy.deepcopy(cfg)
        self.action_horizon = int(self.pol.cfg["motion"]["chunk"])
        mo = cfg["motion"]
        log.info("effective config profile=%s overlay=%s motion arm_vel=%.3f arm_acc=%.3f hand_vel=%.3f "
                 "hand_acc=%s", cfg.get("profile"), selected_overlay, float(mo["arm_max_vel"]),
                 float(mo["arm_max_acc"]), float(mo["hand_max_vel"]),
                 "n/a" if mo.get("hand_max_acc") is None else f"{float(mo['hand_max_acc']):.3f}")

    def reset(self, instruction, task=None, seed=None):
        if self._auto_buf is not None and self.pol is None:
            self._pending_reset = (instruction, task, seed)   # replayed once the profile is chosen
            return
        self._reset_now(instruction, task, seed)

    def _reset_now(self, instruction, task=None, seed=None):
        if not self.pol.cfg["policy"].get("generation_workers", False):
            # Legacy flag-off path: wait before resetting shared planner state.
            job, t0 = self.pol._job, time.monotonic()
            while job is not None and job["result"] is None and time.monotonic() - t0 < 30.0:
                time.sleep(0.02)
        if task not in (None, "sort_fixed"):
            log.warning("task %r: the classical baseline only does rung 1 (colour = bin from scene.yaml)", task)
        info = {"instruction": instruction, "task": task, "seed": seed}
        fixed_first = bool((self.pol.cfg.get("policy") or {}).get("fixed_instruction_first", False))
        if task == "sort_fixed" and not fixed_first:
            # The server reset is the episode contract: only the fixed task receives fixed mapping. The optional
            # finals venue order is explicit scene metadata in policy.fixed_mapping; language tasks never inherit it.
            info["mapping"] = dict(self.pol.fixed_mapping)
        # policy.fixed_instruction_first: no injected mapping; ClassicalPolicy._episode_mapping applies the instruction's
        # full mapping when it parses, else falls back to policy.fixed_mapping itself (sort_fixed only).
        self.pol.reset(info)   # task id: language/fixed/insert routing; seed: skills
        log.info("reset task=%s seed=%s instruction=%r", task, seed, instruction)

    def act(self, obs):
        if self._auto_buf is not None and self.pol is None:
            self._auto_buf.append(obs)
            if len(self._auto_buf) < self._auto_frames:
                return {"arm_q_target": np.zeros((0, 14)), "hand_q_target": np.zeros((0, 14)), "done": False}
            self._decide_profile(self._auto_buf)
            self._auto_buf = None
        a = self.pol.act(obs)
        out = {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
               "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}
        if a.get("arm_tau_ff") is not None:   # motion.arm_tau_ff (organiser PR #4); absent = flag off
            out["arm_tau_ff"] = np.asarray(a["arm_tau_ff"], np.float64)
        return out

    def _decide_profile(self, frames):
        pa = self.probe_cfg["profile_auto"]
        cue = bool(pa.get("require_rgb_cue", False))
        second = bool(pa.get("second_cue", False))
        try:
            kw = {}
            if cue:
                kw["require_rgb_cue"] = True
            if second:
                kw["second_cue"] = True
            profile, info = detect_profile(frames, **kw)
        except Exception:   # fail-safe: an undecidable detection is the REAL robot (keeps every real-only fix on)
            log.exception("[profile] detection raised: defaulting to none (real)")
            profile, info = "none", {"votes": 0, "frames": len(frames), "depth_holes": 1.0, "plane_std_mm": float("inf"),
                                     "palette": "error", "rgb_noise": [], "rgb_cue": cue}
        log.info("[profile] auto=%s votes=%d/%d (depth_holes=%.4f, plane_std_mm=%.2f, palette=%s, frames=%d, "
                 "rgb_noise=%s, rgb_cue=%s)", profile, info["votes"], info["frames"], info["depth_holes"], info["plane_std_mm"],
                 info["palette"], info["frames"], info.get("rgb_noise"), info.get("rgb_cue", False))
        if second:
            log.info("[profile] second_cue: cue_a(holes)_sim=%s cue_b(temporal) changed_frac=%s mean_abs_mm=%s pairs=%s "
                     "cue_b_sim=%s -> %s", info.get("cue_a_sim"), info.get("cue_b_changed_frac"),
                     info.get("cue_b_mean_abs_mm"), info.get("cue_b_pairs"), info.get("cue_b_sim"), profile)
        if info.get("disagree"):
            log.warning("[profile] depth says sim but rgb disagrees in %d/%d frames (require_rgb_cue): deciding %s",
                        info["disagree"], info["frames"], profile)
        extra = None
        if profile == "sim":
            extra = dict(self.probe_cfg["profile_auto"]["sim_overrides"] or {})
            log.info("[profile] sim_overrides applied: %s", sorted(extra))
        self._build(profile, refuse_finals_sim=False, extra_overrides=extra)   # sim from auto-detect may run finals overlays
        if self._pending_reset is not None:
            self._reset_now(*self._pending_reset)
            self._pending_reset = None


def verify_expected_flags(cfg):
    """Refuse to serve when the loaded flags differ from overlays/EXPECTED_FLAGS.json (written by promote.sh).
    No file = unchanged behaviour."""
    from .config import check_expected_flags
    diffs = check_expected_flags(cfg)
    if diffs:
        for d in diffs:
            log.critical("FLAG MISMATCH vs EXPECTED_FLAGS.json: %s", d)
        log.critical("effective config differs from the promoted overlay: refusing to start")
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = ClassicalServerPolicy(profile=os.environ.get("AALTO_PROFILE") or None)
    log.info("classical baseline loaded in %.1f s", time.monotonic() - t0)
    cfg = pol.pol.cfg if pol.pol is not None else pol.probe_cfg   # auto-detect: flags are profile-independent
    from .config import effective_flags  # integration: which opt-in features this image runs (container log)
    on = [f"{k}={v!r}" for k, v, is_on, _, _ in effective_flags(cfg) if is_on]
    log.info("profile=%s overlay=%s flags on: %s", cfg.get("profile"), cfg.get("overlay", "none"),
             ", ".join(on) or "none (stock baseline)")
    verify_expected_flags(cfg)
    serve(pol, port=int(os.environ.get("PORT", 8000)))
