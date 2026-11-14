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

log = logging.getLogger("aaltonomous.baseline")


class ClassicalServerPolicy(Policy):
    name = "classical-rung1"

    def __init__(self, time_limit=None, arm_vel=None, profile=None, overlay=None, **kw):
        tl = float(time_limit or os.environ.get("AALTO_TIME_LIMIT", 120.0))
        # This server is the real-robot entry point. Simulation remains available only by an explicit argument/env.
        effective_profile = profile if profile is not None else os.environ.get("AALTO_PROFILE", "none")
        effective_profile = effective_profile or "none"
        ov = {}
        av = arm_vel or os.environ.get("AALTO_ARM_VEL")
        if av:
            ov["motion.arm_max_vel"] = float(av)
        self.pol = ClassicalPolicy(threaded=True, require_confirm=False, time_limit=tl, overrides=ov,
                                   profile=effective_profile, overlay=overlay, log=lambda m: log.info(m), **kw)
        cfg = self.pol.cfg
        selected_overlay = str(cfg.get("overlay", "none"))
        if cfg.get("profile") == "sim" and (selected_overlay == "submission" or selected_overlay.startswith("finals")):
            raise RuntimeError(f"refusing finals overlay {selected_overlay!r} under profile 'sim'; use profile='none'")
        self.config_snapshot = copy.deepcopy(cfg)
        self.action_horizon = int(self.pol.cfg["motion"]["chunk"])
        mo = cfg["motion"]
        log.info("effective config profile=%s overlay=%s motion arm_vel=%.3f arm_acc=%.3f hand_vel=%.3f "
                 "hand_acc=%s", cfg.get("profile"), selected_overlay, float(mo["arm_max_vel"]),
                 float(mo["arm_max_acc"]), float(mo["hand_max_vel"]),
                 "n/a" if mo.get("hand_max_acc") is None else f"{float(mo['hand_max_acc']):.3f}")

    def reset(self, instruction, task=None, seed=None):
        if not self.pol.cfg["policy"].get("generation_workers", False):
            # Legacy flag-off path: wait before resetting shared planner state.
            job, t0 = self.pol._job, time.monotonic()
            while job is not None and job["result"] is None and time.monotonic() - t0 < 30.0:
                time.sleep(0.02)
        if task not in (None, "sort_fixed"):
            log.warning("task %r: the classical baseline only does rung 1 (colour = bin from scene.yaml)", task)
        info = {"instruction": instruction, "task": task, "seed": seed}
        if task == "sort_fixed":
            # The server reset is the episode contract: only the fixed task receives fixed mapping. The optional
            # finals venue order is explicit scene metadata in policy.fixed_mapping; language tasks never inherit it.
            info["mapping"] = dict(self.pol.fixed_mapping)
        self.pol.reset(info)   # task id: language/fixed/insert routing; seed: skills
        log.info("reset task=%s seed=%s instruction=%r", task, seed, instruction)

    def act(self, obs):
        a = self.pol.act(obs)
        return {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
                "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}


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
    pol = ClassicalServerPolicy(profile=os.environ.get("AALTO_PROFILE", "none"))
    log.info("classical baseline loaded in %.1f s", time.monotonic() - t0)
    from .config import effective_flags  # integration: which opt-in features this image runs (container log)
    on = [f"{k}={v!r}" for k, v, is_on, _, _ in effective_flags(pol.pol.cfg) if is_on]
    log.info("profile=%s overlay=%s flags on: %s", pol.pol.cfg.get("profile"), pol.pol.cfg.get("overlay", "none"),
             ", ".join(on) or "none (stock baseline)")
    verify_expected_flags(pol.pol.cfg)
    serve(pol, port=int(os.environ.get("PORT", 8000)))
