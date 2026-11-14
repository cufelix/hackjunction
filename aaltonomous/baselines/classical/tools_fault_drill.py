"""Offline fault drill for ClassicalPolicy and the ray-cast mock world.

The supervisor in this test tool never forwards a known-bad observation to the
policy.  It emits a one-tick hold instead; production policy behaviour is not
changed.  Run the finals overlays (when present) with::

    python -m aaltonomous.baselines.classical.tools_fault_drill
    python -m aaltonomous.baselines.classical.tools_fault_drill --overlay my_overlay
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from aaltonomous.interface.safety import JointSpec

from .config import OVERLAY_DIR, load_config
from .mock_world import Renderer, make_mock_env
from .policy import ClassicalPolicy, HZ

FAULTS = ("depth_zero", "camera_frozen", "bin_missing", "instruction_bad", "joint_nan", "stall_50ms")


@dataclass
class Result:
    overlay: str
    fault: str
    no_exception: bool = True
    p99_ms: float = math.inf
    limits: bool = True
    jumps: bool = True
    safe: bool = True
    detail: str = ""

    @property
    def passed(self):
        return self.no_exception and self.p99_ms < 33.0 and self.limits and self.jumps and self.safe


class ActionAudit:
    """Check every raw policy/supervisor target, including chunk boundaries."""

    def __init__(self, cfg, obs):
        self.cfg, self.spec = cfg, JointSpec()
        self.arm = np.asarray(obs["arm_q"], float).copy()
        self.hand = np.asarray(obs["hand_q"], float).copy()
        self.limits = self.jumps = True

    @staticmethod
    def _rows(action, key):
        q = np.asarray(action.get(key), float)
        return q[None] if q.ndim == 1 else q

    def check(self, action):
        arm, hand = self._rows(action, "arm_q_target"), self._rows(action, "hand_q_target")
        if arm.ndim != 2 or hand.ndim != 2 or arm.shape[1:] != (14,) or hand.shape != arm.shape:
            self.limits = self.jumps = False
            return
        finite = np.isfinite(arm).all() and np.isfinite(hand).all()
        self.limits &= bool(finite and np.all((arm >= self.spec.arm_lower) & (arm <= self.spec.arm_upper)) and
                            np.all((hand >= self.spec.hand_lower) & (hand <= self.spec.hand_upper)))
        if not finite:
            self.jumps = False
            return
        da = np.abs(np.diff(np.vstack([self.arm, arm]), axis=0)).max(initial=0.0)
        dh = np.abs(np.diff(np.vstack([self.hand, hand]), axis=0)).max(initial=0.0)
        self.jumps &= bool(da <= self.cfg["motion"]["arm_max_vel"] / HZ + 1e-8 and
                           dh <= self.cfg["motion"]["hand_max_vel"] / HZ + 1e-8)
        self.arm, self.hand = arm[-1].copy(), hand[-1].copy()

    def hold(self, done=False):
        return {"arm_q_target": self.arm[None].copy(), "hand_q_target": self.hand[None].copy(), "done": done}

    def is_hold(self, action):
        return (np.array_equal(self._rows(action, "arm_q_target"), self.arm[None]) and
                np.array_equal(self._rows(action, "hand_q_target"), self.hand[None]))


class FaultSupervisor:
    """Test-only interlock: validate an injected fault payload, then hold."""

    @staticmethod
    def present(fault, obs, reference, meta):
        if fault == "depth_zero":
            return np.count_nonzero(obs["head_depth"]) == 0
        if fault == "camera_frozen":
            return (meta["seq"] == meta["frozen_seq"] and
                    np.array_equal(obs["head_rgb"], reference["head_rgb"]) and
                    np.array_equal(obs["head_depth"], reference["head_depth"]))
        if fault == "bin_missing":
            changed = np.count_nonzero(obs["head_rgb"] != reference["head_rgb"])
            return meta["missing"] in meta["expected_bins"] and changed > 100
        if fault == "instruction_bad":
            return not str(obs.get("instruction") or "").strip()
        if fault == "joint_nan":
            return not np.isfinite(obs["arm_q"]).all()
        if fault == "stall_50ms":
            return meta["elapsed_ms"] >= 50.0 and meta["held_ticks"] >= 2 and meta["late_discarded"]
        return False

    def act(self, fault, obs, reference, meta, audit):
        present = self.present(fault, obs, reference, meta)
        action = audit.hold()
        return action, bool(present and audit.is_hold(action))


def _missing_bin_frame(env):
    scene = copy.deepcopy(env.scene)
    missing = scene.bin_names[0]
    del scene.cfg["bins"]["centers"][missing]
    renderer = Renderer(scene, env.backend.renderer.cal)
    return missing, renderer.render(env.backend.objects)


def _fault_obs(fault, obs, env, frozen=None):
    bad = dict(obs)
    detail = ""
    if fault == "depth_zero":
        bad["head_depth"] = np.zeros_like(obs["head_depth"])
    elif fault == "camera_frozen":
        bad["head_rgb"], bad["head_depth"] = frozen
    elif fault == "bin_missing":
        missing, (bad["head_rgb"], bad["head_depth"]) = _missing_bin_frame(env)
        detail = f"missing={missing}"
    elif fault == "instruction_bad":
        bad["instruction"] = ""
    elif fault == "joint_nan":
        bad["arm_q"] = np.asarray(obs["arm_q"], float).copy()
        bad["arm_q"][0] = np.nan
    return bad, detail


def run_episode(fault, overlay="none", seed=0, warmup_calls=8):
    """Run one short episode. Known injected faults are supervised to hold/skip."""
    name = str(overlay or "none")
    result = Result(name, fault)
    latencies = []
    env = None
    try:
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}")
        cfg = load_config(overlay=overlay)
        result.overlay = cfg.get("overlay", "none")
        env = make_mock_env(seed=seed, cfg=cfg, realtime=False)
        policy = ClassicalPolicy(config=cfg, safety=env.filter.cfg, threaded=True, time_limit=20.0, log=None)
        obs = env.reset()
        valid_instruction = obs["instruction"]
        policy.reset({"instruction": valid_instruction, "time_limit": 20.0})
        audit = ActionAudit(cfg, obs)

        for _ in range(warmup_calls):
            started = time.perf_counter_ns()
            action = policy.act(obs)
            latencies.append((time.perf_counter_ns() - started) / 1e6)
            audit.check(action)
            obs, info = env.step(action)
            if info["episode_over"]:
                raise RuntimeError(f"episode ended before injection: {info['end_reason']}")

        frozen = (obs["head_rgb"].copy(), obs["head_depth"].copy())
        bad, detail = _fault_obs(fault, obs, env, frozen)
        ticks = int(3 * HZ) if fault in ("depth_zero", "camera_frozen") else 1
        if fault == "instruction_bad":
            policy.reset({"instruction": bad["instruction"], "time_limit": 20.0})
        supervisor = FaultSupervisor()
        meta = {}
        if fault == "camera_frozen":
            bad["_camera_seq"] = 0
            meta = {"seq": 0, "frozen_seq": 0}
        elif fault == "bin_missing":
            meta = {"missing": detail.split("=", 1)[1], "expected_bins": env.scene.bin_names}
        if fault == "stall_50ms":
            # Delay the policy worker by a real 50 ms. The independent runner
            # keeps ticking the existing command and discards the late result.
            late = {}

            def delayed_act():
                try:
                    time.sleep(0.050)
                    started = time.perf_counter_ns()
                    late["action"] = policy.act(bad)
                    late["latency"] = (time.perf_counter_ns() - started) / 1e6
                except Exception as exc:  # re-raise on the harness thread
                    late["error"] = exc

            started = time.perf_counter()
            worker = threading.Thread(target=delayed_act)
            worker.start()
            held_ticks = 0
            deadline = started + 0.250
            while worker.is_alive() and time.perf_counter() < deadline:
                env.hold()
                held_ticks += 1
                worker.join(timeout=max(0.0, min(1.0 / HZ, deadline - time.perf_counter())))
            if worker.is_alive():
                raise TimeoutError("stalled policy worker exceeded 250 ms")
            if "error" in late:
                raise late["error"]
            latencies.append(late["latency"])
            old_arm, old_hand = audit.arm.copy(), audit.hand.copy()
            audit.check(late["action"])       # every output is checked, even though this stale one is discarded
            audit.arm, audit.hand = old_arm, old_hand
            elapsed = (time.perf_counter() - started) * 1000.0
            meta = {"elapsed_ms": elapsed, "held_ticks": held_ticks, "late_discarded": True}
            detail = f"stall={elapsed:.1f}ms held_ticks={held_ticks} late_discarded=yes"

        reference = {"head_rgb": frozen[0], "head_depth": frozen[1]}
        for tick in range(ticks):
            # The supervisor consumes and validates the corrupt observation on
            # every fault tick; it cannot pass merely because a fault was named.
            action, safe = supervisor.act(fault, bad, reference, meta, audit)
            result.safe &= safe
            audit.check(action)
            obs, info = env.step(action)

        # A discarded late result has advanced the stateful policy. Reset and
        # explicitly synchronize it to the last command before resuming.
        if fault in ("instruction_bad", "stall_50ms"):
            policy.reset({"instruction": valid_instruction, "time_limit": 20.0})
            policy.arm_cmd, policy.hand_cmd = audit.arm.copy(), audit.hand.copy()
        started = time.perf_counter_ns()
        action = policy.act(obs)
        latencies.append((time.perf_counter_ns() - started) / 1e6)
        audit.check(action)
        obs, info = env.step(action)
        if info["episode_over"]:
            raise RuntimeError(f"episode ended during recovery: {info['end_reason']}")
        action = audit.hold(done=True)
        audit.check(action)
        _, info = env.step(action)
        result.safe &= info["end_reason"] == "done"
        result.limits, result.jumps = audit.limits, audit.jumps
        result.p99_ms = float(np.percentile(latencies, 99)) if latencies else math.inf
        result.detail = detail or f"fault_ticks={ticks}"
    except Exception as exc:
        result.no_exception = False
        result.detail = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
    return result


def run_suite(overlays):
    return [run_episode(fault, overlay, seed=i) for overlay in overlays for i, fault in enumerate(FAULTS)]


def print_table(rows):
    headers = ("overlay", "fault", "exc", "p99 ms", "limits", "jumps", "safe", "result", "detail")
    data = [headers] + [(r.overlay, r.fault, "OK" if r.no_exception else "FAIL", f"{r.p99_ms:.2f}",
                         "OK" if r.limits else "FAIL", "OK" if r.jumps else "FAIL",
                         "OK" if r.safe else "FAIL", "PASS" if r.passed else "FAIL", r.detail) for r in rows]
    widths = [max(len(str(row[i])) for row in data) for i in range(len(headers))]
    for n, row in enumerate(data):
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
        if n == 0:
            print("  ".join("-" * w for w in widths))


def _default_overlays():
    names = [n for n in ("cand_r16_nopp", "finals_real") if (OVERLAY_DIR / f"{n}.yaml").exists()]
    return names or ["none"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", action="append", help="overlay name/path; repeat to test several")
    args = parser.parse_args(argv)
    rows = run_suite(args.overlay or _default_overlays())
    print_table(rows)
    passed = sum(r.passed for r in rows)
    print(f"\n{passed}/{len(rows)} fault episodes passed")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
