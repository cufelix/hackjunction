"""Offline tests for the optional, drop-only voice event hook."""
from __future__ import annotations

import time
import unittest

import numpy as np

from .policy import ClassicalPolicy
from .voice_events import VoiceEventEmitter


class _Sink:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append({"event": event, **fields})
        return True


class _FullSocket:
    def setblocking(self, _value):
        pass

    def setsockopt(self, *_args):
        pass

    def sendto(self, *_args):
        raise BlockingIOError()

    def close(self):
        pass


def _stub_policy(sink=None, emit_each_act=False):
    policy = ClassicalPolicy.__new__(ClassicalPolicy)
    policy.cfg = {"safety": {"nan_guard": False}}
    policy._voice = sink
    policy._voice_start_pending = False
    policy.fk_xcheck_enabled = False
    policy._shell = False
    action = {"arm_q_target": np.zeros((1, 14)), "hand_q_target": np.zeros((1, 14)), "done": False}
    if emit_each_act:
        policy._act = lambda _obs: (policy._voice_emit("picked", colour="yellow", shape="cube"), action)[1]
    else:
        policy._act = lambda _obs: action
    return policy, action


class TestVoiceHook(unittest.TestCase):
    def test_phase_mapping_emits_committed_pick_then_placement(self):
        policy = ClassicalPolicy.__new__(ClassicalPolicy)
        policy._voice = _Sink()
        policy._voice_plan = {"n": 12, "picked_at": 4, "placed_at": 8, "picked": False, "placed": False,
                              "pick_valid": False, "colour": "yellow", "shape": "cube", "bin": "left"}
        policy._voice_placed = 0
        policy.scene = type("Scene", (), {"bin_names": ("left", "center", "right")})()
        policy.queue_a = np.zeros((8, 14))
        policy._voice_pick_result(True)
        policy._voice_progress()
        policy.queue_a = np.zeros((4, 14))
        policy._voice_progress()
        policy._voice_progress()  # phase boundary is emitted once
        self.assertEqual(policy._voice.events, [
            {"event": "picked", "colour": "yellow", "shape": "cube"},
            {"event": "placed", "colour": "yellow", "shape": "cube", "bin": "left"},
        ])
        self.assertEqual(policy._voice_placed, 1)

    def test_full_transport_never_blocks_or_raises(self):
        emitter = VoiceEventEmitter(sock=_FullSocket())
        started = time.perf_counter()
        for _ in range(10000):
            self.assertFalse(emitter.emit("picked", colour="red", shape="cube"))
        self.assertLess(time.perf_counter() - started, 0.5)

    def test_missing_sidecar_changes_no_action_and_adds_under_point_two_ms(self):
        baseline, expected = _stub_policy()
        emitter = VoiceEventEmitter(port=45999)  # deliberately no listener
        voiced, _ = _stub_policy(emitter, emit_each_act=True)
        obs = {"t": 0.0}
        for _ in range(100):
            baseline.act(obs); voiced.act(obs)
        n = 5000
        start = time.perf_counter()
        for _ in range(n):
            out0 = baseline.act(obs)
        base_s = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(n):
            out1 = voiced.act(obs)
        voice_s = time.perf_counter() - start
        np.testing.assert_array_equal(out0["arm_q_target"], expected["arm_q_target"])
        np.testing.assert_array_equal(out1["arm_q_target"], expected["arm_q_target"])
        self.assertEqual(out0["done"], out1["done"])
        overhead_s = max(0.0, (voice_s - base_s) / n)
        self.assertLess(overhead_s, 0.0002, f"voice overhead {overhead_s * 1e3:.3f} ms/act")
        emitter.close()


if __name__ == "__main__":
    unittest.main()
