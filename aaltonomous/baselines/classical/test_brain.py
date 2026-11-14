"""Brain layer tests. Pure-stdlib part runs anywhere; the equivalence part needs numpy (box C).
    python3 -m pytest aaltonomous/baselines/classical/test_brain.py"""
from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

import http.server
import threading

from aaltonomous.baselines.classical.brain import (BrainLayer, LlamaCppHTTPBackend, PlannerBackend, StubPlanner,
                                                  allowed_calls, decision_gbnf, decision_json_schema, ev_decide,
                                                  expand_macro, make_backend, validate_decision)

WORLD = {"objects": [{"id": "o0", "color": "red"}, {"id": "o1", "color": "blue"}]}
STATE = {"candidates": ["o0", "o1"]}
CH = allowed_calls("after_look", STATE)


def idx(tool, **args):
    return next(c["i"] for c in CH if c["tool"] == tool and c["args"] == args)


def dec(i, c=0.9):
    return {"choice": i, "confidence": c}


class Det:
    def __init__(self, color, x, y):
        self.color, self.top, self.shape, self.shape_conf = color, (x, y, 0.0), None, 0.0


class TestAllowedCalls(unittest.TestCase):
    def test_closed_set_shapes(self):
        self.assertEqual(len(CH), 2 * 4 + 2)                       # 3 bins + skip per object, look, done
        self.assertEqual([c["i"] for c in CH], list(range(len(CH))))
        f = allowed_calls("after_fail", {"obj_id": "o0"})
        self.assertEqual({c["tool"] for c in f}, {"retry", "skip", "look", "done"})
        p = allowed_calls("after_place", {"obj_id": "o0", "can_look": False})
        self.assertEqual([c["tool"] for c in p], ["verify", "done"])

    def test_done_not_offered_while_work_remains(self):
        ch = allowed_calls("after_place", {"obj_id": "o0", "allow_done": False})
        self.assertNotIn("done", [c["tool"] for c in ch])

    def test_grammar_and_schema_are_index_enums(self):
        g = decision_gbnf(CH.__len__())
        self.assertIn('idx ::= "0" | "1"', g)
        self.assertIn(f'"{len(CH) - 1}"', g)
        self.assertNotIn(f'"{len(CH)}"', g)
        self.assertEqual(decision_json_schema(len(CH))["properties"]["choice"]["maximum"], len(CH) - 1)

    def test_macro_expands_to_tools_only(self):
        out = expand_macro({"tool": "pick_and_verify", "args": {"obj_id": "o0", "bin": "left"}})
        self.assertEqual([s["tool"] for s in out], ["pick_place", "verify"])
        self.assertEqual(out[1]["args"], {"obj_id": "o0"})


class TestValidator(unittest.TestCase):
    def v(self, d, ch=CH, **kw):
        return validate_decision(d, ch, WORLD, **kw)

    def test_valid(self):
        ok, call, conf, err = self.v(dec(idx("pick_place", obj_id="o0", bin="left")))
        self.assertTrue(ok, err)
        self.assertEqual(call["args"], {"obj_id": "o0", "bin": "left"})
        self.assertTrue(self.v(json.dumps(dec(0)))[0])           # JSON string accepted

    def test_invalid_shapes(self):
        for bad in ("nope", {"choice": 0}, {"choice": 99, "confidence": .9}, {"choice": -1, "confidence": .9},
                    {"choice": True, "confidence": .9}, {"choice": 0, "confidence": 1.5},
                    {"choice": 0, "confidence": "hi"}, {"choice": 0, "confidence": .9, "x": 1}, [1, 2]):
            self.assertFalse(self.v(bad)[0], bad)

    def test_tool_args_ids_bins(self):
        for ch in ([{"i": 0, "tool": "rm_rf", "args": {}}],
                   [{"i": 0, "tool": "pick_place", "args": {"obj_id": "o0"}}],
                   [{"i": 0, "tool": "pick_place", "args": {"obj_id": "o9", "bin": "left"}}],
                   [{"i": 0, "tool": "pick_place", "args": {"obj_id": "o0", "bin": "up"}}],
                   [{"i": 0, "tool": "retry", "args": {"obj_id": "o0", "variant": "fast"}}]):
            self.assertFalse(self.v(dec(0), ch)[0], ch)

    def test_step_limit(self):
        self.assertFalse(self.v(dec(0), cfg={"max_steps": 3}, n_done=3)[0])
        self.assertTrue(self.v(dec(0), cfg={"max_steps": 3}, n_done=2)[0])

    def test_reachability_check(self):
        i = idx("pick_place", obj_id="o0", bin="left")
        ok, _, _, err = self.v(dec(i), reach_fn=lambda o, b: (False, "ik"))
        self.assertFalse(ok)
        self.assertIn("not reachable", err[0])
        self.assertTrue(self.v(dec(i), reach_fn=lambda o, b: (True, ""))[0])


class TestEV(unittest.TestCase):
    A = {"tool": "pick_place", "args": {"obj_id": "o0", "bin": "left"}}
    B = {"tool": "pick_place", "args": {"obj_id": "o0", "bin": "right"}}

    def test_rule(self):
        self.assertFalse(ev_decide(self.A, self.A, 0.79)[0])      # below threshold even when it agrees
        self.assertTrue(ev_decide(self.A, self.A, 0.8)[0])        # agrees
        self.assertFalse(ev_decide(self.B, self.A, 0.9)[0])       # disagrees, < 0.95 -> classical
        self.assertTrue(ev_decide(self.B, self.A, 0.95)[0])       # disagrees, >= 0.95 -> override
        self.assertTrue(ev_decide(self.B, None, 0.8)[0])          # classical has no answer
        self.assertFalse(ev_decide(self.B, None, 0.5)[0])         # ... but still needs the threshold
        self.assertTrue(ev_decide(self.B, self.A, 0.6, threshold=0.5, override_conf=0.6)[0])   # configurable


class Scripted(PlannerBackend):
    def __init__(self, out=None, delay=0.0, exc=False):
        self.out, self.delay, self.exc = out, delay, exc

    def decide(self, request_json):
        time.sleep(self.delay)
        if self.exc:
            raise RuntimeError("boom")
        return json.dumps(self.out(json.loads(request_json))) if callable(self.out) else self.out


class TestLayer(unittest.TestCase):
    CL = {"tool": "pick_place", "args": {"obj_id": "o0", "bin": "left"}}

    def layer(self, backend, **cfg):
        return BrainLayer({"timeout_s": 0.3, **cfg}, backend=backend)

    def ask(self, L, wait=1.0):
        L.request("after_look", WORLD, STATE, self.CL)
        L.wait(wait)
        return L.take()

    def test_stub_returns_classical_choice(self):
        L = self.layer(StubPlanner())
        self.assertEqual(self.ask(L), self.CL)
        self.assertEqual(L.stats["brain"], 1)
        self.assertEqual(L.trace, [])                             # no override: identical behaviour

    def test_override_needs_high_confidence(self):
        want = {"choice": idx("pick_place", obj_id="o0", bin="right"), "confidence": 0.9}
        L = self.layer(Scripted(want))
        self.assertIsNone(self.ask(L))
        self.assertEqual(L.stats["classical"], 1)
        L = self.layer(Scripted({**want, "confidence": 0.97}))
        self.assertEqual(self.ask(L)["args"]["bin"], "right")
        L = self.layer(Scripted({**want, "confidence": 0.97}))
        L.request("after_look", WORLD, STATE, self.CL)
        L.wait(1.0)
        self.assertIsNone(L.take(lambda o, b: (False, "ik")))     # unreachable override -> classical
        self.assertEqual(L.stats["reach_reject"], 1)

    def test_invalid_decision_falls_back(self):
        L = self.layer(Scripted("garbage"))
        self.assertIsNone(self.ask(L))
        self.assertEqual(L.stats["invalid"], 1)

    def test_backend_error_falls_back(self):
        L = self.layer(Scripted(exc=True))
        self.assertIsNone(self.ask(L))

    def test_timeout_never_blocks_and_late_answer_is_dropped(self):
        L = self.layer(Scripted({"choice": 0, "confidence": 1.0}, delay=0.8), timeout_s=0.3)
        t0 = time.perf_counter()
        L.request("after_look", WORLD, STATE, self.CL)
        self.assertIsNone(L.take())                               # pending: classical decides, instantly
        self.assertLess(time.perf_counter() - t0, 0.1)
        self.assertEqual(L.stats["pending"], 1)
        time.sleep(0.5)
        self.assertIsNone(L.take())
        self.assertEqual(L.stats["timeout"], 1)
        time.sleep(0.6)                                           # the backend finished late: still dropped
        self.assertIsNone(L.take())
        self.assertEqual(L.req.status, "timeout")

    def test_abstain_only_guard(self):
        """Bug 1: brain may bin only objects the parser abstained on; every dropped override is logged."""
        logs = []
        known = {**WORLD, "objects": [{**WORLD["objects"][0], "resolver_bin": "left"},
                                      {**WORLD["objects"][1], "resolver_bin": None}]}
        # (1) different bin for a parser-mapped object -> ignored and logged
        L = self.layer(Scripted({"choice": idx("pick_place", obj_id="o0", bin="right"), "confidence": 1.0}))
        L.log = logs.append
        L.request("after_look", known, STATE, self.CL, abstain=False)
        L.wait(1.0)
        self.assertIsNone(L.take())
        self.assertTrue(any("[brain_guard] override dropped" in m for m in logs), logs)
        # (2) bin for an abstained object -> used
        L = self.layer(Scripted({"choice": idx("pick_place", obj_id="o1", bin="right"), "confidence": 0.9}))
        L.request("after_look", known, STATE, self.CL, abstain=False)
        L.wait(1.0)
        self.assertEqual(L.take()["args"], {"obj_id": "o1", "bin": "right"})

    def test_brain_not_asked_when_nothing_abstained(self):
        class Count(PlannerBackend):
            n = 0

            def decide(self, rj):
                Count.n += 1
                return json.dumps({"choice": 0, "confidence": 1.0})

        allknown = {**WORLD, "objects": [{**WORLD["objects"][0], "resolver_bin": "left"},
                                         {**WORLD["objects"][1], "resolver_bin": "right"}]}
        L = self.layer(Count())
        L.request("after_look", allknown, STATE, self.CL, abstain=False)
        L.prefetch("after_look", allknown, STATE, self.CL, None, False)
        L.wait(0.2)
        self.assertIsNone(L.take())
        time.sleep(0.2)
        self.assertEqual(Count.n, 0)
        self.assertEqual(L.stats["no_abstain"], 1)

    def test_known_bin_cannot_be_overridden_but_unknown_object_can(self):
        known = {**WORLD, "objects": [{**WORLD["objects"][0], "resolver_bin": "left"},
                                      {**WORLD["objects"][1], "resolver_bin": None}]}
        wrong = {"choice": idx("pick_place", obj_id="o0", bin="right"), "confidence": 1.0}
        L = self.layer(Scripted(wrong))
        L.request("after_look", known, STATE, self.CL, abstain=True)
        L.wait(1.0)
        self.assertIsNone(L.take())
        self.assertEqual(L.stats["bin_reject"], 1)

        unknown = {"choice": idx("pick_place", obj_id="o1", bin="right"), "confidence": 0.85}
        L = self.layer(Scripted(unknown))
        L.request("after_look", known, STATE, self.CL, abstain=False)
        L.wait(1.0)
        self.assertEqual(L.take()["args"], {"obj_id": "o1", "bin": "right"})

    def test_shared_worker_allows_only_one_outstanding_call(self):
        class Blocking(PlannerBackend):
            def __init__(self):
                self.started, self.release, self.n = threading.Event(), threading.Event(), 0

            def decide(self, request_json):
                self.n += 1
                self.started.set()
                self.release.wait(1.0)
                return json.dumps({"choice": 0, "confidence": 1.0})

        backend = Blocking()
        L = self.layer(backend)
        L.request("after_look", WORLD, STATE, self.CL)
        self.assertTrue(backend.started.wait(0.2))
        for _ in range(20):
            L.request("after_look", WORLD, STATE, self.CL)
        self.assertEqual(backend.n, 1)
        self.assertEqual(L.req.status, "busy")
        backend.release.set()


class MockLlama:
    """Mock llama.cpp server. mode: ok | wrong (index out of range) | garbage | low (conf 0.3) | late | health503.
    `answer` = index returned in ok mode. Records the last /completion body."""

    def __init__(self, mode="ok", answer=0, delay=1.0):
        self.mode, self.answer, self.delay, self.bodies, self.gets = mode, answer, delay, [], 0
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                try:
                    self.wfile.write(b)
                except OSError:
                    pass

            def do_GET(self):
                outer.gets += 1
                self._send(503 if outer.mode == "health503" else 200, {"status": "ok"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                if outer.mode == "trickle":
                    payload = json.dumps({"content": json.dumps({"choice": outer.answer,
                                                                  "confidence": 0.99})}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    try:
                        for byte in payload:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(0.03)
                    except OSError:
                        pass
                    return
                if outer.mode == "late":
                    time.sleep(outer.delay)
                txt = {"wrong": '{"choice": 999, "confidence": 0.99}', "garbage": "I think option two.",
                       "low": json.dumps({"choice": outer.answer, "confidence": 0.3}),
                       "giant": "x" * 10000}.get(
                    outer.mode, json.dumps({"choice": outer.answer, "confidence": 0.99}))
                self._send(200, {"content": txt})

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/completion"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class TestLlamaCpp(unittest.TestCase):
    CL = {"tool": "pick_place", "args": {"obj_id": "o0", "bin": "left"}}
    RIGHT = idx("pick_place", obj_id="o0", bin="right")

    def run_mode(self, mode, answer=None, **bk):
        srv = MockLlama(mode, self.RIGHT if answer is None else answer)
        self.addCleanup(srv.close)
        logs = []
        L = BrainLayer({"backend": "llamacpp", "url": srv.url, "timeout_s": 1.0, "request_timeout_s": 0.4, **bk},
                       log=logs.append)
        L.request("after_look", WORLD, STATE, self.CL)
        L.wait(1.5)
        return L, L.take(), srv, logs

    def test_correct_answer_is_used_and_request_is_constrained(self):
        L, call, srv, _ = self.run_mode("ok")
        self.assertEqual(call["args"]["bin"], "right")                  # conf 0.99 >= 0.95 override
        b = srv.bodies[0]
        self.assertEqual(b["temperature"], 0)
        self.assertEqual(b["n_predict"], 32)
        self.assertTrue(b["prompt"].startswith("<|im_start|>user"))   # ChatML wrap (Qwen)
        self.assertIn("root ::=", b["grammar"])
        self.assertIn(f'"{len(CH) - 1}"', b["grammar"])
        self.assertNotIn("classical", b["prompt"])                       # the hint never reaches the model

    def test_agreeing_answer_equals_classical(self):
        _, call, _, _ = self.run_mode("ok", answer=idx("pick_place", obj_id="o0", bin="left"))
        self.assertEqual(call, self.CL)

    def test_wrong_garbage_low_late_all_fall_back(self):
        for mode, key in (("wrong", "invalid"), ("garbage", "invalid"), ("low", "classical"), ("late", "error")):
            L, call, _, _ = self.run_mode(mode)
            self.assertIsNone(call, mode)
            self.assertEqual(L.stats[key], 1, (mode, L.stats))

    def test_late_answer_dropped_even_if_it_arrives(self):
        srv = MockLlama("late", self.RIGHT, delay=0.7)
        self.addCleanup(srv.close)
        L = BrainLayer({"backend": "llamacpp", "url": srv.url, "timeout_s": 0.3, "request_timeout_s": 3.0})
        t0 = time.perf_counter()
        L.request("after_look", WORLD, STATE, self.CL)
        self.assertIsNone(L.take())
        self.assertLess(time.perf_counter() - t0, 0.1)                   # never waits
        time.sleep(1.0)
        self.assertIsNone(L.take())
        self.assertEqual(L.stats["timeout"], 1)

    def test_no_server_fails_closed_asynchronously(self):
        srv = MockLlama("ok")
        url = srv.url
        srv.close()                                                      # nothing listens there any more
        L = BrainLayer({"backend": "llamacpp", "url": url, "request_timeout_s": 0.2})
        self.assertIsInstance(L.backend, LlamaCppHTTPBackend)
        L.request("after_look", WORLD, STATE, self.CL)
        L.wait(1.0)
        self.assertIsNone(L.take())                                      # None = classical policy decides

    def test_backend_construction_never_probes_health(self):
        srv = MockLlama("health503")
        self.addCleanup(srv.close)
        self.assertIsInstance(make_backend("llamacpp", {"url": srv.url}), LlamaCppHTTPBackend)
        self.assertEqual(srv.gets, 0)
        healthy = MockLlama("ok")
        self.addCleanup(healthy.close)
        self.assertTrue(LlamaCppHTTPBackend(healthy.url).healthy())

    def test_response_body_is_capped(self):
        srv = MockLlama("giant")
        self.addCleanup(srv.close)
        b = LlamaCppHTTPBackend(srv.url, timeout_s=1.0, max_response_bytes=128)
        request = json.dumps({"world": WORLD, "choices": CH, "point": "after_look"})
        with self.assertRaisesRegex(ValueError, "exceeds 128 bytes"):
            b.decide(request)

    def test_slow_trickle_obeys_absolute_deadline(self):
        srv = MockLlama("trickle", self.RIGHT)
        self.addCleanup(srv.close)
        b = LlamaCppHTTPBackend(srv.url, timeout_s=0.15)
        request = json.dumps({"world": WORLD, "choices": CH, "point": "after_look"})
        started = time.monotonic()
        with self.assertRaises((TimeoutError, OSError)):
            b.decide(request)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_server_dies_mid_episode(self):
        srv = MockLlama("ok", self.RIGHT)
        L = BrainLayer({"backend": "llamacpp", "url": srv.url, "timeout_s": 1.0, "request_timeout_s": 0.3})
        srv.close()
        L.request("after_look", WORLD, STATE, self.CL)
        L.wait(1.0)
        self.assertIsNone(L.take())                                      # connection refused -> classical

    def test_image_is_sent_as_image_data(self):
        srv = MockLlama("ok", self.RIGHT)
        self.addCleanup(srv.close)
        L = BrainLayer({"backend": "llamacpp", "url": srv.url, "timeout_s": 1.0})
        L.request("after_look", WORLD, STATE, self.CL, image="QUJD")
        L.wait(1.0)
        self.assertEqual(srv.bodies[0]["image_data"], [{"data": "QUJD", "id": 10}])
        self.assertIn("[img-10]", srv.bodies[0]["prompt"])


class Counting(PlannerBackend):
    def __init__(self):
        self.n = 0

    def decide(self, request_json):
        self.n += 1
        return json.dumps({"choice": json.loads(request_json)["classical"], "confidence": 1.0})


class TestPrefetch(unittest.TestCase):
    CL = {"tool": "pick_place", "args": {"obj_id": "o0", "bin": "left"}}

    def w(self, x=0.30):
        return {"task": "t", "instruction": "i", "last_pick": None, "bins": [], "resolver_mapping": {},
                "objects": [{"id": "o0", "color": "red", "shape": None, "shape_conf": 0, "xy": [x, 0.1],
                             "reachable": None, "near_bin": None, "resolver_bin": "left"},
                            {"id": "o1", "color": "blue", "shape": None, "shape_conf": 0, "xy": [0.2, -0.1],
                             "reachable": None, "near_bin": None, "resolver_bin": None}]}   # o1 abstained: the brain is asked

    def test_hit_uses_prefetched_answer_without_new_request(self):
        b = Counting()
        L = BrainLayer({"timeout_s": 1.0}, backend=b)
        L.prefetch("after_look", self.w(), STATE, self.CL)
        L.wait(0.0)
        time.sleep(0.1)
        L.request("after_look", self.w(0.301), STATE, self.CL)           # same 2 cm cell -> same state
        self.assertEqual(L.take(), self.CL)
        self.assertEqual((b.n, L.stats["prefetch_hit"]), (1, 1))

    def test_changed_state_discards_prefetch_and_classical_decides(self):
        b = Counting()
        L = BrainLayer({"timeout_s": 1.0}, backend=b)
        L.prefetch("after_look", self.w(), STATE, self.CL)
        time.sleep(0.1)
        L.request("after_look", self.w(0.40), STATE, self.CL)            # object moved 10 cm: different state
        time.sleep(0.1)
        self.assertEqual((L.stats["prefetch_hit"], L.stats["prefetch_miss"]), (0, 1))
        self.assertEqual(b.n, 2)

    def test_different_classical_or_choices_is_a_miss(self):
        L = BrainLayer({"timeout_s": 1.0}, backend=Counting())
        L.prefetch("after_look", self.w(), STATE, self.CL)
        time.sleep(0.1)
        other = {"tool": "pick_place", "args": {"obj_id": "o1", "bin": "right"}}
        L.request("after_look", self.w(), STATE, other)
        self.assertEqual(L.stats["prefetch_hit"], 0)


OPEN_VOCAB = "The object farthest from the robot goes in the right bin."


class TestResolverAbstain(unittest.TestCase):
    """The finalized parser mapping is shared by the classical planner and brain guard; None lets the model decide."""
    CL = {"tool": "pick_place", "args": {"obj_id": "o0", "bin": "left"}}     # the planner's FIXED default
    RIGHT = {"choice": idx("pick_place", obj_id="o0", bin="right"), "confidence": 0.99}

    def test_resolver_returns_empty_on_open_vocab_with_fallback(self):
        from aaltonomous.baselines.classical.instruction import resolve
        from aaltonomous.interface.tasks import FIXED_MAPPING
        self.assertEqual(resolve(OPEN_VOCAB, fallback={}), {})
        self.assertEqual(resolve(OPEN_VOCAB), dict(FIXED_MAPPING))          # the classical path is unchanged

    def test_fixed_and_explicit_mapping_are_authoritative_without_text(self):
        try:
            from aaltonomous.baselines.classical.policy import ClassicalPolicy
            from aaltonomous.interface.tasks import FIXED_MAPPING
        except ImportError as e:
            self.skipTest(str(e))
        pol = ClassicalPolicy.__new__(ClassicalPolicy)
        pol.brain = object()
        pol.cfg = {"instruction": {"resolve": True}}
        pol.mapping = dict(FIXED_MAPPING)
        mapping, source = pol._resolver_mapping({"task": "sort_fixed", "instruction": ""})
        self.assertEqual((mapping, source), (dict(FIXED_MAPPING), "fixed"))
        pol.mapping = {"red": "right"}       # reset() finalizes an explicit mapping before building the brain guard
        mapping, source = pol._resolver_mapping({"instruction": OPEN_VOCAB, "mapping": {"red": "right"}})
        self.assertEqual((mapping, source), ({"red": "right"}, "explicit"))

    def test_layer_abstain_lets_the_model_through(self):
        for conf, passes in ((0.99, True), (0.85, True), (0.7, False)):
            L = BrainLayer({"timeout_s": 1.0}, backend=Scripted({**self.RIGHT, "confidence": conf}))
            L.request("after_look", WORLD, STATE, self.CL, abstain=True)
            L.wait(1.0)
            call = L.take()
            self.assertEqual(call is not None and call["args"]["bin"] == "right", passes, conf)
        L = BrainLayer({"timeout_s": 1.0}, backend=Scripted({**self.RIGHT, "confidence": 0.85}))
        L.request("after_look", WORLD, STATE, self.CL, abstain=False)       # resolver has an answer: 0.85 < 0.95
        L.wait(1.0)
        self.assertIsNone(L.take())

    def test_policy_open_vocab_sentence_llm_choice_goes_through(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not available")
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        from aaltonomous.baselines.classical.policy import ClassicalPolicy

        class Right(PlannerBackend):
            def decide(self, rj):
                r = json.loads(rj)
                if r["point"] != "after_look":
                    return json.dumps({"choice": r["classical"], "confidence": 1.0})
                i = next((c["i"] for c in r["choices"] if c["tool"] == "pick_place" and c["args"]["bin"] == "right"),
                         r["classical"])
                return json.dumps({"choice": i, "confidence": 0.99})

        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=120.0,
                              overrides={"brain.enabled": True, "brain.debug_wait": True, "brain.prefetch": False,
                                         "instruction.resolve": True})
        pol.brain = BrainLayer(pol.cfg["brain"], backend=Right())
        obs = env.reset()
        pol.reset({"instruction": OPEN_VOCAB, "time_limit": 120.0})
        self.assertTrue(pol._brain_abstain)
        n = 0
        while n < 3000:
            a = pol.act(obs)
            n += 1
            obs, info = env.step(a)
            if pol.plans or info["episode_over"]:
                break
        self.assertTrue(any(t[0] == "after_look" and t[2] == "override" and t[1]["args"]["bin"] == "right"
                            for t in pol.brain.trace), pol.brain.trace)
        self.assertEqual(pol.plans[0].target_bin, "right")                   # and the classical planner executed it


class TestFind(unittest.TestCase):
    def test_extract_queries(self):
        from aaltonomous.baselines.classical.brain.find import extract_queries, match_found
        self.assertEqual(extract_queries(OPEN_VOCAB), ["The object farthest from the robot"])
        q = extract_queries("Put the red cube in the left bin and the blue cylinders in the right bin.")
        self.assertEqual(q[:2], ["the red cube", "the blue cylinders"])
        self.assertIn("the orange cube", q)                       # alias variant offered
        self.assertEqual(extract_queries(""), [])
        d = type("D", (), {"uv": (100.0, 100.0)})
        self.assertEqual(match_found([(110, 95), (400, 400)], [d]), [0])

    def test_find_only_offered_when_queries_given_and_ev_is_threshold_only(self):
        self.assertFalse(any(c["tool"] == "find" for c in CH))
        ch = allowed_calls("after_look", {"candidates": ["o0"], "queries": ["the red cube"]})
        self.assertEqual(sum(c["tool"] == "find" for c in ch), 1)
        f = {"tool": "find", "args": {"query": "the red cube"}}
        cl = {"tool": "pick_place", "args": {"obj_id": "o0", "bin": "left"}}
        self.assertTrue(ev_decide(f, cl, 0.8)[0])
        self.assertFalse(ev_decide(f, cl, 0.79)[0])
        w = {**WORLD, "queries": ["the red cube"]}
        i = next(c["i"] for c in ch if c["tool"] == "find")
        self.assertTrue(validate_decision(dec(i), ch, w)[0])
        self.assertFalse(validate_decision(dec(i), ch, WORLD)[0])  # query not offered in this world

    def test_policy_find_then_pick(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not available")
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        from aaltonomous.baselines.classical.policy import ClassicalPolicy

        class FindThenRight(PlannerBackend):
            def decide(self, rj):
                r = json.loads(rj)
                if r["point"] != "after_look":
                    return json.dumps({"choice": r["classical"], "confidence": 1.0})
                found = r["world"].get("found")
                if found is None:
                    i = next((c["i"] for c in r["choices"] if c["tool"] == "find"), r["classical"])
                    return json.dumps({"choice": i, "confidence": 0.9})
                oid = next(iter(found.values()))[0]
                i = next(c["i"] for c in r["choices"] if c["tool"] == "pick_place" and c["args"] == {"obj_id": oid, "bin": "right"})
                return json.dumps({"choice": i, "confidence": 0.99})

        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=120.0,
                              overrides={"brain.enabled": True, "brain.debug_wait": True, "brain.prefetch": False,
                                         "brain.find": True, "instruction.resolve": True})
        pol.brain = BrainLayer(pol.cfg["brain"], backend=FindThenRight())
        calls = []
        pol._brain_find_fn = lambda rgb, q: (calls.append(q), [pol.last_detections[-1].uv if pol.last_detections else (0, 0)])[1]
        obs = env.reset()
        pol.reset({"instruction": OPEN_VOCAB, "time_limit": 120.0})
        n = 0
        while n < 3000:
            a = pol.act(obs)
            n += 1
            obs, info = env.step(a)
            if pol.plans or info["episode_over"]:
                break
        self.assertEqual(calls[:1], ["The object farthest from the robot"])
        self.assertTrue(any(t[0] == "after_look" and t[1]["tool"] == "pick_place" and t[1]["args"]["bin"] == "right"
                            for t in pol.brain.trace), pol.brain.trace)
        self.assertEqual(pol.plans[0].target_bin, "right")

    def test_default_off_no_queries(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not available")
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        from aaltonomous.baselines.classical.policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None, safety=make_mock_env(seed=0).filter.cfg, overrides={"brain.enabled": True})
        pol.reset({"instruction": OPEN_VOCAB})
        self.assertEqual(pol._brain_queries, [])


class TestSubmissionPackaging(unittest.TestCase):
    def test_submission_image_copies_brain_package(self):
        dockerfile = (Path(__file__).resolve().parents[3] / "tools" / "submission.Dockerfile").read_text()
        self.assertIn("COPY aaltonomous/baselines/classical/brain/ aaltonomous/baselines/classical/brain/", dockerfile)


class TestStubEquivalence(unittest.TestCase):
    def test_mock_episode_identical_with_stub(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not available")
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
        from equivalence_check import episode_digest
        ov = {"brain.enabled": True, "brain.backend": "stub", "brain.debug_wait": True}
        dead = {"brain.enabled": True, "brain.backend": "llamacpp", "brain.url": "http://127.0.0.1:9/completion",
                "brain.debug_wait": True}
        for seed in (0, 1):
            base = episode_digest(seed, 120.0)
            self.assertEqual(base, episode_digest(seed, 120.0, ov))      # stub (prefetch on)
            self.assertEqual(base, episode_digest(seed, 120.0, dead))    # llamacpp, no server -> stub

    def test_stub_prefetch_hits_and_after_place_is_wired(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not available")
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        from aaltonomous.baselines.classical.policy import ClassicalPolicy
        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=120.0,
                              overrides={"brain.enabled": True, "brain.debug_wait": True})
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"], "time_limit": 120.0})
        n = 0
        while n < 20000:
            a = pol.act(obs)
            n += 1
            obs, info = env.step(a)
            if info["episode_over"]:
                break
        self.assertGreater(pol.brain.stats["prefetch_hit"], 0)           # next decisions were ready in time
        self.assertEqual(pol.brain.trace, [])                            # stub never overrides

    def test_scripted_verify_after_place_makes_an_extra_look(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not available")
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        from aaltonomous.baselines.classical.policy import ClassicalPolicy

        class Verify(PlannerBackend):
            def decide(self, rj):
                r = json.loads(rj)
                if r["point"] == "after_place":
                    i = next(c["i"] for c in r["choices"] if c["tool"] == "verify")
                    return json.dumps({"choice": i, "confidence": 0.99})
                return json.dumps({"choice": r["classical"], "confidence": 1.0})

        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=120.0,
                              overrides={"brain.enabled": True, "brain.debug_wait": True, "brain.prefetch": False})
        pol.brain = BrainLayer(pol.cfg["brain"], backend=Verify())
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"], "time_limit": 120.0})
        n = 0
        while n < 20000:
            a = pol.act(obs)
            n += 1
            obs, info = env.step(a)
            if info["episode_over"]:
                break
        self.assertTrue(any(t[0] == "after_place" and t[2] == "override" for t in pol.brain.trace), pol.brain.trace)


if __name__ == "__main__":
    unittest.main()
