"""World JSON, closed-set decision validator, EV rule, stub backend, async request and the BrainLayer.

Contract: at a decision point the code builds the closed list of allowed tool calls; the backend returns an index +
confidence; the validator and the EV rule decide whether that choice replaces what the classical planner would do.
Anything wrong (invalid, late, low confidence, error) -> the classical planner's own choice. Safety / IK / speed checks
stay in the classical planner (nothing here can move the arm)."""
from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
import urllib.error
import urllib.request

from .registry import BINS, POINTS, VARIANTS, TOOLS, allowed_calls, arg_spec, decision_gbnf, expand_macro

DEFAULTS = {"threshold": 0.8, "override_conf": 0.95, "timeout_s": 2.0, "max_steps": 32}
MAX_RESPONSE_BYTES = 64 * 1024


# ------------------------------------------------------------------ world
def world_to_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def build_world(dets, bins, instruction, task, resolver_mapping, resolver_bin_fn, reach_fn=None, near_bin_fn=None,
                ids=None, last_pick=None):
    """Detections -> world dict (ids o0.. in dets order). bins: [(name, xy, source)]; resolver_bin_fn(det) -> bin|None;
    reach_fn(det, bin) -> bool (planner/IK), near_bin_fn(det) -> bool; both optional (field None when absent).
    ids: stable object ids parallel to dets (default o0..); last_pick: {"id","color"} of the pick just executed."""
    def safe(fn, *a):
        try:
            return None if fn is None else bool(fn(*a))
        except Exception:
            return None

    objs = []
    for i, d in enumerate(dets):
        rb = resolver_bin_fn(d)
        objs.append({"id": ids[i] if ids else f"o{i}", "color": d.color, "shape": getattr(d, "shape", None),
                     "shape_conf": round(float(getattr(d, "shape_conf", 0.0) or 0.0), 3),
                     "xy": [round(float(d.top[0]), 4), round(float(d.top[1]), 4)],
                     "reachable": safe(reach_fn, d, rb) if rb is not None else None,
                     "near_bin": safe(near_bin_fn, d), "resolver_bin": rb})
    return {"task": task, "instruction": instruction or "", "objects": objs, "frame": "pelvis",
            "last_pick": last_pick,
            "bins": [{"name": n, "xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)], "source": s}
                     for n, xy, s in bins],
            "resolver_mapping": {("/".join(k) if isinstance(k, tuple) else str(k)): v
                                 for k, v in (resolver_mapping or {}).items()}}


# ------------------------------------------------------------------ validator
def nothing_abstained(point, world, state):
    """True at after_look when every candidate object already has a parser-decided bin (resolver_bin not None):
    the brain may assign bins only to objects the deterministic mapping abstained on, so there is nothing to ask."""
    if point != "after_look":
        return False
    objs = {o.get("id"): o for o in (world or {}).get("objects", [])}
    cands = [objs[c] for c in (state or {}).get("candidates", []) if c in objs]
    if not cands or any("resolver_bin" not in o for o in cands):
        return False
    return all(o["resolver_bin"] is not None for o in cands)


def _same(a, b):
    return a is not None and b is not None and a["tool"] == b["tool"] and a.get("args", {}) == b.get("args", {})


def validate_decision(decision, choices, world, cfg=None, reach_fn=None, n_done=0):
    """-> (ok, call, confidence, [errors]). decision: dict or JSON {"choice": i, "confidence": c}.
    Checks: JSON shape, index in range, confidence 0..1, tool on the whitelist, argument types, obj_id in the world,
    bin in {left, center, right}, step limit (n_done decisions already taken), and for pick_place the planner/IK
    reachability via reach_fn(obj_id, bin) -> (ok, reason)."""
    c = {**DEFAULTS, **(cfg or {})}
    if isinstance(decision, (str, bytes)):
        try:
            decision = json.loads(decision)
        except ValueError as e:
            return False, None, 0.0, [f"not JSON: {e}"]
    if not isinstance(decision, dict) or set(decision) != {"choice", "confidence"}:
        return False, None, 0.0, ["decision must be exactly {choice, confidence}"]
    i, conf = decision["choice"], decision["confidence"]
    if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(choices):
        return False, None, 0.0, [f"choice {i!r} outside 0..{len(choices) - 1}"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or math.isnan(conf) or not 0 <= conf <= 1:
        return False, None, 0.0, ["confidence must be a number in 0..1"]
    if n_done >= int(c["max_steps"]):
        return False, None, 0.0, [f"step limit {c['max_steps']} reached"]
    call = {"tool": choices[i].get("tool"), "args": choices[i].get("args", {})}
    ids, err = {o["id"] for o in (world or {}).get("objects", [])}, []
    lp = (world or {}).get("last_pick")
    if isinstance(lp, dict) and lp.get("id"):
        ids.add(lp["id"])                  # after_place talks about the object that just left the counter
    if call["tool"] not in TOOLS:
        return False, None, 0.0, [f"tool {call['tool']!r} not on the whitelist"]
    spec, args = arg_spec(call["tool"]), call["args"]
    if not isinstance(args, dict) or set(args) != set(spec):
        return False, None, 0.0, [f"args {args!r} != {sorted(spec)}"]
    for k, t in spec.items():
        v = args[k]
        if t == "bin" and v not in BINS:
            err.append(f"bin {v!r} not in {BINS}")
        elif t == "variant" and v not in VARIANTS:
            err.append(f"variant {v!r} not in {VARIANTS}")
        elif t == "query" and v not in (world or {}).get("queries", []):
            err.append(f"query {v!r} not offered")
        elif t == "obj_id" and (not isinstance(v, str) or v not in ids):
            err.append(f"obj_id {v!r} not in the world")
        elif t == "str" and (not isinstance(v, str) or len(v) > 80):
            err.append(f"{k} must be a string <= 80 chars")
    if not err and call["tool"] == "pick_place" and reach_fn is not None:
        ok, why = reach_fn(args["obj_id"], args["bin"])
        if not ok:
            err.append(f"{args['obj_id']} -> {args['bin']} not reachable ({why})")
    return (not err), (None if err else call), float(conf), err


def state_key(point, world, choices, classical, abstain=False):
    """Hash of the decision state: point, objects (id, colour, shape, xy quantised to 2 cm, classical bin), instruction,
    task, the closed choice set and the classical call. Equal key = same question -> a prefetched answer is valid."""
    objs = sorted([o["id"], o["color"], o.get("shape"), round(o.get("xy", [0, 0])[0] / 0.02),
                   round(o.get("xy", [0, 0])[1] / 0.02), o.get("resolver_bin")] for o in world["objects"])
    blob = json.dumps([point, objs, world.get("instruction"), world.get("task"),
                       [[c["tool"], c["args"]] for c in choices], classical, bool(abstain)], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ EV rule
def ev_decide(chosen, classical, conf, threshold=0.8, override_conf=0.95):
    """chosen / classical: {"tool","args"}; classical None = the classical planner has no answer.
    Goes ahead only if conf >= threshold AND (classical has no answer, or agrees, or conf >= override_conf).
    -> (use_brain: bool, reason)."""
    if conf < threshold:
        return False, f"conf {conf:.2f} < {threshold}"
    if chosen["tool"] == "find":      # information only (moves nothing, the next answer is gated again): threshold suffices
        return True, "find: information only"
    if classical is None:
        return True, "classical planner has no answer"
    if _same(chosen, classical):
        return True, "agrees with the classical planner"
    if conf >= override_conf:
        return True, f"override (conf {conf:.2f} >= {override_conf})"
    return False, f"disagrees, conf {conf:.2f} < {override_conf}"


# ------------------------------------------------------------------ backends
class PlannerBackend:
    """decide(request_json) -> decision_json. request = {"point", "world", "choices": [{i, tool, args}], "classical":
    index the classical planner would take (a hint only the stub reads; strip it from an LLM prompt)}.
    Return {"choice": i, "confidence": c}; a real model uses llama.cpp with decision_gbnf(len(choices)).
    Runs in a background thread; may be slow or raise: the 30 Hz loop never waits for it."""

    def decide(self, request_json):  # pragma: no cover
        raise NotImplementedError


class StubPlanner(PlannerBackend):
    """No model: picks the classical planner's own choice, confidence 1.0 -> behaviour identical to the brain off."""

    def decide(self, request_json):
        r = json.loads(request_json)
        return json.dumps({"choice": int(r["classical"]), "confidence": 1.0})


class LlamaCppHTTPBackend(PlannerBackend):
    """llama.cpp server /completion with a GBNF grammar (index enum + confidence), temperature 0, ~24 tokens.
    Optional request["image"] = base64 JPEG of the head RGB (multimodal image_data). Responses have an absolute
    deadline and a size cap. Returns the generated text; validation happens in BrainLayer (validate_decision)."""

    def __init__(self, url="http://127.0.0.1:8080/completion", timeout_s=1.5, n_predict=32, chatml=True,
                 max_response_bytes=MAX_RESPONSE_BYTES):
        self.base = url.rstrip("/")
        if self.base.endswith("/completion"):
            self.base = self.base[:-len("/completion")]
        self.timeout_s, self.n_predict, self.chatml = float(timeout_s), int(n_predict), bool(chatml)
        self.max_response_bytes = int(max_response_bytes)

    @staticmethod
    def _read_bounded(resp, max_bytes, deadline, clock=time.monotonic):
        """Read at most max_bytes before an absolute deadline, including slow-trickle responses."""
        size = resp.headers.get("Content-Length")
        if size is not None and int(size) > max_bytes:
            raise ValueError(f"response body exceeds {max_bytes} bytes")
        sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
        read = getattr(resp, "read1", resp.read)
        chunks, total = [], 0
        while True:
            left = deadline - clock()
            if left <= 0:
                raise TimeoutError("response exceeded absolute deadline")
            if sock is not None:
                try:
                    sock.settimeout(left)
                except OSError:       # urllib may already have buffered the complete body and closed the socket
                    sock = None
            chunk = read(min(8192, max_bytes + 1 - total))
            if clock() > deadline:
                raise TimeoutError("response exceeded absolute deadline")
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"response body exceeds {max_bytes} bytes")

    def healthy(self, timeout=0.5):
        deadline = time.monotonic() + float(timeout)
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=timeout) as r:
                if r.status != 200:
                    return False
                body = self._read_bounded(r, 4096, deadline) or b"{}"
                try:
                    return json.loads(body).get("status", "ok") == "ok"
                except json.JSONDecodeError:
                    return True
        except Exception:
            return False

    @staticmethod
    def prompt(req, with_image=False):
        w = {k: v for k, v in req["world"].items() if k != "frame"}
        lines = [f'{c["i"]}: {c["tool"]}({", ".join(f"{k}={v}" for k, v in c["args"].items())})' for c in req["choices"]]
        return (("[img-10]\n" if with_image else "") +
                "You control a robot sorting coloured objects into bins. World state (JSON):\n" +
                json.dumps(w, sort_keys=True, separators=(",", ":")) + f"\nDecision point: {req['point']}.\n"
                "Allowed tool calls:\n" + "\n".join(lines) +
                '\nAnswer with the index of the best call and your confidence, as {"choice": <index>, "confidence": <0..1>}.\n')

    def decide(self, request_json):
        deadline = time.monotonic() + self.timeout_s
        req = json.loads(request_json)
        img = req.get("image")
        prompt = self.prompt(req, bool(img))
        if self.chatml:
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        body = {"prompt": prompt, "grammar": decision_gbnf(len(req["choices"])),
                "temperature": 0, "n_predict": self.n_predict, "cache_prompt": True}
        if img:
            body["image_data"] = [{"data": img, "id": 10}]
        r = urllib.request.Request(self.base + "/completion", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=self.timeout_s) as resp:
            return str(json.loads(self._read_bounded(resp, self.max_response_bytes, deadline))["content"]).strip()


def make_backend(name, cfg=None, log=None):
    """brain.backend: stub | llamacpp. HTTP availability is checked only by asynchronous decisions, so policy
    construction can never be held up by a health endpoint; request failures fall back to the classical planner."""
    cfg, log = cfg or {}, log or (lambda *a: None)
    if name == "stub":
        return StubPlanner()
    if name == "llamacpp":
        return LlamaCppHTTPBackend(cfg.get("url", "http://127.0.0.1:8080/completion"),
                                   cfg.get("request_timeout_s", 1.5), cfg.get("n_predict", 32),
                                   cfg.get("chatml", True), cfg.get("max_response_bytes", MAX_RESPONSE_BYTES))
    raise ValueError(f"unknown brain.backend {name!r} (stub | llamacpp)")


class BackendWorker:
    """One daemon worker and at most one in-flight backend call. Extra submissions fail closed instead of spawning
    threads behind a stuck backend."""

    def __init__(self, backend):
        self.backend, self._busy = backend, False
        self._lock, self._q = threading.Lock(), queue.Queue(maxsize=1)
        threading.Thread(target=self._run, daemon=True, name="brain-backend").start()

    def submit(self, request_json, done):
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self._q.put_nowait((request_json, done))
        return True

    def _run(self):
        while True:
            request_json, done = self._q.get()
            try:
                res, bad = self.backend.decide(request_json), False
            except Exception as e:
                res, bad = repr(e), True
            with self._lock:
                self._busy = False
            done(res, bad)


class AsyncRequest:
    """backend.decide on a shared bounded worker. poll() never blocks: None until a result arrived within timeout_s
    of submit; a late / failed result is dropped for good (status "timeout" / "error")."""

    def __init__(self, backend, timeout_s=2.0, clock=time.perf_counter, worker=None):
        self.backend, self.timeout_s, self.clock = backend, float(timeout_s), clock
        self.worker = worker or BackendWorker(backend)
        self.status, self._gen = "idle", 0

    def reset(self):
        self._gen += 1
        self.status, self._res, self._tfin, self._ev = "idle", None, None, threading.Event()

    def submit(self, request_json):
        self.reset()
        gen, ev, self.status, self._t0 = self._gen, self._ev, "pending", self.clock()

        def done(res, bad):
            if gen == self._gen:
                self._res, self._tfin, self._bad = res, self.clock(), bad
                ev.set()

        if not self.worker.submit(request_json, done):
            self.status = "busy"
            ev.set()

    def poll(self):
        if self.status == "ready":
            return self._res
        if self.status != "pending":
            return None
        if self._ev.is_set():
            if self._bad:
                self.status = "error"
            elif self._tfin - self._t0 > self.timeout_s:
                self.status = "timeout"
            else:
                self.status = "ready"
                return self._res
        elif self.clock() - self._t0 > self.timeout_s:
            self.status = "timeout"
        return None

    def wait(self, seconds):
        """Tests / equivalence tool ONLY (never the control loop)."""
        if self.status == "pending":
            self._ev.wait(seconds)
        return self.poll()


# ------------------------------------------------------------------ layer
class BrainLayer:
    """One decision at a time: request(point, world, state, classical_call) then take(). The policy always has its
    own classical call ready; take() returns the brain's call only if a validated, EV-approved decision arrived."""

    def __init__(self, cfg, backend=None, log=None, clock=time.perf_counter):
        self.c = {**DEFAULTS, **(cfg or {})}
        self.log = log or (lambda *a: None)
        self.backend = backend or make_backend(self.c.get("backend", "stub"), self.c, self.log)
        self.worker = BackendWorker(self.backend)
        self.req = AsyncRequest(self.backend, self.c["timeout_s"], clock, self.worker)
        self.verbose = not isinstance(self.backend, StubPlanner)    # per-decision log lines only with a real model
        self.reset()

    def median_ms(self):
        l = sorted(self.lat)
        return 1000.0 * l[len(l) // 2] if l else 0.0

    def _note(self, src, why):
        if self.verbose:
            self.log(f"[brain] {self.point} src={src} why={why}")

    def reset(self):
        if getattr(self, "verbose", False) and any(getattr(self, "stats", {}).values()):
            self.log(f"[brain] episode stats {self.stats} lat_ms_median={self.median_ms():.0f}")
        self.req.reset()
        for r in getattr(self, "pre", {}).values():
            r.reset()
        self.pre = {}                       # state key -> AsyncRequest (prefetched)
        self.n_done, self.trace, self.lat = 0, [], []
        self.stats = {"brain": 0, "classical": 0, "pending": 0, "invalid": 0, "bin_reject": 0,
                      "reach_reject": 0, "no_abstain": 0, "timeout": 0, "error": 0,
                      "prefetch_hit": 0, "prefetch_miss": 0}
        self.choices = self.world = self.classical = self.point = None
        self.abstain = False
        self.skipped = False

    def _prepare(self, point, world, state, classical_call, image=None, abstain=False):
        choices = allowed_calls(point, state)
        ci = next((c["i"] for c in choices if _same(c, classical_call)), None)
        payload = {"point": point, "world": world, "choices": choices, "classical": ci if ci is not None else 0}
        if image:
            payload["image"] = image
        return choices, state_key(point, world, choices, classical_call, abstain), world_to_json(payload)

    def prefetch(self, point, world, state, classical_call, image=None, abstain=False):
        """Ask NOW for a decision that will be needed later (the predicted state). Async, never blocks. The answer
        is only used if request() later finds the same state key; otherwise it is dropped."""
        if nothing_abstained(point, world, state):
            return None
        _, key, rj = self._prepare(point, world, state, classical_call, image, abstain)
        if key in self.pre:
            return key
        for k in list(self.pre)[:max(0, len(self.pre) - 7)]:       # keep the newest 8
            self.pre.pop(k).reset()
        r = AsyncRequest(self.backend, self.c["timeout_s"], self.req.clock, self.worker)
        r.submit(rj)
        self.pre[key] = r
        return key

    def request(self, point, world, state, classical_call, image=None, abstain=False):
        """The decision point: reuse a prefetched answer for the same state key, else ask now (async). Returns choices."""
        self.point, self.world, self.classical, self.abstain = point, world, classical_call, abstain
        self.skipped = False
        if nothing_abstained(point, world, state):
            # abstain-only rule: the parser decided every candidate, so there is no bin the brain may assign
            self.skipped = True
            self.req.reset()
            self.choices = []
            return self.choices
        self.choices, key, rj = self._prepare(point, world, state, classical_call, image, abstain)
        hit = self.pre.pop(key, None)
        if hit is not None:
            self.stats["prefetch_hit"] += 1
            self.req.reset()
            self.req = hit
        else:
            if self.pre:
                self.stats["prefetch_miss"] += 1
            self.req.reset()
            self.req = AsyncRequest(self.backend, self.c["timeout_s"], self.req.clock, self.worker)
            self.req.submit(rj)
        return self.choices

    def wait(self, seconds):
        return self.req.wait(seconds)

    def take(self, reach_fn=None):
        """-> call dict the brain wants executed instead of the classical one, or None (= classical decides).
        Never blocks, never raises. reach_fn(obj_id, bin) -> (ok, reason) is applied only to an override."""
        try:
            if self.skipped:
                self.stats["no_abstain"] = self.stats.get("no_abstain", 0) + 1
                self._note("classical", "nothing abstained: brain not asked")
                return None
            raw = self.req.poll()
            if raw is not None and getattr(self.req, "_tfin", None) is not None:
                self.lat.append(self.req._tfin - self.req._t0)
            if raw is None:
                self._note("classical", self.req.status)
                self.stats[self.req.status if self.req.status in ("timeout", "error") else "pending"] += 1
                return None
            ok, call, conf, errs = validate_decision(raw, self.choices, self.world, self.c, None, self.n_done)
            if not ok:
                self.stats["invalid"] += 1
                self.log(f"[brain] decision rejected: {errs[:2]}")
                self._note("classical", "invalid")
                return None
            guard_classical = None if self.abstain else self.classical
            if call["tool"] == "pick_place":
                obj = next((o for o in self.world.get("objects", []) if o.get("id") == call["args"]["obj_id"]), {})
                if "resolver_bin" in obj:
                    known_bin = obj["resolver_bin"]
                    if known_bin is not None and call["args"]["bin"] != known_bin:
                        why = f"known bin for {call['args']['obj_id']} is {known_bin}"
                        self.log(f"[brain_guard] override dropped: {call['args']['obj_id']} brain={call['args']['bin']} "
                                 f"parser={known_bin}")
                        self.stats["bin_reject"] += 1
                        self.trace.append((self.point, call, "bin_reject", why))
                        self._note("classical", why)
                        return None
                    if known_bin is None:       # per-object abstention: the LLM may fill only this missing opinion
                        guard_classical = None
            use, why = ev_decide(call, guard_classical, conf, self.c["threshold"], self.c["override_conf"])
            if not use:
                self.stats["classical"] += 1
                self.trace.append((self.point, call, "classical", why))
                self._note("classical", why)
                return None
            if call["tool"] == "pick_place" and not _same(call, self.classical) and reach_fn is not None:
                ok2, r = reach_fn(call["args"]["obj_id"], call["args"]["bin"])
                if not ok2:
                    self.stats["reach_reject"] += 1
                    self.trace.append((self.point, call, "reach_reject", r))
                    self._note("classical", "reach_reject")
                    return None
            self.n_done += 1
            self.stats["brain"] += 1
            self._note("llm", f"{'agree' if _same(call, self.classical) else 'override'}: {why} conf={conf:.2f} "
                              f"lat_ms={1000 * (self.req._tfin - self.req._t0):.0f}")
            if not _same(call, self.classical):
                self.log(f"[brain] {self.point}: {call} instead of {self.classical} ({why})")
                self.trace.append((self.point, call, "override", why))
            return call
        except Exception as e:
            self.log(f"[brain] take failed ({e!r}), classical decides")
            return None

