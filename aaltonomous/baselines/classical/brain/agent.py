"""brain.closed_loop: the agent loop around PlanLayer.

PlanLayer plans once (plus one re-plan). AgentLoop keeps the plan that is running and, after every primitive outcome
(ok / failed / skipped / found), asks the backend again, asynchronously, with the fresh world JSON and the history of
executed steps + results, at most `closed_loop_max_replans` times. The executor never waits: until the new answer is
validated it keeps running the last valid plan; a timeout / malformed / invalid answer leaves that plan (or the
classical resolver when there is none) in charge.

Retry / abandon is deterministic code, not a model call: after `max_grasp_fails` failed grasps of one object the loop
inserts the same pick with a DIFFERENT grasp variant (skill-memory order when a chooser is given), then gives the
object up (skip: 0 points, never a -1 guess).

Nothing here can move the robot: the output is data, every step passes validate_plan (closed tool set, bin authority,
reach / place dry runs, no-early-done), and the policy dry-runs every motion with the classical planner."""
from __future__ import annotations

import json
import time

from .core import AsyncRequest, world_to_json
from .planner import PlanLayer, validate_plan
from .registry import VARIANTS

HISTORY_LEN = 10


class AgentLoop(PlanLayer):
    """PlanLayer + async re-plan after each outcome, history, retry/abandon, latency log, world-JSON cache."""

    def __init__(self, cfg, backend=None, log=None, clock=None):
        super().__init__(cfg, backend=backend, log=log, clock=clock)

    # ------------------------------------------------------------------ state
    def reset(self):
        super().reset()
        self.executed = []                  # [{tool, args, outcome, reason}] in execution order
        self.fails, self.tried = {}, {}     # obj_id -> failed grasps / grasp variants already used
        self.n_replans = 0                  # adopted closed-loop answers (the initial plan is not counted)
        self.next_req = None
        self.lat = []                       # [(kind, status, ms)] one entry per backend call
        self.stats = {"calls": 0, "adopted": 0, "timeout": 0, "error": 0, "invalid": 0, "busy": 0,
                      "capped": 0, "json_builds": 0, "json_hits": 0, "retry": 0, "abandon": 0}
        self._wj_key, self._wj = None, None
        self._pending_kind = None

    @property
    def max_replans(self):
        return int(self.c.get("closed_loop_max_replans", 6))

    @property
    def call_timeout_s(self):
        return float(self.c.get("call_timeout_s", 1.0))

    @property
    def max_grasp_fails(self):
        return int(self.c.get("max_grasp_fails", 2))

    def wait(self, seconds):
        """Tools / tests ONLY (brain.debug_wait): also wait for the in-flight re-plan. Never the control loop."""
        r = self.req.wait(seconds)
        if self.next_req is not None:
            self.next_req.wait(seconds)
        return r

    # ------------------------------------------------------------------ requests
    def _world_json(self, world, key):
        """World JSON is built once per look (key = the look's identity); later requests of the same look reuse it."""
        if key is not None and key == self._wj_key:
            self.stats["json_hits"] += 1
            return self._wj
        self.stats["json_builds"] += 1
        text = world_to_json(world)
        self._wj_key, self._wj = key, text
        return text

    def _payload(self, world, key, failure=None):
        extra = {"agent": True, "history": [dict(h) for h in self.executed[-HISTORY_LEN:]],
                 "n_predict": int(self.c.get("agent_n_predict", 160)), "timeout_s": self.call_timeout_s}
        if failure is not None:
            extra["failure"] = failure
        # same bytes as world_to_json({"mode": "plan", "world": world, **extra}) (sorted keys), world part cached
        head = world_to_json({"mode": "plan", **extra})
        return head[:-1] + ',"world":' + self._world_json(world, key) + "}"

    def _new_request(self, world, key, failure=None):
        r = AsyncRequest(self.adapter, self.call_timeout_s, worker=self.worker,
                         **({} if self.clock is None else {"clock": self.clock}))
        r.submit(self._payload(world, key, failure))
        self.stats["calls"] += 1
        return r

    def request(self, world, failure=None, key=None):
        """The initial plan (also re-used when it timed out and the policy asks again at the next look)."""
        if (self.requested and self.req.status not in ("timeout", "error", "busy")) or self.n_replans >= self.max_replans:
            return False
        self.req = self._new_request(world, key, failure)
        self.requested, self.world, self._pending_kind = True, world, "plan"
        return True

    def request_next(self, world, key=None, failure=None, why=""):
        """Async re-plan after an outcome. False when capped, still waiting for the previous answer, or the worker is busy."""
        if self.next_req is not None and self.next_req.status == "pending":
            return False
        if self.n_replans >= self.max_replans:
            self.stats["capped"] += 1
            return False
        r = self._new_request(world, key, failure)
        if r.status == "busy":
            self.stats["busy"] += 1
            self.stats["calls"] -= 1
            return False
        self.next_req, self.world = r, world
        self._note_why = why
        return True

    # ------------------------------------------------------------------ answers
    def _record(self, kind, req):
        ms = 1000.0 * ((getattr(req, "_tfin", None) or time.perf_counter()) - getattr(req, "_t0", time.perf_counter()))
        self.lat.append((kind, req.status, round(ms, 1)))
        self.log(f"[agent] call={kind} status={req.status} lat_ms={ms:.0f}")

    def note_initial(self):
        """Log the latency of the initial plan once it is settled (ready / timeout / error)."""
        if self._pending_kind == "plan" and self.req.status in ("ready", "timeout", "error", "busy"):
            self._record("plan", self.req)
            if self.req.status != "ready":
                self.stats[self.req.status] += 1
            self._pending_kind = None

    def poll_next(self, world, **validators):
        """Non-blocking. True = a re-plan was adopted (replaces the not-yet-executed rest of the plan); False = the
        answer was late / failed / invalid (the running plan or the classical resolver goes on); None = nothing new."""
        r = self.next_req
        if r is None:
            return None
        raw = r.poll()
        if raw is None:
            if r.status == "pending":
                return None
            self.next_req = None
            self.stats["timeout" if r.status == "timeout" else "error" if r.status == "error" else "busy"] += 1
            self._record("replan", r)
            self.log(f"[agent] replan {r.status}: keeping the running plan")
            return False
        self.next_req = None
        self._record("replan", r)
        steps, errors = validate_plan(raw, world, max_steps=self.c.get("max_steps", 32),
                                      prior_steps=self._successful(), **validators)
        for e in errors:
            self.log(f"[agent] rejected {e}")
        if not any(s["valid"] for s in steps):
            self.stats["invalid"] += 1
            self.log("[agent] replan had no valid step: keeping the running plan")
            return False
        self.steps, self.index, self.world = steps, 0, world
        self.adopted = True
        self.n_replans += 1
        self.stats["adopted"] += 1
        shown = [{"tool": s["tool"], "args": s["args"], "valid": s["valid"]} for s in steps]
        self.log(f"[agent] adopted replan {self.n_replans}/{self.max_replans}: {json.dumps(shown, separators=(',', ':'))}")
        return True

    def _successful(self):
        return [{"tool": h["tool"], "args": h["args"]} for h in self.executed
                if h.get("outcome") == "ok" and h["tool"] in ("pick_place", "place_at")]

    # ------------------------------------------------------------------ executor
    def next_call(self):
        call = super().next_call()
        if call is not None and self.steps is not None and self.steps[self.index].get("variant"):
            call["variant"] = self.steps[self.index]["variant"]
        return call

    def mark(self, outcome, reason=""):
        if self.steps is not None and self.index < len(self.steps):
            s = self.steps[self.index]
            self.executed.append({"tool": s["tool"], "args": dict(s["args"]), "outcome": outcome,
                                  "reason": str(reason)[:100]})
            if s.get("variant"):
                self.executed[-1]["variant"] = s["variant"]
        super().mark(outcome, reason)

    def on_grasp_failed(self, obj_id, bin_name, chooser=None):
        """A pick_place of obj_id did not take the object. -> ("retry", variant) | ("skip", None) and the step is queued.
        1 .. max_grasp_fails-1 failures: the same grasp again. From max_grasp_fails on: a grasp variant not used yet
        (chooser(untried) -> ordered list, e.g. the skill-memory bandit), at most two of them, then skip."""
        n = self.fails[obj_id] = self.fails.get(obj_id, 0) + 1
        tried = self.tried.setdefault(obj_id, {"same"})
        if n < self.max_grasp_fails:
            variant = "same"
        else:
            untried = [v for v in VARIANTS if v not in tried]
            order = []
            if untried and n < self.max_grasp_fails + 2:
                try:
                    order = [v for v in (chooser(list(untried)) if chooser else untried) if v in untried]
                except Exception:
                    order = []
                variant = (order or untried)[0]
            else:
                variant = None
        if variant is None:
            self.skipped_ids.add(obj_id)
            self.executed.append({"tool": "skip", "args": {"obj_id": obj_id}, "outcome": "ok",
                                  "reason": f"abandoned after {n} failed grasps"})
            self.stats["abandon"] += 1
            self.log(f"[agent] {obj_id}: {n} failed grasps, every variant used -> skip")
            return "skip", None
        tried.add(variant)
        self.stats["retry"] += 1
        step = {"tool": "pick_place", "args": {"obj_id": obj_id, "bin": bin_name}, "valid": True, "errors": [],
                "outcome": None, "variant": variant}
        if self.steps is None:
            self.steps, self.index = [], 0
        self.steps.insert(self.index, step)
        self.log(f"[agent] {obj_id}: failed grasp {n}, retrying with variant {variant!r}")
        return "retry", variant

    def summary(self):
        ms = sorted(m for _, st, m in self.lat if st == "ready")
        return {**self.stats, "replans": self.n_replans, "lat_ms_p50": ms[len(ms) // 2] if ms else None,
                "lat_ms_max": ms[-1] if ms else None}
