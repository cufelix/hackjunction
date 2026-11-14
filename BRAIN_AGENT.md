# Adaptive brain: machine vision + a tool-calling agent

The robot does not only sort known colours. It can be told to handle an object it has never seen ("put the pan in the
left bin") or a relational instruction ("put the striped one right"), and it recovers when a grasp fails. Everything
is behind flags (default OFF; flags off is byte-identical). Overlay: `overlays/finals_r22_agent.yaml`
(= `finals_r21` + the brain flags below, `brain.url = http://127.0.0.1:8080/completion`).

| Flag | What |
|---|---|
| `brain.enabled`, `brain.planner`, `brain.backend=llamacpp` | existing closed-set planner (`brain/planner.py`) |
| `brain.find` | VLM grounding tool `find(description)` (`brain/find.py`, `brain/find_tool.py`) |
| `brain.find_to_detection` | a find point that matches no detection becomes a novel `Detection` from the depth blob (`brain/novel.py`) |
| `brain.closed_loop` | agent loop: async re-plan after every outcome (`brain/agent.py`) |

## Architecture

```
 instruction ──► strict resolver (known colour/shape -> bin)  ── certain bins are AUTHORITY, never overridden
      │                                │
      │ abstains / relation / count    ▼
      ▼                       world JSON (per look, cached): objects{id,colour,shape,xy,resolver_bin,
 ┌───────────────┐                       novel,label,footprint,height,graspable}, bins, table_region,
 │  AgentLoop    │◄── history ──┐        found{query:[ids]}
 │ (async, capped│              │
 │  1.0 s / call)│── plan JSON ─┴──► validate_plan  (closed tool set, bin authority, reach + place dry-run,
 └──────┬────────┘   (GBNF, <=160     no-early-done, only-N, protected ids, graspable)
        │ llama.cpp   tokens)                │ valid steps only
        │ Qwen3-VL-8B Q8                      ▼
        │ 127.0.0.1:8080              executor in policy._choose
        │                                      │ each motion: classical planner dry-run (IK + SafetyFilter)
   find(description)                           ▼
        │  boxes (0..1000 -> px)         Plan ──► act()  (30 Hz loop, never waits for the brain)
        ▼
 match_found(points, detections) ── hit: known id
        └── miss: novel_detection(depth blob at the point, clipped to the VLM box)
                  region grow above the counter plane -> 3D centroid, footprint, height, colour name,
                  generic_grasp (top-down) ; too wide / flat / unsupported -> graspable=false -> skip
```

The model only ever sees ids and emits data (JSON from a GBNF grammar). It never sees or emits joints, poses or code.

## Tools (closed set)

`pick_place(obj_id, bin)` · `place_at(obj_id, table_xy)` (only for swap / next-to) · `find(description)` ·
`skip(obj_id)` · `wait` · `done`. Retry with another grasp variant (`same | closed_tight | demo`) is done in code by the loop.

## The loop

1. First look: one async plan call (timeout `brain.call_timeout_s`, default 1.0 s). Until it arrives, or on timeout /
   malformed / invalid, the classical resolver acts (never a stall; the plan is asked again at the next look, capped).
2. The executor runs the plan one primitive per look. The outcome comes from the next settled look
   (object left its source = ok; still there = failed; novel objects are confirmed from their re-grown blob).
3. After every outcome (ok / failed / skip / find) the loop requests a new plan, async, with the fresh world JSON and the
   history `[{tool,args,outcome,reason}]` of the executed steps, at most `brain.closed_loop_max_replans` (6) times.
   While it computes, the executor keeps running the last valid plan; a validated answer replaces the not-yet-run rest at
   the next look boundary.
4. Retry / abandon (deterministic): failed grasp 1 -> same grasp again; after `brain.max_grasp_fails` (2) failures a
   different variant (skill-memory order when `skills.memory` is on, else registry order), at most two variants, then
   `skip` (0 points, never a -1 guess).
5. `done` is accepted only when no resolvable target remains or the instruction is proven satisfied.

## Latency budget (RTX 5090, Qwen3-VL-8B Q8, measured on box E)

| Step | Budget | Notes |
|---|---|---|
| find() | ~320 ms p50, 420 ms max | 640x480 PNG, HTTP, warm |
| novel blob (region grow + generic_grasp) | ~20-40 ms | look thread, not act() |
| plan call | <= 1.0 s hard timeout | grammar-constrained, `agent_n_predict` 160 tokens, `cache_prompt` |
| act() | unchanged (>= 27 Hz) | all brain work is async / on the look thread; tested with a 0.8 s backend |

Per-call latency is logged: `[agent] call=replan status=ready lat_ms=212`. `AgentLoop.summary()` gives p50/max,
timeouts, retries, abandons, JSON cache hits.

## Safety invariants (each has a test in `test_brain_agent.py`)

1. The brain can never send joints: no tool or key outside the closed set survives `validate_plan`; the agent / novel
   modules contain no access to arm or hand commands.
2. It can never change a bin the strict resolver determined (`bin-authority`; novel objects follow the colour/default
   lookup the same way).
3. It can never declare `done` while resolvable targets remain (no-early-done guard, also at execution time).
4. `place_at` stays inside the safe table area (region + 2 cm IK margin, 6 cm from bins and objects).
5. Every motion is dry-run by the classical planner (IK + SafetyFilter) before it starts; infeasible = failed step.
6. Timeout / malformed / invalid answer = the running plan or the classical behaviour (compared with the brain-off run).
7. Prompt injection in the instruction cannot add tools or ids (length cap, closed grammar, validation).
8. Ungraspable novel objects (too wide > 6 cm, lower than 2 cm, unsupported) are skipped, never attempted.

## One container at the finals

The runner starts only `docker run -d --gpus all -p PORT:8000 IMAGE`, so the brain runs inside the policy image:

- `tools/brain/Dockerfile.policy_brain`: multi-stage; builds `llama-server` (CUDA 12.8, sm_120) from the staged
  `tools/brain/llama.cpp`, copies Qwen3-VL-8B-Instruct Q8_0 + mmproj from `tools/brain/models/`, bakes
  `overlays/finals_r22_agent.yaml` as `overlay.yaml`.
- `tools/brain/entrypoint.policy_brain.sh`: starts llama-server on `127.0.0.1:8080` in the background and immediately
  `exec`s the policy server. It never waits for the LLM; a missing model, a crash or an OOM only means the brain is off
  (classical `finals_r21` behaviour). `BRAIN_DISABLE=1` skips the server.
- `find_tool.py` ships in the policy image with `brain/` (`FindClient` falls back to the bundled copy when
  `/data/brain/find/find_tool.py` does not exist; `brain.find_path` / `brain.find_url` keep their old meaning).
- The default image (`tools/submission.Dockerfile`) is unchanged and brain-free in behaviour (overlay `submission`).
- Not built here. Build: `docker build -f tools/brain/Dockerfile.policy_brain -t hackjunction/policy:finals-agent .`
  Record the llama.cpp commit and both GGUF sha256.

## 60 s judge demo (sim)

Setup: finals_r22_agent overlay, llama sidecar up, sim scene with the usual coloured blocks plus one object the colour
detector has never seen (a grey pan / mug).

- 0:00-0:10 "The robot only knows colours and shapes. Watch what happens with something new." Show the head image:
  the pan has no detection box.
- 0:10-0:30 Instruction: **"Put the pan in the left bin."** Log shows `find('pan')` -> `[novel] ... novel grey object
  footprint=(0.04,0.04) height=0.04 graspable=True` -> `adopted replan` -> `pick_place(o5,left)`. Robot picks the pan, bin = left.
  Point out: nothing was trained on the pan; the VLM located it, depth made it graspable.
- 0:30-0:50 Instruction: **"Put the striped object on the right."** (relational / attribute): find returns the box of the
  matching known block, the loop picks that id and puts it right. Then show one failed grasp (cover the object): `failed
  grasp 1, retrying`, then it succeeds or `skip` after the variant.
- 0:50-1:00 Safety close: "The model only emits JSON from a grammar; a validator, the strict resolver and the classical
  planner's dry-run stand between it and the arm. If the LLM is slow or down, the classical sorter carries on: act() never waits."

## Still needed to run live (box E)

llama servers on 8080 (8B Q8) are already there for sidecar mode; for the single-container finals image: stage the pinned
llama.cpp tree + the two GGUFs under `tools/brain/`, build, and run `replay`/sim with the `finals_r22_agent` overlay.
Tune on real frames: `brain.novel_*` thresholds, the plan prompt wording, and measure plan-call p50 with the real prompt.
