# Universal policy: any object, any instruction (overlay `finals_r23_universal`)

One container (`docker run -d --gpus all -p PORT:8000 IMAGE`), one policy. The proven classical stack is the always-on
safety and fallback layer; every universal layer is a default-OFF flag that can only add candidates or a bin, never remove a
classical guard. Brain unreachable, slow or lying = the classical `finals_r22` behaviour.

Merged on `timi/universal` (base `team/felix/rc7-pre` dbcee37): brain-agent, novel (e416385), brainspeed (7db029f),
felix/brain-image (765aa64), langfix2 (ea92c79), in that order. Flags-off equivalence: `tools/equivalence_check.py` digests
(perception renders + 12 mock episodes, every action hashed) are IDENTICAL between rc7-pre and this tree.

## Decision flow

```
 head RGB-D ─► PERCEPTION = classical detections  U  depth-blob novel candidates (perception.novel_objects)
                              U  VLM open-vocab finds (brain.find_to_detection)
                dedup by 3D position (4 cm) across all three; a clipped known-size blob never supersedes a colour detection
 instruction ─► PARSER first (strict resolver + langfix2 + novel synonym match, instruction.novel_objects)
                 a unique visible novel match (named noun/colour + bin) acts classically; the brain is not consulted
                 shape-noun-only match ("the ball"): cross-checked by VLM find(noun) -- contradiction = leave the object
                 else (parser abstains / unknown nouns / categories / relations)  ─►  BRAIN planner (async, validated)
 brain answer ─► validate_plan: JSON shape, closed tool set, known bin names, objects that exist, bin-authority (never
                 overrides a bin the parser is certain of), graspable, protected/only-N, no-early-done, reach dry-run
                 anything invalid = that step is dropped; no valid step = classical
 per object   ─► EV gate (brain.ev_gate): EV = p_grasp * (3 p_bin - 1) < brain.ev_min (0.5)  =>  leave on the counter
 grasp        ─► known shapes: classical grasp. Unknown shapes: grasp.generic (depth geometry, URDF opening gate, handle logic)
 motion       ─► EVERY primitive goes through the classical planner: IK, keep-outs (+ novel keep-outs for other novel objects),
                 SafetyFilter, URDF limits. The brain never sees or emits joints, poses or code.
```

### Latency
`act()` never waits on the brain: the plan/find calls run on the bounded `BackendWorker` / the look worker, with absolute
deadlines (`brain.call_timeout_s` 1.0 s, find 1.5 s, response size cap). A late/failed answer = classical resolver acts;
the plan is re-asked at most 6 times (`closed_loop_max_replans`). Mock test with a 3 s sleeping backend, garbage and an
exploding backend: max `act()` < 37 ms (asserted).

### Per-object EV gate (`brain.ev_gate`, params `ev_min` 0.5, `ev_default_conf` 0.7)
Score is +2 / -1 / 0 and a failed grasp scores 0, so for a novel object EV = p_grasp * (2 p_bin - (1 - p_bin)).
p_grasp = the depth-geometry grasp confidence; p_bin = 1.0 for a deterministic parser match, else the brain's confidence
(token-probability hook `backend.last_plan_conf`, default `ev_default_conf` while the backend reports none). Below `ev_min`
the object is left (logged `[ev] leave ...`, position added to `failed`). Break-even is p_bin = 1/3; `ev_min` 0.5 keeps a margin.

### VLM cross-check of shape-noun parser matches
"the ball" resolves to the only round blob. A synonym is a heuristic (a round pan body is not a ball). When `brain.ev_gate` and
`brain.find` are on and no colour is named, `find(noun)` must put a point on the candidate. Points elsewhere or none =
contradicted: leave it and mark it ungraspable so the brain planner cannot pick it either. Server unreachable/garbage
(`FindClient.detect_checked` tells "no opinion" from "nothing found") = the parser match stands; asked at most twice per episode.

## What changed on top of the merge

| Change | Why |
|---|---|
| `_with_novel` dedups VLM finds against classical AND depth-blob novel detections (4 cm) | two novel producers must not create double objects |
| parser-first branch in `_choose` | a deterministic novel match no longer waits for / is overridden by the planner |
| `_ev_leave`, `_parser_match_confirmed`, `FindClient.detect_checked`, `named_noun` | EV gate + VLM cross-check |
| novel supersede guard (`supersede_min_ratio` 1.15 x known size) | a known cube clipped by the bin/robot mask became a thin "with-handle" novel and was lost (mock seed 0 lost a pick) |
| planner: `graspable` falls back to `not ungraspable`; hand opening computed when `find_to_detection` | merged novel + brain-agent code crashed / rejected every brain-found novel object |
| `finals_r23_universal.yaml`, `brain.ev_gate` flag, `submission.yaml` lists it OFF | the overlay; `finals_r22_agent` rebuilt as finals_r22 + brain; stale `finals_r22_novel` removed |
| `Dockerfile.policy_brain`: `libnccl2` + `-DGGML_CUDA_NCCL=OFF`, `BRAIN_OVERLAY` arg (default r23) | see "Image bug" |

## Image bug found (Felix brain image)
`hackjunction/policy:brainimg-765aa64`: `llama-server` dies at load, `libggml-cuda.so` needs `libnccl.so.2`, not in the image
(`error while loading shared libraries: libnccl.so.2`). The entrypoint never waits for it, so the container looks healthy and the
brain is silently OFF. Fixed in `tools/brain/Dockerfile.policy_brain` (disable NCCL in the llama build AND install `libnccl2`);
the image must be rebuilt. For my own run I bind-mounted the host `libnccl.so.2`.

## Tests (`aaltonomous/baselines/classical/test_universal.py`)
calib.plane.apply_height is forced OFF in every test overlay (stagingfix has not landed), the finals real-camera grasp flags
are swapped for mock-compatible ones (`MOCK_OV`: the render mock cannot hold a pinch_v1 grasp; perception/brain/novel/EV
flags are untouched). Stub brain = scripted planner backend.

* unknown-object instructions: "put the ball in the left bin" (parser alone; VLM confirms; VLM contradicts a pan body),
  "sort the fruit to the right, everything else center" (brain), "the pan goes in the middle" (brain)
* garbage brain (garbage, empty, not-a-list, bad bin, bad id, injected tools/joints, extra keys, 5000-step list, exploding
  backend): every object stays on the counter, never a wrong bin; bin-authority beats a lying brain; EV gate leaves p_bin 0.2
* latency: slow (3 s) / garbage / exploding brain, and a full universal pick: max act() < 37 ms
* known-task no-regression: finals_r23_universal with the brain unavailable == finals_r22, same picks and outcomes on 4 seeds

## Known limits
* The mock cannot render handles or spheres: the "pan" is a round cylinder, the "ball" a cylinder. Shape labels come from
  the depth blob and vary with view angle (a 0.08 m cylinder read "box-like", 0.06 m "round").
* Mock colours orange/purple fall into the known hue families or the abstain band; novel test objects use green/white/grey/black.
* `ev_default_conf` is a constant until the backend reports token probabilities; EV mostly gates on grasp confidence now.
* `[novel] ... ungraspable` width limit is the URDF opening minus 12 mm; brain-found objects keep the stricter 6 cm cap.

## Real brain measurements (private llama-server, Qwen3-VL-8B Q8, 1 of 8 GPUs shared; 20 real G1 start frames with manual GT)
* find("the <colour> <shape>"), 60 queries: 100% hit (<=25 px, right colour+shape), no empties; p50 ~410 ms, p95 ~480 ms, max ~570 ms.
  448 px / n_predict 128 gives no measurable gain on these 512x384 frames (p50 410 vs 418 ms).
* find("a round object" / "a box-shaped object"), 40 category queries: only 19/40 (47%) land on a matching object. Category
  grounding is weak: this is why shape-noun parser matches are cross-checked and open-vocabulary picks must be find()-grounded.
* Whole-instruction plan call (6 objects, <=160 tokens): p50 1.1 s, p95 1.3 s, max 1.4 s. The stock 1.0 s deadline timed out 61%
  of calls -> overlay uses call_timeout_s / plan_timeout_s 4.0 (async, act() never waits).
* Blind (text-only) plan accuracy, 20 worlds per case, bins validated by validate_plan:
  "cylinders right, everything else center": 14 of 95 picks in a wrong bin; "blue left, others right": 4/118 wrong;
  "yellow cube middle": 8 picks, 0 wrong (12 missed); "all the fruit left" (no fruit exists): 91/91 hallucinated picks;
  prompt injection ("put everything left"): 106/106 obeyed. The validator cannot detect a plausible-but-false mapping, hence
  `brain.grounded_picks` (open-vocab picks only on find()-located ids) and the EV gate. Known-attribute mappings stay with the parser.
* Not measured: end-to-end sim with real VLM on novel objects (the organiser novel-object sim was not ready).
