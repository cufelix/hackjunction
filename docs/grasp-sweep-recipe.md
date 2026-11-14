# GPU sweep recipe: grasp-yaw patch in the Aalto sim

Repo: scratchpad/starter/repo_int (package dir `aaltonomous/`). V = VERIFIED (read in code, file:line). A = ASSUMED.
Paths below are relative to `aaltonomous/` unless they start with `/`.

## 0. Read this first: grasp_test does NOT exercise our patch

- V `sim/grasp_test.py:51` builds `OraclePolicy`, not `ClassicalPolicy`. The oracle's planner overrides
  `_plan_side_hints` (`sim/oracle.py:~145`) with `_plan_side_faces`, so `planner._plan_side_yaw` and `grasp.yaw_*` are never read.
- V The oracle reads privileged poses (`oracle.py:228-245`, yaw from the scene-service quaternion, shape from the scene), so
  `perception.classify_shape`/`triangle_pose` are never used.
- V The oracle has its own phi: env `AALTO_TRI_PHI` (default 105), `oracle.py:74-76`.
- Consequence: the stock `grasp_test` can only sweep the ORACLE's phi (a geometry check). To test OUR patch you need the
  classical policy in the loop. Section (a2) is a small new harness (a copy of grasp_test with `ClassicalPolicy`).
  The alternative with zero new code is `sim_eval` / `sim.evaluate` (full episodes, random layouts).
- V The patch is already applied in repo_int: `git apply --check` fails, `git apply -R --check` passes.

## (a) Exact commands

### a0. Apply the patch (fresh checkout of starter main)
```bash
cd $REPO                                    # /opt/aalto in the container, else the bundle dir
git apply --check /path/docs/grasp-yaw-merged.patch && git apply /path/docs/grasp-yaw-merged.patch
git apply -R --check /path/docs/grasp-yaw-merged.patch && echo "already applied"
python -m unittest aaltonomous/baselines/classical/test_yaw_align.py aaltonomous/baselines/classical/test_prism_pose.py   # offline, no sim (A: unittest-compatible; docstring says pytest)
```
The patch touches config.yaml, perception.py, planner.py and adds prism_pose.py and two test files (V: patch `diff --git` lines).

### a1. Start the sim and the env (V)
Docker (QUICKSTART.md:22-27): `docker run -d --name aalto --gpus all --network host --ipc host -v aalto-kitcache:... aalto-sim:v1 sleep infinity`,
then `docker exec -it aalto bash`, `aalto sim` (headless, log /tmp/aalto_sim.log). Bare metal: `bash aaltonomous/cloud/aalto.sh sim`
(`aalto.sh:28ff`; wraps `sim/run_sim.sh`, which runs `TASK=Aalto-Sort-G129-Dex3-Joint`, default at `aalto.sh:17`).
Client commands need conda env `isaac` and `PYTHONPATH=$REPO`; `aalto.sh py ...` sets both.
2nd isolated sim (`run_sim.sh:10-11`): start with `AALTO_INSTANCE=b`; clients then need
`AALTO_DDS_DOMAIN=7 AALTO_CAM_PORT_BASE=56555 AALTO_SCENE_PORT=56580 AALTO_RGBD_PORT=56560` (`aalto.sh:19-22` exports these defaults when `AALTO_INSTANCE` is set).
DR is off unless the sim was started with `AALTO_DR=train|eval_heldout|<yaml>` (`run_sim.sh:9`). Run the sweep nominal first.

### a2. How to set the overrides per run
Mechanisms that exist (V):
| Mechanism | Evidence | Usable for yaw keys? |
|---|---|---|
| `load_config(path, overrides, profile)`: dotted keys via `_set` | `baselines/classical/config.py:25-30, 33-48` | yes, any key |
| `sim_eval --set '<json>'` (dotted keys), `--config`, `--profile`, `--arm-vel` | `sim_eval.py:24,26,27,42-45` | yes |
| env `AALTO_PROFILE` (profile NAME only) | `config.py:38`, `config.yaml:10,133` | no (names `sim` / `none`) |
| `policy_server` env: `AALTO_TIME_LIMIT`, `AALTO_ARM_VEL`, `AALTO_PROFILE` only | `policy_server.py:34-38` | NO dotted-key override |
| `run.py` | `--config --profile --arm-vel` only (`run.py:207-213`) | only via a yaml |
| any `AALTO_*` env var for yaw keys | grep: none exist | no |

Trap (V): sim README says `AALTO_PROFILE=1` prints a per-step timing breakdown (README:68) but `config.load_config` reads the same var as a profile name and raises on `1`. Never set `AALTO_PROFILE=1` in a client shell.

Smallest safe mechanism for the policy-server / `sim.evaluate` path (PROPOSED, not in repo): 3 lines in `policy_server.py`
after the `AALTO_ARM_VEL` block (line 38): `import json` at top, then
```python
ov.update(json.loads(os.environ.get("AALTO_CFG_SET", "{}")))   # e.g. {"grasp.yaw_align": true, "grasp.yaw_phi_deg": 105}
```
For `sim_eval` no change is needed. Flags that must be on for the yaw path (V: `config.yaml:37-38,57`; `perception.py:49-53,149-156`;
`planner.py:320-325`): `perception.classify_shape`, `perception.triangle_pose`, `grasp.yaw_align`. Why `classify_shape` is needed: the
scene has 3 shapes (`scene.yaml:32`), so `scene_shape` is None and the blob is classified by `prism_pose.classify_blob` (`prism_pose.py:100-119`).
The yaw path then needs shape == "triangle", finite `apex_yaw`, `apex_conf >= yaw_min_conf` (0.5) (`planner.py:320-325`); otherwise it silently falls back to the stock hints.

Override JSON for one cell:
```
{"perception.classify_shape": true, "perception.triangle_pose": true, "grasp.yaw_align": true, "grasp.yaw_phi_deg": 105}
```
(other knobs, defaults V `config.yaml:58-65`: yaw_hint_weight 3.0, yaw_tol_deg 12, yaw_max_hints 2, yaw_min_conf 0.5, yaw_tri_shift true, yaw_ref_deg 25, yaw_max_dev_deg 75.)

### a3. grasp_test for triangles (ORACLE path, geometry check only) (V `sim/grasp_test.py:19-31,55`)
Args: `--shapes triangle`, `--colors` (cycled; colour fixes the bin and therefore the arm: red=left bin, yellow=centre, blue=right, `tasks.py:14`),
`--yaws` deg in the MAT frame, `--xs` mat x (m), `--ys` mat y (m), `--time-limit` (default 22 sim s), `--out f.json`, `-v`, `--trace`.
Cases = product(shapes x yaws x xs x ys); it runs ONE episode per case with env seed 0. There is no arm, no trials and no seed flag.
Mat to pelvis (V `scene.py:73-77`, `scene.yaml:20`): pelvis_x = 0.14 + y_mat, pelvis_y = -0.45 + x_mat. So pelvis x=0.30 means `--ys 0.16`,
and pelvis y = +0.05..+0.20 means `--xs 0.50,0.55,0.60,0.65`.
```bash
# left arm (red -> left bin), phi=105, apex headings 0/90/270 (layout yaw psi = -A: 0, -90, +90) at pelvis y=+0.05..0.20
AALTO_TRI_PHI=105 python -m aaltonomous.sim.grasp_test --shapes triangle --colors red --yaws 0,-90,90 \
    --xs 0.50,0.55,0.60,0.65 --ys 0.16 --out runs/gt_oracle_L_phi105.json
# right arm: pelvis y = -0.05..-0.20 -> xs 0.40,0.35,0.30,0.25 and blue; mirrored apex: psi = +A -> same yaw set
AALTO_TRI_PHI=105 python -m aaltonomous.sim.grasp_test --shapes triangle --colors blue --yaws 0,-90,90 \
    --xs 0.40,0.35,0.30,0.25 --ys 0.16 --out runs/gt_oracle_R_phi105.json
```
Caveat (V `oracle.py:~151-181`): the oracle uses the SAME phi sign for both arms (candidates apex+phi+k120, filtered by `FINGER_REF_DEG`), while our patch mirrors phi for the right arm (`planner.py:~335-336`). The oracle right-arm numbers therefore do not validate the patch's right arm (see section e).

### a4. The harness that tests the patch: `sim/grasp_test_cls.py` (NEW file, PROPOSED; needs no repo edits)
Same structure as `grasp_test.py:33-86` but with `ClassicalPolicy`, head RGB-D on (as `sim_eval.py:38-39`), and the instrumentation of section (c).
Pose list: section (b). Run:
```bash
python -m aaltonomous.sim.grasp_test_cls --arms left,right --phis 60,90,105,120,135 --baseline --time-limit 45 --out runs/sweep1.json
```
Code (copy to `aaltonomous/sim/grasp_test_cls.py`):
```python
"""Classical-planner yaw sweep: one triangle per episode, cells = (arm, phi | baseline)."""
import argparse, json, os, tempfile, threading, time
from pathlib import Path
import numpy as np
from aaltonomous.sim import evaluate as ev

HZ = 30.0
def poses(arm):
    """(pelvis_y, apex_heading_deg) list, x_pelvis fixed 0.30. Right arm = mirror image (y, A) -> (-y, -A)."""
    base = [(y, A) for A in (0, 90, 270) for y in (0.05, 0.10, 0.15, 0.20)] + [(0.20, 180)]   # 13 poses
    s = 1 if arm == "left" else -1
    return [(s * y, (s * A) % 360) for y, A in base]

def to_layout(arm, y_pel, A_deg):
    """pelvis (0.30, y, apex heading A) -> mat layout. V scene.py:73-77, layout.py:45-50: x_mat=0.45+y, y_mat=x-0.14, psi=-A."""
    return {"color": "red" if arm == "left" else "blue", "shape": "triangle",
            "x": round(0.45 + y_pel, 4), "y": round(0.30 - 0.14, 4), "yaw": float(-np.radians(A_deg))}

def wrap120(d_deg):                      # signed difference mod 120 deg
    return (d_deg + 60) % 120 - 60

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="left,right"); ap.add_argument("--phis", default="60,90,105,120,135")
    ap.add_argument("--baseline", action="store_true"); ap.add_argument("--time-limit", type=float, default=45.0)
    ap.add_argument("--set", default=None, help="extra overrides json"); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-poses", type=int, default=99); ap.add_argument("--out", default="runs/sweep.json")
    a = ap.parse_args(argv)
    from aaltonomous.deploy.runner import Recorder, run_episode
    from aaltonomous.interface import env as env_mod
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    ev.apply_instance_env()
    sc = SceneClient(a.host)
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=a.time_limit, realtime=True, img_server_ip=a.host,
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    tmp, rows, stop = Path(tempfile.mkdtemp(prefix="gtc_")), [], threading.Event()
    cells = ([("base", None)] if a.baseline else []) + [(f"phi{p}", float(p)) for p in a.phis.split(",")]
    extra = json.loads(a.set) if a.set else {}
    try:
        for arm in a.arms.split(","):
            for cell, phi in cells:
                ov = {"perception.classify_shape": True, "perception.triangle_pose": True,
                      "grasp.yaw_align": phi is not None, **extra}
                if phi is not None: ov["grasp.yaw_phi_deg"] = phi
                pol = ClassicalPolicy(config=load_config(None, ov, None), safety=env.filter.cfg, threaded=True,
                                      time_limit=a.time_limit, log=None)
                st = {"t_plan": [], "yaw_calls": 0, "yaw_none": 0}
                op = pol.planner.plan
                def timed(*x, _op=op, **k):
                    t = time.perf_counter(); r = _op(*x, **k); st["t_plan"].append(time.perf_counter() - t); return r
                pol.planner.plan = timed
                oy = pol.planner._plan_side_yaw
                def wy(*x, _oy=oy, **k):
                    st["yaw_calls"] += 1; r = _oy(*x, **k); st["yaw_none"] += r is None; return r
                pol.planner._plan_side_yaw = wy
                client = ev.LocalClient(pol)
                for i, (y, A) in enumerate(poses(arm)[: a.max_poses]):
                    lay = [to_layout(arm, y, A)]
                    env_mod.sample_layout = lambda rng, _l=lay, **kw: [dict(o) for o in _l]
                    for k in st: st[k] = [] if k == "t_plan" else 0
                    ev.home_arms(env, sc)
                    t0 = time.monotonic()
                    summ, _ = run_episode(env, client, i, i, Recorder(tmp / f"{arm}_{cell}_{i:02d}"), stop, "sort_fixed")
                    while pol._job is not None and pol._job.get("result") is None: time.sleep(0.05)
                    wall = time.monotonic() - t0
                    ev.settle(sc, 3.0, float(env.scene.cfg["success"]["at_rest_speed"]))
                    out = sc.evaluate("sort_fixed", env.mapping)["outcomes"][0]["outcome"]
                    p0 = pol.plans[0] if pol.plans else None
                    d0 = pol.last_detections[0] if pol.last_detections else None   # NB: last look; see (c)
                    rows.append({"arm": arm, "cell": cell, "phi": phi, "i": i, "y": y, "A": A, "outcome": out,
                                 "end": summ.get("end_reason"), "wall_s": round(wall, 1), "sim_ticks": summ.get("ticks"),
                                 "plan_ok": bool(p0 and p0.ok), "plan_side": getattr(p0, "side", None),
                                 "plan_reason": getattr(p0, "reason", None), "plan_s": [round(t, 2) for t in st["t_plan"]],
                                 "yaw_calls": st["yaw_calls"], "yaw_none": st["yaw_none"],
                                 "yaw_used": bool(p0 and p0.ok and str(p0.reason).startswith("yaw-aligned")),
                                 "n_det": len(pol.last_detections), "events": pol.events[-6:]})
                    print(f"{arm:5s} {cell:6s} #{i:02d} y={y:+.2f} A={A:3d} -> {out:8s} side={rows[-1]['plan_side']} "
                          f"yaw_used={rows[-1]['yaw_used']} plan_s={rows[-1]['plan_s']} wall={wall:.0f}s", flush=True)
                    json.dump(rows, open(a.out, "w"), indent=1, default=str)      # incremental: survives a crash
    finally:
        client.close(); env.close()
if __name__ == "__main__":
    main()
```
Notes on the harness (A until run once; smoke-test with `--arms left --phis 105 --max-poses 2` first, about 3 min):
- `pol.last_detections` after the episode is from the LAST look (object already in the bin, so usually empty). To log the first look's apex_yaw
  error, add inside `timed()` before calling `_op`: `st["det0"] = st.get("det0") or [(d.shape, d.apex_yaw, d.apex_conf) for d in pol.last_detections]`
  (`plan` is called with the Detection as first arg: use `x[0]`), and compare with `wrap120(np.degrees(apex_yaw) - A)`.
- `pol.plans`/`pol.events` are reset by `policy.reset` (`policy.py:~84-85`), which `LocalClient.reset` calls at each episode start (`evaluate.py:73-74`).
- Arm gating: colour decides the arm (red: left bin so side0 = left, `planner.py:~226-228`), but the planner falls back to the other arm and then relocation
  if the first fails, so ALWAYS read `plan_side`; a right-arm result inside a "left" cell is a failed-left cell, count it separately.

### a5. Full-episode alternatives
(1) No new code, patch path, random layouts (3-6 objects, about 1/3 triangles), V `sim_eval.py:21-28`:
```bash
for CFG in base yaw105; do
  case $CFG in base)  S='{"perception.classify_shape":true,"perception.triangle_pose":true,"grasp.yaw_align":false}';;
               yaw105) S='{"perception.classify_shape":true,"perception.triangle_pose":true,"grasp.yaw_align":true,"grasp.yaw_phi_deg":105}';; esac
  python -m aaltonomous.baselines.classical.sim_eval --seeds $(seq 1000 1011) --time-limit 120 --set "$S" --log runs/se_$CFG.jsonl
done
```
`--log` appends one json line per episode with `outcomes` and `events` (V `sim_eval.py:62-65,69-72`). `events` carry `execute ... ` plan summaries but not plan.reason; grep "yaw-aligned" will NOT work there (see (c)).
(2) `sim.evaluate` (V `evaluate.py:300-322`) runs the policy over the websocket protocol. Needs the proposed `AALTO_CFG_SET` (a2):
```bash
AALTO_CFG_SET='{"perception.classify_shape":true,"perception.triangle_pose":true,"grasp.yaw_align":true,"grasp.yaw_phi_deg":105}' \
AALTO_TIME_LIMIT=120 python -m aaltonomous.baselines.classical.policy_server &          # :8000
python -m aaltonomous.sim.evaluate --policy ws://localhost:8000 --team yaw105 --task sort_fixed --trials 12 --seed 1000 \
    --time-limit 120 --no-cameras --out runs/eval        # scores in runs/eval/<team>_sort_fixed_<variant>_<ts>/eval.json
```
Without the `AALTO_CFG_SET` change this runs the stock config (yaw_align false). `--policy oracle` runs the ORACLE (`evaluate.py:~170`), not the patch. Same `--seed` for base and yaw gives identical layouts (V `evaluate.py:~183-185`: seed = `--seed`+i).

## (b) The matrix

Cells: phi in {60, 90, 105, 120, 135} x arm {left, right} = 10 yaw cells, plus 2 baseline cells (`grasp.yaw_align=false`, one per arm) = 12 cells, 13 poses each = 156 episodes.
Same poses and same episode index (= env seed) in every cell (harness uses `i` as seed and the same `poses(arm)`).
Pose list (pelvis frame, x = 0.30 fixed; A = heading pelvis x-forward, CCW toward +y left, of one vertex of the triangle; 13 poses):
```python
base = [(y, A) for A in (0, 90, 270) for y in (0.05, 0.10, 0.15, 0.20)] + [(0.20, 180)]
left  = [( y, A % 360) for y, A in base]       # y > 0, red -> left bin,  mat x = 0.45 + y = 0.50..0.65, mat y = 0.16
right = [(-y, (-A) % 360) for y, A in base]    # mirror image; blue -> right bin, mat x = 0.45 - y = 0.40..0.25
# layout yaw (mat frame, what the scene service takes) = -A in radians, A=0 -> 0, A=90 -> -90 deg, A=270 -> +90 deg
```
Evidence for the conversion (V): `scene.yaml:20` origin [0.14,-0.45,0.01], `scene.py:73-77` (mat x -> pelvis y, mat y -> pelvis x),
`layout.py:45-50` + `test_layout.py:50-56` (mesh apex heading in pelvis = -layout yaw).
Why these poses (A = windows measured on CPU, taken from the brief, not re-verified here): apex 0/90/270 at all four |y|, apex 180 only at |y| = 0.20, 45 excluded.
Because a triangle repeats every 120 deg, the physically distinct classes are A mod 120: 0 -> 0, 90 -> 90, 270 -> 30, 180 -> 60, 45 -> 45. So the 12 core poses cover 3 classes x 4 y,
the 13th adds class 60. A: the measured windows were per true vertex heading; since the planner and the object only see A mod 120, check that the CPU table agrees across 0/120/240 before trusting the "45 never" row.
Risks to check before the sweep (A):
- `|y| = 0.20` lies outside `layout.region_x [0.28, 0.62]` (`scene.yaml:67`, "kept inside the head camera view"): mat x = 0.65 / 0.25. The object may be partly out of the head view: perception failures there are not grasp failures.
  Run the first 2 poses of that column with `plan_ok`/`n_det` logged before the full sweep.
- Cross-body: arm and bin are tied by colour (left arm + red/left bin, same side as the object) so no relocation is expected; any `plan_side != arm` or reason starting `relocate` is logged as a non-cell result.
- DR off, one physics seed per pose: the cell success rate over 13 is coarse (one pose = 7.7 points; 95 % binomial CI about +-25 points). Decide a phi winner only on a gap of >= 3 poses, then confirm on the winner vs baseline with a second pass (poses jittered by `yaw +-10 deg`, or `AALTO_DR=train` sim).

## (c) Success metric and output

`grasp_test.py` (V `:58-76,85-86`): per case prints `idx shape color yaw x y -> outcome final=(x,y,z)`, then `shape n/N correct`;
`--out` writes a json list with keys shape, color, yaw, x, y, outcome, end, final, events (last 6 oracle events). `outcome` comes from `sc.evaluate("sort_fixed", env.mapping)["outcomes"][0]["outcome"]`; success = `"correct"` (object in its colour's bin) (V `sim_eval.py:60`, `scene.py:244-251`).
Success rate per cell: group rows by (arm, cell), `mean(outcome == "correct")`. Harness rows (a4) add: `plan_ok`, `plan_side`, `plan_reason`, `yaw_used`, `plan_s` (list, s per `planner.plan` call), `yaw_calls`, `yaw_none`, `wall_s`.
```python
import json, collections, statistics as S
rows = json.load(open("runs/sweep1.json")); g = collections.defaultdict(list)
for r in rows: g[(r["arm"], r["cell"])].append(r)
for (arm, cell), v in sorted(g.items()):
    n = len(v); ok = sum(r["outcome"] == "correct" for r in v)
    yaw = [r for r in v if r["yaw_calls"]]                      # episodes where the yaw path actually ran
    fb = sum(r["yaw_none"] > 0 or not r["yaw_used"] for r in yaw)  # yaw path ran but a stock-hint plan won (or nothing)
    ps = [r["plan_s"][0] for r in v if r["plan_s"]]
    print(f"{arm:5s} {cell:6s} success {ok}/{n}  yaw_ran {len(yaw)}  fallback {fb}/{len(yaw)}  "
          f"plan_s median {S.median(ps):.2f} max {max(ps):.2f}  wall median {S.median(r['wall_s'] for r in v):.0f}s")
```
Definitions: fallback rate = (episodes where `_plan_side_yaw` was called and returned None, i.e. `yaw_none > 0`) / (episodes where it was called). `yaw_used` is true when the first plan's
reason starts with `yaw-aligned` (V `planner.py:376`, text `yaw-aligned (<deg> deg, err <deg>)`; the achieved finger heading and error are in that string, parse them for the (e) check).
Planning time per pick: the wrapper around `planner.plan` (stock code does not log it; V grep: only `Plan.duration`, the execution time, exists, `planner.py:55`). Report the first call (baseline vs yaw cell, same pose): that is the cost of the extra candidate IK tries.
If `yaw_calls == 0` in a yaw cell, the flags/perception did not trigger (shape not "triangle", `apex_conf < 0.5`, or no detection): that is a perception problem and the cell is not a valid test; log `n_det`.
A baseline cell must show `yaw_calls == 0` everywhere (sanity check of the flag).

## (d) Wall time

Inputs (V): sim runs at RTF 0.70 alone, 0.59 with 2 sims, 0.50-0.53 with 3 (`sim/README.md:58`); "120 s sim ~ 220 s wall" (README:69); QUICKSTART:41 "one baseline trial ~75 s" (oracle, rung 1);
planning "0.1 to few s" per look (`policy_server.py:6-7`); oracle single pick limit 22 s sim (`grasp_test.py:26`); first shader compile 5-10 min (QUICKSTART:40).
Per episode (A, not measured): home_arms ~4 sim s + settle 0.3 s + look/plan 1-3 s wall (1 pick) + pick, carry, drop and return ~18-22 sim s + 2nd look + `settle` 3 s
= roughly 30 sim s -> 30 / 0.65 + 5 = about 50 s wall. Use 45-75 s (low from the QUICKSTART 75 s figure for the oracle, high if planning + relays are slow).
| Item | Estimate |
|---|---|
| One cell (13 episodes), one sim | 13 x 60 s = about 13 min (10-16 min) |
| 12 cells (156 episodes), one sim | about 2.6 h (2.0-3.4 h) |
| Same with 2 sims (`AALTO_INSTANCE=b`; each sim about 19 % slower, RTF 0.70 -> 0.59), cells split by arm | about 1.6 h (needs about 12 CPU cores free) |
| Smoke test (`--arms left --phis 105 --max-poses 2`) | about 3 min plus sim start |
| Sim start, first time | 5-10 min shader compile, later 20 s - 2 min |
| Oracle `grasp_test` (a3) 12 poses, one phi, one arm | 12 x (22 / 0.65 + 8) = about 8 min |
| `sim_eval` 12 seeds x 2 configs (a5.1) | 24 x about 200 s = about 80 min (A: all episodes run to the 120 s limit; they end earlier when the policy says done) |
Trim if time is short: phi {90, 105, 120} x both arms + baseline = 8 cells = about 1.7 h on one sim, about 65 min on two.
Unmeasured: `ClassicalPolicy` construction per cell (Pinocchio model load, A: few s), the `AaltoEnv` first reset.

## (e) Sign-convention check (Felix)

Frames (all V):
1. Layout/mat yaw psi (CCW from mat +x; mesh apex along mesh +y): `layout.py:45-50`.
2. Mesh yaw in the pelvis/world frame = `t = -pi/2 - psi` (`mat_yaw_to_pelvis`, `layout.py:45-50`). Apex heading in the pelvis frame = `t + pi/2 = -psi`
   (`test_layout.py:50-56` checks `[-sin t, cos t]` = apex direction). Mesh apex +y at t=0: `oracle.py:71-73` comment, `export.py` (centroid at origin, apex +y).
3. Oracle: `det.yaw` = quaternion yaw of the object body = `t` (`oracle.py:241`); finger heading `a = yaw + pi/2 + phi + k*2pi/3` (`oracle.py:76,161`), i.e. apex heading + phi + k*120, and `_tri_shift(a - (yaw + pi/2))` (`oracle.py:171`), i.e. phi measured from the apex heading. Same convention as the patch's `apex + s*phi + k*120` and `_tri_shift(a - apex_yaw)` (`planner.py:335-339, 369-371`) for the LEFT arm.
4. `perception._yaw` (`perception.py:223-230`) is NOT the mesh yaw: `cv2.minAreaRect` angle of the top-face points in pelvis (x, y), folded to [-45, 45) deg ("info only", `perception.py:27`). It is unrelated to apex heading and the patch does not read it.
   It is the same orientation convention (CCW from pelvis +x, A: OpenCV angle of raw (x,y) numbers) but only mod 90.
5. `triangle_pose` apex_yaw (`prism_pose.py:72-97`, `perception.py:30,168-173`): pelvis xy (x forward, y left), heading centroid -> a vertex, in [prefer-60, prefer+60) = [30, 150) deg with `prefer_yaw = pi/2`. So it equals `-psi (mod 120)`, not necessarily `-psi`.
   The docstring "prefer_yaw = +y = the scene's rot=0 apex" is MAT-frame wording; in pelvis frame rot=0 apex is +x. Harmless because only mod 120 is used (`planner.py:339`: k = 0..2).
6. Mismatch found (V): oracle right arm uses `a = apex + phi + k120` (same sign as left, `oracle.py:161`, ref +25 deg via `FINGER_REF_DEG`, `oracle.py:77`), the patch right arm uses `apex - phi + k120`
   (`planner.py:335-336`). Mirroring is geometrically right for a chiral hand (reflect scene y -> -y: apex -> -apex, finger heading -> -heading), but the oracle's phi table (config.yaml:58, "left arm, n=6 per phi") says nothing about the right arm.

Expected values (phi = 105, defaults; computed by hand from the code, A until run):
| layout yaw psi | apex heading A=-psi | `apex_yaw` expected (mod 120 equal) | left finger heading (patch) | oracle left | right finger heading (patch) | oracle right |
|---|---|---|---|---|---|---|
| 0 | 0 | 120 | -15 deg | -15 | +15 | -15 |
| -90 | 90 | 90 | -45 deg | -45 | -15 | +75 |
| -20 | 20 | 140 | +5 deg | +5 | +35 | +5 |
Rule: left = the `apex_yaw + 105 + k120` candidate within 75 deg of -25 (nearest first, max 2); right (patch) = `apex_yaw - 105 + k120` within 75 deg of +25.

Single-prism test on the box (about 10 min, A that it runs as written; uses the sim and the harness above):
```python
# t_sign.py  (python -m aaltonomous.sim.t_sign), env isaac, sim running (Aalto-Sort)
import numpy as np, time
from aaltonomous.sim.scene_client import SceneClient
from aaltonomous.sim import evaluate as ev
from aaltonomous.interface import env as env_mod
from aaltonomous.interface.env import AaltoEnv
from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.policy import ClassicalPolicy
ev.apply_instance_env(); sc = SceneClient("127.0.0.1")
env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=60, realtime=True, img_server_ip="127.0.0.1", head_rgbd=True)
ov = {"perception.classify_shape": True, "perception.triangle_pose": True, "grasp.yaw_align": True, "grasp.yaw_phi_deg": 105}
pol = ClassicalPolicy(config=load_config(None, ov, None), safety=env.filter.cfg, time_limit=60, log=None)
w = lambda d: (d + 180) % 360 - 180
for psi_deg, y_pel, arm in [(0, 0.10, "left"), (-90, 0.10, "left"), (-20, 0.10, "left"), (0, -0.10, "right")]:
    lay = [{"color": "red" if arm == "left" else "blue", "shape": "triangle", "x": 0.45 + y_pel, "y": 0.16, "yaw": np.radians(psi_deg)}]
    env_mod.sample_layout = lambda rng, _l=lay, **kw: [dict(o) for o in _l]
    ev.home_arms(env, sc); obs = env.reset(); time.sleep(1.5); obs = env._obs()
    q = next(o for o in sc.state()["objects"] if o["active"])["quat_xyzw"]
    t = np.degrees(np.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)))
    print(f"psi={psi_deg}: sim mesh yaw t={t:.1f} expected {w(-90-psi_deg):.1f} -> apex heading {w(t+90):.1f} expected {w(-psi_deg):.1f}")   # step 1
    dets = pol.perception.detect(obs["head_rgb"], obs["head_depth"]); d = dets[0]
    print(f"   perception apex_yaw={np.degrees(d.apex_yaw):.1f} conf={d.apex_conf:.2f}; diff to -psi mod 120 = {((np.degrees(d.apex_yaw) + psi_deg + 60) % 120) - 60:.1f}; _yaw (minAreaRect, mod 90) = {np.degrees(d.yaw):.1f}")  # step 2
    pol.planner.mapping = {"red": "left", "blue": "right", "yellow": "center"}
    p = pol.planner.plan(d, env.filter.arm_cmd.copy(), env.filter.hand_cmd.copy())
    print("   plan:", p.side, p.reason)                                                                              # step 3: "yaw-aligned (<deg> deg, err <e>)"
    exp = pol.planner.yaw_hint_angles(d.apex_yaw, arm); print("   hint angles (deg):", [round(np.degrees(a), 1) for a in exp])
```
Pass criteria: (1) sim mesh yaw `t == -90 - psi` and apex heading `t+90 == -psi` (within 3 deg). (2) `apex_yaw` equals `-psi` mod 120 within about 10 deg (p90 < 10 deg is the claim at `prism_pose.py:22`) and `apex_conf >= 0.5`.
(3) `plan.reason` is `yaw-aligned (X deg, err E)` with X the table value for that arm (psi 0 left: -15, right: +15), E <= 12.
Physical check of the sign (A, only the sim can show it): run the same pose in oracle `grasp_test` (a3, `--trace -v`) and in the harness. For the LEFT arm the planner's finger heading must match the oracle's logged `face-aligned (<deg> deg, err ..)` (event text `oracle.py:~176`) to within 12 deg. For the RIGHT arm they will differ by 30 deg at psi 0 (see item 6): that is expected, run both signs with `--set '{"grasp.yaw_phi_deg": -105}'` on the right-arm cells ONLY IF the mirrored phi underperforms (yaw_phi_deg is multiplied by s=-1 for the right arm, so -105 means +105 on the right = the oracle's un-mirrored choice).
