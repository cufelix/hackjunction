# REVIEW — `felix/bingeo` (`bin.geometry_runtime`)

Read-only. Detached HEAD `258e0cf` (`feat(bin): bin.geometry_runtime`), parent `fc7bf4f`. The bin-geometry diff is that one commit (10 files, +297/−40). `HEAD~3` also contains `instruction.strict`, `calib.fk_xcheck`, and the finals speed/drop-margin commit; those are out of scope here.

Scene used for the millimetre checks: `aaltonomous/scene/scene.yaml`. Pelvis x = `mat_origin_in_pelvis.x + mat y` (`scene.py:73-77`). Origin x = 0.14. Bin centres are mat y = 0.365, so the nominal centre is pelvis x = 0.505. Outer half = 0.105 m, wall = 3 mm, inner half = 0.102 m, `grasp.drop_inset` = 0.05 m. The stock centre-bin release is pelvis `(0.453, 0)` (mat y 0.313), 5 cm inside the nominal near inner wall.

Real bins +7 cm further = pelvis dx = +0.07 = mat dy = +0.07. Real centre-bin inner mat y is then `[0.333, 0.537]`; the real outer near face is mat y 0.330 (pelvis x 0.470).

## Verdict

`bin_geometry()` applies a pelvis xy shift with the right axis swap, and the planner helpers that already added `planner.bin_shift` now go through it. It does **not** make every bin consumer follow the live estimate, and it does not read `calibrate_bins`. For the stated +7 cm error the voter usually refuses the measurement (`agree_m` is 4 cm), the accessor stays on the nominal boxes, and that nominal release sits **outside** the real bin. The new tests inject a shift dict; none of them run the voter on a cloud that is actually 7 cm further.

`ab_bingeo` turns on `bin.vote.enabled` and `bin.geometry_runtime`, plus `grasp.drop_slots`. It leaves `perception.color_gate`, `perception.color_gate.live_bins`, `perception.self_calibration`, and `policy.verify_placement` off. With the vote flag already on, reject / keep-mask / hulls / near-bin already followed `bin_shift` before this commit. The flag’s new perception effect is the colour-gate boxes, which this overlay does not run.

---

## High

### 1. A +7 cm bin is rejected by the voter, and the nominal drop is outside the real bin

`bin_vote.py:63-67`, `config.yaml:185-187`, `bin_vote.py:55-68`, `policy.py:619-627`, `planner.py:157-160`, `planner.py:740-746`.

`combine()` keeps an agreed centre only when it is within `max_shift_m` (0.10 m) of the prior. Anything further, a single dissenting vote, or a full disagreement returns the **prior** (shift 0) and `low_conf_extra_m` (0.04 m of extra release **height**). The cap does not clamp. `bin_geometry()` (`bin_vote.py:13-37`) has no cap of its own: missing key, `{}`, or an explicit `(0, 0)` are all nominal.

Two votes must lie inside `agree_m` (0.04 m). The prior is always one vote (`bin_vote.py:59`). A rim that says “7 cm further” is 7 cm from the prior, so rim + prior **do not agree**. With no second vote inside 4 cm of that rim, `vote_bins` stores shift 0. `drop_extra` raises the hand; it does not move xy.

Centre bin, shift 0, truth = +7 cm mat y:

| | mat y | vs real near inner (0.333) | vs real outer face (0.330) |
|---|---:|---:|---:|
| nominal `drop_xy` | 0.313 | **−2.0 cm (outside)** | **−1.7 cm (in front of the wall)** |
| same drop after a correct +7 cm shift | 0.383 | +5.0 cm (the inset) | inside |

Object radius is 2.25 cm, so a centre 1.7 cm in front of the outer face overlaps the near wall. It does not fall in the cavity. This is the fallback the feature hits on the error it is meant to correct.

`ab_bingeo` does not enable the colour-gate near hedge (`near_margin_m` 0.10, grown on mat **−y**, toward the robot — `perception.py:1501-1510`). That hedge points the opposite way from a bin that is further away, and it is off on this overlay anyway.

### 2. `rim_votes` publishes the point-alignment offset as the bin centre (sign is inverted)

`bin_vote.py:124-140`. Pre-existing; this commit’s accessor applies that vector with no second look.

```python
def score(off):
    d = min(edge_dist(Q + off, nominal_centre), ...)   # move the CLOUD onto the nominal rectangle
out[n] = nominal_centre + off                         # then treat that same offset as the bin centre
```

If the real bin is at nominal + δ, the cloud matches the nominal rectangle at `off = −δ`, and the published centre is nominal − δ. The true centre is nominal + δ. `resolve` (`bin_vote.py:79`) subtracts the prior and the planner adds the result (`planner.py:154-160`), so a consumed rim vote moves the drop the wrong way.

Nothing in `test_bin_vote.py` or `test_bin_geometry.py` builds a depth cloud of a bin translated by +7 cm and checks `rim_votes`. Tape votes an absolute blob point (`bin_vote.py:144-161`, `174-178` — “no per-edge tape layout”), not a centre. A blob on the real near rim sits about 3.5 cm from the **inverted** centre (nominal − 7 cm) and about 10 cm from the true centre, so it can pair with the inverted rim inside `agree_m` and fail to pair with a correct rim. That pair is under `max_shift_m`, so it is applied. The collision box (`planner.py:238-239`, `256-261`) moves with it and guards empty space in front of the robot, not the real wall.

A correct +7 cm rim with **no** tape is the case in finding 1: discarded, shift 0, drop short. An inverted rim that finds a partner inside 4 cm is worse than shift 0.

### 3. `drop_slot_exclude` stays at the nominal pelvis point, and `ab_bingeo` uses slots

`planner.py:212-217`, `config.yaml:263`, `overlays/ab_bingeo.yaml:33`.

Exclusion runs inside `_drop_slots0` (nominal inner box) against the fixed point `[0.453, -0.308]`, then `drop_slots` adds the shift (`planner.py:175-178`). Right-bin slot 2 **is** that point (near wall + 5 cm, y = inner edge stepped by 6 cm). Match tolerance is 5 mm.

After pelvis dx = +0.07 the slot is `(0.523, -0.308)`, 7 cm from the exclude point, so it is kept. The comment on the constant says that slot is outside the head camera. The success path of this feature (a real translation) drops into the slot the exclude was added to hide. Shift 0 still excludes it, which is why the existing slot tests stay green.

---

## Medium

### 4. Consumers that still read nominal `scene.yaml` bins

Switched, when `planner.bin_shift` is set (planner always; perception only if `geometry_runtime` or `bin.vote` / `live_bins` — `perception.py:1150-1155`):

| Consumer | Where |
|---|---|
| `bin_xy`, `drop_xy`, `drop_slots` | `planner.py:154-178` (nominal point, then `+ shift`) |
| `_in_bin_footprint`, lateral drop margin | `planner.py:338-363` |
| `bin_clearance`, `hits_bin_wall` | `planner.py:229-265` via `_mat_centre` / `inner_lo` |
| carry / lower check | `planner.py:881-893` (those two functions) |
| `_reject`, `_keep_mask`, `_bin_hulls_mask`, `_near_bin` | `perception.py:1184-1216`, `1390-1394` via `_bin_centres_mat` |
| colour-gate boxes and the workspace near edge | `perception.py:1468`, `1488`, `1507` |
| arm choice, nearest-bin order, decision time, brain bin list, endgame “over bin” | `policy.py:984`, `1175`, `1223`, `1420`, `1638` |

Still nominal (not the accessor):

| Consumer | Lines | What a +7 cm bin does |
|---|---|---|
| `Perception._bin_footprint_mask` | `perception.py:530-545`, called from the table-plane fit at `609` | Mask hole sits on the nominal footprint. Real walls stay in the plane sample; a strip of real counter is cut out. `online_pitch_cal` is off in `ab_bingeo`. |
| `Perception._calibrate_bin_rims` | `perception.py:379-388` | Pitch fit targets the nominal far rims. A 7 cm depth error is ~7° at 0.6 m, inside the ±12° search, so it can be absorbed as a fake pitch. Flag off on this overlay. |
| `Perception.calibrate_bins` | `perception.py:808-846`, `914` | Fits an **xy offset on detections** to the fixed walls. It does not write `bin_shift`. See finding 5. |
| `Scene.bin_of` / `bin_inner_box` via `policy._score_bin` | `policy.py:1825-1829`, `scene.py:123-130`, `215-222` | Scoring inner box is nominal. The +7 cm near-wall drop (mat y 0.383) is still inside the nominal inner span `[0.263, 0.467]`, so a good near drop still scores. The rear 7 cm of the real cavity (mat y 0.467–0.537) does not: `bin_of` returns None while a shifted `_reject` says `in_bin`. `verify_placement` is off in `ab_bingeo`; the path is live when that flag is on (`policy.py:1852-1871`). |
| Lid holes / `classify_insert` | `scene.py:147-164`, `187-198` | Lid stays on the nominal left bin. `policy.insert_guard` is on in `ab_bingeo` (abstain), so this overlay does not drop on the lid. |
| `grasp.drop_slot_exclude` | finding 3 | |
| Sim meshes, oracle, `scene/export.py`, layout cards | `sim/aalto_task/scene_cfg.py:192-207`, `sim/oracle.py` | Sim bins are the nominal scene. Correct for sim; they are not the real-table estimate. |

`_mat_centre(name, c)` (`planner.py:225-227`) ignores `c` and re-reads the nominal centre. Callers pass that same centre, so clearance matches today. A caller that passed a different `c` would silently get the accessor centre.

Half-extents (`outer_size`, wall height, thickness) stay nominal everywhere. That is right for a pure translation. A bin that is also the wrong size is not handled.

### 5. `calibrate_bins` is not an input to the accessor, and it can stack with the vote

`perception.py:1150-1155`, `780-809`, `914`, `1295`, `1346`, `1370`; `config.yaml:76-78`; `policy.py:549-567` runs calibration **before** the vote (`619-627`).

The accessor’s only shift source is `planner.bin_shift`, filled by `vote_bins`. `calibrate_bins` writes `xy_offset`, added to every detection, and refuses anything outside `max_offset_m` (0.03 m). A 7 cm placement error is rejected there (“offset outside gate”), so on this specific error the two do not both fire. A 2–3 cm error can: detections move by the offset **and** the bins move by the vote. The colour-gate comment (`perception.py:1499-1502`) still describes calibration as “moves the detections onto the nominal bins” and treats a successful calibration as “an estimate” that **cancels the near hedge** without moving the box (`1504-1510`). `estimate` looks at `live_bins` only; a `geometry_runtime` shift with `live_bins` off does not count.

`ab_bingeo` has self-calibration off, so this overlay does not stack. The flag text (`config.py:107`, `config.yaml:182`) says the accessor follows “the voted shift” and “needs `bin.vote.enabled`”. It does not follow `calibrate_bins`.

### 6. On `ab_bingeo` the new flag does not move the masks or the planner beyond `bin.vote`

`policy.py:108-110`, `perception.py:1153`, `planner.py:150-152`, `overlays/ab_bingeo.yaml:46-47`, `overlays/ab_binvote.yaml`.

The planner applies `self.bin_shift` whether or not `geometry_runtime` is set. `bin.vote.enabled` already set `perception.bin_vote_on`, which already moved `_bin_centres_mat`. `geometry_runtime` adds the same shift for the colour gate without `live_bins` (`perception.py:1153`, `gate=True`). `ab_bingeo` does not enable the colour gate (`config.yaml:165`). Relative to `ab_binvote`, the overlay’s pick, drop, collision, and detection masks are the same code path. The commit message and the FLAGS blurb (“every bin consumer … planner drop/slots/collision”) describe the refactor, not a new planner gate.

---

## Low

### 7. Absurd shifts are only stopped in `combine`, and only by throwing the vote away

`bin_vote.py:66-67` versus `bin_vote.py:22-36`. A hand-set `bin_shift` of 0.5 m is applied (NaNs too). The voter never emits that: over 10 cm becomes shift 0, not 10 cm. An agreed 9.9 cm shift is applied in full, including a pure lateral 9.9 cm, which parks a 4.5 cm object on the side wall of a 21 cm bin (inner half 10.2 cm, inset 5 cm). Forward, the same cap cannot push the near-inset drop out of the **far** wall of a bin that really is +7 cm: drop-to-far-inner clearance stays about 12 cm. It can make the side bins unreachable (nominal centre already r ≈ 0.57 m); IK failure then refuses the plan, which is the safe direction.

### 8. Stale colour-gate hedge

`perception.py:1499-1510`. The local `shift` is computed only for `estimate` and is not what moves the box. The docstring still says “live_bins off = nominal”, which is false once `geometry_runtime` is on (`test_bin_geometry.py:158-163` locks the new behaviour). The 10 cm hedge grows mat −y. The physical error in this review is mat +y.

### 9. Brain bin tag ignores the vote

`policy.py:1637-1638`. Positions come from `planner.bin_xy` (shifted). The string is `"calibrated"` only when `calibrate_bins` succeeded, otherwise `"scene"`, including after a vote.

---

## Frames, units, sign

| | |
|---|---|
| Shift units | metres, pelvis xy, same frame as `planner.bin_xy` |
| Stored on | `planner.bin_shift[name] = (dx, dy)` (`policy.py:627`) |
| Pelvis | `centre_pelvis += (dx, dy)`; `inner_pelvis` mins/maxes `+= (dx, dy)` (`bin_vote.py:30-36`) |
| Mat | `centre_mat += (dy, dx)` because pelvis `(x, y) = origin + (mat y, mat x)` (`bin_vote.py:30-33`, `scene.py:73-80`) |
| Further from the robot | **+pelvis x = +mat y**. Nominal near edge is the small mat y. `test_bin_geometry.py:18-19` and `:67` use this sign and check it. |
| Opposite convention already in tree | `test_bin_vote_perception.py:21` and `test_color_gate.py:268` inject **−7 cm** pelvis x and call that the drop_2 shift. The accessor will apply whichever sign the voter returns. Those tests do not establish the physical sign. |
| z | not shifted. Rim height stays `counter_z + wall_height`. |

The swap itself is correct and covered by `TestAccessor.test_shift_moves_everything_in_both_frames`.

---

## Drop target and carry vs the real bin

Inside the **estimated** bin, by construction, whenever the shift is a pure translation: `_drop_xy0` / `_drop_slots0` are inset from the nominal inner walls, then the same vector is added (`planner.py:157-178`). `test_bin_geometry.py:91-99` checks that. `_adjusted_drop_target` refuses a candidate outside `inner_pelvis` and falls back to that stock point (`planner.py:362-367`).

Against the **real** bin:

- Vote correct and equal to +7 cm: centre-bin release is 5.0 cm inside the real near inner wall. Object fits (2.25 cm radius). `bin_clearance` / `hits_bin_wall` use that same box, so the checker and the wall agree. The release height stays above the rim (`drop_above_rim` 0.04 m, `planner.py:742`) except on the `open` phase, which skips the rim check (`planner.py:877-884`). A matching shift should not plan a grasp point through the wall.
- Vote missing / rejected (the +7 cm vs prior case): release is 2.0 cm outside the real inner wall and 1.7 cm in front of the real outer face. The checker still guards the **nominal** box, whose outer near face is 7 cm closer (pelvis x 0.400 vs real 0.470). The planned grasp point does not enter the real cavity. Fingertips are not checked against the real wall, because that wall is outside the nominal footprint. `hand_points` includes fingertips (`kinematics.py:94-100`); this review did not run FK, so fingertip contact is a residual, not a measured hit.
- Vote sign flipped and accepted: the drop moves further in front of the real bin, and the checker moves with the wrong box (finding 2).
- Per-bin shifts that are not one rigid row: a bin that stayed at the prior is checked at the nominal place while its real wall is 7 cm further. A carry into a neighbour that did move can cross that real wall without a hit. `resolve` only saves the left/right **y** order (`bin_vote.py:75-78`).

Sampling is every 3rd control tick (`planner.py:868`, `881`). A fast segment can step over a thin wall between samples. That predates this commit.

Within the 10 cm cap, a forward overshoot cannot put the near-inset drop out the far side of a bin that is truly +7 cm (about 12 cm of clearance left). The miss is on the **near** side, when the applied shift is short of the truth by more than the 5 cm inset.

---

## Tests

What exists and would catch a regression of the accessor math:

- `test_bin_geometry.py` — nominal vs +7 cm pelvis x in both frames (`53-70`); planner `bin_xy` / `drop_xy` / slots / footprint (`73-103`); one-point `hits_bin_wall` and `bin_clearance` (`105-125`); `_reject` / `_near_bin` / hull mask / colour-gate boxes under `geometry_runtime` alone (`132-163`); flag default and `ab_bingeo` wiring (`166-172`).
- `test_bin_vote.py:150-207` — small shift on drop, slots, clearance, wall; over-max and disagreement stay at the prior (`53-83`).
- `test_bin_vote_perception.py` — reject, keep mask, hulls, near-bin follow `bin_vote_on` for a **−7 cm** shift. Does not pass through `geometry_runtime`.

What those tests do not catch:

| Gap | Why it matters |
|---|---|
| No `rim_votes` / `vote_bins` on a synthetic bin translated +7 cm in mat y | Findings 1 and 2. Injecting `bin_shift` never checks the sign the fitter publishes. |
| No assertion that shift 0 leaves `drop_xy` outside an inner box moved +7 cm | The failure mode is the nominal numbers, which the “flag off is the old numbers” test treats as success. |
| No test that `drop_slot_exclude` is applied in the shifted frame | Finding 3. `test_classical.py:510` locks the nominal right-slot coordinate. |
| No test that `scene.bin_of` / `_score_bin` sees a point in the shifted far half | Finding 4. |
| No test that `calibrate_bins` success does not also move `bin_geometry`, and does not stack on `xy_offset` | Finding 5. |
| `test_exclusion_mask_and_hulls_follow` (`test_bin_geometry.py:151-156`) never calls `_keep_mask` | The 3D keep path is only covered for `bin_vote_on` in `test_bin_vote_perception.py:77-80`. Same `_bin_centres_mat`, so a hull-only break would be caught; a depth-path break under `geometry_runtime` alone would not. |
| Collision tests use `_FakeKin` with one point | They do not plan a carry from the object region to `drop_xy` and test it against a box whose walls are 7 cm further than the shift. |
| No test that `geometry_runtime` without a filled `bin_shift` is a no-op through `ClassicalPolicy._act` | The flag’s real effect on this overlay is easy to mistake for a behaviour change. |

A regression test that would have failed on this branch for the stated hardware error: render or synthesize wall points at nominal mat y + 0.07, run `vote_bins`, then require `drop_xy` inside that inner box and `drop_slots("right")` still to omit the FOV slot. Today the first assert fails (shift 0, drop at mat y 0.313) and, if the vote is forced to +0.07, the second fails (slot 2 kept).
