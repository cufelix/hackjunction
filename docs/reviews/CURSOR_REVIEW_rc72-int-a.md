# Adversarial review: `team/felix/rc72-int-a` vs `team/felix/rc7-pre`

Scope: `git diff team/felix/rc7-pre...team/felix/rc72-int-a -- aaltonomous tools`. `tools` is empty. The diff is the three r22 opt-ins on the rc7 tree: `policy.verify_landing`, soft squeeze / soft held, and novel perception + generic grasp + instruction match + obstacles. Line numbers are `team/felix/rc72-int-a`.

`finals_r22`, `finals_r22_verify`, `finals_r22_softgrip`, and `finals_r22_novel` all set `policy.generation_workers: true` and `bin.vote.async_extract: true`, so the look runs on a snapshot worker (`policy.py:198-202`, `policy.py:1211-1233`). Only `_LOOK_COMMIT_FIELDS` (`policy.py:98-105`) is published back (`policy.py:1346-1352`).

With every new flag off, the new branches stay off: `novel_objects` is not called, `_landing_on()` is false, `soft` stays `None`, and `soft_overclose` is not applied. `soft_overclose` (`config.py:628-640`) clips to the URDF limits shrunk by `soft_limit_margin_rad` 0.03. No new state machine parks in `_hold()` with no exit. On the r22 overlays `act()` polls the look thread and returns a hold; the occupancy scan and the second detect run on that thread.

## 1. High — landing state is computed on the worker and never committed, so occupancy, the refund cap, and the regrant cap do not exist on r22

`policy.py:98-105` (commit list), `policy.py:1629-1663` (`_landing_review_inner`), `policy.py:1677-1724` (requeue and regrant), `policy.py:3036-3039` (`_start` arms `_landing` on the live policy).

`_landing`, `_occ_prev`, `landing_refunds`, and `landing_regrants` are not commit fields. The worker is a shallow copy. `worker._occ_prev = occ` and `worker._landing = None` and `worker.landing_refunds += 1` rebind the worker only. `attempts` / `attempt_tracks` / `slot_used` are commit fields, so the pops and the slot marks do stick.

`finals_r22_verify` turns `verify_landing` on next to `generation_workers`. After a release:

- Live `_occ_prev` stays `None`. `classify_landing` (`landing.py:56-61`) builds an empty delta. A bin that gained volume is never `ok` and never `wrong_bin`. The depth path, including the shadowed-bin case, does not run on this overlay.
- Live `_landing` stays armed. Every later look classifies that same release again. `landing_events.append` and `bounce_count[...] =` mutate the shared containers, so the log grows a duplicate event per look.
- `landing_refunds` on the live policy stays 0, so `landing_max_refunds` (4) never trips. `landing_regrants` stays 0, so `landing_regrants_max` (2) never trips. Each look that still sees a same-colour counter detection more than `near_m` 0.05 from the original `pre_xy` refunds again and plans another pick. `max_total_attempts` 18 is the only stop, i.e. the rest of the 120 s on one roller. A miss into another bin is −1; `wrong_bin` is log-only and, with an empty delta, is not even logged.

A worker that started before `reset` mutates the containers it captured. `reset` (`policy.py:426-432`) replaces those containers and bumps `_look_generation`, and a mismatched generation is not committed. The cross-episode write is contained. The same-episode commit gap is the failure.

Minimal fix: add `_landing`, `_occ_prev`, `landing_events`, `landing_refunds`, `landing_regrants`, `bounce_count`, and `_landing_excl_logged` to `_LOOK_COMMIT_FIELDS`, and deep-copy `_landing` / `bounce_count`. Re-run `test_landing` through `ClassicalPolicy(..., overlay="finals_r22_verify")` so the worker, not a direct `_look`, is what gets asserted.

## 2. High — a bounce refunds the old pose and keeps the drop slot

`policy.py:1683-1692` (refund searches `src_xy` only), `policy.py:1698-1702` (`dataclasses.replace` moves the detection, not the track), `policy.py:1742-1751` (`_settle_slot`), `policy.py:3090-3092` (slot marked at `_start`). `grasp.drop_slots: true` and `grasp.drop_slot_grid: true` on every r22 overlay. `same_object_radius` is 0.03 (`config.yaml:562`). The grid is two rows of `drop_slot_count` 3.

`classify_landing` calls a same-colour detection a bounce when it is more than 0.05 m from every `pre_xy` (`landing.py:69-73`). The unit test's own bounce is about 16 cm (`test_landing.py:71-80`).

The refund pops one `attempts` entry near `src_xy` and decrements the track near `src_xy`. It does not move either record onto `det.top`. The next `_start` appends a fresh count at the new grasp. A second roll past `retry_assoc_radius` (0.06 m with `retry_vary_grasp` on) sees count 0, so `max_attempts` 2 and `max_attempts_vary` 3 never fire. Finding 1 means the refund counter does not accumulate either, so this repeats until 18 starts.

`_settle_slot` returns the slot only when a same-colour detection is still within 0.06 m of the grasp. A 16 cm roll fails that test. `slot_used` is committed, so the slot stays spent. One roller walks the grid. With `drop_slot_overflow: counter`, the next object of that colour is left on the counter (0).

Minimal fix: on a bounce, move the matched attempt and the same-colour track onto `det.top[:2]`, including once refunds are exhausted. If `slot_pending` is this colour, `slot_used[bin].discard(slot)` and clear `slot_pending` before `_settle_slot`. Add a two-hop test at the 16 cm spacing that also turns `grasp.drop_slots` on.

## 3. High — soft abort samples a mid-lift tick, and the r22 empty-close bias is subtracted twice

`policy.py:3427-3438` (`_pop` cuts a chunk only when `carry` is set), `policy.py:3298-3308` (close abort uses `soft_margin` 0.12), `policy.py:1166-1172` and `policy.py:1084-1092` (soft baseline finishes before held calibration starts), `held.py:221-238` (`excess = score - ratio * travel`, and `score` already subtracted `self.reference`).

`motion.chunk` is 10 (`config.yaml:514`). The chunk cut at `lift_end` runs only when `held.read_through_carry` is on. `finals_r22_softgrip` leaves that flag off, and it leaves `grasp.held_detector: finger_sum` and `grasp.held_calibration.enabled` on.

Close-sample smear. The soft read is `hold_start + ceil(1.18 * 30)` (`held.py:179-180`). A chunk that starts two ticks before that read still emits eight ticks (0.27 s) of `lift_check` before the next `hand_q` is classified. The abort then applies the close margin 0.12 to a reading taken after the arm has started the slow lift. A grasp whose excess falls from 0.15 to 0.10 during that window is above the lift margin 0.08 and below 0.12, so `_abort_empty_grasp` opens the hand. The object drops. Score 0, or −1 if it falls into a bin.

Reference applied twice. On this overlay the episode-start order is fixed: the soft baseline is queued at the end of `settle` and `soft_calibrate` runs while `reference` is still zeros (`ready` is false, so the abort is correctly suppressed during the baseline). Held calibration then closes to `closed_tight`, not the over-closed command, and on success stores `reference = command - measured`. `SOFTGRIP.md` measures that shortfall at 0.06–0.43 rad per joint. The ratio already equals the full empty over-closed error, because it was computed with a zero reference. At the pick, `excess = extra_stall - (that empty pair error)`.

Take the top of the measured band, 0.43 rad on each joint of a finger (pair 0.86). A cube that stalls the fingers at the closed posture, while the command is over-closed by 0.50 and 1.00 (`SOFTGRIP.md` table), adds `(0.50 - 0.43) + (1.00 - 0.43) = 0.64` rad beyond the empty hand. `excess = 0.64 - 0.86 = -0.22`, which is under 0.12. The hand opens on a real grasp. The soft tests construct a detector with `episode_start_reference` of zeros and never load `held_calibration`, so they stay green.

Minimal fix: in `_pop`, stop the chunk on `close_end` as well as `lift_end`, with `read_through_carry` off. In the soft branch of `held()`, do not subtract `self.reference`; the empty-close ratio is the bias. Keep `_held_abort_due`'s `not det.ready` guard. Add one test that loads `finals_r22_softgrip` (both calibrations on) and asserts a 0.43 rad/joint empty lag plus a cube stalled at `closed` is held.

## 4. High — one novel colour match is sent to the first bin named in the sentence, and an oversized blob deletes the known cube

`novel_objects.py:220-227` and `novel_objects.py:273` (size gate and supersede), `novel_objects.py:285-300` (`_BIN.search` is the first bin word; one surviving match wins), `planner.py:42-46` (`lookup_detection_bin` returns `novel_target_bin` for any novel detection). `finals_r22_novel` turns on all four novel flags on top of the sort policy. Known-size window is 0.035–0.070 m (`novel_objects.py:216`, default at `novel_objects.py:220`).

`match_instruction("yellow left blue center red right", [one red novel])` has no unknown token (`cube` is reserved, the colours and bin names are reserved). `colours` is yellow, blue, and red, so the single red novel matches, and the destination is `left`, the first bin word. Finals fixed mapping is yellow left, blue center, red right. The red object is placed in the left bin (−1).

The same look deletes the known detection when the depth blob is outside 3.5–7.0 cm and its centroid is within `known_match_px` 45 of that detection (`novel_objects.py:219-227`, `273-274`). A 5 cm cube whose 2nd–98th percentile footprint comes out at 8 cm is removed from the colour sorter. If the sentence has an extra word (`cubes`, `objects`), the matcher abstains, `novel_target_bin` stays `None`, and `_choose` skips it (`policy.py:2018-2019`). That cube scores 0. If the sentence is the terse colour list above, it scores −1 in the first bin.

`test_green_ball_left_maps_exactly` only checks `"green ball left"`. `test_unmatched_novel_is_not_picked_and_is_an_obstacle` uses one bin and a black obstacle. Neither sentence is a three-bin sort, and neither test asserts that a known cube outside the 7 cm window survives.

Minimal fix: pair each colour token with the bin word that follows it, and refuse a match when more than one bin is named unless that pairing is unique. Do not supersede a known-family detection; leave it on the colour path when the blob is merely outside `known_size_m`. Add the sentence `yellow left, blue center, red right` with one red novel and assert the target is `right`, plus a 8 cm red blob on top of a known red cube that asserts the known detection is still the one planned.

## 5. High — a bounce rebuilds the detection with `dataclasses.replace` and drops the novel grasp

`policy.py:1698-1702`. `Detection` declares `novel`, `label`, `footprint`, `height_m`, `ungraspable` (`perception.py`, the new fields). `novel_target_bin`, `graspable`, `generic_required`, `generic_width`, `footprint_dims` are assigned in `augment_detections` (`novel_objects.py:259-271`) and are not fields. `dataclasses.replace` keeps the fields and drops the rest.

`_match_novel` runs before the landing review (`policy.py:1455-1462`) and sets `novel_target_bin` on the original object. The bounce then swaps in the replaced detection. `lookup_detection_bin` sees `novel=True` and `novel_target_bin` missing, returns `None`, and `_choose` leaves the object on the counter (0).

With finding 1 the pending release never clears, so any later same-colour novel that sits more than 5 cm from the original `pre_xy` is rebuilt this way on every look. A novel that was the thing that bounced is never retried. A novel that appears after a same-colour cube was placed loses its instruction bin for the rest of the episode.

Minimal fix: copy the dynamic attributes in `_landing_requeue` (or store `novel_target_bin`, `graspable`, `generic_width`, `generic_required` on the dataclass). Retarget the attempt as in finding 2 so the rebuilt detection is the one the track follows.

## 6. Low — the new tests pass on a path the overlays do not run

- `test_landing.py` builds `ClassicalPolicy` with only `policy.verify_landing: true` (`test_landing.py:19-41`) and calls `_look` / `_start` directly. That writes the live policy, so findings 1 and 5 stay green. `test_bounce_to_the_counter_is_repicked_with_jitter` asserts `attempts == []`, which is the popped old record, and never checks the count at the bounced pose or `slot_used`. `drop_slots` is off in `LandingBase`.
- `test_follows_the_runtime_bin_pose` (`test_landing.py:153-160`) stubs `detect` and never calls `_look`. The two `inside_bin` asserts do not check that the nominal spot is planned.
- `test_flag_off_is_stock_and_perfect_landings_make_the_flag_a_no_op` (`test_landing.py:231-237`) compares `{}` with `{policy.verify_landing: false}` in this same build. Both take the flag-off path. It does not compare trajectories to `team/felix/rc7-pre`.
- `test_act_never_blocks_on_the_landing_review` (`test_landing.py:242-260`) uses `threaded=True` and not `generation_workers`, sleeps 20 ms while the job is in flight, and allows 50 ms. The loop is 30 Hz; the bar for this review is 37 ms.
- `test_soft_grip.py` `SOFT_ON` (`test_soft_grip.py:20-21`) does not set `grasp.held_calibration.enabled`. `test_act_latency_is_unchanged_by_the_flags` compares medians of `_pop` and allows +2 ms; it never runs the held-calibration close.
- `test_novel_objects.py` never loads `finals_r22_novel` and never passes a multi-bin sort sentence (finding 4).

Minimal fix: one episode per overlay (`finals_r22_verify`, `finals_r22_softgrip`, `finals_r22_novel`, and all three flags together) that goes through `act()` with `generation_workers` left on, asserts a two-hop bounce shares one track and an empty `slot_used`, asserts the soft close sample is taken on the last hold tick with both calibrations enabled, and asserts the three-bin sentence does not move a red novel to the left bin. Tighten the latency bound to 37 ms with no sleep inside the measured call.
