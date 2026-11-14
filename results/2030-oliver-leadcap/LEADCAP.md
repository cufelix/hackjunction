# Lead-cap offline check (mock env, 6 seeds, 120 s, no commits)

Trees: rc7-pre `/home/oliver/work/leadcap` (b65de8e, overlay finals_r22). Softgrip = `/home/oliver/work/leadcap-sg`
(origin/felix/rc72-softgrip + local merge of rc7-pre, since softgrip alone deadlocks in staging; overlay finals_r22_softgrip,
no such overlay exists in rc7-pre). Script: `leadcap_run.py`, raw logs `out/`. Filter `max_clipped_steps` raised to 1e9 so every
clip is counted; "abort" = more than 30 clipped steps in an episode (real filter would latch and hold).

safety.py: `hand_lead`/`arm_lead` go through `_clamp` -> `reasons.append(...)` -> `clipped_steps += 1` (safety.py:225-230, 239-246).
A lead clamp therefore IS a clipped step, and > 30 aborts. Lead is vs the MEASURED joint, not the last command.

Modes: stiff = mock tau 0.05; soft = hand_tau 0.133 s (kd/kp of Dex3 1.5/0.2); softblock = soft + a held cube stops the
finger at closure 0.8 (ASSUMPTION, mock has no contact; the real overclose would sit against the cube like this).

Totals over 6 seeds (29 objects). picks/placed/clips/abort-episodes/bad drops.

| overlay | mode | config | picks | placed | clips | abort eps | bad drops | clip reasons |
|---|---|---|---|---|---|---|---|---|
| finals_r22 | stiff | a,b(0.3),c(0.15),d(arm+hand 0.3) | 28 | 28/29 | 0 | 0 | 0 | none |
| finals_r22 | soft | a, b, d | 28 | 28/29 | 0 | 0 | 0 | none |
| finals_r22 | soft | c hand 0.15 | 28 | 28/29 | 1196 | 6/6 | 0 | hand_lead (thumb_2, index_0/middle_0, thumb_1) |
| finals_r22 | softblock | a, b, d | 28 | 28/29 | 0 | 0 | 0 | none |
| finals_r22 | softblock | c hand 0.15 | 28 | 28/29 | 5395 | 6/6 | 0 | hand_lead (thumb_1 4451, thumb_2, index_0/middle_0) |
| softgrip | softblock | a default | 28 | 28/29 | 24 | 0 | 0 | hand_position (index_0/middle_0, 1 step/episode) |
| softgrip | softblock | b hand 0.3 | 28 | 28/29 | 5542 | 6/6 | 0 | hand_lead (index_1/middle_1, index_0/middle_0), hand_position, hand_velocity |
| softgrip | softblock | c hand 0.15 | 28 | 28/29 | 7136 | 6/6 | 0 | hand_lead (all finger joints), hand_position, hand_velocity |
| softgrip | softblock | d arm+hand 0.3 | 28 | 28/29 | 5542 | 6/6 | 0 | same as b; arm_lead never fires |
| softgrip | soft (no block) | a..d | 73 | 0 | 24 (c: 3414) | 0 (c: 6) | 47 | invalid: mock never blocks, finger_sum detector reads every close as empty (identical in a) |

Notes
- finals_r22 baseline (no overclose): hand_max_lead 0.3 and arm_max_lead 0.3 never clip, even with lag. 0.15 clips on every pick in soft modes.
- `arm_lead` never fired in any run (arm_max_vel 1.2-ish rad/s x lag stays under 0.3).
- Softgrip overclose is what breaks at 0.3: thumb/finger targets sit past the cube-contact angle, measured stays at contact, lead > 0.3 for the whole squeeze -> thousands of clips, abort.
- Softgrip default (no lead cap) already has 24 clipped steps (hand_position, 1/episode): fails `--gate-max-clips 0` on its own. Separate finding.
- Dropped objects: none attributable to the cap (bad drops 0 in all valid modes). After an abort the real filter holds output, so the pick in progress would stall instead; mock here ignores the abort.
- Caveat: soft model is first-order lag only; real Dex3 stiction/velocity limit and contact angle are not modelled. softblock contact closure 0.8 is a guess.

## Smallest fix (not implemented)
Policy flag `hand.lead_guard: <lead - margin>` (e.g. 0.25 for a 0.3 cap): clamp each emitted hand target row to
`obs["hand_q"] +- (lead - margin)` before it leaves the policy.
- Where: `policy.py:755 act()`, next to the `_im_emit` mirror step (apply after it, so index==middle is preserved), or inside
  `policy.py:790 _limit_motion_output` which already reads `obs["hand_q"]` / `obs["arm_q"]` (`start()` helper).
- Chunks (k,14) are validated per tick against the measured q at that tick, so clamp every row against the chunk-start
  measured q (conservative); the soft close then advances by `lead` per chunk, so close_hold_s / held-detector waits need to cover that.
- For soft overclose, cheaper alternative: cap the overclose depth in `planner.py:950` / `config.py:623 soft_overclose` to lead-margin
  beyond the contact angle instead of clamping every tick.
