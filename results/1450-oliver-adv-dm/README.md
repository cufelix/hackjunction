# Adversarial "real-like" sim: material for the organisers' readiness DM

The stock simulator was edited to match the finals setup; tooling is in `tools/adv/` on branch `oliver/adv-sim`.
- **Real colours:** objects render in the real orange/yellow and bins in the real orange, with bin tape.
- **Real bin offsets:** the bins are shifted dx 7 ± 3 cm.
- **Camera error:** the head-camera pitch error measured on the real robot is applied.
- **Image noise:** lighting/exposure/white-balance variation, plus depth holes and noise.

Profiles: `nominal` (stock sim), `real_mild` (colours + bins), `real_hard` (+ camera error, lighting, depth noise).

## Results, `cand_r16_nopp` (current sim submission), sort_fixed, paired seeds from 1000

| Profile   | Seeds | Mean score / episode | Correct objects | Median episode (sim s) |
|-----------|-------|----------------------|-----------------|------------------------|
| nominal   | 5     | +3.8                 | 19/20           | 48.8                   |
| real_mild | 5 (10 running) | +0.2         | 1/20            | n/a                    |
| real_hard | 10    | ~+0.2                | 2/40            | 88.5                   |

The stock-sim baseline on the same seeds is +3.4, so the edited sim at `nominal` is faithful.

**Why objects are lost on real_hard** (38 of 40; `tools/adv/loss_breakdown.py`):
- 42 % wrong position (camera pitch error).
- 42 % phantom objects (orange bin rims read as objects).
- 8 % wrong colour, 5 % dropped in carry, 3 % no plan.

**What we do with it:** each fix is A/B-tested in this sim against these numbers before it is turned on: online
camera-pitch calibration, colour gate with a 3D bin footprint, hand-eye calibration from the visible hands, and bin vote.

## Images
- `adv_profiles_side_by_side.png`: head camera, nominal | real_mild | real_hard, same seed.
- `real_hard_head.png`: head camera in the real_hard profile.

**Known gap:** one object still renders blue rather than the real purple. A fix is in progress before these images are used externally.
