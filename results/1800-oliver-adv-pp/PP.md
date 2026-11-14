# Principal point: stock sim vs real-camera pp (rc1)

Box g, sim instance c, rc1 @ 7a4168e9 (overlay rc1, `perception.principal_point: file`), sort_fixed,
seeds 1000-1009, paired. Held-out seeds (seed % 4 == 3): 1003, 1007.

- A = `adv_nominal`: stock Isaac render, principal point at the image centre (319.5, 239.5), while the
  policy reads calibration.yaml (325.04, 250.93). A ~6 / ~11 px mismatch.
- B = `adv_pp`: the render is shifted so the image principal point equals calibration.yaml, which is what
  the real D435i delivers if the factory intrinsics hold.

| | mean | std | correct / wrong / counter of 40 | median sim s |
|---|---|---|---|---|
| A nominal | +2.60 | 1.43 | 27 / 1 / 12 | 66.6 |
| B real pp | +3.80 | 0.75 | 38 / 0 / 2 | 51.9 |

Paired B-A: **+1.20 (se 0.31)**, wins 7, losses 0, ties 3. Tuning seeds +1.00 (n=8), held-out +2.00 (n=2).
0 clips, 0 stops, gate PASS in both.

## What it means

- The stock sim under-scores rc1 by about 1.2 points per episode because of a render artefact that the real
  camera does not have. With the image matching the calibration file, rc1 places 38/40 with 0 wrong bins.
- `principal_point: file` (rc1 default) is the right setting for the real robot. Do not ship `center`.
  rc7's `sim_center_principal_point` (profile=sim only) is safe for the real robot.
- Any sim A/B where one arm centres the principal point and the other does not is not a fair comparison;
  check this before reading rc1 vs rc7 on box A.

Files: `compare.txt` (paired table), `nominal_episodes.json`, `pp_episodes.json`.
