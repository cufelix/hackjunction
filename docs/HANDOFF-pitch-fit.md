# HANDOFF: camera-pitch fit (Oliver → Timi / Felix)

**Task (Timi, msg f79feed8):** is the 8–23 cm bin offset in the real HF head frames caused by camera pitch?
**Answer: YES.** A pitch offset of the head camera vs `calibration.yaml` explains most of it.

| quantity | value |
|---|---|
| fitted pitch (15 frames, 5 episodes) | **−7.5 ± 1.8°** (range −9.45 … −3.76) |
| rim rms, no pitch correction | 8–23 cm |
| rim rms, pitch-only fit | **2–3 cm** |
| per-episode variation | e12 ≈ −4.2° vs e01 ≈ −9.2° → **not a constant** |

Negative = the real head camera is pitched **UP** (less down-tilt) than the URDF extrinsics assume, so the
bins project farther/higher than nominal. Most likely the waist/head pose differed between teleop episodes.

## Implications (the important part)

1. **Do not bake a pitch constant.** It varies per episode (−4° … −9°). This is evidence FOR the per-episode
   online bin/camera calibration (`perception.self_calibration`, `oliver/bin-detect`), not a fixed offset.
2. The residual 2–3 cm after pitch is the true bin-placement / wall-height uncertainty — within the
   bin-detect `max_offset` gate.
3. Pitch / camera dz / counter_z are **partially degenerate**: adding dz barely changes rms (pitch+dz fits
   −5…−12°, dz +1…+7 cm). The angle is model-dependent by ~±2° (lab geometry zc=−0.05, wall=0.045 gives
   −9.4 ± 2.3°), but "several degrees of pitch, not moved bins" is robust to that choice.

## Method

- Bins assumed at nominal `scene.yaml` xy. Camera pose perturbed by pitch dθ about its optical x
  (T′ = T·Rx(dθ)), optionally + translation. Models: `pitch`, `pitch_z`, `pitch_xz`.
- Far-rim pixels = top orange edge per column. The 3 bins merge into ONE orange component (tape + shared
  hue), so components can't separate them; the envelope is split by u at the nominal projected bin centres.
  Wall-like columns only (vertical extent ≥ 15 px, rim in the top 42% of the frame); real-bin hue gate
  H ≤ 9 or ≥ 170 (objects are H 9–15).
- Each rim pixel → ray through the perturbed pose → rim plane z = counter_z + wall_height. Cost = trimmed
  (worst 10%) rms distance to the nominal far-rim line (x = bin_x + half, y clipped to the bin span).
  Nelder-Mead, 3 starts, |dθ| ≤ 20°, |shift| ≤ 20 cm.
- The pitch signature is the **differential y-shift of the side bins vs the centre bin** — a pure row
  translation cannot produce it.

**Validation (blind, synthetic):** renders with known pitch −6/−3/0/+3/+6° are recovered to ≤ 0.1° with
rim rms ≤ 0.6 cm (`--synth-only`, and `tools/test_pitch_fit.py`, 9 tests).

## Per-episode results (scene.yaml: counter_z = 0.010, wall = 0.060)

| ep | no-pitch rms | pitch fit | pitch rms |
|----|--------------|-----------|-----------|
| e01 | ~22.3 cm | −9.2° | ~2.8 cm |
| e04 | ~21.3 cm | −8.8° | ~2.0 cm |
| e08 | ~19.2 cm | −8.1° | ~2.1 cm |
| e12 | ~9.1 cm  | −4.2° | ~2.8 cm |
| e16 | ~17.1 cm | −7.4° | ~2.9 cm |

Full per-frame JSON: `/var/tmp/oliver-handoff/pitch_fit_results.json` (also reproducible with the command below).

## Run it

CPU-only: numpy, cv2, scipy. No GPU, no box needed if the frames are local.

```sh
python3 tools/pitch_fit.py --synth-only                      # validate the fitter (~3 s)
python3 tools/pitch_fit.py --frames /var/tmp/hf_g1/frames --out results.json
python3 tools/pitch_fit.py --frames <any dir of head RGB pngs> --out results.json
python3 tools/pitch_fit.py --counter-z -0.05 --wall-height 0.045   # geometry sensitivity
python -m pytest tools/test_pitch_fit.py -q                  # 9 tests
```

**Felix offer (msg 30f24ad7):** run it on box C's full `/data/datasets/g1_pick_cube` frame set to confirm
the per-episode pitch spread on more than these 15 frames. Extract cam_high frames per episode first.

## Reuse in online calibration (Timi)

`RimFit`, `far_rim_bins`, `rx` are importable (`tools/pitch_fit.py`). Suggested use: on the first settled
look of an episode, fit pitch from the bin rims and feed it as the camera-pose prior into
`self_calibration` / bin-detect (instead of assuming the URDF pose). Guard it like the other calibrations:
require 3 bins, rms below a gate, |dθ| ≤ 20°, else keep the configured pose. One log line per episode.

## Files

- `tools/pitch_fit.py` — fitter + CLI (branch `oliver/pitch-fit`, base `2380e83`)
- `tools/test_pitch_fit.py` — 9 CPU tests (synthetic recovery, extraction, guards)
- bundle: `/var/tmp/oliver-handoff/pitch-fit.bundle` — `git fetch <bundle> oliver/pitch-fit:oliver/pitch-fit`
- results: `/var/tmp/oliver-handoff/pitch_fit_results.json`
- local frames analysed: `/var/tmp/hf_g1/frames/*.png` (15 cam_high frames, episodes e01/e04/e08/e12/e16)

Nothing pushed (team rule). Warren msgs: f79feed8 (task) → 3fe2d6a5 (result) → 30f24ad7 (offer) → this handoff.
