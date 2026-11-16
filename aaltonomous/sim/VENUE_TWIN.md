# Venue twin: `venue_20261115`

Opt-in sim variant that looks like the finals table of 2026-11-15. On the real robot the policy grabbed a
red-orange paper-wrapped bin instead of a cube; the default sim bins are plain orange boxes
(`scene.yaml` `bins.rgb` [230,110,30]) and never showed this. **Nothing changes unless `AALTO_DR=venue_20261115`.**

## What it changes

Selected like every other DR variant: `AALTO_DR=venue_20261115` when the sim starts (config
`aaltonomous/sim/dr/venue_20261115.yaml`, loaded by `aalto_task/dr.py`). `evaluate.py` needs no flag: it reads the
sim's DR config name (variant label `venue_20261115`) and calls `randomize` with `seed` per trial.

| Surface | Sim colour |
|---|---|
| Bin outer walls + rim | red-orange paper (`bin_outer_rgb`) |
| Bin inside (floor + inner faces) | cardboard brown (`bin_inner_rgb`) |
| Table (counter box and the printed mat, whose texture is replaced by a flat colour) | `table_rgb` |
| Object classes red / yellow / blue | venue orange / yellow / blue (`object_rgb`; the "red" class IS the orange parts) |
| Jitter per episode | +-4 % per channel and 0.9-1.1 brightness per surface (paper 0.65-1.1), key light 0.9-1.5x, plus the usual pose / camera noise / mass / friction |

Hook: `hooks._apply_venue` sets `diffuseColor` of the existing `UsdPreviewSurface` shaders. The bin walls are single
3 mm boxes (one material), so "orange outside, brown inside" is done with a 0.6 mm visual-only liner cube (no collision)
on the inner face of each wall, created once under the wall prim (`dr.bin_liner_specs`). No USD asset or geometry
is edited; bin physics and sizes are unchanged. Sampled sRGB values are converted to linear before being used.

## Sampled colours (sRGB 0-255)

Source: `knowledge/humanoid/real_runs/larp-interns_20261115_091835/episode_000.mp4`, 640x480, frame at t=5 s
(`ffmpeg -ss 5 -i episode_000.mp4 -frames:v 1 f_5.png`; robot arm not in view). Median over the patch (x,y,w,h in px):

| Surface | Patch | Measured | Used |
|---|---|---|---|
| Bin paper, camera-facing front faces (3 bins) | L (40,215,140,28) / C (250,238,160,22) / R (490,228,130,22) | (99,40,25) / (103,42,32) / (106,47,28) | hue of these, brightness of the lit side face below |
| Bin paper, lit side face (left bin) | (200,90,25,100) | (172,80,61) | **(172,68,47)** |
| Bin inner back wall (centre, shaded) / floor | (290,110,100,40) / (50,125,120,40) | (90,81,71) / (130,141,127) | **(126,113,99)** (shaded wall x ~1.4) |
| Table | (130,305,100,35), (20,420,200,50), (0,60,15,100), (330,290,150,30) | (143,135,131), (137,141,136), (137,139,133), (146,135,130) | **(141,136,132)** |
| Blue cube | top (268,332,32,12) / front (255,360,50,28) | (84,134,226) / (82,132,225) | **(84,134,226)** |
| Orange cube (class red) | top (392,318,30,12) / left (375,340,12,30) | (244,147,81) / (133,57,25) | **(244,147,81)** |
| Orange cylinder | top (380,405,55,25) | (250,168,114), overexposed | not used (same class) |
| Yellow cube | top (470,312,40,14) / left (455,340,18,20) | (213,255,182) overexposed / (151,118,53) | **(232,205,95)** (hue of the sides, brightened) |
| Yellow prism | top (280,440,40,20) | (209,253,180) overexposed | same class colour |

Caveats: the camera has a neutral/greenish white balance (table reads grey, not brown) and the paper is 0.6x darker on
camera-facing faces than on the lit side face; the "Used" paper and yellow values are therefore derived (measured
hue, lit brightness), not raw medians. The raw numbers above are all you need to re-derive them; edit the yaml to change.
Re-check with `aalto snap` that a sim head frame looks like the real one before trusting a result.

## Run it (GPU box, container `aalto`, see `aaltonomous/cloud/QUICKSTART.md`)

```bash
cd /opt/aalto && git fetch && git checkout timi/venue-twin   # or copy the branch into the container
aalto test                                    # offline unit tests incl. test_venue.py (should be OK)
aalto stop; AALTO_DR=venue_20261115 aalto sim # restart the sim with the variant (log /tmp/aalto_sim.log must say
                                              # "domain randomization ON: .../venue_20261115.yaml")
aalto snap --out /tmp/venue_snap              # look at head.png: orange paper bins outside, brown inside, flat table
# the policy image under test (serves ws on 8000), 20 trials with the same seeds as the nominal baseline:
aalto eval --container <policy-image:tag> --team venue-twin --task sort_fixed --trials 20 --seed 1000 \
    --out runs/venue --variant venue_20261115
# policy already served elsewhere: aalto eval --policy ws://localhost:8000 --team venue-twin --task sort_fixed --trials 20 --seed 1000 --out runs/venue
# identical call without the variant for the A/B baseline:
aalto stop; aalto sim; aalto eval --container <policy-image:tag> --team nominal --task sort_fixed --trials 20 --seed 1000 --out runs/nominal
# same thing without the wrapper: python -m aaltonomous.sim.evaluate --container <policy-image:tag> --team venue-twin \
#     --task sort_fixed --trials 20 --seed 1000 --out runs/venue
```

Results: `runs/venue/<team>_sort_fixed_venue_20261115_<stamp>/eval.json` (per trial outcomes, score, `dr_params`) and
the per-episode logs (`episode_NNN.npz` / `.jsonl`) next to it (`deploy.runner.Recorder`); the policy's own logs hold
its grasp targets and release_confirm messages.

## What counts as "reproduced the failure"

Compared with the nominal run on the same seeds, in the venue run at least one of:
- the commanded grasp / hand target (arm + hand close in the episode log) lies on a bin wall or rim rather than on a cube,
  and the hand closes on it (video: hand on a box wall);
- the hand closes on the bin and the object count in the bin does not change; the policy's release step then reports
  "hand still not empty" (release_confirm) or never releases;
- scored outcomes show cubes left on the counter (`counter`) while the policy spent the episode on a bin, with
  score clearly below the nominal run.
It is NOT reproduced if the venue run behaves like the nominal one (same grasp targets on cubes). In that case try
`bin_outer_brightness: [1.0, 1.0]` + `bin_outer_rgb: [200, 90, 50]` (closer to the orange cube) and report back.
After a fix, rerun the same command: score back to nominal, no grasp on a bin wall.
