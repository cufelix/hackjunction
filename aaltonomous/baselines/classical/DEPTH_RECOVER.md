# Depth blob recovery

`perception.depth_blob_recover` is a default-OFF recall path for objects that
have valid head depth but were missed by the normal HSV connected components.
It runs only from `Perception.detect()` on the policy's scheduled look updates,
not on ordinary control ticks. The real runner's threaded policy mode runs it
in the existing look worker, while the recovery pass itself is budgeted below
5 ms.

## Design

1. Downsample aligned head depth by `stride` (default 6, an 80×107 map).
2. Reuse the existing counter-plane fitter. If the fit is unavailable, use the
   known `scene.counter_z` horizontal plane.
3. Keep points 1.2–15 cm above that plane, within the existing mat/workspace
   geometry, outside current bin footprints, and outside the projected FK
   arm/hand mask. Cluster the small height map with 8-connected components.
4. Skip a component matching a normal colour detection by IoU or image-centroid
   distance. Classify the remaining blob using the active `hsv_set`, lowering
   only its saturation floor. A best-class confidence and runner-up margin are
   both required; otherwise the blob is unknown and no candidate is emitted.
5. Estimate the normal metric top centre from depth. The optional
   `shape_hint` uses footprint aspect/fill and top flatness conservatively. It
   can label a strongly elongated flat prism footprint, but abstains on the
   ambiguous square-vs-circle case at the default stride.

Internal `red` and `blue` labels correspond to the real orange and purple
pieces. The recovery path never emits a new/open-set colour.

## Parameters

All tuning is under `perception.depth_blob_recover_params` in `config.yaml`.
The main gates are `min_height_m`/`max_height_m`, `min_blob_cells`,
`sat_floor`, `min_colour_conf`, `min_colour_margin`, `match_iou`, and
`match_centroid_px`. The default stride-6 implementation has a synthetic timing
test requiring median recovery time below 5 ms after warm-up.

## Sim A/B

With the Isaac sim and RGB-D service running, use identical seeds:

```bash
python -m aaltonomous.baselines.classical.sim_eval --seeds 0 1 2 3 4 5 6 7 \
  --log /tmp/depth-recover-off.jsonl

python -m aaltonomous.baselines.classical.sim_eval --seeds 0 1 2 3 4 5 6 7 \
  --set '{"perception.depth_blob_recover": true}' \
  --log /tmp/depth-recover-on.jsonl
```

Compare score/correct count, misses, wrong placements, and the
`[depth_recover]` log (`components`, `matched`, `unknown`, `accepted`, and
`elapsed_ms`). Keep the flag off unless the ON arm improves recall without new
wrong placements and stays below the timing budget.
