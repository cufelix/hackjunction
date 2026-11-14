# results

Experiment results for the Junction 2026 sim work. Each experiment is one folder.

## Convention

```
results/<HHMM>-<owner>-<experiment>/
  README.md        setup, results table, conclusion
  summary.json     machine-readable numbers (per arm, per object, per shape)
  *.mp4            at most 3 compressed sim videos (<= 20 MB each, 640 px, crf 30)
```

- `<HHMM>` is the start time of the run (local time, 24 h), `<owner>` is the person who ran it, `<experiment>` is a short slug.
- The README states the engine commit, overlay flags, task, seeds and gate, so the table can be read on its own.
- Some folders also keep a `report.md` with the raw per-episode breakdown. It is optional.

## Folders

| folder | content |
|---|---|
| `0400-timi-pinch-v1-vs-demo` | pinch posture arms: pinch_v1 vs pinch_demo, sort_fixed |
| `0440-timi-pinch-yaff-slots-hybrid` | pinch + y_affine, slots dropped, hybrid posture (arms 3-7) |
| `0521-timi-baseline-clock-check` | baseline sanity: sim clock vs wall clock on Modal |
| `0915-timi-submission-arms-retry-vary` | submission baseline, depthgate, pp_center, prism_offset, retry_vary (seeds 1000-1019) |
| `1056-timi-cand-r16-submission-nopp` | candidate r16 (retry_vary_slots + pp_center) vs submission, seeds 1040-1069 |
| `1206-timi-cand-r16-nopp-c036154` | cand_r16_nopp at c036154 (retry safety hardening) |
| `1400-timi-language` | language lane: open-vocab eval, normalization, sealed re-checks |
| `1200-timi-baseline-videos` | sim footage of an earlier candidate (3 mp4, 640x400) |

## Pending

- Recording of cand_r16_nopp (c036154) videos with sort_fixed seeds 1040/1041 and sort_language seed 1 was still running on the sim box when this was written. Those videos are not in this branch yet. A follow-up folder will be added once they are finished.
