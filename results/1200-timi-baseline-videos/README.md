# Baseline sim videos (earlier candidate)

Sim footage of an earlier candidate policy, recorded in Isaac Sim. Used as a visual reference for the pinch/sort behaviour. No metrics here; the numbers are in the sibling experiment folders.

## Files

| file | task | seed | length |
|---|---|---|---|
| `sort_fixed_seed2.mp4` | sort_fixed | 2 | 53 s |
| `sort_language_seed1.mp4` | sort_language | 1 | 56 s |
| `highlight_grasp_release_10s.mp4` | grasp/release highlight | n/a | 10 s |

## Encoding

- H.264, 640 x 400 px (checked with ffprobe), compressed to crf 30 for the repo. Each file is about 1.2 MB or less.

## Overlay

- Overlay info (episode, object, score counter) is not recorded for these files. Treat them as raw footage.

## Setup

- The engine commit and overlay flags of the recorded candidate were not written down at recording time, so they are unknown here.
