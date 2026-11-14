# Humanoid: verified facts with file:line

For Felix's build handbook. Every row was read from the starter repo at commit `36daf5b44dad46b8f448a5124ded7faa9947be1e` (`github.com/zupermiez/aaltonomous-humanoid`) on 2026-11-13. Links: `https://github.com/zupermiez/aaltonomous-humanoid/blob/36daf5b/<path>#L<line>`. Paths below are relative to the repo root. Line numbers can drift if the organisers push updates; the SHA is the anchor.
Anything NOT in this table is an inference or a web claim, see `humanoid-research-log.md` and `humanoid-grokbot-report.md`.

## Scoring and finals (`aaltonomous/docs/participants/RULES_AND_SCORING.md`)
| Fact | Line |
|---|---|
| IN a bin = centre inside inner walls, below the rim, at rest (<2 cm/s) at episode end | 25 |
| Safety abort after more than 30 clipped steps, points so far kept | 42 |
| Sim score = mean over seeded episodes on the held-out sim variant | 47 |
| Sim gate: 0 clipped steps (`--gate-max-clips 0`) | 53 |
| Finals: 3 real-robot episodes (fixed, language known template, language unseen) | 63 |
| Unseen-instruction episode weight x1.5 | 66 |
| Final ranking = real-robot score, sim score breaks ties | 67 |
| Finals settings (objects, layout, time limit) not announced in advance | 68 |

## The baseline's gaps (`aaltonomous/baselines/classical/`)
| Fact | Where |
|---|---|
| Rung 2 falls back silently to the fixed mapping: `self.mapping = info.get("mapping") or dict(FIXED_MAPPING)` | `policy.py:73` |
| Perception is "Shape-agnostic" | `perception.py:1` |
| Yaw is computed but "info only" | `perception.py:27` |
| Sim: 31/38 objects (82%) in-process, misses 6 triangles + 1 cube | `README.md:22` |
| Sim: 29/38 (76%) as Docker image, 0 clips, 30.0 Hz, gate PASS | `README.md:23` |
| Real robot: "not run yet (M8)" (the FAQ says it did one pick: conflict, ask a mentor) | `README.md:26` |
| Index/middle tips 57 mm apart, a 45 mm object wedges, so the pinch is thumb vs ONE finger | `config.yaml:42` |

## The grasp ceiling
| Fact | Where |
|---|---|
| Oracle with privileged poses ~84% on rung 1, "needs >90%: triangle grasps (thumb vs one finger spins the prism)" | `aaltonomous/sim/README.md:77` |

## Instructions (`aaltonomous/interface/tasks.py`)
| Fact | Line |
|---|---|
| Fixed mapping red left, yellow center, blue right | 14 |
| 8 training templates | 20 |
| Colours and bins are drawn with `rng.permutation`, so mappings are always full permutations (6 possible) | 45-46 |
| Difficulty knobs: finals may add colour + shape, negation, held-out paraphrases (`STARTER_PATHS.md:106`); "No public G1 colour + shape sorting dataset exists" | `STARTER_PATHS.md:73` |

## Scene (`aaltonomous/scene/scene.yaml`)
| Fact | Line |
|---|---|
| Object size 45 mm for every shape | 29 |
| Bin outer 21 x 21 cm, MEASURED on the real boxes 2026-11-13 | 49 |
| Bin wall height 6 cm is a "(default) measure the real box", NOT measured | 50 |
| Knocked off = centre 3 cm below the counter surface | 76 |

## Interface and safety (`aaltonomous/docs/participants/INTERFACE.md`)
| Fact | Line |
|---|---|
| Error reply from the server: 0.2 s hold, 1.0 s safe stop, episode ends | 66-67 |
| Counter keep-out for wrist and palm frames: x 0.14-1.20, y -0.55..0.55, z < 0.03 m | 143 |
| Slow `act()` is held up to 30 s, held ticks do not count toward the rate | 144 |
| Self-collision and fingertips vs table are NOT checked (only wrist and palm frames) | 147 |
| Wrist cameras: intrinsics placeholder, extrinsics `T_link_optical` null until calibrated | 171 |

## Submission and finals hardware
| Fact | Where |
|---|---|
| Finals PC: one RTX 5090 (32 GB), wired to the robot | `FAQ.md:23` |
| No force or tactile in `obs`, only `hand_q` tells what the fingers did (this is the only grasp-feedback signal) | `FAQ.md:52` |
| External API calls at own risk, no network provided | `FAQ.md:66` |
| Held-out variant: different textures, lighting, object poses | `FAQ.md:93` |
| Listening within 120 s of `docker run` | `SUBMISSION.md:12` |
| Must fit 32 GB VRAM | `SUBMISSION.md:14` |
| Image at most 40 GB uncompressed | `SUBMISSION.md:16` |
| Push to a registry, pin the digest | `SUBMISSION.md:128-133` |

## Sim randomisation (`aaltonomous/sim/dr/train.yaml`)
Line 2 says the ranges are "(default) guesses until the venue capture". Values: depth noise 1 mm + 0.2% of range, 1% holes, head pose jitter +-1 cm / +-1 deg (lines 28-32).

## Not in the repo (from the official Junction page the user pasted, behind login)
Judging weights 40 / 25 / 20 / 15. Top ~5 by sim performance run on the real G1. Mentors Erkka Lehtonen and Viktor Solianoi. The page showed "Status: Closed" (meaning unknown).
