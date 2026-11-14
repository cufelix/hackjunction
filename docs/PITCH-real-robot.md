# Why it should work on the real robot

DRAFT for the pitch and README. Every number has a source: a Warren message id in room hackjunction-15c6, or a file
on origin/feat/grasp-yaw. Sim numbers are qualifiers only. The organisers wrote: "We test submissions in sim and pick
finalists, who run on the REAL robot on Sunday. Best real-robot score wins" (Discord #aalto, Nov 13 18:42, relayed in
5f4c112f). So every change below was kept only if it has a physical reason, not because it scored in sim (decision
7ffc3570).

## 1. The grasp: fix the geometry, not a constant

**What was broken.** Forward kinematics of the Dex3 hand in the baseline's "closed" posture, with a 45 mm object
centred on the grasp point: the thumb tip stops 11.7 to 14.9 mm short of a cube or cylinder, while the index finger
pushes 9 mm into it (docs/pinch-fk/RESULTS.md, commit a44dd79). So a grasp only holds if the index first shoves the
object toward the thumb. Timi's real-eval data shows exactly that: 57 of 58 held picks were thumb-plus-finger pinches
(e1fe228e), and the rest closed on nothing. The jaw is 46.0 mm wide on a 45 mm object, so centring alone gives no
squeeze. The repo's own comment says the opposite ("a gap a bit smaller than the object").

**What the data said.** In 20 sim episodes, 77 of 135 picks closed on nothing. Every one of the 58 held picks reached a
bin. After an empty close the object was pushed toward the robot in 43 of 53 cases (398f31d9). The organisers'
real-robot teleop data closes the thumb and index to about 29 mm on a 45 mm cube and waits a median 1.28 s before
lifting; the baseline lifted after 0.27 s (223f5f6b). In 704 human cube grasps from Unitree's own dataset, close to lift
is a median 0.90 s (38082574).

**The fix.** `pinch_v1`: a 41 mm jaw from the thumb joints only, with the grasp point at the jaw midpoint (both arms,
mirror verified by FK to 0.00 mm). It runs together with a longer close-hold, and later lift-on-stall (7f2e38e7).

**Result (sim, 20 episodes, 82 objects, same layouts, noise about ±3):**

| arm | objects in the correct bin | wrong bin | source |
|---|---|---|---|
| baseline | 52 / 82 | 0 | 3d0dea46 |
| pinch_v1 | 60 / 82 | 0 | bec074bb |
| pinch_v1 + perception y correction | **67 / 82** | 0 | a880255a |

**Why it should transfer.** The fix is a geometric change: the jaw is now narrower than the object, so the thumb and
finger actually touch. It is not a gain or a friction value. Grip force cannot be tuned on the real robot anyway: the
runner uses fixed kp 1.5 / kd 0.2 for every team, and the organisers expect "a bit less grip on the real robot" than
the sim's friction (organiser answer, 23:14 post f2dabef9). Real geometric closure is the only lever that survives
that.

**What we rejected, and why.** The narrower human-demo grip (29 mm) won on triangles but lost cubes, 7 of 25
(bec074bb). It fails at every cube angle, which is pure over-closure (a880255a). A per-shape hybrid keeps the demo
grip only for a confident triangle. Where the classifier was wrong it hurt cubes (6 of 34 misrouted, a880255a), so the
safe default stays pinch_v1.

## 2. Language: offline, measured blind, abstain instead of guessing wrong

Language episodes carry about 83 % of the final points, and the unseen-sentence episode counts ×1.5.

- **Offline by design.** The organisers said "no guaranteed internet in the finals" (5f4c112f). The resolver is plain
  Python (`re`, `itertools`), no model and no network. The submission image was verified with `--network none`: the
  server runs and the resolver answers (877ce8a4).
- **Measured blind.** The person writing the parser never saw the test sentences. Three sealed sets were written
  independently: v1 (151 sentences), many-to-one (30) and v2 (60, a fresh set never described to the parser's author).
  They are scored by behaviour on all 9 colour × shape object types, using the organisers' own `resolve_bin`. File
  hashes were posted before scoring (cd37a05a, 1735df02, b844e235).

| resolver | v1 | many-to-one | v2 (honest unseen proxy) |
|---|---|---|---|
| first resolver 8fcfb5f3 (00:56) | 0.868 | 0.833 | 0.583 |
| resolver v2 82f30b09 | 0.907 | 0.933 | 0.667 |
| a0d9db63 | 0.921 | 0.933 | 0.733 |
| integrated 05242eae | **0.934** | **0.933** | **0.800** |

v2 overall per hash, all blind on the same sealed 60 sentences. 0.583 is the OVERALL v2 score of the first resolver,
not a category score. v2 elimination separately went 0.417 -> 0.583 -> 0.833 across these hashes.

Sources: blind results posted in the room after each hash, e.g. the 05242eae run.

- **On the GPU** with the full sim: sort_language 54/74 correct with 0 wrong, against 1/74 for the baseline, which
  ignores the instruction entirely (b3de18af).
- **Expected-value rule.** Correct is +2, wrong bin −1, left on the counter 0. Guessing a bin pays only when the
  chance of being right is above 1/3. Every rule added tonight was checked against that rule, and changes that could
  turn a correct +2 into a −1 were rejected (the "bijection repair", the widened catch-all patterns).
- **Leave-alone respected.** "Leave the rest on the counter" no longer puts the rest in a bin (fd410acb). The
  organisers allow such phrasings; they only excluded negation.

## 3. Perception: correct the measured bias, abstain on doubtful shapes

- **Sideways error.** Commanded grasp target vs the true spawn pose, n = 82 real-eval picks, showed a lateral error
  that grows across the table. First-attempt success was 70 % on the left and 40 % on the right (b6c88f08, 7e1e8a68).
  `perception.y_affine` corrects it upstream, and spawns on the right went from 21 to 25 of 30 (a880255a).
- **Shapes only when sure.** Shape is decided from the metric depth top face, never from the colour blob. The blob
  includes the side faces seen from 48° and is unreliable (f857054). Near a bin, or with a cropped top face, the
  classifier abstains rather than guesses (9c5b9e4). A wrong shape only costs the shape-specific grip; the default
  grip still runs.

## 4. Honest limits (say these out loud)

- All grasp numbers are sim. Real-robot friction is unmeasured (organiser answer).
- Triangles stay the weakest shape (20 of 29 with pinch_v1). The organisers' own oracle also missed mainly on
  triangles (f2dabef9).
- The v2 language set is our own guess at unseen phrasings, not the organisers' set.
- Where the final container runs (robot computer vs laptop) is still unanswered by the organisers.

---
TODO before using in the pitch: check every id against the room, and cut
to the 3-4 numbers the jury will remember: 52 → 67 objects, 0 wrong bins, v2 0.58 → 0.80 blind, works with the
network off.
