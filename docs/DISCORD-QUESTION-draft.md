# Draft question for #aalto (Junction 2026, challenge 320) — NOT POSTED

Owner: **Felix or Tim posts this.** Claude must not post to Discord (public posting on Felix's
account is his call). Requested by `@claude-timi/main`; he confirmed draft + channel read only.
Trimmed to three questions on his instruction, 2026-11-14 05:57.

---

## THE POST (paste this)

> Hi! A few questions about the finals for challenge 320:
> 1. Where does our container actually run — on the robot's onboard computer, or on our own laptop
>    connected over ethernet? And does it have a GPU available?
> 2. Will you set any environment variables or a config profile for the finals run, or does our image
>    run with exactly the defaults we built it with?
> 3. Will the objects be red / yellow / blue as in the sim scene, or the orange / yellow / purple set
>    from `config_lab.yaml`?
> Thanks!

---

## Why each one matters (for us — not part of the post)

**Q1 — where the container runs.** Still genuinely unanswered. erkka's 00:00 reply answered the Dex3
gains but said nothing about *where* the image executes, and it decides whether we can rely on a GPU
at all. Another team (`yo_mama`, #aalto 01:20) asked the same thing and also has no answer, so a +1
from us is cheap. It also bears on whether sim-calibrated constants (`y_affine`, `min_conf`) may live
in the qualification image — see "Parked" below.

**Q2 — env / profile.** If the finals runner injects env vars or selects a config profile, anything
we hardcode as a default can be silently overridden on the day.

**Q3 — object colours. Highest value of the three.** If the real objects are orange and purple rather
than red and blue, every instruction we parse and every colour threshold we tuned is aimed at the
wrong words. We ship a colour-alias pre-pass (`purple/violet/lilac/magenta` → `blue`, `orange` → `red`
except when it describes a box) to cover it blind, but a one-line answer turns that hedge into a known
quantity. If only one question gets answered, make it this one.

> **Caveat on Q3:** `config_lab.yaml` is **not in this repo**. I could not verify the
> orange/yellow/purple list myself, so Q3 is phrased from Timi's description of that file rather than
> from anything I checked. Worth a glance before posting in case the filename or the colours are off.

---

## Already answered — do NOT re-ask

From **erkka from aalto university [LeRo]**, #aalto, 2026-11-14 00:00, replying to Oliver's questions:

> Oracle on the sim clock: not re-measured yet. The ~84 % was measured before the clock fix, and
> triangle grasps were the main miss there too. Your 76–82 % is in the same range.
> Dex3 gains: yes. The real runner uses kp 1.5 / kd 0.2, fixed and the same for every team. You only
> send joint targets.
> Pad friction: we haven't measured it. The pads are rubber. The sim uses static 1.2 / dynamic 1.0, on
> the grippy side, so expect a bit less grip on the real robot.
> Finals instructions: colour comes first. If several teams handle colour, we may also test
> shape-based instructions. No negation.

Consequences already acted on: negation is a dead branch (decision (a), Timi, 04:57); colour is the
right priority; shape stays alive but second; expect **less** grip on the real robot than in sim.

---

## Parked — deliberately NOT in the post

Kept so nobody has to re-derive them; Timi capped the post at three questions.

1. **Same image for sim and robot?** "For the real-robot run, do finalists submit a new container
   image, or does the exact image from the sim qualification get run on the robot? Is there a cut-off
   for submitting it?" — decides whether sim-only tuned constants (`y_affine` +4/30 on right-side
   spawns, `min_conf` off sim confidence logs) may ship in the qualification image. Timi's 5f4c112f §1
   excludes sim-only constant tuning from the submission. Partly folded into Q1.
2. **Calibration time on the day.** "Will teams get a couple of minutes with the robot before scoring
   starts, enough to take a camera snapshot and calibrate colour thresholds?" — if yes, colour
   thresholds become measured rather than guessed, which changes what we build Saturday. Dropped from
   the post on Timi's call, not because it stopped mattering.
