# Pitch video: 2-minute script and shot list

Junction 2026, challenge 320 "Aaltonomous Humanoid". All numbers below come from the measured facts list, resolver `21af849894f526b1` (the final candidate), scored by a teammate on sealed sets.

> ## GATE CLEARED — the voice-over can be recorded (2026-11-14 08:58)
>
> **Nothing blocks shot 14 any more.**
>
> The submission is image **`hackjunction/policy:integ-75e95b2`**, containing resolver
> **`21af849894f526b1`**. Every number in this script was measured on that exact resolver:
>
> * Blind set, 358 unseen sentences: expected value **84.5 % -> 96.8 %**, sentences scoring -1
>   **239 -> 67**, abstentions **132 -> 0**.
> * Second corpus, held-out **test** split, 3276 sentences: exact match **82.2 % -> 87.9 %**.
> * Oliver's sealed blind on `21af8498` **passed**, independently and against a behaviour scorer
>   over all nine (colour, shape) types: v1 0.934 -> 0.947, many-to-one 28/30 -> **30/30**,
>   v2 0.800 -> 0.833, no drop anywhere.
> * All **8 `sort_language` instructions from the organisers' `tasks.py`** give identical output to
>   the previous resolver, so the sim figures in shots 5 and 11 hold.
>
> **Two things to keep straight when recording.** The second corpus reads 92.8 % expected value and
> 91.7 % exact match over both halves; the script deliberately quotes the **test split**, 82.2 % ->
> 87.9 %, because the later gains land mostly in the development half and Q3 is the overfitting
> question. And the sim figures are sim figures — the line "the real robot has not been scored"
> stays in, in shot 13 and in Q1.
>
> If the shipping hash ever changes from `21af8498`, this gate reopens: get the held-out figures for
> whatever ships, update the 1:25-1:41 segment, Q2 and Q3, re-export `docs/pitch-cards/card10.png`,
> and re-check the word count — the segment is sized for 311 words.
>
> Sim numbers in shots 5 and 11 are **not** affected: Timi confirmed they come from box A seeds 0-7
> on resolver `fd410acb` (r14) and `fd410acb`+alias (r15b, 0 changes), and the `sort_language`
> instructions use the organisers' 8 templates from `tasks.py`, which the old resolver already
> handled. With the new resolver those numbers are the same or conservative, never overstated. Sim numbers are sim numbers; the real robot has not been scored.

## 1. Script (120 s)

| Time | Voice-over | On screen |
|---|---|---|
| 0:00-0:09 | Three orange boxes. Three to five small objects. One typed sentence: red on the left, yellow in the middle, blue on the right. | Cold open. Sentence typed into the instruction box; cut to the three boxes and objects on the table. |
| 0:09-0:19 | This is Väiski, a Unitree G1. Its head and wrist cameras find the objects, and the overlay shows what it sees. Two minutes on the clock. | Robot-view video, head and wrist cameras with overlay. A 2:00 countdown in the corner. |
| 0:19-0:35 | Watch the sim run. Cubes, cylinders, prisms: each goes where the sentence says. Across seeds zero to seven, it made zero wrong placements. Fixed sorting scored thirty-one to thirty-two out of thirty-eight. Language sorting, twenty-eight to twenty-nine out of thirty-six. | Sim running, sped up. Lower third: "Sim, box A, seeds 0-7". Scoring output: fixed 31-32 / 38, language 28-29 / 36, wrong placements 0. |
| 0:35-0:51 | Why spend so long on the sentence? Because of the scoring. A correct fixed sort pays one point. A correct language sort pays two. A wrong one costs one either way. Language is about eighty-three percent of the final points. | Scoring table: sort_fixed +1 / -1, sort_language +2 / -1. "~83% of final points" highlighted. |
| 0:51-1:08 | So the robot does the arithmetic. Guessing a bin pays three p minus one, where p is its confidence. Declining pays zero. So it only guesses above one third. Otherwise it abstains instead of guessing wrong. That rule is why the wrong count is zero. | Simple graphic: "guess = 3p - 1, decline = 0", threshold line at 1/3. Then back to the 0 wrong placements line. |
| 1:08-1:25 | We never graded our own homework. Every resolver change was scored on sealed sentence sets the implementer never saw, by a different teammate. We wrote our prediction down first. One change predicted zero sentences would move. Zero moved, across two hundred forty-one. | Terminal: test suite running green. Then a "prediction: 0 change / result: 0 change / 241 sentences" card. |
| 1:25-1:41 | On 358 unseen sentences, expected value rose from 84.5 to 96.8 percent. Sentences scoring minus one fell from 239 to 67. And the 132 it used to give up on, it now reads. Fewer abstentions because it understands more, not because it gambles more. | Before/after card: EV 84.5% -> 96.8%; -1 sentences 239 -> 67; abstentions 132 -> 0. Second line: held-out test split, 3276 sentences, exact match 82.2% -> 87.9%. |
| 1:41-1:53 | The safety gate allows zero clips and needs at least twenty-seven hertz. Our runs pass with zero clips at thirty. The whole resolver is Python standard library. No model, no network, fully deterministic. | Scoring output showing 0 clips, 30 Hz. Then the dependency list: Python standard library only. |
| 1:53-2:00 | Those are sim numbers. The real robot has not been scored. Väiski guesses only when it is sure. | Robot-view clip, last frame holds. End card: "Väiski, Aaltonomous Humanoid". |

Timing check: segments run 9 + 10 + 16 + 16 + 17 + 17 + 16 + 12 + 7 = 120 s.

## 2. Shot list

| # | Shot | Duration | Asset |
|---|---|---|---|
| 1 | Screen-record of the instruction being typed ("red on the left, yellow in the middle, blue on the right") | 4 s | **MISSING** (clean typed-sentence recording) |
| 2 | Table with the three orange boxes and objects, third-person | 5 s | **MISSING** (wide shot of the real set-up; use a sim frame if we cannot film it) |
| 3 | Robot-view: head and wrist cameras with overlay | 10 s | Have |
| 4 | Sim running a full sort, box A, sped up | 12 s | Have |
| 5 | Scoring output: fixed 31-32 / 38, language 28-29 / 36, 0 wrong | 5 s | Have |
| 6 | Scoring table graphic: +1/-1 vs +2/-1, "~83% of final points" | 8 s | **Have** `docs/pitch-cards/card6.png` |
| 7 | EV rule graphic: 3p - 1 vs 0, threshold at 1/3 | 10 s | **Have** `docs/pitch-cards/card7.png` |
| 8 | Terminal: test suite running and passing | 6 s | Have |
| 9 | "Prediction 0 / result 0 / 241 sentences" card | 5 s | **Have** `docs/pitch-cards/card9.png` |
| 10 | Before/after card: EV 84.5 -> 96.8, -1 sentences 239 -> 67, abstentions 132 -> 0, test split 82.2 -> 87.9 | 12 s | **Have** `docs/pitch-cards/card10.png` — **numbers subject to the gate above** |
| 11 | Scoring output: 0 clips, 30 Hz | 5 s | Have |
| 12 | Dependency list: Python standard library only | 5 s | **Have** `docs/pitch-cards/card12.png` |
| 13 | Closing robot-view frame and end card | 7 s | Have (frame); end card **Have** `docs/pitch-cards/card13.png` |
| 14 | Voice-over recording, one take, 311 words (~120 s at a natural 2.6 words/sec) | n/a | **MISSING** |

Cards 6, 7, 9, 10, 12 and the end card 13 are built: `docs/pitch-cards/cards.html` renders all six
at exactly 1920x1080, and the PNGs next to it are straight exports. The HTML is self-contained (no network, no
CDN, no web fonts), so re-exporting after a number changes is just re-screenshotting the section.

Shots 1, 2 and 14 are the ones that most need doing before anything else. Real-robot footage of a sort is **MISSING** unless the robot-view video already shows one; do not imply it is scored.

## 3. Strongest 10-second cut

Shots 1, 3, 4 and 5, in that order, about 2.5 s each: the typed sentence, the robot-view overlay, the sim sorting, and the scoring line "0 wrong placements (sim)".

Voice-over: "One typed sentence. Red left, yellow middle, blue right. Zero wrong placements in sim, because it abstains instead of guessing wrong."

Why: it shows the task, the robot working, and the one result that is both true and unusual, with the reason in the same breath. Judges decide fast, and this answers "does it work" and "why should I trust it" at once.

## 4. Likely judge questions

**Q1. Has this run on the real robot, and what score did it get?**
The numbers we show are simulation numbers: box A, seeds 0-7, fixed 31-32 out of 38, language 28-29 out of 36, zero wrong placements. We have not scored the real robot and we do not claim to have. What carries over is the part we measured separately: the safety gate passes in our runs with zero clips at 30 Hz, and the instruction resolver is deterministic, so the same sentence gives the same answer on the robot as in sim.

**Q2. Why does it abstain? Doesn't skipping an object lose points?**
Skipping scores zero, and a wrong placement scores minus one, so declining is better than a bad guess. A guess at confidence p pays 3p - 1 in expectation, which beats zero only above one third. The robot guesses above that line and abstains below it. That is why wrong placements are zero in sim. Because a correct language sort pays two, it still guesses whenever it is more than a third sure. On the blind 358-sentence set the same rule took abstentions from 132 to zero while expected value rose from 84.5 to 96.8 percent, so the improvement came from resolving more sentences correctly, not from refusing more.

**Q3. How do you know the resolver is not just overfit to your own test sentences?**
Every resolver change was scored on sealed, held-out sentence sets that the implementer never saw, scored by a different team member. Predictions were registered before measuring, so a change that did nothing was caught instead of celebrated: one predicted zero sentences would change, and exactly zero changed across 241 sentences. The 358-sentence set was unseen, and on the held-out test split of a second corpus — 3276 sentences, scored by the same teammate — exact match went from 82.2 to 87.9 percent. We quote the test split rather than the whole corpus on purpose: the later changes gained more on the development half, and reporting that number here would be answering an overfitting question with a number the question is about. The resolver is Python standard library only, with no model and no network, so it is fully deterministic and auditable. The organisers also confirmed the finals instructions are colour-first with no negation.
