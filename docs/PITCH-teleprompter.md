# Teleprompter — shot 14 voice-over, one take

Read only the **bold** text. The grey line under each block is what is on screen, so you can
pace to it; do not read it aloud. Timings are cumulative from zero.

> GATE CLEARED, you can record — checked 2026-11-14 09:11 CET. The submission is image
> `hackjunction/policy:integ-75e95b2` with resolver `21af849894f526b1`; Oliver's sealed blind on it
> passed, and every number below was measured on that exact resolver. If the shipping resolver ever
> stops being `21af8498`, stop and re-check the gate at the top of `PITCH-video-script.md` —
> segment 6 would then be out of date.
>
> Nothing in this box is read aloud. Only the bold blocks below are.

Target pace: **2.6 words per second**. If a block feels rushed, it is the writing that is long,
not your delivery — tell Claude and the block gets cut, rather than you speeding up.

---

### 1.  0:00-0:09   (9 s, 23 words, 2.6 w/s)

**Three orange boxes. Three to five small objects. One typed sentence: red on the left, yellow in the middle, blue on the right.**

> on screen: Cold open. Sentence typed into the instruction box; cut to the three boxes and objects on the table.

---

### 2.  0:09-0:19   (10 s, 26 words, 2.6 w/s)

**This is Väiski, a Unitree G1. Its head and wrist cameras find the objects, and the overlay shows what it sees. Two minutes on the clock.**

> on screen: Robot-view video, head and wrist cameras with overlay. A 2:00 countdown in the corner.

---

### 3.  0:19-0:35   (16 s, 40 words, 2.5 w/s)

**Watch the sim run. Cubes, cylinders, prisms: each goes where the sentence says. Across seeds zero to seven, it made zero wrong placements. Fixed sorting scored thirty-one to thirty-two out of thirty-eight. Language sorting, twenty-eight to twenty-nine out of thirty-six.**

> on screen: Sim running, sped up. Lower third: "Sim, box A, seeds 0-7". Scoring output: fixed 31-32 / 38, language 28-29 / 36, wrong placements 0.

---

### 4.  0:35-0:51   (16 s, 40 words, 2.5 w/s)

**Why spend so long on the sentence? Because of the scoring. A correct fixed sort pays one point. A correct language sort pays two. A wrong one costs one either way. Language is about eighty-three percent of the final points.**

> on screen: Scoring table: sort_fixed +1 / -1, sort_language +2 / -1. "~83% of final points" highlighted.

---

### 5.  0:51-1:08   (17 s, 45 words, 2.6 w/s)

**So the robot does the arithmetic. Guessing a bin pays three p minus one, where p is its confidence. Declining pays zero. So it only guesses above one third. Otherwise it abstains instead of guessing wrong. That rule is why the wrong count is zero.**

> on screen: Simple graphic: "guess = 3p - 1, decline = 0", threshold line at 1/3. Then back to the 0 wrong placements line.

---

### 6.  1:08-1:25   (17 s, 42 words, 2.5 w/s)

**We never graded our own homework. Every resolver change was scored on sealed sentence sets the implementer never saw, by a different teammate. We wrote our prediction down first. One change predicted zero sentences would move. Zero moved, across two hundred forty-one.**

> on screen: Terminal: test suite running green. Then a "prediction: 0 change / result: 0 change / 241 sentences" card.

---

### 7.  1:25-1:41   (16 s, 44 words, 2.8 w/s)

**On 358 unseen sentences, expected value rose from 84.5 to 96.8 percent. Sentences scoring minus one fell from 239 to 67. And the 132 it used to give up on, it now reads. Fewer abstentions because it understands more, not because it gambles more.**

> on screen: Before/after card (`docs/pitch-cards/card10.png`): EV 84.5% -> 96.8%; -1 sentences 239 -> 67; abstentions 132 -> 0. Second line: held-out test split, 3276 sentences, exact match 82.2% -> 87.9%.

---

### 8.  1:41-1:53   (12 s, 33 words, 2.8 w/s)

**The safety gate allows zero clips and needs at least twenty-seven hertz. Our runs pass with zero clips at thirty. The whole resolver is Python standard library. No model, no network, fully deterministic.**

> on screen: Scoring output showing 0 clips, 30 Hz. Then the dependency list: Python standard library only.

---

### 9.  1:53-2:00   (7 s, 18 words, 2.6 w/s)

**Those are sim numbers. The real robot has not been scored. Väiski guesses only when it is sure.**

> on screen: Robot-view clip, last frame holds. End card: "Väiski, Aaltonomous Humanoid".

---

Total: **311 words**, 119.6 s at 2.6 w/s. Budget is 120 s.

If you fluff a line, keep going and do a second take of the whole thing; it is two minutes,
and one continuous take sounds better than a splice.