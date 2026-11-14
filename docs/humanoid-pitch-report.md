# Grokbot "Hackathon Helper" report: what wins robot demos

Condensed from the bot's reply, 2026-11-14 ~01:30. All claims UNVERIFIED unless I say otherwise (I opened none of the links). Nothing built.

## Is the challenge public?
Bot found no public mention of "humanoid" or "Unitree" on 2026.hackjunction.com or the Junction Platform pages it could open. (The brief IS on the Platform behind login; Oliver pasted it. Felix stored it verbatim as `docs/challenge-320-aaltonomous-humanoid.md` on his machine.)

## What winners did
1. A humanoid that sorts things can win if the demo is polished: Humanoid Hack Tokyo, team AG1 tied 3rd with G1 object sorting plus a control GUI, praised for completeness and real industrial use (aiworkstyle.net/ainews/17199).
2. Tie it to a real problem. LeRobot worldwide runner-up sorted medicine for seniors; a judge said "it was really the execution" (theaiinnovator.com team-behind-ai-robot-that-folds-laundry-wins-hackathon). Another winner noted top spots go to "real, relatable tasks".
3. Judges ask "why a humanoid?" (Japan G1 hackathon). Humanoid Hack Tokyo 2's top prize went to Rescue G1 for proving it worked on the real robot.
4. Show the full pipeline, teleop data to ML to the real robot (Tokyo).
5. Junction 2023-2025 robotics winners: not found. 2025 overall winner WAVELET was a simulation with no robot.

## How judges score originality and generality
- RoboCup@Home finals: robotics jury scores efficacy and elegance, innovation, difficulty; outside jury scores originality and presentation ("story-telling is to be rewarded"), everyday usefulness, success, 0-10 each.
- Generality = unseen test cases (ManiSkill challenges test on unseen objects; models with almost equal training scores ended up more than 10 points apart on test).
- LeRobot hackathon criteria: originality, engineering challenge, usefulness and real-world value. One top-30 team skipped generic pick-and-place on purpose to stand out.
- Volting Cup (G1 sorting, IROS 2026): points per object moved, points off per drop, so reliability counts directly.
- How Aalto scores its own 25% and 20%: not found.

## Failure honesty
Recover live (a LeRobot runner-up reset and reran after a cable crash, judges "looked impressed"). State the real success rate (winners said about 70%). Show an ablation (Real Robot Challenge winners: 40 real trials per variant). Turning a miss into a "feature" is risky and unconfirmed.

## Strong in sim, weaker on the real robot
- Habitat CVPR 2019: sim success barely predicted real success (rank correlation 0.18, 9 reversals) because agents exploited sim quirks (arXiv 1912.06321).
- Real Robot Challenge (UCD): excellent sim results did not carry over, real cube slid more and fingers got stuck; fix was stuck detection plus random recovery moves (arXiv 2109.15233).
- ManiSkill2: 91% sim became 60% real, blamed on depth-sensor differences (arXiv 2302.04659). Lesson: more sensor-noise randomisation.
- A small camera shift broke a LeRobot policy.
- No named hackathon team found that topped the sim board then lost on the real robot.

## Bot's 3-minute pitch list
1. Real robot (40%): run live, backup video ready, state success rate up front, retry and keep talking on a miss.
2. Generality (25%): sim clip with new colours, positions and lighting, plus one number: success on trained setups vs new ones.
3. Originality (20%): name the one thing done differently from the baseline, show a before/after bar chart proving it helped.
4. Presentation (15%): open with why sorting matters (warehouses, recycling); let the robot do the talking; keep own talking under a minute.
5. Sim vs real honesty: one slide on what broke on the real robot and what you would fix next.

## My note
Items 1-4 of the pitch list match our own plan: an ablation of baseline vs rotation-aware grasp vs parser (rung 1 and rung 2 numbers) is the cleanest "originality" evidence we can make without building something exotic.
