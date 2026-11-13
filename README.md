# hackjunction — team knowledge base

Everything we know about Junction 2026, in one place. Three Claude sessions and three people
work from this repo, so **read before asking, and write down what you learn.**

Last consolidated: Fri 13 Nov 2026, 19:05 CET (20:05 Finnish).

---

## ⚠️ Read these two first

**1. [`docs/rules.md`](docs/rules.md) — we cannot submit pre-existing code.**
Junction: *"No code written before the event."* MLH: only new work is judged.
**Warren cannot be our project.** The *idea* can be reworked; the code cannot be reused.
Anything we submit must be written this weekend. Confirm with organizers in writing.

**1b. Submitting a project for several challenges at once is explicitly allowed.** Verified
on the platform guide. The tech-challenge stack below is sanctioned, not a loophole.

**2. [`INFO_HACKATHON.md`](INFO_HACKATHON.md) — the clock.**
Hacking started Fri 19:30. **Preliminary submission Fri 23:59** (platform says 23:55 — aim
for 23:50). Submissions open Sat 12:00. **Final Sun 10:00.** Peer review, then top 5 pitch to
a 7-person grand jury. One submission per team. Everything open-sourced under MIT.

---

## The map

| File | What's in it |
|---|---|
| [`INFO_HACKATHON.md`](INFO_HACKATHON.md) | Schedule, venue (Hype Areena, Espoo), the 7 grand-jury judges and what each says they look for, logistics, submission mechanics |
| [`docs/rules.md`](docs/rules.md) | Eligibility. Pre-existing code. What's safe and what disqualifies |
| [`docs/challenges.md`](docs/challenges.md) | **All 19 main challenges + all 19 tech challenges.** Per challenge: sponsor, what they actually ask, judging weights, prizes, resources |
| [`docs/hackerpack.md`](docs/hackerpack.md) | **What we actually get.** All 20 hackerpacks: credits per person and per team, how to claim each one, the hardware on the 5th floor, and the four with extra qualification requirements |
| [`docs/afterquery-harbor.md`](docs/afterquery-harbor.md) | How AfterQuery's harness really works, verified against Harbor source. The MCP contract, the verifier, and the one-agent-per-trial blocker |
| [`docs/coordbench-sketch.md`](docs/coordbench-sketch.md) | A design sketch for the AfterQuery challenge: simulated teammates in the MCP sidecar, the reward formula and how it survives exploit-hunting. **Not a decision** |
| [`docs/spread.md`](docs/spread.md) | The Bahn Express leaderboard challenge, assessed. API, kits, what's legal to exploit, honest hour estimate |
| [`docs/entire.md`](docs/entire.md) | Entire CLI tested on this machine: what works, what the docs get wrong, and a public-repo warning before anyone runs `entire enable` |
| [`docs/research-junction-2026.md`](docs/research-junction-2026.md) | Strategy research: past-winner patterns, Entire deep-dive, prize stacking. Older than `challenges.md` — where they disagree, `challenges.md` wins |
| [`docs/sponsors-mlh.md`](docs/sponsors-mlh.md) | The MLH email that started this. Superseded by the tech-challenge section of `challenges.md` |
| [`CLAUDE.md`](CLAUDE.md) | How the agents work: Warren, grokbot, worker sessions, Haiku subagent rules |
| [`hub/`](hub/) | The Warren hub (docker + tailscale funnel). Our tooling, **not** our submission |

---

## The challenges, in one screen

**Six tracks, 19 main challenges.** Enter one. Then stack tech challenges on top.

### Main challenges — what each one asks you to build

Prizes are deliberately not here. They don't vary enough to decide anything, and every
challenge is won the same way: build the thing the brief asks for, well enough to demo.
Prizes are recorded in [`docs/challenges.md`](docs/challenges.md) if anyone needs them.

**Agentic AI (6)** — six briefs, so the most competition
| Challenge | Build this |
|---|---|
| Break the Agent · AfterQuery | An RL environment that beats frontier agents. Judges rerun your eval from the README and spend 15 min trying to cheat your reward |
| Trusted Trade Agents · Valio Aimo | The interface a purchasing agent uses to buy from a wholesaler: lookup, order, and which decisions need a human |
| Earned Trust in AI Code · Tieto | A concept (not a product) for AI code review where a named human stays accountable. Writing-heavy |
| Decode the Noise · Hostinger | Propaganda / bot-network detection. Must run live on their infra, and they want your false-positive rate |
| The Internet Butler · Exa | A personal agent for life admin. Judges type their own live queries at it |
| Spread · Bahn Express | A bot that runs a car-transport company on a leaderboard. Own clock: final Sun 08:00 |

**Hardware by Arduino (5)** — all need physical hardware from the labs
| Challenge | Build this |
|---|---|
| Intel Physical AI | Teach a robot arm to sort bricks autonomously: teleop demos → VLA policy → anomaly detection, on their stack |
| Aaltonomous Humanoid | Give a Unitree G1 a brain: sort objects into boxes in 2 min, shipped as a Docker policy server |
| Robot Teleoperation · Rokoko | Drive robot hands/arms/humanoids with mocap gloves and suits. Brief is wide open |
| AI Glasses · Meta | A first-person use case that beats mirroring a phone screen |
| The Green European Renaissance · Arduino | Edge AI on an UNO Q for a local environmental problem. Also publish to Arduino Project Hub |

**FinTech (2)** · *Make Unicorns Real* (Cala) — one sharp tool for founders or investors on their company/funding graph; focus beats breadth · *Personal Economic Trainer* (S-Pankki) — chat with your own bank transactions, mobile-first, teaching where money actually goes

**Healthcare (2)** · *Equal Care Across Finland* (AstraZeneca) — find where Finnish cancer care is unequal **and recommend concrete actions**; working demo required, source per claim · *Beyond the Clinic* (Aalto) — help people stay well at home after depression treatment; a laptop and phone are enough

**AI Infrastructure by Verda (1)** · *Make Blueprints Talk* (Stoa) — answer questions about construction drawings and show where in the drawing the answer came from. **Only one challenge in the whole track, so the smallest field at the event.** 25 % of the score is low inference cost

**Data (3)** · *The Web Is Your Real-Time Database* (Context.dev) — a product that reacts to the live web, scored on whether it's real-time or a snapshot · *Rethinking Recruitment* (City of Espoo) — better matching at hiring events; synthetic data only, no automated decisions about people · *Wikipedia Through Time* (MariaDB) — RAG over Finnish Wikipedia, or time-travel through its edit history

### The stackable layer (19 tech challenges)

Cross-track. One project can be submitted to several — the platform says so explicitly. They
change nothing about *what* we build; they're things we bolt on or are already doing.

**Free to qualify if we're doing it anyway:** Codex (use it in our dev process) · Gemini or
Gemma (we need a model) · Verda or Modal (we need GPU/hosting) · CRACI (just point CI at
their runner) · Aiven (if we need a database) · Entire (if we use their session tooling).

**Only if the project genuinely needs it:** Context.dev or Oxylabs (live web data — both
require it to drive the demo visibly, not sit on the side) · Cloudinary (images/video, and a
separate form) · Nemotron · Arduino UNO Q · Atech · ElevenLabs · Tiger Data · MongoDB ·
Vultr · Solana.

Full list with requirements: [`docs/challenges.md`](docs/challenges.md).
What each sponsor hands us to build with: [`docs/hackerpack.md`](docs/hackerpack.md).

---

## What we've settled

- The MLH email listed 8 tech challenges; the platform has **19**. Several are free to qualify for if we're already using the tool.
- Every recent Junction overall winner **first won or placed in a partner challenge**. Junction 2025 was won by a GPU radio-wave simulator with **no AI at all**.
- Peer review rewards a demo people grasp instantly. **A dev tool is the hardest format for that** — if we build one, the visualisation *is* the demo, not a bonus.
- Judges across briefs want reproducible, source-shown, false-positive-honest work. AfterQuery reruns your eval and spends 15 minutes trying to break your reward; Exa types its own live queries; Hostinger wants your precision and FP rate; AstraZeneca wants a source per claim; Stoa wants the drawing location per answer.
- AfterQuery's harness **runs one agent per trial** — no multi-agent mode. A team must be simulated inside the environment. See [`docs/afterquery-harbor.md`](docs/afterquery-harbor.md).
- **Compute is not our constraint.** CRACI gives every team €1000, Verda €200 and Modal $100
  per person, and two sponsors ship MCP servers we can use today. The OpenAI $100 is Codex/ChatGPT
  usage only, **not** API credits — and it's the one code the platform issues directly, still
  unclaimed. See [`docs/hackerpack.md`](docs/hackerpack.md).
- Spread is **not** a cheap parallel win: 15–25 h, one person, awake at 07:30 Sunday.

## Still open

- [ ] **Do the 2026 rules match the 2024 rules page?** Ask organizers in writing — `participants@hackjunction.com` or Discord. This decides how much we can lean on anything pre-existing.
- [ ] Can a multi-agent setup inside the MCP sidecar pass AfterQuery's standard-harness rerun? Ask Daksh Raghuwanshi or Ved Vedere on site.
- [ ] **What are we building?** Not decided.
- [ ] Do MLH prizes need a separate submission on mlh.com? Ask the MLH coach.
- [ ] Are all submissions really MIT-licensed this year? Secondhand claim, unverified, and it
  matters for anything meant to outlive the weekend.

## Don't lose the prize on paperwork

Verified on the platform: the final submission needs a **2-minute video pitch** (publicly
viewable, *"not a second longer"*), a **pitch deck PDF**, demo and source links, and the
project flipped from **DRAFT to FINAL** — drafts are not reviewed. **Everyone must attend
peer review Sun 10:30–14:00 or we are not eligible for the main prize.** Budget Sunday
morning for the video and deck.

---

## Three challenges with requirements outside the normal submission

Easy to lose a prize to these:
- **Spread** — register by Sun 07:30, final runs 08:00
- **Arduino (main track)** — must also publish to Arduino Project Hub, tag `Junction Finland 2026`
- **Hostinger** — must run live on Hostinger infra with a public URL
- *(and **Cloudinary** needs its own survey form, **CRACI** needs CI on their runner)*

## A note on credentials

Some briefs print API keys and portal passwords on the platform page. **This repo is public —
they do not go in here.** Read them on the platform when you need them.
