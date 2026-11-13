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
| [`docs/afterquery-harbor.md`](docs/afterquery-harbor.md) | How AfterQuery's harness really works, verified against Harbor source. The MCP contract, the verifier, and the one-agent-per-trial blocker |
| [`docs/spread.md`](docs/spread.md) | The Bahn Express leaderboard challenge, assessed. API, kits, what's legal to exploit, honest hour estimate |
| [`docs/entire.md`](docs/entire.md) | Entire CLI tested on this machine: what works, what the docs get wrong, and a public-repo warning before anyone runs `entire enable` |
| [`docs/research-junction-2026.md`](docs/research-junction-2026.md) | Strategy research: past-winner patterns, Entire deep-dive, prize stacking. Older than `challenges.md` — where they disagree, `challenges.md` wins |
| [`docs/sponsors-mlh.md`](docs/sponsors-mlh.md) | The MLH email that started this. Superseded by the tech-challenge section of `challenges.md` |
| [`CLAUDE.md`](CLAUDE.md) | How the agents work: Warren, grokbot, worker sessions, Haiku subagent rules |
| [`hub/`](hub/) | The Warren hub (docker + tailscale funnel). Our tooling, **not** our submission |

---

## The challenges, in one screen

**Six tracks, 19 main challenges.** Enter one. Then stack tech challenges on top.

### Main challenges by track

**Agentic AI (6)** — the crowded track
| Challenge | By | Prize |
|---|---|---|
| Trusted Trade Agents | Valio Aimo | **4 000 / 2 000 / 1 000 €** |
| Break the Agent | AfterQuery | **3 000 €** + work trial |
| The Internet Butler | Exa | $2 500 + $5 000 credits |
| Decode the Noise | Hostinger | 2 000 / 1 000 / 500 € + 1 yr hosting |
| Earned Trust in AI Code | Tieto | 2 000 / 1 000 / 500 € |
| Spread | Bahn Express | 2 000 / 1 000 / 500 € |

**Hardware by Arduino (5)**
| Challenge | By | Prize |
|---|---|---|
| Intel Physical AI | Ignite Next × Intel | **5 000 / 3 000 / 2 000 €** — biggest cash at the event |
| Aaltonomous Humanoid | Aalto | 2 000 € + top 5 run on the real robot |
| Robot Teleoperation | Rokoko | Hardware ~€4 990 / €2 295 / €1 095 |
| AI Glasses for Everyone | Meta | Ray-Ban Meta per member |
| The Green European Renaissance | Arduino | Hardware kits |

**FinTech (2)** · Make Unicorns Real (Cala, ~5 000 € Comet + credits) · Personal Economic Trainer (S-Pankki, Airbnb vouchers 2 000/1 000/500 €)

**Healthcare (2)** · Equal Care Across Finland (AstraZeneca, 1 000/750/250 €) · Beyond the Clinic (Aalto, prizes TBA)

**AI Infrastructure by Verda (1)** · Make Blueprints Talk (Stoa, 1 000 € + work trial) — **only one challenge in the whole track, smallest field at the event**

**Data (3)** · The Web Is Your Real-Time Database (Context.dev, 2 000 €) · Rethinking Recruitment (City of Espoo, 1 500/1 000/500 €) · Wikipedia Through Time (MariaDB, 1 000 € + credits + certs)

### The stackable layer — where the money actually is

These run across every track and combine with whatever we enter.

| | By | Prize |
|---|---|---|
| **Best Use of Codex** | OpenAI | **$10 000 / $5 000 / $2 500 API credits** — open challenge |
| **Verda Compute Track** | Verda | **$10 000 cloud credits** / $2 500 / $2 500 |
| **Best Use of Gemini API** | Google DeepMind | **4 000 € + $1 000 GCloud per member** / 2 000 / 1 000 € |
| Best use of Modal | Modal | $2 000 cash + $3 000 credits |
| Best agentic usecase of Oxylabs Web API | Oxylabs | €1 000 voucher + $500/member |
| Best Use of Aiven | Aiven | 1 000 / 750 / 500 € |
| Best Use of Gemma | Google DeepMind | €1 000 × top 3, unranked |
| Best Use of Context.dev | Context.dev | €750 / €250 |
| Best Use of Cloudinary | Cloudinary | $500 Amazon **per member** — needs a separate form |
| Best Use of Nemotron | NVIDIA | RTX 5080 × 2 teams |
| Best use of CRACI | Craci | LEGO / Red Bull / Slush lunch — just needs CI on their runner |
| Best use of Arduino UNO Q, Atech | Arduino, Atech | Hardware kits |
| MLH swag tier | MLH | Entire, ElevenLabs, Tiger Data, MongoDB, Vultr, Solana — one prize each, no cash |

**Codex + Verda + Gemini alone are worth more than any main challenge, and all three are near-free for a team already coding with agents on GPUs.** That stack is independent of what we build.

---

## What we've settled

- The MLH email **badly undersold** the tech challenges: it showed 8 mostly-swag prizes; the platform has 19 and the money is in the ones it never mentioned.
- Every recent Junction overall winner **first won or placed in a partner challenge**. Junction 2025 was won by a GPU radio-wave simulator with **no AI at all**.
- Peer review rewards a demo people grasp instantly. **A dev tool is the hardest format for that** — if we build one, the visualisation *is* the demo, not a bonus.
- Judges across briefs want reproducible, source-shown, false-positive-honest work. AfterQuery reruns your eval and spends 15 minutes trying to break your reward; Exa types its own live queries; Hostinger wants your precision and FP rate; AstraZeneca wants a source per claim; Stoa wants the drawing location per answer.
- AfterQuery's harness **runs one agent per trial** — no multi-agent mode. A team must be simulated inside the environment. See [`docs/afterquery-harbor.md`](docs/afterquery-harbor.md).
- Spread is **not** a cheap parallel win: 15–25 h, one person, awake at 07:30 Sunday.

## Still open

- [ ] **Do the 2026 rules match the 2024 rules page?** Ask organizers in writing — `participants@hackjunction.com` or Discord. This decides how much we can lean on anything pre-existing.
- [ ] Can a multi-agent setup inside the MCP sidecar pass AfterQuery's standard-harness rerun? Ask Daksh Raghuwanshi or Ved Vedere on site.
- [ ] **What are we building?** Not decided.
- [ ] Do MLH prizes need a separate submission on mlh.com? Ask the MLH coach.

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
