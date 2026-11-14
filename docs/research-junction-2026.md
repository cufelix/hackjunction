# Junction 2026 — strategy research

Deep research run by grokbot `reshercher`, 13 Nov 2026 ~15:00–16:00 CET.
Complements [`INFO_HACKATHON.md`](../INFO_HACKATHON.md) (schedule, venue, judges,
logistics) — this file is the **what to build and which prizes to chase** half.

Anything the bot could not confirm is marked **UNVERIFIED**. Don't treat an
estimate as a fact in the pitch.

---

## TL;DR — the recommendation

**Build CoordBench:** a test environment on top of Warren that scores multi-agent
coding teams on conflicts, wasted tokens, and whether the scoring itself can be gamed.

Stacks: **AfterQuery** (agent evaluation) + **Verda** (€10k credits, open to every team)
+ **MLH Entire** + **MLH Gemini**.

Why: two of the seven grand-jury judges map directly onto it — Daksh Raghuvanshi
(AfterQuery) wants teams who "can explain exactly why an agent fails… whose reward holds
when someone tries to break it", and Arturs Polis (Verda CTO) wants hard problems with
"a clear path to production and scale". It reuses Warren instead of starting cold.

Risk: the AfterQuery brief isn't public yet. **Read it in the platform before 19:30.**
If it doesn't fit, fall back to #2 below.

### The other two directions
2. **Warren Brain** — multi-agent coding hub using Entire Graph + Brain as shared memory,
   with a before/after token benchmark. Stacks AI Infrastructure (Verda) + Entire +
   Tiger Data + DEV post. Very shippable, but reads as a developer tool → weaker for the
   overall prize.
3. **Valio Aimo "Trusted Trade Agents"** — purchasing agents working in a Warren room,
   handing off to a human by voice when a decision needs one. Stacks ElevenLabs + Gemini +
   Tiger Data. Real buyer problem, but it means learning a new domain and ElevenLabs will
   be crowded.

---

## Tracks and partner challenges

Six tracks: Hardware, AI Infrastructure, Healthcare, FinTech, Agentic AI, Data.
Only Hardware (Arduino) and AI Infrastructure (Verda) have a confirmed sponsor;
the Healthcare / FinTech / Agentic AI / Data sponsors are **UNVERIFIED**.

Confirmed briefs:

| Partner | Brief | Prize |
|---|---|---|
| **Verda** | "Best use of Verda" — any Verda service, in any challenge | €10k credits 1st, €2.5k 2nd & 3rd; **every team gets credits** |
| **AfterQuery** | Agent environments and evaluations: why agents fail, scoring that survives exploits | UNVERIFIED |
| **Valio Aimo** | "Trusted Trade Agents" — purchasing agents with clear human-in-the-loop points | UNVERIFIED (was €2k/€1.5k/€500 in 2025) |
| **Cala** | "Make Unicorns Real" — data for founders and VCs | UNVERIFIED |
| **Oxylabs** | — | €1k Apple voucher + $500 credits per winner |
| Others confirmed, prizes UNVERIFIED | MariaDB (Wikipedia edit history + vector search), Meta (AI glasses), Hostinger, Exa (search API), Atech (physical API for agents), NVIDIA (Nemotron), City of Espoo | |

Sources: [Verda blog](https://verda.com/blog/junction-2026),
[AfterQuery announcement](https://www.linkedin.com/posts/hackjunction_junction2026-activity-7523719475117064192-QY_W),
[Valio Aimo announcement](https://www.linkedin.com/posts/hackjunction_junction2026-activity-7520462011562942464-bReX),
[Oxylabs](https://x.com/Oxylabs_io/status/2120168906510000450),
[challenge archive](https://www.hackjunction.com/challenges).

**2025 as a proxy:** ~291 projects across ~24 partner challenges ≈ **10 teams per challenge**.
Overall winner got €10k.

**Crowding (inferred, not measured):** consumer and big-brand challenges draw the most
teams; industrial, government and technically hard ones draw the fewest. Expect Meta,
DeepMind, Agentic AI, Exa and Oxylabs crowded; AfterQuery, MariaDB, Hardware and the niche
MLH prizes quiet. Bot's estimate: ~500–650 teams total (**UNVERIFIED**).

---

## MLH side challenges — real integration cost

All eight confirmed on the [MLH Junction prizes page](https://www.mlh.com/events/junction-40/prizes).
Full prize list and what each asks for: [`sponsors-mlh.md`](./sponsors-mlh.md).

| Prize | Difficulty | Time to first call | Card? | Catch |
|---|---|---|---|---|
| [Gemini](https://aistudio.google.com) | Easy | 10–15 min | No | Free tier is Flash models only; Google may use the data |
| [ElevenLabs](https://www.mlh.com/partners/elevenlabs) | Easy (TTS) / medium (voice agents) | 15–20 min | No | MLH code gives Creator tier; MLH's own pages disagree on 1 vs 3 months |
| [Entire](https://www.mlh.com/partners/entire) | Easy–medium | 15 min | No | $10 MLH credit needs a GitHub account ≥30 days old; only Trail Agents cost money |
| [Tiger Data](https://www.mlh.com/partners/tigerdata) | Easy–medium | 20–30 min | No | $1k credit for 30 days; judges want hypertables, continuous aggregates and compression *shown* |
| [MongoDB Atlas](https://www.mlh.com/partners/mongodb) | Easy | 15–20 min | No | Free tier 512 MB; dull unless you use vector search |
| [Vultr](https://www.mlh.com/partners/vultr) | Medium | 45–90 min | **Yes** | Overlaps Verda, which is free and pays far more → **skip** |
| [Solana](https://www.mlh.com/partners/solana) | Medium–hard | ~1 h | No | Toolchain + faucet limits, no fit with Warren → **skip** |
| [DEV writing](https://www.mlh.com/events/junction-40/challenges/01a120eb-00a4-1251-a834-e3c3be34055d) | Easy | 1–2 h writing | No | Can be published after the event |

20-minute qualifiers: Gemini, ElevenLabs TTS, MongoDB, DEV. Qualifying ≠ winning.
Tiger Data and Entire need a few hours to do convincingly.

---

## Entire — deep dive

**What it is:** "the Git forge for agent-human collaboration", founded by **Thomas Dohmke**
(GitHub's former CEO). $60M seed Feb 2026, forge launched 26 Oct 2026, MLH partnership
announced 12 Nov 2026. Real and usable today.

- CLI is open source: [`entireio/cli`](https://github.com/entireio/cli)
- Docs: [docs.entire.com](https://docs.entire.com) · docs MCP server at `https://docs.entire.io/mcp`
- The `entire mcp` server itself is **read-only with only two tools** — Graph and Brain are
  driven through CLI commands, not MCP.
- ⚠️ The docs and the hack guide disagree on whether `entire brain mcp` exists.
  **Check `--help` on the installed version.**

### Setup

Installer is piped from `https://entire.com/install.sh` (read it before running it), then:

```sh
entire enable -y --agent claude-code
entire login
entire plugin install graph && entire graph setup --repo .
entire plugin install brain && entire brain setup --no-backfill
entire brain connect
```

### How it bolts onto Warren
- Agent finishes work → its commit carries a **checkpoint ID** → posted to the Warren room
  so other agents can pick it up.
- Handoff = `entire brain brief` or `entire session resume`.
- Before granting a **file claim**, Warren runs `entire graph impact` and claims the
  affected code too → fewer conflicts. *(This is the genuinely novel bit.)*
- `entire brain connect` = one shared memory store across machines.
- A task can become a **Trail**: one agent does the work, another reviews it.

### The number for the slide
Take 8–10 real tasks on the Warren repo. Run each **twice** in a fresh Claude Code session —
once without Graph/Brain, once with. Compare with:

```sh
entire checkpoint tokens <B> --compare <A> --json
```

Add handoff cost (tokens agent #2 spends picking up agent #1's half-done task). Slide reads:

> Median −X% tokens per task (n=10, same model), same success rate, −Y% file reads, −Z% handoff cost.

Report cache-read and fresh tokens **separately**. Do **not** present Entire's "up to 71%"
as our own number.

### Submission requirements
List the Entire features used; include demo, repo, and a **public checkpoint link + checkpoint ID**.
[Hack guide](https://hack.entire.io/) · [MLH challenge page](https://www.mlh.com/events/junction-40/challenges/01a182c8-4f6f-d173-81a9-16b415ed5a08)

**Competition:** ~3–10 teams for this prize (bot's estimate, **UNVERIFIED**).

---

## Past winners — the pattern

- **2025 overall:** [WAVELET](https://github.com/ron0studios/WAVELET) — GPU radio-wave
  simulator over a 3D city model. **No AI at all.** Won Sensofusion's track + €10k overall.
- **2025, 2nd:** LoreDash — solo entry turning Wikipedia articles into games, placed in
  Supercell's challenge.
- **2024:** BREAKDANCE — solo inventory app with an AI-generated rap video.
  [Story](https://www.hackjunction.com/blog/frank-sandqvist-story)

**Patterns that matter for us:**
1. Every recent overall winner **first won or placed in a partner challenge**. Pick a
   partner challenge and win it; the overall prize comes through that door.
2. Peer review rewards a demo people **get instantly** — visual or playable.
3. Technical depth **plus** polish. Winners had a live demo *and* a short video.

A dev-tool demo is the hardest kind to make instantly legible to peers. If we go
CoordBench or Warren Brain, the visualisation is not optional — it *is* the demo.

---

## Prize stacking, ranked

1. **AfterQuery / Agentic AI + Verda + Entire + Gemini** — tight fit, two judges aligned,
   reuses Warren. Risk: AfterQuery brief not public yet.
2. **AI Infrastructure (Verda) + Entire + Tiger Data + DEV** — very shippable, weaker for
   the overall prize (reads as a developer tool).
3. **Valio Aimo + ElevenLabs + Gemini + Tiger Data** — real problem, new domain to learn,
   ElevenLabs crowded.
4. **Cala + Oxylabs + Gemini + MongoDB** — Oxylabs pays well, likely crowded.

---

## Open questions to settle tonight

- [ ] **Log in to the platform and read the challenge briefs — AfterQuery's first.**
      Watch the 18:00 opening videos.
- [ ] **Ask the organisers whether building on pre-existing code (Warren) is allowed.**
      **UNVERIFIED** and it decides the whole plan. Say in the submission that Warren
      existed before either way.
- [ ] Ask the MLH coach whether MLH prizes need a **separate** submission on mlh.com.
- [ ] Claim the Entire MLH promo code early (GitHub account must be ≥30 days old).

## Judging mechanics worth remembering
Each partner picks its own challenge winners → in-person peer review picks the top 5 →
those 5 pitch live to the 7-person grand jury. **One submission per team.** Submissions are
open-sourced under MIT.

---

*Bot also produced a longer `report.md` with every source; it lives in the grokbot
`reshercher` channel, not in this repo.*
