# Junction 2026 — all 19 main challenges

Read off the platform (`hackjunction.app/participate/45/challenges`) on Fri 13 Nov 2026,
~18:50 CET, with ~16 h on the clock. This is the official brief content, not research:
where a number here disagrees with [`research-junction-2026.md`](./research-junction-2026.md),
**this file wins**.

Percentages are the sponsors' own judging weights. Schedule and logistics are in
[`INFO_HACKATHON.md`](../INFO_HACKATHON.md).

The **Tech Challenges** are at the bottom of this file — 19 more, all stackable on top of
whichever main challenge we pick. **Read that section before deciding anything**: it holds
more stackable money than the main challenges do.

---

## Agentic AI Track (6)

### Break the Agent — AfterQuery · 3 000 € / 1 500 € / 500 € + work trial + merch
Build an **RL environment** that defeats today's best agents, show real failure modes, and
prove the reward reflects real progress. Setting is free: game, scientific sim, software
task, real-world process. Needs a clear goal, an action space, and a checkable reward. Run a
frontier agent in it and compare against a stronger reference run your team demonstrates.
Optional stretch: train a small model on experience from the environment and show it improves.

**Judging:** The gap (30 %) — frontier agent consistently below your reference run, both
reproducible across repeated runs · Meaningful difficulty (30 %) — you can explain the
agent's mistakes and tie them to real decisions; zero points if the gap comes from withheld
info, trick wording or broken tools · Reward integrity (25 %) — survives judges actively
hunting exploits; doing nothing or stopping early must score near zero · Craft (15 %) —
judges run your eval **from the README with one command** and reproduce your numbers.

**How judging runs:** 10-min demo, then judges rerun everything themselves with the standard
harness — your reference run, three agent runs, a do-nothing baseline, a stop-early baseline —
plus **15 minutes actively trying to break your reward**. Tiebreaker: a smaller model that
improves after training in your environment.

**They provide:** Harbor harness running OpenCode, connected to your environment **over MCP**;
a scoring skeleton; frontier model API credits per team; on-site mentors (Daksh Raghuwanshi,
Ved Vedere) for environment design reviews and reward testing.

**Our take:** the closest fit to CoordBench by a distance, and the MCP harness means our
Warren work is on the right side of the interface. The rubric is unusually explicit — note
that 55 % of it (gap + craft + reward integrity partly) is *reproducibility*, not cleverness.
One-command README reproduction is not optional. Judges breaking the reward for 15 minutes is
the thing most teams will fail.

### Trusted Trade Agents — Valio Aimo · 4 000 € / 2 000 € / 1 000 €
Highest cash in the track. Prototype the **interface and operating model** that lets an
external purchasing agent interact with a foodservice wholesaler's systems — not a
marketplace, the interface. Must enable: structured product/price/availability lookup;
inquiry → order on fixed prices; and permission + human-approval management (what the agent
decides alone vs. what goes to a human). Optional: delivery/sustainability comparison,
substitution handling, human-readable audit trail.

**Judging:** Working end-to-end demo (30 %) — full request → lookup → order/approval flow in
the mocked environment, not isolated pieces · Solving the user need (20 %) · Autonomy and
controls (20 %) — guardrail design and auditable logging · Robustness to exceptions (15 %) —
partial availability, substitutions, delivery constraints · Openness and path to production
(15 %) — open documented standards, not a proprietary format.

**They provide:** synthetic catalog, simulated suppliers, mock APIs with realistic prices,
lead times and sustainability labels. Resource portal at `junction26.valioaimo.fi`
(password + API key are on the challenge page on the platform — **not copied into this repo**).
Mentors across commercial, online sales and IT/data governance. x402 / Nevermined named as
relevant context, not required.

**Our take:** best cash-to-effort ratio on the board, and "guardrails + human sign-off +
audit trail" is the same shape as Warren's claim/release model. Needs a new domain learned.

### The Internet Butler / Your Personal Assistant — Exa · $2 500 + $5 000 credits / $1 500 / $500
Personal agent that takes tedious research-heavy life admin off people's hands — is this rent
increase legal, which benefits does my family qualify for, which restaurant for the
anniversary. Demo on a real case.

**Judging (5 × 20 %):** use of Exa to get information that *could not* come from training
data · usefulness to a specific group of real people · clarity in showing sources ·
originality · **demo works on live queries the judges type themselves**.

**They provide:** $50 Exa credits per participant (code `EXA50JUNCTION` at
dashboard.exa.ai/billing), full API (search with fast/balanced/deep modes, contents API,
Websets, Monitors, hosted MCP server, Python/JS SDKs), demo gallery at demos.exa.ai,
engineers on site + Discord.

**Our take:** will be crowded — it's the most "just build an agent" brief here. The live-query
criterion is the filter; most teams will demo a canned path and fall over.

### Earned Trust in AI Code — Tieto · 2 000 € / 1 000 € / 500 €
How does AI code review keep pace with AI-written code while a **named human** stays
accountable? Define the concept and show it works on one sample agent-written change with its
requirement ID: what the AI found, what it **refused to decide**, and why. Show how the
evidence lets a named human accept the change without a full read.

**Deliverables:** concept card (recommendation, principles, trade-offs, limits), live
walkthrough, lessons log, demo video. **Explicitly out of scope: a finished product.** It's a
reference implementation illustrating a concept.

**Judging:** Clarity (30 %) · Practicality (25 %) — works in a real PR flow, named human owns
each decision · Measurement (25 %) — reverts/incidents don't rise as review time falls, time
to merge, share routed to humans · Evidence (20 %).

Judges will ask: *what would make you not trust your own output? who owns the decision? how
would this scale?* Not accepted: "the AI said it was fine", an unowned merge, a risk score
nobody can explain.

**Our take:** this is a *concept* challenge, not a build challenge — unusual, and it rewards
writing over code. Entire's Trails primitive maps onto it almost exactly if we go that way.

### Decode the Noise — Hostinger · 2 000 € / 1 000 € / 500 € + 1 yr hosting + domain
Detect coordinated propaganda, bot amplification or bad-actor behaviour from text, social
content or web-traffic patterns. Four suggested angles: spot the swarm (coordinated
inauthentic behaviour from timing/account similarity/text reuse), name the technique
(classify manipulation and explain it in plain language), clean the traffic (malicious bots
vs. real visitors in server logs), trust the source (provenance scoring).

**Hard requirement:** deployed and **running live on Hostinger infrastructure** with a public
URL judges can open.

**Judging:** detection quality — accuracy, false-positive handling, explainability (40 %) ·
technical execution & live deployment (25 %) · originality & impact (20 %) · demo clarity (15 %).
They explicitly want **precision and false-positive rate shown**, not a happy-path demo.

**They provide:** VPS KVM 2 per team (2 vCPU, 8 GB RAM, 100 GB NVMe, free domain, Docker,
one-click Ollama/n8n templates), AI Router credits for 150+ models (Anthropic/OpenAI/
DeepSeek/Mistral/xAI, OpenAI- and Anthropic-SDK compatible), AI Builder for a front end, and
a **Hostinger MCP server + CLI that plugs into Claude Code** (docs.hostinger.com/hostinger-connector).

### Spread — Bahn Express · 2 000 € / 1 000 € / 500 €
⚠️ **Different format — a live leaderboard competition, with its own clock.** Run a simulated
car-transport company for a week in Finland: dealers ask for quotes, you price against a
competitor, win the car, post the drive as a gig and set the driver fee. Score = gross margin
(dealer revenue − driver pay), with **≥ 90 % of cars arriving by deadline**.

**Timing:** ranking rounds open Fri 19:30 · **final Sun 08:00**, each team plays its own copy
of a secret week, runs ~34 min unattended · **registration closes Sun 07:30**. Top 10 by
margin present 5–10 slides to a jury, which scores elegance, clarity, creativity and
robustness. Reaching top 10 is enough — after that margin stops counting.

**Start:** `spread.bahnexpress.com/join`, event code `spread-kmu9z6`. Handbook, rules, API
reference, TypeScript and Python kits, leaderboard all on the site. There's an
`llms.txt` to hand straight to an agent.

**Our take:** the only challenge with a mechanical score we could grind rather than pitch, and
the 07:30 Sunday cutoff is a trap if nobody is awake. Their own system plays every round as a
benchmark to beat.

---

## Hardware Track by Arduino (5)

### Intel Physical AI Challenge — Ignite Next × Intel · 5 000 € / 3 000 € / 2 000 €
**Biggest cash prize at the event.** Each team gets a station: SO-101 robot arm + leader arm
for teleoperation, top-view and inspection cameras (one a RealSense D415), Duplo bricks you
mark as defective yourself. Teleoperate to record demos → fine-tune a VLA policy in Physical
AI Studio on cloud GPUs → in parallel train an Anomalib model for good/defective → combine
into a fully autonomous inspection cell, optimised with OpenVINO, running on the laptop's
CPU/GPU/NPU.

**Judging:** task success (30 %) — accuracy, cycle time, robustness to shifted/cluttered
bricks · Intel technology integration (30 %) — OpenVINO Physical AI, Robotics AI Suite, NPU
and iGPU offload · dataset and pipeline quality (20 %) · system efficiency (20 %) — latency,
thermal headroom, power.

**They provide:** 25 stations, Core Ultra Series 3 laptops with 64 GB RAM, cloud GPUs, 3–5
Intel robotics engineers on site daily, hands-on session Sat 12:00. There are **Challenge
Terms you must read and accept**.

**Our take:** highest cash, but 30 % of the score is Intel-stack integration — not winnable
without committing the whole weekend to their toolchain.

### Aaltonomous Humanoid — Aalto University · 2 000 € + lab visit; top ~5 run on the real robot
Give a Unitree G1 ("Väiski") a brain: sort 3–5 printed objects (cubes, cylinders, triangular
prisms, ~4.5 cm, red/yellow/blue) into three identical unmarked orange boxes **within 2
minutes**, fully autonomously. Two modes: fixed sorting (red→left, yellow→centre, blue→right)
and **sort by instruction** with phrasings you haven't seen. Stretch: push objects through
matching holes in a lid.

Your policy sees head camera (colour + depth), both wrist cameras, joint states and the
instruction; returns target joint positions **30×/second**. Ship it as a **Docker policy
server**; their runner drives it through a safety filter.

**Judging:** real-world result on the actual robot (40 %) · generality when things change
(25 %) · originality (20 %) · presentation (15 %).

**They provide:** Isaac Sim/Isaac Lab Docker image with the exact scene, GPU compute from
Verda, real-robot datasets published to Hugging Face **during** the event (links in Junction
Discord), starter repo `github.com/zupermiez/aaltonomous-humanoid`, classical baseline,
evaluator, ~30 Aalto Robotics Club volunteers.

⚠️ Simulator needs an **RTX-class GPU (≥24 GB, RTX Pro 6000 / L40S / A6000 / 4090 / 5090)** —
it ray-traces the cameras, so **it will not render on A100 or H100**. On-demand instances,
not spot. Safety gate to reach the real robot: no crashes or safety stops, within joint and
speed limits, ≥ 27 commands/sec.

### The Robot Teleoperation Challenge — Rokoko · Smartgloves II + SDK (~€4 990) / Smartsuit Pro II (€2 295) / Smartgloves I (€1 095)
Control robotic hands, arms or humanoids with Rokoko mocap suits and gloves, physically or in
simulation. Brief is deliberately open — "surprise us". They're interested in precise
manipulation (fragile objects, tools, liquids), collecting human demonstrations to train
autonomous policies, and creative/entertainment uses.

**Judging:** technical execution (35 %) · originality (25 %) · usefulness and impact (25 %) ·
how well it demonstrates human-controlled robotics (15 %).

**They provide:** Smartgloves II, Smartsuit Pro II, Coil Pro, Device SDK, access to the
Unitree G1 **coordinated with the humanoid challenge**, possibly a Booster T1. Access needs a
Rokoko ID + a form + approval by their reps — start early. Prizes are hardware, no cash.

### The Green European Renaissance — Arduino · Ventuno Q / UNO Q 4G + kits / Plug and Make Kit
Edge AI on the Arduino UNO Q for local environmental and social problems — local flora and
fauna, energy saving, elderly independence, industrial impact, pollution, livable cities.
Framed explicitly as European tech sovereignty against megacorp AI.

**Judging:** technical implementation & Edge AI (30 %) · environmental & social impact (30 %) ·
creativity (20 %) · working physical prototype (20 %).

⚠️ **Double submission required:** to be eligible you must publish on **Arduino Project Hub**
(projecthub.arduino.cc, tag `Junction Finland 2026`) *in addition* to the Junction platform.
Hardware kits from the Arduino Hardware Lab, 5th floor. Prizes are hardware, no cash.

### AI Glasses for Everyone, Everyday — Meta · Ray-Ban Meta glasses per winning team member
Show how a first-person field of view enables interactions that beat mirroring a phone
screen. Display and audio-first experiences weighted equally, as are web apps and Device
Access Toolkit mobile integrations. Prototype must handle permissions, errors, performance
trade-offs, connectivity and device state changes. Accessibility for people with all
abilities is called out repeatedly.

**Judging (4 × 25 %):** creativity and use case · technical execution · real-world value and
feasibility · UX including input, accessibility and trust.

**They provide:** 10 × Meta Ray-Ban Display and 10 × RBM Gen 2 **shared** for the event,
6 engineers at the booth. Best teams may be invited to Meta's AI forum on Wed 18 Nov.
Only one prize tier — no 2nd or 3rd.

---

## FinTech Track (2)

### Make Unicorns Real — Cala · ~5 000 € Cala Comet + 1 000 € credits / 500 € + merch
Build on Cala's structured public data (companies, people, funding rounds, investors, and the
connections between them, via API or MCP) to solve a real need for founders or investors.
Explicitly: **focus beats breadth** — one need solved brilliantly beats a platform doing ten
things halfway. They want something usable the day after, with Slush starting right after.

**Judging (4 × 25 %):** real need · execution end to end · originality · **Cala core to the
product, not an add-on**.

**Prize detail:** each winning member becomes a "Cala Comet" — one hackathon of their choice
per month for three months, travel and accommodation covered. ~€5 000 for a team of four.

**They provide:** unlimited credits (code `MAKEUNICORNSREAL` at console.cala.ai → Billing →
Extra credits), docs.cala.ai, co-founder/CTO Issey Masuda on site. **Workshop Sat 11:00**
"Tips & tricks to Make Unicorns Real" — includes what the judges look for.

### Personal Economic Trainer — S-Pankki · 2 000 € / 1 000 € / 500 € Airbnb gift cards
Let people chat with their own banking data, on a large synthetic transaction dataset modelled
on real S-Bank data. From "how much did I spend on taxis last month" to "what kind of spender
am I", "what's safe to spend this week", "how much if I cut dining out 20 %".

They want a **pedagogical angle**: the three baskets — mandatory regular (rent), mandatory
irregular (broken washing machine), and everyday money that slips through your fingers
(coffees). MyData spirit: let users explore freely rather than showing fixed bank-chosen
charts. Optional: benchmark against open data (Statistics Finland); PSD2 multi-bank view.

**Recommended output: a mobile app**, or a web app embeddable in one — their customers are on
mobile. **Judging (5 × 20 %):** financial literacy impact · originality of AI integration ·
clarity of the mobile UX · feasibility · potential to integrate into S-mobiili.

Note: prizes are Airbnb gift cards, not cash.

---

## Healthcare Track (2)

### Equal Care Across Finland — AstraZeneca · 1 000 € / 750 € / 250 €
Build an AI system that finds where Finland's cancer care pathway is unequal, locates where
in the pathway the gap emerges (detection, primary→specialist referral, diagnosis, treatment
start), and **recommends concrete evidence-based actions**. "Diagnosing the problem is not
enough."

**Minimum requirements:** reliable public evidence with sources shown for every claim; cover
at least one indicator tied to Goal 3 of the national cancer strategy; actionable proposals;
**a working demo — slides alone are not sufficient**.

**Judging:** actionability (30 %) · insight (25 %) · data quality & transparency (20 %) ·
execution in a live demo (15 %) · clarity (10 %) — a decision-maker must get the key message
in **under two minutes**.

⚠️ **Key insights session at the AstraZeneca stand Friday 20:00.** Data on Finnish cancer care
is limited and closing that gap is part of the challenge — expect to combine public sources
with mentor interviews. Public aggregate data only, no patient-level data. Three PDFs linked
on the platform (in Finnish).

### Beyond the Clinic — Aalto University · prizes TBA
Help people stay well at home after TMS treatment for severe depression, using anything from
natural stimuli to a phone, music, games, light or AI. Can be **closed-loop**: sense how
someone is doing, respond with something that helps, learn what works for that person.
No special hardware needed — a laptop and a phone are enough.

Sensing ideas: phone (sound, voice, movement, facial expression, sleep, routines), wearables
(HRV, sleep, activity). Product directions: everyday companion, adaptive sensory experience,
game, smart home, social, clinician view.

**Judging (4 × 25 %):** usefulness · ease of use (would someone recovering from depression
actually use it daily) · smart sensing · creativity. Possible TMS and live EEG demos at the booth.

⚠️ Prizes still "to be announced" as of 18:50 Friday.

---

## AI Infrastructure Track by Verda (1)

### Make Blueprints Talk — Stoa · 1 000 € + work trial / 750 € / 250 €
**The only challenge in this track.** Answer natural-language questions about construction
drawings and BIM models, accurately and verifiably. A great answer is correct **and shows
where in the drawing it came from** so a human verifies it in seconds. They explicitly want
solutions next year's frontier models won't obsolete — hybrids of frontier models, computer
vision, 3D, geometry, small fine-tuned models.

**Judging (4 × 25 %):** accuracy · how well it visualises its reasoning for human
verification · creativity · **low inference cost**.

**Dataset:** real technical drawings in many formats and languages —
preview 0.5 GB and full 10 GB, both linked on the platform page (Supabase storage).

Stoa: Helsinki, AI-native OS for construction, backed by Lifeline Ventures (who also have a
grand-jury seat) and founders of Wolt/Supercell/SiloAI; Finland's largest-ever pre-seed.

**Our take:** one challenge in the track = smallest field at the event, and "low inference
cost" as a full 25 % is a criterion most teams will ignore entirely. Against it: a 10 GB
domain dataset to get to grips with, and nothing we've built transfers.

---

## Data Track (3)

### The Web Is Your Real-Time Database — Context.dev · 2 000 € / 500 € / swag
Build a product that reacts to the **live** web. Themes: scrape the economy (live indicators
from product prices or job listings), the web watched (agents triggered by site or news
changes), brand in a box (apps that brand themselves from a sign-up email), truth at internet
speed (fact-checking with sources). Examples given: GPU rental price comparison, founder
activity leaderboard, cve.now, personalised onboarding.

**Judging (5 × 20 %):** real-world impact · creative use of live web data · completeness of
demo · technical execution (latency and accuracy) · **real-timeness — does it react to change
over time or take a one-off snapshot**.

**They provide:** 50 000 non-expiring API credits (sign up at context.dev/signup, redeem
`JUNCTION26`), 50 concurrent requests, search/scrape/crawl/extract, brand and company
entities, news monitors with webhooks, batch API, PDF parsing. Mentors on site Fri to 23:00,
Sat 09:00 → Sun 02:00, Sun 07:00 → end.

⚠️ **Their CTO Beer van der Drift is also on the 7-person grand jury.** He publicly looks for
teams that "move fast, stay pragmatic, and find the one unique insight". Worth knowing
whichever challenge we pick.

### Rethinking Recruitment — City of Espoo · 1 500 € / 1 000 € / 500 €
Help companies and talent connect at Espoo's recruitment and matchmaking events, now that
both sides increasingly interact through AI-generated material — "matching decisions based on
synthetic representations rather than genuine capabilities". Any stage: before, during or
after an event.

**Judging:** impact & relevance (45 %) · innovation & UX (35 %) · responsible & feasible
execution (20 %).

⚠️ **Hard constraints:** synthetic data only; **no automated profiling or automated
decision-making about individuals without human involvement**; minimise personal data;
demonstrate EU AI Act-compliant privacy; avoid bias by nationality, language, education or
career path. Winner presents to Espoo's business development and innovation leaders and the
national city network.

### Wikipedia Through Time — MariaDB · 1 000 € cash + 1 000 € UpCloud credits + certifications / 500 € / 250 €
Build the tool Wikipedia is missing, with MariaDB at the core. Two tracks:
**RAG** (native vector search → grounded, source-cited Q&A over Wikipedia) — recommended for
most teams; **SVT** (system-versioned tables → what an article said on a given date, edit
wars, how a story was rewritten) — for advanced teams; or **both**, e.g. "what did Wikipedia
say about this in 2015 and what changed?" Combining them earns explicit recognition.

**Judging (5 × 20 %):** effective and creative use of MariaDB · implementation and scalability
across the provided datasets · usefulness to readers or editors · design and demo quality ·
originality.

**They provide:** three dataset sizes from **Finnish** Wikipedia (history of Finland) — a
multilingual embedding model is recommended, but no Finnish needed; three starter repos under
`github.com/MariaDB`; mentors including Kaj Arnö and Robert Silén. MariaDB was created in
Finland by Monty Widenius.

---

## What jumps out

**Cash ranking:** Intel 5 000 € → Valio Aimo 4 000 € → AfterQuery 3 000 € → Hostinger /
Tieto / Context.dev / Bahn / Stoa-track 2 000 € → the rest below.

**Thinnest fields (fewest competing briefs):** AI Infrastructure has **one** challenge,
Healthcare and FinTech two each. Agentic AI has six and will be the crowded track.

**Three challenges have requirements outside the normal submission:**
- **Spread** — register by Sun 07:30, final runs Sun 08:00
- **Arduino** — must also publish to Arduino Project Hub with the event tag
- **Hostinger** — must be live on Hostinger infra with a public URL

**Named "we will try to break it" judging:** only AfterQuery. 15 minutes of judges hunting
reward exploits, plus reruns from our README. That's the hardest bar here and the one fewest
teams will clear — which is exactly why it's worth taking.

**Recurring theme across briefs:** judges want *reproducible, source-shown, false-positive-
honest* work, not happy-path demos. AfterQuery reruns your eval, Exa types its own live
queries, Hostinger wants your precision and FP rate, AstraZeneca wants a source per claim,
Stoa wants the drawing location for every answer. Whatever we build, the demo has to survive
someone else's hands.

---

# Tech Challenges (19) — the stackable layer

These run **across every track**. You enter them alongside your main challenge, with the same
project. Several explicitly say so ("open challenge", "applies to any track", "you can
participate alongside any other Junction challenge").

**The MLH email badly undersold this.** That mail listed eight prizes, mostly swag. The
platform has nineteen, and the real money is in the ones the mail never mentioned: OpenAI,
Google DeepMind, Verda, Modal, Aiven, Oxylabs. The six "by Major League Hacking" entries
below are the swag tier; everything else is run by the sponsor directly and pays properly.

## Worth real money

| Challenge | By | Prize |
|---|---|---|
| **Best Use of Codex** | OpenAI | **$10 000 / $5 000 / $2 500 API credits** |
| **Verda Compute Track** | Verda | **$10 000 cloud credits + hoodies** / $2 500 / $2 500 |
| **Best Use of Gemini API** | Google DeepMind | **4 000 € + $1 000 GCloud per team member** / 2 000 € / 1 000 € |
| **Best use of Modal** | Modal | **$2 000 cash + $3 000 credits** |
| **Best agentic usecase of Oxylabs Web API** | Oxylabs | €1 000 Apple voucher + $500 credits per member |
| **Best Use of Aiven** | Aiven | 1 000 € / 750 € / 500 € |
| **Best Use of Gemma** | Google DeepMind | €1 000 per team, top 3, unranked |
| **Best Use of Context.dev** | Context.dev | €750 / €250 / swag |
| **Best Use of Cloudinary** | Cloudinary | $500 Amazon gift cards **per team member** |
| **Best Use of Nemotron** | NVIDIA | 1× RTX 5080, to each of the two best teams |

### What each actually requires

**Best Use of Codex (OpenAI)** — explicitly an open challenge, combinable with anything. Use
Codex *as part of your development process* to build and ship. The bar is demonstrating you
meaningfully used it, not building a product around it. This is the single cheapest $10 000
on the board for a team already coding with agents.

**Verda Compute Track** — "pick any challenge at Junction and use Verda products to build
something you couldn't build on a laptop." Named examples: teaching a humanoid to move,
training locomotion or manipulation policies. European data centres, 100 % renewable. Free
for all Junction teams. Note the Aaltonomous Humanoid challenge already runs its GPU compute
on Verda, so that pairing is pre-blessed.

**Best Use of Gemini API (Google DeepMind)** — open innovation, no predefined problem. Agent,
dev tool, creative app, anything. Judged on a meaningful, innovative, technically impressive
use. ⚠️ This is **not** the MLH "Gemini swag kit" prize from the email — it's DeepMind's own,
and it's worth 4 000 € plus $1 000 per person in cloud credits.

**Best use of Modal** — serve an open model on a GPU, run agent code in Sandboxes, fine-tune,
or fan out a batch job, in plain Python with no Dockerfiles or Kubernetes. They're after "the
most ambitious working AI system built on Modal". Directly aimed at the "my model runs on a
laptop but not behind a URL the judges can open" problem.

**Best agentic usecase of Oxylabs Web API** — use their live search and scraping so the agent
acts on what's true on the web right now and can show its sources. Enter alongside your main
challenge.

**Best Use of Aiven** — at least one Aiven service as a meaningful part of the project. Their
own line: one service used well scores as high as several bolted on. Aiven Runtime can host
the app or agent.

**Best Use of Nemotron (NVIDIA)** — Nemotron 3.5 Super VL, their newest open model: 120 B
parameters but only 12 B active per token, reads text, images, documents and video in one
workflow. Applies to any track. Wants reasoning, tool use, or coordinated workflows.

**Best Use of Context.dev: Fully Structured Web** — ⚠️ hard minimum: at least one Context.dev
endpoint **in the core flow, not a side feature**, and the data must *visibly drive something
in the demo*. Submit a GitHub repo and a demo. (Their CTO is also on the grand jury.)

**Best Use of Cloudinary** — images/video for a more engaging experience, using one of the
four tools at `cld.media/hackathons`. ⚠️ **Requires a separate form**:
`cld.media/hackathon-survey`. Prize is per team member.

**Best use of CRACI** — switch a GitHub Actions job to `runs-on: craci` and every build
records what goes into your software, exposes it via their API, and can auto-fix
vulnerabilities. Explicitly: a project built mainly for another challenge qualifies, as long
as its CI/CD runs on CRACI. Prizes are LEGO / Red Bull / a lunch invite — low value, but
close to free if we have CI at all.

**Atech Real World Challenge** — build a physical device for a specific real person using
Atech modules as the body and AI as the brain. Pro Atech kit for the winners.

**Best use of Arduino UNO Q** — open to all teams, combinable with any challenge. Edge AI on
device. UNO Q + Plug and Make kit per winning member. (Distinct from the main-track Arduino
challenge, which additionally requires publishing to Project Hub.)

## The MLH swag tier

All "by Major League Hacking", all single "Best Overall" prize, no cash:

| Challenge | Prize |
|---|---|
| Best Use of Entire | Mini portable projector |
| Best Use of ElevenLabs | Wireless earbuds |
| Best Use of Tiger Data | Stream Deck Mini |
| Best Use of MongoDB Atlas | M5Stack IoT kit |
| Best Use of Vultr | M5Stack mini programmable monitor |
| Best Use of Solana | Unruggable hardware wallets |

Entire's full ask is in [`research-junction-2026.md`](./research-junction-2026.md) — Sessions,
Graph and/or Brain, as many as possible, judged partly on measurable token efficiency.

## What this changes

The stackable layer looks lucrative, but read it correctly: **these are prizes to win, not
credits handed out for entering.** Codex and Gemini are open to every team at the event,
which makes them the most contested prizes here. Entering several is cheap — one project
can be submitted to multiple challenges — so they are cheap lottery tickets, not expected
income. A realistic bundle for a multi-agent / dev-tooling project:

**Codex ($10 000 credits) + Verda ($10 000 credits) + Gemini (4 000 € + $1 000/person) +
Modal ($2 000 + $3 000) + Entire (projector) + Aiven (1 000 €)**

Qualifying for Codex, Gemini and Verda costs us close to nothing — we are already coding with
agents, we already need GPU and we already need a model. **Winning them is a different
question**: every team can enter, so assume heavy competition and do not plan around the money. Modal and Aiven are an afternoon each if we
need hosting anyway. That stack is independent of which main challenge we enter.

Two that need a non-obvious extra step, easy to miss: **Cloudinary** needs its own survey
form, and **CRACI** needs CI actually running on their runner.
