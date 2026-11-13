# Demo and pitch research

Status: first pass, 2026-11-13. Web search only (grokbot has no xAI credentials, so no live X/web via grok). Nothing here is built or tested.

## What is verified

- Main prize is judged by an expert panel; partner challenges are judged separately by the partners, with their own criteria on the track pages. Source: Junction 2016 material via Wikipedia and Junction notes; 2025 criteria not found.
- Older Junction criteria (2016 to 2019): solves a real problem, creativity, originality, functionality; partners wanted working demos with an intuitive UI. Dated, not confirmed for 2026.
- Voting uses pairwise comparison (Gavel). Each judge sees two projects for a short time, so the demo must land fast. Wikipedia notes technical difficulties at some events.
- The event ends with a demo expo on Sunday.
- WAVELET (2025 winner) is already noted in `docs/research-junction-2026.md`.

## What the search found for bot-guard demos

- Commercial tools that explain verdicts: Prosopo says each verdict comes with a human-readable reason. Irdeto generates plots of bot behaviour with a text explanation. This supports the "shows its work" angle.
- Cloudflare Turnstile: the blocked visitor gets whatever message the site owner wrote ("Failed the bot check"). A vendor-comparison blog says blocked users get no clear way to report a false positive (secondary source).
- No hackathon winner with a live bot-vs-human visitor demo turned up. Related projects are on-chain (ETHIndia Persona, ETHOnline VigilBot) or Twitter accounts (Bot Detective). Prior art for this exact demo looks thin.
- Nothing found on Anubis or CrowdSec block-reason UX. Read their docs directly before claiming anything.

## Second pass (web search, 2026-11-13 evening)

Prior art on block reasons, confidence and FP rate. Secondary sources unless noted; check primary docs before quoting.

- **Anubis**: visitor gets a proof-of-work page (lighter meta-refresh for low-suspicion clients). Maintainers accept false positives as a tradeoff; JS-off users can be locked out (Gentoo forum report). A 418 response means blocked with no challenge page. No confidence score, no FP rate. Sources: xeiaso.net/blog/2025/anubis, anubis.techaro.lol/blog/release/v1.20.0, fzakaria.com/2026/07/09/who-does-anubis-actually-stop.
- **CrowdSec**: `cscli decisions list` gives a REASON (the scenario) and SOURCE (local, manual, console, CAPI). `cscli alerts list -i IP` shows the matching log lines. Decision management in the Console needs an Enterprise plan. No FP rate found. A detection does not always become a ban. Sources: doc.crowdsec.net decisions pages, docs.nethserver.org CrowdSec page.
- **Cloudflare Turnstile**: block message is whatever the site owner writes; no confidence shown to the visitor (docs.cloudflare.com Turnstile).
- Read together: reasons exist (CrowdSec), confidence and live FP rate do not appear in any of the three. Consistent with Tim's market check.

Junction judging, extra facts:
- 2025: 291 projects, 40+ partners; challenge-track winners are picked by the partner companies; the overall winner by peer review (Aalto news, Junction Medium post).
- Gavel compares each project against the previous one the judge saw (Anish Athalye, github.com/techx/gavel). Junction per-project time was not found.
- The Junction Notion pages (Reviewing, Participant Guidebook) are JS-only: WebFetch and the Notion connector both fail. A human has to open them.

## Third pass: hackathon bot and fake-account demos (subagent, web only)

Coverage is thin: Devpost galleries did not surface, most detail comes from ETHGlobal write-ups, not from watching demos. No verified 2022-2026 winner was explicitly sockpuppet or coordinated-behaviour detection. That gap is itself a finding.

| Project | URL | Demo | Prize verified | Lesson |
|---|---|---|---|---|
| DeCartography (ETHNewYork 2022) | ethglobal.com/showcase/decartography-u9prm | Wallet-cluster graphs showing "separate" voters are coordinated; no demo video | Y, Optimism sponsor track | Cluster graph is the right image, but won only a sponsor track |
| Sybil Slayer (ETHGlobal Istanbul) | ethglobal.com/showcase/sybil-slayer-7wtmc | Connect wallet, get a sybil-likelihood | N | One button to a score, no "why", no prize |
| Sybil Daddy | ethglobal.com/showcase/sybil-daddy-orjwp | Three signals combined into a score that scales airdrops | N | Signal breakdown is closer to what we want |
| Sybil Game (ETHGlobal NY 2023) | ethglobal.com/showcase/sybil-game-tam2e | Trivia gated by World ID (prevention, not detection) | Y, sponsor prizes | Stopping bots inside something people play wins sponsor prizes |
| HawkEye (ETHGlobal Singapore 2024) | ethglobal.com/showcase/hawkeye-uitr4 | Camera on keyboard, keystrokes matched to inputs, score under 5 = human | Y, 1st on a sponsor prize | A physical, checkable gesture is a memorable moment |
| HughKnew (Clawckathon 2026) | resemble.ai/resources/could-you-tell-your-grandsons-voice-from-a-clone-of-it-hughknew-was-built-to-find-out-safely | Practice scam call, harder second round shows the stakes | Y, 1st place | Two-beat demo: easy catch, then the stake |

Failure case: NearByNow won at ConUHacks, was accused of hard-coded results, and was cleared by MLH (theconcordian.com/?p=74542). Detectors that cannot show their work get challenged.

Unverified, do not cite: Junction X Budapest 2024 deepfake winner (fetch blocked), Unmask on lablab.ai (dead link), Halo finalist status.

Lessons (from the agent, to test not to trust):
1. One moment carries the demo: a real account flipping to a verdict on screen.
2. A bare score won nothing; show which signals fired next to confidence and FP rate.
3. Nobody in the set showed a false positive being handled. A real human letting through plus a measured FP rate is an open slot.
4. Verified wins were sponsor-track wins, which fits picking the sponsor track and saying so on slide one.

## Fourth pass: detector evasion pitfalls and honest evaluation (subagent, web only)

| Pitfall | Source | Honest practice for 48h |
|---|---|---|
| Botometer-based studies counted humans as bots | arxiv.org/abs/2207.11474 (abstract only read) | Hand-audit ~50 flagged items, report precision with a Wilson interval |
| Labels reflect collection method; shallow trees come near SOTA | arxiv.org/pdf/2301.07015 (via summary) | Trivial 3-feature baseline first |
| Feature or label leakage | arxiv.org/pdf/2303.17251v1 | Feature-provenance table, ablation |
| User-agent-derived labels | arxiv.org/pdf/1711.05098 | Never use UA as label source and feature |
| Unseen bot classes (LOBO) | preprints.org/manuscript/202610.0500/v1; LOBO paper URL UNVERIFIED | Hold out whole campaigns; split by IP/ASN and day, not session |
| Easy synthetic bots flatter numbers | arxiv.org/pdf/2412.02266 | Build an adaptive attacker, report recall per tier, label synthetic traffic |

Calibration has no verified source; practice is a reliability curve on a held-out day, threshold chosen on a validation day. No source found for same-IP leakage in session splits.

Datasets:
- AUTH web robot detection, zenodo.org/records/3477932: CC BY 4.0, 4,091,155 requests, 27,061 IPs, server logs only, IPs partly masked, labels built by the authors. Verified.
- M4D / Bournemouth mouse-and-logs dataset: NO verified download. zenodo.org/record/5549439 is the paper PDF only; m4d.iti.gr has no data link. Probably needs an email to the authors. Do not plan an eval on it.
- TwiBot-22, twibot22.github.io: CC BY-NC-ND 4.0, bot benchmark not coordination.
- No open dataset found combining account-level data with mouse or scroll signals for sockpuppets. The behaviour-fingerprint eval needs self-collected traffic and must be labelled as such.

## Open, needs a human or a better source

1. Junction 2026 judging criteria for the Hostinger challenge and for the main prize. Read the platform track page and the Reviewing section of the participant guidebook.
2. How long a Gavel comparison lasts. Ask an organizer.
3. Anubis and CrowdSec: what a blocked visitor and the owner actually see (primary docs).

## Hypotheses to test (not findings)

- A pairwise judge with a few minutes per project favours one visible moment over a metric table. For this idea that moment would be the sockpuppet reveal from the Hostinger mentor.
- The false-positive rate shown live is the differentiator, since Hostinger weights it at 40 %.
