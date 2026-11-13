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

## Open, needs a human or a better source

1. Junction 2026 judging criteria for the Hostinger challenge and for the main prize. Read the platform track page and the Reviewing section of the participant guidebook.
2. How long a Gavel comparison lasts. Ask an organizer.
3. Anubis and CrowdSec: what a blocked visitor and the owner actually see (primary docs).

## Hypotheses to test (not findings)

- A pairwise judge with a few minutes per project favours one visible moment over a metric table. For this idea that moment would be the sockpuppet reveal from the Hostinger mentor.
- The false-positive rate shown live is the differentiator, since Hostinger weights it at 40 %.
