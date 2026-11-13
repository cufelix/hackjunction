# Idea brainstorm: making the bot guard score on Impact

Status 2026-11-13 evening, research only. Written by Oliver's session after reading the Junction guidebook (`junction-guidebook-notes.md`). Nothing here is built.

## Why this exists

Junction scores Innovation, Impact (society) and Implementation. Peers at a table and 7 stage judges read those three, not precision and FPR. A bot guard for site owners is a dev tool, and dev tools score weakest on Impact. This doc collects the framings and the objections.

## Verified facts usable in the pitch

- Cloudflare's own post-mortem (blog.cloudflare.com/18-november-2025-outage/): a Bot Management "feature file" got duplicate rows, grew past a fixed size and "caused the bots module to trigger an error". Impact from 11:20 or 11:28 UTC (the post gives both), main impact resolved 14:30, all resolved 17:06. Core CDN and security services returned HTTP 5xx, Turnstile failed to load. "Not caused, directly or indirectly, by a cyber attack." The post gives no traffic percentage and names no customers.
- LocalImpact survey (localimpact.com/research/state-of-fake-reviews, owner survey, US): 72% of local business owners received a fake review in 12 months, 42% say they were definitely targeted by a coordinated fake-review attack, 37% suspect it. It is a survey of owners' belief, not an incidence rate.
- Prior-art matrix (`demo-research.md`, fifth pass): no product shows a calibrated probability tied to a measured FP rate, an owner feedback loop, or an "undecided" state.
- Anubis docs accept false positives and note JS-off users can be locked out.
- Not found: any independent measurement of how often humans fail bot checks. Vendors claim as low as 0.01% as marketing.

## Five framings

1. **The bouncer who shows his receipt.** Harm: real people wrongly locked out (VPN, privacy browsers, assistive tech, shared Wi-Fi). Every block shows a reason, a blocked human gets a one-click appeal that feeds a measured FP rate.
2. **"I don't know" as a feature.** Theme is idealism and caution. The "watching" state refuses to accuse without evidence and says what it still needs.
3. **European sovereignty.** Self-hosted on an EU VPS, no single US proxy in front of every visitor. Claim is only as strong as the verified outage facts above. GDPR angle unverified.
4. **Protect the small publisher.** Coordinated fake reviews and comment brigades hit small sites; the owner gets a daily summary of what looks coordinated and what is uncertain.
5. **The accountable algorithm.** A public ledger of how often the guard was wrong, with a confidence interval, next to every verdict.

Current lean: 1 + 2 as the story, 4 as the demo setting, 5 as the artifact.

## Red-team (model role-play, opinion not evidence)

Three reviewers: non-technical hacker, senior security engineer, stage judge.

Weakest points:
1. **Label bias.** The "trusted human stream" only represents who it samples. Attackers who mimic it defeat it. A tight CI on a biased label is worse than no number. No full answer; say it on the slide. Needs a one-sentence answer to "calibrated against whom?".
2. **Appeals are gameable.** Bots can click "I'm human". Mitigation: proof of work, rate limits, report FP rate with and without appeals. Still partial.
3. **Impact unproven.** No target user, no pilot, no "share of people helped". Very low-traffic sites have too little data to calibrate.
4. **Innovation is packaging.** Calibration and abstention are known; the engineer will compare to CrowdSec + Anubis + logistic regression. Concede the detector is standard, the product is the measurement layer.
5. **Stage-judge market question.** Cloudflare is free. Evidence that owners want an FP dashboard does not exist; do not invent a market size.

Changes proposed:
- **A. Abstention as the headline, and measured.** A live chart of decisions forced versus deferred against a baseline that must choose, and how many real humans were spared a wrongful block.
- **B. One narrow victim, proven on data.** Privacy-browser and JS-off humans. State "of N such humans a hard-block guard stops X%, we stop Y%". Needs a dataset slice; unknown whether AUTH has one.
- **C. Open scorecard.** Export the FP-rate-with-CI report in a format any owner can publish.

## Open questions

- Which dataset can supply a privacy-tool or JS-off human slice (needed for B)? Tim.
- What does Hostinger supply as labelled data, and what do their judges care about? Felix at the booth.
- How does the table demo reset in seconds for ten reviewers, offline from venue Wi-Fi? (Design only, nothing built.)
- Pending: grokbot's report on pitch structure, harm numbers and small-site-owner complaints.
