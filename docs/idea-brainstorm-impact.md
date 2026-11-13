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

## Real complaints from owners and visitors (subagent, web, 2026-11-13)

Only pages that opened are listed as verified. Gaps: no forum posts found on fake signups or comment spam; the Hostinger and WordPress forums did not open.

| Source | Who | What (short) |
|---|---|---|
| github.com/TecharoHQ/anubis/issues/95 | visitors | JS-off users locked out; older devices broken; image challenge is an accessibility problem; maintainer: "Please just enable JavaScript" |
| github.com/TecharoHQ/anubis/issues/1248 (Nov 2025) | visitor + FreeBSD ports maintainer | Brave challenged on wiki.freebsd.org, a second machine was not; owner side: crawler load "all maxed out for days on end" |
| news.ycombinator.com/item?id=44968145 (Aug 2025) | commenters | "legitimate users of non-mainstream browsers are blocked" |
| lwn.net/SubscriberLink/1028558/6640bdca9c856920/ (Jul 2025) | LWN | comment display suppressed for anonymous users under scraper load; Duke library pilot saw 12 people report problems in one week |
| lemmy.4d2.org/post/2397977/3181208 | small site owner | "AI went nuts on my website and generated a $155 excessive bandwidth bill" on a low-traffic site |

UNVERIFIED (snippets only): Cloudflare captcha loops even without VPN (forum.palemoon.org/viewtopic.php?p=259369); Hostinger help page says strict privacy extensions can trigger the verification page (hostinger.com/support/?p=16180); an r/Hostinger thread and a LowEndTalk thread about Hostinger robot verification support.

Takeaways:
- Blocked humans are a small quiet group (12 reports in a week at Duke); owners have no count of how many they lose. "Challenged versus solved" on a dashboard would be new.
- Owners cannot tell why a visitor was blocked; maintainers ask reporters to define "blocked" and send the error text.
- Scraper pain hits low-traffic sites (LWN, FreeBSD, a $155 bill).
- Skeptics say walls work only "because attackers haven't tried yet", so owners want evidence of adaptation, not just friction.
- Candidate "real human locked out" story with a named, dated source: Anubis issue 1248 or the Duke pilot.

## Converged direction (Felix and Oliver agree, Tim not yet confirmed, 2026-11-13 20:30)

- **P1 is the product:** self-hosted visitor guard on the owner's own VPS. The earlier "type any domain, see who pushes it on Bluesky" idea (P2) is dropped as a headline: Jetstream is unsigned and not meant for anti-abuse, naming accounts risks GDPR Art. 9, labelled data does not exist, and Felix's 150 s measurement found the largest near-duplicate cluster was 2 accounts.
- **Demo: the judge is the traffic.** A reviewer scans a QR on their own phone, the guard on the Hostinger VPS shows a verdict about them: score, evidence, and "not enough evidence" when that is the honest answer. They can switch on a VPN, turn off JavaScript, or use a privacy browser and see whether they are wrongly accused. No install, no scraping, no third-party accounts.
- **Impact line:** every false positive is a person the site turned away (VPN, JS off, old phone, shared or CGNAT IP). Cloudflare measured it: CGNAT IPs are rate limited three times more often (median bot rate 4.8% vs 4.7%).
- **Design rules to write down:**
  1. Never key a verdict on IP alone (venue Wi-Fi is one shared IP).
  2. Abstention must come from a real uncertainty threshold, not a script.
  3. Show at least one case the guard got wrong.
  4. A replay is always labelled as replay.
  5. No beacon (mouse/scroll) layer: legal risk, spoofable, no external eval data (M4D dataset not downloadable).
- **Pitch order sketch (2-min video, 3-min stage):** one real wrongly blocked human (Anubis issue 1248 or the Duke pilot) 15 s; live QR demo 60 s; numbers 45 s (Cloudflare 3x CGNAT, our measured FP with CI, abstention rate); how it differs 30 s; runs on a 2-core VPS 15 s.
- **Working name:** "the detector that must be able to apologize", subtitle "I don't know as a feature".
