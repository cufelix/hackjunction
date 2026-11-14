# Who it's for — the user decision

**Decided Fri 13 Nov, 20:35 CET, by Felix.** Everything else in this folder is research;
this is a choice. If this choice changes, the architecture changes with it, so change this
file first.

> **The user is someone who runs their own server and can deploy the detector onto it.**
> A developer, a sysadmin, a small agency running sites for clients — Hostinger's **VPS**
> customer, not their shared-hosting customer.
>
> Positioning, one line: **detection you deploy yourself, and your logs never leave your box.**

## Why this user and not the five obvious ones

| Candidate | Why not |
|---|---|
| A platform trust-and-safety team | The default assumption of every project in this space, so the most crowded. They have their own teams, we cannot reach their data, and Hostinger is not a platform. |
| A journalist or fact-checker | Real need, but it is *investigation*, not detection — judged on depth we cannot reach in 36 h, and they would never run a tool on a VPS. |
| The reader | Nobody has ever gone looking for a propaganda checker. No distribution, no measurable value. |
| Hostinger's own abuse team | The ideal user in principle and **unbuildable** — cross-site detection needs Hostinger's fleet, which we do not have. This was a real mistake made earlier today; see `hostinger-decode-the-noise.md`. |
| A non-technical site owner (the restaurant) | Closest runner-up, and the right instinct about *whose pain* it is. But they cannot install anything, so there is no demo where they use it and no path where they adopt it. |

## What picking this user buys us, against the rubric

- **The judge can install it themselves.** Not "watch our demo" but *"run this on your own VPS,
  then attack your own site and watch it get caught."* That is the strongest available answer to
  the brief's own *"not just a demo that works on the happy path"* — and it only exists because
  the user is a person with a server. Speaks directly to **technical execution and live
  deployment, 25 %**.
- **Privacy stops being a compromise and becomes the reason to choose it.** Self-hosted means the
  logs never leave the operator's machine. For an EU operator under GDPR, *"no data leaves your
  server"* is the whole argument for a self-hosted tool over a hosted service. The GDPR exposure
  this project started with does not get mitigated — it disappears.
- **Auditability stops being a claim.** This user reads the rules. A self-hosted detector is
  auditable by definition, so *"is the explanation the real reason?"* is answered by a file
  rather than a slide. Speaks to **explainability, inside the 40 %**.
- **The mentor on the booth owns this product.** The challenge page names **Minijus Savickas,
  VPS Product**. The user we are choosing is literally the customer of the person who will be
  standing there judging it. That is checkable, not a guess.
- **Herd immunity becomes buildable again.** Earlier today the "only a host sees the swarm" idea
  had to be retracted, because it needs Hostinger's fleet. With self-hosting it returns in a form
  we can actually build: each operator keeps their own logs, and they can *opt in* to sharing
  **detection signatures, not data**. We do not need somebody else's fleet, because the operators
  come to us. It is also a natural open-source story.

## What it costs, honestly

- 🔴 **The demo becomes a terminal and a dashboard, and that is the real risk.** Peer review is
  Sunday 10:30–14:00, judged by exhausted strangers at a table, and **a dev tool is the hardest
  format to make legible in fifteen seconds.** This has to be designed against deliberately, not
  hoped away. If we pick this user, the visualisation is not a bonus — it is the deliverable.
- **"Propaganda" narrows.** A server operator experiences bots, scrapers and credential stuffing
  — *traffic*. Manipulation reaches them through the comment, review and form surfaces. That is a
  legitimate reading: the brief lists *"a monitoring service for site owners"* as an acceptable
  form. But we say that out loud and **do not claim to detect state disinformation.**
- **Impact reads smaller to a grand jury.** 20 % is originality and real-world impact, and a
  sysadmin tool is less moving than "protecting democracy". Two counters: Hostinger hosts millions
  of these operators, so the aggregate is not small; and the grand jury is the *second* audience —
  the Hostinger judges award this challenge, and this is their customer.

## What this choice excludes

No analysis of social platforms we do not own. No cross-site view (only a host has that). No
investigative tooling. **No profiling of readers** — we measure the behaviour of requests, not the
behaviour of people. That last one is a design position we can state on stage, and almost nobody
else will take it.

## Consequences for the architecture

- The three-tier shape (cheap deterministic rules → an expensive model on the uncertain band
  only → a human) survives and fits this user well: they understand and will inspect the cheap
  tier, and they are the ones who can label the uncertain band.
- **The "world political situation" tier is dead.** It was already statistically indefensible
  (request-level data joined to week-level events, no power, guaranteed spurious correlation) —
  but the stronger reason is simpler: **it is worth nothing to a person protecting their own
  server.** News belongs in this product only as context shown next to a finding, never as a
  detector.
- Default behaviour is **observe and report, not block.** For this user the damage from
  over-blocking is worse than from under-blocking, which is also why their false-positive rate
  is the product rather than a methodology note.

## Still open

- [ ] Does the judge-installs-it-themselves demo actually work at a noisy venue, or does it need
      a pre-provisioned box per judge?
- [ ] How do we make a self-hosted tool legible to a peer reviewer in fifteen seconds? Unsolved,
      and it is the biggest risk in this decision.
- [ ] Do we ship the opt-in signature sharing, or only describe it? It is the strongest
      "outlives the weekend" story we have.
