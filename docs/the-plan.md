# The plan, as it stands at midnight — a proposal for Felix

**Written 13 Nov 2026 21:30 CET, during the build freeze Felix called. Nothing here is decided
except where it says Felix decided it.** This exists because tonight's reasoning is scattered
across thirteen Warren messages and six documents, and at midnight we start building.

---

## What we build

**A visitor guard the site owner installs on their own server, which shows its verdict to the
person it is judging.**

One sentence for the submission form:

> Bot and coordinated-abuse detection for your own server that explains every verdict in plain
> language, tells you how sure it is, shows you how often it is wrong, and says *"not enough
> evidence"* instead of guessing — including to the visitor it just judged.

## Who it is for

Unchanged from [`who-its-for.md`](./who-its-for.md), **which Felix decided at 20:35**: someone who
runs their own server. Hostinger's VPS customer — the customer of Minijus Savickas, the VPS
Product mentor who judges this challenge.

⚠️ **The two-mode revision I added to that file at 21:10 should be reverted if this plan is
accepted.** It proposed a public mode that scrapes a typed domain; this plan replaces that with a
better answer to the same need (see *The demo*). Awaiting Felix.

## The demo — nobody types a domain, the judge is the traffic

The peer reviewer opens our site on their own phone. **The tool shows them its verdict about
them**, live: the score, the evidence that produced it, and the *"I am not sure"* state.

Then we invite them to look suspicious: turn on a VPN, disable JavaScript, use a privacy browser,
hand it to someone on an older phone. They watch whether it **wrongly accuses them** — and watch
it say *not enough evidence* rather than block.

**Why this and not "type a domain":**
- zero install for the judge, which was the whole point of Felix's instinct
- no third-party scraping, so **no Art 9 exposure** and no live handle on screen — it keeps the
  hard rule from [`the-five-seconds.md`](./the-five-seconds.md)
- the traffic is genuinely real, because the judge makes it
- and it is the one thing the prior art does not do: **nobody shows the verdict to the person
  being judged** (Cloudflare's reasons are Enterprise-only; Turnstile and Super Bot Fight Mode
  show nothing; Anubis gives the visitor an opaque error)

Corroboration, via Oliver, for leaning on live over polished: among the Junction 2025 finalists
the winner had the weakest pitch craft but the most impressive thing **running live**, and the
jury forgave the glitches. (Secondhand, pitches are public on YouTube — worth watching one.)

## Why it scores on Junction's three criteria

Oliver read the official guidebook: **Innovation, Impact on society, Implementation.** Precision
and false-positive rate are *not* Junction criteria — they are Hostinger's. So the honest-numbers
work has to be told as impact, and it can be, because **a false positive is a person**.

| Criterion | Our answer |
|---|---|
| **Innovation** | Nobody explains a verdict to the visitor, publishes their own error rate, or ships *"insufficient evidence"* as a first-class output. Verified against primary docs, not assumed. |
| **Impact on society** | The people today's bot walls lock out: VPN users, JavaScript-off users, older devices, privacy browsers, shared Wi-Fi and CGNAT. **With real complaints, not hypotheticals** — Anubis issue #95 (maintainer: *"Please just enable JavaScript"*), Brave challenged on wiki.freebsd.org (issue #1248), and the HN thread on non-mainstream browsers being blocked. And the systemic case: **Cloudflare's own Bot Management feature file took down ChatGPT, X and Spotify for ~6 hours on 18 Nov 2025 — not an attack**, verified in their post-mortem. Opaque detection is itself the risk. |
| **Implementation** | Runs live on a Hostinger VPS with a public URL. The cheapest detector tier is two integers (below). The labelled eval set exists and is open. |

## What the detector is made of

**Tier 1 — deterministic, two integers, measured tonight.** The posts-to-accounts ratio from
[`experiment-01`](./experiment-01-firehose-yield.md) generalises to requests-per-identity: 1.03
for organic, 32.0 for a feed posting every 4.7 seconds. **~30× separation, no model.** This is the
tier the owner can read and audit, which is why it is first.

**Tier 2 — a model on the uncertain band only**, where tier 1 abstains.

**Tier 3 — `gemini-decision-1` for the calibrated number.** Verified: real, launched 11 Nov 2026,
1–1000 questions per request, 262k context, flat latency to 100 questions, free tier of 200 RPM /
10k RPD. **It emits zero text tokens**, so it can never write the explanation — the explanation is
composed by us from the features that produced the score, which is the discipline we wanted
anyway. Watch the three traps: probabilities clamped to 0.01–0.99, choice questions always pick
something (so every one needs an explicit *none of these*), and `confidence` is only
`max(p, 1-p)`, not a measured uncertainty.

**Explicitly not in it:** no JS beacon (three independent reasons, all recorded), no reader
profiling, no world-news tier, no live handle on screen, no claim to detect state disinformation.

## Evaluation

Each layer against a dataset somebody else labelled: **AUTH** (zenodo 3477932, CC-BY, 4,091,155
requests, labelled sessions with features) for traffic, **PTC** (zenodo 3952415, CC-BY) for
propaganda technique, **IO Honduras** for coordination. Three panels, **never merged into one
number**. Account-disjoint and time-disjoint splits.

🔴 **The open problem on the critical path — label bias.** Oliver's red team is right: a "trusted
human stream" only represents the population it samples, and logged-in users are systematically
unlike anonymous ones. **Proposal: do not use our own trusted-human traffic as positive labels at
all.** Report the false-positive rate only on AUTH, where someone else did the labelling, and
report our own traffic as a separate, clearly-labelled observation. This needs Tim's verdict
before any number goes on a slide.

## What Felix needs to decide

1. **Accept this plan, or keep the public typed-domain mode?** If kept, the Art 9 and Jetstream
   constraints apply and the two-panel separation is mandatory.
2. **Accept the impact reframing** — false positives told as *people wrongly locked out*?
3. **Push the five local commits?** Nothing is on origin.
4. **The preliminary, by 23:55** — challenge plus a short description. Draft text is ready.

## Still open, inherited

- [ ] Read Junction's three criteria first-hand; `INFO_HACKATHON.md` records different grand-jury
      criteria from the platform, and the difference may be peer review vs grand jury.
- [ ] Resolve Cloudflare Logpush's Free-vs-Enterprise conflict if connected mode ever offers it.
- [ ] Confirm Context.dev's credit figure on the platform.
- [ ] Hostinger booth: is AI Audit really unavailable on VPS, and can the CDN export logs?
