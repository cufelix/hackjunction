# Experiment 01 — does a typed domain yield anything on the live firehose?

**Run 13 Nov 2026 21:22 CET, 150 seconds, Bluesky Jetstream, keyless.** Probe:
[`experiments-domain-yield.py`](./experiments-domain-yield.py). Privacy: DIDs hashed to 12 chars,
never printed; no handles collected; nothing retained.

This tested the one assumption [`the-five-seconds.md`](./the-five-seconds.md) marked as unverified
and on which the whole demo rested.

## Raw result

```
6899 posts in 150s (45.9/s); 1341 carried a link (19.4%); 592 distinct domains

domain                      posts  accts   dup-text clusters
klipy.com                     145    141
youtube.com                    74     70
youtu.be                       57     51   1 cluster, biggest spans 2 accounts
bsky.app                       48     45   1 cluster, biggest spans 2 accounts
xvactu.fr                      32      1
substack.com                   23     19
dlvr.it                        23     15
blogspot.com                   22     10   1 cluster, biggest spans 2 accounts
globo.com                      19      5   5 clusters, biggest spans 2 accounts
x.com                          19     14
kick.com                       17      3
vero.co                        15      1
nytimes.com                    12     12
promobit.com.br                12      1
businessinsider.es              9      1
lavieweekly.com                 9      1
ebird.org                       8      1

domains with >=  2 mentions: 148      >= 3: 81      >= 5: 37      >= 10: 19
```

## 🔴 Finding 1: the demo moment I designed did not appear

**"Forty accounts, one fingerprint" is not visible from a cold start on a live firehose.** Across
6,899 posts, the largest near-duplicate text cluster spanned **two** accounts. Not forty. Not ten.
Two.

Three possible reasons, and we cannot yet tell them apart: 150 s is a short window and coordinated
campaigns may operate over hours; Bluesky's own moderation may already remove the obvious ones; or
our fingerprint (exact normalised-text match) is too strict, because real coordination paraphrases.

**Consequence for the demo:** a judge typing a domain cannot be promised a sockpuppet cluster
within five seconds. Promising it would have been a demo that works on the happy path only —
exactly what the brief warns against. **Good thing we measured instead of building it.**

## ✅ Finding 2: a different signal is there instantly, and it is cleaner

Look at the posts-to-accounts ratio rather than text similarity:

| domain | posts | accounts | ratio |
|---|---|---|---|
| `klipy.com` | 145 | 141 | **1.03** — organic, a widely used embed service |
| `youtube.com` | 74 | 70 | **1.06** — organic |
| `nytimes.com` | 12 | 12 | **1.00** — organic |
| `globo.com` | 19 | 5 | **3.8** |
| `kick.com` | 17 | 3 | **5.7** |
| `vero.co` | 15 | 1 | **15.0** |
| `xvactu.fr` | 32 | 1 | **32.0** — one account, a post every 4.7 seconds |

**That is a ~30× separation between organic and automated, available in seconds, computable from
two integers.** One account posting a domain's links 32 times in 150 seconds is not a human
reading and sharing; it is a feed. `dlvr.it` and `buff.ly` appearing in the top list is
corroboration — those are scheduling services, i.e. automation we can name.

This is **exactly the cheap deterministic first tier** the three-tier architecture asked for,
and it is now empirically grounded in our own measurement rather than assumed.

## What changes in the design

1. **The five-second moment becomes the ratio, not the cluster.** Type a domain, and the
   organic/automated split is visible immediately, with the two counts as the entire explanation.
   It is legible without a legend, which is what pairwise review needs.
2. **Sockpuppet clustering moves behind an accumulation window.** The product is a **monitor that
   runs continuously**, not a search engine answering from cold — which is what a real deployment
   does anyway. For the demo this means a warm buffer that has been collecting for hours before
   judging, labelled with how long it has been watching.
3. **"How long have I been watching" becomes a first-class number on screen**, next to the verdict.
   It is the honest version of the middle column: *not enough evidence yet, and here is the window
   I have.*
4. **Baseline rates are now measurable, which is what the false-positive panel needs.** 19.4 % of
   posts carry a link; 592 domains appeared in 150 s; only 19 reached ten mentions. So a typed
   domain will often have **little or no traffic**, and the empty answer has to be a designed
   state, not a blank screen.

## Caveats, stated plainly

- **One 150-second run, one time of day, one region's firehose.** Not a benchmark. Ratios will
  differ by hour and by domain popularity.
- The eTLD+1 extraction is crude (a hardcoded two-part suffix list), fine for a yield measurement
  and not fit for production.
- A high ratio **is not an accusation**. A news outlet's own auto-posting account scores exactly
  like a spam feed. This measurement identifies *automation*, which is the benign-automation class
  in the four-class task definition — not *undesired* automation, and certainly not propaganda.
  Conflating them is the Botometer mistake.

## Next experiment

- [ ] Run the same probe for 30–60 minutes and ask whether cross-account text clusters appear at
      all with a longer window. This decides whether the sockpuppet tab is real or lab-only.
- [ ] Re-run with near-duplicate matching (MinHash over shingles) instead of exact match, to test
      whether the strict fingerprint was the reason clusters were absent.
