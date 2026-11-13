# The five seconds

**Status: a design proposal, written during the Friday-night freeze. Not a decision, not built.**

Oliver found that Junction's peer review runs on **Gavel, which is pairwise**: a reviewer does
not score us against a rubric, they are asked *which of these two projects is better*. That
changes the target. Depth does not win a pairwise comparison — **one moment the reviewer
understands without being told** wins it. We do not have that moment designed yet. This file
proposes it.

Everything below assumes the **public mode** from [`who-its-for.md`](./who-its-for.md): the
reviewer types a domain and installs nothing.

---

## The moment: separation the reviewer watches happen

A reviewer walks up. They type a domain — theirs, or one from the news. Within seconds, real
posts pushing that domain's links arrive from the live Bluesky firehose as individual cards.

Then, while they are still watching, **some of the cards move.** Cards that share a
fingerprint pull together into a single cluster and get one label. The rest stay scattered.

Nobody has to explain what happened. The reviewer saw forty separate things become one thing.
That is the entire pitch delivered in four seconds, with no words.

**Why this and not a graph.** Every project in this space ends in a node-link diagram, and a
graph is a picture of a conclusion someone else reached. Separation is the conclusion *being
reached*, live, on data the reviewer chose. They cannot suspect a fixture, because they picked
the domain.

**Why it is credible.** The source is verified first-hand and needs no key: Bluesky Jetstream,
`wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post`,
measured at ~40 posts/sec, records carry `langs` and `facets`. It is a real firehose of things
real people are posting right now.

## The second moment, which is the one that wins

Three columns, never merged into one number:

| **Coordinated** | **Watching** | **Clear** |
|---|---|---|
| clustered, with the evidence that clustered it | *not enough evidence yet* — and what is missing, counted: "4 accounts, need 2 more" / "need 40 more minutes" | everything the detector looked at and let through |

The middle column is the product. Every competing demo shows a detector catching something.
**No demo shows a detector declining to accuse**, naming what it would need, and changing its
mind on camera when the evidence arrives. A reviewer who sees a tool admit uncertainty trusts
the other two columns more, not less — and it is the only one of the three a competitor cannot
copy by Sunday, because it requires having thought about false positives from the start.

The right-hand column matters more than it looks: a detector that only ever shows its hits is
unfalsifiable. Showing the traffic it cleared is what makes the precision figure mean anything.

## What we must not claim while doing it

🔴 **Public mode mostly surfaces coordination, not propaganda.** On an open firehose, clustered
link-pushing is usually spam, engagement farming and marketing — not state information
operations. The content layer only adds "propaganda" where it actually fires on a technique.
So the label on a cluster is **"coordinated push"**, and "propaganda" is a second, separate
badge that appears only when the content layer says so, with the technique named.

Felix's framing — *type a domain and it flags all the propagandists* — is the right product
instinct and the wrong promise. What we can honestly put on screen is: *these accounts are
pushing this domain together, here is the evidence, and here is whether the text uses known
propaganda techniques.* Saying more than that is the single easiest way for a judge with
domain knowledge to take us apart in the 3 minutes of questions.

## The two ways this demo dies, and the fix

1. **A quiet domain produces nothing.** Type a small site and the screen stays empty — a dead
   demo in a pairwise comparison is worse than a boring one. Fix: the field is pre-filled with
   a domain that has live volume, the reviewer is invited to change it, and an empty result is
   itself a displayed verdict (*"no coordinated push on this domain in the last N minutes"*)
   rather than a blank panel. An honest empty answer still demonstrates the tool.
2. **The venue network fails.** Fix: a recorded replay, **labelled as a replay on screen**,
   never presented as live. We say which one the reviewer is looking at. A demo caught faking
   live data loses the challenge whose own brief asks for a false-positive rate.

## What this costs us to be true

The clustering has to run fast enough to visibly move cards within seconds of the first posts
arriving — so the fingerprint for public mode must be cheap: near-duplicate text (MinHash),
posting-time proximity, account age, and link-target identity. The expensive content model
runs *after* a cluster forms, on the cluster, not on every post. That ordering is also what
keeps the per-verdict cost honest.

## Open

- [ ] Does a mid-size domain actually produce visible clusters on Bluesky within ~60 s? **Needs
      a measurement before we commit the demo to it** — the firehose rate is verified, the
      clustering yield on a chosen domain is not.
- [ ] What does the connected mode's equivalent moment look like, for the Hostinger mentor who
      asked for traffic matched to behaviour? Probably the same separation, driven by request
      sequences instead of text.
