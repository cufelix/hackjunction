# Spread (Bahn Express) — assessed, and it is not a side bet

**Verdict: it eats one person for the weekend, ~15–25 hours. Only worth it if someone owns it
completely.** Assessed 13 Nov 2026 ~20:00 local from `spread.bahnexpress.com` (`/llms.txt`,
`/kit/API.md`, `/kit/RULES.md`, `/kit/HANDBOOK.md`, `/judging`), read-only — nothing
registered.

Brief and prizes: [`challenges.md`](./challenges.md#spread--bahn-express--2000--1000--500).

## The API

REST, Bearer auth, one key per team from `/join`.

```
POST /v1/worlds              POST .../actions     (1–50 per call)
POST .../advance             GET  .../events?after=
GET  .../drivers             GET  .../score
POST .../fork                (practice worlds only)
```

Events by polling **or** WebSocket (`.../stream?after=seq`, reconnects by seq) — both work.

**Practice worlds run on your own clock** via `advance`, so a week simulates in seconds. That
makes local tuning loops cheap and is where the real edge is.

**Rate limits:** the practice limit is shared and shrinks as more teams practise — ~1000 req/s
at 10 teams, ~50 req/s at 200. The final allows 20 actions/s and 50 reads/s, hard 429 above.

**Latency constraint:** each price request must be answered within 15 world-minutes ≈ **4.5
real seconds** in the final. Miss it and the dealer takes your last quote.

## Running unattended

Designed for it. The kit has a `--final` flag that waits for the final world. The clock runs
itself: ~18 s per world-hour, ~34 min total. If the bot crashes you can restart and it rejoins
the same week — but every request missed meanwhile goes to the competitor. They recommend a
cloud VM near Stockholm.

## The kits, and how far from winning they are

TypeScript (starter 394 lines, client 307, types 550) and Python (starter 432, client 262,
types 555). No dependencies.

**The starter loses money:** House starter sits at **−€287k over 5 runs, 55 % on time,
unranked**. The bar is House best at about **+€60k over 5 runs** (~€12k/week at level 8).

Left for us to write: parsing order notes, a pricing model learned from won and lost quotes,
driver-fee escalation (drivers wait if they expect raises, and drop bookings when the fee is
thin), and protecting the 90 %-on-time constraint.

**Penalties bite:** late refunds 10–100 %, 2× the price for an undelivered car, plus €150 and
€100 fines. Missing 90 % on time means **not ranked at all**.

## What's legal to exploit

- **Allowed:** unlimited practice worlds on your own seeds, `fork`, `advance`. Offline
  grid-search of parameters at high speed is the real edge.
- **Not quoting a request is legal** and only costs a little dealer goodwill. Cancelled orders
  don't count toward the 90 %. **Being selective about which cars to win is the main lever.**
- **Forbidden:** forking or revealing ranking runs.
- No rule found on LLM use or multiple accounts. Leave that alone.

## The schedule trap

Ranking rounds opened Fri 19:30. **Registration closes Sun 07:30; the final launches Sun
08:00** and runs ~34 min unattended. Someone must be awake for that, or the bot must already
be on a VM.

## Why it might still be tempting

Winning needs only **top 10 by margin**, then 5–10 slides to a jury that scores elegance,
clarity, creativity and robustness. After top 10, margin stops counting. At Fri 20:01 the
leaderboard showed House best +€60k ranked and **other teams 0 played** — with ~18 entries,
top 10 may be a low bar.

But "top 10" still means clearing 90 % on time and making money, which is the 12–20 hour part.
