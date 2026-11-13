# Eligibility rules — read this before writing any code

**Bottom line: we cannot submit Warren, or anything built on Warren's existing code, as our
project.** Both Junction's rules and MLH's forbid it.

## What Junction's rules say

Source: [Junction Rules](https://junctionhq.notion.site/Junction-Rules-5d5856395abd48e4a8740e533ffc5577),
read through Notion's public API on 13 Nov 2026.

> "No code written before the event."

> "All code outside of open source libraries must be written at the event. If you use any
> code written specifically for this idea in advance, or work on a new project that is
> somehow related to an already existing company, you will be disqualified."

Also from that page: organizers must be able to access the source code; teams are 1–5;
at least one challenge must be entered; the pitch video is 2 minutes or less.

⚠️ **Caveat on the source.** That page is titled *official rules for Junction 2024* and was
last edited 2025-06-23. No 2026 version was found, so it is the latest visible version — but
it is **not confirmed as the 2026 rules**. Treat it as binding until an organizer says
otherwise, because the downside is disqualification.

## What MLH's rules say — stricter

Source: `github.com/MLH/mlh-hackathon-rules`, `Rules.md`. Junction follows the MLH Code of
Conduct, and the MLH side prizes are judged under these.

- **Rule 5** — "All work on a project should be done at the hackathon."
- **Rule 8** — you may rework an old *idea* "as long as they do not re-use code".
- **Rule 9** — "Working on a project before the event and open-sourcing it for the sole
  purpose of using the code during the event is against the spirit of the rules and is not
  allowed."
- **Rule 10** — only new work is judged.

## What this means for us

| Plan | Verdict |
|---|---|
| Submit Warren, or CoordBench built on Warren's codebase | **Disqualifying** under both rule sets |
| Warren as a published open-source library our new project depends on | The only reading that could work — but judges score only new code, and MLH rule 9 catches us if Warren looks like it was published so we could use it here |
| Rebuild the coordination we need from scratch this weekend, taking only the *idea* from Warren | **Clean.** Rule 8 explicitly permits reworking an old idea without reusing code |

**The safe move, and it is cheap:** ask the organizers in writing *now* —
`participants@hackjunction.com` or the Junction Discord — and keep the answer. Do that before
building on anything pre-existing.

### Practical consequence for the project

Whatever we build this weekend has to be written this weekend. Warren is still useful to us
as **tooling for how we work** (the three agent sessions coordinating through it) — that's us
using a tool, not submitting it. What goes in the repo we submit must be new.

If the idea survives without the old code — a benchmark that scores multi-agent coding teams
doesn't actually need Warren's implementation, just *some* hub — then CoordBench is still
live, and we write the hub fresh. See
[`afterquery-harbor.md`](./afterquery-harbor.md) for what that hub has to look like to run
inside AfterQuery's harness.

## Still unverified

- Whether the 2026 rules differ from the 2024 page.
- How past winners handled pre-existing code (no clean repo evidence found).
- Whether individual challenge partners add their own "built during the event" clauses.
