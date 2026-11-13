# Rules and submission — read this before writing any code

## Verified first-hand on the platform

Read directly off `hackjunction.app/participate/45/guide` while signed in, 13 Nov 2026
19:08 CET. These are not secondhand:

- **Team size is 1–4.** (The older rules page said 1–5.)
- **"You are allowed to submit a project for multiple challenges."** Stacking tech challenges
  on one project is explicitly sanctioned.
- On **Fri 23:59** you submit *which challenge you plan to tackle*. You may change it later.
- **Final submission Sun 10:00.** The form closes automatically, no late submissions. You
  must set the project from **DRAFT to FINAL** — *"Draft projects are not reviewed."*
- **Peer review Sun 10:30–14:00, everyone must be there.** *"If you or your team do not take
  part in the peer reviewing, you are not eligible for the main prize."* There's a
  "not present" button reviewers press if a team isn't at its table.
- Top 5 pitch on stage: **exactly 3 minutes, then mics are killed**, plus 3 minutes of
  judges' questions. You can demo from your own laptop.

### What the final submission form requires

Project name, punchline, description · challenge · technologies used · **a 2-minute video
pitch** publicly viewable (*"not a second longer than 2 minutes, and we will be very strict"*)
· demo and source code links · source code publicity · pitch deck (PDF) · team phone numbers ·
Discord username.

⚠️ Two of these are quietly expensive and nobody plans for them: the **video** and the
**pitch deck**. Budget Sunday morning for both, before 10:00.

---

## Eligibility — the part that constrains what we build

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
last edited 2025-06-23. The platform's own Guide page does **not** carry the code rules, so
this remains the governing text we can find. Treat it as binding — the downside is
disqualification.

**A second, 2026-dated source agrees.** Another team's public knowledge base
(`github.com/madviking/junction2026`, commit 178b449, 13 Nov 2026) transcribes a 2026 rules
text with the same no-pre-event-code wording, and its other claims that we *could* check
first-hand — team size 1–4, mandatory peer review for main-prize eligibility — both turned
out correct. Two further claims from it we have **not** verified and should not rely on:
that **all submissions are licensed MIT** (2024 allowed a private repo), and the exact 2026
numbering. If the MIT claim is right it matters for anything meant to outlive the weekend —
worth confirming before the final submission.

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
