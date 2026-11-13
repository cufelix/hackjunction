# CoordBench — design sketch

**Status: a sketch, not a decision. Nothing is built. We have not picked a project.**

A candidate answer to AfterQuery's [Break the Agent](./challenges.md#break-the-agent--afterquery--3000--1500--500--work-trial--merch):
an RL environment that scores a coding agent on how well it *coordinates* with teammates.
Written against the harness constraints in [`afterquery-harbor.md`](./afterquery-harbor.md)
and the eligibility limits in [`rules.md`](./rules.md) — **all of it would be new code**.

## The shape

Harbor runs one agent per trial, so the team is simulated inside the MCP sidecar: one
OpenCode agent plus **two scripted teammates**, `bob` and `cara`.

- **main container** — a small Python repo (~6 files) and a git checkout. The agent gets one
  ticket, e.g. *"add retry logic to client.py"*.
- **sidecar `hub`** — FastMCP over streamable-http. Runs the teammates, each with a fixed
  per-seed plan: which files it touches, at which tick, what message it posts. Their commits
  go to their own branches in a shared volume.
- **tick = one MCP call by the agent.** Deterministic time, no wall clock. Each teammate
  posts its plan at tick 0, so nothing is hidden; updates land at fixed ticks
  (*"bob changed fetch() to return (data, status) at tick 6"*).
- **MCP tools, all ours:** `read_inbox()`, `post(msg)`, `claim(paths)`, `release(paths)`,
  `list_claims()`, `status()`, optionally `submit()`.
- **seeded conflicts** — the agent's ticket overlaps a teammate's in a way that *is* visible
  from messages and claims: a shared helper whose signature bob changes, or a file cara
  claimed. ~5 seeds, varying which overlap appears.

### Scripted teammates, not LLM ones

Deterministic per seed, free to run, reproducible by judges, and every failure is
explainable — which matters when over half the rubric is reproducibility and the judges
rerun everything themselves. LLM teammates add noise to "the gap" (30 %), burn credits, and
let a judge blame a flaky teammate for the agent's mistakes. Keep them behind a
`--teammates=llm` flag as a stretch, never for scoring.

## What it exposes — the "meaningful difficulty" 30 %

1. Never reads the inbox mid-task → builds on a stale interface → tests break after merge
2. Edits a file someone else claimed → merge conflict
3. Duplicates work a teammate announced
4. Over-claims, grabbing everything → blocks teammates → their tickets fail
5. Chats endlessly → burns ticks and tokens with no progress

All are real decisions by the agent. **None come from withheld information or broken tools** —
which is exactly what scores zero on that criterion.

## The verifier (`tests/test.sh` → `reward.json`)

Pull the event log from `hub:/log` — kept on the hub's side, append-only through the tools.
Apply any teammate commits not yet landed, merge bob + cara + agent in a fixed order, run a
**hidden** test suite on the merged result (stays in `tests/`, never copied into the container).

```
reward = team_pass_rate × (conflicts ? 0.5 : 1) − 0.1·min(1, wasted_ticks/budget)
```

`team_pass_rate` covers **all three tickets**, so hogging and blocking cost the agent too.
Report sub-scores alongside: `merge_conflicts`, `stale_api_uses`, `claims_violated`, `ticks`.

## Reward integrity — the 25 % judges attack for 15 minutes

| Exploit | Why it fails |
|---|---|
| Do nothing | Agent's ticket fails. **Gate team pass rate on the agent's own ticket, multiplied in**, so this scores 0 — not 0.33 from teammates |
| `submit()` immediately | Same gate |
| Claim everything | Teammates blocked, their tickets fail, score drops |
| Edit the tests | Hidden, never in the container |
| Fake the inbox or claims | Log lives in the sidecar, append-only |
| Delete teammates' code to avoid conflicts | Hidden tests cover their tickets |
| Hard-code answers | Tickets vary per seed, hidden tests use seeded values |

## The reference run (needed for "the gap", 30 %)

`solution/solve.sh` — a scripted, protocol-perfect policy: reads the inbox, claims only its
own files, adapts to bob's API change, submits. Should score ~1.0.

If they want an agent rather than a script: the same frontier model with a short coordination
system prompt (*"check inbox every N edits"*). **The gap between that and bare OpenCode is the
story we tell.**

## Cost

Hub ~200 lines (FastMCP + a dict of teammate plans), toy repo + 5 seeds, `test.sh` plus a
merge/score script. Realistically one evening for a first pass.

## Open risks

- **Will OpenCode use MCP tools it wasn't told about?** `instruction.md` has to name the hub.
  If the gap exists only because the agent didn't know the tools were there, that's "withheld
  information" and scores **zero** on meaningful difficulty.
- **Do judges accept simulated teammates as multi-agent?** Ask Daksh Raghuwanshi or Ved
  Vedere on site before committing the weekend to this.
