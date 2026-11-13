# Break the Agent — how the harness actually works

Verified against Harbor's source (shallow clone at HEAD, 13 Nov 2026) by a worker session,
not from docs. Covers AfterQuery's "Break the Agent" challenge
([brief](./challenges.md#break-the-agent--afterquery--3000--1500--500--work-trial--merch)).

**Headline: there is no step/reset/reward MCP protocol.** MCP is only the channel the agent
uses to call tools. The reward comes from a verifier script that runs after the agent stops.
If you were picturing a Gym-style env behind MCP, that's wrong.

## The pieces

**Harbor** — `github.com/harbor-framework/harbor` (`laude-institute/harbor` redirects there).
The Terminal-Bench team's agent-evaluation framework. ~7k stars, pushed today, actively
maintained.

```sh
uv tool install harbor
harbor run -p <task_dir> -a opencode -m <provider/model>
harbor run -p <task_dir> -a oracle          # sanity run against the reference solution
harbor jobs start --config cfg.yaml --yes   # job-config form
```

**OpenCode** — `github.com/anomalyco/opencode` (formerly `sst/opencode`). Harbor installs it
inside the task container via nvm/npm and runs `opencode run --format=json`, parsing the
JSON-lines output into a trajectory. Agent names: `opencode`, `opencode-v2`.

## The MCP contract

**There are no required tool names — we choose the tools.** The environment is a Harbor
*task*; the MCP server is just a Docker sidecar the agent can call.

`task.toml`:
```toml
[[environment.mcp_servers]]
name = "mcp-server"
transport = "streamable-http"   # or "sse" (default), or "stdio" with command/args
url = "http://mcp-server:8000/mcp"
```

`environment/docker-compose.yaml` adds the sidecar next to `main` (the agent container).
Harbor writes an OpenCode config whose `mcp` block is built from the task's servers —
stdio becomes `{type:"local", command:[...]}`, sse/streamable-http becomes
`{type:"remote", url, oauth:false}` (`src/harbor/agents/installed/opencode.py:844`).

Server shape, from `examples/tasks/hello-mcp` and `examples/jobs/mcp-proof`: a `FastMCP("x")`
with `@mcp.tool()` functions, `mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)`,
plus a `/health` route for the compose healthcheck.

MCP servers can also be injected at runtime without touching the task:
`agents: [{name: opencode, mcp_servers: [...]}]` in the job yaml, or `--mcp-config .mcp.json`.

## How the reward actually gets produced

After the agent finishes, Harbor runs `tests/test.sh`. It must write **either** a float to
`/logs/verifier/reward.txt` **or** a flat `{name: number}` JSON to `/logs/verifier/reward.json`.
Values must be numeric and finite or the run fails
(`src/harbor/verifier/verifier.py:69-124`).

The verifier can query the sidecar to check what the agent actually did over MCP — the
`mcp-proof` example curls `http://runtime-mcp-server:8000/proof`, which is server-side state
set by the tool call. **That's the pattern for us:** the hub logs claims, conflicts and token
spend; `test.sh` pulls that log and scores it.

"Step" and "reset" are ours to design as MCP tools. Each Harbor trial is a fresh container —
that *is* the reset. `task.toml` also has `multi_step_reward_strategy` (`mean` or `final`)
for multi-step tasks; not yet investigated.

## Packaging

Standard Harbor task directory:

```
instruction.md
task.toml
environment/{Dockerfile, docker-compose.yaml, sidecar/}
solution/solve.sh      # the oracle — doubles as our "stronger reference run"
tests/test.sh          # the verifier
```

It's Docker. `github.com/nathaniellaurent/harbor-dungeon-maker` is a public template from a
prior AfterQuery submission with an oracle, naive baselines, pass@k runs across models and a
hidden verifier seed set — it matches the judging rubric almost 1:1. Worth reading before
writing anything.

## ⚠️ The blocker for CoordBench

**Harbor runs ONE agent per trial. There is no multi-agent mode.** CoordBench's whole premise
is scoring a *team* of agents.

Two ways out, neither free:
1. Simulate the team inside the environment — scripted or LLM teammates behind the MCP
   sidecar, with the OpenCode agent as one member of the team.
2. Write a custom Harbor agent.

**Ask the AfterQuery mentors on site (Daksh Raghuwanshi, Ved Vedere) whether a multi-agent
setup inside the sidecar is acceptable to the "standard harness" rerun**, since the judges
rerun everything themselves with their harness. If option 1 is acceptable the challenge stays
open to us; if not, CoordBench as conceived does not fit this challenge.

## Smaller gotchas

- Harbor's OpenCode mapping **drops MCP `headers`** — the sidecar can't use auth headers, so
  rely on the compose network instead.
- The sidecar must be ready before `main` starts: `depends_on` with `service_healthy`.

## Unverified

- **The scoring skeleton AfterQuery promises is not public.** Their GitHub org only has
  `AfterQuery/storebench`; no Junction starter repo found. Assume it's handed out on site.
  Likely just Harbor's `tests/test.sh` → `reward.txt` template (Harbor ships task generators).
- `multi_step_reward_strategy` semantics.
