# Entire — tested on this machine, here's what actually works

Installed and exercised 13 Nov 2026 ~19:00 CET by a worker session, in `/tmp/entire-scratch`
(a local clone with origin removed). **Our repo was not touched.** CLI 0.13.4, graph plugin
v0.6.0, brain plugin v0.5.1.

Why we care: "Best Use of Entire" is an MLH prize and Entire's primitives overlap our problem
space. The ask and the prize are in
[`challenges.md`](./challenges.md#the-mlh-swag-tier) and
[`research-junction-2026.md`](./research-junction-2026.md#entire--deep-dive).

---

## 🔴 Before anyone runs `entire enable` in our repo

**`entire enable` makes checkpoints — which contain full agent transcripts — get pushed to
the git remote on push. Our GitHub repo is public. That would publish every transcript.**
Secret redaction is "best effort" by their own description.

If we use it, pick one first:
```sh
entire configure --checkpoint-push-policy manual
# or send checkpoints somewhere private:
entire configure --checkpoint-remote <private-repo>
```

Two more side effects worth knowing before running it in our tree:
- `entire enable -y --agent claude-code` installs **10 Claude Code hooks** into the repo's
  `.claude/settings.json` and creates `.entire/`.
- `entire graph setup --repo .` writes `AGENTS.md` and `.entire/agent-guide.md`, adds a
  SubagentStart hook to `.claude/settings.local.json`, and **modifies `CLAUDE.md`**.

## What works without logging in

`enable`, `checkpoints`, `graph`, `brain setup`, `checkpoint tokens` — **none of these need
`entire login`.** That matters: the whole token-efficiency measurement can be scripted with
no account and nothing manual from Felix.

| Command | Status |
|---|---|
| `entire enable -y --agent claude-code` | ✅ works (see warning above) |
| `entire graph setup --repo .` | ✅ works |
| `entire graph impact` | ✅ works, offline, no login |
| `entire brain setup --no-backfill` | ✅ works — built in 3.4 s, BM25 only |
| `entire checkpoint tokens <B> --compare <A> --json` | ✅ **works, tested on real checkpoints** |
| `entire brain connect` | ❌ fails without a remote |
| `entire brain mcp` | ❌ **does not exist** |
| `entire login` | not run — needs Felix |

## `entire brain mcp` does not exist — the hack guide is stale

```
$ entire brain mcp --help
unknown command "mcp" for "entire-brain"
```

Not hidden either; `--help-all` doesn't list it, and the Brain docs agree: *"Brain v0.5.1 does
not provide the `entire brain mcp` command."* The hack guide's
`claude mcp add entire-brain -- entire brain mcp` is **wrong**. That settles the contradiction
flagged in the research doc.

The only MCP server is the core `entire mcp` — stdio, read-only, exposes agent-help and
status, and is hidden from `entire --help`. **Brain is driven through the CLI, not MCP.**

`brain` subcommands: setup, add, agent-guide, webhook, config, overview, brief, query, get,
multi-get, dash, recall, remember, distill, memories (sync/gc/import/vitality), connect,
disconnect, verify, patterns, inspect, show, viz, status, stats, workspace, bundle, memory,
entities, repair, docs, refresh, watch, stop, start, reset, repo-identity, path, capabilities,
doctor, version.

Also: brain leaves a background watcher running (`entire-brain workspace watch --distill`).
⚠️ **`entire brain stop` does not kill it** — it only prints "background brain work paused"
and the process keeps running. Kill the pid yourself.

## The token number — we can produce it, fully scripted

`entire checkpoint tokens <B> --compare <A> --json` **exists and was tested** against two
headless `claude -p --model haiku` runs in the scratch clone. Each commit picked up an
`Entire-Checkpoint: <id>` trailer; prefix IDs work.

JSON gives `tokens{total, input, cache_read, cache_write, output, api_calls}` plus model,
session_id and recommendations[]. Under `comparison`, each field carries
`{baseline, current, change, change_percent, direction}`, with `status`, `cache_read_caveat`
and `qualification`.

**Cache-read and fresh tokens come back separately** — exactly what the slide needs, and
exactly what we said we'd report honestly rather than quoting Entire's "up to 71 %".

So the A/B harness is fully automatable: run the same tasks twice with `claude -p` in an
enabled repo, once with graph/brain and once without, diff the checkpoints.

## `entire graph impact`

Offline, no login. This is the piece that could make a coordination hub smarter about file
claims — claim the affected code, not just the named file.

```sh
entire graph impact --symbol NAME|<file>:<line> --repo . \
  [--depth 1|2] [--limit 15] [--format text|json] [--exclude-tests] \
  [--profile syntax-only|fast|full]
```

JSON returns `focus{id, file_path, start_line, …}` and
`callers{total, direct, transitive, entries[{endpoint, relation, depth, via, call_site}]}`;
per the help text also callees, type consumers, data flows and co-changing files.

Cold index took 36 s on a large Go repo — `entire graph index` warms the cache.

## What needs Felix

- `entire login` (browser redirect to auth.entire.io, or `--device` for a device code)
- Installing the **Entire GitHub App** on the repo — org admin
- `entire brain connect` / hosted sync
- The **public checkpoint link** the submission requires:
  `https://entire.io/gh/OWNER/REPO/commit/FULL_SHA` — needs the GitHub App and the branch pushed
- The MLH promo code

## Submission checklist (from the hack guide)

- which Entire features we used
- demo link or recording
- source with setup instructions
- an accessible checkpoint link **plus its ID**

## Install notes

`install.sh` was read before running. Clean: checksum-verified, no sudo, doesn't edit rc files
on its own. Three things worth knowing:
- it also installs a second binary, `git-remote-entire`, for `entire://` clones
- it ends by running a hidden `entire curl-bash-post-install`, which only offers to add shell
  completion to `~/.bashrc` after an explicit "Yes" (verified in `cmd/entire/cli/setup.go:2439`)
- it downloads from install.entire.com first, falling back to GitHub releases

`~/.claude/settings.json` and `~/.claude.json` were checksummed before and after: unchanged.

**Leftovers on this machine:** `/tmp/entire-scratch`, `/tmp/entire-cli-src`, the brain watcher
process, and binaries in `~/.local/bin` and `~/.local/share/entire`.

## Unverified

- Whether `brain connect` also requires login (it failed earlier, on the missing remote).
- The MLH promo code mechanics — the hack guide has no claim wording, and the MLH page wasn't
  opened. "GitHub account ≥ 30 days old" is still unconfirmed.
