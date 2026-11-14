# Verda GPU box — the setup recipe that actually works

**Source: Tim's Claude (`@claude-timi/main`), who ran it end to end on box A and asked for it to be
written down here.** Committed by `@claude-felix/claude` on 14 Nov 2026, 01:25 CET. Every fix below
was derived from pip's own error messages on a clean box, not from reverted commits — `RULES_AND_SCORING.md:87-88`
applies, and only commit *titles* were read from history. PR #1 is a public deployment PR.

Verified on: Verda `1RTXPRO6000.30V`, driver 580.178.04, Ubuntu 24.04, starter repo `main` at `36daf5b`.
**Whole run ≈ 25 minutes.**

---

## 🔴 Money first: `shutdown` keeps billing. Only `delete` stops it.

Tim's Claude, verbatim finding: stopping a Verda instance **continues to charge**; charging stops
only when the instance is **deleted**. At **$2.81/h** an idle forgotten box is ~$67/day. Nobody
should leave a box shut down overnight and assume it is free.

Two more facts, as of 14 Nov 01:00 CET:

- ⚠️ **`verda.com` is serving a Cloudflare Registrar "services suspended" page** — console *and*
  `api.verda.com`, with `datacrunch.io` redirecting to it. HTTP 200 from those hosts is a parking
  page, not the service. Only the **legacy API `https://api.datacrunch.io/v1`** works (valid TLS),
  and Tim's account can see and drive box A through it. So **no new API keys and no console** until
  the domain is fixed.
- **Credits are per person, on separate accounts** — not a pool. Tim's Verda account is **prepaid
  with no card**, so it cannot be charged past its credit. Whether Felix's and Oliver's accounts
  behave the same is **unknown and is their own call to check.**

## Instance

| | |
|---|---|
| type | `1RTXPRO6000.30V` on-demand, $2.81/h (free in FIN-02 and FIN-03) |
| image | `24.04.cuda13.4.docker` |
| OS disk | **300 GB** |

Then clone the starter repo and run `bash aaltonomous/cloud/setup_instance.sh` with the four fixes
below. **Without them `setup_instance.sh` does not finish on a clean box** — pip constraint
conflicts.

## The four fixes

1. **Extra apt packages** (the prereq step prints them itself):

   ```
   libturbojpeg libglu1-mesa libxt6 libvulkan1 libegl1 libgl1
   libxrandr2 libxi6 libsm6 libice6 libxkbcommon0 libglib2.0-0t64
   ```

2. **The `isaacsim` step gets its own copy of the constraints**, with the exact pins of
   `isaacsim-kernel 6.2.0.0`:

   ```
   click==8.3.3
   urllib3==2.7.0
   psutil==5.9.8
   llvmlite==0.46.0
   fsspec==2026.2.0
   ```

3. **Global `isaac-constraints.txt`** gets only `psutil==5.9.8` (rl-games wants < 6) and
   `fsspec==2026.2.0`. Two pins must be **left alone**:

   | leave at | because |
   |---|---|
   | `llvmlite` **0.50.0** | `numba 0.68.1` inside Isaac Lab wants ≥ 0.50 |
   | `onnxruntime` **1.22.1** | `unitree_sim_isaaclab` requirements pin exactly that |

4. **The `isaaclab` step gets a constraints copy with the `rerun-sdk` pin removed** — Isaac Lab
   wants ≥ 0.29, the freeze has 0.23.4. That single exclusion was enough; no extras had to be
   switched off, and `rl-games` passed via `psutil 5.9.8`.

> 🔴 **Do not apply PR #1 globally.** It sets `llvmlite 0.46` and `onnxruntime 1.26` globally,
> which breaks Isaac Lab and the `repos` step respectively.

## 🔴 First start: black cameras are expected, and the fix is a restart

On a fresh box every camera can come up black — **RGB mean 0, depth 0 % valid**. Root cause found
on box A: the **RTX renderer wedges on first start**. The kit log shows

```
[omni.rtx] make sure you call cmdBegin before calling cmdEnd
RTX streaming did not complete within 30.0 s
```

and the shaders never begin compiling — **GPU sits at 0 %**. No code, driver or DR change is needed;
driver 580 is fine and 595 is not required.

**Fix: `aalto.sh stop` then `aalto.sh sim`.** After that, box A reported `aalto check` **9/9 PASS**,
depth valid 100 %, median 565 mm.

**Runbook for every box:** if the probe reports RGB mean 0 or depth 0 % valid, or the log has the
`cmdBegin`/`cmdEnd` error, or the GPU is at 0 % — **restart the sim immediately, do not wait.**

> A correct first start takes **about 4 minutes**, because it compiles shaders (~250 MB cache).
> **If it says "ready" in ~50 s, be suspicious** — that is the wedged path.

Tim keeps the detail in `knowledge/humanoid/08-sim-rendering-debug.md`.

## Verify, and the two results worth knowing

`setup_instance.sh` on `36daf5b` ends with **66 tests: 1 fail, 1 skip, 1 expected failure.**

- The only FAIL is `test_scene.py:235 test_cubes_and_bins_visible`. The out-of-frame corners are the
  `z=0` corners at `[0.2575|0.3025|0.5975|0.6425, 0.0975, 0]` — the **near edge of the spawn region
  sits below the bottom edge of the image.** This reproduces on a clean box, so it is a property of
  the scene, not of anyone's install. Not a broken environment.
- ✅ **`deploy/test_deploy.py::test_stuck_policy_latency_cap` PASSES on a clean box.** An earlier
  11–12 s failure on a laptop was host load, not a bug. That question is closed.

## ⚠️ `aalto baseline` runs the ORACLE, not the classical baseline

`aalto baseline` passes `--policy oracle`. For the step-0 reference measurement both have to be run,
separately and sequentially:

- the **classical baseline** as a submission server, via `--policy ws://`;
- the **oracle** on the sim clock, because per the organisers it was never re-measured after the
  clock fix (see `BUILD-HANDBOOK.md` §3.4.2).

## ⚠️ `mock_world` draws triangles as boxes

Tim's Claude's note for build item #7, originally Oliver's finding: **`mock_world` renders the
triangular prism as a box.** So a CPU test of a shape classifier against the mock **cannot verify
the triangle** — the one shape the classifier most needs to get right. A real check needs an Isaac
render. Box A can supply frames with **`aalto.sh snap`** once the cameras are up.
