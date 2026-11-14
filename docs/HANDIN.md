# Ten-minute hand-in checklist

This checklist assumes the chosen finals overlay has already been promoted to `overlays/submission.yaml`, reviewed,
and committed by a human. `tools/handin.sh` deliberately refuses a dirty source tree. Use a Docker host with a warm
build cache if the hand-in window itself is ten minutes.

## 0:00–1:00 — freeze the exact source

- Confirm the intended commit and a completely clean tree:

  ```sh
  git rev-parse HEAD
  git status --short
  ```

- Confirm the required robot model is a real file inside the build context, not a symlink:

  ```sh
  test -f third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf
  test ! -L third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf
  ```

- Do not start if either check fails. Record the full Git SHA in the submission notes.

## 1:00–2:00 — run the release checks

```sh
../v16/bin/python -m unittest \
  aaltonomous.baselines.classical.test_flags \
  aaltonomous.baselines.classical.test_release_gate \
  tools.test_handin -v
../v16/bin/python tools/release_gate.py finals_r22_chunk
```

The release gate must say `RELEASE GATE PASS`. If another overlay is intentionally promoted, use that overlay name
instead and record the choice.

## 2:00–7:00 — build, push, and measure

Make sure `docker login` has already authenticated the target registry. Pass a fully qualified registry host,
repository, and unique non-`latest` tag explicitly; the script never stores a registry or credentials:

```sh
tools/handin.sh <registry>/<namespace>/<image>:<tag>
```

The default platform is `linux/amd64`. The script performs all of these checks in order:

1. refuses a dirty Git tree and a missing, empty, or symlinked URDF;
   it also verifies the pinned URDF SHA-256 and records it;
2. builds `tools/submission.Dockerfile` for the selected platform;
3. records the local image ID, Docker-reported size, and root-filesystem layer count;
4. starts the container with `--network none` and times the exact policy-server ready line;
5. uses loopback inside that container for one reset and action, validates the action, and requires at least 27 policy
   ticks per wall second;
6. dumps the effective flags from inside the image;
7. pushes using Docker's login state and records the manifest digest returned by the registry;
8. removes the locally built image, pulls `repository@sha256:…`, gives it a fresh local tag, and repeats the smoke;
9. refuses the result if the pulled image's flags differ; and
10. atomically writes `HANDIN.json`.

For a local rehearsal without publishing, omit the image argument. The `digest` fields will correctly be `null`:

```sh
tools/handin.sh
```

## 7:00–8:00 — inspect the evidence

```sh
../v16/bin/python -m json.tool HANDIN.json
```

Check all of the following:

- `git_sha` equals the frozen commit;
- `urdf_sha256` is `8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735`;
- `digest_reference` is the requested repository plus `@sha256:…`, never a local image ID;
- `size_bytes` and `layer_count` are non-zero;
- local and pulled `verification` entries both have `pass: true`;
- `first_action_policy_hz` is at least 27;
- the flags dump shows the expected submission overlay and values.

Copy the final `registry digest:` line into the submission form. Submit by digest where the form permits it.

## 8:00–9:00 — independent pull command

Use the `digest_reference` printed in `HANDIN.json`, not the mutable tag:

```sh
docker pull --platform linux/amd64 <registry>/<namespace>/<image>@sha256:<digest>
```

The hand-in script has already smoke-tested that digest-pulled image. This extra command is a quick operator check
that the reference was copied correctly.

## 9:00–10:00 — hand over the immutable record

- Upload or archive `HANDIN.json` with the submission notes.
- Record the tag, digest reference, Git SHA, size, layer count, cold start, and first-action latency.
- Keep the pulled image and its fresh local tag until the submission is accepted.
- Do not rebuild under the same tag after recording the digest. A new build gets a new tag and a new `HANDIN.json`.

## Optional multi-architecture manifest

This takes longer and always requires a registry destination because a multi-platform build cannot be loaded as one
local image:

```sh
tools/handin.sh --multi-arch <registry>/<namespace>/<image>:<tag>
```

It publishes a `linux/amd64,linux/arm64` manifest, then clean-pulls and smokes the selected `--platform` (amd64 by
default) by the registry's manifest digest.

## Chunk-size recommendation

The slow-sim rate model is `policy_hz_rt = ticks_per_reply × rtf × 30`; the gate is 27 Hz. The new flags only change
reply sizes and add no waiting or synchronous work to `act()`.

| Slow-sim RTF | Current chunk 10 | Current hold 6 | Recommended chunk 15 | Recommended hold 10 |
|---:|---:|---:|---:|---:|
| 0.15 | 45 Hz (+18) | 27 Hz (+0) | 67.5 Hz (+40.5) | 45 Hz (+18) |
| 0.30 | 90 Hz (+63) | 54 Hz (+27) | 135 Hz (+108) | 90 Hz (+63) |

Use `finals_r22_chunk` (`policy.chunk_ticks: 15`, `policy.hold_ticks: 10`) for the slow simulator. The trade-off is at
most five extra trajectory ticks (0.167 simulated seconds) before the next policy request and four extra hold ticks
(0.133 simulated seconds). With the flags off, the existing 10/6 behavior is unchanged.

## Fast failure guide

- Dirty tree: stop; identify every listed file. Do not bypass the check.
- URDF failure: put the real file at the exact required path. Do not use an out-of-context symlink.
- Ready timeout or first-action rate below 27 Hz: keep the image unpublished and inspect the container log printed by
  the script.
- Push authentication failure: fix `docker login`; never paste credentials into the command or repository.
- Clean-pull removal failure: another local tag points at the built image. Remove that explicit tag, then rerun from a
  clean tree.
- Digest or pulled-smoke failure: do not submit the tag; a tag without verified digest evidence is not a hand-in.
