# Wendy packaging for the policy server

Manifests for the classical policy image built by `tools/submission.Dockerfile`. The image `CMD` is `python3 -m aaltonomous.baselines.classical.policy_server`, which binds `0.0.0.0:8000` (websocket, msgpack). This folder does not modify `aaltonomous/deploy`, `aaltonomous/interface`, or the Dockerfile.

Nothing here has been deployed. Commands are from Wendy docs fetched 2026-11-14 (`docs.wendy.dev`). Anything not run on the venue hardware is marked **[?]**.

| File | Use |
|---|---|
| `wendy.json` | RTX 5090 PC, `linux/amd64`, app id `vaiski-policy`. Readiness TCP 8000, 120 s. Network `host` + `http` port 8000. No GPU. |
| `wendy.gpu.json` | Optional brain sidecar only. App id `vaiski-brain`, TCP 8080, plus the `gpu` entitlement. No sidecar image in this folder. |
| `wendy.arm64.json` | Same policy manifest for the G1 onboard Jetson (`linux/arm64`). Needs the Dockerfile changes in `ARM64.md` first. |
| `smoke.sh` | Local build and cold-start measurement. Does not deploy. |
| `ARM64.md` | What a copy of the Dockerfile must change for arm64. |

The policy process does not open cameras or DDS. Observations arrive on the websocket. `network` mode `host` is set because bridge mode does not publish a port onto the device LAN, and the runner has to dial `ws://<device>:8000`. Host mode also removes container network isolation. The `http` entitlement advertises port 8000 to the agent; the protocol on that port is websocket, not a browser page.

`wendy.gpu.json` matches the brain URL baked in config (`http://127.0.0.1:8080/completion`). The submission overlay ships with `brain.enabled: false` and `brain.backend: stub`, so the policy image does not call that port. Do not point `wendy run` of the policy image at `wendy.gpu.json`.

## Deploy over LAN

Install the CLI on the laptop only when someone asks. Docs currently use `https://install.wendy.dev/cli.sh`. Download the script, read it, then run it. Confirm `wendy --version`. The same for `wendy-agent` on the target (`https://install.wendy.dev/agent.sh`): read it before running, refuse it if apt wants to remove Docker or replace `containerd.io`, then check `sudo systemctl is-active wendy-agent`. [?]

Wendy reads `wendy.json` from the project directory and accepts `--dockerfile` as a **bare filename in that directory**, not a path. The submission Dockerfile lives at `tools/submission.Dockerfile` and its `COPY` lines use the repo root as the build context. From a checkout that is not under `/tmp` or `/private/tmp` (Apple Container reports an empty context there; a Docker builder on macOS may also refuse this scratchpad path) [?] :

```bash
cd <repo-root>
ln -sfn tools/wendy/wendy.json wendy.json
ln -sfn tools/submission.Dockerfile Dockerfile

wendy discover
# one-shot LAN scan, prints addresses and exits:
wendy discover --type lan --timeout 5s

wendy run --device <address-from-discover> \
  --build-type docker \
  --dockerfile Dockerfile \
  --skip-cloud-registration \
  --chunking force \
  --wait-ready
```

- `<address-from-discover>` is the hostname or `host:port` the scan prints. Do not assume the agent port is 50051. [?]
- `--build-type docker` forces the Dockerfile when the repo also contains Python project files.
- `--skip-cloud-registration` is the documented offline deploy flag.
- `--chunking force` uses content-defined chunk transfer and does not fall back to a registry push. Whether that reaches the venue machine with no internet is unverified. [?]
- `--wait-ready` blocks until the TCP 8000 probe in `wendy.json` succeeds or the 120 s timeout fires.
- For the Jetson, symlink `tools/wendy/wendy.arm64.json` instead of `wendy.json`, and only after the arm64 Dockerfile copy exists. The 5090 and the Jetson are different devices; use one manifest per device. Both use app id `vaiski-policy`.

Logs stream in the attached `wendy run` terminal. A separate `wendy logs` command was not in the run reference used here. [?]

Stop:

```bash
wendy --device <address-from-discover> device apps stop vaiski-policy
```

Leave the two symlinks uncommitted unless the team decides the repo root should carry them.

Do not pass `--env` unless the organisers ask. The server defaults `AALTO_TIME_LIMIT` to 120 and `AALTO_PROFILE` to `sim` (sim-tuned). `AALTO_PROFILE=none` is the robot base profile. It is not set in these manifests. [?]

## What to verify on the day

1. `bash tools/wendy/smoke.sh` on a machine with Docker. It prints `cold-start: <seconds>s`. The budget is 120 s. This has not been run. [?]
2. `wendy discover --type lan --timeout 5s` shows the machine the organisers named. Write down the address and port.
3. `wendy run ... --wait-ready` reaches readiness and the log line is the classical server listening on 8000, with the baked overlay (brain off).
4. From the runner host: `python -m aaltonomous.deploy.replay --synthetic 100 --policy ws://<device>:8000`. Exit 0. [?]
5. `docker run --rm --entrypoint python3 <image> tools/flags.py` matches the overlay that was supposed to ship.
6. Platform of the running image is `linux/amd64` on the 5090. Do not load that image on PC2.
7. Only one app is controlling motion. This policy server does not talk to the robot; the runner does. No motion test without the physical E-stop in a person's hand and the robot on the gantry.

## Fallback

If discovery is empty, ask for the address and try `wendy device info --device <ip>`, then the same `wendy run --device <ip>`. Factory G1 PC2 Ethernet in the Wendy G1 guide is `192.168.123.164` (user `unitree`). Confirm before using it. [?]

If chunk-diff or the agent cannot take the image, build and run with Docker on the target. Same command the Dockerfile header and the submission spec use:

```bash
docker build -f tools/submission.Dockerfile -t hackjunction/policy:venue .
docker run -d --name vaiski-policy --gpus all -p 8000:8000 hackjunction/policy:venue
```

The image runs without a GPU. On a host with no NVIDIA container runtime, drop `--gpus all`. To move an image with no registry: `docker save hackjunction/policy:venue | ssh <target> docker load`. [?]

Stop the fallback container with `docker stop vaiski-policy && docker rm vaiski-policy`.

## Open questions

1. Which box runs the agent — the RTX 5090 (`wendy.json`) or G1 PC2 (`wendy.arm64.json` plus a new Dockerfile)? Ubuntu version, Docker version, and sudo on that box? [?]
2. Does `--chunking force` deliver the image across the venue switch with no registry and no cloud? Is there a size cap (submission spec says ≤ 40 GB uncompressed)? [?]
3. Does `wendy discover` work on their switch (mDNS), or do we get a raw IP? What port is the agent on? [?]
4. Is host networking accepted for a websocket-only policy, and does the `http` entitlement make `wendy run` open a browser onto the msgpack port? [?]
5. On a normal Ubuntu + NVIDIA driver install (not WendyOS), does the agent honour the `gpu` entitlement the way WendyOS does? Needed only for `vaiski-brain`. [?]
6. Who is holding the physical E-stop, and is the gantry the place the robot stays for any test that can move it? [?]
7. Should the on-robot process run `AALTO_PROFILE=none`? These manifests do not set it. [?]
8. Jetson module and JetPack on this PC2. The arm64 CUDA tag on Docker Hub is SBSA, so it is the wrong base even though the manifest is multi-arch. [?]
9. There is no brain-sidecar Dockerfile. `wendy.gpu.json` cannot be deployed until that image exists and listens on 8080. [?]
