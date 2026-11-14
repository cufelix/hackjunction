# Finals brain sidecar

This packages Qwen3-VL-8B-Instruct Q8_0 behind llama.cpp's HTTP server. The
policy remains the existing policy server; only the `brain_llm` overlay and its
sidecar URL are selected. CUDA **12.8 or newer is required** for the RTX 5090's
Blackwell `sm_120` target.

## Build context

From the repository root, stage already-downloaded artifacts as follows:

```text
tools/brain/
├── llama.cpp/                         # pinned, tested llama.cpp source tree
└── models/
    ├── Qwen3VL-8B-Instruct-Q8_0.gguf
    └── mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf
```

The Dockerfile never downloads llama.cpp or model files. Record the llama.cpp
commit and both GGUF SHA-256 hashes with the delivered images. A recent
llama.cpp revision with Qwen3-VL support is required.

Build and start both services:

```bash
docker compose -f tools/brain/compose.finals.yaml build
docker compose -f tools/brain/compose.finals.yaml up -d
docker compose -f tools/brain/compose.finals.yaml ps
python3 -m aaltonomous.deploy.replay --synthetic 1 --policy ws://localhost:8000
```

The policy is available on port 8000. The brain is internal on port 8080.
Check it through Compose:

```bash
docker compose -f tools/brain/compose.finals.yaml exec brain \
  curl --fail http://127.0.0.1:8080/health
docker compose -f tools/brain/compose.finals.yaml logs --tail=100 policy brain
```

To prove the fail-open deployment behavior, run the policy alone (there is no
policy dependency on `brain`):

```bash
docker compose -f tools/brain/compose.finals.yaml stop brain
docker compose -f tools/brain/compose.finals.yaml up -d --no-deps policy
```

Backend connection failures and timeouts happen on a bounded background
worker, so they never block policy startup or the control loop. A failed brain
request is treated as no brain override and the classical policy continues.
Where the resolver/bin lookup itself abstains, the object remains on the
counter (`skip`, zero points rather than a wrong-bin penalty); known classical
mappings still run unchanged.

## Capacity estimates

- Brain image: approximately 11–14 GB unpacked (roughly 9–10 GB of GGUF files,
  plus CUDA runtime and llama-server). Confirm with `docker image inspect`.
- Policy image: approximately 0.8–1.5 GB. A raw two-image `docker save` bundle
  will usually be about 11–14 GB; leave at least 25 GB free for load/unpack.
- GPU memory: budget 12–15 GiB at `-c 8192` (Q8_0 weights, mmproj, KV cache,
  CUDA workspace). This fits comfortably on a 32 GiB RTX 5090.

These are planning estimates; exact GGUF conversion and llama.cpp revision
change the numbers. Measure on box E with `docker images` and `nvidia-smi`.

## Fully offline handoff

On the build machine:

```bash
docker compose -f tools/brain/compose.finals.yaml build
docker save -o finals-images.tar \
  hackjunction/policy:finals hackjunction/brain:qwen3vl8b-q8
sha256sum finals-images.tar > finals-images.tar.sha256
```

Copy `finals-images.tar`, its checksum, this compose file, and the recorded
llama.cpp/GGUF hashes to the finals machine. Then, without network access:

```bash
sha256sum -c finals-images.tar.sha256
docker load -i finals-images.tar
docker compose -f tools/brain/compose.finals.yaml up -d --no-build --pull never
docker compose -f tools/brain/compose.finals.yaml ps
```

Preflight the finals host with `nvidia-smi` and
`docker run --rm --gpus all hackjunction/brain:qwen3vl8b-q8 --version` before
the demo. The image's normal entrypoint uses `-ngl 99`, both GGUF files,
`0.0.0.0:8080`, and an 8192-token context.

## Risks and fallback

- **No GPU access:** the brain container will fail or be unusably slow. Start
  only `policy`; its classical path remains operational. For an intentional CPU
  experiment, remove `-ngl 99` (or use `-ngl 0`). A typical desktop CPU may
  produce only about 5–20 tokens/s: a short decision is roughly 2–10 s after
  prompt ingestion, beyond the policy's 1.5–2 s request budget; multimodal
  requests can take tens of seconds.
- **Only one container allowed:** fold the brain builder/runtime artifacts and
  `/models` into the policy image, use the `brain_llm` overlay with its default
  loopback URL, and add a tiny init script that backgrounds `llama-server` then
  `exec`s the current policy command. Do not wait for brain health before
  starting policy. Expect the combined image to be about 12–15 GB.
- **CUDA/driver mismatch:** RTX 5090 needs CUDA >=12.8 and a host NVIDIA driver
  new enough for CUDA 12.8. Failure appears as no CUDA device, an `sm_120`
  loading error, or an unhealthy sidecar; policy still falls back.
- **Unsupported llama.cpp snapshot or mismatched mmproj:** model load fails or
  multimodal tokens are wrong. Freeze and test the exact source commit and
  GGUF pair on box E before `docker save`.
