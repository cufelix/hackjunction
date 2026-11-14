# arm64 image changes (G1 PC2 / Jetson)

`wendy.arm64.json` targets `linux/arm64`. `tools/submission.Dockerfile` is the x86 submission image and stays as it is. Apply the changes below to a **copy** of that file before anyone deploys `wendy.arm64.json`. This copy has not been built. [?]

The current file:

```dockerfile
FROM nvidia/cuda:12.8.0-base-ubuntu22.04
...
RUN pip3 install --no-cache-dir numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml \
        msgpack "websockets>=12" pin==3.3.1
...
CMD ["python3", "-m", "aaltonomous.baselines.classical.policy_server"]
```

`CMD` already starts `aaltonomous.baselines.classical.policy_server` on port 8000 (`PORT`, default 8000, bind `0.0.0.0`). Leave that line.

## 1. Base image: multi-arch, and not the Jetson CUDA image

`nvidia/cuda:12.8.0-base-ubuntu22.04` is a multi-arch manifest. Docker Hub lists `linux/amd64` and `linux/arm64` (checked 2026-11-14). The arm64 layers set `NVARCH=sbsa`. That is server (SBSA) arm64, not Jetson Tegra / L4T.

G1 PC2 runs Unitree's Ubuntu image, arm64, 20.04 or 22.04 (Wendy's G1 guide). WendyOS install guides do not apply, and the WendyOS GPU page's JetPack 7.2 / CUDA 13 line is for WendyOS, not this PC2 image. Which Jetson module and which JetPack the venue PC2 actually has is unverified. [?]

For the policy image, change the copy to a non-CUDA multi-arch base and build it explicitly for arm64:

```dockerfile
FROM ubuntu:22.04
```

```bash
docker buildx build --platform linux/arm64 -f <arm64-dockerfile> -t hackjunction/policy:arm64 .
```

`ubuntu:22.04` publishes `linux/amd64` and `linux/arm64`. Ubuntu 22.04's `python3` is 3.10, which matches the wheels below. Keep `apt-get install python3 python3-pip libgomp1`.

A cross-build from an x86 host needs Docker Buildx and qemu/binfmt. Enabling binfmt on a shared machine was left for an explicit go-ahead. Prefer a native build on the Jetson, or on an arm64 builder, until that is agreed. [?]

## 2. No CUDA in the policy

The classical server does not import torch or CUDA. The Dockerfile header states the image is CPU only (HSV + depth, Pinocchio IK) and runs the same with or without `--gpus`. `ClassicalServerPolicy` loads the URDF, calibration, and the baked overlay, then `serve()` listens on 8000.

Drop the CUDA base on the arm64 copy. Do not add `--gpus` for this image. Do not install the CUDA toolkit or a Jetson L4T CUDA base into the policy image.

GPU belongs on the optional brain sidecar (`wendy.gpu.json`, app id `vaiski-brain`, TCP 8080), which is a different app. The baked submission overlay ships `brain.enabled: false` and `brain.backend: stub`. There is no sidecar Dockerfile in this folder.

## 3. Wheel pins

Keep the pins. On 2026-11-14 PyPI had aarch64 wheels for every pinned distribution, for the CPython that Ubuntu 22.04 apt installs (3.10):

| Pin | aarch64 artifact |
|---|---|
| `numpy==2.2.6` | `manylinux_2_17_aarch64` for `cp310` |
| `scipy==1.15.2` | `manylinux_2_17_aarch64` for `cp310` |
| `opencv-python-headless==4.10.0.84` | `cp37-abi3` `manylinux_2_17_aarch64` (any CPython ≥ 3.7) |
| `pin==3.3.1` | `manylinux_2_28_aarch64` for `cp310` (also cp38–cp313) |
| `pyyaml`, `msgpack`, `websockets>=12` | pure Python / ordinary wheels |

`manylinux_2_28` needs glibc ≥ 2.28. Ubuntu 22.04 (glibc 2.35) and Ubuntu 20.04 (glibc 2.31) both clear that. If the base image's `python3` is not CPython 3.8–3.13, `pin==3.3.1` will not resolve; fix the base Python, not the pin.

The arm64 image has not been built, so an `import cv2` or `import pinocchio` failure from a missing shared library is still possible. The existing build check (`ClassicalServerPolicy()` before the image is tagged) is the place that fails. Add the apt package the dynamic linker names. Do not loosen the pins to get past it. [?]
