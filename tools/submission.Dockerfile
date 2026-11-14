# Team submission: the classical baseline + the integration branch's opt-in features (timi/integration).
# Based on aaltonomous/docker/baseline.Dockerfile (36daf5b) with two changes:
#   1. interface/tasks.py is copied (the 36daf5b baseline.Dockerfile misses it: policy.py imports FIXED_MAPPING).
#   2. The feature flags come from a BAKED-IN overlay file: overlays/submission.yaml -> classical/overlay.yaml.
#      The organisers run `docker run -d --gpus all -p <port>:8000 <image>` with no env, args or volumes, so the
#      overlay file is the only switch. As committed every flag is OFF = stock baseline behaviour.
#
#   docker build -f tools/submission.Dockerfile -t hackjunction/policy:<tag> .        # from the repo root
#   docker run --rm -p 8001:8000 hackjunction/policy:<tag>                            # (+ --gpus all on the GPU box)
#   python -m aaltonomous.deploy.replay --synthetic 100 --policy ws://localhost:8001
#   docker run --rm --entrypoint python3 hackjunction/policy:<tag> tools/flags.py     # flags baked into the image
#
# CPU only (HSV + depth perception, Pinocchio IK); runs the same with or without --gpus. Context: needs
# third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf (a real file, not a symlink out of the context).
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
# Same pins as baseline.Dockerfile (pin 3.3.1 needs numpy >= 2.2).
RUN pip3 install --no-cache-dir numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml \
        msgpack "websockets>=12" pin==3.3.1

WORKDIR /app
COPY third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous/__init__.py aaltonomous/
COPY aaltonomous/deploy/__init__.py aaltonomous/deploy/protocol.py aaltonomous/deploy/policy_server.py aaltonomous/deploy/
COPY aaltonomous/interface/__init__.py aaltonomous/interface/safety.py aaltonomous/interface/calibration.py \
     aaltonomous/interface/tasks.py \
     aaltonomous/interface/joint_spec.yaml aaltonomous/interface/calibration.yaml aaltonomous/interface/
COPY aaltonomous/scene/__init__.py aaltonomous/scene/scene.py aaltonomous/scene/scene.yaml aaltonomous/scene/
COPY aaltonomous/baselines/__init__.py aaltonomous/baselines/
COPY aaltonomous/baselines/classical/*.py aaltonomous/baselines/classical/config.yaml aaltonomous/baselines/classical/
COPY aaltonomous/baselines/classical/overlays/ aaltonomous/baselines/classical/overlays/
# THE flag switch: the submission overlay, baked in where load_config() looks for it.
COPY aaltonomous/baselines/classical/overlays/submission.yaml aaltonomous/baselines/classical/overlay.yaml
COPY tools/flags.py tools/
# Build-time checks (fail the build, not the run): config + overlay + URDF + calibration load; print the flag set.
RUN python3 -c "from aaltonomous.baselines.classical.policy_server import ClassicalServerPolicy; ClassicalServerPolicy()" \
 && python3 tools/flags.py \
 && python3 -c "import sys; from aaltonomous.baselines.classical.config import BAKED_OVERLAY, resolve_overlay; \
sys.exit(0 if resolve_overlay() == BAKED_OVERLAY else 'baked overlay not picked up')"

EXPOSE 8000
CMD ["python3", "-m", "aaltonomous.baselines.classical.policy_server"]
