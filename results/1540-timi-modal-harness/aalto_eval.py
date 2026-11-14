"""Aalto humanoid Isaac Sim evaluation on Modal (L40S), one sim + one eval per container.

Image = organisers' Dockerfile recipe (aaltonomous/cloud/setup_instance.sh, one step per layer) at repo 36daf5b
plus the junction install fixes in modal/recipe/ (boxA.diff, pip overrides) and the Vulkan libs Modal needs.

  PY=<HJ_ROOT>/.venv-modal/bin/python
  $PY -m modal run modal/aalto_eval.py --mode check                          # sim + `aalto check` + snap
  $PY -m modal run modal/aalto_eval.py --seeds 1000 --trials 2               # one oracle sort_fixed eval
  $PY -m modal run modal/aalto_eval.py --seeds 2000,3000,4000 --trials 2     # 3 containers in parallel
  # own policy server from a local worktree (mounted at /work, started inside the same container):
  AALTO_WORKTREE=/path/to/worktree $PY -m modal run modal/aalto_eval.py --seeds 1000 --trials 5 \
      --policy-cmd "python -m aaltonomous.baselines.classical.policy_server"

Results: modal/runs/<run_id>/ locally (and the Modal Volume `aalto-runs`). Shader/kit cache: Volume `aalto-kitcache`
(written only when it is empty or with --save-cache; parallel containers only read it).
"""
from __future__ import annotations

import io
import json
import os
import re
import shlex
import socket
import subprocess
import tarfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/zupermiez/aaltonomous-humanoid"
REPO_REV = "36daf5b"
AALTO = "/opt/aalto"
CONDA = "/root/miniforge3"
KIT = f"{CONDA}/envs/isaac/lib/python3.12/site-packages/isaacsim/kit"
CACHE_PATHS = [f"{KIT}/cache", f"{KIT}/data", "/root/.nv", "/root/.cache/warp", "/root/.cache/ov"]
CACHE_TAR = "/cache/kitcache-{driver}.tar"  # per NVIDIA driver: Modal hosts differ (610.57 / 620.70), shaders do not carry over

APT = [  # aaltonomous/cloud/Dockerfile + Modal Vulkan probe (modal/RESULT.md)
    "ca-certificates", "curl", "git", "unzip", "cmake", "build-essential", "iproute2", "procps", "less",
    "libturbojpeg", "libglu1-mesa", "libxt6", "libvulkan1", "libegl1", "libgl1", "libxrandr2", "libxi6", "libsm6",
    "libice6", "libxkbcommon0", "libglib2.0-0t64", "libgomp1",
    "libxext6", "libx11-6", "libxcb1", "libglvnd0", "libxcursor1", "libxinerama1", "vulkan-tools",
]
ENV = {
    "DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC", "HOME": "/root",
    "NVIDIA_VISIBLE_DEVICES": "all", "NVIDIA_DRIVER_CAPABILITIES": "all",
    "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",  # Mesa llvmpipe is also present
    "OMNI_KIT_ACCEPT_EULA": "YES", "OMNI_KIT_ALLOW_ROOT": "1", "PYTHONUNBUFFERED": "1",
    "CONDA_ROOT": CONDA, "AALTO_NO_GPU": "1",
}
SETUP = f"cd {AALTO} && bash aaltonomous/cloud/setup_instance.sh"

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .env(ENV)
    .apt_install(*APT)
    .run_commands(f"git clone -q {REPO_URL} {AALTO} && git -C {AALTO} -c advice.detachedHead=false checkout -q {REPO_REV}")
    .add_local_dir(HERE / "recipe", f"{AALTO}/recipe", copy=True)
    .run_commands(
        f"cd {AALTO} && git apply recipe/boxA.diff && "
        f"sed -i 's#/data/#{AALTO}/recipe/#g' aaltonomous/cloud/setup_instance.sh && "
        "grep -c recipe/ aaltonomous/cloud/setup_instance.sh"
    )
    .run_commands(f"{SETUP} prereq && {SETUP} conda")
    .run_commands(f"{SETUP} torch")
    .run_commands(f"{SETUP} isaacsim kit")
    .run_commands(f"{SETUP} isaacsim rest")
    .run_commands(f"{SETUP} isaaclab")
    .run_commands(f"{SETUP} cyclonedds && {SETUP} repos")
    .run_commands(f"{SETUP} assets")
    # verify: test_scene.TestHeadCameraView.test_cubes_and_bins_visible fails upstream at 36daf5b (also on box A),
    # so only the imports are gating; the test summary stays in the build log.
    .run_commands(f"{SETUP} aalto && ({SETUP} verify 2>&1 | tee /tmp/verify.log; grep -q 'imports OK' /tmp/verify.log) "
                  f"&& ln -sf {AALTO}/aaltonomous/cloud/aalto.sh /usr/local/bin/aalto")
    .env({"AALTO_NO_GPU": ""})
)

WORKTREE = os.environ.get("AALTO_WORKTREE")
if WORKTREE:  # mounted at container start, no image rebuild
    image = image.add_local_dir(WORKTREE, "/work", ignore=[".git", "**/runs", "**/third_party", "**/__pycache__"])

app = modal.App("aalto-eval", image=image)
cache_vol = modal.Volume.from_name("aalto-kitcache", create_if_missing=True)
runs_vol = modal.Volume.from_name("aalto-runs", create_if_missing=True)


def _sh(cmd: str, log: Path | None = None, timeout: float | None = None, check: bool = False) -> tuple[int, str]:
    t0 = time.time()
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    line = f"\n$ {cmd}\n[rc={r.returncode} {time.time() - t0:.1f}s]\n{out}"
    print(line[-4000:], flush=True)
    if log:
        with open(log, "a") as f:
            f.write(line)
    if check and r.returncode:
        raise RuntimeError(f"failed rc={r.returncode}: {cmd}\n{out[-2000:]}")
    return r.returncode, out


def _aalto(args: str, log: Path | None = None, timeout: float | None = None, check=False):
    return _sh(f"bash {AALTO}/aaltonomous/cloud/aalto.sh {args}", log, timeout, check)


def _ping(log: Path):
    code = ("import json; from aaltonomous.sim.scene_client import SceneClient as S; c=S(timeout_ms=3000); "
            "print('PING', json.dumps(c.info(), default=str))")
    rc, out = _aalto(f"py -c {shlex.quote(code)}", log, 60)
    m = re.search(r"^PING (.*)$", out, re.M)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return {"raw": m.group(1)}


def _kit_log() -> Path | None:
    logs = sorted(Path(f"{KIT}/logs").rglob("kit_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _render_ok(log: Path) -> tuple[bool, str]:
    rc, out = _aalto(f"py {AALTO}/recipe/probe_depth.py", log, 90)
    m = re.findall(r"rgb .* mean ([\d.]+) \| depth .* nonzero ([\d.]+)%", out)
    if not m:
        return False, out[-300:]
    rgb, nz = float(m[-1][0]), float(m[-1][1])
    return (rgb > 1.0 and nz > 50.0), f"rgb mean {rgb}, depth nonzero {nz}%"


def _wedged() -> bool:
    k = _kit_log()
    return bool(k) and "cmdBegin before calling cmdEnd" in k.read_text(errors="ignore")


def _start_sim(rd: Path, t: dict, sim_env: str) -> None:
    """Start the sim, wait for the scene service AND a valid head RGB-D frame; restart once if RTX wedges."""
    for attempt in range(2):
        t0 = time.time()
        _aalto("sim", rd / "steps.log", 700)  # returns after <=10 min; the sim keeps running (nohup)
        while time.time() - t0 < 1500 and _ping(rd / "steps.log") is None:
            time.sleep(10)
        t[f"sim_ready_s_{attempt}"] = round(time.time() - t0, 1)
        while time.time() - t0 < 1800:
            ok, msg = _render_ok(rd / "steps.log")
            t["render"] = msg
            if ok:
                t[f"render_ok_s_{attempt}"] = round(time.time() - t0, 1)
                return
            if _wedged():
                t[f"wedged_{attempt}"] = True
                break
            time.sleep(20)
        _aalto("stop", rd / "steps.log", 120)
        time.sleep(5)
    raise RuntimeError(f"sim never rendered: {t}")


def _wait_port(port: int, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), 1).close()
            return True
        except OSError:
            time.sleep(2)
    return False


@app.function(gpu="L40S", cpu=8.0, memory=32768, timeout=3 * 3600, max_containers=10,  # workspace GPU limit = 10
              single_use_containers=True,  # 1 input = 1 fresh container (no leftover policy server / sim)
              volumes={"/cache": cache_vol, "/runs": runs_vol})
def run_eval(policy: str = "oracle", task: str = "sort_fixed", trials: int = 2, seed: int = 1000,
             settle: float = 3.0, extra_args: str = "", mode: str = "eval", policy_cmd: str = "",
             sim_env: str = "", run_id: str = "", save_cache: bool = False) -> dict:
    """mode: eval | check (aalto check + snap, no eval). policy_cmd: started in a copy of /work (else /opt/aalto),
    must listen on :8000; then policy defaults to ws://localhost:8000. sim_env: e.g. 'AALTO_DR=train TASK=...'."""
    t: dict = {"entry_unix": time.time()}
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S") + f"_{mode}_{task}_s{seed}"
    rd = Path("/tmp/run") / run_id
    rd.mkdir(parents=True)
    for k, v in (kv.split("=", 1) for kv in shlex.split(sim_env)):
        os.environ[k] = v
    os.environ["AALTO_LOG"] = str(rd / "sim.log")

    egl = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if not egl.exists():
        egl.parent.mkdir(parents=True, exist_ok=True)
        egl.write_text('{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}')
    _sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; nproc; free -g | head -2; "
        "vulkaninfo --summary 2>&1 | grep -E 'deviceName|driverName' ", rd / "steps.log")

    t0 = time.time()
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip().splitlines()[0]
    cache_tar = CACHE_TAR.format(driver=driver)
    t["driver"] = driver
    had_cache = Path(cache_tar).exists()
    if not had_cache:  # fall back to the cache of another driver (kit data, DerivedDataCache still help)
        others = sorted(Path("/cache").glob("kitcache-*.tar"), key=lambda p: p.stat().st_mtime)
        if others:
            _sh(f"tar -xf {others[-1]} -C /", rd / "steps.log", 900)
            t["cache_from"] = others[-1].name
    else:
        _sh(f"tar -xf {cache_tar} -C / && du -sh {KIT}/cache", rd / "steps.log", 900)
    t["cache_restore_s"] = round(time.time() - t0, 1)
    t["had_cache"] = had_cache

    summary: dict = {}
    pol_proc = None
    try:
        _start_sim(rd, t, sim_env)
        t["sim_up_since_entry_s"] = round(time.time() - t["entry_unix"], 1)
        json.dump(_ping(rd / "steps.log"), open(rd / "sim_ping_before.json", "w"), indent=1, default=str)
        if mode == "check":
            t0 = time.time()
            rc, out = _aalto("check", rd / "check.log", 900)
            t["check_rc"], t["check_s"] = rc, round(time.time() - t0, 1)
            summary["check"] = [l for l in out.splitlines() if l.startswith(("PASS", "FAIL"))]
            _aalto(f"snap --out {rd}/snap", rd / "steps.log", 120)
        else:
            if policy_cmd:
                wd = AALTO
                if Path("/work").exists():  # writable copy + the image's third_party (URDF for Calibration)
                    wd = "/tmp/work"
                    _sh(f"rm -rf {wd} && cp -a /work {wd} && ([ -e {wd}/third_party ] || ln -s {AALTO}/third_party {wd}/third_party)",
                        rd / "steps.log", check=True)
                cmd = (f"source {CONDA}/etc/profile.d/conda.sh && conda activate isaac && cd {wd} && "
                       f"export PYTHONPATH={wd}:{AALTO} && exec {policy_cmd}")
                if _wait_port(8000, 1):
                    raise RuntimeError(":8000 already in use before our policy server started")
                pol_proc = subprocess.Popen(["bash", "-c", cmd], stdout=open(rd / "policy.log", "w"),
                                            stderr=subprocess.STDOUT, start_new_session=True)
                if not _wait_port(8000, 300):
                    raise RuntimeError("policy server did not open :8000 in 300 s, see policy.log")
                if policy == "oracle":
                    policy = "ws://localhost:8000"
            t0 = time.time()
            rc, out = _aalto(f"eval --policy {shlex.quote(policy)} --task {task} --trials {trials} --seed {seed} "
                             f"--settle {settle} --out {rd}/eval {extra_args}", rd / "eval.log", 3 * 3600 - 1800)
            t["eval_rc"], t["eval_s"] = rc, round(time.time() - t0, 1)
            summary["eval_tail"] = out.splitlines()[-25:]
            for ej in rd.rglob("eval.json"):
                summary["eval_summary"] = json.load(open(ej)).get("summary")
        json.dump(_ping(rd / "steps.log"), open(rd / "sim_ping_after.json", "w"), indent=1, default=str)
    except Exception as e:  # keep logs on failure
        t["error"] = repr(e)[:2000]
    finally:
        if pol_proc is not None:  # stop the policy server (whole process group)
            try:
                os.killpg(pol_proc.pid, 15)
                pol_proc.wait(20)
            except Exception:
                pass
        _aalto("stop", rd / "steps.log", 120)
        k = _kit_log()
        if k:
            _sh(f"cp {k} {rd}/kit.log")

    rendered = any(k.startswith("render_ok_s_") for k in t)
    if rendered and (not had_cache or save_cache):
        t0 = time.time()
        paths = " ".join(p for p in CACHE_PATHS if Path(p).exists())
        _sh(f"tar -cf {cache_tar}.tmp {paths} && mv {cache_tar}.tmp {cache_tar} && ls -la /cache", rd / "steps.log")
        cache_vol.commit()
        t["cache_save_s"] = round(time.time() - t0, 1)

    t["total_in_container_s"] = round(time.time() - t["entry_unix"], 1)
    summary["timing"] = t
    json.dump(summary, open(rd / "modal_summary.json", "w"), indent=1, default=str)
    _sh(f"mkdir -p /runs && cp -r {rd} /runs/")
    runs_vol.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(rd, arcname=run_id)
    summary["run_id"], summary["tar"] = run_id, buf.getvalue()
    return summary


@app.function(cpu=4.0, memory=8192, timeout=1800)
def run_tests(cmd: str) -> str:
    """CPU only: run `cmd` (e.g. python -m unittest ...) in a writable copy of the mounted worktree (/work, else
    /opt/aalto) with the conda env and the image's third_party. Returns the combined output."""
    wd = AALTO
    if Path("/work").exists():
        wd = "/tmp/work"
        _sh(f"rm -rf {wd} && cp -a /work {wd} && ([ -e {wd}/third_party ] || ln -s {AALTO}/third_party {wd}/third_party)",
            check=True)
    rc, out = _sh(f"source {CONDA}/etc/profile.d/conda.sh && conda activate isaac && cd {wd} && "
                  f"export PYTHONPATH={wd} && {cmd}", timeout=1700)
    return f"[rc={rc}]\n{out}"


def _save(res: dict, t_call: float, tag: str = "") -> None:
    out = HERE / "runs" / tag if tag else HERE / "runs"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(res.pop("tar")), mode="r:gz") as tf:
        tf.extractall(out, filter="data")
    res["timing"]["cold_start_s"] = round(res["timing"]["entry_unix"] - t_call, 1)
    print(json.dumps({k: v for k, v in res.items() if k != "eval_tail"}, indent=1, default=str))
    print("\n".join(res.get("eval_tail", [])))
    print(f"-> {out / res['run_id']}")


@app.local_entrypoint()
def main(mode: str = "eval", seeds: str = "1000", trials: int = 2, task: str = "sort_fixed", policy: str = "oracle",
         settle: float = 3.0, extra_args: str = "", policy_cmd: str = "", sim_env: str = "", save_cache: bool = False,
         gpu: str = "", cpu: float = 0.0, tag: str = ""):
    """seeds: '1000,1010' or a range '1000-1019' (one container per seed). tag: run name prefix + runs/<tag>/."""
    if mode == "test":   # CPU unit tests of the mounted worktree; --policy-cmd carries the test command
        print(run_tests.remote(policy_cmd or "python -m unittest discover -s aaltonomous/baselines/classical -t ."))
        return
    opts = {k: v for k, v in (("gpu", gpu), ("cpu", cpu)) if v}
    fn = run_eval.with_options(**opts) if opts else run_eval
    t_call = time.time()
    seed_list = []
    for part in seeds.split(","):
        a, _, b = part.partition("-")
        seed_list += list(range(int(a), int(b) + 1)) if b else [int(a)]
    calls = [(s, fn.spawn(policy=policy, task=task, trials=trials, seed=s, settle=settle, extra_args=extra_args,
                          mode=mode, policy_cmd=policy_cmd, sim_env=sim_env, save_cache=save_cache,
                          run_id=f"{tag}_s{s}" if tag else ""))
             for s in seed_list]
    failed = []
    for s, c in calls:
        try:
            _save(c.get(), t_call, tag)
        except Exception as e:  # one bad container must not lose the others
            failed.append(s)
            print(f"seed {s}: FAILED {e!r}"[:2000])
    print(f"total wall {time.time() - t_call:.0f} s for {len(calls)} container(s); failed seeds: {failed}")
