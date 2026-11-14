"""Head-camera frame under an adv profile: set_profile -> randomize(seed) -> home pose -> reset -> frame (same path as
/data/work/oliver/ab/dump_home.py). Usage (adv instance env sourced, sim idle):
    python dump_profile_frame.py <out_dir> <profile|nominal> [seed=1000]
-> <out_dir>/<profile>.png (head RGB) + <profile>_depth.png."""
import os, sys, numpy as np, cv2
from aaltonomous.interface.env import AaltoEnv
from aaltonomous.sim.scene_client import SceneClient
from aaltonomous.sim.evaluate import home_arms, apply_instance_env

out, prof = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
apply_instance_env()
sc = SceneClient("127.0.0.1")
sc.call(cmd="set_profile", name="adv_nominal" if prof == "nominal" else f"adv_{prof}")
env = AaltoEnv(backend="sim", task="sort_fixed", seed=seed, time_limit=120, realtime=True, img_server_ip="127.0.0.1",
               head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")), cameras=True)
home_arms(env, sc)
params = sc.call(cmd="randomize", seed=seed).get("params")
env.rng = np.random.default_rng(seed)
obs = env.reset()
for _ in range(15):
    obs, info = env.step({"arm_q_target": env.filter.arm_cmd.copy(), "hand_q_target": env.filter.hand_cmd.copy()})
rgb, d = obs["head_rgb"], obs["head_depth"]
os.makedirs(out, exist_ok=True)
cv2.imwrite(f"{out}/{prof}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
dv = d.astype(np.float32); dv = dv / max(float(np.nanmax(dv)), 1e-6) * 255
cv2.imwrite(f"{out}/{prof}_depth.png", np.nan_to_num(dv).astype(np.uint8))
print("profile", prof, "adv params:", None if not params else {k: params["adv"][k] for k in ("cam", "bin_dx_m")} if params.get("adv") else None,
      "applied", params.get("applied") if params else None, "depth_zero_frac", float((d == 0).mean()))
sys.stdout.flush(); os._exit(0)
