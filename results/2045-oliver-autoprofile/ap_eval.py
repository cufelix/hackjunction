# drop_2 ep<40 only. Head depth is absent in drop_2, so run (A) depth=None (what the tool can do) and
# (B) a worst-case adversarial probe: real RGB + a perfect synthetic sim-like depth (flat, no holes), to test
# whether the palette veto alone keeps real frames at 'none'.
import sys, os, collections, numpy as np, cv2
sys.path.insert(0, "/home/oliver/work/autoprofile")
from aaltonomous.baselines.classical.profile_detect import detect_profile
F = "/var/tmp/hf_drop2/frames"
eps = collections.defaultdict(list)
for fn in sorted(os.listdir(F)):
    if not fn.endswith(".png"): continue
    ep = int(fn[1:3])
    if ep >= 40: continue
    eps[ep].append(os.path.join(F, fn))
def flat_depth(h, w):
    v = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    return (600 + 0.8 * v).astype(np.float32)   # smooth tilted plane in mm, no holes, no noise
res = {"A": collections.Counter(), "B": collections.Counter()}
pal = collections.Counter(); frames = 0; missB = []
for ep, paths in sorted(eps.items()):
    rgbs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    frames += len(rgbs)
    win = (rgbs * 5)[:5]                      # 4 key frames per episode -> window of 5 (repeat first)
    pa, ia = detect_profile([{"head_rgb": r, "head_depth": None} for r in win]); res["A"][pa] += 1
    pb, ib = detect_profile([{"head_rgb": r, "head_depth": flat_depth(*r.shape[:2])} for r in win]); res["B"][pb] += 1
    for p in ib["per_frame"][:len(rgbs)]: pal[p["palette"]] += 1
    if pb != "none": missB.append((ep, ib["votes"], [p["palette"] for p in ib["per_frame"]]))
print("episodes", len(eps), "frames", frames)
print("A depth=None (drop_2 as is):", dict(res["A"]))
print("B real RGB + sim-like flat depth (palette veto only):", dict(res["B"]))
print("B per-frame palette:", dict(pal))
print("B misses:", missB[:12])
r = cv2.cvtColor(cv2.imread(eps[0][0]), cv2.COLOR_BGR2RGB)
h, w = r.shape[:2]
# exact plane in 1/z (what _plane models): 1/z linear in (u,v)
v = np.arange(h, dtype=np.float64)[:, None].repeat(w, 1); u = np.arange(w, dtype=np.float64)[None, :].repeat(h, 0)
z = (1.0 / (1/600 + 1e-6 * v + 1e-7 * u)).astype(np.float32)
p, i = detect_profile([{"head_rgb": r, "head_depth": z}] * 5)
print("exact-plane probe:", p, {k: i["per_frame"][0][k] for k in ("holes", "plane_std_mm", "depth_sim", "palette", "sim_px", "real_px")})
p, i = detect_profile([{"head_rgb": r, "head_depth": flat_depth(h, w)}] * 5)
print("linear probe:", p, {k: i["per_frame"][0][k] for k in ("holes", "plane_std_mm", "depth_sim", "palette", "sim_px", "real_px")})
