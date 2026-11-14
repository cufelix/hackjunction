# Near-wall x of the bins from drop_2 ep<40 tape labels (bottom edge = bin base on the table), monocular:
# ray from the nominal head camera, optionally pitched, intersected with a table plane z. No depth in drop_2.
import os, sys, json, numpy as np
os.chdir("/home/oliver/work/ab-r21"); sys.path.insert(0, ".")
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene
cal = Calibration.load(); sc = Scene.load()
c = cal.cams["head"] if "head" in cal.cams else list(cal.cams.values())[0]
K = np.asarray(c.K, float) if hasattr(c, "K") else None
T0 = c.T_pelvis_optical
z0 = getattr(sc, "counter_z", None)
print("counter_z nominal", z0, "K", None if K is None else K.round(1).tolist())
rows = [json.loads(l) for l in open("/home/oliver/work/e1-audit/results/1148-opencode-e1/labels_clean.jsonl")]
tape = [r for r in rows if r.get("kind") == "bin_tape" and int(r.get("ep", 99)) < 40 and r.get("bbox_px")]
def rot_y(a):
    ca, sa = np.cos(a), np.sin(a); return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
def hit(u, v, pitch_deg, zt):
    d = np.linalg.inv(K) @ np.array([u, v, 1.0]); R = T0[:3, :3]; t = T0[:3, 3]
    R = rot_y(np.radians(-pitch_deg)) @ R   # + pitch_deg = nose down (steeper), about pelvis y through the camera
    dw = R @ d
    s = (zt - t[2]) / dw[2]
    return t + s * dw
for pitch in (0.0, -3.0, -5.7, -7.5, -9.3):
    for dz in (0.0, 0.05):
        xs = []
        for r in tape:
            b = r["bbox_px"]; u = (b[0] + b[2]) / 2; v = b[3]
            p = hit(u, v, pitch, (z0 or 0.01) + dz)
            if abs(p[1]) < 0.12: xs.append(p[0])       # centre bin only
        xs = np.array(xs)
        print(f"pitch {pitch:+.1f} deg  table +{dz*100:.0f} cm: centre-bin near base x median {np.median(xs):.3f} p10 {np.percentile(xs,10):.3f} p90 {np.percentile(xs,90):.3f} (n={len(xs)}) vs nominal 0.400")
