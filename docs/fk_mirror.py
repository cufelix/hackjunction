"""FK mirror check: is the right grasp point (point_in_palm * [1,-1,1]) the exact mirror of the left one?
Run from outside the repo: PYTHONPATH=repo_env venv_env/bin/python fk_mirror.py"""
import numpy as np

from aaltonomous.baselines.classical.config import load_config, hand_postures, home_arm, mirror_left_to_right
from aaltonomous.baselines.classical.kinematics import ArmKinematics, TIP, FINGER_LINKS
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

np.set_printoptions(precision=4, suppress=True)
M = np.diag([1.0, -1.0, 1.0])
mm = lambda v: np.round(np.asarray(v) * 1000, 1)
# arm joint axes (URDF): sh_pitch y, sh_roll x, sh_yaw z, elbow y, wr_roll x, wr_pitch y, wr_yaw z.
# Under the y-reflection a rotation about y keeps its sign, about x / z flips it.
ARM_MIRROR = np.array([1, -1, -1, 1, -1, 1, -1], float)

cfg = load_config()
spec = JointSpec()
kin = ArmKinematics(cfg, spec)
post = hand_postures(cfg, spec)
home = home_arm(cfg, spec)
print("profile", cfg["profile"], "| offset L", mm(kin.offset["left"]), "R", mm(kin.offset["right"]))
print("finger frames:", [f"<side>_{f}" for f in FINGER_LINKS], "tip offsets in link (ASSUMED, kinematics.TIP):", TIP)

# ---------------------------------------------------------------- (1) grasp point vs fingertips, palm frame
for name in ("closed", "preshape"):
    print(f"\n== {name}: hand14 = {post[name]}")
    tips = {}
    for side in ("left", "right"):
        t = kin.tips_in_palm(side, post[name])  # rows: index, middle, thumb
        tips[side] = t
        g = kin.offset[side]
        d = np.linalg.norm(t - g, axis=1)
        print(f"  {side:5s} tips_in_palm(mm) idx {mm(t[0])} mid {mm(t[1])} thb {mm(t[2])}")
        print(f"  {side:5s} |g-tip| mm: index {d[0]*1000:.1f}  middle {d[1]*1000:.1f}  thumb {d[2]*1000:.1f}"
              f"  -> nearer finger: {'index' if d[0] < d[1] else 'middle'}")
    print("  max |tip_R - M tip_L| mm:", mm(np.abs(tips["right"] - tips["left"] @ M).max()))

# ---------------------------------------------------------------- (3) frame mirror at random mirrored arm q
rng = np.random.default_rng(1)
worst_R = worst_p = worst_g = 0.0
for _ in range(200):
    qL = rng.uniform(kin.lo[:7], kin.hi[:7])
    arm = np.r_[qL, ARM_MIRROR * qL]
    kin._fk(arm)
    PL, PR = kin.data.oMf[kin.palm["left"]], kin.data.oMf[kin.palm["right"]]
    worst_R = max(worst_R, np.abs(PR.rotation - M @ PL.rotation @ M).max())
    worst_p = max(worst_p, np.abs(PR.translation - M @ PL.translation).max())
    gL, _ = kin.grasp_pose("left", arm)
    gR, _ = kin.grasp_pose("right", arm)
    worst_g = max(worst_g, np.abs(gR - M @ gL).max())
print(f"\n(3) 200 random y-mirrored arm q: max|R_R - M R_L M| {worst_R:.2e}, max|p_R - M p_L| {worst_p*1000:.4f} mm,"
      f" max|g_R - M g_L| {worst_g*1000:.4f} mm")
# hand joints: posture mirror rule vs exact geometric mirror (thumb0 is about palm y -> same sign)
worst_h = worst_h2 = 0.0
lo, hi = spec.hand_lower[:7], spec.hand_upper[:7]
for _ in range(200):
    hl = rng.uniform(lo, hi)
    hr = mirror_left_to_right(hl)
    hr_exact = hr.copy(); hr_exact[0] = hl[0]
    tl = kin.tips_in_palm("left", np.r_[hl, np.zeros(7)])
    worst_h = max(worst_h, np.abs(kin.tips_in_palm("right", np.r_[np.zeros(7), hr]) - tl @ M).max())
    worst_h2 = max(worst_h2, np.abs(kin.tips_in_palm("right", np.r_[np.zeros(7), hr_exact]) - tl @ M).max())
print(f"    random hand q: mirror_left_to_right tips err {worst_h*1000:.2f} mm; with thumb0 sign kept {worst_h2*1000:.4f} mm")

# ---------------------------------------------------------------- (2) planner IK at a counter pose
sc = Scene.load()
z = sc.counter_z + 0.0225
print(f"\n(2) counter_z {sc.counter_z:.4f}, target z {z:.4f}")
res = {}
for side, y in (("left", 0.10), ("right", -0.10)):
    h = np.asarray(cfg["grasp"]["finger_hints"][0], float) * (M.diagonal() if side == "right" else 1.0)
    kin.hint = h / np.linalg.norm(h)
    sl = ArmKinematics.sl(side)
    q, ok, err, tilt = kin.solve(side, np.r_[0.30, y, z], home, top_down=True, extra_seeds=[home[sl]])
    g, a = kin.grasp_pose(side, q)
    kin._fk(q)
    R = kin.data.oMf[kin.palm[side]].rotation.copy()
    tips = kin.hand_points(side, q, post["closed"])[5:]
    res[side] = (q[sl], g, R, tips)
    print(f"  {side:5s} ok {ok} err {err*1000:.2f} mm tilt {np.degrees(tilt):.1f} deg  q7 {q[sl]}")
    print(f"        grasp pt (pelvis, mm) {mm(g)}  approach {a}")
qL, gL, RL, tL = res["left"]
qR, gR, RR, tR = res["right"]
print("  q_R - mirror(q_L):", qR - ARM_MIRROR * qL)
print("  g_R - M g_L (mm):", mm(gR - M @ gL), " max|R_R - M R_L M|:", f"{np.abs(RR - M @ RL @ M).max():.2e}")
print("  closed tips pelvis R - M L (mm):", mm(np.abs(tR - tL @ M).max()))
