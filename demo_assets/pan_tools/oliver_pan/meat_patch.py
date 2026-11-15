"""Add the 'meat in pan' demo prop to the pan sim tree + run_pan.py (idempotent). Run with the isaac python on G."""
import math
from pathlib import Path

TREE = Path("/data/aaltonomous-humanoid-pan/aaltonomous")
RUN = Path("/data/work/oliver/pan/tools/run_pan.py")


def sub(p, old, new, once=True):
    s = p.read_text()
    if new in s:
        return
    assert old in s, (p, old[:60])
    p.write_text(s.replace(old, new, 1 if once else -1))


# 1) layout: obj_brown_meat_0 when AALTO_MEAT=1
lay = TREE / "sim/aalto_task/layout.py"
sub(lay, 'PANCAKE_NAME = "obj_tan_pancake_0"',
    'PANCAKE_NAME = "obj_tan_pancake_0"\nMEAT_NAME = "obj_brown_meat_0"   # demo prop (AALTO_PAN=1 + AALTO_MEAT=1): steak slab in the pan')
sub(lay, "    return names + ([PAN_NAME] if pan_enabled() else []) + pancake",
    '    meat = [MEAT_NAME] if pan_enabled() and os.environ.get("AALTO_MEAT", "0") == "1" else []\n'
    "    return names + ([PAN_NAME] if pan_enabled() else []) + pancake + meat")

# 2) scene_cfg: colour + mass
cfg = TREE / "sim/aalto_task/scene_cfg.py"
sub(cfg, "PANCAKE_RGB, PANCAKE_MASS = (0.92, 0.89, 0.84), 0.01",
    "PANCAKE_RGB, PANCAKE_MASS = (0.92, 0.89, 0.84), 0.01\n"
    "MEAT_RGB, MEAT_MASS = (0.24, 0.20, 0.18), 0.06   # dark low-saturation brown: S~64, V~61 (< red's S/V 70 floor)")
sub(cfg, '    if name == "tan":\n        return PANCAKE_RGB',
    '    if name == "tan":\n        return PANCAKE_RGB\n    if name == "brown":\n        return MEAT_RGB')
sub(cfg, '{"pan": PAN_MASS, "pancake": PANCAKE_MASS}', '{"pan": PAN_MASS, "pancake": PANCAKE_MASS, "meat": MEAT_MASS}')


# 3) meat mesh: 90 x 60 x 20 mm superellipse slab, 3 mm chamfered top/bottom edges (convex), mm, z 0..20
def meat_stl(path, a=45.0, b=30.0, h=20.0, ch=3.0, n=4.0, seg=48):
    def ring(ia, ib, z):
        pts = []
        for i in range(seg):
            t = 2 * math.pi * i / seg
            c, s = math.cos(t), math.sin(t)
            pts.append((ia * math.copysign(abs(c) ** (2 / n), c), ib * math.copysign(abs(s) ** (2 / n), s), z))
        return pts
    rings = [ring(a - ch, b - ch, 0.0), ring(a, b, ch), ring(a, b, h - ch), ring(a - ch, b - ch, h)]
    tris = []
    for r0, r1 in zip(rings, rings[1:]):
        for i in range(seg):
            j = (i + 1) % seg
            tris += [(r0[i], r0[j], r1[j]), (r0[i], r1[j], r1[i])]
    for r, z, up in ((rings[0], 0.0, False), (rings[-1], h, True)):
        c = (0.0, 0.0, z)
        for i in range(seg):
            j = (i + 1) % seg
            tris.append((c, r[i], r[j]) if up else (c, r[j], r[i]))
    with open(path, "w") as f:
        f.write("solid meat\n")
        for p, q, r in tris:
            u = [q[k] - p[k] for k in range(3)]
            v = [r[k] - p[k] for k in range(3)]
            nn = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            ln = math.sqrt(sum(x * x for x in nn)) or 1.0
            f.write("facet normal %g %g %g\n outer loop\n" % tuple(x / ln for x in nn))
            for pt in (p, q, r):
                f.write("  vertex %.4f %.4f %.4f\n" % pt)
            f.write(" endloop\nendfacet\n")
        f.write("endsolid meat\n")


meat_stl(TREE / "scene/assets/object_meat.stl")

# 4) run_pan: meat entry, cluttered variants, meat pose in the poller, meat-in-pan metric
sub(RUN, '    for ci, shape, xy in others:\n',
    '    if os.environ.get("AALTO_MEAT") == "1":   # steak slab on the pan base: base top + half the 20 mm slab + 1 mm\n'
    '        lay.append(entry("brown", "meat", (px, py), 0.0, float(os.environ.get("PAN_TOP_MAT", "0.004")) + 0.010 + 0.001))\n'
    '    for ci, shape, xy in others:\n')
sub(RUN, '    "egg_left":    ((0.30, 0.10), -90.0, [(1, "cube", (0.27, -0.25))]),\n',
    '    "egg_left":    ((0.30, 0.10), -90.0, [(1, "cube", (0.27, -0.25))]),\n'
    "    # cluttered 'hold the pan, do not spill the meat' scene: >= 5 cm from pan + handle, all x <= 0.35 (bin band 0.37)\n"
    '    "meat_right":  ((0.30, -0.10), 90.0, [(0, "cube", (0.27, 0.20)), (1, "cylinder", (0.33, 0.27)),\n'
    '                                          (2, "cube", (0.25, 0.28)), (0, "triangle", (0.30, -0.27))]),\n'
    '    "meat_left":   ((0.30, 0.10), -90.0, [(1, "cube", (0.27, -0.20)), (2, "cylinder", (0.33, -0.27)),\n'
    '                                          (0, "cube", (0.25, -0.28)), (1, "triangle", (0.30, 0.27))]),\n')
sub(RUN, '                p = objs.get(PAN)\n                if p is not None:\n'
         '                    self.rows.append({"t": round(time.monotonic() - t0, 2), "pos": p["pos"], "quat": p["quat_xyzw"],\n'
         '                                      "bodies": bodies})',
    '                p = objs.get(PAN)\n                if p is not None:\n'
    '                    m = objs.get("obj_brown_meat_0")\n'
    '                    self.rows.append({"t": round(time.monotonic() - t0, 2), "pos": p["pos"], "quat": p["quat_xyzw"],\n'
    '                                      "bodies": bodies, "meat": None if m is None else m["pos"]})')
sub(RUN, "def closest_approach(rows):",
    '''def meat_in_pan(rows, sc_scene):
    """Meat centre inside the rim radius (pan frame xy) and above the pan base, at: the end of the lifted phase
    (last sample with the pan >= 3 cm up = end of the hold, before the put-back) and the end of the episode."""
    rows = [r for r in rows if r["pos"][0] > 0.0 and r.get("meat")]
    if not rows:
        return None
    z0 = float(sc_scene.mat_to_pelvis(np.array([0.0, 0.0, PAN_Z_MAT]))[2])
    rim_in, base_top = float(os.environ.get("PAN_RIM_IN_M", "0.072")), -float(os.environ.get("PAN_BASE_LOCAL_M", "0.026"))

    def check(r):
        lp = quat_xyzw_to_R(r["quat"]).T @ (np.asarray(r["meat"]) - np.asarray(r["pos"]))
        return {"in": bool(math.hypot(lp[0], lp[1]) < rim_in and lp[2] > base_top), "t": r["t"],
                "meat_in_pan_frame_mm": [int(round(1000 * v)) for v in lp]}
    lifted = [r for r in rows if r["pos"][2] - z0 >= 0.03]
    out = {"end_of_episode": check(rows[-1])}
    if lifted:
        out["end_of_hold"] = check(lifted[-1])
        out["lifted_s"] = round(lifted[-1]["t"] - lifted[0]["t"], 2)
        out["in_frac_while_lifted"] = round(float(np.mean([check(r)["in"] for r in lifted])), 3)
    return out


def closest_approach(rows):''')
sub(RUN, '            v["closest"] = closest_approach(poll.rows)\n',
    '            v["closest"] = closest_approach(poll.rows)\n            v["meat"] = meat_in_pan(poll.rows, env.scene)\n')
sub(RUN, 'f"end={res[\'end_reason\']} t={res[\'wall_s\']}s", flush=True)',
    'f"end={res[\'end_reason\']} t={res[\'wall_s\']}s", flush=True)\n            if v.get("meat"):\n'
    '                print(f"{vname:11s} meat={json.dumps(v[\'meat\'])}", flush=True)')
print("patched")
