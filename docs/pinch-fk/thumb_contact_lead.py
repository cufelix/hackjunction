"""Per-joint lead at first contact (commanded - joint value at first touch), preshape -> candidate closed posture,
linear joint interpolation, object centred on point_in_palm = jaw midpoint of the candidate.
Usage: PYTHONPATH=repo_env venv_env/bin/python thumb_contact_lead.py  (candidates: thumb-only family scan)"""
import itertools
import numpy as np

_src = open(__file__.replace("thumb_contact_lead", "thumb_contact_combo")).read().split("cl = post[")[0]
_g = {}; exec(_src, _g)
pts, sd, SH, post, spec = _g["pts"], _g["sd"], _g["SH"], _g["post"], _g["spec"]
L = slice(0, 7); pre = post["preshape"][L]; cl = post["closed"][L]
lo7, hi7 = spec.hand_lower[:7], spec.hand_upper[:7]
FR = np.linspace(0, 1, 201)
FINGER_J = {0: (5, 6), 1: (3, 4), 2: (0, 1, 2)}  # tip row -> its joints (index, middle, thumb)


def evaluate(h, p=None):
    tp = pts(h)[3:]
    jaw = np.linalg.norm(tp[2] - tp[0])
    if p is None:
        p = 0.5 * (tp[2] + tp[0])
    rows = [sd(pts(pre + f * (h - pre))[3:], p)[1] for f in FR]
    zg, fin = sd(tp, p)
    out, worst = {}, 0.0
    for k in SH:
        leads = {}
        for i, js in FINGER_J.items():
            hit = next((f for f, r in zip(FR, rows) if r[k][i] <= 0), None)
            if hit is not None:
                for j in js:
                    leads[j] = abs((1 - hit) * (h[j] - pre[j]))
        lmax = max(leads.values()) if leads else 0.0
        worst = max(worst, lmax)
        out[k] = (fin[k], lmax, leads)
    return jaw, p, zg, out, worst


def show(h, tag):
    jaw, p, zg, out, worst = evaluate(h)
    print(f"\n[{tag}] hand {np.round(h, 3).tolist()} jaw {jaw*1000:.1f} mm  point_in_palm {np.round(p, 4).tolist()}  worst lead {worst:.3f} rad")
    for k, (fin, lmax, leads) in out.items():
        jl = " ".join(f"{spec.hand_names[j].replace('left_hand_', '')}:{v:.3f}" for j, v in sorted(leads.items()))
        print(f"   {k:6s} pen idx/mid/thb {fin[0]:+.1f}/{fin[1]:+.1f}/{fin[2]:+.1f}  max lead {lmax:.3f}  [{jl}]")
    return jaw, worst


# thumb-only family: tighten thumb_1/2 from closed; keep the posture with best cube0 balance per jaw width
fam = {}
for t1, t2 in itertools.product(np.arange(-0.1, 0.71, 0.025), np.arange(0.6, 1.51, 0.025)):
    h = cl.copy(); h[1], h[2] = t1, t2
    if np.any(h < lo7 + 0.02) or np.any(h > hi7 - 0.02):
        continue
    tp = pts(h)[3:]; jaw = np.linalg.norm(tp[2] - tp[0]); mid = 0.5 * (tp[2] + tp[0])
    key = round(jaw * 1000)
    if 38 <= key <= 46:
        _, s = sd(tp, mid)
        if not all(s[k][1] > 0 for k in SH):
            continue
        c = abs(mid[2] - 0.015) * 1000 + abs(s["cube0"][0] - s["cube0"][2])
        if key not in fam or c < fam[key][0]:
            fam[key] = (c, h)
res = {}
for key in sorted(fam):
    jaw, worst = show(fam[key][1], f"thumb-only jaw ~{key} mm")
    res[key] = worst
ok = [k for k, w in res.items() if w <= 0.15]
print("\njaw widths with worst lead <= 0.15 rad:", ok, "-> tightest", min(ok) if ok else None)

# chosen pinch_v1 postures (from thumb_contact_pinch.py)
show(np.array([0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]), "pinch_v1 41 mm")
show(np.array([0.0, 0.05, 1.2, -1.0, -0.6, -1.0, -0.6]), "pinch_v1 39 mm backup")

# ---------------------------------------------------------------- pinch_demo (organisers' teleop)
print("\n==== pinch_demo; joint order", [n.replace("left_hand_", "") for n in spec.hand_names[:7]])
D_CL = np.array([0.00, 0.92, 0.25, -0.50, -1.41, -0.49, -1.21])
D_PRE = np.array([-0.67, 0.92, 0.15, -0.28, -0.02, -0.19, -0.05])
print("within limits:", bool(np.all((D_CL >= lo7) & (D_CL <= hi7)) and np.all((D_PRE >= lo7) & (D_PRE <= hi7))))
tp = pts(D_CL)[3:]
print("closed tips palm mm idx/mid/thb:", np.round(tp * 1000, 1).tolist())
auto = 0.5 * (tp[:2].mean(0) + tp[2])
print(f"gap thb-idx {np.linalg.norm(tp[2]-tp[0])*1000:.1f} thb-mid {np.linalg.norm(tp[2]-tp[1])*1000:.1f} idx-mid {np.linalg.norm(tp[1]-tp[0])*1000:.1f} mm;"
      f" auto {np.round(auto, 4).tolist()} jaw-mid {np.round(0.5*(tp[2]+tp[0]), 4).tolist()}")
pre = D_PRE  # evaluate() interpolates from the module-level `pre`
show(D_CL, "pinch_demo from demo preshape")
pre = post["preshape"][L]
show(D_CL, "pinch_demo from our preshape")
pp = pts(D_PRE)
print("demo preshape tips palm mm:", np.round(pp[3:] * 1000, 1).tolist(), " ours:", np.round(pts(post['preshape'][L])[3:] * 1000, 1).tolist())
