#!/usr/bin/env python3
"""Real-frame recall/precision on the public organiser drop_2 head frames (no depth, recorded arm pose).

Inputs (all outside git): <data>/frames/e{ep:02d}{a,b,c,z}.png, <data>/states.json (name -> 28 joint state),
<data>/labels.json (per-GT colour/shape/centroid/locus; visible_on_mat rows are the recall denominator; the E1 clean labels).
Episodes >= 40 are HELD OUT: refused unless --heldout is passed (use once, at the very end; no labels exist for them,
so only detection counts are reported there).

  python tools/real_recall.py --data D --overlay finals_r23 [--set k=v ...] --split train|dev|all|heldout --out r.json
Split: train = even episodes < 40, dev = odd episodes < 40.
Matching = E1 contract (greedy, 25 px nearest centroid, file order, one used-set per frame over ALL GT rows), colour-aware
(orange<->red, purple<->blue). Precision = matched detections / all detections (lower bound: unlabelled real objects count as FP).
"""
from __future__ import annotations
import argparse, json, math, os, sys
from collections import Counter, defaultdict

CANON = {"red": "orange", "orange": "orange", "blue": "purple", "purple": "purple", "yellow": "yellow"}
MATCH_PX = 25.0
import re
# second GT row of one physical object (its shadowed side face labelled apart from the top face): not a separate target
SECONDARY = re.compile(r"shadowed (lower|side)|shadowed face|lower face|lower side|upper side|side face", re.I)


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def in_split(ep, split):
    if split == "heldout":
        return ep >= 40
    if ep >= 40:
        return False
    return {"all": True, "train": ep % 2 == 0, "dev": ep % 2 == 1}[split]


def detect(a):
    import cv2
    import numpy as np
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.perception import Perception
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    cfg = load_config(overlay=a.overlay, overrides=parse_set(a.set))
    p = Perception(cfg, Scene.load(), Calibration.load())
    if not a.nominal_counter_z:
        # drop_2 has no depth, so the runtime plane fit cannot run: use the configured no-depth prior (r23: 0.15 m), exactly
        # what the policy falls back to on the robot. --nominal-counter-z keeps the sim placeholder (0.01 m) of the E1 audit.
        cz = p.counter_z_prior()
        if cz is not None:
            p.scene.counter_z_runtime = cz
    states = json.load(open(os.path.join(a.data, "states.json")))
    names = sorted(n for n in os.listdir(os.path.join(a.data, "frames")) if n.endswith(".png"))
    res = {}
    global REJ
    REJ = {}
    for fn in names:
        name = fn[:-4]
        ep = int(name[1:3])
        if ep >= 40 and not a.heldout:
            continue
        if not in_split(ep, a.split):
            continue
        s = states.get(name)
        rq = (np.asarray(s[:14], float), np.asarray(s[14:28], float)) if s and len(s) >= 28 else None
        rgb = cv2.cvtColor(cv2.imread(os.path.join(a.data, "frames", fn)), cv2.COLOR_BGR2RGB)
        kw = {"robot_q": rq} if rq is not None else {}
        dets, rej = p.detect(rgb, None, debug=True, include_bins=False, **kw)
        REJ[name] = [dict(color=d.color, uv=[float(d.uv[0]), float(d.uv[1])], n_px=int(d.n_px), why=str(w)) for d, w in rej
                     if getattr(d, "uv", None) is not None]
        res[name] = [dict(color=d.color, uv=[float(d.uv[0]), float(d.uv[1])], n_px=int(d.n_px),
                          shape=getattr(d, "shape", None)) for d in dets]
    return res


REJ = {}


def miss_reasons(dets, labels):
    """For each visible GT not matched colour-aware: nearest rejected blob (any colour) within 25 px -> its reason, else no_blob;
    a wrong-colour accepted detection within 25 px -> wrong_colour:<det colour>."""
    out = []
    for g in labels["gt"]:
        fr = g["frame"]
        if fr not in dets or g["locus"] != "visible_on_mat":
            continue
        c = g["centroid"]
        ok = [d for d in dets[fr] if math.hypot(d["uv"][0] - c[0], d["uv"][1] - c[1]) <= MATCH_PX]
        if any(CANON.get(d["color"]) == g["colour"] for d in ok):
            continue
        if ok:
            why = "wrong_colour:" + ok[0]["color"]
        else:
            rj = [r for r in REJ.get(fr, []) if math.hypot(r["uv"][0] - c[0], r["uv"][1] - c[1]) <= MATCH_PX]
            why = ("rej:" + rj[0]["why"].split(":")[0] + ("/" + rj[0]["why"].split(":")[1][:20] if ":" in rj[0]["why"] else "")) if rj else "no_blob"
        out.append(dict(frame=fr, cls=f"{g['colour']}-{g['shape']}", uv=c, why=why))
    return out


def score(dets, labels, split):
    gts = defaultdict(list)
    for g in labels["gt"]:
        gts[g["frame"]].append(g)
    cls = defaultdict(lambda: [0, 0, 0])   # n, matched colour-aware, matched colour-agnostic
    obj = [0, 0]                            # object-level: visible rows minus secondary side-face rows
    nd = nm = 0
    fp = Counter()
    wrongcol = 0
    for fr, dl in dets.items():
        used = set()
        gl = gts.get(fr, [])
        matched_det = set()
        for g in gl:
            best, bd, bany, bad = None, 1e9, None, 1e9
            for j, d in enumerate(dl):
                if j in used:
                    continue
                dist = math.hypot(d["uv"][0] - g["centroid"][0], d["uv"][1] - g["centroid"][1])
                if dist <= MATCH_PX and dist < bad:
                    bany, bad = j, dist
                if dist <= MATCH_PX and CANON.get(d["color"]) == g["colour"] and dist < bd:
                    best, bd = j, dist
            if g["locus"] == "visible_on_mat":
                k = f"{g['colour']}-{g['shape']}"
                cls[k][0] += 1
                cls[k][1] += best is not None
                cls[k][2] += bany is not None
                if not SECONDARY.search(g["note"]):
                    obj[0] += 1
                    obj[1] += best is not None
            if best is not None:
                used.add(best)
                matched_det.add(best)
        for j, d in enumerate(dl):
            nd += 1
            if j in matched_det:
                nm += 1
            else:
                fp[CANON.get(d["color"], d["color"])] += 1
    tot = [sum(v[i] for v in cls.values()) for i in range(3)]
    out = dict(split=split, frames=len(dets), visible_n=tot[0], matched=tot[1], recall=round(tot[1] / max(tot[0], 1), 3),
               recall_obj=round(obj[1] / max(obj[0], 1), 3), obj_n=obj[0], recall_agnostic=round(tot[2] / max(tot[0], 1), 3), dets=nd, det_matched=nm,
               precision=round(nm / max(nd, 1), 3), fp=nd - nm, fp_by_colour=dict(fp),
               per_class={k: dict(n=v[0], matched=v[1], recall=round(v[1] / max(v[0], 1), 3)) for k, v in sorted(cls.items())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--overlay", default="finals_r23")
    ap.add_argument("--set", action="append")
    ap.add_argument("--split", default="train", choices=["train", "dev", "all", "heldout"])
    ap.add_argument("--heldout", action="store_true", help="required for --split heldout (one-shot, final evaluation only)")
    ap.add_argument("--nominal-counter-z", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.split == "heldout" and not a.heldout:
        sys.exit("REFUSED: episodes 40-47 are held out; pass --heldout for the single final evaluation")
    dets = detect(a)
    if a.split == "heldout":
        c = Counter(CANON.get(d["color"], d["color"]) for dl in dets.values() for d in dl)
        res = dict(split="heldout", frames=len(dets), dets=sum(c.values()), dets_by_colour=dict(c), note="no GT labels exist for ep 40-47")
    else:
        res = score(dets, json.load(open(os.path.join(a.data, "labels.json"))), a.split)
    res["overlay"], res["overrides"] = a.overlay, parse_set(a.set)
    if a.out:
        miss = miss_reasons(dets, json.load(open(os.path.join(a.data, "labels.json")))) if a.split != "heldout" else []
        res["miss_reasons"] = dict(Counter((m["cls"].split("-")[0], m["why"]) for m in miss).most_common(40)) if False else \
            {f"{k[0]}|{k[1]}": v for k, v in Counter((m["cls"], m["why"]) for m in miss).most_common(60)}
        json.dump(dict(result=res, dets=dets, rej=REJ, miss=miss), open(a.out, "w"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
