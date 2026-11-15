"""Stress shape_id.classify_top_face on the 108 sim top-face clouds (knowledge/humanoid/shape_features_dataset.npz) with
real-like degradations: sparse points, depth/xy noise, a clipped (occluded / cut by a bin) top face. Reports, per
condition, label accuracy and how many WRONG labels survive a confidence gate (the shape-rule safety question)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aaltonomous.baselines.classical.shape_id import classify_top_face  # noqa: E402

D = np.load(sys.argv[1], allow_pickle=True)
meta = json.loads(str(D["meta"]))
rng = np.random.default_rng(0)


def degrade(xy, n, sigma, clip):
    xy = np.asarray(xy, float)
    if clip:   # cut away a strip of the face (occlusion by hand / bin exclusion)
        c = xy.mean(0); ang = rng.uniform(0, 2 * np.pi); d = np.array([np.cos(ang), np.sin(ang)])
        proj = (xy - c) @ d
        xy = xy[proj < np.quantile(proj, 1 - clip)]
    if len(xy) > n:
        xy = xy[rng.choice(len(xy), n, replace=False)]
    return xy + rng.normal(0, sigma, xy.shape)


print(f"{'n':>5} {'sigma_mm':>8} {'clip':>5} | labelled acc | wrong@conf>=0 >=.3 >=.5 >=.7 | abstain")
for n in (1500, 300, 150, 80, 50):
    for sig in (0.0, 0.002, 0.004):
        for clip in (0.0, 0.2, 0.35):
            tot = ok = ab = 0; wrong = {0: 0, .3: 0, .5: 0, .7: 0}; kept = {0: 0, .3: 0, .5: 0, .7: 0}
            for rep in range(3):
                for i, m in enumerate(meta):
                    xy = degrade(D[f"top_{i:03d}"], n, sig, clip)
                    lab, conf = classify_top_face(xy)
                    tot += 1
                    if lab is None:
                        ab += 1; continue
                    ok += lab == m["shape"]
                    for th in wrong:
                        if conf >= th:
                            kept[th] += 1; wrong[th] += lab != m["shape"]
            print(f"{n:5d} {sig*1000:8.1f} {clip:5.2f} | {ok/max(tot-ab,1):.3f} ({tot-ab}/{tot}) | "
                  + " ".join(f"{wrong[t]:3d}/{kept[t]:3d}" for t in wrong) + f" | {ab}")
