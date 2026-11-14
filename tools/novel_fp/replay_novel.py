"""Replay dumped sim frames through perception and count novel detections (false-positive gate for oliver/novel-fp).

    python3 tools/novel_fp/replay_novel.py <frames_dir> [--overlay finals_r23_universal] [--png <dir>] [--expect-novel ep]

Config = what the policy server runs in sim: load_config(overlay, profile="sim", overrides=profile_auto.sim_overrides).
Per episode: frames with >= 1 novel detection and the labels seen. Gate for stock scenes: 0 novel detections in
every frame. With --png, the first frame that has novel detections is saved with each novel det's uv + label drawn.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load_cfg(overlay, fp_filter=None):
    from aaltonomous.baselines.classical.config import load_config
    probe = load_config(overlay=overlay)
    extra = dict((probe.get("profile_auto") or {}).get("sim_overrides") or {})
    if fp_filter is not None:
        extra["perception.novel.fp_filter"] = fp_filter
    return load_config(overlay=overlay, profile="sim", overrides=extra)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--overlay", default="finals_r23_universal")
    ap.add_argument("--png")
    ap.add_argument("--max-t", type=float, default=1e9, help="only frames with t <= this (s)")
    ap.add_argument("--static-rad", type=float, default=0.0,
                    help="only frames whose arm_q moved <= this (max abs, rad) since the previous kept frame: the policy "
                         "looks with a still arm, and a moving arm's image lags its joints (FK mask misaligned)")
    ap.add_argument("--fp-filter", choices=("on", "off"), help="force perception.novel.fp_filter (default: overlay's)")
    ap.add_argument("--tag", default="", help="suffix for the report json (two code versions on one frames dir)")
    a = ap.parse_args(argv)

    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    cfg = load_cfg(a.overlay, None if a.fp_filter is None else a.fp_filter == "on")
    print(f"fp_filter={cfg['perception'].get('novel', {}).get('fp_filter')}", flush=True)
    pol = ClassicalPolicy(config=cfg, log=lambda *_: None)
    per = pol.perception
    tot_frames = tot_bad = 0
    report = {}
    for f in sorted(Path(a.frames).glob("*.npz")):
        d = np.load(f)
        meta = json.loads(f.with_suffix(".json").read_text()) if f.with_suffix(".json").exists() else {}
        labels, bad, n, saved = Counter(), 0, 0, False
        for i in range(len(d["t"])):
            if float(d["t"][i]) > a.max_t:
                continue
            if a.static_rad > 0 and (i == 0 or float(np.max(np.abs(d["arm_q"][i] - d["arm_q"][i - 1]))) > a.static_rad):
                continue
            rgb, depth = d["rgb"][i], d["depth"][i]
            dets = per.detect(rgb, depth, robot_q=(d["arm_q"][i].astype(float), d["hand_q"][i].astype(float))) or []
            nov = [x for x in dets if getattr(x, "novel", False)]
            n += 1
            if nov:
                bad += 1
                labels.update(x.label for x in nov)
                if a.png and not saved:
                    import cv2
                    img = np.ascontiguousarray(rgb[..., ::-1]).copy()
                    for x in nov:
                        u, v = map(int, x.uv)
                        cv2.circle(img, (u, v), 6, (0, 0, 255), 2)
                        cv2.putText(img, x.label, (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    Path(a.png).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(Path(a.png) / f"{f.stem}_f{i:03d}.png"), img)
                    saved = True
        tot_frames += n
        tot_bad += bad
        report[f.stem] = {"frames": n, "frames_with_novel": bad, "labels": dict(labels),
                          "layout": [f"{o.get('color')} {o.get('shape')}" for o in meta.get("layout", [])]}
        print(f"{f.stem}: {bad}/{n} frames with novel  {dict(labels)}", flush=True)
    print(f"TOTAL: {tot_bad}/{tot_frames} frames with >= 1 novel detection")
    json.dump(report, sys.stderr if False else open(Path(a.frames) / f"replay_{a.overlay}{a.tag}.json", "w"), indent=1)
    return 0 if tot_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
