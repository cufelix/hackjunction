"""Validation CLI for profile_detect: per-frame signals and the sim/none share over saved head frames.

usage: python tools/profile_detect_eval.py <dir> [--episodes A:B] [--window N]
<dir> holds frames as *.npz (keys head_rgb, head_depth) or *.png + *.npy (depth). Paths containing "drop_2" are
held out: any episode index >= 40 is refused. Only directory input is supported (no LeRobot decoder here).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aaltonomous.baselines.classical.profile_detect import detect_profile  # noqa: E402

HELD_OUT_EP = 40


def _episode(path):
    m = re.search(r"ep(?:isode)?[_-]?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _load(path):
    if path.endswith(".npz"):
        z = np.load(path)
        return {"head_rgb": z["head_rgb"], "head_depth": z["head_depth"]}
    import cv2
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return {"head_rgb": rgb, "head_depth": np.load(path[:-4] + ".npy")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--episodes", default=None, help="A:B (B exclusive) episode index range")
    ap.add_argument("--window", type=int, default=5, help="frames per decision (profile_auto.auto_frames)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.root, "**", "*.npz"), recursive=True)
                   or [p for p in glob.glob(os.path.join(a.root, "**", "*.png"), recursive=True)
                       if os.path.exists(p[:-4] + ".npy")])
    if "drop_2" in os.path.abspath(a.root) and a.episodes and int(a.episodes.split(":")[1]) > HELD_OUT_EP:
        sys.exit("REFUSED: drop_2 episodes >= 40 are held out")
    if "drop_2" in os.path.abspath(a.root):
        eps = [_episode(f) for f in files]
        if any(e is None or e >= HELD_OUT_EP for e in eps):
            sys.exit("REFUSED: drop_2 episodes >= 40 are held out (and unindexed files are not checked)")
    if a.episodes:
        lo, hi = (int(x) for x in a.episodes.split(":"))
        files = [f for f in files if _episode(f) is not None and lo <= _episode(f) < hi]
    if not files:
        sys.exit("no frames found")
    frames = [_load(f) for f in files]
    results = []
    for i in range(0, len(frames) - a.window + 1, a.window):
        prof, info = detect_profile(frames[i:i + a.window])
        results.append(prof)
        for f, p in zip(files[i:i + a.window], info["per_frame"]):
            print(f"{os.path.basename(f)} holes={p['holes']:.4f} plane_std_mm={p['plane_std_mm']:.2f} "
                  f"palette={p['palette']} vote={'sim' if p['sim'] else 'none'}")
        print(f"window {i // a.window}: auto={prof} votes={info['votes']}/{info['frames']}")
    n = len(results)
    if n:
        print(f"share sim={sum(r == 'sim' for r in results) / n:.2f} none={sum(r == 'none' for r in results) / n:.2f} "
              f"windows={n}")


if __name__ == "__main__":
    main()
