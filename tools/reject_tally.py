#!/usr/bin/env python3
"""Run classical RGB perception over a frame folder and print rejection counts by colour.

PNG files are read as RGB with no depth. NPZ files need an ``rgb`` array and may
contain ``depth``; either may be a single frame or a leading stack of frames.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaltonomous.baselines.classical.config import load_config  # noqa: E402
from aaltonomous.baselines.classical.perception import Perception  # noqa: E402
from aaltonomous.interface.calibration import Calibration  # noqa: E402
from aaltonomous.scene.scene import Scene  # noqa: E402


def frames(path):
    for file in sorted(path.rglob("*")):
        suffix = file.suffix.lower()
        if suffix == ".png":
            bgr = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"warning: unreadable image {file}", file=sys.stderr)
                continue
            yield file, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None
        elif suffix == ".npz":
            with np.load(file) as data:
                if "rgb" not in data:
                    print(f"warning: {file} has no rgb array", file=sys.stderr)
                    continue
                rgb = np.asarray(data["rgb"])
                depth = np.asarray(data["depth"]) if "depth" in data else None
                rgbs = rgb[None] if rgb.ndim == 3 else rgb
                depths = [None] * len(rgbs) if depth is None else (depth[None] if depth.ndim == 2 else depth)
                if len(depths) != len(rgbs):
                    print(f"warning: frame count differs in {file}", file=sys.stderr)
                    continue
                for i, (im, dep) in enumerate(zip(rgbs, depths)):
                    yield Path(f"{file}#{i}"), im, dep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--overlay", default="finals_r22_gate")
    ap.add_argument("--profile", default="none", help="none keeps real-camera file intrinsics")
    args = ap.parse_args(argv)
    cfg = load_config(profile=args.profile, overlay=args.overlay,
                      overrides={"perception.reject_log": True, "perception.still_vote": False})
    calibration = Calibration.load()
    camera = calibration.cams[cfg["perception"]["camera"]]
    expected_rgb = (camera.height, camera.width, 3)
    perception = Perception(cfg, Scene.load(), calibration, log=lambda _msg: None)
    accepted, count = Counter(), 0
    for name, rgb, depth in frames(args.folder):
        if rgb.shape != expected_rgb or (depth is not None and depth.shape != expected_rgb[:2]):
            print(f"warning: {name} has shape rgb={rgb.shape}, depth="
                  f"{None if depth is None else depth.shape}; expected {expected_rgb}", file=sys.stderr)
            continue
        dets, _ = perception.detect(rgb, depth, debug=True)
        count += 1
        accepted.update(d.color for d in dets)
    tally = perception.rejection_table()
    names = (("orange", "red"), ("yellow", "yellow"), ("purple", "blue"))
    stages = sorted({stage for values in tally.values() for stage in values})
    width = max([5, *(len(stage) for stage in stages), len("accepted")])
    print(f"frames: {count}")
    print(f"{'stage':<{width}}  " + "  ".join(f"{label:>8}" for label, _ in names))
    print(f"{'accepted':<{width}}  " + "  ".join(f"{accepted[key]:8d}" for _, key in names))
    for stage in stages:
        print(f"{stage:<{width}}  " + "  ".join(f"{tally.get(key, {}).get(stage, 0):8d}" for _, key in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
