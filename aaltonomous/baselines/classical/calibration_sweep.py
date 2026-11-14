"""CPU-only synthetic sweep for the classical head-camera bin self-calibration.

Run from the repository root:
    python -m aaltonomous.baselines.classical.calibration_sweep --trials 100 --seed 2026
"""
from __future__ import annotations

import argparse

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import Renderer
from .perception import Perception


def _scene_objects(scene, rng, n=3):
    layout = scene.cfg["layout"]
    objects, used = [], []
    colors = list(rng.permutation(["red", "yellow", "blue"]))[:n]
    shapes = ("cube", "cylinder")
    while len(objects) < n:
        x, y = rng.uniform(*layout["region_x"]), rng.uniform(*layout["region_y"])
        if any(np.hypot(x - a, y - b) < 0.09 for a, b in used):
            continue
        used.append((x, y))
        height = scene.object_height()
        objects.append({"color": colors[len(objects)], "shape": shapes[len(objects) % len(shapes)],
                        "yaw": rng.uniform(-0.8, 0.8), "size": scene.size, "height": height,
                        "pos": scene.mat_to_pelvis([x, y, height / 2])})
    return objects


def _rotvec_matrix(rotvec):
    """Rodrigues formula without adding a scipy dependency to the offline mock."""
    v = np.asarray(rotvec, float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def jittered_camera(T_nominal, translation, rotvec):
    """Match hooks.py: world-frame translation and pre-multiplied rotation offset."""
    T = np.asarray(T_nominal, float).copy()
    T[:3, :3] = _rotvec_matrix(rotvec) @ T[:3, :3]
    T[:3, 3] += np.asarray(translation, float)
    return T


def run_sweep(trials=100, seed=2026, depth_noise_mm=2.0, hole_frac=0.05):
    rng = np.random.default_rng(seed)
    scene, cal, cfg = Scene.load(), Calibration.load(), load_config()
    nominal = cal._T("head")
    before, after, times = [], [], []
    calibration_failures = detection_failures = 0
    offsets = []

    for _ in range(int(trials)):
        translation = rng.uniform(-0.01, 0.01, 3)
        rotvec = rng.uniform(-np.radians(1.0), np.radians(1.0), 3)
        actual = jittered_camera(nominal, translation, rotvec)
        objects = _scene_objects(scene, rng, n=3)
        rgb, depth = Renderer(scene, cal, T_pelvis_optical=actual).render(
            objects, rng, depth_noise_mm=depth_noise_mm, hole_frac=hole_frac)
        perception = Perception(cfg, scene, cal)

        raw = perception.detect(rgb, depth)
        raw_by_color = {d.color: d for d in raw}
        calibration = perception.calibrate_bins(depth)
        calibrated = perception.detect(rgb, depth)
        calibrated_by_color = {d.color: d for d in calibrated}
        if not calibration.success:
            calibration_failures += 1
        else:
            offsets.append(calibration.xy_offset.copy())
        times.append(calibration.elapsed_ms)

        complete = True
        for obj in objects:
            color = obj["color"]
            if color not in raw_by_color or color not in calibrated_by_color:
                complete = False
                continue
            truth = obj["pos"][:2]
            before.append(float(np.linalg.norm(raw_by_color[color].top[:2] - truth)))
            after.append(float(np.linalg.norm(calibrated_by_color[color].top[:2] - truth)))
        if not complete:
            detection_failures += 1

    def stats(values):
        values = np.asarray(values, float) * 1000.0
        return {"mean_mm": float(np.mean(values)) if len(values) else np.inf,
                "p95_mm": float(np.percentile(values, 95)) if len(values) else np.inf,
                "n": int(len(values))}

    return {
        "trials": int(trials),
        "before": stats(before),
        "after": stats(after),
        "calibration_failures": calibration_failures,
        "calibration_failure_rate": calibration_failures / max(int(trials), 1),
        "detection_failures": detection_failures,
        "detection_failure_rate": detection_failures / max(int(trials), 1),
        "calibration_time_mean_ms": float(np.mean(times)),
        "calibration_time_p95_ms": float(np.percentile(times, 95)),
        "offsets": np.asarray(offsets),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--depth-noise-mm", type=float, default=2.0)
    parser.add_argument("--hole-frac", type=float, default=0.05)
    args = parser.parse_args(argv)
    out = run_sweep(args.trials, args.seed, args.depth_noise_mm, args.hole_frac)
    print(f"trials={out['trials']} calibration_failures={out['calibration_failures']} "
          f"({100 * out['calibration_failure_rate']:.1f}%) detection_failures={out['detection_failures']} "
          f"({100 * out['detection_failure_rate']:.1f}%)")
    print("top-centre XY error (mm)       mean      p95       n")
    for name in ("before", "after"):
        row = out[name]
        print(f"{name:<28} {row['mean_mm']:7.2f}  {row['p95_mm']:7.2f}  {row['n']:6d}")
    print(f"calibration CPU time (ms) mean={out['calibration_time_mean_ms']:.1f} "
          f"p95={out['calibration_time_p95_ms']:.1f}")


if __name__ == "__main__":
    main()
