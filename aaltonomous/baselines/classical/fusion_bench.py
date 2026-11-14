"""Offline validation of perception.depth_fusion + perception.track on synthetic mock_world sequences.
    python -m aaltonomous.baselines.classical.fusion_bench [n_seq]
Sequence: 3-4 still objects, per-frame depth noise sigma 2-3 mm, 5-10 % holes, flying pixels on object edges, and
occasional single-frame phantom blobs (a coloured patch with consistent depth, 1 frame only)."""
from __future__ import annotations

import sys
import time

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import Renderer
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
H = SC.object_height()


def make_objects(rng, n=3):
    L = SC.cfg["layout"]
    objs, used = [], []
    for col in ("red", "yellow", "blue", "red")[:n]:
        while True:
            x, y = rng.uniform(*L["region_x"]), rng.uniform(*L["region_y"])
            if all(np.hypot(x - a, y - b) > 0.09 for a, b in used):
                break
        used.append((x, y))
        objs.append(dict(color=col, shape="cube", yaw=rng.uniform(-0.8, 0.8), size=SC.size, height=H,
                         pos=SC.mat_to_pelvis([x, y, H / 2])))
    return objs


class Sequence:
    """Clean render once (fast), then per-frame sensor noise; phantom frames re-render with one extra object."""

    def __init__(self, rng, n_obj=3, sigma_mm=2.5, hole=0.08, fly=0.3):
        self.rng, self.sigma, self.hole, self.fly = rng, sigma_mm, hole, fly
        self.objs = make_objects(rng, n_obj)
        self.rend = Renderer(SC, CAL)
        self.rgb, self.depth = self.rend.render(self.objs)

    def truth(self):
        return [(o["color"], o["pos"] + [0, 0, o["height"] / 2]) for o in self.objs]

    def phantom_frame(self):
        L = SC.cfg["layout"]
        while True:
            x, y = self.rng.uniform(*L["region_x"]), self.rng.uniform(*L["region_y"])
            if all(np.linalg.norm(SC.mat_to_pelvis([x, y, 0])[:2] - o["pos"][:2]) > 0.08 for o in self.objs):
                break
        ph = dict(color=str(self.rng.choice(["red", "yellow", "blue"])), shape="cube", yaw=0.3, size=SC.size,
                  height=H, pos=SC.mat_to_pelvis([x, y, H / 2]))
        return self._noise(*self.rend.render(self.objs + [ph])), ph

    def frame(self):
        return self._noise(self.rgb, self.depth)

    def _noise(self, rgb, depth):
        r = self.rng
        d = depth.astype(np.float32)
        d = d + r.normal(0, self.sigma, d.shape)
        d[r.random(d.shape) < self.hole] = 0
        # flying pixels: edge pixels (depth step) get a value between object and background
        gx = np.abs(np.diff(depth.astype(np.int32), axis=1, prepend=depth[:, :1].astype(np.int32)))
        edge = (gx > 20) & (r.random(d.shape) < self.fly)
        d[edge] = r.uniform(300, 900, edge.sum())
        return rgb, np.clip(d, 0, 65535).astype(np.uint16)


def _cfg(fusion, track, **kw):
    ov = {"perception.depth_fusion.enabled": fusion, "perception.track.enabled": track}
    ov.update(kw)
    return load_config(overlay="none", overrides=ov)


def run(n_seq=6, n_frames=12, phantom_rate=0.15, seed=0, verbose=True):
    """Returns {mode: dict(err_xy_mm list, err_z_mm list, phantoms, t_ms)}. One 'detect call' per frame, like the
    policy's observe ticks + final look; error is that of every real object on the frame (miss = excluded, counted)."""
    modes = {"single": (False, False), "fused": (True, False), "tracked": (False, True), "both": (True, True)}
    res = {m: dict(exy=[], ez=[], phantom=0, miss=0, t=[], pos={}) for m in modes}
    for s in range(n_seq):
        rng = np.random.default_rng(seed + s)
        seq = Sequence(rng, sigma_mm=3.0, hole=0.10)
        frames = []
        for f in range(n_frames):
            if f >= 5 and rng.random() < phantom_rate:
                (rgb, depth), ph = seq.phantom_frame()
                frames.append((rgb, depth, ph["pos"]))
            else:
                frames.append((*seq.frame(), None))
        truth = seq.truth()
        for m, (fu, tr) in modes.items():
            p = Perception(_cfg(fu, tr), SC, CAL)
            if p.temporal:
                p.reset_stream()
            for f, (rgb, depth, ph) in enumerate(frames):
                t0 = time.perf_counter()
                dets = p.detect(rgb, depth)
                if p.fuser is not None:
                    p.fuser.push(depth)
                res[m]["t"].append((time.perf_counter() - t0) * 1000)
                if f < 5:
                    continue   # settled history
                for col, top in truth:
                    c = [d for d in dets if d.color == col and np.linalg.norm(d.top[:2] - top[:2]) < 0.03]
                    if c:
                        d = min(c, key=lambda d: np.linalg.norm(d.top[:2] - top[:2]))
                        res[m]["exy"].append(1000 * np.linalg.norm(d.top[:2] - top[:2]))
                        res[m]["ez"].append(1000 * abs(d.top[2] - top[2]))
                        res[m]["pos"].setdefault((s, col), []).append(d.top[:2].copy())
                    else:
                        res[m]["miss"] += 1
                if ph is not None:
                    res[m]["phantom"] += sum(1 for d in dets if np.linalg.norm(d.top[:2] - ph[:2]) < 0.03)
    return res


def table(res):
    lines = [f"{'mode':8} {'xy mean':>8} {'xy p95':>7} {'xy max':>7} {'jitter':>7} {'z mean':>7} {'miss':>5} {'phantoms':>9} {'ms/frame':>9}"]
    for m, r in res.items():
        e = np.array(r["exy"])
        jit = np.mean([1000 * np.linalg.norm(np.std(np.array(v), axis=0)) for v in r["pos"].values() if len(v) > 2])
        lines.append(f"{m:8} {e.mean():8.2f} {np.percentile(e, 95):7.2f} {e.max():7.2f} {jit:7.2f} {np.mean(r['ez']):7.2f} "
                     f"{r['miss']:5d} {r['phantom']:9d} {np.mean(r['t']):9.1f}")
    return "\n".join(lines)


def confirm_latency(n_seq=6, **track):
    """Frames until every real object is reported, tracker with passthrough_cold OFF (strict). 1 = first frame."""
    out = []
    for s in range(n_seq):
        seq = Sequence(np.random.default_rng(100 + s))
        cfg = _cfg(False, True, **{"perception.track.passthrough_cold": False})
        p = Perception(cfg, SC, CAL)
        for f in range(1, 9):
            dets = p.detect(*seq.frame())
            if len({d.color for d in dets}) >= 3 and len(dets) >= 3:
                out.append(f)
                break
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(table(run(n)))
    print("frames to confirm (strict tracker, M=2 of 3):", confirm_latency(n))
