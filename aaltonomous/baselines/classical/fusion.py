"""Opt-in temporal perception helpers (all default OFF; see perception.depth_fusion / perception.track).

DepthFuser  ring buffer of the last N head-depth frames while the robot is still. fuse() replaces the depth only
            inside the given ROI mask (the colour blobs): per pixel the median over the valid samples, valid only if
            >= min_valid samples are valid AND >= min_valid of them lie within tol_mm of that median (flying edge
            pixels and one-off outliers are dropped to 0 = hole). Everything outside the ROI is left untouched, so
            the cost is O(N * roi_px), not O(N * 640*480).
Tracker     frame-to-frame association of detections (same colour, nearest neighbour within radius_m in pelvis xy),
            running-mean -> EMA position smoothing, and an M-of-N confirmation rule: an object is only reported
            after it was seen in >= confirm_m of the last window_n frames (a one-frame phantom never is).
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np


class DepthFuser:
    def __init__(self, cfg):
        self.n = max(2, int(cfg.get("frames", 3)))
        self.min_valid = max(1, int(cfg.get("min_valid", 2)))
        self.tol = float(cfg.get("tol_mm", 8.0))
        self.range = tuple(cfg.get("range_mm", (150, 2000)))
        self.buf = deque(maxlen=self.n - 1)      # past frames; the current frame is the n-th
        self.lock = threading.Lock()             # observe() may run detect (-> fuse) in a worker thread

    def reset(self):
        with self.lock:
            self.buf.clear()

    def push(self, depth):
        if depth is not None:
            with self.lock:
                if self.buf and self.buf[-1].shape != depth.shape:
                    self.buf.clear()
                self.buf.append(np.array(depth, copy=True))

    def fuse(self, depth, roi):
        """depth (H,W) current frame, roi (H,W) bool/uint8. Returns a new depth with fused values inside roi
        (the input is never modified). With no history the input is returned as is."""
        with self.lock:
            past = list(self.buf)
        if depth is None or not past or past[-1].shape != depth.shape:
            return depth
        flat = np.flatnonzero(roi)
        if len(flat) == 0:
            return depth
        S = np.stack([f.take(flat) for f in past] + [depth.take(flat)], axis=1).astype(np.int16)   # (K, n)
        lo, hi = self.range
        bad = (S < lo) | (S > hi)
        cnt = S.shape[1] - bad.sum(1, dtype=np.int8)
        S[bad] = 30000                                  # sorts last; |30000 - median| is never within tol
        S.sort(axis=1)                                  # valid samples first (ascending), then the sentinels
        k = np.arange(S.shape[0])
        med = (S[k, np.maximum(cnt - 1, 0) // 2].astype(np.int16) + S[k, np.minimum(cnt // 2, S.shape[1] - 1)]) // 2
        med = np.where(cnt > 0, med, 0).astype(np.int16)
        near = (np.abs(S - med[:, None]) <= self.tol).sum(1, dtype=np.int8)
        good = (cnt > 0) & (near >= min(self.min_valid, S.shape[1]))
        out = depth.copy()
        out.reshape(-1)[flat] = np.where(good, med, 0).astype(depth.dtype)
        return out


class Track:
    __slots__ = ("id", "color", "pos", "n", "hist", "det", "seen_now")

    def __init__(self, tid, det, window):
        self.id, self.color = tid, det.color
        self.pos = np.array(det.top, float)
        self.n = 1
        self.hist = deque([True], maxlen=window)
        self.det, self.seen_now = det, True


class Tracker:
    def __init__(self, cfg):
        self.radius = float(cfg.get("radius_m", 0.02))
        self.m = int(cfg.get("confirm_m", 2))
        self.window = max(self.m, int(cfg.get("window_n", 3)))
        self.alpha = float(cfg.get("alpha", 0.4))
        self.drop_after = int(cfg.get("drop_after", 2))           # consecutive misses before a track is forgotten
        self.passthrough = bool(cfg.get("passthrough_cold", True))  # fewer than m frames since reset -> cannot judge
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.tracks, self.frames, self.next_id = [], 0, 1
            self.misses = {}

    def update(self, dets):
        """One frame of detections in -> the confirmed ones (smoothed, with .track_id) out. Thread-safe."""
        with self.lock:
            return self._update(dets)

    def _update(self, dets):
        import copy
        self.frames += 1
        for t in self.tracks:
            t.seen_now = False
        cand = sorted(((np.linalg.norm(d.top[:2] - t.pos[:2]), i, j) for i, d in enumerate(dets)
                       for j, t in enumerate(self.tracks) if t.color == d.color), key=lambda x: x[0])
        used_d, used_t, match = set(), set(), {}
        for dist, i, j in cand:
            if dist > self.radius:
                break
            if i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            match[i] = j
        for i, j in match.items():
            t = self.tracks[j]
            t.n += 1
            t.pos += max(self.alpha, 1.0 / t.n) * (np.asarray(dets[i].top, float) - t.pos)
            t.hist.append(True)
            t.det, t.seen_now = dets[i], True
        for i, d in enumerate(dets):
            if i not in used_d:
                self.tracks.append(Track(self.next_id, d, self.window))
                self.next_id += 1
        keep = []
        for t in self.tracks:
            if not t.seen_now:
                t.hist.append(False)
            recent = list(t.hist)[-self.drop_after:]
            if t.seen_now or any(recent):
                keep.append(t)
        self.tracks = keep
        cold = self.passthrough and self.frames < self.m
        out = []
        for t in self.tracks:
            if not t.seen_now:
                continue
            if sum(t.hist) >= self.m or cold:
                d = copy.copy(t.det)
                d.top = t.pos.copy()
                d.track_id = t.id
                out.append(d)
        return out
