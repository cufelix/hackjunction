"""Debug-only frame dump (debug.frame_dump_dir; off by default). Every `every`-th tick the incoming head RGB (and the
wrist RGBs if present) is saved as an annotated JPEG: detected blobs (colour + shape), planner state / plan phase and
the last action. Annotation + JPEG encoding run on a worker thread (bounded queue, frames dropped when it is full), so
act() only copies the frame. Never enabled in an overlay / submission."""
from __future__ import annotations

import os
import queue
import threading

import numpy as np

_DRAW = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 120, 0), "yellow": (0, 220, 255)}


class FrameDumper:
    def __init__(self, out_dir, every=2):
        import cv2  # only imported when the flag is on
        self.cv2, self.dir, self.every = cv2, str(out_dir), max(1, int(every))
        os.makedirs(self.dir, exist_ok=True)
        self.q = queue.Queue(maxsize=8)
        self.n = 0
        self.ep = 0
        threading.Thread(target=self._work, daemon=True).start()

    def new_episode(self):
        self.ep += 1
        self.n = 0

    def tick(self, obs, pol):
        self.n += 1
        if (self.n - 1) % self.every or obs.get("head_rgb") is None:
            return
        try:
            ph = ""
            cur = getattr(pol, "_dbg_plan", None)
            if cur is not None and pol.state == "executing" and pol.queue_a is not None:
                plan, n = cur
                k = n - len(pol.queue_a)
                ph = next((nm for nm, s, e in (plan.phases or []) if s <= k < e), "")
                ph = f"{plan.color}/{getattr(plan, 'target_bin', '')} {ph}"
            dets = [(d.color, d.shape, tuple(d.uv)) for d in (pol.last_detections or [])]
            hand = float(np.mean(pol.hand_cmd)) if pol.hand_cmd is not None else 0.0
            arm = None if pol.arm_cmd is None else np.round(np.asarray(pol.arm_cmd)[[0, 1, 3, 7, 8, 10]], 2).tolist()
            imgs = {"head": np.array(obs["head_rgb"], copy=True)}
            for k in ("left_wrist_rgb", "right_wrist_rgb"):
                if obs.get(k) is not None:
                    imgs[k.split("_")[0] + "_wrist"] = np.array(obs[k], copy=True)
            self.q.put_nowait((self.ep, self.n, float(obs.get("t", 0.0)), pol.state, ph, dets, hand, arm, imgs))
        except queue.Full:
            pass
        except Exception:
            pass  # debug aid: never disturb the control loop

    def _work(self):
        cv2 = self.cv2
        while True:
            ep, n, t, state, ph, dets, hand, arm, imgs = self.q.get()
            try:
                for name, rgb in imgs.items():
                    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if name == "head":
                        for c, shp, (u, v) in dets:
                            p = (int(u), int(v))
                            col = _DRAW.get(c, (255, 255, 255))
                            cv2.circle(im, p, 14, col, 2)
                            cv2.putText(im, f"{c}/{shp or '?'}", (p[0] + 16, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 2)
                            cv2.putText(im, f"{c}/{shp or '?'}", (p[0] + 16, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        col, 1)
                        lines = [f"t={t:5.1f} tick={n} state={state}", f"plan: {ph or '-'}",
                                 f"hand={hand:+.2f} arm={arm}"]
                        for i, s in enumerate(lines):
                            cv2.rectangle(im, (0, 4 + 20 * i), (8 * len(s) + 10, 22 + 20 * i), (0, 0, 0), -1)
                            cv2.putText(im, s, (4, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imwrite(os.path.join(self.dir, f"e{ep:02d}_{n:05d}_{name}.jpg"), im,
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
            except Exception:
                pass
