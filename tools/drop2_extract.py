"""drop_2 frame extraction: 3 frames per episode (20/50/80% of its video span) + episode manifest.
Run: venv/bin/python drop2_extract.py   (needs av, pyarrow; ~2-4 min for the full 14.5k-frame av1 decode)
"""
import json

import argparse

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default="/var/tmp/hf_drop2", help="dataset dir (meta/, videos/, frames/ out)")
ROOT = _ap.parse_args().root
OUT = f"{ROOT}/frames"
import os
os.makedirs(OUT, exist_ok=True)

tasks = pq.read_table(f"{ROOT}/meta/tasks.parquet").to_pydict()
task_of = dict(zip(tasks["task_index"], tasks["__index_level_0__"]))
ep = pq.read_table(f"{ROOT}/meta/episodes/chunk-000/file-000.parquet",
                   columns=["episode_index", "tasks", "length",
                            "videos/observation.images.cam_high/from_timestamp",
                            "videos/observation.images.cam_high/to_timestamp"]).to_pydict()

want = {}          # timestamp -> (episode, tag)
manifest = []
for i in range(len(ep["episode_index"])):
    e = ep["episode_index"][i]
    t0 = ep["videos/observation.images.cam_high/from_timestamp"][i]
    t1 = ep["videos/observation.images.cam_high/to_timestamp"][i]
    task = ep["tasks"][i][0] if isinstance(ep["tasks"][i], list) else ep["tasks"][i]
    manifest.append(dict(episode=e, length=ep["length"][i], task=task, t0=t0, t1=t1))
    for tag, f in (("a", 0.2), ("b", 0.5), ("c", 0.8), ("z", 0.97)):
        want[round(t0 + (t1 - t0) * f, 3)] = (e, tag)

keys = sorted(want)
c = av.open(f"{ROOT}/videos/observation.images.cam_high/chunk-000/file-000.mp4")
stream = c.streams.video[0]
time_base = float(stream.time_base)
next_k = 0
saved = 0
for frame in c.decode(video=0):
    t = float(frame.pts * time_base)
    while next_k < len(keys) and t >= keys[next_k] - 0.02:
        e, tag = want[keys[next_k]]
        name = f"e{e:02d}{tag}"
        cv2.imwrite(f"{OUT}/{name}.png", frame.to_ndarray(format="bgr24"))
        saved += 1
        next_k += 1
    if next_k >= len(keys):
        break
c.close()
json.dump(manifest, open(f"{ROOT}/manifest.json", "w"), indent=1)
print(f"saved {saved}/{len(keys)} frames to {OUT}")
by_task = {}
for m in manifest:
    by_task[m["task"]] = by_task.get(m["task"], 0) + 1
for t, n in by_task.items():
    print(f"  {n:2d} eps: {t}")
