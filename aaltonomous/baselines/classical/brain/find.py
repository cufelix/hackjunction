"""find(): VLM grounding as a brain tool (behind brain.find, default off). Qwen3-VL on llama.cpp returns boxes for a
phrase; the policy maps the points back onto its own detections (Detection.uv), so the model only ever sees object ids.
Weak spots: imperative sentences (extract_queries takes the noun phrase) and aliases (red -> orange: both are offered)."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

BUNDLED_FIND_TOOL = Path(__file__).with_name("find_tool.py")   # the copy that ships in the brain sidecar

_VERB = re.compile(r"^(?:please\s+)?(?:put away|put|place|move|sort|take|drop|throw|stack|carry|bring|transfer|send|"
                   r"deposit|fetch|grab|pick up|pick)\s+", re.I)
_CUT = re.compile(r"\b(?:in|into|to|on|onto|inside|at|goes|go|belongs?|should|must|needs?|is|are)\b.*$", re.I)
_SPLIT = re.compile(r",|;|:|\band\b|\bthen\b|\bbut\b|\bwhile\b", re.I)
ALIASES = {"red": ("orange",), "orange": ("red",)}


def extract_queries(text, max_n=4):
    """Instruction -> short noun phrases to ground ("The object farthest from the robot goes in the right bin" ->
    ["the object farthest from the robot"]); colour aliases are added as extra variants. Closed set for the brain."""
    out = []
    for part in _SPLIT.split(str(text or "").strip().rstrip(".!")):
        p = _CUT.sub("", _VERB.sub("", part.strip())).strip(" .")
        if 1 <= len(p.split()) <= 8 and p.lower() not in ("bin", "the bin", "it", "them", "everything else", "the objects", "objects", "the items", "the pieces"):
            out.append(p)
    for q in list(out):
        for w, alts in ALIASES.items():
            if re.search(rf"\b{w}\b", q, re.I):
                out += [re.sub(rf"\b{w}\b", a, q, flags=re.I) for a in alts]
    seen, res = set(), []
    for q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            res.append(q)
    return res[:max_n]


def match_found(points, dets, radius_px=40.0):
    """Image points -> indices of the detections whose uv is nearest (<= radius_px), one detection per point."""
    hit = []
    for (px, py) in points:
        best, bd = None, radius_px
        for i, d in enumerate(dets):
            dd = ((d.uv[0] - px) ** 2 + (d.uv[1] - py) ** 2) ** 0.5
            if dd <= bd:
                best, bd = i, dd
        if best is not None and best not in hit:
            hit.append(best)
    return hit


class FindClient:
    """Wraps /data/brain/find/find_tool.py (path configurable). __call__(rgb, query) -> [(x, y)] image points, [] on
    any failure. Needs cv2 for the PNG encode; both missing -> empty (the brain then just sees no hits)."""

    def __init__(self, path="/data/brain/find/find_tool.py", url=None, timeout=1.5):
        self.path, self.url, self.timeout, self._mod = path, url, timeout, None

    def _load(self):
        if self._mod is None:
            path = self.path if self.path and Path(self.path).is_file() else str(BUNDLED_FIND_TOOL)
            spec = importlib.util.spec_from_file_location("brain_find_tool", path)
            self._mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._mod)
        return self._mod

    def __call__(self, rgb, query):
        try:
            import cv2
            import numpy as np
            ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
            if not ok:
                return []
            kw = {"timeout": self.timeout}
            if self.url:
                kw["url"] = self.url
            return [tuple(r["point"]) for r in self._load().find(buf.tobytes(), query, **kw)]
        except Exception:
            return []

    def detect(self, rgb, query):
        """Like __call__ but keeps the boxes: [{"point": (x, y), "bbox": [x1, y1, x2, y2], "label": str}], [] on failure.
        brain.find_to_detection needs the box to clip the depth region grow."""
        try:
            import cv2
            import numpy as np
            ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
            if not ok:
                return []
            kw = {"timeout": self.timeout}
            if self.url:
                kw["url"] = self.url
            return [{"point": tuple(r["point"]), "bbox": list(r.get("bbox") or []), "label": str(r.get("label", ""))}
                    for r in self._load().find(buf.tobytes(), query, **kw)]
        except Exception:
            return []
