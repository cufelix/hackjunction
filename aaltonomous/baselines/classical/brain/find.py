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


def downscale_rgb(rgb, max_side):
    """HxWx3 array -> (array, (orig_w, orig_h)). max_side <= 0 or already small enough: unchanged. Aspect kept (INTER_AREA).
    The model's boxes are normalised, so find_tool scales them back to orig_w/orig_h (original pixel coordinates)."""
    import numpy as np
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    m = int(max_side or 0)
    if m <= 0 or max(h, w) <= m:
        return a, (w, h)
    import cv2
    s = m / float(max(h, w))
    return cv2.resize(a, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA), (w, h)


class FindClient:
    """Wraps /data/brain/find/find_tool.py (path configurable). __call__(rgb, query) -> [(x, y)] image points, [] on
    any failure. Needs cv2 for the PNG encode; both missing -> empty (the brain then just sees no hits)."""

    def __init__(self, path="/data/brain/find/find_tool.py", url=None, timeout=1.5, max_side=0, n_predict=400,
                 layout="image_first"):
        self.path, self.url, self.timeout, self._mod = path, url, timeout, None
        self.max_side, self.n_predict, self.layout = int(max_side or 0), int(n_predict), layout

    def _load(self):
        if self._mod is None:
            path = self.path if self.path and Path(self.path).is_file() else str(BUNDLED_FIND_TOOL)
            spec = importlib.util.spec_from_file_location("brain_find_tool", path)
            self._mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._mod)
        return self._mod

    def _kw(self, small, osz):
        """Only non-default knobs are passed, so an older find_tool.py at find_path keeps working unchanged."""
        kw = {"timeout": self.timeout}
        if self.n_predict != 400:
            kw["n_predict"] = self.n_predict
        if self.layout != "image_first":
            kw["layout"] = self.layout
        if tuple(small.shape[:2][::-1]) != tuple(osz):
            kw["orig_size"] = osz
        return kw

    def __call__(self, rgb, query):
        try:
            import cv2
            import numpy as np
            small, osz = downscale_rgb(rgb, self.max_side)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR))
            if not ok:
                return []
            kw = self._kw(small, osz)
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
            small, osz = downscale_rgb(rgb, self.max_side)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR))
            if not ok:
                return []
            kw = self._kw(small, osz)
            if self.url:
                kw["url"] = self.url
            return [{"point": tuple(r["point"]), "bbox": list(r.get("bbox") or []), "label": str(r.get("label", ""))}
                    for r in self._load().find(buf.tobytes(), query, **kw)]
        except Exception:
            return []
