"""find(image, description, url) -> [{bbox,point,label,confidence}] via Qwen3-VL on llama.cpp.
Qwen3-VL emits bbox_2d NORMALISED to 0-1000 (verified: GT orange cube 440,310 px -> centre ~(687,655))."""
import json, base64, re, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPT = ('Locate every instance that matches: {d}. Output bbox_2d coordinates in JSON: '
          '[{{"bbox_2d":[x1,y1,x2,y2],"label":"..."}}]. If nothing matches, output [].')

def _png_size(b):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")
    return 640, 480

def _parse(txt):
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    s = m.group(1) if m else txt
    try:
        data = json.loads(s)
    except Exception:
        data = []
        for mm in re.finditer(r'"bbox_2d"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]'
                              r'(?:\s*,\s*"label"\s*:\s*"([^"]*)")?', s):
            data.append({"bbox_2d": [float(mm.group(i)) for i in range(1, 5)], "label": mm.group(5) or ""})
    if isinstance(data, dict):
        data = [data]
    return data if isinstance(data, list) else []

def find(image, description, url=URL, timeout=60, return_meta=False):
    """image: path or bytes (PNG/JPEG). Returns list of dicts, [] on any failure."""
    meta = {"latency": None, "raw": ""}
    try:
        b = open(image, "rb").read() if isinstance(image, str) else image
        w, h = _png_size(b)
        mime = "image/png" if b[:4] == b"\x89PNG" else "image/jpeg"
        body = {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(b).decode()}},
                    {"type": "text", "text": PROMPT.format(d=description)}]}],
                "temperature": 0, "max_tokens": 400}
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout))
        meta["latency"] = time.time() - t
        txt = r["choices"][0]["message"]["content"]
        meta["raw"] = txt
        out = []
        n = 0
        for it in _parse(txt):
            try:
                x1, y1, x2, y2 = [float(v) for v in it["bbox_2d"]]
            except Exception:
                continue
            x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
            bb = [x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000]
            bb = [max(0, min(w, bb[0])), max(0, min(h, bb[1])), max(0, min(w, bb[2])), max(0, min(h, bb[3]))]
            out.append({"bbox": [round(v, 1) for v in bb],
                        "point": [round((bb[0] + bb[2]) / 2, 1), round((bb[1] + bb[3]) / 2, 1)],
                        "label": str(it.get("label", "")),
                        # llama.cpp gives no per-box score; heuristic: 1.0 if single detection, else 1/n
                        "confidence": 0.0})
        for o in out:
            o["confidence"] = round(1.0 / len(out), 2) if out else 0.0
        res = out
    except Exception as e:
        meta["error"] = repr(e)
        res = []
    return (res, meta) if return_meta else res

if __name__ == "__main__":
    import sys
    print(json.dumps(find(sys.argv[1], sys.argv[2]), indent=1))
