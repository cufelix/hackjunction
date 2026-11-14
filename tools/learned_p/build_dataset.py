#!/usr/bin/env python3
"""Build one CSV row per executed pick attempt from classical policy.log files.

New runs with ``decision.learned_p`` enabled contain an exact ``[learned_p]``
feature line.  Older runs are supported by joining grasp_posture, grasp_diag,
confidence_order, speed_tier and classical execute/verification lines.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
EXECUTE = re.compile(
    rf"\[classical t=\s*({FLOAT})\]\s+execute\s+(\w+)\s+(left|right)\s+arm -> (\w+) bin\s+"
    rf"grasp=\(({FLOAT}),({FLOAT}),({FLOAT})\).*?\s({FLOAT})s\s+safety_clips=")
POSTURE = re.compile(rf"\[grasp_posture\] t=({FLOAT}) (\w+) (left|right) label=([^ ]+)")
DIAG = re.compile(rf"\[grasp_diag\] t=({FLOAT}) (\w+) (left|right) target=(\w+) held=(yes|no|n/a)")
OUTCOME = re.compile(r"\[speed_tier\] (\w+) (gone: clean|still on the counter: failure)")
LEARNED = re.compile(r"\[learned_p\]\s+(.*)")
PAIR = re.compile(r"([A-Za-z_]+)=([^ ]+)")
PRIOR_ITEM = re.compile(r"(\w+)/([^ ]+) p=(\d+(?:\.\d+)?)")
RESET = re.compile(r"reset .*?seed=([^ ]+)")

FIELDS = [
    "episode", "source_log", "attempt_id", "shape", "colour", "hand", "target_bin", "attempt_index",
    "pelvis_x", "pelvis_y", "distance", "nearest_neighbor_distance", "time_remaining", "p_prior",
    "p_prior_source", "duration_s", "label", "outcome_source",
]


def _number(value):
    if value in (None, "", "na", "n/a", "None", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prior(shape, x, y, target):
    shape_p = {"cube": 0.9, "cylinder": 0.8, "triangle": 0.6}.get(shape, 0.75)
    radius = math.hypot(x, y)
    margin = 1.0 - 0.5 * min(max((radius - 0.35) / 0.10, 0.0), 1.0)
    bin_y = {"left": 0.21, "center": 0.0, "centre": 0.0, "right": -0.21}.get(target, 0.0)
    if abs(bin_y) > 0.1 and abs(y) > 0.05 and bin_y * y < 0.0:
        margin *= 0.75
    return shape_p * margin


def _find_logs(paths):
    out = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name == "policy.log":
            out.append(path)
        elif path.is_dir():
            out.extend(path.rglob("policy.log"))
        else:
            raise FileNotFoundError(path)
    return sorted(set(p.resolve() for p in out))


def _closest(attempts, t, colour, unresolved=False):
    candidates = [a for a in attempts if a["colour"] == colour and (not unresolved or a.get("label") is None)]
    return min(candidates, key=lambda a: abs(a["t"] - t), default=None)


def parse_log(path, time_limit=120.0, association_radius=0.06):
    attempts, learned, postures = [], [], []
    prior_items, episode, episode_start = [], path.parent.name, 0
    for line in path.read_text(errors="replace").splitlines():
        match = RESET.search(line)
        if match:
            episode = f"{path.parent.name}:seed={match.group(1)}"
            episode_start = len(attempts)
            learned, postures, prior_items = [], [], []
        match = LEARNED.search(line)
        if match:
            data = {k: v for k, v in PAIR.findall(match.group(1))}
            learned.append(data)
            continue
        match = POSTURE.search(line)
        if match:
            postures.append({"t": float(match.group(1)), "colour": match.group(2), "hand": match.group(3),
                             "shape": match.group(4) if match.group(4) not in ("None", "none") else "unknown"})
            continue
        if "[confidence_order]" in line:
            prior_items = [(c, s, float(p)) for c, s, p in PRIOR_ITEM.findall(line)]
            continue
        match = EXECUTE.search(line)
        if match:
            t, colour, hand, target, x, y, _z, duration = match.groups()
            t, x, y, duration = map(float, (t, x, y, duration))
            posture = min((p for p in postures if p["colour"] == colour),
                          key=lambda p: abs(p["t"] - t), default=None)
            if posture is not None and abs(posture["t"] - t) > 0.25:
                posture = None
            exact = min((p for p in learned if p.get("colour") == colour and _number(p.get("t")) is not None),
                        key=lambda p: abs(_number(p.get("t")) - t), default=None)
            if exact is not None and abs(_number(exact.get("t")) - t) > 0.25:
                exact = None
            shape = (exact or {}).get("shape") or (posture or {}).get("shape") or "unknown"
            previous = [a for a in attempts[episode_start:] if a["colour"] == colour
                        and math.hypot(a["pelvis_x"] - x, a["pelvis_y"] - y) <= association_radius]
            attempt_index = int(_number((exact or {}).get("attempt")) or (max(
                (a["attempt_index"] for a in previous), default=0) + 1))
            prior = _number((exact or {}).get("p_prior"))
            prior_source = "learned_log" if prior is not None else None
            if prior is None:
                matches = [item for item in prior_items if item[:2] == (colour, shape)]
                first_matches = bool(prior_items and prior_items[0][:2] == (colour, shape))
                if first_matches or len(matches) == 1:
                    prior, prior_source = matches[0][2], "confidence_order"
                else:
                    # Old non-decision runs lack the perception terms needed to recreate p exactly.
                    # Keep the historical shape/reach prior, but identify it as reconstructed.
                    prior, prior_source = _prior(shape, x, y, target), "reconstructed"
            preferred_hand = "left" if target == "left" or (target in ("center", "centre") and y > 0.0) else "right"
            attempts.append({
                "episode": episode, "source_log": str(path), "attempt_id": len(attempts) + 1, "t": t,
                "shape": shape, "colour": colour, "hand": (exact or {}).get("hand", preferred_hand),
                "target_bin": (exact or {}).get("target_bin", target), "attempt_index": attempt_index,
                "actual_target": (exact or {}).get("actual_target", target),
                "pelvis_x": _number((exact or {}).get("x")) or x,
                "pelvis_y": _number((exact or {}).get("y")) if _number((exact or {}).get("y")) is not None else y,
                "distance": _number((exact or {}).get("distance")) or math.hypot(x, y),
                "nearest_neighbor_distance": _number((exact or {}).get("nn")),
                "time_remaining": _number((exact or {}).get("time_remaining")) or max(0.0, time_limit - t),
                "p_prior": prior, "p_prior_source": prior_source,
                "duration_s": duration, "label": None, "outcome_source": "",
            })
            continue
        match = DIAG.search(line)
        if match:
            t, colour, _hand, _target, held = match.groups()
            attempt = _closest(attempts[episode_start:], float(t), colour)
            if attempt is not None and abs(attempt["t"] - float(t)) <= 0.25 and held in ("yes", "no"):
                attempt["held"] = held
            continue
        match = OUTCOME.search(line)
        if match:
            colour, status = match.groups()
            attempt = next((a for a in reversed(attempts[episode_start:])
                            if a["colour"] == colour and not a.get("authoritative")), None)
            if attempt is not None:
                attempt["label"] = int(status == "gone: clean")
                attempt["outcome_source"] = "still_on_table" if not attempt["label"] else "gone_clean"
                attempt["authoritative"] = True
            continue
        match = re.search(r"verified (\w+) in (\w+) slot", line)
        if match:
            attempt = next((a for a in reversed(attempts[episode_start:]) if a["colour"] == match.group(1)), None)
            if attempt is not None:
                attempt.update(label=1, outcome_source="verified_place", authoritative=True)
            continue
        match = re.search(r"release of (\w+) not verified", line)
        if match:
            attempt = next((a for a in reversed(attempts[episode_start:]) if a["colour"] == match.group(1)), None)
            if attempt is not None:
                attempt.update(label=0, outcome_source="not_verified", authoritative=True)

    # Older logs have no placement verifier. held=no is a definite miss; held=yes is the best available success
    # signal because the logged target is the resolved correct bin. Without diagnostics, a nearby retry means miss.
    for i, attempt in enumerate(attempts):
        retry = any(b["episode"] == attempt["episode"] and b["colour"] == attempt["colour"] and
                    math.hypot(b["pelvis_x"] - attempt["pelvis_x"], b["pelvis_y"] - attempt["pelvis_y"])
                    <= association_radius for b in attempts[i + 1:])
        if attempt["actual_target"] in ("counter", "unknown", "abstain"):
            attempt.update(label=0, outcome_source="non_target_drop")
        elif attempt["label"] is None and attempt.get("held") == "no":
            attempt.update(label=0, outcome_source="grasp_diag_empty")
        elif attempt["label"] is None and retry:
            attempt.update(label=0, outcome_source="nearby_retry")
        elif attempt["label"] is None and attempt.get("held") == "yes":
            attempt.update(label=1, outcome_source="grasp_diag_held")
        elif attempt["label"] is None:
            attempt.update(label=int(not retry), outcome_source="nearby_retry" if retry else "inferred_disappearance")
        attempt.pop("t", None)
        attempt.pop("held", None)
        attempt.pop("authoritative", None)
        attempt.pop("actual_target", None)
    return attempts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="run directories or individual policy.log files")
    parser.add_argument("-o", "--output", required=True, help="output CSV")
    parser.add_argument("--time-limit", type=float, default=120.0, help="episode limit when absent from logs")
    args = parser.parse_args(argv)
    logs = _find_logs(args.paths)
    rows = [row for log in logs for row in parse_log(log, args.time_limit)]
    if not rows:
        parser.error("no pick attempts found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in FIELDS} for row in rows)
    positives = sum(row["label"] for row in rows)
    print(f"wrote {len(rows)} attempts from {len(logs)} episodes to {output} ({positives} success, {len(rows)-positives} failure)")


if __name__ == "__main__":
    main()
