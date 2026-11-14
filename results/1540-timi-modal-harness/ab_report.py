"""Summarise A/B arms (one Modal container per episode) against the box A STEP 0 classical baseline.

    python3 modal/ab_report.py modal/runs/<arm_dir> [...]          # markdown tables to stdout
    python3 modal/ab_report.py --baseline                          # the baseline itself, same tables

Per arm: score, correct/wrong/counter/off, per shape, cubes by yaw (flat |yaw mod 90| < 22.5 deg vs diagonal),
picks/episode (`execute` lines of the classical server), [grasp_diag] held=yes/no/n/a, gate, RTF.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "aalto-starter/work/step0/runs/classical1/step0_classical_settle3_sort_fixed_nominal_20261114_001120"
OUTS = ("correct", "wrong", "counter", "off")


def objects(trial):
    """(shape, yaw, outcome) per object: layout entries matched to outcome names obj_<color>_<shape>_<k>."""
    seen, by = collections.Counter(), {o["name"]: o["outcome"] for o in trial["outcomes"]}
    res = []
    for ob in trial["layout"]:
        key = (ob["color"], ob["shape"])
        name = f"obj_{ob['color']}_{ob['shape']}_{seen[key]}"
        seen[key] += 1
        res.append((ob["shape"], ob.get("yaw"), by.get(name, "?")))
    return res


EXEC = re.compile(r"\] execute (\w+)\s+(left|right)\s+arm .*?grasp=\(([-+\d.]+),([-+\d.]+)")
CALIB = re.compile(r"head calibration (accepted|fallback)\S*:? .*?xy=\(([-+\d.]+), *([-+\d.]+)\)")


def arm_of_objects(trial, log):
    """Arm of the LAST pick of each layout object: execute lines matched to the nearest same-colour object
    (layout mat xy -> pelvis: x = 0.14 + y_mat, y = -0.45 + x_mat). None = never picked."""
    arms = [None] * len(trial["layout"])
    for col, side, gx, gy in EXEC.findall(log):
        cands = [(math.hypot(0.14 + o["y"] - float(gx), -0.45 + o["x"] - float(gy)), i)
                 for i, o in enumerate(trial["layout"]) if o["color"] == col]
        if cands:
            arms[min(cands)[1]] = side
    return arms


def cube_class(yaw):
    d = math.degrees(yaw) % 90.0
    return "flat" if min(d, 90 - d) < 22.5 else "diagonal"


def arm_trials(d: Path):
    """One container per episode: <ep_dir>/eval/*/eval.json + <ep_dir>/policy.log, aligned."""
    trials, logs = [], []
    for ej in sorted(d.rglob("eval.json")):
        tr = json.load(open(ej))["trials"]
        pl = ej.parents[2] / "policy.log"
        trials += tr
        logs += [pl.read_text(errors="ignore") if pl.exists() else ""] * len(tr)
    return trials, logs


def summarize(name, trials, logs):
    oc, shape, cube = collections.Counter(), collections.defaultdict(collections.Counter), \
        collections.defaultdict(collections.Counter)
    armshape = collections.defaultdict(collections.Counter)
    sidearm = collections.defaultdict(collections.Counter)
    single_log = len(logs) == 1 and len(trials) > 1      # baseline: one server log for all episodes
    for i, t in enumerate(trials):
        lg = "" if single_log else (logs[i] if i < len(logs) else "")
        arms = arm_of_objects(t, lg) if lg else [None] * len(t["layout"])
        for (sh, yaw, o), arm, ob in zip(objects(t), arms, t["layout"]):
            armshape[(arm or "none", sh)][o] += 1
            py = -0.45 + ob["x"]                                   # spawn y in the pelvis frame (left +)
            side = "left" if py > 0.05 else ("right" if py < -0.05 else "middle")
            sidearm[(side, arm or "none", sh)][o] += 1
            oc[o] += 1
            shape[sh][o] += 1
            if sh == "cube" and yaw is not None:
                cube[cube_class(yaw)][o] += 1
    text = "\n".join(dict.fromkeys(logs)) if single_log else "\n".join(logs)
    calib = [(k, float(x), float(y)) for k, x, y in CALIB.findall(text)]
    picks = len(re.findall(r"\] execute ", text))
    held = collections.Counter(re.findall(r"\[grasp_diag\].*?held=(yes|no|n/a)", text))
    n = len(trials)
    flags = []
    for line in (logs[0].splitlines() if logs else []):
        if line.startswith("profile=") or line.startswith("  ON ") or re.match(r"^\d+ of \d+ flags on", line):
            flags.append(line.rstrip())
        if re.match(r"^\d+ of \d+ flags on", line):
            break
    return {"flags": flags, "calib": calib,
        "arm_shape": {f"{a}/{sh}": dict(c) for (a, sh), c in sorted(armshape.items())},
        "side_arm_shape": {f"{sd}/{a}/{sh}": dict(c) for (sd, a, sh), c in sorted(sidearm.items())},
        "name": name, "episodes": n, "seeds": sorted(t["seed"] for t in trials),
        "score": oc["correct"], "objects": sum(oc.values()), "outcomes": dict(oc),
        "shape": {k: dict(v) for k, v in shape.items()}, "cube_yaw": {k: dict(v) for k, v in cube.items()},
        "picks": picks, "picks_per_ep": picks / n if n and logs else None, "held": dict(held),
        "gate_fail": [t["seed"] for t in trials if not t.get("gate_ok", True)],
        "clips": sum(t.get("safety", {}).get("clipped_steps", 0) for t in trials),
        "hz_min": min((t.get("command_hz", 0) for t in trials), default=None),
        "rtf_mean": sum(t.get("rtf", 0) for t in trials) / n if n else None,
        "end": dict(collections.Counter(t.get("end_reason") for t in trials)),
    }


def frac(c: dict):
    n = sum(c.values())
    return f"{c.get('correct', 0)}/{n}" if n else "-"


def table(rows):
    out = ["| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | "
           "held yes/no/n.a. | gate fails | clips | Hz min | RTF |", "|" + "---|" * 15]
    for r in rows:
        o = r["outcomes"]
        h = r["held"]
        out.append(
            f"| {r['name']} | {r['episodes']} | **{r['score']}/{r['objects']}** | "
            f"{'/'.join(str(o.get(k, 0)) for k in OUTS)} | {frac(r['shape'].get('cube', {}))} | "
            f"{frac(r['shape'].get('cylinder', {}))} | {frac(r['shape'].get('triangle', {}))} | "
            f"{frac(r['cube_yaw'].get('flat', {}))} | {frac(r['cube_yaw'].get('diagonal', {}))} | "
            f"{r['picks_per_ep']:.2f} | " if r["picks_per_ep"] is not None else "- | ")
        out[-1] += (f"{h.get('yes', 0)}/{h.get('no', 0)}/{h.get('n/a', 0)} | {len(r['gate_fail'])} | {r['clips']} | "
                    f"{r['hz_min']:.2f} | {r['rtf_mean']:.3f} |")
    return "\n".join(out)


def baseline():
    trials, _ = arm_trials(BASELINE)
    trials = [t for t in trials if 1000 <= t["seed"] <= 1019]
    pol = sorted(glob.glob(str(BASELINE.parent / "policy_server*.log")))
    # policy_server_full.log covers sort_fixed then sort_language (started 01:09:14): keep the sort_fixed part
    lines = Path(pol[0]).read_text(errors="ignore").splitlines() if pol else []
    per_seed, cur = collections.defaultdict(list), None
    for l in lines:
        if l[:19] >= "2026-11-14 01:09:14":
            break
        m = re.search(r"reset task=\S+ seed=(\d+)", l)
        if m:
            cur = int(m.group(1))
        if cur is not None:
            per_seed[cur].append(l)
    logs = ["\n".join(per_seed.get(t["seed"], [])) for t in trials]
    return summarize("baseline box A (flags off)", trials, logs)


if __name__ == "__main__":
    rows = [baseline()]
    for d in sys.argv[1:]:
        if d.startswith("--"):
            continue
        tr, lg = arm_trials(Path(d))
        rows.append(summarize(Path(d).name, tr, lg))
    print(table(rows))
    print("\n| arm | " + " | ".join(f"{a} {sh}" for a in ("left", "right") for sh in ("cube", "cylinder", "triangle"))
          + " | never picked |")
    print("|" + "---|" * 8)
    for r in rows:
        cells = [frac(r["arm_shape"].get(f"{a}/{sh}", {})) for a in ("left", "right")
                 for sh in ("cube", "cylinder", "triangle")]
        none = sum(sum(v.values()) for k, v in r["arm_shape"].items() if k.startswith("none/"))
        print(f"| {r['name']} | " + " | ".join(cells) + f" | {none} |")
    print("\nSpawn-y side (pelvis y: left > +5 cm, right < -5 cm) x picking arm, correct/n per shape:")
    print("\n| arm run | spawn side | picked by | cube | cylinder | triangle | all |")
    print("|" + "---|" * 7)
    for r in rows:
        for sd in ("left", "middle", "right"):
            for a in ("left", "right", "none"):
                cells = [r["side_arm_shape"].get(f"{sd}/{a}/{sh}", {}) for sh in ("cube", "cylinder", "triangle")]
                tot = collections.Counter()
                for c in cells:
                    tot.update(c)
                if sum(tot.values()):
                    print(f"| {r['name']} | {sd} | {a} | " + " | ".join(frac(c) for c in cells) + f" | {frac(tot)} |")
    for r in rows[1:]:
        print(f"\n{r['name']} flags (tools/flags.py, first episode):\n" + "\n".join(r["flags"]))
        if r["calib"]:
            xs = sorted(c[1] for c in r["calib"]); ys = sorted(c[2] for c in r["calib"])
            kinds = collections.Counter(c[0] for c in r["calib"])
            print(f"head calibration: {dict(kinds)}; dx median {xs[len(xs)//2]:+.1f} [{xs[0]:+.1f}, {xs[-1]:+.1f}] mm, "
                  f"dy median {ys[len(ys)//2]:+.1f} [{ys[0]:+.1f}, {ys[-1]:+.1f}] mm")
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
