"""aggregate.py <runs_dir> [arm labels as ARM=Label,...] -> markdown tables of our novel-object eval (stdout)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

runs = Path(sys.argv[1])
labels = dict(kv.split("=", 1) for kv in sys.argv[2].split(",")) if len(sys.argv) > 2 else {}
rows = defaultdict(list)
for f in sorted(runs.glob("*/results.jsonl")):
    for line in f.read_text().splitlines():
        t = json.loads(line)
        t["arm"] = f.parent.name
        rows[(t["scene"], t["arm"])].append(t)
scenes = sorted({k[0] for k in rows})
arms = sorted({k[1] for k in rows})


def fmt(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


print("| scene | arm | seeds | targets correct / wrong / left / off | distractors kept / disturbed | all-correct eps | clips | min cmd Hz | min policy Hz | mean sim ep s | mean wall ep s | end reasons |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
tot = defaultdict(lambda: defaultdict(int))
for s in scenes:
    for a in arms:
        ts = rows.get((s, a))
        if not ts:
            continue
        tg = {k: sum(t["target"][k] for t in ts) for k in ("correct", "wrong", "left", "off")}
        nt = sum(t["n_target"] for t in ts)
        dk = sum(t["distractors"]["kept"] for t in ts)
        dd = sum(t["distractors"]["disturbed"] for t in ts)
        er = defaultdict(int)
        for t in ts:
            er[t["end_reason"]] += 1
        hz = [t["command_hz"] for t in ts if t.get("command_hz") is not None]
        rt = [t["policy_hz_rt"] for t in ts if t.get("policy_hz_rt") is not None]
        sims = [t["sim_s"] * t["rtf"] for t in ts if t.get("sim_s") is not None and t.get("rtf")]   # summary "wall_s" is WALL seconds; x rtf = sim seconds
        wall = [t["episode_wall_s"] for t in ts]
        clips = sum((t.get("clipped_steps") or 0) for t in ts)
        print(f"| {s} | {labels.get(a, a)} | {len(ts)} | {tg['correct']}/{nt} / {tg['wrong']} / {tg['left']} / {tg['off']} | "
              f"{dk} / {dd} | {sum(t['all_correct'] for t in ts)}/{len(ts)} | {clips} | {fmt(min(hz) if hz else None)} | "
              f"{fmt(min(rt) if rt else None)} | {fmt(sum(sims) / len(sims) if sims else None)} | {fmt(sum(wall) / len(wall))} | "
              f"{', '.join(f'{k}:{v}' for k, v in sorted(er.items()))} |")
        for k in ("correct", "wrong", "left", "off"):
            tot[a][k] += tg[k]
        tot[a]["targets"] += nt
        tot[a]["dk"] += dk
        tot[a]["dd"] += dd
        tot[a]["eps"] += len(ts)
        tot[a]["allc"] += sum(t["all_correct"] for t in ts)
print("\n| arm | target objects correct | wrong | left on counter | off | distractors disturbed | all-correct episodes |")
print("|---|---|---|---|---|---|---|")
for a in arms:
    t = tot[a]
    print(f"| {labels.get(a, a)} | {t['correct']}/{t['targets']} | {t['wrong']} | {t['left']} | {t['off']} | {t['dd']}/{t['dk'] + t['dd']} | {t['allc']}/{t['eps']} |")
