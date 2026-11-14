"""funnel.py <runs_dir> : per (scene, arm) root-cause summary from policy logs + results.jsonl (stdout markdown)."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

runs = Path(sys.argv[1])
rows = defaultdict(list)
for arm_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
    rj = arm_dir / "results.jsonl"
    if not rj.exists():
        continue
    for line in rj.read_text().splitlines():
        t = json.loads(line)
        log = arm_dir / f"{t['scene']}_s{t['seed']}_policy.log"
        txt = log.read_text() if log.exists() else ""
        det = re.findall(r"nothing left to sort \((\d+) detections\)", txt)
        ex = re.findall(r"execute (\w+)\s+(\w+)\s+arm -> (\w+) bin", txt)
        held = len(re.findall(r"held_abort", txt))
        miss = Counter(re.findall(r"bin_lookup\] miss color=(\w+) shape=(\w+)", txt))
        nov = re.findall(r"\[novel\] matched ([\w -]+) -> (\w+)", txt)
        skip = re.findall(r"skip (\w+)\s.*?\[(\w+)\]: (.{0,90})", txt)
        t["_f"] = dict(det=int(det[-1]) if det else None, ex=len(ex), held=held, miss=sum(miss.values()), nov=nov, skip=skip,
                       ex_bins=[e[2] for e in ex])
        rows[(arm_dir.name, t["scene"])].append(t)
print("| arm | scene | detections at last look | picks attempted | grasp aborted (close sample empty) | novel matches | skips (reason) | bin_lookup misses | dominant cause |")
print("|---|---|---|---|---|---|---|---|---|")
for (arm, scene), ts in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    f = [t["_f"] for t in ts]
    ex, held = sum(x["ex"] for x in f), sum(x["held"] for x in f)
    nov = sorted({f"{a}->{b}" for x in f for a, b in x["nov"]})
    sk = Counter(f"{c}:{r[:60]}" for x in f for _, c, r in x["skip"])
    miss = sum(x["miss"] for x in f)
    corr = sum(t["target"]["correct"] for t in ts)
    if ex == 0 and miss and not nov:
        cause = "instruction not grounded to any visible object (no bin) -> abstain"
    elif ex == 0 and nov and sk:
        cause = "novel match skipped by grasp/height gate"
    elif ex == 0 and miss == 0:
        cause = "perception: nothing detected"
    elif ex and held >= ex * 0.6 and corr == 0:
        cause = "grasp failed in sim (hand closed on nothing)"
    elif ex:
        cause = "picks attempted; mixed"
    else:
        cause = "?"
    print(f"| {arm} | {scene} | {sorted({x['det'] for x in f if x['det'] is not None})} | {ex} | {held} | {', '.join(nov) or '-'} | "
          f"{'; '.join(f'{k} x{v}' for k, v in sk.most_common(2)) or '-'} | {miss} | {cause} |")
