#!/usr/bin/env python3
"""robustness_table.py: markdown robustness table from adv-sim runs and Modal policy logs.

    robustness_table.py [--adv DIR ...] [--modal DIR ...] [--rows A,B,...]
                        [--cond-map COND=OVERLAY ...] [--out FILE]

--adv DIR    a run dir <profile>__<overlay>__<sha>__s<seed0>n<n>/ holding episodes.json (written by
             tools/adv/adv_report.py episodes), or a parent dir of such run dirs. Condition = profile.
--modal DIR  searched recursively for modal_summary.json (one per episode). Condition = nominal, unless
             --cond-map says otherwise. Overlay and sha come from the group name <overlay>_<sha>.
--rows       overlay names in order: the first is the base, the rest are candidates. A bare overlay name
             takes every sha of it; overlay@sha picks one. Default: every row found, sorted.
--cond-map   COND=OVERLAY: Modal runs of OVERLAY are condition COND, filed under the base row (the first
             --rows entry; the overlay itself when --rows is absent). E.g. inject_p7=ab_inj_p7.
--out        write the markdown there instead of stdout.

Episode score: the evaluator score; an episode that ended in a safety stop (end_reason safe_stop / aborted /
crash / policy_connection_lost) counts -6. Cell: mean ± population std over episodes, correct/objects,
wrong count, n episodes, safety stops when any. Paired: a candidate and the base in the same condition with
identical seed lists give Δ mean and Δ (mean - std), candidate minus base.
Stdlib only.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

SAFETY_SCORE = -6
STOP_REASONS = ("safe_stop", "aborted", "crash", "policy_connection_lost")
COND_ORDER = ["nominal", "real_mild", "real_hard", "out_of_range", "inject_p7"]
SHA_SUFFIX = re.compile(r"^(?P<overlay>.+)_(?P<sha>[0-9a-f]{6,40})$")


def split_group(group):
    """'cand_r16_nopp_9a28c41' -> ('cand_r16_nopp', '9a28c41'); no sha suffix -> (group, '')."""
    m = SHA_SUFFIX.match(group)
    return (m.group("overlay"), m.group("sha")) if m else (group, "")


def episode_score(score, stop):
    return SAFETY_SCORE if stop else score


def adv_run_dirs(paths):
    runs = []
    for p in paths:
        if not os.path.isdir(p):
            sys.exit(f"--adv: not a directory: {p}")
        if os.path.exists(os.path.join(p, "episodes.json")):
            runs.append(p)
        else:
            runs += sorted(os.path.dirname(f) for f in glob.glob(os.path.join(p, "*", "episodes.json")))
    return runs


def load_adv_run(run):
    """Episodes of one adv run; overlay, sha and condition (profile) come from the dir name."""
    parts = os.path.basename(os.path.normpath(run)).split("__")
    if len(parts) < 3:
        sys.exit(f"adv run dir name is not <profile>__<overlay>__<sha>__...: {run}")
    profile, overlay, sha = parts[0], parts[1], parts[2]
    with open(os.path.join(run, "episodes.json")) as f:
        data = json.load(f)
    eps = []
    for e in data["episodes"]:
        stop = bool(e.get("safety_stop"))
        eps.append(dict(overlay=overlay, sha=sha, cond=profile, seed=e["seed"],
                        score=episode_score(e["score"], stop), correct=e.get("correct") or 0,
                        objects=e.get("n_objects") or 0, wrong=e.get("wrong") or 0, stop=stop))
    return eps


def modal_summary_files(paths):
    out = []
    for p in paths:
        if not os.path.isdir(p):
            sys.exit(f"--modal: not a directory: {p}")
        for root, _dirs, names in os.walk(p):
            out += [os.path.join(root, n) for n in names if n == "modal_summary.json"]  # skips ._ AppleDouble
    return sorted(out)


def load_modal_episode(path, cond_map, base):
    with open(path) as f:
        e = json.load(f)["eval_summary"]
    if e.get("trials") != 1:
        sys.exit(f"{path}: expected one trial per episode, got {e.get('trials')}")
    overlay, sha = split_group(e["team"])
    stop = any(r in STOP_REASONS for r in (e.get("end_reasons") or {}))
    oc = e.get("outcome_counts") or {}
    if overlay in cond_map:
        cond, row, sha = cond_map[overlay], (base or overlay), ""
    else:
        cond, row = "nominal", overlay
    return dict(overlay=row, sha=sha, cond=cond, seed=e["seeds"][0],
                score=episode_score(e["score_mean"], stop), correct=e["objects_correct"],
                objects=e["objects"], wrong=oc.get("wrong", 0), stop=stop)


def gather(adv_paths, modal_paths, cond_map, base):
    eps = []
    runs = adv_run_dirs(adv_paths)
    for run in runs:
        eps += load_adv_run(run)
    modal_files = modal_summary_files(modal_paths)
    for f in modal_files:
        eps.append(load_modal_episode(f, cond_map, base))
    print(f"{len(eps)} episodes: {len(runs)} adv runs, {len(modal_files)} modal episodes", file=sys.stderr)
    return eps


def assign_row_keys(eps):
    """Row = overlay; overlay@sha only when one overlay has more than one sha in the data."""
    shas = {}
    for e in eps:
        if e["sha"]:
            shas.setdefault(e["overlay"], set()).add(e["sha"])
    for e in eps:
        split = e["sha"] and len(shas.get(e["overlay"], ())) > 1
        e["row"] = f"{e['overlay']}@{e['sha']}" if split else e["overlay"]


def resolve_rows(names, all_rows):
    if not names:
        return sorted(all_rows)
    out = []
    for n in names:
        hits = sorted(r for r in all_rows if r == n or r.split("@")[0] == n)
        if not hits:
            sys.exit(f"--rows: no runs for {n}")
        out += [h for h in hits if h not in out]
    return out


def summarize(eps):
    s = [e["score"] for e in eps]
    m = st.mean(s)
    sd = st.pstdev(s) if len(s) > 1 else 0.0
    return dict(n=len(s), mean=m, std=sd, ms=m - sd,
                correct=sum(e["correct"] for e in eps), objects=sum(e["objects"] for e in eps),
                wrong=sum(e["wrong"] for e in eps), stops=sum(e["stop"] for e in eps),
                seeds=sorted(e["seed"] for e in eps))


def fmt_cell(s):
    if s is None:
        return "–"
    txt = (f"{s['mean']:+.2f} ± {s['std']:.2f} ({s['correct']}/{s['objects']} obj, "
           f"{s['wrong']} wrong, n={s['n']}")
    if s["stops"]:
        txt += f", {s['stops']} stop"
    return txt + ")"


def render(rows, conds, stats):
    base = rows[0]
    lines = ["| overlay | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for r in rows:
        lines.append(f"| {r} | " + " | ".join(fmt_cell(stats.get((r, c))) for c in conds) + " |")
    lines += ["", "Cell: mean score per episode ± population std · correct/objects · wrong · n episodes "
              "· safety stops (scored -6)."]
    paired = []
    for cand in rows[1:]:
        for c in conds:
            b, a = stats.get((base, c)), stats.get((cand, c))
            if not b or not a:
                continue
            if b["seeds"] != a["seeds"]:
                paired.append(f"| {cand} | {c} | {a['n']} vs {b['n']} | seeds differ, not paired | |")
            else:
                paired.append(f"| {cand} | {c} | {a['n']} | {a['mean'] - b['mean']:+.2f} "
                              f"| {a['ms'] - b['ms']:+.2f} |")
    if paired:
        lines += ["", f"### Paired vs base `{base}` (same seeds, candidate − base)", "",
                  "| candidate | condition | n | Δ mean | Δ (mean − std) |", "|---|---|---|---|---|"] + paired
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Markdown robustness table (adv-sim + Modal).")
    ap.add_argument("--adv", action="append", default=[], metavar="DIR")
    ap.add_argument("--modal", action="append", default=[], metavar="DIR")
    ap.add_argument("--rows", default="", help="comma-separated overlays, base first")
    ap.add_argument("--cond-map", action="append", default=[], metavar="COND=OVERLAY")
    ap.add_argument("--out", metavar="FILE")
    a = ap.parse_args(argv)
    if not a.adv and not a.modal:
        ap.error("give --adv and/or --modal")
    names = [x.strip() for x in a.rows.split(",") if x.strip()]
    cond_map = {}
    for item in a.cond_map:
        if "=" not in item:
            ap.error(f"--cond-map wants COND=OVERLAY, got {item!r}")
        cond, ov = item.split("=", 1)
        cond_map[ov.strip()] = cond.strip()
    base = names[0].split("@")[0] if names else None
    eps = gather(a.adv, a.modal, cond_map, base)
    if not eps:
        sys.exit("no episodes found")
    assign_row_keys(eps)
    rows = resolve_rows(names, {e["row"] for e in eps})
    eps = [e for e in eps if e["row"] in rows]
    groups = {}
    for e in eps:
        groups.setdefault((e["row"], e["cond"]), []).append(e)
    stats = {k: summarize(v) for k, v in groups.items()}
    conds_seen = {c for _, c in groups}
    conds = [c for c in COND_ORDER if c in conds_seen] + sorted(c for c in conds_seen if c not in COND_ORDER)
    text = render(rows, conds, stats)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main(sys.argv[1:])
