"""Tests for tools/robustness_table.py with synthetic adv and Modal fixtures in the real schemas."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robustness_table as rt  # noqa: E402


def adv_episode(seed, score, correct, wrong, n_objects, stop=False, end="done"):
    # Schema written by tools/adv/adv_report.py episodes (score is already -6 for a safety stop).
    return {"seed": seed, "episode": seed, "n_objects": n_objects, "correct": correct, "wrong": wrong,
            "counter": 0, "off": 0, "raw_score": score, "safety_stop": stop,
            "score": -6 if stop else score, "clips": 0, "command_hz": 30.0, "policy_hz_rt": 77.5,
            "end_reason": end, "sim_s": 160.0, "rtf": 0.3, "adv": None}


def write_adv(root, profile, overlay, sha, seed0, episodes):
    d = root / f"{profile}__{overlay}__{sha}__s{seed0}n{len(episodes)}"
    d.mkdir()
    (d / "episodes.json").write_text(json.dumps({"meta": {}, "stats": {}, "episodes": episodes}))
    return d


def write_modal(root, group, seed, score, correct, objects, wrong=0, end_reasons=None):
    # Schema of modal_summary.json (eval_summary block), one trial per episode directory.
    ep = root / group / f"{group}_s{seed}"
    ep.mkdir(parents=True)
    summary = {
        "eval_tail": ["  SIM GATE  PASS"],
        "eval_summary": {
            "policy": "policy-server", "team": group, "task": "sort_fixed", "variant": "nominal",
            "dr": None, "trials": 1, "seeds": [seed], "score_mean": float(score), "score_std": 0.0,
            "score_frac": 1.0, "objects": objects, "objects_correct": correct, "object_success": 1.0,
            "episodes_all_correct": 1, "outcome_counts": {"correct": correct, **({"wrong": wrong} if wrong else {})},
            "end_reasons": end_reasons or {"done": 1}, "safety_clipped_steps_total": 0,
            "safety_clipped_steps_max": 0, "effective_hz_min": 30.0, "command_hz_min": 29.98,
            "policy_hz_rt_min": 77.5, "clock": "sim", "rtf_min": 0.3, "rtf_mean": 0.3,
            "gate": {"passed": True, "failed_trials": [], "max_clips": 0, "min_hz": 27.0},
        },
        "timing": {"eval_s": 171.4, "eval_rc": 0},
    }
    (ep / "modal_summary.json").write_text(json.dumps(summary))
    # AppleDouble sidecar that must be ignored.
    (ep / "._modal_summary.json").write_text("not json")
    return ep


def test_adv_dir_name_and_stop_scoring(tmp_path):
    run = write_adv(tmp_path, "real_mild", "cand", "bbbbbbbb", 1000, [
        adv_episode(1000, 4, 4, 0, 4),
        adv_episode(1001, 5, 1, 0, 3, stop=True, end="safe_stop"),
    ])
    eps = rt.load_adv_run(str(run))
    assert [(e["overlay"], e["sha"], e["cond"]) for e in eps] == [("cand", "bbbbbbbb", "real_mild")] * 2
    assert [e["score"] for e in eps] == [4, -6]
    assert eps[1]["stop"] is True and eps[1]["correct"] == 1 and eps[1]["objects"] == 3


def test_modal_episode_schema_and_stop(tmp_path):
    ep = write_modal(tmp_path, "cand_r16_nopp_9a28c41", 1054, 3, 3, 3)
    e = rt.load_modal_episode(str(ep / "modal_summary.json"), {}, None)
    assert (e["overlay"], e["sha"], e["cond"], e["seed"]) == ("cand_r16_nopp", "9a28c41", "nominal", 1054)
    assert (e["score"], e["correct"], e["objects"], e["wrong"], e["stop"]) == (3, 3, 3, 0, False)
    ep2 = write_modal(tmp_path, "base_1234abc", 1000, 2, 1, 3, wrong=1, end_reasons={"safe_stop": 1})
    e2 = rt.load_modal_episode(str(ep2 / "modal_summary.json"), {}, None)
    assert e2["score"] == -6 and e2["stop"] is True and e2["wrong"] == 1


def test_modal_discovery_skips_appledouble(tmp_path):
    write_modal(tmp_path, "rc1_eb31647", 1000, 1, 1, 2)
    files = rt.modal_summary_files([str(tmp_path)])
    assert len(files) == 1 and files[0].endswith("modal_summary.json")
    assert not os.path.basename(files[0]).startswith("._")


def test_split_group():
    assert rt.split_group("cand_r16_nopp_9a28c41") == ("cand_r16_nopp", "9a28c41")
    assert rt.split_group("K1_baseline_36daf5b") == ("K1_baseline", "36daf5b")
    assert rt.split_group("rc1") == ("rc1", "")


def test_summarize_mean_std_and_totals():
    eps = [dict(score=3, correct=3, objects=3, wrong=0, stop=False, seed=1),
           dict(score=-6, correct=1, objects=3, wrong=1, stop=True, seed=2)]
    s = rt.summarize(eps)
    assert s["mean"] == pytest.approx(-1.5)
    assert s["std"] == pytest.approx(4.5)
    assert s["ms"] == pytest.approx(-6.0)
    assert (s["correct"], s["objects"], s["wrong"], s["stops"], s["n"]) == (4, 6, 1, 1, 2)
    assert rt.fmt_cell(s) == "-1.50 ± 4.50 (4/6 obj, 1 wrong, n=2, 1 stop)"
    assert rt.fmt_cell(None) == "–"


def test_row_keys_split_only_when_sha_differs():
    eps = [dict(overlay="cand", sha="aaaaaaaa"), dict(overlay="cand", sha="bbbbbbbb"),
           dict(overlay="base", sha="cccccccc"), dict(overlay="base", sha="")]
    rt.assign_row_keys(eps)
    assert [e["row"] for e in eps] == ["cand@aaaaaaaa", "cand@bbbbbbbb", "base", "base"]


def test_resolve_rows_bare_name_expands_and_unknown_fails():
    keys = {"cand@aaaaaaaa", "cand@bbbbbbbb", "base"}
    assert rt.resolve_rows(["base", "cand"], keys) == ["base", "cand@aaaaaaaa", "cand@bbbbbbbb"]
    assert rt.resolve_rows([], keys) == sorted(keys)
    with pytest.raises(SystemExit):
        rt.resolve_rows(["nope"], keys)


def test_cond_map_files_injected_run_under_base(tmp_path):
    adv = tmp_path / "adv"
    adv.mkdir()
    write_adv(adv, "nominal", "base", "aaaaaaaa", 1000, [adv_episode(1000, 2, 2, 0, 2)])
    modal = tmp_path / "modal"
    write_modal(modal, "ab_inj_p7_1234abc", 1000, 1, 2, 3, wrong=1)
    cond_map = {"ab_inj_p7": "inject_p7"}
    eps = rt.gather([str(adv)], [str(modal)], cond_map, "base")
    rt.assign_row_keys(eps)
    inj = [e for e in eps if e["cond"] == "inject_p7"]
    assert len(inj) == 1 and inj[0]["row"] == "base" and inj[0]["score"] == 1


def test_markdown_end_to_end_with_paired_deltas(tmp_path):
    adv = tmp_path / "adv"
    adv.mkdir()
    # base nominal: seeds 1000, 1001 -> scores 3 and a safety stop (-6)
    write_adv(adv, "nominal", "base", "aaaaaaaa", 1000,
              [adv_episode(1000, 3, 3, 0, 3), adv_episode(1001, -6, 1, 0, 3, stop=True, end="safe_stop")])
    # candidate nominal: same seeds -> scores 4 and 2
    write_adv(adv, "nominal", "cand", "bbbbbbbb", 1000,
              [adv_episode(1000, 4, 4, 0, 4), adv_episode(1001, 2, 2, 1, 3)])
    # seeds differ in real_mild -> not paired
    write_adv(adv, "real_mild", "base", "aaaaaaaa", 1000, [adv_episode(1000, 1, 1, 0, 2)])
    write_adv(adv, "real_mild", "cand", "bbbbbbbb", 1001, [adv_episode(1001, 1, 1, 0, 2)])
    # modal injected run filed under base
    modal = tmp_path / "modal"
    write_modal(modal, "ab_inj_p7_1234abc", 1000, 2, 3, 3)
    out = tmp_path / "table.md"
    rt.main(["--adv", str(adv), "--modal", str(modal), "--rows", "base,cand",
             "--cond-map", "inject_p7=ab_inj_p7", "--out", str(out)])
    md = out.read_text()
    assert md.splitlines()[0] == "| overlay | nominal | real_mild | inject_p7 |"
    assert "| base | -1.50 ± 4.50 (4/6 obj, 0 wrong, n=2, 1 stop) | +1.00 ± 0.00 (1/2 obj, 0 wrong, n=1) | +2.00 ± 0.00 (3/3 obj, 0 wrong, n=1) |" in md
    assert "| cand | +3.00 ± 1.00 (6/7 obj, 1 wrong, n=2) | +1.00 ± 0.00 (1/2 obj, 0 wrong, n=1) | – |" in md
    assert "| cand | nominal | 2 | +4.50 | +8.00 |" in md
    assert "| cand | real_mild | 1 vs 1 | seeds differ, not paired | |" in md
    assert "Paired vs base `base`" in md


def test_cli_needs_an_input():
    with pytest.raises(SystemExit):
        rt.main([])


def test_cli_runs_as_script(tmp_path):
    adv = tmp_path / "adv"
    adv.mkdir()
    write_adv(adv, "nominal", "base", "aaaaaaaa", 1000, [adv_episode(1000, 2, 2, 0, 2)])
    res = subprocess.run([sys.executable, "-I", os.path.join(HERE, "robustness_table.py"),
                          "--adv", str(adv)], capture_output=True, text=True, check=True)
    assert "| base | +2.00 ± 0.00 (2/2 obj, 0 wrong, n=1) |" in res.stdout
