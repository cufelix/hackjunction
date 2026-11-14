import json

from tools.lang_blind import generate, organiser_cases, read_jsonl, score


def test_generation_is_seeded_and_has_disjoint_template_splits():
    a = generate(300, 17)
    b = generate(300, 17)
    assert a == b
    assert len(a) == 300
    assert {r["split"] for r in a} == {"dev", "heldout"}
    dev = {r["template"] for r in a if r["split"] == "dev"}
    held = {r["template"] for r in a if r["split"] == "heldout"}
    assert not dev.intersection(held)
    assert {r["category"] for r in a} == {
        "direct", "zip", "directional", "colon", "conditional", "ordinal", "ambiguous"
    }
    assert {r["mode"] for r in a} == {"color", "shape", "both"}
    assert all(not r["gold"] for r in a if r["category"] == "ambiguous")


def test_jsonl_round_trip_and_external_defaults(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = generate(4, 9)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert read_jsonl([str(path)]) == rows

    path.write_text(json.dumps({"text": "red goes left", "gold": {"red": "left"}}) + "\n")
    row = read_jsonl([str(path)])[0]
    assert row["split"] == row["category"] == "external"


def test_published_templates_are_8_times_36_and_stay_exact():
    rows = organiser_cases()
    assert len(rows) == 8 * 36
    got = score(rows, "finals_r21")["metrics"]
    assert got["exact_rate"] == 1.0
    assert got["wrong_bin_rate"] == 0.0
    assert got["points_per_object"] == 2.0


def test_blind_sample_improves_exact_without_adding_wrong_bins():
    rows = generate(3000, 17)
    before = score(rows, "finals_r21")["metrics"]
    after = score(rows, "ab_langzip")["metrics"]
    assert after["exact_rate"] > before["exact_rate"]
    assert after["wrong_bin_rate"] <= before["wrong_bin_rate"]
    assert after["exact"] + after["wrong_bin"] + after["abstain"] == after["cases"]
