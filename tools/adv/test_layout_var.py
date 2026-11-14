#!/usr/bin/env python3
"""Offline test of the layout-variation knobs (prism_lying_frac, n_objects, shape_weights, colour_weights).

    uv run -q --no-project --with numpy --with pyyaml --with opencv-python-headless --with trimesh \\
        python tools/adv/test_layout_var.py [--old-patch <git-ref>]

Builds scratch trees: OLD = stock + adv_sim.patch of git ref --old-patch (default 19b8224, the pre-layout-var patch), NEW = stock + current patch.
Checks: (1) old profiles draw identical params per seed in OLD and NEW; (2) new knobs deterministic, capped, shapes /
colours respected; (3) lying pose geometry on the real STL; (4) the patch round-trips.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
PROF = REPO / "tools/adv/profiles"


def tree(patch_text):
    d = Path(tempfile.mkdtemp(prefix="advtree_"))
    (d / "aaltonomous/sim").mkdir(parents=True)
    shutil.copytree(REPO / "aaltonomous/sim/aalto_task", d / "aaltonomous/sim/aalto_task")
    shutil.copy(REPO / "aaltonomous/sim/evaluate.py", d / "aaltonomous/sim/evaluate.py")
    subprocess.run(["patch", "-p1", "-s"], input=patch_text, text=True, cwd=d, check=True)
    return d


def load_dr(d, name):
    spec = importlib.util.spec_from_file_location(name, d / "aaltonomous/sim/aalto_task/dr.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-patch", default="19b8224", help="git ref whose tools/adv/adv_sim.patch is the OLD baseline")
    a = ap.parse_args()
    new_patch = (REPO / "tools/adv/adv_sim.patch").read_text()
    old_patch = subprocess.run(["git", "show", f"{a.old_patch}:tools/adv/adv_sim.patch"], cwd=REPO, text=True,
                               capture_output=True, check=True).stdout
    old_d, new_d = tree(old_patch), tree(new_patch)
    old, new = load_dr(old_d, "dr_old"), load_dr(new_d, "dr_new")
    bins = ("left", "center", "right")

    # (1) old profiles: identical draws per seed
    n = 0
    for f in sorted(PROF.glob("adv_*.yaml")):
        A = (yaml.safe_load(f.read_text()) or {}).get("adv")
        if not A or any(k in A for k in new._VAR_KEYS):
            continue
        for seed in list(range(25)) + [1000, 1003, None]:
            o, w = old.sample_adv(A, seed, bins), new.sample_adv(A, seed, bins)
            assert json.dumps(o, sort_keys=True, default=str) == json.dumps(w, sort_keys=True, default=str), (f.name, seed)
            assert "var" not in w
            n += 1
    print(f"ok 1: {n} (profile, seed) draws identical old vs new")

    # real2_all gains only the `var` entry; everything else identical
    A = yaml.safe_load((PROF / "adv_real2_all.yaml").read_text())["adv"]
    assert "prism_lying_frac" in A
    A_old = {k: v for k, v in A.items() if k != "prism_lying_frac"}
    for seed in range(10):
        w = new.sample_adv(A, seed, bins)
        v = w.pop("var")
        assert json.dumps(w, sort_keys=True, default=str) == json.dumps(old.sample_adv(A_old, seed, bins), sort_keys=True, default=str)
        assert v["prism_lying_frac"] == 0.5
    print("ok 1b: real2_all = old draws + var (frac 0.5)")

    # (2) new knobs: deterministic, capped, respected
    cfg = yaml.safe_load((REPO / "aaltonomous/scene/scene.yaml").read_text())
    geom = new.layout_geom(cfg)
    cap = 3 * 3 * geom["copies"]
    Av = yaml.safe_load((PROF / "adv_layout_var.yaml").read_text())["adv"]
    assert Av["pp_px"] == [325.042, 250.927] and Av["prism_lying_frac"] == 0.5
    from aaltonomous.interface import tasks  # noqa: E402  (stock repo module, numpy only)
    sizes, lies, tri = set(), 0, 0
    for seed in range(200):
        V1, V2 = new.sample_adv(Av, seed, bins)["var"], new.sample_adv(Av, seed, bins)["var"]
        assert V1 == V2
        base = tasks.sample_layout(np.random.default_rng(seed), shapes=tuple(cfg["objects"]["shapes"]),
                                   **{"n_min": cfg["layout"]["n_cubes"][0], "n_max": cfg["layout"]["n_cubes"][1],
                                      "region": (tuple(cfg["layout"]["region_x"]), tuple(cfg["layout"]["region_y"])),
                                      "min_spacing": cfg["layout"]["min_spacing"], "yaw_range": geom["yaw_range"]})
        l1, l2 = new.vary_layout(base, V1, geom), new.vary_layout(base, V1, geom)
        assert l1 == l2 and 3 <= len(l1) <= 6, len(l1)
        sizes.add(len(l1))
        cnt = {}
        for o in l1:
            cnt[(o["color"], o["shape"])] = cnt.get((o["color"], o["shape"]), 0) + 1
            assert o["color"] in geom["colors"] and o["shape"] in geom["shapes"]
        assert max(cnt.values()) <= geom["copies"]
        for i, o in enumerate(l1):
            for j in range(i):
                assert np.hypot(o["x"] - l1[j]["x"], o["y"] - l1[j]["y"]) >= geom["min_spacing"] - 2e-3
        slots = new.lying_layout_slots(l1, V1)
        assert slots == new.lying_layout_slots(l1, V2)
        assert all(l1[i]["shape"] == "triangle" for i in slots)
        lies += len(slots)
        tri += sum(o["shape"] == "triangle" for o in l1)
    assert len(sizes) > 1 and 0.35 < lies / tri < 0.65, (sizes, lies / tri)
    print(f"ok 2: deterministic, 3..6 objects (sizes {sorted(sizes)}), <= copies per type, lying share {lies / tri:.2f}")
    # weights: only cubes of one colour, capped at copies; n_objects above capacity is capped
    V = dict(new.sample_adv({"n_objects": [10, 12], "shape_weights": {"cube": 1}, "colour_weights": {"red": 1}}, 1)["var"])
    out = new.vary_layout(base, V, geom)
    assert 1 <= len(out) <= geom["copies"] and all((o["color"], o["shape"]) == ("red", "cube") for o in out), out
    V = new.sample_adv({"n_objects": 100, "shape_weights": {"triangle": 1}}, 2)["var"]
    out = new.vary_layout(base, V, geom)
    assert len(out) <= 3 * geom["copies"] and {o["shape"] for o in out} == {"triangle"}
    assert new.vary_layout(base, new.sample_adv({"prism_lying_frac": 1.0}, 1)["var"], geom) == base   # no layout knob: unchanged
    print("ok 2b: weights respected, never more than exist")

    # (3) lying pose on the real mesh: rests on the surface, apex up, prism axis horizontal
    import trimesh
    m = trimesh.load(REPO / "aaltonomous/scene/assets/object_triangle.stl", force="mesh")
    v = (np.asarray(m.vertices) * 0.001) - np.array([0, 0, (m.bounds[0][2] + m.bounds[1][2]) / 2 * 0.001])
    d = {"height": 0.045, "apothem": 0.045 / 3}
    for yaw in (-2.0, 0.0, 1.1):
        x, y, z, w = new.lying_quat(yaw)
        Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        zc = d["height"] / 2 - (d["height"] / 2 - d["apothem"])           # centre height above the surface
        zmin = (v @ Rm.T)[:, 2].min() + zc
        assert abs(zmin) < 1e-3, zmin                                       # base face on the surface (fillet 2.5 mm slack)
        assert abs((Rm @ [0, 0, 1])[2]) < 1e-9                              # prism axis horizontal
        assert (Rm @ [0, 1, 0])[2] > 0.999                                  # apex up
        assert abs(np.linalg.norm([x, y, z, w]) - 1) < 1e-9
    print("ok 3: lying pose rests on the surface (STL), apex up")

    # (4) patch round-trips: re-diff the NEW tree against stock -> identical patch text
    out = ""
    for f in ("aalto_task/dr.py", "aalto_task/hooks.py", "aalto_task/scene_cfg.py", "evaluate.py"):
        r = f"aaltonomous/sim/{f}"
        out += subprocess.run(["diff", "-u", "--label", f"a/{r}", "--label", f"b/{r}", str(REPO / r), str(new_d / r)],
                              capture_output=True, text=True).stdout
    assert out == new_patch, "patch does not round-trip"
    print("ok 4: patch applies cleanly and re-diffs to itself")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    main()
