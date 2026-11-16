"""Venue replay: the real Perception.detect() colour path on real finals frames (no depth, no GPU, no simulator).
Frames + hand labels: test_data/venue_20261115/. Harness: tools/venue_replay.py. Run:
    python -m unittest aaltonomous.baselines.classical.test_venue_replay -v

What is (and is not) asserted, honestly:
  * 1063951 behaviour (bin_size_guard off) reproduces the venue failure: on the clean look of run 1 (the frame the policy
    chose "execute red right arm -> right bin" from) an accepted red candidate lies on the right bin.
  * the hotfix bin_size_guard alone does NOT change anything on these frames: its area rule needs a blob > 4x an object,
    the bin's wall blob is ~1x (its 3D rules need depth, which the recordings do not have -> not exercised here).
  * with perception.bin_image_guard (+ the wider gate_below_bins first-run band) no accepted red/yellow/blue candidate
    lies on a labelled bin in the arm-free frames and all 5 objects of every frame are still found as the right colour class.
  * NOT asserted: shape labels (without depth the plane-fallback shape is unreliable), positions, depth stages.
Frames where a robot hand is in view (arm_visible) only get the object-recall assertion: the policy's FK arm mask, which
needs joint states, removes hand blobs; here they show up as 'bin' hits on the left bin and are reported, not asserted.
"""
import importlib.util
import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_spec = importlib.util.spec_from_file_location("venue_replay", os.path.join(ROOT, "tools", "venue_replay.py"))
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)

FIX = "guard+bin_image+first_run_0.55"
_CACHE = {}


def _res(config):
    if config not in _CACHE:
        _CACHE[config] = vr.evaluate(configs=[config], verbose=False)[config]
    return _CACHE[config]


def _clean(res):
    return {f: r for f, r in res.items() if not r["lab"]["arm_visible"]}


class TestVenueReplay(unittest.TestCase):
    def test_fixtures_present_and_small(self):
        frames = vr.load_frames()
        self.assertGreaterEqual(len(frames), 8)
        self.assertLess(sum(os.path.getsize(p) for _, p in frames), 3 * 1024 * 1024)
        for lab, _ in frames:
            self.assertEqual(len(lab["objects"]), 5)
            self.assertGreaterEqual(len(lab["bins"]), 3)

    def test_legacy_reproduces_bin_as_red(self):
        r = _res("legacy")["r1_t12"]
        self.assertTrue(any(c["colour"] == "red" and c["kind"] == "bin" for c in r["on_bin"]))
        self.assertEqual(r["first_red"]["kind"], "bin")      # ranked first: the bin wall is nearest to the red bin
        self.assertEqual(r["first_red"]["name"], "right")

    def test_size_guard_alone_is_noop_without_depth(self):
        legacy, guard = _clean(_res("legacy")), _clean(_res("guard"))
        for f in legacy:
            self.assertEqual(len(legacy[f]["on_bin"]), len(guard[f]["on_bin"]), f)
        self.assertGreater(sum(len(r["on_bin"]) for r in guard.values()), 0)   # failure still there with the hotfix

    def test_fix_no_accepted_candidate_on_a_bin(self):
        for f, r in _clean(_res(FIX)).items():
            self.assertEqual(r["on_bin"], [], f"{f}: accepted on bin {[(c['colour'], c['area'], c['uv']) for c in r['on_bin']]}")
            if r["first_red"] is not None:
                self.assertEqual(r["first_red"]["kind"], "object", f)

    def test_bin_image_guard_alone_residual(self):
        """bin_image_guard alone removes the bin-wall blobs; one 1015-px lit edge strip of the LEFT bin survives (r2_t12)."""
        res = _clean(_res("guard+bin_image"))
        left = {f: len(r["on_bin"]) for f, r in res.items() if r["on_bin"]}
        self.assertLessEqual(sum(left.values()), 1, left)
        for f, r in res.items():
            for c in r["on_bin"]:
                self.assertEqual(c["name"], "left")

    def test_real_objects_still_found(self):
        for config in ("legacy", "guard", FIX):
            for f, r in _res(config).items():
                self.assertEqual(r["missed"], [], f"{config} {f}: missed {r['missed']}")
                self.assertEqual(len(r["found"]), 5)

    def test_flag_default_off_and_no_depth_needed(self):
        from .config import load_config
        self.assertFalse(load_config(overlay="none")["perception"]["bin_image_guard"])

    def test_shipping_overlay_has_the_fix_on_and_it_holds(self):
        from .config import load_config
        p = load_config(overlay="finals_r25_gen")["perception"]
        self.assertTrue(p["bin_image_guard"])
        self.assertEqual(p["gate_below_bins_params"]["first_run_top_frac"], 0.55)
        for f, r in _clean(_res("shipping")).items():
            self.assertEqual(r["on_bin"], [], f)
        for f, r in _res("shipping").items():
            self.assertEqual(r["missed"], [], f)


if __name__ == "__main__":
    unittest.main()
