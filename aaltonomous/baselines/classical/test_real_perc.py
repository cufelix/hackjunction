"""Real-only perception recall flags (finals_r23_realperc): default OFF, ON for real, forced OFF by sim_overrides."""
import unittest

import numpy as np

from .config import load_config
from .perception import Perception
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

REAL_FLAGS = ["perception.real_merge_split", "perception.real_border_partial", "perception.split_by_solidity",
              "perception.prism_speckle_floor", "perception.color_gate.fk_silhouette"]


def _get(cfg, dotted):
    for k in dotted.split("."):
        cfg = cfg[k]
    return cfg


class TestRealPercFlags(unittest.TestCase):
    def test_new_flags_default_off(self):
        for name in ("finals_r23", "rc1", "submission"):
            cfg = load_config(overlay=name)
            self.assertFalse(cfg["perception"]["real_merge_split"], name)
            self.assertFalse(cfg["perception"]["real_border_partial"], name)

    def test_real_overlay_on_sim_overrides_off(self):
        real = load_config(overlay="finals_r23_realperc")
        so = real["profile_auto"]["sim_overrides"]
        for k in REAL_FLAGS:
            self.assertTrue(_get(real, k), k)
            self.assertIs(so.get(k), False, k)
        sim = load_config(overlay="finals_r23_realperc", overrides=dict(so))
        for k in REAL_FLAGS:
            self.assertFalse(_get(sim, k), k)

    def test_real_overlay_only_adds_real_flags_to_r23(self):
        a, b = load_config(overlay="finals_r23"), load_config(overlay="finals_r23_realperc")
        sa, sb = dict(a["profile_auto"]["sim_overrides"]), dict(b["profile_auto"]["sim_overrides"])
        self.assertEqual({k: v for k, v in sb.items() if k not in REAL_FLAGS}, sa)

        def flat(d, p=""):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict) and k != "sim_overrides":
                    out.update(flat(v, p + k + "."))
                else:
                    out[p + k] = v
            return out
        fa, fb = flat(a), flat(b)
        diff = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k) and k != "overlay" and "sim_overrides" not in k}
        self.assertEqual(diff, set(REAL_FLAGS) | {"name"} & diff)


def _perc(overrides):
    cfg = load_config(overlay="finals_r23", overrides=overrides)
    return Perception(cfg, Scene.load(), Calibration.load())


class TestMergeSplit(unittest.TestCase):
    def _blob(self):
        m = np.zeros((480, 640), np.uint8)
        m[200:260, 200:260] = 1       # cube
        m[210:250, 255:295] = 1       # touching neighbour
        return m

    def test_flag_lowers_split_gate(self):
        blob = self._blob()
        area = int(blob.sum())
        cen = np.array([245.0, 230.0])
        off = _perc({})
        on = _perc({"perception.real_merge_split": True, "perception.real_merge_split_params": {"area_factor": 0.01, "min_part_frac": 0.01}})
        exp = off.expected_area_px(blob, cen, None)
        self.assertLessEqual(area, 1.5 * exp)          # stock gate does not even try
        self.assertEqual(len(off.split_blob(blob, area, cen, None)), 1)
        self.assertGreaterEqual(len(on.split_blob(blob, area, cen, None)), 2)



class TestRealPerc3(unittest.TestCase):
    KEYS = ["perception.real_merge_split", "perception.prism_speckle_floor", "perception.color_gate.fk_silhouette"]

    def test_realperc3_on_for_real_off_in_sim(self):
        real = load_config(overlay="finals_r23_realperc3")
        so = real["profile_auto"]["sim_overrides"]
        for k in self.KEYS:
            self.assertTrue(_get(real, k), k)
            self.assertIs(so.get(k), False, k)
        self.assertFalse(_get(real, "perception.real_border_partial"))
        self.assertFalse(_get(real, "perception.split_by_solidity"))
        sim = load_config(overlay="finals_r23_realperc3", overrides=dict(so))
        for k in self.KEYS:
            self.assertFalse(_get(sim, k), k)


if __name__ == "__main__":
    unittest.main()
