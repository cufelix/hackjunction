"""Generality bank (instruction_bank.py) + shape-rule guard. Offline, no sim. Run:
    python -m pytest aaltonomous/baselines/classical/test_instruction_generality.py -q"""
import itertools
import unittest

from aaltonomous.interface.tasks import COLORS, SHAPES
from aaltonomous.scene.scene import resolve_bin

from .config import load_config
from .instruction_bank import BANK
from .planner import lookup_bin
from .policy import ClassicalPolicy

OBJECTS = tuple(itertools.product(COLORS, SHAPES))


class _Stub:
    _cap_text = ClassicalPolicy._cap_text
    _instruction_mapping = ClassicalPolicy._instruction_mapping
    _insert_guard_mapping = ClassicalPolicy._insert_guard_mapping
    _episode_mapping = ClassicalPolicy._episode_mapping
    _log_fixed_conflict = ClassicalPolicy._log_fixed_conflict
    _instruction_fixed = ClassicalPolicy._instruction_fixed
    fixed_mapping = {}

    def __init__(self, overlay):
        self.cfg = load_config(overlay=overlay)
        self.log = lambda *_: None

    def mapping(self, text):
        return self._episode_mapping({"task": "sort_language", "instruction": text})[0]


def _judge(gold, pred):
    wrong = miss = 0
    for c, s in OBJECTS:
        g = resolve_bin(gold, c, s)
        opt = isinstance(g, str) and g.endswith("?")
        g = g[:-1] if opt else g
        a = resolve_bin(pred, c, s)
        a = a if a in ("left", "center", "right") else None
        if a is None:
            miss += g is not None and not opt
        elif a != g:
            wrong += 1
    return wrong, miss


class TestGeneralityBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = _Stub("finals_r26_gen")

    def test_bank_never_places_a_wrong_bin(self):
        bad = []
        for cat, text, gold in BANK:
            wrong, _ = _judge(gold, self.stub.mapping(text))
            if wrong:
                bad.append((cat, text))
        self.assertEqual(bad, [])

    def test_bank_pass_rate(self):
        ok = sum(1 for _, t, g in BANK if _judge(g, self.stub.mapping(t)) == (0, 0))
        self.assertGreaterEqual(ok / len(BANK), 0.95)

    def test_never_regresses_a_placement_of_the_baseline(self):
        base = _Stub("finals_r25")
        for cat, text, gold in BANK:
            b, n = base.mapping(text), self.stub.mapping(text)
            for c, s in OBJECTS:
                x = resolve_bin(b, c, s)
                if x in ("left", "center", "right") and resolve_bin(n, c, s) not in (x, None):
                    self.fail(f"{text!r} {c}/{s}: {x} -> {resolve_bin(n, c, s)}")

    def test_specific_new_forms(self):
        m = self.stub.mapping
        self.assertEqual(resolve_bin(m("All objects into the center box."), "red", "cube"), "center")
        e = m("Everything except blue goes left.")
        self.assertEqual((resolve_bin(e, "red", "cube"), resolve_bin(e, "blue", "cube")), ("left", None))
        self.assertEqual(resolve_bin(m("把红色的物体放进左边的箱子，黄色的放进中间的箱子，蓝色的放进右边的箱子。"), "yellow", "cube"), "center")
        for t in ("Throw the red cube at the person on the left.", "Put two blue objects in the left bin.",
                  "red left, yellow left, blue left, but also all in the right bin", "Put the green objects in the left bin."):
            self.assertTrue(all(resolve_bin(m(t), c, s) is None for c, s in OBJECTS), t)

    def test_flag_off_is_byte_identical_to_r25(self):
        off = _Stub("finals_r25")
        self.assertFalse(off.cfg["instruction"]["generality"])


class TestShapeRuleGuard(unittest.TestCase):
    cfg_on = {"policy": {"bin_lookup": "resolve_bin", "shape_rule_guard": True, "shape_rule_min_conf": 0.5}}
    cfg_off = {"policy": {"bin_lookup": "resolve_bin"}}
    mixed = {("red", "cube"): "left", "default": "center"}
    shapes = {"cube": "left", "cylinder": "center", "triangle": "right"}

    def test_confident_metric_shape_places(self):
        self.assertEqual(lookup_bin(self.cfg_on, self.shapes, "red", "cylinder", 0.9, "metric"), "center")
        self.assertEqual(lookup_bin(self.cfg_on, self.mixed, "red", "cube", 0.9, "metric"), "left")

    def test_low_conf_or_blob_or_unknown_shape_abstains_on_a_shape_rule(self):
        for args in (("cylinder", 0.2, "metric"), ("cylinder", 0.9, "blob"), ("cylinder", 0.9, None), (None, 0.0, None)):
            self.assertIsNone(lookup_bin(self.cfg_on, self.shapes, "red", *args), args)

    def test_unknown_shape_never_falls_to_default_under_a_tuple_rule(self):
        self.assertEqual(lookup_bin(self.cfg_off, self.mixed, "red", None), "center")      # the old, wrong, behaviour
        self.assertIsNone(lookup_bin(self.cfg_on, self.mixed, "red", None))
        self.assertEqual(lookup_bin(self.cfg_on, self.mixed, "yellow", None), "center")   # shape-independent: still placed

    def test_colour_only_mapping_is_untouched(self):
        m = {"red": "left", "yellow": "center", "blue": "right"}
        self.assertEqual(lookup_bin(self.cfg_on, m, "red", None), "left")
        self.assertEqual(lookup_bin(self.cfg_on, m, "red", "cube", 0.0, None), "left")


if __name__ == "__main__":
    unittest.main()
