"""policy.fixed_mapping: the real-robot colour->bin fallback (organisers P0). Run:
    python -m unittest aaltonomous.baselines.classical.test_fixed_mapping -v
"""
import unittest

from aaltonomous.interface.tasks import FIXED_MAPPING

from .config import load_config
from .policy import ClassicalPolicy
from .mock_world import make_mock_env

REAL = {"yellow": "left", "blue": "center", "red": "right"}
FINALS = ("finals_r18", "finals_r19", "finals_r19b", "finals_r19c", "finals_r20", "finals_r21", "finals_r22_pregrasp", "finals_real")
SIM = ("submission", "cand_r17", "rc1")


def mapping(overlay, instruction, **kw):
    pol = ClassicalPolicy(config=load_config(overlay=overlay), log=None, safety=make_mock_env(seed=0).filter.cfg, **kw)
    pol.reset({"instruction": instruction} if instruction is not False else {})
    return pol.mapping


class TestFixedMapping(unittest.TestCase):
    def test_finals_empty_missing_whitespace_use_the_real_mapping(self):
        for o in FINALS:
            for ins in ("", None, "   ", False, " \n\t "):
                self.assertEqual(mapping(o, ins), REAL, (o, ins))

    def test_instruction_text_keeps_priority(self):
        for o in FINALS:
            m = mapping(o, "Put the red objects in the left bin.")
            self.assertEqual(m.get("red"), "left", o)
            self.assertNotEqual(m, REAL, o)

    def test_strict_garbled_text_abstains(self):
        for o in ("finals_r19", "finals_r20"):
            self.assertTrue(load_config(overlay=o)["instruction"]["strict"])
            self.assertEqual(mapping(o, "asdf qwer"), {"default": None}, o)

    def test_sim_overlays_keep_the_old_fixed_mapping(self):
        for o in SIM:
            for ins in ("", None, "   "):
                self.assertEqual(mapping(o, ins), FIXED_MAPPING, (o, ins))

    def test_bad_values_raise(self):
        bad = ({"red": "left", "yellow": "center"}, {"red": "left", "yellow": "left", "blue": "right"},
               {"red": "left", "yellow": "center", "blue": "up"}, {"red": "left", "yellow": "center", "green": "right"},
               "left", [1, 2, 3])
        for b in bad:
            with self.assertRaises(ValueError, msg=repr(b)):
                load_config(overlay="submission", overrides={"policy.fixed_mapping": b})
        load_config(overlay="submission", overrides={"policy.fixed_mapping": REAL})


if __name__ == "__main__":
    unittest.main()
