"""policy.fixed_mapping: explicit sort_fixed venue-scene metadata (organisers P0). Run:
    python -m unittest aaltonomous.baselines.classical.test_fixed_mapping -v
"""
import unittest

from aaltonomous.interface.tasks import FIXED_MAPPING

from .config import load_config
from .policy import ClassicalPolicy

REAL = {"yellow": "left", "blue": "center", "red": "right"}
FINALS = ("finals_r18", "finals_r19", "finals_r19b", "finals_r19c", "finals_r20", "finals_r21", "finals_r22", "finals_r22_recall", "finals_r22_cal", "finals_real")
SIM = ("submission", "cand_r17", "rc1")


def mapping(overlay, instruction, task=None, reset_mapping=None):
    pol = ClassicalPolicy.__new__(ClassicalPolicy)
    pol.cfg = load_config(overlay=overlay)
    fm = pol.cfg["policy"]["fixed_mapping"]
    pol.fixed_mapping = dict(fm) if fm else dict(FIXED_MAPPING)
    pol.log = lambda *_: None
    pol._cap_text = ClassicalPolicy._cap_text.__get__(pol)
    pol._instruction_mapping = ClassicalPolicy._instruction_mapping.__get__(pol)
    pol._insert_guard_mapping = ClassicalPolicy._insert_guard_mapping.__get__(pol)
    info = {"instruction": instruction, "task": task} if instruction is not False else {"task": task}
    if reset_mapping is not None:
        info["mapping"] = reset_mapping
    return ClassicalPolicy._episode_mapping(pol, info)[0]


class TestFixedMapping(unittest.TestCase):
    def test_finals_sort_fixed_uses_explicit_venue_scene_mapping(self):
        for o in FINALS:
            for ins in ("", None, "   ", False, " \n\t "):
                self.assertEqual(mapping(o, ins, task="sort_fixed"), REAL, (o, ins))

    def test_language_blank_missing_whitespace_truthily_abstains(self):
        for o in FINALS:
            for ins in ("", None, "   ", False, " \n\t "):
                self.assertEqual(mapping(o, ins, task="sort_language"), {"default": None}, (o, ins))

    def test_instruction_text_keeps_priority(self):
        for o in FINALS:
            m = mapping(o, "Put the red objects in the left bin.", task="sort_language")
            self.assertEqual(m.get("red"), "left", o)
            self.assertNotEqual(m, REAL, o)

    def test_strict_garbled_text_abstains(self):
        for o in ("finals_r19", "finals_r20"):
            self.assertTrue(load_config(overlay=o)["instruction"]["strict"])
            self.assertEqual(mapping(o, "asdf qwer", task="sort_language"), {"default": None}, o)

    def test_sim_overlays_keep_the_old_fixed_mapping(self):
        for o in SIM:
            for ins in ("", None, "   "):
                self.assertEqual(mapping(o, ins, task="sort_fixed"), FIXED_MAPPING, (o, ins))

    def test_all_mapping_flags_off_keeps_legacy_fixed_default(self):
        self.assertEqual(mapping("none", "", task=None), FIXED_MAPPING)
        self.assertEqual(mapping("none", "ignored language", task="sort_language"), FIXED_MAPPING)

    def test_reset_supplied_mapping_is_authoritative_for_both_task_types(self):
        supplied = {"red": "right", "yellow": "left", "blue": "center"}
        for task in ("sort_fixed", "sort_language"):
            self.assertEqual(mapping("finals_r21", "red goes left", task=task, reset_mapping=supplied), supplied)

    def test_server_reset_passes_mapping_only_for_identified_sort_fixed(self):
        from .policy_server import ClassicalServerPolicy

        class Stub:
            _job = None
            cfg = {"policy": {}}
            fixed_mapping = REAL

            def reset(self, info):
                self.info = info

        server = ClassicalServerPolicy.__new__(ClassicalServerPolicy)
        server.pol = Stub()
        server.reset("ignored", task="sort_fixed", seed=7)
        self.assertEqual(server.pol.info["mapping"], REAL)
        self.assertEqual(server.pol.info["task"], "sort_fixed")
        server.reset("yellow goes right", task="sort_language", seed=8)
        self.assertNotIn("mapping", server.pol.info)
        self.assertEqual(server.pol.info["task"], "sort_language")

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
