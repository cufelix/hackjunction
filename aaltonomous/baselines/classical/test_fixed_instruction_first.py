"""policy.fixed_instruction_first: sort_fixed uses the instruction's mapping only when it resolves EVERY venue colour."""
import unittest
from unittest import mock

from .policy import ClassicalPolicy

VENUE = {"yellow": "left", "blue": "center", "red": "right"}
FULL = {"yellow": "right", "blue": "left", "red": "center"}   # differs from the venue on purpose


def _policy(flag, parsed=None, raises=None):
    p = ClassicalPolicy.__new__(ClassicalPolicy)
    p.cfg = {"policy": {"fixed_instruction_first": flag}, "instruction": {"resolve": True}}
    p.fixed_mapping = dict(VENUE)
    p.logs = []
    p.log = p.logs.append
    if raises is not None:
        p._instruction_mapping = mock.Mock(side_effect=raises)
    else:
        p._instruction_mapping = mock.Mock(return_value=parsed)
    return p


INFO = {"task": "sort_fixed", "instruction": "sort the blocks"}


class TestFixedInstructionFirst(unittest.TestCase):
    def test_full_instruction_mapping_beats_venue(self):
        p = _policy(True, parsed=dict(FULL))
        mapping, _ = p._episode_mapping(INFO)
        self.assertEqual(mapping, FULL)
        self.assertTrue(any("instruction resolves all colours" in m for m in p.logs))

    def test_partial_instruction_mapping_keeps_venue(self):
        p = _policy(True, parsed={"yellow": "right", "blue": "left"})   # red missing
        self.assertEqual(p._episode_mapping(INFO)[0], VENUE)

    def test_abstain_or_unparsed_keeps_venue(self):
        self.assertEqual(_policy(True, parsed={"default": "abstain"})._episode_mapping(INFO)[0], VENUE)
        self.assertEqual(_policy(True, parsed=None)._episode_mapping(INFO)[0], VENUE)

    def test_flag_off_keeps_venue(self):
        p = _policy(False, parsed=dict(FULL))
        self.assertEqual(p._episode_mapping(INFO)[0], VENUE)
        self.assertFalse(any("instruction resolves" in m for m in p.logs))

    def test_error_in_instruction_falls_back_to_venue(self):
        p = _policy(True, raises=RuntimeError("boom"))
        self.assertEqual(p._episode_mapping(INFO)[0], VENUE)   # never raises
        self.assertTrue(any("instruction-first failed" in m for m in p.logs))


if __name__ == "__main__":
    unittest.main()
