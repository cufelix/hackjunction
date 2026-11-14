"""Does the wiring in policy.py actually reach the resolver? CPU only, no sim, no URDF.

Run before spending GPU time:  python3 -m pytest aaltonomous/baselines/classical/test_instruction_wiring.py

`ClassicalPolicy.__init__` loads a URDF and the IK solver, so instantiating it here is not an
option. `_instruction_mapping` only ever touches `self.cfg` and `self.log`, so it is called
unbound on a stub -- which exercises the real method body, including the relative import of
`.instruction` that would silently fall back to FIXED_MAPPING if the module were missing from
the image.
"""
from __future__ import annotations

import unittest

from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.interface.tasks import FIXED_MAPPING

LANGUAGE = ("Sort the objects: blue into left, red into center, yellow into right.")


class Stub:
    """Just enough of a policy for `_instruction_mapping`."""

    def __init__(self, resolve):
        self.cfg = {"instruction": {"resolve": resolve}}
        self.logged = []
        self.log = self.logged.append

    mapping = staticmethod(ClassicalPolicy._instruction_mapping)
    _cap_text = ClassicalPolicy._cap_text


def call(resolve, info):
    s = Stub(resolve)
    return ClassicalPolicy._instruction_mapping(s, info), s.logged


class TestWiring(unittest.TestCase):
    def test_flag_off_is_byte_for_byte_the_old_behaviour(self):
        """The step-0 reference run must not move: off -> None -> `or dict(FIXED_MAPPING)`."""
        got, logged = call(False, {"instruction": LANGUAGE})
        self.assertIsNone(got)
        self.assertEqual(logged, [])

    def test_flag_on_resolves_the_sentence(self):
        got, _ = call(True, {"instruction": LANGUAGE})
        self.assertEqual(got, {"blue": "left", "red": "center", "yellow": "right"})
        self.assertNotEqual(got, FIXED_MAPPING, "a language episode must not sort by FIXED")

    def test_no_text_no_mapping(self):
        for info in ({}, None, {"instruction": ""}, {"instruction": None}):
            self.assertEqual(call(True, info)[0], {"default": None}, info)

    def test_garbage_abstains_instead_of_using_fixed_mapping(self):
        got, _ = call(True, {"instruction": "do the needful"})
        self.assertEqual(got, {"default": None})

    def test_a_resolver_that_raises_costs_nothing(self):
        """reset() is under a 1.0 s watchdog; an exception here would be a BAD_END."""
        class Boom(Stub):
            pass

        s = Boom(True)
        s.cfg = {"instruction": {"resolve": True}}

        # A dict subclass that explodes on the .get("instruction") the method does next. It has to
        # be NON-EMPTY: `info or {}` swaps an empty dict for a plain one, which never raises.
        class Evil(dict):
            def get(self, *a, **k):
                raise RuntimeError("parser bug")

        self.assertEqual(ClassicalPolicy._instruction_mapping(s, Evil(instruction="x")), {"default": None})
        self.assertTrue(any("instruction resolve failed" in m for m in s.logged), s.logged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
