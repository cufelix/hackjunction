"""instruction.venue_veto: viewer-perspective sides, bin negations, bare ordinals, foreign negations abstain on every object
in the real finals overlay (finals_r25_gen), the robot's own left/centre/right still parse, and sim_overrides turn it off. Run:
    python -m pytest aaltonomous/baselines/classical/test_venue_veto.py -q"""
import unittest

from .config import load_config
from .generality import venue_veto
from .policy import ClassicalPolicy

ABSTAIN = (
    "Yellow things to my left, blue things to my right, red in the middle.",
    "From where I'm standing, put red on the left, yellow in the centre and blue on the right.",
    "Red to the viewer's left bin, blue to the viewer's right bin, yellow centre.",
    "Put the orange ones in the bin on the audience's left.",
    "Put the yellow ones anywhere but the left bin.",
    "Red in the first box, yellow in the second box, blue in the third box.",
    "Put red in bin 1 and blue in bin 3.",
    "Žluté ne do levé krabice, červené doprava.",
    "Rote nicht nach links, gelbe in die Mitte.",
    "Punaiset ei vasempaan laatikkoon, keltaiset keskelle.",
)
KEEP = (    # the robot's own sides must keep parsing
    "Put the red objects in your left bin, the yellow in the middle bin and the blue in your right bin.",
    "From the robot's perspective: red left, yellow centre, blue right.",
    "Yellow to the left from your point of view, orange to the right from your point of view, purple in the center.",
    "Blue in the third bin from the left, red in the first bin from the left, yellow in the second.",
    "Put the red objects in the left bin, and the same for the yellow ones, but in the right bin.",
    "Everything but blue goes left.",
    "red -> left, yellow -> center, blue -> right",
    "Laita punaiset vasempaan laatikkoon, keltaiset keskelle ja siniset oikealle.",
    "Červené do levé krabice, žluté doprostřed, modré do pravé.",
)


class VenueVeto(unittest.TestCase):
    def test_veto_regexes(self):
        for t in ABSTAIN:
            self.assertIsNotNone(venue_veto(t), t)
        for t in KEEP:
            self.assertIsNone(venue_veto(t), t)

    def _mapping(self, overlay, text, overrides=None):
        class S:
            _cap_text = ClassicalPolicy._cap_text
            _instruction_mapping = ClassicalPolicy._instruction_mapping
            _episode_mapping = ClassicalPolicy._episode_mapping
            _insert_guard_mapping = ClassicalPolicy._insert_guard_mapping
            _log_fixed_conflict = ClassicalPolicy._log_fixed_conflict
            _instruction_fixed = ClassicalPolicy._instruction_fixed
            fixed_mapping = {}
        s = S()
        s.cfg = load_config(overlay=overlay, overrides=overrides)
        s.log = lambda *_: None
        return s._episode_mapping({"task": "sort_language", "instruction": text})[0]

    def test_real_overlay_abstains(self):
        for t in ABSTAIN:
            m = self._mapping("finals_r25_gen", t)
            self.assertTrue(all(v is None for v in m.values()), (t, m))

    def test_real_overlay_keeps_robot_sides(self):
        m = self._mapping("finals_r25_gen", KEEP[0])
        self.assertEqual((m.get("red"), m.get("yellow"), m.get("blue")), ("left", "center", "right"))

    def test_flag_off_is_inert(self):
        cfg = load_config(overlay="finals_r25_gen", overrides={"instruction.venue_veto": False})
        self.assertFalse(cfg["instruction"]["venue_veto"])
        self.assertFalse(load_config(overlay="finals_r25")["instruction"]["venue_veto"])

    def test_sim_overrides_turn_it_off(self):
        sim = load_config(overlay="finals_r25_gen")["profile_auto"]["sim_overrides"]
        self.assertIs(sim["instruction.venue_veto"], False)


if __name__ == "__main__":
    unittest.main()
