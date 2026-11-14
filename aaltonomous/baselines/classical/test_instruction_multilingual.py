"""Finnish/Czech instruction canonicalization under instruction.multilingual."""
from copy import deepcopy
import unittest

from .config import FLAGS, load_config
from .instruction import resolve
from .normalize import canonicalize_multilingual, strict_mapping


def strict_multilingual(text):
    canonical = canonicalize_multilingual(text)
    return canonical, strict_mapping(canonical, resolve(canonical, fallback={}))


FI_CASES = (
    ("Laita violetti kuutio vasemmalle.", {("blue", "cube"): "left"}),
    ("Pane oranssi sylinteri oikealle.", {("red", "cylinder"): "right"}),
    ("Siirrä keltainen prisma keskelle.", {("yellow", "triangle"): "center"}),
    ("Violetit vasemmalle.", {"blue": "left"}),
    ("Oranssit oikealle.", {"red": "right"}),
    ("Keltaiset keskelle.", {"yellow": "center"}),
    ("Violetti kuutio menee vasempaan laatikkoon.", {("blue", "cube"): "left"}),
    ("Oranssi sylinteri kuuluu oikeaan laatikkoon.", {("red", "cylinder"): "right"}),
    ("Keltainen kolmio menee keskimmäiseen laatikkoon.", {("yellow", "triangle"): "center"}),
    ("Vasemmalle violetti kuutio.", {("blue", "cube"): "left"}),
    ("Oikealle oranssi sylinteri.", {("red", "cylinder"): "right"}),
    ("Keskelle keltainen prisma.", {("yellow", "triangle"): "center"}),
    ("Laita violetit vasemmalle ja oranssit oikealle.", {"blue": "left", "red": "right"}),
    ("Pane keltaiset keskelle ja violetit oikealle.", {"yellow": "center", "blue": "right"}),
    ("Siirrä kuutiot vasemmalle.", {"cube": "left"}),
    ("Pane sylinterit oikealle.", {"cylinder": "right"}),
    ("Laita kolmiot keskelle.", {"triangle": "center"}),
    ("Kaikki violetit kuutiot vasemmalle.", {("blue", "cube"): "left"}),
    ("Violetti vasemmalle, keltainen keskelle, oranssi oikealle.",
     {"blue": "left", "yellow": "center", "red": "right"}),
    ("Pane purppura oikealle ja keltaiset vasemmalle.", {"blue": "right", "yellow": "left"}),
)


CZ_CASES = (
    ("Dej fialovou kostku doleva.", {("blue", "cube"): "left"}),
    ("Polož oranžový válec doprava.", {("red", "cylinder"): "right"}),
    ("Přesuň žlutý hranol doprostřed.", {("yellow", "triangle"): "center"}),
    ("Hoď fialové kostky vlevo.", {("blue", "cube"): "left"}),
    ("Dej oranžové válce vpravo.", {("red", "cylinder"): "right"}),
    ("Polož žluté trojúhelníky do středu.", {("yellow", "triangle"): "center"}),
    ("Fialový patří vlevo.", {"blue": "left"}),
    ("Oranžový patří vpravo.", {"red": "right"}),
    ("Žlutý patří doprostřed.", {"yellow": "center"}),
    ("Vlevo fialovou kostku.", {("blue", "cube"): "left"}),
    ("Vpravo oranžové válce.", {("red", "cylinder"): "right"}),
    ("Doprostřed žluté hranoly.", {("yellow", "triangle"): "center"}),
    ("Dej fialové vlevo a oranžové vpravo.", {"blue": "left", "red": "right"}),
    ("Polož žluté doprostřed a fialové doprava.", {"yellow": "center", "blue": "right"}),
    ("Přesuň kostky doleva.", {"cube": "left"}),
    ("Hoď válce doprava.", {"cylinder": "right"}),
    ("Dej hranoly do středu.", {"triangle": "center"}),
    ("Všechny fialové kostky vlevo.", {("blue", "cube"): "left"}),
    ("Fialové vlevo, žluté doprostřed, oranžové vpravo.",
     {"blue": "left", "yellow": "center", "red": "right"}),
    ("Fialový doleva a žlutý doprava.", {"blue": "left", "yellow": "right"}),
)


AMBIGUOUS = (
    "Laita violetti vasemmalle tai oikealle.",             # two destinations / unknown disjunction
    "Laita vihreä kuutio vasemmalle.",                     # unsupported colour
    "Älä laita oranssia sylinteriä oikealle.",             # negation and unsupported inflections
    "Jos kuutio on violetti, laita se vasemmalle.",        # conditional / pronoun binding
    "Laita vasemmalle.",                                   # destination without selector
    "Dej fialovou kostku doleva nebo doprava.",             # two destinations / disjunction
    "Dej modrou kostku vlevo.",                             # unsupported colour
    "Nedávej oranžový válec vpravo.",                       # negated command
    "Pokud je hranol žlutý, dej ho doprostřed.",            # conditional / pronoun binding
    "Fialové vlevo a oranssit oikealle.",                   # mixed Czech/Finnish
)


class TestMultilingualMappings(unittest.TestCase):
    def test_40_finnish_and_czech_sentences(self):
        self.assertEqual(len(FI_CASES) + len(CZ_CASES), 40)
        for text, expected in FI_CASES + CZ_CASES:
            canonical, got = strict_multilingual(text)
            self.assertEqual(got, expected, (text, canonical, got))

    def test_10_ambiguous_or_partial_sentences_abstain(self):
        self.assertEqual(len(AMBIGUOUS), 10)
        for text in AMBIGUOUS:
            canonical, got = strict_multilingual(text)
            self.assertEqual(canonicalize_multilingual(text), "unparsed multilingual instruction", text)
            self.assertEqual(got, {"default": None}, (text, canonical, got))


class TestMultilingualFlag(unittest.TestCase):
    def test_registered_default_off_and_false_path_is_identical(self):
        self.assertIn("instruction.multilingual", FLAGS)
        self.assertFalse(load_config(overlay="none")["instruction"]["multilingual"])
        samples = [text for text, _ in FI_CASES[:3] + CZ_CASES[:3]]

        from .policy import ClassicalPolicy
        current = load_config(overlay="rc1")
        explicit_off = deepcopy(current)
        explicit_off["instruction"]["multilingual"] = False
        legacy = deepcopy(current)
        legacy["instruction"].pop("multilingual")

        def bare(cfg):
            policy = object.__new__(ClassicalPolicy)
            policy.cfg, policy.fixed_mapping, policy.log = cfg, None, lambda _: None
            return policy

        for text in samples + ["Put the red cubes in the left bin.", "qqq zzz"]:
            self.assertEqual(bare(explicit_off)._instruction_mapping({"instruction": text}),
                             bare(legacy)._instruction_mapping({"instruction": text}), text)

    def test_multilingual_only_does_not_enable_english_normalizer(self):
        from .policy import ClassicalPolicy
        text = "Put the cans in the right receptacle and the bricks in the left receptacle."
        cfg = load_config(overrides={"instruction.resolve": True, "instruction.multilingual": True,
                                     "instruction.normalize": False})
        policy = object.__new__(ClassicalPolicy)
        policy.cfg, policy.fixed_mapping, policy.log = cfg, None, lambda _: None
        self.assertEqual(policy._instruction_mapping({"instruction": text}), resolve(text))
        self.assertNotIn("cylinder", policy._instruction_mapping({"instruction": text}))

    def test_common_english_collisions_pass_through_byte_identically(self):
        for text in ("Do put the red cube in the left bin.",
                     "Place it in the window pane.",
                     "Put the box on the right."):
            self.assertEqual(canonicalize_multilingual(text), text)

    def test_policy_wires_the_flag_before_strict_resolve(self):
        from .policy import ClassicalPolicy
        cfg = load_config(overrides={"instruction.resolve": True, "instruction.multilingual": True,
                                     "instruction.strict": True})
        # _instruction_mapping is intentionally isolated from robot construction; keep this language
        # test independent of Pinocchio and the external URDF checkout.
        policy = object.__new__(ClassicalPolicy)
        policy.cfg, policy.fixed_mapping, policy.log = cfg, None, lambda _: None
        self.assertEqual(policy._instruction_mapping({"instruction": "Dej fialovou kostku doleva."}),
                         {("blue", "cube"): "left"})
        self.assertEqual(policy._instruction_mapping({"instruction": AMBIGUOUS[0]}), {"default": None})
        self.assertEqual(policy._instruction_mapping({"instruction": 7}), {"default": None})


if __name__ == "__main__":
    unittest.main()
