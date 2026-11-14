"""Closed-grammar multilingual canonicalization under instruction.multilingual."""
from copy import deepcopy
import time
import unittest

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


EXTRA_CASES = {
    "de": (
        ("Lege den gelben Würfel in den linken Behälter.", {("yellow", "cube"): "left"}),
        ("Sortiere violette Dosen in die Mitte.", {("blue", "cylinder"): "center"}),
        ("Verschiebe orangene Prismen nach rechts.", {("red", "triangle"): "right"}),
        ("Wirf rote Schachteln ins linke Fach.", {("red", "cube"): "left"}),
        ("Stell blaue Büchsen in den rechten Korb.", {("blue", "cylinder"): "right"}),
        ("Lege fliederfarbene Blöcke ins mittlere Fach.", {("blue", "cube"): "center"}),
        ("Sortieren gelbe Zylinder und blaue Würfel links.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Rote Dächer rechts, lila Würfel links.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
    "sv": (
        ("Lägg gula kuber i vänstra facket.", {("yellow", "cube"): "left"}),
        ("Sortera lila burkar till mitten.", {("blue", "cylinder"): "center"}),
        ("Flytta orangea prismor till höger.", {("red", "triangle"): "right"}),
        ("Släpp röda lådor i vänstra korgen.", {("red", "cube"): "left"}),
        ("Placera blå cylindrar till höger.", {("blue", "cylinder"): "right"}),
        ("Lägg syrenfärgade block i mitten.", {("blue", "cube"): "center"}),
        ("Sortera gula cylindrar och blå kuber vänster.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Röda tak höger, lila kuber vänster.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
    "es": (
        ("Pon el cubo amarillo en el contenedor izquierdo.", {("yellow", "cube"): "left"}),
        ("Coloca las latas moradas en el centro.", {("blue", "cylinder"): "center"}),
        ("Mueve los prismas naranjas a la derecha.", {("red", "triangle"): "right"}),
        ("Suelta las cajas rojas en la cesta izquierda.", {("red", "cube"): "left"}),
        ("Clasifica los cilindros azules a la derecha.", {("blue", "cylinder"): "right"}),
        ("Pon los bloques lilas en el compartimento central.", {("blue", "cube"): "center"}),
        ("Ordena los cilindros amarillos y cubos azules a la izquierda.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Triangulos rojos derecha, cubos violetas izquierda.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
    "fr": (
        ("Mettez les cubes jaunes dans le bac gauche.", {("yellow", "cube"): "left"}),
        ("Place les canettes violettes au centre.", {("blue", "cylinder"): "center"}),
        ("Déplace les prismes orange vers la droite.", {("red", "triangle"): "right"}),
        ("Lâchez les boîtes rouges dans le panier gauche.", {("red", "cube"): "left"}),
        ("Triez les cylindres bleus à droite.", {("blue", "cylinder"): "right"}),
        ("Mets les blocs lilas dans le compartiment central.", {("blue", "cube"): "center"}),
        ("Placez les cylindres jaunes et les cubes bleus à gauche.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Toits rouges droite, cubes pourpres gauche.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
    "it": (
        ("Metti i cubi gialli nel contenitore sinistro.", {("yellow", "cube"): "left"}),
        ("Colloca le lattine viola al centro.", {("blue", "cylinder"): "center"}),
        ("Sposta i prismi arancioni a destra.", {("red", "triangle"): "right"}),
        ("Lascia le scatole rosse nel cestino sinistro.", {("red", "cube"): "left"}),
        ("Smista i cilindri blu a destra.", {("blue", "cylinder"): "right"}),
        ("Metti i blocchi lilla nel scomparto centrale.", {("blue", "cube"): "center"}),
        ("Ordina i cilindri gialli e i cubi azzurri a sinistra.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Tetti rossi destra, cubi violetti sinistra.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
    "pl": (
        ("Połóż żółtą kostkę w lewym pojemniku.", {("yellow", "cube"): "left"}),
        ("Umieść fioletowe puszki na środku.", {("blue", "cylinder"): "center"}),
        ("Przenieś pomarańczowe pryzmaty na prawo.", {("red", "triangle"): "right"}),
        ("Wrzuć czerwone pudełka do lewego kosza.", {("red", "cube"): "left"}),
        ("Sortuj niebieskie walce na prawo.", {("blue", "cylinder"): "right"}),
        ("Przesuń liliowe bloki do środka.", {("blue", "cube"): "center"}),
        ("Sortuj żółte walce oraz niebieskie sześciany na lewo.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Czerwone dachy na prawo, fioletowe kostki na lewo.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
    "pt": (
        ("Coloque os cubos amarelos no recipiente esquerdo.", {("yellow", "cube"): "left"}),
        ("Ordene as latas roxas no centro.", {("blue", "cylinder"): "center"}),
        ("Mova os prismas laranja para a direita.", {("red", "triangle"): "right"}),
        ("Solte as caixas vermelhas no cesto esquerdo.", {("red", "cube"): "left"}),
        ("Classifique os cilindros azuis à direita.", {("blue", "cylinder"): "right"}),
        ("Ponha os blocos lilás no compartimento central.", {("blue", "cube"): "center"}),
        ("Ordene os cilindros amarelos e cubos azuis à esquerda.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Telhados vermelhos direita, cubos violetas esquerda.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
    "uk": (
        ("Поклади жовтий куб у лівий кошик.", {("yellow", "cube"): "left"}),
        ("Сортуй фіолетові банки в центр.", {("blue", "cylinder"): "center"}),
        ("Перемісти помаранчеві призми праворуч.", {("red", "triangle"): "right"}),
        ("Кинь червону коробку в лівий контейнер.", {("red", "cube"): "left"}),
        ("Помісти сині циліндри праворуч.", {("blue", "cylinder"): "right"}),
        ("Поклади бузкові блоки в центр.", {("blue", "cube"): "center"}),
        ("Сортуй жовті циліндри і сині куби ліворуч.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Chervoni dakhy pravoruch, fioletovi kuby livoruch.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
    "ru": (
        ("Положи жёлтый куб в левый контейнер.", {("yellow", "cube"): "left"}),
        ("Сортируй фиолетовые банки в центр.", {("blue", "cylinder"): "center"}),
        ("Перемести оранжевые призмы вправо.", {("red", "triangle"): "right"}),
        ("Брось красную коробку в левую корзину.", {("red", "cube"): "left"}),
        ("Помести синие цилиндры вправо.", {("blue", "cylinder"): "right"}),
        ("Положи сиреневые блоки в центр.", {("blue", "cube"): "center"}),
        ("Сортируй жёлтые цилиндры и синие кубы влево.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Krasnye kryshi vpravo, fioletovye kuby vlevo.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
    "sk": (
        ("Polož žltú kocku do ľavého koša.", {("yellow", "cube"): "left"}),
        ("Trieď fialové plechovky do stredu.", {("blue", "cylinder"): "center"}),
        ("Presuň oranžové hranoly doprava.", {("red", "triangle"): "right"}),
        ("Hoď červené škatule do ľavého kontajnera.", {("red", "cube"): "left"}),
        ("Premiestni modré valce doprava.", {("blue", "cylinder"): "right"}),
        ("Daj orgovánové bloky do stredu.", {("blue", "cube"): "center"}),
        ("Zoraď žlté valce a modré kocky doľava.",
         {("yellow", "cylinder"): "left", ("blue", "cube"): "left"}),
        ("Červené strechy doprava, fialové kocky doľava.",
         {("red", "triangle"): "right", ("blue", "cube"): "left"}),
    ),
}


ENGLISH_GOLDEN = (
    ("Sort the objects: blue into left, red into center, yellow into right.",
     {"blue": "left", "red": "center", "yellow": "right"}),
    ("Put all red cubes in the left bin.", {("red", "cube"): "left"}),
    ("Put it somewhere.", {"red": "left", "yellow": "center", "blue": "right"}),
    ("Red goes left, but not blue.", {"red": "left", "blue": "right"}),
    ("Put all red cubes in the left bin, yellow cylinders in the center bin, and blue triangles in the right bin.",
     {("red", "cube"): "left", ("yellow", "cylinder"): "center", ("blue", "triangle"): "right"}),
    ("Red goes left, yellow goes right, everything else goes centre.",
     {"red": "left", "yellow": "right", "default": "center"}),
    ("Put the yellow cubes on the table and red in the left bin.", {"yellow": "left", "red": "left"}),
    ("Put the red cube in the left bin and the blue cylinders in the right bin.",
     {("red", "cube"): "left", ("blue", "cylinder"): "right"}),
)


AMBIGUOUS = (
    "Laita violetti vasemmalle tai oikealle.",             # two destinations / unknown disjunction
    "Laita vihreä kuutio vasemmalle.",                     # unsupported colour
    "Älä laita oranssia sylinteriä oikealle.",             # negation and unsupported inflections
    "Jos kuutio on violetti, laita se vasemmalle.",        # conditional / pronoun binding
    "Laita vasemmalle.",                                   # destination without selector
    "Dej fialovou kostku doleva nebo doprava.",             # two destinations / disjunction
    "Dej zelenou kostku vlevo.",                            # unsupported colour
    "Nedávej oranžový válec vpravo.",                       # negated command
    "Pokud je hranol žlutý, dej ho doprostřed.",            # conditional / pronoun binding
    "Lege den smaragdgrünen Würfel in den linken Behälter.", # unsupported colour
)


class TestMultilingualMappings(unittest.TestCase):
    def test_40_finnish_and_czech_sentences(self):
        self.assertEqual(len(FI_CASES) + len(CZ_CASES), 40)
        for text, expected in FI_CASES + CZ_CASES:
            canonical, got = strict_multilingual(text)
            self.assertEqual(got, expected, (text, canonical, got))

    def test_eight_sentences_for_each_added_language(self):
        self.assertEqual(set(EXTRA_CASES), {"de", "sv", "es", "fr", "it", "pl", "pt", "uk", "ru", "sk"})
        for language, cases in EXTRA_CASES.items():
            self.assertGreaterEqual(len(cases), 8, language)
            for text, expected in cases:
                canonical, got = strict_multilingual(text)
                self.assertEqual(got, expected, (language, text, canonical, got))

    def test_mixed_language_sentence(self):
        text = "Lege the amarillo cube dans le bac gauche."
        canonical, got = strict_multilingual(text)
        self.assertEqual(canonical, "put yellow cube in the left bin")
        self.assertEqual(got, {("yellow", "cube"): "left"})

    def test_english_golden_is_byte_identical_and_parses_as_before(self):
        for text, expected in ENGLISH_GOLDEN:
            self.assertEqual(canonicalize_multilingual(text), text)
            self.assertEqual(resolve(canonicalize_multilingual(text)), expected, text)

    def test_normalization_timing_and_cache(self):
        text = ("Lege " + "bitte " * 320 + "gelbe Würfel links.").ljust(2000)
        self.assertEqual(len(text), 2000)
        canonicalize_multilingual.cache_clear()
        start = time.perf_counter_ns()
        canonical = canonicalize_multilingual(text)
        uncached_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.assertEqual(canonical, "put yellow cube in the left bin")
        start = time.perf_counter_ns()
        canonicalize_multilingual(text)
        cached_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.assertLess(uncached_ms, 5.0, uncached_ms)
        self.assertLess(cached_ms, 0.1, cached_ms)
        self.assertEqual(canonicalize_multilingual.cache_info().hits, 1)

    def test_10_ambiguous_or_partial_sentences_abstain(self):
        self.assertEqual(len(AMBIGUOUS), 10)
        for text in AMBIGUOUS:
            canonical, got = strict_multilingual(text)
            self.assertEqual(canonicalize_multilingual(text), "unparsed multilingual instruction", text)
            self.assertEqual(got, {"default": None}, (text, canonical, got))


class TestMultilingualFlag(unittest.TestCase):
    def test_registered_default_off_and_false_path_is_identical(self):
        from .config import FLAGS, load_config
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
        from .config import load_config
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
        from .config import load_config
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
