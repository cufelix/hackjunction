"""Independent open stand-in corpus: 48 sentences, 8 in each of six structural families.
Scored on behaviour through resolve_bin over all 9 colour x shape types, like probe_families.py."""
import sys
sys.path.insert(0, "/home/felix/projects/hackjunction/policy")
from instruction import resolve

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")


def resolve_bin(mapping, color, shape=None):
    """Verbatim logic of scene/scene.py:43 -- most specific first."""
    for key in ((color, shape), color, shape, "default"):
        if key in mapping:
            return mapping[key]
    return None


def score(mapping, truth):
    """truth: (color, shape) -> bin, for all 9. Returns (n_correct, n_unmapped)."""
    ok = unmapped = 0
    for c in COLORS:
        for s in SHAPES:
            got = resolve_bin(mapping, c, s)
            if got is None:
                unmapped += 1
            elif got == truth(c, s):
                ok += 1
    return ok, unmapped


def by_color(d):
    return lambda c, s: d[c]


def by_shape(d):
    return lambda c, s: d[s]


def pairs(d, rest):
    """Compound truth: named (colour, shape) pairs, everything else -> rest."""
    return lambda c, s: d.get((c, s), rest)


def except_key(key, key_bin, rest, kind):
    """All-but truth: the named key goes to key_bin, every other value of that kind to rest."""
    if kind == "color":
        return lambda c, s: key_bin if c == key else rest
    return lambda c, s: key_bin if s == key else rest


CASES = [
    # --- (1) ordering_bin_first -------------------------------------------------------
    ("The left-hand bin gets blue objects, the right-hand bin holds red ones, and the centre "
     "bin receives yellow ones.", by_color({"blue": "left", "red": "right", "yellow": "center"})),
    ("The middle bin is where the yellow pieces go; the rightmost bin takes the blue pieces; "
     "the leftmost bin gets the red pieces.",
     by_color({"yellow": "center", "blue": "right", "red": "left"})),
    ("Centre bin receives cylinders, left bin holds triangles, right bin is for cubes.",
     by_shape({"cylinder": "center", "triangle": "left", "cube": "right"})),
    ("The right bin is for yellow, the left bin takes blue and the centre bin holds red.",
     by_color({"yellow": "right", "blue": "left", "red": "center"})),
    ("Right bin: triangles. Left bin: cylinders. Middle bin: cubes.",
     by_shape({"triangle": "right", "cylinder": "left", "cube": "center"})),
    ("The leftmost bin is where red objects go, the centre bin is where blue objects go, and "
     "the rightmost bin is where yellow objects go.",
     by_color({"red": "left", "blue": "center", "yellow": "right"})),
    ("The middle bin holds cubes; the left bin gets cylinders; the right bin takes triangles.",
     by_shape({"cube": "center", "cylinder": "left", "triangle": "right"})),
    ("The centre bin receives red items, then the right bin receives yellow items, then the "
     "left bin receives blue items.",
     by_color({"red": "center", "yellow": "right", "blue": "left"})),
    # --- (2) elimination --------------------------------------------------------------
    # key eliminated
    ("Red goes left and blue goes centre; the remaining colour goes right.",
     by_color({"red": "left", "blue": "center", "yellow": "right"})),
    # bin eliminated
    ("Yellow objects go left, red objects go centre, and blue objects go in the last bin.",
     by_color({"yellow": "left", "red": "center", "blue": "right"})),
    # both eliminated
    ("The left bin takes blue, the centre bin takes red, and the remaining colour goes in the "
     "remaining bin.", by_color({"blue": "left", "red": "center", "yellow": "right"})),
    # key eliminated (shape)
    ("Cubes go right, cylinders go centre, and the last shape goes in the leftmost bin.",
     by_shape({"cube": "right", "cylinder": "center", "triangle": "left"})),
    # key and bin eliminated, bin-last
    ("Whatever colour has not been mentioned goes in the bin nobody has used: blue goes left "
     "and yellow goes right.", by_color({"blue": "left", "yellow": "right", "red": "center"})),
    # both eliminated (shape)
    ("Triangles go in the left bin and cubes go in the right bin; the last bin gets what is "
     "left.", by_shape({"triangle": "left", "cube": "right", "cylinder": "center"})),
    ("Yellow goes in the centre bin, red goes in the right-hand bin, and the leftover colour "
     "takes the leftover bin.", by_color({"yellow": "center", "red": "right", "blue": "left"})),
    ("Put the cylinders in the middle, the triangles on the right, and send the remaining "
     "shape to the remaining bin.",
     by_shape({"cylinder": "center", "triangle": "right", "cube": "left"})),
    # --- (3) all_but (the named key is EXCLUDED) --------------------------------------
    ("All but the red objects go left; red goes right.",
     except_key("red", "right", "left", "color")),
    ("Everything except the cubes goes in the centre bin; the cubes go left.",
     except_key("cube", "left", "center", "shape")),
    ("All except blue belong on the left, and blue belongs in the right-hand bin.",
     except_key("blue", "right", "left", "color")),
    ("Everything but the triangles goes in the right bin, while triangles go in the centre.",
     except_key("triangle", "center", "right", "shape")),
    ("All the objects except yellow ones go in the centre bin; yellow ones go in the rightmost "
     "bin.", except_key("yellow", "right", "center", "color")),
    ("Excluding the cylinders, everything goes left. Cylinders go right.",
     except_key("cylinder", "right", "left", "shape")),
    ("Every colour except red goes in the middle bin, and red goes in the left one.",
     except_key("red", "left", "center", "color")),
    ("With the exception of the cubes, all shapes go in the right bin; cubes go in the centre "
     "bin.", except_key("cube", "center", "right", "shape")),
    # --- (4) shape_default ------------------------------------------------------------
    ("Cubes go left; all the other shapes go right.",
     by_shape({"cube": "left", "cylinder": "right", "triangle": "right"})),
    ("Put triangles in the centre bin and whatever remains in the left bin.",
     by_shape({"triangle": "center", "cube": "left", "cylinder": "left"})),
    ("Cylinders in the rightmost bin, everything that remains in the leftmost bin.",
     by_shape({"cylinder": "right", "cube": "left", "triangle": "left"})),
    ("Triangles go on the right; any leftover objects go in the middle bin.",
     by_shape({"triangle": "right", "cube": "center", "cylinder": "center"})),
    ("The cubes belong in the left-hand bin, and all remaining pieces belong in the centre "
     "bin.", by_shape({"cube": "left", "cylinder": "center", "triangle": "center"})),
    ("Cylinders go in the centre bin, and the other shapes go in the right bin.",
     by_shape({"cylinder": "center", "cube": "right", "triangle": "right"})),
    ("Send the cubes to the centre, then whatever remains to the left.",
     by_shape({"cube": "center", "cylinder": "left", "triangle": "left"})),
    ("Cubes and cylinders go in the left bin; any leftover objects go in the right bin.",
     by_shape({"cube": "left", "cylinder": "left", "triangle": "right"})),
    # --- (5) many_to_one --------------------------------------------------------------
    ("Yellow and blue share the centre bin, and red goes in the left bin.",
     by_color({"yellow": "center", "blue": "center", "red": "left"})),
    ("Cubes and cylinders in the centre, triangles on the right.",
     by_shape({"cube": "center", "cylinder": "center", "triangle": "right"})),
    ("Red and blue objects both go in the rightmost bin; yellow objects go in the left-hand "
     "bin.", by_color({"red": "right", "blue": "right", "yellow": "left"})),
    ("Triangles and cubes go in the left bin together, and cylinders go in the middle.",
     by_shape({"triangle": "left", "cube": "left", "cylinder": "center"})),
    ("Both red and yellow belong in the centre bin, and blue belongs in the right bin.",
     by_color({"red": "center", "yellow": "center", "blue": "right"})),
    ("Cylinders and triangles share the right bin; cubes go left.",
     by_shape({"cylinder": "right", "triangle": "right", "cube": "left"})),
    ("Yellow objects and blue objects go in the left bin; red objects go in the centre bin.",
     by_color({"yellow": "left", "blue": "left", "red": "center"})),
    ("Red and yellow items go to the middle bin, and blue items go to the left bin.",
     by_color({"red": "center", "yellow": "center", "blue": "left"})),
    # --- (6) compound -----------------------------------------------------------------
    ("Red cubes go in the left bin, blue cylinders in the centre bin, and everything else "
     "goes right.", pairs({("red", "cube"): "left", ("blue", "cylinder"): "center"}, "right")),
    # FILLER shape: "cubes" is the same across all three pairs, so this really keys on colour
    ("Red cubes go left, yellow cubes go centre, blue cubes go right.",
     by_color({"red": "left", "yellow": "center", "blue": "right"})),
    # FILLER shape: "cylinders" is the same across all three pairs
    ("Left bin: red cylinders; centre bin: yellow cylinders; right bin: blue cylinders.",
     by_color({"red": "left", "yellow": "center", "blue": "right"})),
    # FILLER shape: "triangles" is the same across all three pairs
    ("Send blue triangles to the leftmost bin, red triangles to the middle bin and yellow "
     "triangles to the rightmost bin.",
     by_color({"blue": "left", "red": "center", "yellow": "right"})),
    ("Yellow cylinders go left and red cubes go right; everything else goes in the centre bin.",
     pairs({("yellow", "cylinder"): "left", ("red", "cube"): "right"}, "center")),
    # FILLER shape: "triangles" then "ones" is the same across all three pairs
    ("Blue triangles go in the right-hand bin, red triangles in the left-hand bin, and yellow "
     "triangles in the centre.", by_color({"blue": "right", "red": "left", "yellow": "center"})),
    ("Red cubes and red cylinders go left; the other red pieces go centre; all yellow and "
     "blue things go right.",
     pairs({("red", "cube"): "left", ("red", "cylinder"): "left", ("red", "triangle"): "center"},
           "right")),
    ("Blue cubes go to the left bin, yellow triangles go to the centre bin, and the rest go "
     "to the right bin.", pairs({("blue", "cube"): "left", ("yellow", "triangle"): "center"},
                                "right")),
]

assert len(CASES) == 48

FAMILIES = ("ordering_bin_first", "elimination", "all_but", "shape_default", "many_to_one",
            "compound")
tot = bad = 0
fam = {f: 0 for f in FAMILIES}
shortest = {}
for i, (text, truth) in enumerate(CASES):
    f = FAMILIES[i // 8]
    m = resolve(text)
    ok, un = score(m, truth)
    tot += 1
    flag = "ok  " if ok == 9 else "FAIL"
    if ok != 9:
        bad += 1
        if f not in shortest or len(text) < len(shortest[f][0]):
            shortest[f] = (text, m)
    else:
        fam[f] += 1
    print(f"{flag} {ok}/9 unmapped={un}  [{f}] {text[:72]}")
    if ok != 9:
        print(f"       got {m}")
print()
for f in FAMILIES:
    print(f"{f}: {fam[f]}/8")
print(f"\n{tot - bad}/{tot} sentences fully correct")
