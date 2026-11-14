"""My own probes in the three families Oliver DESCRIBED (not his sealed sentences).
Scored on behaviour through resolve_bin over all 9 colour x shape types, like his scorer."""
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


CASES = [
    # --- (2) elimination, bin-first order --------------------------------------------
    ("The left bin is for blue, the right bin takes red, and the middle one gets everything "
     "else, i.e. the yellow stuff.", by_color({"blue": "left", "red": "right", "yellow": "center"})),
    ("The left bin is for blue, the right one for red; the remaining colour takes the "
     "remaining bin.", by_color({"blue": "left", "red": "right", "yellow": "center"})),
    ("Blue in the centre bin, yellow in the left bin, and the remaining colour goes to the "
     "remaining bin.", by_color({"blue": "center", "yellow": "left", "red": "right"})),
    ("Put blue on the left and red in the middle; everything else, i.e. the yellow items, "
     "goes right.", by_color({"blue": "left", "red": "center", "yellow": "right"})),
    ("The centre bin takes yellow, the left bin red, and whatever colour is left takes the "
     "bin that is left.", by_color({"yellow": "center", "red": "left", "blue": "right"})),
    # --- (3) shape_default -----------------------------------------------------------
    ("Cubes go in the left bin and all remaining pieces go in the right bin.",
     by_shape({"cube": "left", "cylinder": "right", "triangle": "right"})),
    ("Put the triangles in the centre bin; whatever remains goes left.",
     by_shape({"triangle": "center", "cube": "left", "cylinder": "left"})),
    ("Cylinders to the right bin, all remaining objects to the middle.",
     by_shape({"cylinder": "right", "cube": "center", "triangle": "center"})),
    ("Triangles on the left, everything that remains on the right.",
     by_shape({"triangle": "left", "cube": "right", "cylinder": "right"})),
    # --- (3) two-shape merge ---------------------------------------------------------
    ("Cubes and cylinders both go in the left bin, triangles go in the right bin.",
     by_shape({"cube": "left", "cylinder": "left", "triangle": "right"})),
    ("Cubes and triangles both belong in the centre bin; cylinders belong on the right.",
     by_shape({"cube": "center", "triangle": "center", "cylinder": "right"})),
    ("Both the cubes and the cylinders go right, the triangles go left.",
     by_shape({"cube": "right", "cylinder": "right", "triangle": "left"})),
    # --- colour merge, the sibling case ----------------------------------------------
    ("Red and yellow both go in the left bin, blue goes in the right bin.",
     by_color({"red": "left", "yellow": "left", "blue": "right"})),
    # --- controls that already worked (must not regress) -----------------------------
    ("Put the red objects in the left bin, the yellow objects in the center bin and the blue "
     "objects in the right bin.", by_color({"red": "left", "yellow": "center", "blue": "right"})),
    ("Put the cubes in the left bin, the cylinders in the center bin and the triangles in the "
     "right bin.", by_shape({"cube": "left", "cylinder": "center", "triangle": "right"})),
    ("Put the red objects in the left bin and everything else in the right bin.",
     by_color({"red": "left", "yellow": "right", "blue": "right"})),
]

tot = bad = 0
for text, truth in CASES:
    m = resolve(text)
    ok, un = score(m, truth)
    tot += 1
    flag = "ok  " if ok == 9 else "FAIL"
    if ok != 9:
        bad += 1
    print(f"{flag} {ok}/9 unmapped={un}  {text[:72]}")
    if ok != 9:
        print(f"       got {m}")
print(f"\n{tot - bad}/{tot} sentences fully correct")
