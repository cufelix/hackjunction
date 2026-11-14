"""OUR novel-object scenes (NOT official). Same table / bins layout as the stock scene, 4-5 objects each.

Every object: key (name prefix obj_<colour>_<shape>), gt = the bin it must end in ("left"/"center"/"right") or None
= distractor that should stay on the counter. Instructions are what the policy hears; gt is defined by us.
Slots are mat coordinates (x along the counter, y away from the robot); `deep` slots (y 0.175) take the long objects.
"""
from __future__ import annotations

import numpy as np

SLOTS = [(0.31, 0.175, True), (0.40, 0.125, False), (0.47, 0.175, True), (0.55, 0.125, False), (0.62, 0.175, True)]
BIG = {"banana", "marker", "pan", "mug", "can"}

O = lambda colour, shape, gt: {"colour": colour, "shape": shape, "gt": gt}   # noqa: E731

SCENES = {
    "S1_ball_left": {
        "instruction": "Put the ball in the left bin and everything else in the center bin.",
        "what": "novel shape (ball) by name + 'everything else' default; 3 known objects",
        "objects": [O("green", "ball", "left"), O("red", "cube", "center"), O("yellow", "cylinder", "center"),
                    O("blue", "triangle", "center")]},
    "S2_mug_right": {
        "instruction": "Put the mug in the right bin.",
        "what": "novel object with handle by name; unseen colours (white, purple) as distractors that must stay",
        "objects": [O("natural", "mug", "right"), O("white", "cylinder", None), O("red", "cube", None),
                    O("purple", "triangle", None)]},
    "S3_fruit_left": {
        "instruction": "Fruit goes in the left bin, everything else goes in the right bin.",
        "what": "category word (fruit = banana); yellow cube is a colour-confusable distractor",
        "objects": [O("natural", "banana", "left"), O("yellow", "cube", "right"), O("red", "cylinder", "right"),
                    O("blue", "triangle", "right")]},
    "S4_unseen_colours": {
        "instruction": "Green things in the left bin, white things in the center bin and purple things in the right bin.",
        "what": "unseen colours (green, white, purple) on known shapes; black cube must stay",
        "objects": [O("green", "cube", "left"), O("white", "cylinder", "center"), O("purple", "triangle", "right"),
                    O("black", "cube", None)]},
    "S5_can_marker": {
        "instruction": "Put the can in the center bin and the marker in the right bin.",
        "what": "two novel objects (tall can, thin marker) with two separate targets; known distractors stay",
        "objects": [O("natural", "can", "center"), O("natural", "marker", "right"), O("red", "cube", None),
                    O("yellow", "triangle", None)]},
    "S6_pan_czech": {
        "instruction": "Dej pánev do pravého boxu a všechno ostatní do levého boxu.",
        "what": "Czech instruction, small pan (handle); 5 objects incl. unseen colours",
        "objects": [O("black", "pan", "right"), O("green", "ball", "left"), O("blue", "cube", "left"),
                    O("red", "cylinder", "left"), O("white", "triangle", "left")]},
    "S7_ball_finnish": {
        "instruction": "Laita pallo vasempaan laatikkoon ja kaikki muut keskimmäiseen laatikkoon.",
        "what": "Finnish instruction (ball left, rest center), purple ball",
        "objects": [O("purple", "ball", "left"), O("yellow", "cylinder", "center"), O("blue", "cube", "center"),
                    O("green", "cylinder", "center")]},
    "S8_known_with_clutter": {
        "instruction": "Put the red objects in the left bin, the yellow objects in the center bin and the blue objects in the right bin.",
        "what": "control: the stock sort_fixed sentence with novel clutter (green ball, mug) that must stay untouched",
        "objects": [O("red", "cube", "left"), O("yellow", "cylinder", "center"), O("blue", "triangle", "right"),
                    O("green", "ball", None), O("natural", "mug", None)]},
}


def build_layout(scene_id, seed):
    """-> (layout for the scene service, gt {object name: bin|None}, meta). Deterministic in (scene, seed).
    Layout entries carry an extra `gt` key (ignored by the sim)."""
    sc = SCENES[scene_id]
    rng = np.random.default_rng(1000 * (list(SCENES).index(scene_id) + 1) + seed)
    objs = sc["objects"]
    n = len(objs)
    slots = list(range(n)) if n >= len(SLOTS) else sorted(rng.permutation(len(SLOTS))[:n].tolist())
    deep = [i for i in range(len(SLOTS)) if SLOTS[i][2]]
    big = [i for i, o in enumerate(objs) if o["shape"] in BIG]
    # choose n slots such that there are enough deep ones for the big objects, then give big objects deep slots
    for _ in range(200):
        pick = sorted(rng.permutation(len(SLOTS))[:n].tolist())
        if sum(1 for s in pick if s in deep) >= len(big):
            break
    pick_deep = [s for s in pick if s in deep]
    pick_rest = [s for s in pick if s not in deep] + pick_deep[len(big):]
    assign = {}
    for i, s in zip(big, rng.permutation(pick_deep)[:len(big)]):
        assign[i] = int(s)
    rest_slots = [s for s in pick if s not in assign.values()]
    for i, s in zip([i for i in range(n) if i not in big], rng.permutation(rest_slots)):
        assign[i] = int(s)
    layout, gt = [], {}
    counts = {}
    for i, o in enumerate(objs):
        x, y, _ = SLOTS[assign[i]]
        x += float(rng.uniform(-0.006, 0.006))
        y += float(rng.uniform(-0.006, 0.006))
        shape = o["shape"]
        if shape in ("ball", "pan", "banana", "mug", "can", "marker"):
            yaw = float(rng.uniform(-0.7, 0.7))               # pelvis yaw: handle / long axis roughly away from the robot
        else:
            yaw = float(rng.uniform(-np.pi / 4, np.pi / 4))   # stock mat-yaw range
        e = {"color": o["colour"], "shape": shape, "x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3), "gt": o["gt"]}
        layout.append(e)
        k = counts.get((o["colour"], shape), 0)
        counts[(o["colour"], shape)] = k + 1
        gt[f"obj_{o['colour']}_{shape}_{k}"] = o["gt"]
    return layout, gt


if __name__ == "__main__":
    for sid in SCENES:
        for s in range(3):
            lay, gt = build_layout(sid, s)
            print(sid, s, [(e["color"], e["shape"], e["x"], e["y"]) for e in lay])
