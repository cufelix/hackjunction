# Real-frame recall A/B (drop_2 ep 0-39, clean GT)

Tool: `tools/recall_ab.py` on branch `oliver/recall-ab`. Visible-only recall uses the E1 fair-recall contract (greedy, 25 px,
conservative notes + current-pose FK). It reproduces the published baseline exactly (submission 321/538 = 0.597, every
class identical). drop_2 has NO head depth, so every arm runs with depth=None: a worst case for depth-dependent gates.
robot_q comes from the recorded joint state. Episodes 40-47 were never opened.

- `RECALL-AB.md`: Timi's arms (finals_r21, finals_r22_recall, each recallfix flag alone).
- `RECALL-ATTRIB.md`: which finals_r21 flag costs the recall.
- `arms/*/arm.json` (tree, sha, overlay, overrides) + `fair_recall.json`.

Headline: no recallfix arm beats 0.597 without raising FP. In finals_r21 the colour gate costs 87 visible objects
(0.362 -> 0.524 without it), and hue_guard costs 23 objects (0.405 without it) for +5 FP.
