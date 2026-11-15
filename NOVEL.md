# R22 novel-object adaptivity

## Design

The known orange/purple/yellow (sim red/blue/yellow) detector and its shape/bin path remain authoritative. The new path is four independent, default-off flags:

| Flag | Responsibility | Failure behaviour |
|---|---|---|
| `perception.novel_objects` | Find 1–25 cm above-counter depth components inside the counter workspace, outside bin footprints and the FK robot mask; describe footprint, PCA axes, height, broad colour and coarse geometry. | Return the already accepted known detections. |
| `grasp.generic` | Make a top-down minor-axis pinch, or move the point onto a detected handle. The maximum width is the open thumb-to-finger span measured from our G1/Dex3 URDF minus 12 mm. | `UNGRASPABLE`; no plan. |
| `instruction.novel_objects` | Ground a destination only when one visible novel candidate uniquely matches the named colour/shape synonym. | No bin means no pick. An unknown noun may be offered to the existing bounded asynchronous brain as a closed visible-novel list. |
| `policy.novel_obstacles` | Add every other visible novel footprint/height as a 3-D keep-out during hand-trajectory checks. | Reject the plan and try another candidate. |

Depth is sampled on a 3-pixel grid before connected components. Candidate geometry is then metric in the pelvis frame, not inferred from apparent image size. The measured incremental median in the synthetic 640×480 regression is below 8 ms.

Known-task protection is asymmetric on purpose:

- A known-size component near an existing colour detection stays on the old path.
- Known-size geometry in any known colour family stays exclusively on the old path even if its colour detector abstains. Novel perception cannot bypass hue/placement gates.
- A novel detection never inherits the fixed colour-to-bin mapping. Only a unique novel instruction match—or a validated brain choice for the instruction’s named bin—sets `novel_target_bin`.
- The brain sees only currently visible novel candidate IDs in this fallback, cannot select a known object, and cannot change the instruction’s destination bin. Its existing worker and absolute timeout keep `act()` non-waiting.
- Width, confidence, target match and IK/safety checks must all pass. Unmatched and ungraspable objects remain on the counter.

The pan protrusion test first requires a metrically elongated thick body, then looks for a coherent, one-sided thin extension. Its grasp centre is computed only from handle depth pixels. Palm-x follows the handle, so thumb-to-finger closure is perpendicular to the handle axis; the disc is never the grasp target.

Use `overlays/finals_r22_novel.yaml` for the complete finals-r21 behaviour plus all four novel layers.

## Judge-facing sim demo

1. Start with a normal known cube beside a green ball, black box, red marker and toy pan. Show the overlay labels: the cube remains `red/cube/depth`; the four additions read `green/round`, `black/box-like`, `red/elongated`, and `blue/with-handle`.
2. Say “green ball left”. Highlight that only the green round candidate receives `left`; every other novel object is rendered as an obstacle rather than a candidate.
3. Show the planned ball pinch and the URDF-derived 104 mm conservative opening gate.
4. Say “red marker right”. Show palm alignment along the marker and closure across its minor width.
5. Say “blue pan center”. Draw the pan body centre, handle axis, handle-only grasp point and perpendicular closing axis. The grasp marker must sit on the handle, not the disc.
6. Say an unresolved phrase such as “put the gizmo right”. With the brain disabled/stubbed, show `ABSTAIN` and no motion. If demonstrating the brain, show the closed candidate IDs and timeout; a known cube or a different bin must be rejected.
7. End by switching all four flags off and replaying the known scene, showing identical known detections/actions.

Keep the real-robot demo conservative: one object at a time, clear spacing, low first-object speed, and no claim that an object is graspable until the overlay shows width and confidence above their gates.

