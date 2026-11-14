# Team

Team name: hackjunction

Fill every bracket before you submit. Do not paste personal details from memory.

## Members

| Name | Role | Contact | Links |
|---|---|---|---|
| Oliver Seidl | Perception and robustness: adversarial sim, real-frame recall audit, camera calibration | [email] | [LinkedIn] [GitHub] |
| Timi | Coordination, language, real-robot strategy | [email] | [LinkedIn] [GitHub] |
| Felix | Integration and release: release candidates, resolver, promote and gate | [email] | [LinkedIn] [GitHub] |

Add for each person if the form asks: [city / country], [school or employer], [skills], [photo].

## What each person did

Evidence is the commit history in the team repo. Git author names are handles: `Oliver Seidl` and `oliver-claude` are Oliver, `Dymyt-ry` is Timi, `cufelix` is Felix.

**Oliver** (commits by `Oliver Seidl` and `oliver-claude`)
- Adversarial sim and its reports: `tools/adv/`, lying prisms and layout knobs, and the principal-point result.
- Real-frame recall audit on `drop_2`: `tools/recall_ab.py`.
- Calibration: counter plane, camera pitch fit, head pose from the arms in depth, principal point, `inject_cam_error` test flag.
- Readiness document and `promote.sh` tests: `READINESS.md`.

**Felix** (commits by `cufelix`, mostly release candidates)
- Release candidates rc1 onward, `promote.sh`, release gate and the expected-flags check.
- Resolver and language normaliser work (strict abstain, colour names, Finnish and Czech words), see `aaltonomous/baselines/classical/instruction.py` and `normalize.py`.
- Also calibration code (`calib.plane.apply_height`, bin vote), the fault-drill harness and the architecture doc.

**Timi** (commits by `Dymyt-ry`)
- Language: structured instruction passes, Finnish and Czech support.
- Optional language model sidecar (Qwen3-VL).
- Grasp work: cage grasp, staggered release, prism grasp, wrist alignment.
- Fault-drill suite, hand-in script and Wendy package.

## Open items

- [Confirm the real-robot strategy line for Timi. Commits cannot show it.]
- [Fill contact details.]
