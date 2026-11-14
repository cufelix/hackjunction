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

Evidence is the commit history on `origin/oliver/*`, `origin/timi/*` and `origin/felix/*` in the team repo. Git author names are handles. We mapped them as `Oliver Seidl` and `oliver-claude` to Oliver, `Dymyt-ry` to Timi (the author of the `timi/*` snapshot commits) and `cufelix` to Felix. [Confirm the handles.]

**Oliver** (about 40 commits by `Oliver Seidl` and `oliver-claude` across 22 `oliver/*` branches)
- Adversarial sim and its reports: `oliver/adv-sim`, `oliver/adv-real2`, lying prisms and layout knobs.
- Real-frame recall audit on `drop_2`: `oliver/recall-ab`, `oliver/drop2-analysis`.
- Calibration: counter plane, camera pitch fit, head pose from the arms in depth, principal point, `inject_cam_error` test flag.
- Readiness document and `promote.sh` tests: `oliver/readiness`.
- Matches the role you gave.

**Felix** (the most commits, mostly on `felix/*` release-candidate branches)
- Release candidates `felix/rc1` to `felix/rc72-*`, `promote.sh`, release gate and the expected-flags check.
- Resolver and language normaliser work (strict abstain, colour names, Finnish and Czech words), see the `felix/lang*` branches.
- Also calibration code (`calib.plane.apply_height`, bin vote), the fault-drill harness and the architecture doc.
- Matches the role. Add language, calibration code and fault drills.

**Timi** (many commits on `timi/*` branches, author `Dymyt-ry`)
- Language: structured instruction passes, Finnish and Czech support (`timi/multilang`, `timi/fixlang`, `timi/langzip`).
- Optional language model sidecar: `timi/brain*` branches.
- Grasp work: cage grasp, staggered release, prism grasp, wrist alignment.
- Fault-drill suite, hand-in script and Wendy package (`timi/wendy-pkg`).
- Language matches. The git history does not show coordination or real-robot strategy, since those are not commits. Add the sidecar, grasp work, fault drills and packaging.

## Open items

- [Confirm the handle to person mapping above.]
- [Confirm the real-robot strategy line for Timi. Git history cannot show it.]
- [Fill contact details.]
