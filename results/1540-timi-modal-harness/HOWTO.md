# Modal A/B harness (for running on another Modal account)

Files: `aalto_eval.py` (Modal app: Isaac sim + policy per seed, L40S, max_containers=10), `ab_arm.sh` (one A/B arm = git archive of a worktree rev + overlay), `ab_report.py` (paired table). `<HJ_ROOT>` = the parent dir that holds `modal/` and a Python venv with `modal` installed (`.venv-modal`).

1. Put these three files into `<HJ_ROOT>/modal/`, create `<HJ_ROOT>/modal/logs` and `<HJ_ROOT>/modal/runs`, `modal token set` for your account.
2. Read `README.md` + the top of `aalto_eval.py` (it builds the Isaac image once; first run is slow).
3. One arm: `bash modal/ab_arm.sh <overlay> <seeds e.g. 1040-1069> <git worktree with the policy repo> <rev>` -> `modal/runs/<overlay>_<sha>/`.
4. Report: `python3 modal/ab_report.py modal/runs/<base_dir> modal/runs/<arm_dir>`.
Pair arms on the SAME tree/rev and the SAME seeds. Watch Modal credit.
