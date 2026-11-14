HANDOFF @claude-oliver/claude → @agent-opencode-oliver/qwen (Claude at 95 % usage). From now on, send Oliver's lane to OpenCode.

BOXES (adv sim H2; tools oliver/adv-sim 2a67b88, see tools/adv/REPLICATE.md; runs under /data/work/oliver/adv/runs):
- G sim c: ab_plane_table real_hard s1000 n10. G sim b (ADV_INSTANCE=b, port 8003): ab_plane_rim real_hard s1000 n10.
  Code: /data/work/oliver/adv/ck_7688de2 (= oliver/h2-overlays 7688de2 = felix/pitch-cal-fixes + overlays).
  Compare paired vs base real_hard cand_r16_nopp (F runs) / cand_r17 (+0.10, 2 correct / 1 wrong / 37 counter).
  First seed s1000: table 2/3, rim 1/3, base 0/3.
- F: cand_r17 real_mild s1000 n10 running; then queue /data/work/oliver/adv/h2qF.sh → ab_plane_cgate_live, ab_cgate_live
  real_hard from ck_96b07ab. ⚠ Before they start, swap to Felix's FINAL color-gate SHA (rules 2b workspace + 1b max
  area + Codex C1 fixes): git archive <sha> aaltonomous > tar, scp, extract to ck_<sha>, write a NEW h2qF.sh and
  `mv` it over (never edit a running bash script in place). Boxes have no git clone of our repo: ship code as a tar.
- ON rule: mean−std > base on real_hard, nominal ≥ base −1 obj, 0 stops, 0 extra wrong, ≤2 % time.

RUNNING HAIKU WORKERS (finish on their own; review results before pushing):
1. J8 overlap_look: ~/work/overlap-look (branch oliver/overlap-look on felix/rc1). Review the diff, check that equivalence
   is 13/13 IDENTICAL vs clean rc1 and the full suite passes, then push + kind=done with SHA to Timi/Felix.
2. Colour sweep: scratchpad/colour_sweep/REPORT.md (robust vs robust_v2 confusion) → post the recommendation to Felix
   (colour P0 robust_v2: red 0-22, yellow 24-52, 22-24 abstain).
3. Robustness table → results/1530-oliver-robustness-table/ (pushes itself).

DONE TODAY (all pushed): oliver/inject-cam-error 8946194, oliver/hand-calib-v4 d11bb1b (in rc1, OFF),
oliver/adv-sim 2a67b88 + H1 loss breakdown, oliver/robustness-table 444f138 (W6), oliver/promote 0be4817 (J10),
results: DM material 1450-oliver-adv-dm (DM_DRAFT.md approved by Timi; Oliver sends it), adv logs 1515.
Finding: the −1 in base real_hard s1001 = orange cube read as yellow (top-face H 19-21 is inside robust yellow 18-52).

TODO NEXT: A/B Felix's robust_v2 on G real_hard + nominal when the plane arms finish; add the finished runs to the
1515 logs; then H2 order: ab_plane_b1_inj_p7, ab_vote, ab_iso.
RULES: no IPs/root@/keys in commits or Warren; drop_2 ep 40-47 held-out; flags default OFF + equivalence 13/13.
