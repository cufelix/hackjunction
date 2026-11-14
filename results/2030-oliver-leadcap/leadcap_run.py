"""Lead-cap offline measurement. usage: leadcap_run.py TREE OVERLAY CONFIG MODE SEEDS  -> one JSON line per seed + summary."""
import sys, time, json, dataclasses, collections
tree, overlay, cfgname, mode, seeds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
sys.path.insert(0, tree)
import numpy as np
from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.baselines.classical.config import load_config
from aaltonomous.interface.safety import SafetyConfig, SafetyFilter, JointSpec
CFG = {"a_default": {}, "b_hand0.3": {"hand_max_lead": 0.3}, "c_hand0.15": {"hand_max_lead": 0.15},
       "d_arm0.3_hand0.3": {"arm_max_lead": 0.3, "hand_max_lead": 0.3}}[cfgname]
TL = 120.0
LOG = []
OVR = {'grasp.held_detector': 'stall'} if 'softgrip' not in overlay else {}   # mock never blocks fingers -> finger_sum reads every close as empty
FC = 0.8   # assumed closure at which a held cube blocks the fingers (soft+block mode)

def episode(seed):
    kw = {}
    if mode in ("soft", "softblock"): kw["hand_tau"] = 0.2 / 1.5   # kd/kp of real Dex3
    env = make_mock_env(seed=seed, **kw)
    sc = SafetyConfig(dt=1 / 30.0, max_clipped_steps=10**9, **CFG)   # uncapped so we see every clip; abort tested below
    env.filter = SafetyFilter(JointSpec(), sc)
    b = env.backend
    drops = []
    od = b._drop
    def _drop(ob):
        od(ob); drops.append(bool(ob["pos"][2] > -0.5 and env.scene.bin_of(env.scene.pelvis_to_mat(ob["pos"])) is not None))
    b._drop = _drop
    if mode == "softblock":
        adv = b.advance
        def advance(dt):
            adv(dt)
            for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
                if b.held[side] is not None:
                    o, c = b.post["open"][sl], b.post["closed"][sl]
                    if b._closure(side) > FC:
                        b.hand_q[sl] = o + FC * (c - o)
        b.advance = advance
    pol = ClassicalPolicy(config=load_config(overlay=overlay, overrides=OVR), safety=sc, time_limit=TL, log=LOG.append)
    obs = env.reset(); pol.reset({"instruction": obs["instruction"], "time_limit": TL})
    n = 0; steps_by = collections.Counter(); first = None
    while True:
        job = getattr(pol, "_job", None); pt = getattr(pol, "_pitch_thread", None); bv = getattr(pol, "_bv_job", None)
        if ((job is not None and job.get("result") is None) or (pt is not None and pt.is_alive())
                or any(j["result"] is None for j in list(pol._jobs.values())) or (bv is not None and bv.get("result") is None)):
            time.sleep(0.002); continue
        a = pol.act(obs); n += 1
        obs, info = env.step(a)
        for r in info["clip_reasons"]: steps_by[r.split(":")[0]] += 1
        if info["clipped"] and first is None: first = n
        if info["episode_over"] or n > 20000: break
    joints = collections.Counter()
    for ev in env.filter.events:
        for r in ("hand_lead", "arm_lead"):
            for j in ev.get("details", {}).get(r, []): joints[j] += 1
    out = b.outcomes(pol.mapping)
    return dict(seed=seed, picks=len(pol.plans), placed=sum(o == "correct" for o in out), wrong=sum(o == "wrong" for o in out),
                n_obj=len(out), clipped=env.filter.clipped_steps, would_abort=env.filter.clipped_steps > 30,
                by_reason=dict(steps_by), joints=dict(joints), bad_drops=sum(1 for d in drops if not d), end=info["end_reason"], t=round(info["t"], 1))

if "--log" in sys.argv: pass
rows = []
for s in range(seeds):
    r = episode(s); rows.append(r); print(json.dumps(r), flush=True)
print("SUMMARY", json.dumps(dict(cfg=cfgname, mode=mode, overlay=overlay, picks=sum(r["picks"] for r in rows),
      placed=sum(r["placed"] for r in rows), objs=sum(r["n_obj"] for r in rows), clips=sum(r["clipped"] for r in rows),
      aborts=sum(r["would_abort"] for r in rows), bad_drops=sum(r["bad_drops"] for r in rows))), flush=True)
