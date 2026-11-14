"""policy.safety_shell helpers: 20 % reserve against the SafetyFilter limits (pure numpy, no policy state).

The SafetyFilter (interface/safety.py) limits only per-joint command velocity (arm_max_vel, hand_max_vel); it has no
acceleration or jerk limit, so only velocity is checked here. A trajectory that would exceed (1 - reserve) x limit is
stretched in time (never compressed) until it fits; a trajectory that already fits is returned untouched."""
from __future__ import annotations

import dataclasses

import numpy as np

EPS = 1e-9


def slow_factor(path, start, limit_step):
    """>= 1: how many times slower `path` (n, d) must run so no joint steps more than `limit_step` per tick."""
    if path is None or not len(path):
        return 1.0
    steps = np.abs(np.diff(np.vstack([np.asarray(start, float)[None], np.asarray(path, float)]), axis=0))
    return max(1.0, float(steps.max()) / limit_step)


def stretch(path, start, factor):
    """Same polyline, ceil(n*factor) ticks (linear interpolation, ends on path[-1]): speed <= 1/factor of before."""
    path = np.asarray(path, float)
    n = len(path)
    m = int(np.ceil(n * factor - EPS))
    pts = np.vstack([np.asarray(start, float)[None], path])
    u = np.arange(1, m + 1) * (n / m)
    return np.stack([np.interp(u, np.arange(n + 1), pts[:, c]) for c in range(path.shape[1])], axis=1)


def reserve_factor(arm, hand, arm0, hand0, dt, arm_vmax, hand_vmax, reserve):
    lim = 1.0 - reserve
    return max(slow_factor(arm, arm0, arm_vmax * lim * dt), slow_factor(hand, hand0, hand_vmax * lim * dt))


def reserve_pair(arm, hand, arm0, hand0, *limits):
    """(arm, hand) scaled together to the reserve; unchanged (same objects) when already within it."""
    k = reserve_factor(arm, hand, arm0, hand0, *limits)
    if k <= 1.0 + EPS or not len(arm):
        return arm, hand
    return stretch(arm, arm0, k), stretch(hand, hand0, k)


def reserve_plan(plan, arm0, hand0, *limits):
    """planner.Plan scaled to the reserve (copy); phases / close_hold / abort branch re-indexed to the new tick grid."""
    if plan is None or not getattr(plan, "ok", False) or plan.arm is None or not len(plan.arm):
        return plan
    k = reserve_factor(plan.arm, plan.hand, arm0, hand0, *limits)
    if k <= 1.0 + EPS:
        return plan
    n = len(plan.arm)
    f = int(np.ceil(n * k - EPS)) / n
    ix = lambda i: int(round(i * f))
    abort = plan.abort
    if abort is not None:
        ls, aa, ah = abort
        a0 = plan.arm[ls - 1] if ls > 0 else arm0
        h0 = plan.hand[ls - 1] if ls > 0 else hand0
        abort = (ix(ls), stretch(aa, a0, f), stretch(ah, h0, f)) if len(aa) else abort
    return dataclasses.replace(
        plan, arm=stretch(plan.arm, arm0, k), hand=stretch(plan.hand, hand0, k),
        phases=[(nm, ix(s), ix(e)) for nm, s, e in plan.phases],
        close_hold=None if plan.close_hold is None else tuple(ix(i) for i in plan.close_hold), abort=abort)
