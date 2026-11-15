"""motion.arm_tau_ff = gravity: feed-forward arm torque for the REAL robot (organiser PR #4, action key `arm_tau_ff`).

The runner's arm PD (kp 80 shoulders/elbow, 40 wrists, target clipped to measured +-0.12 rad) has no gravity term; the
teleop app that recorded the demos sends tau = rnea(q_target, 0, 0). Without it the loaded arm sags below its command.
tau here = pinocchio generalized gravity of the full G1 URDF (fixed pelvis, legs/waist at 0) at the COMMANDED arm q and
the commanded hand q, in the runner's arm_q row order, scaled by kappa and clamped like the runner (+-15 Nm shoulders/
elbow, +-2.5 Nm wrists, never above the URDF effort), NaN -> 0. Own pinocchio model: callable from any thread.
"""
from __future__ import annotations

import numpy as np
import yaml

from aaltonomous.interface.safety import DEFAULT_SPEC, DEFAULT_URDF, JointSpec

TAU_MAX = np.array([15.0, 15.0, 15.0, 15.0, 2.5, 2.5, 2.5] * 2)   # = organiser env.ARM_TAU_MAX


class GravityFF:
    def __init__(self, spec: JointSpec | None = None, urdf=DEFAULT_URDF, kappa=1.0):
        import pinocchio as pin
        self.pin = pin
        self.spec = spec or JointSpec()
        self.kappa = float(kappa)
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.q0 = pin.neutral(self.model)
        idx = lambda n: self.model.joints[self.model.getJointId(n)]
        self.arm_iq = np.array([idx(n).idx_q for n in self.spec.arm_names])
        self.arm_iv = np.array([idx(n).idx_v for n in self.spec.arm_names])
        self.hand_iq = np.array([idx(n).idx_q for n in self.spec.hand_names])
        with open(DEFAULT_SPEC) as f:
            effort = np.array([j["effort"] for j in yaml.safe_load(f)["arm"]], float)
        self.tau_max = np.minimum(TAU_MAX, effort)

    def row(self, arm_q, hand_q=None):
        """(14,) torque for one commanded pose."""
        q = self.q0.copy()
        q[self.arm_iq] = np.asarray(arm_q, float)
        if hand_q is not None:
            q[self.hand_iq] = np.asarray(hand_q, float)
        g = self.pin.computeGeneralizedGravity(self.model, self.data, q)[self.arm_iv]
        tau = np.nan_to_num(self.kappa * g, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(tau, -self.tau_max, self.tau_max)

    def rows(self, arm_q, hand_q=None):
        """(k,14) for (k,14) targets, (14,) for a (14,) target; None if the shape is not a pose or chunk."""
        a = np.asarray(arm_q, float)
        if a.ndim == 1 and a.shape == (14,):
            return self.row(a, hand_q)
        if a.ndim != 2 or a.shape[1] != 14:
            return None
        h = None if hand_q is None else np.asarray(hand_q, float)
        return np.stack([self.row(a[i], None if h is None or h.ndim != 2 or len(h) != len(a) else h[i])
                         for i in range(len(a))])
