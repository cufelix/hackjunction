"""Pick-and-place planning for one detected object: Cartesian waypoints -> IK -> 30 Hz joint trajectories.

Sequence (one arm; the other holds its current target):
  home -> pregrasp at carry height (joint space, hand to preshape) -> descend (Cartesian) -> close -> lift -> carry
  above the bin (Cartesian, else joint space direct / via a raised midpoint) -> lower to drop height -> release
  (hand.release) -> retreat up -> home (hand back to open)
Each side is tried with every grasp.finger_hints direction (fingers forward can touch the centre bin's wall).
Cross-body (neither arm can do object -> bin): a RELOCATION plan picks with one arm and puts the object down on
the counter at a spot from which the bin-side arm has a full plan; the policy's next look picks it up from there.
Every planned tick is checked before it is used:
  * per-joint speed <= motion.arm_max_vel (trapezoidal time scaling on the inf-norm path length),
  * hand points (palm, grasp point, finger links) over the counter footprint stay above the counter
    (counter_clearance in transit; fingertip_clearance during the grasp itself),
  * hand points over a bin footprint (+1 cm) stay above its rim, except while releasing,
  * a dry run through a fresh SafetyFilter (same config as the env) produces 0 clipped ticks.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import time

import numpy as np

from aaltonomous.interface.safety import JointSpec, SafetyConfig, SafetyFilter
from aaltonomous.interface.tasks import FIXED_MAPPING
from aaltonomous.scene.scene import resolve_bin

from .bin_vote import bin_geometry
from .kinematics import ArmKinematics
from .config import soft_hand_active, soft_overclose


def lookup_bin(cfg, mapping, color, shape=None):
    """Target bin of a detection. policy.bin_lookup: colour (stock: mapping[colour]) | resolve_bin (scene.resolve_bin:
    (colour, shape) -> colour -> shape -> "default"; shape None = abstain -> only colour / default apply).
    None or "abstain" = do not pick (the object stays on the counter: 0 points, never -1)."""
    if (cfg.get("policy") or {}).get("bin_lookup", "colour") != "resolve_bin":
        return mapping.get(color)
    b = resolve_bin(mapping, color, shape)
    return None if b in (None, "abstain") else b


def lookup_detection_bin(cfg, mapping, det):
    """Target for a detection, resolving generic grasps without a shape guess."""
    if getattr(det, "novel", False):
        return getattr(det, "novel_target_bin", None)
    if getattr(det, "generic_required", False):
        if getattr(det, "generic_width", None) is None:
            return None
        b = resolve_bin(mapping, det.color, None)
        return None if b in (None, "abstain") else b
    return lookup_bin(cfg, mapping, det.color, getattr(det, "shape", None))


@dataclass
class Plan:
    ok: bool
    side: str | None = None
    color: str = ""
    reason: str = ""
    target_bin: str | None = None
    source_bin: str | None = None
    drop_slot: int | None = None
    corrective: bool = False
    waypoints: dict = field(default_factory=dict)      # name -> grasp point (pelvis)
    ik: dict = field(default_factory=dict)             # name -> (ok, pos_err, tilt_deg)
    arm: np.ndarray | None = None                      # (T, 14)
    hand: np.ndarray | None = None                     # (T, 14)
    phases: list = field(default_factory=list)         # (name, start tick, end tick)
    posture: tuple | None = None                       # (posture, shape label, conf, path) with grasp.close_posture_by_shape
    close_hold: tuple | None = None                    # (first, end) tick of the closed-hand hold (lift_on_stall)
    release_stages: tuple | None = None                # (first start, DDS 5/6 start, ramp end) ticks
    abort: tuple | None = None                         # legacy field; early servo-error abort is intentionally disabled
    safety_clips: int | None = None
    deferred: bool = False                             # planner.candidate_budget: planning budget exceeded (not infeasible)
    grasp_key: tuple | None = None                     # policy.retry_vary_grasp: (grasp xy, finger heading, posture)

    @property
    def duration(self):
        return 0.0 if self.arm is None else len(self.arm) / 30.0

    def summary(self):
        if not self.ok:
            return f"{self.color:<6} SKIP ({self.reason})"
        wp = " ".join(f"{k}=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})" for k, v in self.waypoints.items()
                      if k in ("grasp", "drop"))
        tilt = max(t for _, _, t in self.ik.values())
        err = max(e for _, e, _ in self.ik.values()) * 1000
        return (f"{self.color:<6} {self.side:<5} arm -> {self.target_bin} bin  {wp}  IK ok max_err={err:.1f}mm "
                f"max_tilt={tilt:.0f}deg  {self.duration:.1f}s  safety_clips={self.safety_clips}")


@dataclass
class SafeRoute:
    """Proved open -> vertical escape -> home route, or a fail-closed reason with no commands."""
    ok: bool
    arm: np.ndarray | None = None
    hand: np.ndarray | None = None
    phases: list = field(default_factory=list)
    reason: str = ""


def time_scale(Qs, vmax, amax, dt):
    """Piecewise-linear joint path through Qs (n, d) -> samples (T, d) at dt, EXCLUDING Qs[0], with a trapezoidal
    profile on the inf-norm arc length: no joint ever moves faster than vmax."""
    Qs = np.asarray(Qs, float)
    seg = np.abs(np.diff(Qs, axis=0)).max(axis=1)
    S = float(seg.sum())
    if S < 1e-9:
        return np.empty((0, Qs.shape[1]))
    ta = vmax / amax
    if amax * ta * ta > S:                       # triangle profile
        ta = np.sqrt(S / amax)
        vpk = amax * ta
        T = 2 * ta
    else:
        vpk = vmax
        T = 2 * ta + (S - amax * ta * ta) / vmax
    n = max(1, int(np.ceil(T / dt)))
    t = np.arange(1, n + 1) * (T / n)
    s = np.where(t < ta, 0.5 * amax * t ** 2,
                 np.where(t <= T - ta, 0.5 * amax * ta ** 2 + vpk * (t - ta), S - 0.5 * amax * (T - t) ** 2))
    s = np.clip(s, 0, S)
    cum = np.r_[0.0, np.cumsum(seg)]
    out = np.empty((n, Qs.shape[1]))
    for k, sk in enumerate(s):
        i = min(int(np.searchsorted(cum, sk, side="right")) - 1, len(seg) - 1)
        f = 0.0 if seg[i] < 1e-12 else (sk - cum[i]) / seg[i]
        out[k] = Qs[i] + f * (Qs[i + 1] - Qs[i])
    return out


def ramp(a, b, vmax, dt, amax=None):
    """Move a -> b (excluding a); legacy linear ramp, or acceleration-shaped when ``amax`` is supplied."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if amax is not None:
        return time_scale([a, b], float(vmax), float(amax), dt)
    n = int(np.ceil(np.abs(b - a).max() / (vmax * dt) + 1e-9))
    if n == 0:
        return np.empty((0, len(a)))
    return a + (b - a) * (np.arange(1, n + 1) / n)[:, None]


class MotionOutputLimiter:
    """Cheap stateful per-joint velocity/acceleration limiter for the policy's final command stream.

    Planner trajectories are normally already within the same limits.  Keeping this second, stateful boundary is
    what makes queue replacement (home/endgame/empty-grasp abort) obey them too.  ``now`` accounts for a delayed
    policy request; samples within an action chunk remain spaced by the runner's fixed control period.
    """

    def __init__(self, arm_vmax, arm_amax, hand_vmax, hand_amax, dt, arm_lim=None, hand_lim=None):
        self.arm_vmax, self.arm_amax = float(arm_vmax), float(arm_amax)
        self.hand_vmax, self.hand_amax = float(hand_vmax), float(hand_amax)
        self.dt = float(dt)
        # (lower, upper) URDF joint limits: the acceleration-shaped step can overshoot a target at the limit.
        self.arm_lim, self.hand_lim = arm_lim, hand_lim
        self.reset()

    def reset(self):
        self.arm_q = self.hand_q = None
        self.arm_v = self.hand_v = None
        self.last_time = None

    @staticmethod
    def _step(target, q, velocity, vmax, amax, dt):
        desired_v = (target - q) / dt
        velocity = np.clip(desired_v, velocity - amax * dt, velocity + amax * dt)
        velocity = np.clip(velocity, -vmax, vmax)
        return q + velocity * dt, velocity

    def limit(self, arm, hand, arm_start, hand_start, now):
        arm, hand = np.asarray(arm, float), np.asarray(hand, float)
        if not len(arm):
            return arm.copy(), hand.copy()
        if arm.shape != hand.shape or arm.ndim != 2 or arm.shape[1] != 14:
            raise ValueError(f"motion output wants matching (n,14) paths, got arm={arm.shape} hand={hand.shape}")
        now = float(now)
        if self.arm_q is None:
            self.arm_q, self.hand_q = np.asarray(arm_start, float).copy(), np.asarray(hand_start, float).copy()
            self.arm_v, self.hand_v = np.zeros(14, float), np.zeros(14, float)
            self.last_time = now
        # The previous chunk's last command is due at last_time.  A late request grants the first new target the
        # actual extra elapsed time; normal 30 Hz chunks use exactly dt.
        first_time = max(float(self.last_time) + self.dt, now + self.dt)
        first_dt = first_time - float(self.last_time)
        out_a, out_h = np.empty_like(arm), np.empty_like(hand)
        sample_time = float(self.last_time)
        for i in range(len(arm)):
            step_dt = first_dt if i == 0 else self.dt
            self.arm_q, self.arm_v = self._step(
                arm[i], self.arm_q, self.arm_v, self.arm_vmax, self.arm_amax, step_dt)
            self.hand_q, self.hand_v = self._step(
                hand[i], self.hand_q, self.hand_v, self.hand_vmax, self.hand_amax, step_dt)
            if self.arm_lim is not None:
                self.arm_q = np.clip(self.arm_q, *self.arm_lim)
            if self.hand_lim is not None:
                self.hand_q = np.clip(self.hand_q, *self.hand_lim)
            out_a[i], out_h[i] = self.arm_q, self.hand_q
            sample_time = first_time if i == 0 else sample_time + step_dt
        self.last_time = sample_time
        return out_a, out_h


def staggered_release_ramp(a, b, side, vmax, dt, delay_s):
    """Release thumb + DDS slots 3/4 now and native DDS slots 5/6 after ``delay_s``.

    The returned path is precomputed, so policy ``act()`` only streams targets. ``second_tick`` is the zero-based
    tick offset at which slots 5/6 first change.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("staggered release endpoints must be equal-length vectors")
    if side not in ("left", "right"):
        raise ValueError(f"unknown hand side {side!r}")
    off = 0 if side == "left" else 7
    delayed = np.zeros(len(a), bool)
    delayed[off + 5:off + 7] = True
    duration = np.abs(b - a) / float(vmax)
    second_tick = int(np.ceil(float(delay_s) / float(dt) - 1e-12))
    offsets = np.where(delayed, second_tick, 0)
    move_ticks = np.ceil(duration / float(dt) + 1e-9).astype(int)
    n = int(np.max(offsets + move_ticks))
    if n == 0:
        return np.empty((0, len(a))), 0
    elapsed = np.maximum(0.0, (np.arange(n, dtype=float)[:, None] + 1.0 - offsets[None]) * float(dt))
    frac = np.ones((n, len(a)), float)
    moving = duration > 0.0
    frac[:, moving] = np.minimum(1.0, elapsed[:, moving] / duration[moving])
    out = a + (b - a) * frac
    out[-1] = b
    return out, second_tick


class Planner:
    _lying = False   # grasp.lying_prism: the pick being planned is a lying prism
    def __init__(self, cfg, scene, postures, home, kin: ArmKinematics | None = None, safety: SafetyConfig | None = None,
                 spec: JointSpec | None = None, mapping=None, dt=1.0 / 30.0):
        self.cfg, self.scene, self.post, self.home = cfg, scene, postures, np.asarray(home, float)
        self.spec = spec or JointSpec()
        self.kin = kin or ArmKinematics(cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=dt)
        self.dt = dt
        self._posture = None   # grasp.close_posture_by_shape: posture of the pick being planned
        self._apex = None      # grasp.posture_apex_gate: (pass, apex_conf, max angle) of the pick being planned
        self.vary = None       # policy.retry_vary_grasp: {"posture": name | None, "yaw": rad | None} for the next plan()
        self._lying = False    # grasp.lying_prism: the pick being planned is a lying prism (grasp height / posture)
        self._carry_hint = None   # policy.retry_vary_grasp: (hint, weight) for carry/lower/retreat of a heading variant
        self.object_speed_scale = None  # policy.first_object_speed_scale: arm phase speed x this (a copy, per plan)
        self.height = float(cfg["perception"].get("object_height") or scene.object_height())
        self.mapping = mapping or dict(FIXED_MAPPING)  # sort_fixed: fixed colour -> bin position
        self.bin_shift = {}    # bin.vote: bin name -> xy shift (m) added to bin_xy/drop_xy/drop_slots; {} = nominal
        self.drop_extra = {}   # bin.vote: bin name -> extra release height (m) when the bin position is low-confidence
        self.log = None          # perception.neutral_obstacles: one-line fail-open log (policy sets it when the flag is on)
        self._neutral_deadline = None   # plan(): monotonic time after which the obstacle-constrained search stops
        self._neutral_fallback = None   # plan(): first complete plan that only failed the obstacle check
        self.neutral_src = None  # perception.neutral_obstacles: object with .neutral_obstacles (set by the policy); None = off
        self.bin_conf = {}     # bin.vote: bin name -> confidence in bin_shift (1.0 agree, 0.5 prior only, 0.3 disagree)
        self.generic_hand_opening = None
        if cfg["grasp"].get("generic", False) or (cfg.get("brain") or {}).get("find_to_detection", False):
            tips = self.kin.tips_in_palm("left", self.post["open"])
            measured = float(np.linalg.norm(tips[:2].mean(0) - tips[2]))
            self.generic_hand_opening = max(
                0.0, measured - float(cfg["grasp"].get("generic_opening_margin_m", 0.012)))

    @property
    def counter_keepout_z(self):
        """Planner-side table keepout source; one scene assignment updates every planner consumer atomically."""
        return self.scene.counter_z

    def commit_counter_height(self, z):
        """Commit the accepted episode height before planning any motion (single shared source of truth)."""
        self.scene.counter_z_runtime = None if z is None else float(z)

    def hand_ramp(self, start, target):
        """Velocity-only legacy ramp, acceleration-shaped only when motion.output_caps is enabled."""
        mo = self.cfg["motion"]
        amax = mo.get("hand_max_acc") if mo.get("output_caps") is not None else None
        return ramp(start, target, mo["hand_max_vel"], self.dt, amax=amax)

    # ------------------------------------------------------------------ geometry helpers
    def _bin_xy0(self, name):
        return self.scene.mat_to_pelvis(np.r_[self.scene.cfg["bins"]["centers"][name], 0.0])[:2]

    def bin_geometry(self, name=None, shift=None):
        """Runtime bin geometry (bin_vote.bin_geometry) of the planner's current bin_shift; `shift` overrides (tests)."""
        return bin_geometry(self.scene, self.bin_shift if shift is None else shift, name)

    def bin_xy(self, name):
        return self.bin_geometry(name)["centre_pelvis"]

    def drop_xy(self, name, inset="cfg"):
        c = self._drop_xy0(name, inset)
        sh = self.bin_geometry(name)["shift"]
        return c + sh if sh is not None else c

    def _drop_xy0(self, name, inset="cfg"):
        """Release point (pelvis xy) in bin `name`: the point of the inner footprint nearest the pelvis, `grasp.drop_inset`
        inside the inner walls (the published box row puts the box centres past the arm's reach). None -> centre.
        `inset` other than "cfg" overrides grasp.drop_inset (the novel near-rim retry uses 0.0)."""
        if inset == "cfg":
            inset = self.cfg["grasp"].get("drop_inset")
        c = self._bin_xy0(name)
        if inset is None:
            return c
        lo, hi = self.scene.bin_inner_box(name)
        a, b = self.scene.mat_to_pelvis(np.r_[lo[:2], 0.0])[:2], self.scene.mat_to_pelvis(np.r_[hi[:2], 0.0])[:2]
        lo2 = np.minimum(np.minimum(a, b) + inset, c)
        hi2 = np.maximum(np.maximum(a, b) - inset, c)
        return np.clip(np.zeros(2), lo2, hi2)

    def drop_slots(self, name):
        out = self._drop_slots0(name)
        sh = self.bin_geometry(name)["shift"]
        return [p + sh for p in out] if sh is not None else out

    def _drop_slots0(self, name):
        """Reach-biased release points in the near-inner-wall band of a bin.

        The first point is closest to the robot centre line.  In legacy mode,
        consecutive points are at least ``drop_slot_spacing`` apart and every
        object centre keeps ``drop_slot_wall_margin`` from the inner side walls.  With
        ``drop_slot_grid`` on, two reach-biased rows use the physical object
        width plus 1 mm as their collision-free centre spacing; the stock
        three slots per row therefore become six.
        """
        g = self.cfg["grasp"]
        spacing = float(g.get("drop_slot_spacing", 0.06))
        margin = float(g.get("drop_slot_wall_margin", 0.03))
        count = max(1, int(g.get("drop_slot_count", 3)))
        lo, hi = self.scene.bin_inner_box(name)
        corners = self.scene.mat_to_pelvis(np.array([[x, y, 0.0] for x in (lo[0], hi[0])
                                                     for y in (lo[1], hi[1])]))
        xlo, xhi = corners[:, 0].min(), corners[:, 0].max()
        ylo, yhi = corners[:, 1].min() + margin, corners[:, 1].max() - margin
        x = min(xhi, xlo + float(g.get("drop_inset") or 0.0))
        grid = bool(g.get("drop_slot_grid", False))
        if grid:
            # Dense enough for six objects but not for overlap: the extra millimetre avoids a floating-point
            # equality at exactly one object width.  Start at the wall margin (rather than drop_inset) to keep
            # row 2 close to the legacy side-bin reach envelope.
            spacing = float(self.scene.size) + 0.001
            margin = max(margin, float(self.scene.size) / 2)
            ylo, yhi = corners[:, 1].min() + margin, corners[:, 1].max() - margin
            x = xlo + margin
        if ylo > yhi:
            return [self._drop_xy0(name)]
        if ylo <= 0.0 <= yhi:
            ys = [0.0]
            for k in range(1, count):
                step = ((k + 1) // 2) * spacing
                ys.append(step if k % 2 else -step)
        else:
            start = ylo if ylo > 0.0 else yhi
            direction = 1.0 if ylo > 0.0 else -1.0
            ys = [start + direction * spacing * k for k in range(count)]
        row = []
        for y in ys:
            if ylo - 1e-9 <= y <= yhi + 1e-9 and all(abs(y - p[1]) >= spacing - 1e-9 for p in row):
                row.append(np.array([x, np.clip(y, ylo, yhi)]))
        if grid:
            xs = [x0 for x0 in (x, x + spacing) if x0 <= xhi - margin + 1e-9]
            out = [np.array([x0, p[1]]) for x0 in xs for p in row]
        else:
            out = row
        # Slots only (#6 or grasp.drop_slots; never used with both off): drop slots the head camera cannot verify
        excl = [np.asarray(e, float) for e in (g.get("drop_slot_exclude") or [])]
        keep = [p for p in out if all(np.linalg.norm(p - e) > 0.005 for e in excl)]
        out = keep or out
        return out

    def _over_counter(self, p):
        pm = self.scene.pelvis_to_mat(p)
        c = self.scene.cfg["counter"]
        mat, size = np.asarray(c["mat_size"], float), np.asarray(c["counter_size"], float)
        lo, hi = mat / 2 - size / 2, mat / 2 + size / 2
        return np.all((pm[..., :2] >= lo) & (pm[..., :2] <= hi), axis=-1)

    def _mat_centre(self, name, c):
        """Bin centre in MAT coords, moved by bin.vote's shift (pelvis x = mat y, pelvis y = mat x) if the bin has one."""
        return self.bin_geometry(name)["centre_mat"]

    def _wall(self, wall):
        """Bin wall height + planner.bin_wall_extra_m (default 0.0 -> exactly `wall`): the extra rim clearance."""
        extra = float((self.cfg.get("planner") or {}).get("bin_wall_extra_m", 0.0))
        return wall + extra if extra else wall

    def _rim(self, name):
        """Rim height (pelvis z) of one bin: measured or nominal, plus planner.bin_wall_extra_m."""
        return self.scene.counter_z + self._wall(self.scene.rim_wall(name))

    def bin_clearance(self, side, arm_q, hand_q, margin=0.01):
        """Lowest height (m) above the bin rims of the hand points over a bin's outer footprint (+margin), inf if none.
        Bins are static boxes in the sim and on the table: only the drop itself may go below a rim."""
        pts = self.kin.hand_points(side, arm_q, hand_q)
        pm = self.scene.pelvis_to_mat(pts)
        b = self.scene.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + margin
        out = np.inf
        for name, c in b["centers"].items():
            rim = self._rim(name)   # perception.rim_z_from_depth; nominal otherwise
            over = np.all(np.abs(pm[:, :2] - self._mat_centre(name, c)) <= half, axis=1)
            if over.any():
                out = min(out, float((pts[over, 2] - rim).min()))
        return out

    def hits_bin_wall(self, side, arm_q, hand_q, allowed_inside=None):
        """Whether sampled hand points enter a bin wall volume.

        For a corrective pick, points may be below the rim only while they are
        inside the named bin's inner cavity.  Other bins remain fully blocked.
        """
        pts = self.kin.hand_points(side, arm_q, hand_q)
        pm = self.scene.pelvis_to_mat(pts)
        b = self.scene.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2
        for name, ctr in b["centers"].items():
            below = pts[:, 2] < self._rim(name) - 1e-3
            over = np.all(np.abs(pm[:, :2] - self._mat_centre(name, ctr)) <= half, axis=1)
            if name == allowed_inside:
                g = self.bin_geometry(name)   # the cavity moves with the bin
                lo, hi = g["inner_lo"], g["inner_hi"]
                cavity = np.all((pm[:, :2] >= lo[:2]) & (pm[:, :2] <= hi[:2]), axis=1)
                over &= ~cavity
            if np.any(over & below):
                return True
        return False

    def neutral_boxes(self):
        """perception.neutral_obstacles: [(xy_lo, xy_hi, z_top)] of the latest frame's obstacles that have a 3D
        footprint (image-only ones never count). [] when the flag is off."""
        src = self.neutral_src
        if src is None:
            return []
        return [(o.xy_lo, o.xy_hi, o.z_top) for o in getattr(src, "neutral_obstacles", ()) if o.has_footprint]

    def neutral_clearance(self, side, arm_q, hand_q, boxes=None):
        """Smallest horizontal gap (m, negative = inside) between a hand point that is not above an obstacle's top
        and the obstacle's xy footprint; inf with no obstacle. Points higher than z_top + margin clear it."""
        boxes = self.neutral_boxes() if boxes is None else boxes
        if not boxes:
            return np.inf
        pts = self.kin.hand_points(side, arm_q, hand_q)
        margin = float((self.cfg["perception"].get("neutral_params") or {}).get("margin_m", 0.02))
        out = np.inf
        for lo, hi, z_top in boxes:
            low = pts[pts[:, 2] <= z_top + margin]
            if len(low) == 0:
                continue
            d = np.maximum(np.maximum(lo - low[:, :2], low[:, :2] - hi), 0.0)
            gap = np.hypot(d[:, 0], d[:, 1])
            inside = np.all((low[:, :2] >= lo) & (low[:, :2] <= hi), axis=1)
            out = min(out, -1.0 if inside.any() else float(gap.min()))
        return out

    def clearance(self, side, arm_q, hand_q, fingers_only=False):
        """Lowest height (m) above the counter surface of the hand points that are over the counter (inf if none)."""
        pts = self.kin.hand_points(side, arm_q, hand_q)
        if fingers_only:
            pts = pts[2:]
        over = self._over_counter(pts)
        if not over.any():
            return np.inf
        return float((pts[over, 2] - self.scene.counter_z).min())

    def _route_pose_safe(self, arm_q, hand_q, sides, clearance, allowed_inside=None):
        """Equivalent planner-side runtime table/bin keepout used where interface/safety.py cannot be changed."""
        for side in sides:
            if self.clearance(side, arm_q, hand_q) < float(clearance) - 2e-3:
                return False
            inside = allowed_inside if len(sides) == 1 else None
            if self.hits_bin_wall(side, arm_q, hand_q, allowed_inside=inside):
                return False
        return True

    def _lift_from_contact_on(self):
        return bool((self.cfg.get("planner") or {}).get("lift_from_contact", False))

    def _contact_lift(self, sides, arm, hand, allowed_inside=None, rise=0.06):
        """planner.lift_from_contact: pure vertical lift of both grasp points (hand posture unchanged, half speed).
        Each sample must not go below its own side's start clearance - 1 mm and must not enter a bin wall.
        Returns (arm, hand) or None when IK or the clearance proof fails."""
        mo = self.cfg["motion"]
        floors = {s: self.clearance(s, arm, hand) - 1e-3 for s in sides}
        p0 = {s: np.asarray(self.kin.grasp_pose(s, arm)[0], float) for s in sides}
        n = max(1, int(np.ceil(rise / float(mo["cart_step"]))))
        cur, waypoints = arm.copy(), [arm.copy()]
        for i in range(1, n + 1):
            for s in sides:
                z = p0[s][2] + rise * i / n
                q, ok, _, _ = self.kin.solve(s, np.r_[p0[s][:2], z], cur, top_down=False, restarts=False)
                if not ok or np.abs(q - cur).max() > float(mo.get("max_joint_step", 0.4)):
                    return None
                cur = q
            waypoints.append(cur.copy())
        a = time_scale(waypoints, mo["arm_max_vel"] * 0.5, mo["arm_max_acc"] * 0.5, self.dt)
        h = np.repeat(hand[None], len(a), 0)
        for k in range(len(a)):
            for s in sides:
                if self.clearance(s, a[k], h[k]) < floors[s] or self.hits_bin_wall(s, a[k], h[k], allowed_inside=allowed_inside):
                    return None
        return a, h

    def safe_home_route(self, arm_from, hand_from, sides=("left", "right"), allowed_inside=None, open_first=True):
        """Prove an open-hand, vertical escape above the runtime table/rims, then a dry-run-safe route home.
        open_first=False: lift-first-then-open (hand as-is during lift + home, opened at home, same keepout proof).

        This is deliberately fail closed: no joint-space fallback is returned when IK, runtime keepouts, bin walls,
        or the final SafetyFilter replay cannot be proved clip-free.
        """
        arm_from, hand_from = np.asarray(arm_from, float), np.asarray(hand_from, float)
        sides = tuple(s for s in sides if s in ("left", "right"))
        if arm_from.shape != (14,) or hand_from.shape != (14,) or not sides:
            return SafeRoute(False, reason="invalid recovery pose/sides")
        if not np.all(np.isfinite(arm_from)) or not np.all(np.isfinite(hand_from)):
            return SafeRoute(False, reason="non-finite recovery pose")
        mo, g = self.cfg["motion"], self.cfg["grasp"]
        cur_a, cur_h = arm_from.copy(), hand_from.copy()
        arm_parts, hand_parts, phases = [], [], []

        def add(name, a, h):
            nonlocal cur_a, cur_h
            if not len(a):
                return
            start = sum(len(x) for x in arm_parts)
            arm_parts.append(np.asarray(a, float)); hand_parts.append(np.asarray(h, float))
            phases.append((name, start, start + len(a)))
            cur_a, cur_h = arm_parts[-1][-1].copy(), hand_parts[-1][-1].copy()

        hand_open = cur_h.copy()
        for side in sides:
            sl = ArmKinematics.sl(side)
            hand_open[sl] = self.post["open"][sl]
        if self._lift_from_contact_on() and not self._route_pose_safe(cur_a, cur_h, sides, 0.003, allowed_inside):
            lift = self._contact_lift(sides, cur_a, cur_h, allowed_inside)
            if lift is None:
                return SafeRoute(False, reason="lift_from_contact: no proved vertical lift from the contact start")
            add("lift_from_contact", *lift)
        if open_first:
            h = ramp(cur_h, hand_open, mo["hand_max_vel"], self.dt)
            a = np.repeat(cur_a[None], len(h), 0)
            for k in range(len(h)):
                if not self._route_pose_safe(a[k], h[k], sides, 0.003, allowed_inside):
                    return SafeRoute(False, reason="opening hand enters runtime table/bin keepout")
            add("open", a, h)

        wall = float(self.scene.cfg["bins"]["wall_height"])
        safe_z = self.counter_keepout_z + max(float(g["carry_height"]), wall + self.height + 0.03)
        escape_floor = self.counter_keepout_z + wall + 0.003
        for side in sides:
            p0 = np.asarray(self.kin.grasp_pose(side, cur_a)[0], float)
            hand_min = float(np.min(self.kin.hand_points(side, cur_a, cur_h)[:, 2]))
            target_z = max(safe_z, p0[2] + max(0.0, escape_floor - hand_min))
            if p0[2] >= target_z - 1e-4:
                continue
            n = max(1, int(np.ceil((target_z - p0[2]) / float(mo["cart_step"]))))
            waypoints, seed = [], cur_a.copy()
            for z in np.linspace(p0[2], target_z, n + 1)[1:]:
                q, ok, _, _ = self.kin.solve(side, np.r_[p0[:2], z], seed, top_down=False, restarts=False)
                if not ok or np.abs(q - seed).max() > float(mo.get("max_joint_step", 0.4)):
                    return SafeRoute(False, reason=f"unproved vertical lift IK ({side})")
                waypoints.append(q); seed = q
            a = time_scale([cur_a] + waypoints, mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
            h = np.repeat(cur_h[None], len(a), 0)
            for k in range(len(a)):
                if not self._route_pose_safe(a[k], h[k], sides, 0.003, allowed_inside):
                    return SafeRoute(False, reason=f"vertical lift enters runtime table/bin keepout ({side})")
            add("vertical_lift", a, h)

        for side in sides:
            if float(np.min(self.kin.hand_points(side, cur_a, cur_h)[:, 2])) < escape_floor:
                return SafeRoute(False, reason=f"vertical lift does not clear runtime bin rim ({side})")

        home = cur_a.copy()
        for side in sides:
            home[ArmKinematics.sl(side)] = self.home[ArmKinematics.sl(side)]
        a = time_scale([cur_a, home], mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
        h = np.repeat(cur_h[None], len(a), 0)
        for k in range(len(a)):
            if not self._route_pose_safe(a[k], h[k], sides, mo["counter_clearance"]):
                return SafeRoute(False, reason="route home enters runtime table/bin keepout")
        add("home", a, h)
        if not open_first:   # lift-first-then-open: open at home, proved with the same keepout check
            h = ramp(cur_h, hand_open, mo["hand_max_vel"], self.dt)
            a = np.repeat(cur_a[None], len(h), 0)
            for k in range(len(h)):
                if not self._route_pose_safe(a[k], h[k], sides, 0.003, allowed_inside):
                    return SafeRoute(False, reason="opening hand at home enters runtime table/bin keepout")
            add("open_at_home", a, h)
        if not arm_parts:
            return SafeRoute(True, np.empty((0, 14)), np.empty((0, 14)), phases)
        arm, hand = np.vstack(arm_parts), np.vstack(hand_parts)
        clips = self.dry_run_safety(arm_from, hand_from, arm, hand)
        if clips:
            return SafeRoute(False, reason=f"{clips} safety clips in recovery dry run")
        return SafeRoute(True, arm, hand, phases)

    # ------------------------------------------------------------------ IK helpers
    def _ik(self, side, p, seed, name, plan):
        q, ok, err, tilt = self.kin.solve(side, p, seed, top_down=True, extra_seeds=[self.home[ArmKinematics.sl(side)]])
        plan.ik[name] = (ok, err, float(np.degrees(tilt)))
        plan.waypoints[name] = np.asarray(p, float)
        return q, ok

    def _cartesian(self, side, p0, p1, q0, name, plan):
        """IK waypoints along the straight line p0 -> p1 (excluding p0), seeded continuously. A step whose joints
        jump more than `max_joint_step` is bisected (up to 3 times) before it counts as a failure."""
        n = max(1, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / self.cfg["motion"]["cart_step"])))
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        qs, q, info = [], q0, [True, 0.0, 0.0]
        jmax = float(self.cfg["motion"].get("max_joint_step", 0.4))

        def seg(pa, pb, q, depth):
            qn, ok, err, tilt = self.kin.solve(side, pb, q, top_down=True, restarts=False)
            if ok and np.abs(qn - q).max() <= jmax:
                info[1], info[2] = max(info[1], err), max(info[2], float(np.degrees(tilt)))
                return [qn]
            if depth >= 3:
                plan.reason = plan.reason or (f"IK jump on {name}" if ok else "")
                return None
            pm = 0.5 * (pa + pb)
            a = seg(pa, pm, q, depth + 1)
            b = a and seg(pm, pb, a[-1], depth + 1)
            return a + b if b else None

        for k in range(1, n + 1):
            out = seg(p0 + (p1 - p0) * (k - 1) / n, p0 + (p1 - p0) * k / n, q, 0)
            if out is None:
                plan.ik[name] = (False, info[1], info[2])
                return None
            qs += out
            q = out[-1]
        plan.ik[name] = (True, info[1], info[2])
        plan.waypoints[name] = p1
        return qs

    def _joint_carry(self, side, q_from, p_from, p_to, z_min, name, plan):
        """Joint-space move to an IK solution at p_to (direct, else via a raised midpoint), accepted if the grasp
        point never dips below z_min on the way."""
        def low(qa, qb):
            return any(self.kin.grasp_pose(side, qa + f * (qb - qa))[0][2] < z_min for f in np.linspace(0, 1, 15))

        q_to, ok = self._ik(side, p_to, q_from, name, plan)
        if ok and not low(q_from, q_to):
            return [q_to]
        for dz in (0.06, 0.12):
            mid = 0.5 * (np.asarray(p_from, float) + p_to) + [0, 0, dz]
            q_mid, ok1 = self._ik(side, mid, q_from, name + "_mid", plan)
            if not ok1 or low(q_from, q_mid):
                continue
            q_to, ok2 = self._ik(side, p_to, q_mid, name, plan)
            if ok2 and not low(q_mid, q_to):
                plan.reason = ""
                return [q_mid, q_to]
        plan.ik[name] = (False, 0.0, 0.0)
        plan.reason = f"{name}: no joint-space path above the bin rim"
        return None

    def _carry_corridor_on(self):
        return bool((self.cfg.get("planner") or {}).get("carry_corridor", False))

    def _bin_outer_pelvis(self, name):
        """(xlo, xhi, ylo, yhi) of the OUTER footprint of bin `name` in pelvis xy, bin.vote shift included."""
        g = self.bin_geometry(name)
        (x0, y0), (x1, y1) = g["outer_lo"][:2], g["outer_hi"][:2]
        corners = self.scene.mat_to_pelvis(np.array([[x, y, 0.0] for x in (x0, x1) for y in (y0, y1)]))
        return corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max()

    def _carry_crosses_foreign_bin(self, bin_name, p0_xy, p1_xy, margin):
        """True when the pelvis xy segment p0 -> p1 (50 samples) lies inside the outer footprint (+margin) of a bin other
        than `bin_name`."""
        p0, p1 = np.asarray(p0_xy, float), np.asarray(p1_xy, float)
        pts = p0 + np.linspace(0.0, 1.0, 50)[:, None] * (p1 - p0)
        for n in self.scene.bin_names:
            if n == bin_name:
                continue
            xlo, xhi, ylo, yhi = self._bin_outer_pelvis(n)
            if np.any((pts[:, 0] >= xlo - margin) & (pts[:, 0] <= xhi + margin)
                      & (pts[:, 1] >= ylo - margin) & (pts[:, 1] <= yhi + margin)):
                return True
        return False

    def _carry_via(self, side, bin_name, points, up, plan):
        """planner.carry_corridor: lift -> via (in front of the bin row, in line with the target bin) -> above_bin, only
        when the straight carry would clip a foreign bin. None -> the caller runs the stock carry."""
        if not up or bin_name not in self.scene.bin_names:
            return None
        margin = float((self.cfg.get("planner") or {}).get("carry_corridor_margin_m", 0.03))
        lift, above = points["lift"], points["above_bin"]
        if not self._carry_crosses_foreign_bin(bin_name, lift[:2], above[:2], margin):
            return None
        x_front = min(self._bin_outer_pelvis(n)[0] for n in self.scene.bin_names) - margin
        via = np.r_[min(x_front, float(lift[0])), float(above[1]), float(lift[2])]
        reason0 = plan.reason
        leg_a = self._cartesian(side, lift, via, up[-1], "carry", plan)
        leg_b = leg_a and self._cartesian(side, via, above, leg_a[-1], "carry_via", plan)
        if leg_b:
            plan.waypoints["carry"] = np.asarray(above, float).copy()   # the carry still ends above the bin
            plan.waypoints["carry_via"] = via
            return leg_a + leg_b
        for key in ("carry", "carry_via"):
            plan.ik.pop(key, None)
            plan.waypoints.pop(key, None)
        plan.reason = reason0
        (self.log or print)("[carry_corridor] via failed, stock carry")
        return None

    def _in_bin_footprint(self, name, xy):
        """True when pelvis-frame xy lies inside the inner footprint of bin `name`, bin.vote shift included."""
        xlo, xhi, ylo, yhi = self.bin_geometry(name)["inner_pelvis"]
        return bool(xlo <= xy[0] <= xhi and ylo <= xy[1] <= yhi)

    def _adjusted_drop_target(self, side, bin_name, xy, z, seed, z_max=None):
        """Conservative bin release (grasp.drop_lateral_margin_m, grasp.drop_height_extra_m) -> (xy, z, adjusted).

        The lateral margin moves only the y coordinate toward the (bin.vote-shifted) bin centre, never past it. The
        height rises by drop_height_extra_m, capped at z_max (the carry height; never below the stock release). The
        candidate must lie inside the bin footprint and pass a direct IK check, else the stock target is returned.
        Both flags at 0 -> the stock target untouched, no IK call."""
        g = self.cfg["grasp"]
        dz = float(g.get("drop_height_extra_m", 0.0))
        margin = float(g.get("drop_lateral_margin_m", 0.0))
        base_xy = np.asarray(xy, float)
        if dz == 0.0 and margin == 0.0:
            return base_xy, float(z), False
        cand_xy = base_xy.copy()
        dy = float(self.bin_xy(bin_name)[1] - cand_xy[1])
        cand_xy[1] += np.sign(dy) * min(abs(dy), margin)
        z_new = float(z) + dz
        if z_max is not None:
            z_new = min(z_new, max(float(z_max), float(z)))
        if not self._in_bin_footprint(bin_name, cand_xy):
            return base_xy, float(z), False
        candidate = np.r_[cand_xy, z_new]
        _, ok, _, _ = self.kin.solve(side, candidate, seed, top_down=True,
                                     extra_seeds=[self.home[ArmKinematics.sl(side)]])
        return (cand_xy, z_new, True) if ok else (base_xy, float(z), False)

    # ------------------------------------------------------------------ main
    def plan(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True, drop_slot=None, source_bin=None,
             clearance_obstacles=()):
        """perception.neutral_obstacles: the obstacle-constrained search is bounded by neutral_params.plan_timeout_s. If no
        plan passes the obstacle check in that time, fail OPEN: return the first complete plan that only violated the
        obstacle check (the stock plan) and log one line. No footprint obstacle (or flag off): exactly _plan_core."""
        if not self.neutral_boxes():
            return self._plan_core(det, arm_q, hand_q, bin_name, obstacles, relocate, drop_slot, source_bin,
                                   clearance_obstacles)
        budget = float((self.cfg["perception"].get("neutral_params") or {}).get("plan_timeout_s", 1.0))
        self._neutral_deadline, self._neutral_fallback = time.monotonic() + budget, None
        try:
            r = self._plan_core(det, arm_q, hand_q, bin_name, obstacles, relocate, drop_slot, source_bin,
                                clearance_obstacles)
            fb = self._neutral_fallback
        finally:
            self._neutral_deadline, self._neutral_fallback = None, None
        if not r.ok and fb is not None:
            (self.log or print)(f"[neutral] no plan clears the neutral obstacles within {budget:.1f}s: fail open, stock plan "
                                f"for {det.color} ({fb.side})")
            return fb
        return r

    def _plan_core(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True, drop_slot=None, source_bin=None,
                   clearance_obstacles=()):
        """Plan for Detection `det` starting from the commanded arm_q/hand_q (14,). Tries the preferred arm (the
        target bin's side; for the centre bin the object's side), then the other one. If neither arm can do the
        whole pick -> bin (cross-body: the arm that reaches the object cannot reach the bin), plans a RELOCATION
        instead (`grasp.relocate`): pick + place on the counter at a spot from which an arm can finish the job;
        the next look re-detects the object there. obstacles: (n, 2) pelvis xy of the other detected objects.
        Returns the first Plan that passes every check, else the last failure."""
        g = self.cfg["grasp"]
        bin_name = bin_name or lookup_detection_bin(self.cfg, self.mapping, det)
        by = self.bin_xy(bin_name)[1] if bin_name is not None else 0.0
        ref = by if abs(by) > 0.1 else det.top[1]
        side0 = "left" if ref > g["y_split"] else "right"
        last = Plan(False, color=det.color, reason=f"no bin for colour {det.color!r}")
        if bin_name is None:
            return last
        reasons, good = [], []
        for side in (side0, "right" if side0 == "left" else "left"):
            p = self._plan_side_hints(det, side, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float),
                                      drop_slot=drop_slot, source_bin=source_bin,
                                      clearance_obstacles=clearance_obstacles)
            if p.ok:
                if g.get("arm", "best") != "best" or self._lying_reach_on(det):   # lying: default arm first
                    return p
                good.append(p)
            else:
                reasons.append(f"{side}: {p.reason}")
                last = p
        if good:
            return min(good, key=lambda p: p.duration)
        last.reason = "; ".join(reasons)
        if relocate and g.get("relocate", {}).get("enabled", True):
            r = self.plan_relocation(det, bin_name, np.asarray(arm_q, float), np.asarray(hand_q, float), obstacles,
                                     drop_slot=drop_slot, clearance_obstacles=clearance_obstacles)
            if r.ok:
                return r
            last.reason += f"; relocate: {r.reason}"
        return last

    def plan_place_at(self, det, arm_q, hand_q, table_xy, obstacles=()):
        """Dry-run/produce the existing pick-and-place motion with a table release point.

        Geometry policy (table bounds, 6 cm clearances, 2 cm margin) is kept in
        the caller's validator.  This method supplies the same IK, trajectory,
        collision and SafetyFilter checks as ordinary bin placement.
        """
        xy = np.asarray(table_xy, float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            return Plan(False, color=det.color, target_bin="counter", reason="invalid table_xy")
        g = self.cfg["grasp"]
        side0 = "left" if xy[1] > g["y_split"] else "right"
        last, good, reasons = Plan(False, color=det.color, target_bin="counter"), [], []
        for side in (side0, "right" if side0 == "left" else "left"):
            p = self._plan_side_hints(det, side, None, np.asarray(arm_q, float), np.asarray(hand_q, float),
                                      place_xy=xy)
            if p.ok:
                if g.get("arm", "best") != "best":
                    return p
                good.append(p)
            else:
                reasons.append(f"{side}: {p.reason}")
                last = p
        if good:
            return min(good, key=lambda p: p.duration)
        last.reason = "; ".join(reasons) or "no safe table placement"
        return last

    def relocation_spots(self, det, bin_name, obstacles=()):
        """Candidate counter spots (pelvis xy) for a relocation, nearest the target bin first."""
        rc = self.cfg["grasp"].get("relocate", {})
        sc = self.scene
        L = sc.cfg["layout"]
        m = float(rc.get("region_margin", 0.02))
        xs = np.linspace(L["region_x"][0] - m, L["region_x"][1] + m, 9)
        ys = np.linspace(L["region_y"][0] - m, L["region_y"][1] + m, 4)
        pts = sc.mat_to_pelvis(np.array([[x, y, 0.0] for x in xs for y in ys]))[:, :2]
        clear = float(rc.get("clearance", L["min_spacing"]))
        obs = np.asarray(list(obstacles), float).reshape(-1, 2)
        keep = [p for p in pts if np.linalg.norm(p - det.top[:2]) > 0.06
                and (len(obs) == 0 or np.linalg.norm(obs - p, axis=1).min() > clear)]
        b = self.bin_xy(bin_name)
        keep.sort(key=lambda p: np.linalg.norm(p - b))
        return keep[: int(rc.get("max_spots", 6))]

    def plan_relocation(self, det, bin_name, arm_q, hand_q, obstacles=(), drop_slot=None,
                        clearance_obstacles=()):
        """Cross-body fallback: pick `det` with an arm that reaches it and put it down at a counter spot from which
        some arm has a full plan to `bin_name` (checked from home). Returns a Plan with target_bin 'counter'."""
        last = Plan(False, color=det.color, reason="no relocation spot")
        if self._lying_reach_on(det):   # its re-plan from the spot would use the stale ridge / prism_xy
            last.reason = "lying prism: no relocation"
            return last
        H = self._object_height(det)
        for spot in self.relocation_spots(det, bin_name, obstacles):
            if getattr(det, "novel", False):
                d2 = copy.copy(det)   # preserve opt-in dynamic novel geometry fields
                d2.top = np.r_[spot, self.scene.counter_z + H]
            else:
                from dataclasses import replace
                d2 = replace(det, top=np.r_[spot, self.scene.counter_z + H])
            by = self.bin_xy(bin_name)[1]
            fin_side = "left" if (by if abs(by) > 0.1 else spot[1]) > 0 else "right"
            fin = self._plan_side_hints(d2, fin_side, bin_name, self.home.copy(), self.post["open"].copy(),
                                        drop_slot=drop_slot, clearance_obstacles=clearance_obstacles)
            if not fin.ok:
                last.reason = f"no {fin_side} plan from spot ({spot[0]:+.2f},{spot[1]:+.2f}): {fin.reason}"
                continue
            side = "right" if fin_side == "left" else "left"
            p = self._plan_side_hints(det, side, bin_name, arm_q, hand_q, place_xy=spot,
                                      clearance_obstacles=clearance_obstacles)
            if p.ok:
                p.reason = f"relocate for the {fin_side} arm"
                return p
            last = p
            last.reason = f"{side} cannot place at ({spot[0]:+.2f},{spot[1]:+.2f}): {p.reason}"
        return last

    def plan_to_counter(self, det, arm_q, hand_q, source_bin, obstacles=()):
        """Remove a detected wrong-bin object to a clear counter spot.

        This is the score-safe fallback when no target-bin slot has a valid
        plan: moving wrong -> counter changes -1 to 0 and a later look can try
        the intended bin again.
        """
        last = Plan(False, color=det.color, source_bin=source_bin, corrective=True,
                    reason="no safe counter cleanup spot")
        target_bin = lookup_detection_bin(self.cfg, self.mapping, det)
        for spot in self.relocation_spots(det, target_bin, obstacles):
            for side in ("left", "right"):
                p = self._plan_side_hints(det, side, target_bin, np.asarray(arm_q, float), np.asarray(hand_q, float),
                                          place_xy=spot, source_bin=source_bin)
                if p.ok:
                    p.reason = f"remove from wrong {source_bin} bin for retry"
                    return p
                last = p
        last.reason = last.reason or "no safe counter cleanup spot"
        return last

    def _plan_side_hints(self, det, side, bin_name, arm_q, hand_q, **kw):
        """Triangle prisms with `grasp.yaw_align`: finger directions derived from the detected apex yaw first; else
        (and as the fallback) _plan_side with each finger direction of grasp.finger_hints in turn (fingers forward
        can reach over the centre bin's wall when the object is close to it; then fingers inward/sideways)."""
        if self._generic_on(det):
            # Width is valid only across the proposal's minor axis. Never fall
            # through to legacy orientations after accepting a generic proposal.
            return self._plan_side_generic(det, side, bin_name, arm_q, hand_q, **kw)
        if self._lying_on(det):
            if kw.get("place_xy") is None and kw.get("source_bin") is None:
                p = self._plan_side_lying(det, side, bin_name, arm_q, hand_q, **kw)
                if p is not None:
                    return p      # else: the current (standing-prism) path below
            if self._lying_reach_on(det):   # grasp.lying_prism_arm_by_reach: no standing-prism grasp of a lying prism
                return Plan(False, side=side, color=det.color, target_bin=bin_name,
                            reason=f"lying prism: no {side} ridge heading reachable")
        vy = (self.vary or {}).get("yaw")
        # A corrective cleanup has place_xy but still needs a changed grasp after a miss. Ordinary relocations keep
        # the legacy free-yaw planning because their destination search is already the fallback path.
        if vy is not None and (kw.get("place_xy") is None or kw.get("source_bin") is not None):
            return self._plan_side_heading(det, side, bin_name, arm_q, hand_q, float(vy), **kw)
        if self._yaw_align_on(det) and kw.get("place_xy") is None:
            p = self._plan_side_yaw(det, side, bin_name, arm_q, hand_q, **kw)
            if p is not None:
                return p
        return self._plan_side_legacy(det, side, bin_name, arm_q, hand_q, **kw)

    def _plan_side_legacy(self, det, side, bin_name, arm_q, hand_q, **kw):
        hints = self.cfg["grasp"].get("finger_hints") or [self.cfg["grasp"].get("finger_hint", [1.0, 0.0, 0.0])]
        saved, first = self.kin.hint, None
        try:
            for h in hints:
                h = np.asarray(h, float) * ([1.0, -1.0, 1.0] if side == "right" else 1.0)
                self.kin.hint = h / np.linalg.norm(h)
                p = self._plan_side(det, side, bin_name, arm_q, hand_q, **kw)
                if p.ok:
                    return p
                first = first or p
            return first
        finally:
            self.kin.hint = saved

    # ------------------------------------------------------------------ yaw-aligned triangle grasp (grasp.yaw_align)
    def _yaw_align_on(self, det):
        g = self.cfg["grasp"]
        ya = getattr(det, "apex_yaw", None)
        pose_ok = (ya is not None and np.isfinite(ya)
                   and float(getattr(det, "apex_conf", 1.0)) >= float(g.get("yaw_min_conf", 0.5)))
        if getattr(det, "shape", None) != "triangle":
            return False
        if g.get("tri_vertex_align", False):
            if getattr(det, "shape_path", None) != "metric":  # standing top face only
                return False
            fallback = getattr(det, "vertex_yaw", None)
            if fallback is None or not np.isfinite(fallback):
                fallback = getattr(det, "yaw", None)
            return bool(pose_ok or (fallback is not None and np.isfinite(fallback)))
        return bool(g.get("yaw_align", False) and pose_ok)

    def _triangle_yaw(self, det):
        """(vertex yaw, vertex-gap mode). The robust pose wins; rectangle/top-face yaw is the bounded fallback."""
        g = self.cfg["grasp"]
        ya = getattr(det, "apex_yaw", None)
        pose_ok = (ya is not None and np.isfinite(ya)
                   and float(getattr(det, "apex_conf", 1.0)) >= float(g.get("yaw_min_conf", 0.5)))
        if g.get("tri_vertex_align", False):
            if getattr(det, "shape_path", None) != "metric":
                return None, False
            if pose_ok:
                return float(ya), True
            for name in ("vertex_yaw", "yaw"):
                value = getattr(det, name, None)
                if value is not None and np.isfinite(value):
                    return float(value) % (2.0 * np.pi / 3.0), True
        return (float(ya), False) if pose_ok and g.get("yaw_align", False) else (None, False)

    def yaw_hint_angles(self, apex_yaw, side):
        """Candidate finger directions (pelvis-frame angle, rad), best first, for an equilateral triangle whose apex
        (any of the 3 vertices) points along `apex_yaw` (pelvis frame, x forward, y left). Finger direction = apex
        direction + s*phi + k*120 deg, s = +1 left arm, -1 right arm (the mirror image; phi is calibrated on the left
        hand). Candidates further than yaw_max_dev_deg from the arm's natural direction (-yaw_ref_deg left, +yaw_ref_deg
        right = forward, slightly inward) are dropped; the rest are sorted nearest first."""
        g = self.cfg["grasp"]
        s = 1.0 if side == "left" else -1.0
        ref = -s * np.radians(float(g.get("yaw_ref_deg", 25.0)))
        phi = s * np.radians(float(g.get("yaw_phi_deg", 105.0)))
        out = []
        for k in range(3):
            a = apex_yaw + phi + 2 * np.pi * k / 3
            da = abs((a - ref + np.pi) % (2 * np.pi) - np.pi)
            if da <= np.radians(float(g.get("yaw_max_dev_deg", 75.0))):
                out.append((da, a))
        return [a for _, a in sorted(out)][: int(g.get("yaw_max_hints", 2))]

    def vertex_hint_angles(self, vertex_yaw, side):
        """Palm-x headings whose index-middle gap (palm-z) lies on a triangle vertex line."""
        g = self.cfg["grasp"]
        s = 1.0 if side == "left" else -1.0
        ref = -s * np.radians(float(g.get("yaw_ref_deg", 25.0)))
        phi = s * np.radians(float(g.get("tri_vertex_phi_deg", 90.0)))
        out = []
        for k in range(3):
            a = float(vertex_yaw) + phi + 2.0 * np.pi * k / 3.0
            da = abs((a - ref + np.pi) % (2.0 * np.pi) - np.pi)
            if da <= np.radians(float(g.get("yaw_max_dev_deg", 75.0))):
                out.append((da, a))
        return [a for _, a in sorted(out)][: int(g.get("yaw_max_hints", 2))]

    @staticmethod
    def _tri_shift(phi, r):
        """Midpoint (m, along f) of the triangle's extent in direction f at angle phi from the apex direction
        (vertices r = 2/3 altitude from the centroid): phi=0 -> +7.5 mm towards the apex, phi=60 deg -> -7.5 mm."""
        v = np.radians([0.0, 120.0, 240.0])
        return 0.5 * (r * np.cos(phi - v).max() - r * np.cos(phi + np.pi - v).max())

    def _finger_err(self, plan, side, a):
        """Angle (deg) between the palm x axis (finger direction) at the end of the descend and the heading a."""
        e = next((e for name, s, e in plan.phases if name == "descend"), None)
        if e is None:
            return float("inf")   # no descend phase to measure -> never counts as aligned
        self.kin._fk(plan.arm[e - 1])
        x = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0]
        d = np.arctan2(x[1], x[0]) - a
        return float(np.degrees(abs((d + np.pi) % (2 * np.pi) - np.pi)))

    # ------------------------------------------------------------------ lying prism (grasp.lying_prism)
    def _lying_on(self, det):
        ry, pxy = getattr(det, "ridge_yaw", None), getattr(det, "prism_xy", None)
        return (bool(self.cfg["grasp"].get("lying_prism", False)) and getattr(det, "prism_orientation", None) == "lying"
                and ry is not None and np.isfinite(ry) and pxy is not None and bool(np.all(np.isfinite(pxy))))

    def _lying_reach_on(self, det):
        return bool(self.cfg["grasp"].get("lying_prism_arm_by_reach", False)) and self._lying_on(det)

    def lying_headings(self, ridge_yaw, side):
        """Candidate palm-x headings (pelvis rad), best first, for a prism lying with its ridge along `ridge_yaw`. The jaw
        closes along palm x (URDF FK: thumb tip vs index/middle tips differ in palm x only), so the closing axis is across
        the ridge: ridge +- 90 deg. Candidates further than yaw_max_dev_deg from the arm's natural direction are dropped."""
        g = self.cfg["grasp"]
        s = 1.0 if side == "left" else -1.0
        ref = -s * np.radians(float(g.get("yaw_ref_deg", 25.0)))
        out = []
        for k in (-1, 1):
            a = float(ridge_yaw) + k * np.pi / 2
            a = (a + np.pi) % (2 * np.pi) - np.pi
            da = abs((a - ref + np.pi) % (2 * np.pi) - np.pi)
            if da <= np.radians(float(g.get("yaw_max_dev_deg", 75.0))):
                out.append((da, a))
        return [a for _, a in sorted(out)]

    def _plan_side_lying(self, det, side, bin_name, arm_q, hand_q, **kw):
        """Grasp across the ridge: palm-x heading = ridge +- 90 deg (strong IK hint, achieved heading within
        grasp.lying_yaw_tol_deg), grasp point = the footprint centroid (det.prism_xy). Grasp height + posture: see
        _plan_side_inner / _posture_choice (self._lying). None -> caller falls back to the standing-prism path."""
        g = self.cfg["grasp"]
        saved = (self.kin.hint, self.kin.w_hint)
        try:
            self._lying = True
            for a in self.lying_headings(float(det.ridge_yaw), side):
                d2 = copy.copy(det)
                d2.top = np.r_[np.asarray(det.prism_xy, float), det.top[2]]
                self.kin.hint, self.kin.w_hint = np.array([np.cos(a), np.sin(a), 0.0]), float(g.get("yaw_hint_weight", 3.0))
                p = self._plan_side(d2, side, bin_name, arm_q, hand_q, **kw)
                if p.ok:
                    err = self._finger_err(p, side, a)
                    if err <= float(g.get("lying_yaw_tol_deg", 10.0)):
                        p.reason = f"lying prism across ridge ({np.degrees(a):.0f} deg, err {err:.0f})"
                        return p
        finally:
            self.kin.hint, self.kin.w_hint = saved
            self._lying = False
        return None

    def _near_rim_retry_on(self, det, place_xy, bin_name):
        return (bool(self.cfg["grasp"].get("novel_near_rim_retry", False)) and place_xy is None and bin_name is not None
                and (getattr(det, "novel", False) or self._generic_on(det)))

    def _generic_on(self, det):
        enabled = ((bool(self.cfg["grasp"].get("generic", False))
                    or bool((self.cfg.get("brain") or {}).get("find_to_detection", False))) if getattr(det, "novel", False)
                   else bool(self.cfg["grasp"].get("generic_fallback", False)))
        return (enabled
                and getattr(det, "generic_required", False)
                and getattr(det, "generic_width", None) is not None)

    def _object_height(self, det):
        """Measured height for accepted generic geometry; configured height otherwise."""
        if self._generic_on(det):
            return float(getattr(det, "height", det.top[2] - self.scene.counter_z))
        return self.height

    def _plan_side_generic(self, det, side, bin_name, arm_q, hand_q, **kw):
        """Align palm-x (the fingers) with the proposal's major axis."""
        g = self.cfg["grasp"]
        if getattr(det, "novel", False):
            score = float(getattr(det, "generic_confidence", 0.0) or 0.0)
            if not getattr(det, "graspable", not getattr(det, "ungraspable", False)) or score < float(g.get("generic_min_confidence", 0.5)):
                return Plan(False, side=side, color=det.color, target_bin=bin_name,
                            reason=f"novel graspability {score:.2f} below threshold")
            if float(det.generic_width) > self.generic_hand_opening:
                return Plan(False, side=side, color=det.color, target_bin=bin_name,
                            reason=f"novel width {1000 * det.generic_width:.0f} mm exceeds URDF opening "
                                   f"{1000 * self.generic_hand_opening:.0f} mm")
        yaw = getattr(det, "yaw", None)
        if yaw is None or not np.isfinite(yaw):
            return Plan(False, side=side, color=det.color, target_bin=bin_name,
                        reason="generic proposal has no finite yaw")
        saved = (self.kin.hint, self.kin.w_hint)
        sign = 1.0 if side == "left" else -1.0
        natural = -sign * np.radians(float(g.get("yaw_ref_deg", 25.0)))
        # The fingers extend along palm x; with them parallel to the object's
        # major axis, the thumb-to-finger closure spans the reported minor-axis
        # width. Try both directions of the undirected major axis.
        angles = [float(yaw), float(yaw) + np.pi]
        angle = min(angles, key=lambda a: abs((a - natural + np.pi) % (2 * np.pi) - np.pi))
        deviation = abs((angle - natural + np.pi) % (2 * np.pi) - np.pi)
        max_dev = 90.0 if getattr(det, "novel", False) else float(g.get("generic_yaw_max_dev_deg", 45.0))
        if deviation > np.radians(max_dev):
            return Plan(False, side=side, color=det.color, target_bin=bin_name,
                        reason=f"generic yaw {np.degrees(angle):.0f} deg outside natural range")
        try:
            self.kin.hint = np.array([np.cos(angle), np.sin(angle), 0.0])
            self.kin.w_hint = float(g.get("generic_yaw_hint_weight", 3.0))
            p = self._plan_side(det, side, bin_name, arm_q, hand_q, **kw)
            if not p.ok:
                return p
            err = self._finger_err(p, side, angle)
            if err > float(g.get("generic_yaw_tol_deg", 15.0)):
                p.ok = False
                p.reason = f"generic yaw error {err:.0f} deg"
                return p
            p.reason = f"generic yaw ({np.degrees(angle):.0f} deg), width {1000 * det.generic_width:.0f} mm"
            return p
        finally:
            self.kin.hint, self.kin.w_hint = saved

    def _plan_side_yaw(self, det, side, bin_name, arm_q, hand_q, **kw):
        """Try the yaw-derived finger directions (strong hint weight, grasp point moved to the pinch midpoint).
        Returns the first Plan whose achieved finger direction is within yaw_tol_deg, else None (-> legacy hints)."""
        g = self.cfg["grasp"]
        tri_yaw, vertex_mode = self._triangle_yaw(det)
        if tri_yaw is None:
            return None
        saved = (self.kin.hint, self.kin.w_hint)
        try:
            angles = self.vertex_hint_angles(tri_yaw, side) if vertex_mode else self.yaw_hint_angles(tri_yaw, side)
            for a in angles:
                f = np.array([np.cos(a), np.sin(a), 0.0])
                d2 = copy.copy(det)
                if not vertex_mode and g.get("yaw_tri_shift", True):  # pinch midpoint along f is not the centroid
                    d2.top = det.top + self._tri_shift(a - tri_yaw, 2.0 / 3.0 * float(self.scene.size)) * f
                self.kin.hint, self.kin.w_hint = f, float(g.get("yaw_hint_weight", 3.0))
                p = self._plan_side(d2, side, bin_name, arm_q, hand_q, **kw)
                if p.ok:
                    err = self._finger_err(p, side, a)
                    if err <= float(g.get("yaw_tol_deg", 12.0)):
                        mode = "vertex-aligned" if vertex_mode else "yaw-aligned"
                        p.reason = f"{mode} ({np.degrees(a):.0f} deg, err {err:.0f})"
                        return p
        finally:
            self.kin.hint, self.kin.w_hint = saved
        return None

    def _plan_side_heading(self, det, side, bin_name, arm_q, hand_q, a, **kw):
        """policy.retry_vary_grasp: plan with the finger heading `a` (pelvis rad) as a strong IK hint (yaw_hint_weight);
        the plan only counts if the achieved heading is within yaw_tol_deg (else a failed Plan)."""
        g = self.cfg["grasp"]
        saved = (self.kin.hint, self.kin.w_hint)
        try:   # the strong heading only for pregrasp / descend / lift; carry on with the stock forward hint
            h0 = np.asarray(self.cfg["grasp"].get("finger_hint", [1.0, 0.0, 0.0]), float)
            h0 = h0 * ([1.0, -1.0, 1.0] if side == "right" else 1.0)
            self._carry_hint = (h0 / np.linalg.norm(h0), float(g.get("finger_hint_weight", 0.3)))
            self.kin.hint = np.array([np.cos(a), np.sin(a), 0.0])
            self.kin.w_hint = float(g.get("yaw_hint_weight", 3.0))
            p = self._plan_side(det, side, bin_name, arm_q, hand_q, **kw)
        finally:
            self.kin.hint, self.kin.w_hint = saved
            self._carry_hint = None
        if p.ok:
            err = self._finger_err(p, side, a)
            if err > float(g.get("yaw_tol_deg", 12.0)):
                p.ok, p.reason = False, f"retry heading {np.degrees(a):.0f} deg not reached (err {err:.0f})"
        return p

    def finger_heading(self, plan):
        """Achieved finger heading (pelvis rad, palm x axis) at the end of the descend, None without a descend."""
        e = next((e for name, s, e in plan.phases if name == "descend"), None)
        if e is None or plan.arm is None or plan.side not in ("left", "right"):
            return None
        self.kin._fk(plan.arm[e - 1])
        x = self.kin.data.oMf[self.kin.palm[plan.side]].rotation[:, 0]
        return float(np.arctan2(x[1], x[0]))

    def _posture_choice(self, det):
        """grasp.close_posture_by_shape: (posture name, grasp point or None, label, conf, path). demo only for a
        confident metric triangle; everything else (incl. unknown shape / blob / abstain) keeps the configured posture."""
        g = self.cfg["grasp"]
        label = getattr(det, "shape", None)
        conf = float(getattr(det, "shape_conf", 0.0) or 0.0)
        path = getattr(det, "shape_path", None)
        if self._lying:   # grasp.lying_prism: the demo posture is for standing prisms
            return g.get("close_posture", "closed"), None, label, conf, path
        width = getattr(det, "generic_width", None)
        if getattr(det, "generic_required", False) and width is not None:
            if float(width) <= float(g.get("generic_narrow_width", 0.035)):
                return "closed_demo", np.asarray(g["point_in_palm_demo"], float), None, conf, "generic_width"
            return g.get("close_posture", "closed"), None, None, conf, "generic_width"
        thumb_extra = float((g.get("cage_thumb_extra_rad") or {}).get(label, 0.0))
        cage_ok = (g.get("cage_mode", False) and thumb_extra > 0.0 and path == "metric"
                   and conf >= float(g.get("cage_shape_min_conf", 0.5)))
        if cage_ok and label == "triangle" and g.get("posture_apex_gate", False):
            from .apex_gate import apex_gate_pass
            ac, ang = getattr(det, "apex_gate", None) or (0.0, None)
            cage_ok = apex_gate_pass(ac, ang, float(g.get("posture_apex_min_conf", 0.5)),
                                     float(g.get("posture_apex_max_angle_deg", 75.0)))
            self._apex = (cage_ok, ac, ang)
        if cage_ok:
            return f"cage_{label}", np.asarray(g["cage_point_in_palm"], float), label, conf, path
        if label == "triangle" and path == "metric" and conf >= float(g.get("posture_by_shape_min_conf", 0.5)):
            if not g.get("posture_apex_gate", False):
                return "closed_demo", np.asarray(g["point_in_palm_demo"], float), label, conf, path
            # grasp.posture_apex_gate: the demo posture also needs a real prism top (apex_gate.py), else fall back
            from .apex_gate import apex_gate_pass
            ac, ang = getattr(det, "apex_gate", None) or (0.0, None)
            ok = apex_gate_pass(ac, ang, float(g.get("posture_apex_min_conf", 0.5)),
                                float(g.get("posture_apex_max_angle_deg", 75.0)))
            self._apex = (ok, ac, ang)
            if ok:
                return "closed_demo", np.asarray(g["point_in_palm_demo"], float), label, conf, path
        return g.get("close_posture", "closed"), None, label, conf, path

    def _plan_side(self, det, side, bin_name, arm_q, hand_q, place_xy=None, drop_slot=None, source_bin=None,
                   clearance_obstacles=()):
        g = self.cfg["grasp"]
        vp = (self.vary or {}).get("posture")
        border = self._border_posture(det)
        generic = self._generic_on(det)
        if not (g.get("close_posture_by_shape", False) or g.get("cage_mode", False)) \
                and vp is None and border is None and not generic:
            self._posture = None
            return self._plan_side_inner(det, side, bin_name, arm_q, hand_q, place_xy, drop_slot, source_bin,
                                         clearance_obstacles)
        self._apex = None
        if not generic and (vp is not None or border is not None):   # policy.retry_vary_grasp / grasp.border_posture: forced posture
            name = vp if vp is not None else border
            pt = np.asarray(g["point_in_palm_demo"], float) if name == "closed_demo" else None
            label, conf, path = (getattr(det, "shape", None), float(getattr(det, "shape_conf", 0.0) or 0.0),
                                 getattr(det, "shape_path", None))
        else:
            name, pt, label, conf, path = self._posture_choice(det)
        saved = self.kin.offset
        if pt is not None:
            self.kin.offset = {"left": pt, "right": pt * np.array([1.0, -1.0, 1.0])}
        self._posture = name
        try:
            p = self._plan_side_inner(det, side, bin_name, arm_q, hand_q, place_xy, drop_slot, source_bin,
                                      clearance_obstacles)
        finally:
            self.kin.offset, self._posture = saved, None
        p.posture = (name, label, conf, path) if self._apex is None else (name, label, conf, path, self._apex)
        return p

    def _border_posture(self, det):
        """grasp.border_posture (FLAG, default off): an abstain_border detection (label None, blob cut by the image
        border = near counter edge) -> the demo posture instead of grasp.close_posture. None = flag off / not applicable."""
        if not self.cfg["grasp"].get("border_posture", False):
            return None
        if getattr(det, "shape", None) is None and getattr(det, "shape_path", None) == "abstain_border":
            return "closed_demo"
        return None

    def _plan_side_inner(self, det, side, bin_name, arm_q, hand_q, place_xy=None, drop_slot=None, source_bin=None,
                         clearance_obstacles=()):
        cfg, g, mo = self.cfg, self.cfg["grasp"], self.cfg["motion"]
        sc, zc = self.scene, self.scene.counter_z
        plan = Plan(False, side=side, color=det.color, target_bin=bin_name if place_xy is None else "counter",
                    source_bin=source_bin, drop_slot=drop_slot if place_xy is None else None,
                    corrective=source_bin is not None)
        if self._neutral_deadline is not None and self._neutral_fallback is not None \
                and time.monotonic() > self._neutral_deadline:
            plan.reason = "neutral obstacle search budget exhausted"
            return plan
        sl = ArmKinematics.sl(side)
        H, post = self._object_height(det), self.post
        hfrac = g["height_frac"]
        if self._lying:   # grasp.lying_prism: height = triangle altitude, grasp at the section centroid (= apothem)
            dims = sc.object_dims("triangle")
            H, hfrac = 3.0 * dims["apothem"], 1.0 / 3.0   # altitude; centroid of the section = altitude / 3
        if self._generic_on(det):
            hlo, hhi = cfg["perception"].get("height_range", (0.4, 2.0))
            if not np.isfinite(H) or not float(hlo) * self.height <= H <= float(hhi) * self.height:
                plan.reason = f"generic object height {1000 * H:.0f} mm outside safe range"
                return plan
        hand_pre = hand_q.copy(); hand_pre[sl] = post["preshape"][sl]
        hand_cls = hand_q.copy()
        posture_name = self._posture or g.get("close_posture", "closed")
        soft = soft_hand_active(cfg)   # grasp.soft_squeeze on a soft hand: over-closed target, long hold, slow lift_check
        target = post[posture_name][sl]
        if soft and not posture_name.startswith("cage_") and not self._generic_on(det):   # stock close postures only
            target = soft_overclose(cfg, self.spec, post[posture_name], post["preshape"])[sl]
        if posture_name.startswith("cage_"):  # generated targets are clipped once more at the command seam
            target = np.clip(target, self.spec.hand_lower[sl], self.spec.hand_upper[sl])
        hand_cls[sl] = target
        hand_open = hand_q.copy(); hand_open[sl] = post["open"][sl]
        hand_rel = hand_q.copy(); hand_rel[sl] = post[g.get("release_posture", "preshape")][sl]

        # grasp height: object mid-height, raised until the preshaped fingers clear the counter
        xy = det.top[:2] + np.asarray((g.get("arm_xy_offset") or {}).get(side, [0.0, 0.0]), float)  # per-arm, TUNE
        z = (det.top[2] - H / 2) if source_bin is not None else (zc + hfrac * H)
        lhf = g.get("left_hand_fix") or {}   # grasp.left_hand_fix.shift_z: left-arm triangle grasp height sweep
        if side == "left" and getattr(det, "shape", None) == "triangle" and float(lhf.get("shift_z", 0.0) or 0.0):
            z += float(lhf["shift_z"])
        for _ in range(4):
            qg, ok = self._ik(side, np.r_[xy, z], arm_q, "grasp", plan)
            if not ok:
                plan.reason = f"IK grasp {side}"
                return plan
            c = self.clearance(side, qg, hand_pre, fingers_only=True)
            if c >= g["fingertip_clearance"] - 1e-3:
                break
            z += g["fingertip_clearance"] - c
        else:
            plan.reason = "fingers hit the counter"
            return plan
        # Thin novel handles/markers need the soft Dex3 pads to straddle the
        # top edge after the counter-clearance raise.  Permit the configured
        # grasp point up to 6 mm above the measured top; the known/fallback
        # path retains its original 8 mm-below-top guard.
        top_margin = 0.006 if getattr(det, "novel", False) else -0.008
        if source_bin is None and z > zc + H + top_margin:
            plan.reason = f"grasp point {100 * (z - zc):.1f} cm above counter: above the object top (point_in_palm?)"
            return plan

        wall = self._wall(sc.rim_wall())   # + planner.bin_wall_extra_m; rim_wall: perception.rim_z_from_depth
        z_carry = zc + max(g["carry_height"], wall + H + 0.03)
        z_drop = zc + self._wall(sc.rim_wall(bin_name)) + H / 2 + g["drop_above_rim"]
        if self.drop_extra.get(bin_name):  # bin.vote low-confidence bin: release higher, never above the carry height
            z_drop = min(z_drop + self.drop_extra[bin_name], z_carry)
        if place_xy is not None:  # relocation / place_at: put it down on the counter (release just above the grasp height)
            bxy, z_drop = np.asarray(place_xy, float), z + 0.005   # bin_name may be None here
        elif drop_slot is None:  # stock: one release point per bin (#6 off)
            bxy = self.drop_xy(bin_name)
        else:                  # #6: the assigned drop slot
            slots = self.drop_slots(bin_name)
            bxy = slots[min(int(drop_slot), len(slots) - 1)]
        stock_bxy, stock_z_drop = np.asarray(bxy, float).copy(), float(z_drop)
        drop_adjustment_on, adjusted = False, False
        if place_xy is None and (g.get("drop_height_extra_m", 0.0) or g.get("drop_lateral_margin_m", 0.0)):
            drop_adjustment_on = True
            bxy, z_drop, adjusted = self._adjusted_drop_target(side, bin_name, bxy, z_drop, qg, z_max=z_carry)
        P = {"pregrasp": np.r_[xy, max(z + g["approach_height"], z_carry) + float(g.get("approach_extra_m", 0.0))], "grasp": np.r_[xy, z], "lift": np.r_[xy, z_carry],
             "above_bin": np.r_[bxy, z_carry], "drop": np.r_[bxy, z_drop], "retreat": np.r_[bxy, z_drop + g["retreat_up"]]}

        q_pre, ok = self._ik(side, P["pregrasp"], qg, "pregrasp", plan)  # same IK branch as the grasp
        if not ok:
            plan.reason = f"IK pregrasp {side}"
            return plan
        plan.waypoints["grasp"] = np.r_[xy, z]
        down = self._cartesian(side, P["pregrasp"], P["grasp"], q_pre, "descend", plan)
        up = down and self._cartesian(side, P["grasp"], P["lift"], down[-1], "lift", plan)
        hint_saved = (self.kin.hint, self.kin.w_hint)

        def drop_leg(points):
            if self._carry_hint is not None:   # retry heading variant: the wrist may turn back while carrying
                self.kin.hint, self.kin.w_hint = self._carry_hint
            try:
                carry_ = (self._carry_via(side, bin_name, points, up, plan)
                          if place_xy is None and self._carry_corridor_on() else None)
                carry_ = carry_ or (up and (self._cartesian(side, points["lift"], points["above_bin"], up[-1], "carry", plan)
                                            or self._joint_carry(side, up[-1], points["lift"], points["above_bin"],
                                                                 points["drop"][2], "carry", plan)))
                lower_ = carry_ and self._cartesian(side, points["above_bin"], points["drop"], carry_[-1],
                                                    "lower", plan)
                retreat_ = lower_ and self._cartesian(side, points["drop"], points["retreat"], lower_[-1],
                                                      "retreat", plan)
                return carry_, lower_, retreat_
            finally:
                self.kin.hint, self.kin.w_hint = hint_saved

        carry, lower, retreat = drop_leg(P)
        if not retreat and adjusted:
            # Endpoint IK cannot prove a continuous lower/retreat path: retry the complete stock drop leg, so an
            # adjusted release can never reject a pick that the flag-off planner could execute.
            for key in ("carry", "carry_mid", "carry_via", "lower", "retreat"):
                plan.ik.pop(key, None)
                plan.waypoints.pop(key, None)
            plan.reason = ""
            P["above_bin"] = np.r_[stock_bxy, z_carry]
            P["drop"] = np.r_[stock_bxy, stock_z_drop]
            P["retreat"] = np.r_[stock_bxy, stock_z_drop + g["retreat_up"]]
            adjusted = False
            carry, lower, retreat = drop_leg(P)
        if not retreat and self._near_rim_retry_on(det, place_xy, bin_name):
            # novel/generic: the drop leg failed -> release once at the bin's near rim (no inset); clearances unchanged
            first = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
            for key in ("carry", "carry_mid", "lower", "retreat"):
                plan.ik.pop(key, None)
                plan.waypoints.pop(key, None)
            plan.reason = ""
            near = self.drop_xy(bin_name, inset=0.0)
            P["above_bin"] = np.r_[near, z_carry]
            P["drop"] = np.r_[near, stock_z_drop]
            P["retreat"] = np.r_[near, stock_z_drop + g["retreat_up"]]
            carry, lower, retreat = drop_leg(P)
            if retreat:
                plan.waypoints["drop_near_rim"] = P["drop"].copy()
            else:
                second = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
                plan.reason = f"{first} | near-rim retry: {second}"
        if drop_adjustment_on:
            plan.waypoints["drop_adjusted" if adjusted else "drop_fallback"] = P["drop"].copy()
        if not retreat:
            plan.reason = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
            return plan
        plan.waypoints["drop"] = P["drop"]

        vmax, amax, hv = mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"]
        home = arm_q.copy(); home[sl] = self.home[sl]
        arm_parts, hand_parts, phases = [], [], []
        cur_a, cur_h = arm_q.copy(), hand_q.copy()

        def move(name, Qs, hand_to=None, scale=1.0):
            nonlocal cur_a, cur_h
            if self.object_speed_scale is not None:   # <= 1 on top of the (capped) arm_max_vel: never faster
                scale *= float(self.object_speed_scale)
            a = time_scale([cur_a] + list(Qs), vmax * scale, amax * scale, self.dt)
            h = np.repeat(cur_h[None], len(a), 0)
            if hand_to is not None:  # hand moves during the arm move (ends no later than the arm)
                hr = self.hand_ramp(cur_h, hand_to)
                if len(hr) > len(a):
                    a = np.vstack([a, np.repeat((a[-1] if len(a) else cur_a)[None], len(hr) - len(a), 0)])
                    h = np.repeat(cur_h[None], len(a), 0)
                h[:len(hr)] = hr
                h[len(hr):] = hand_to
                cur_h = np.asarray(hand_to, float).copy()
            add(name, a, h)

        def hand(name, to, hold_s):
            nonlocal cur_h
            if name == "open" and g.get("staggered_release", False):
                h, second_tick = staggered_release_ramp(
                    cur_h, to, side, hv, self.dt, float(g.get("staggered_release_delay_s", 0.30)))
            else:
                h = self.hand_ramp(cur_h, to)
                second_tick = None
            n_ramp = len(h)
            h = np.vstack([h, np.repeat(np.asarray(to, float)[None], int(round(hold_s / self.dt)), 0)])
            start = sum(len(x) for x in arm_parts)
            if name == "close":
                plan.close_hold = (start + n_ramp, start + len(h))
            elif second_tick is not None:
                plan.release_stages = (start, start + second_tick, start + n_ramp)
            cur_h = np.asarray(to, float).copy()
            add(name, np.repeat(cur_a[None], len(h), 0), h)

        def add(name, a, h):
            nonlocal cur_a
            if len(a):
                start = sum(len(x) for x in arm_parts)
                arm_parts.append(a); hand_parts.append(h)
                phases.append((name, start, start + len(a)))
                cur_a = a[-1].copy()

        move("to_pregrasp", [q_pre], hand_to=hand_pre)
        slow_v = float(g.get("approach_slow_mps", 0.0))
        if slow_v > 0.0 and len(down) >= 2:   # grasp.approach_slow_mps: the last approach_slow_dist_m is slow (real demo)
            n_tail = min(len(down) - 1, max(1, int(np.ceil(float(g.get("approach_slow_dist_m", 0.05)) / mo["cart_step"]))))
            t_min = min(n_tail * float(mo["cart_step"]), float(g.get("approach_slow_dist_m", 0.05))) / slow_v
            sc0 = float(self.object_speed_scale) if self.object_speed_scale is not None else 1.0
            a_head = time_scale([cur_a] + list(down[:-n_tail]), vmax * sc0, amax * sc0, self.dt) \
                if len(down) > n_tail else np.repeat(cur_a[None], 0, 0)
            start = a_head[-1].copy() if len(a_head) else cur_a.copy()
            sc = sc0
            for _ in range(8):
                a_tail = time_scale([start] + list(down[-n_tail:]), vmax * sc, amax * sc, self.dt)
                if len(a_tail) * self.dt >= t_min - 1e-9:
                    break
                sc *= max(0.2, 0.95 * len(a_tail) * self.dt / t_min)
            a = np.vstack([a_head, a_tail]) if len(a_head) else a_tail
            add("descend", a, np.repeat(cur_h[None], len(a), 0))
        else:
            move("descend", down)
        ls = mo.get("lift_on_stall") or {}
        close_hold_s = mo["close_hold_s"]
        if ls.get("enabled", False):  # hold up to timeout_s; the policy starts the lift early on a finger stall
            close_hold_s = max(close_hold_s, float(ls.get("timeout_s", 1.2)))
        if soft:   # the servo lags ~0.38 s and is soft: hold until the squeeze has built and the read is trustworthy
            close_hold_s = max(close_hold_s, float(g["soft_lag_s"]) + float(g["soft_settle_s"])
                               + float(cfg["held"]["soft_read_after_s"]))
        hand("close", hand_cls, close_hold_s)
        cs = min(1.0, float(mo.get("carry_scale", 1.0)))  # motion.carry_scale <= 1: slower lift + carry only
        if soft and len(up) > 1:   # lift_check: first soft_lift_check_m slowly; the policy re-reads the grip at its end
            k = min(len(up) - 1, max(1, int(np.ceil(float(g["soft_lift_check_m"]) / float(mo["cart_step"])))))
            move("lift_check", up[:k], scale=cs * float(g["soft_lift_check_scale"]))
            move("lift", up[k:], scale=cs)
        else:
            move("lift", up, scale=cs)
        move("carry", carry + lower, scale=cs)
        rw = g.get("release_wait_s")  # grasp.release_wait_s: the retreat starts >= this long after the open command
        hand("open", hand_rel, mo["open_hold_s"] if rw is None else max(mo["open_hold_s"], float(rw)))  # widest posture: `open` alone left the cube wedged in the sim
        move("retreat", retreat)
        move("home", [home], hand_to=hand_open)
        plan.arm, plan.hand, plan.phases = np.vstack(arm_parts), np.vstack(hand_parts), phases

        # clearance over the whole trajectory (grasp phases use the fingertip margin)
        grasp_ticks = np.zeros(len(plan.arm), bool)
        near = ("descend", "close", "lift_check", "lift") + (("carry", "open", "retreat") if place_xy is not None else ())
        for name, s, e in phases:
            if name in near:
                grasp_ticks[s:e] = True
        closing = np.zeros(len(plan.arm), bool)
        for name, s, e in phases:
            closing[s:e] |= name == "close" or (place_xy is not None and name == "open")
        for k in range(0, len(plan.arm), 3):
            # closing: the thumb swings through its lowest point; touching the mat lightly is acceptable
            need = 0.003 if closing[k] else g["fingertip_clearance"] if grasp_ticks[k] else mo["counter_clearance"]
            c = self.clearance(side, plan.arm[k], plan.hand[k])
            if c < need - 2e-3:
                name = next(n for n, s, e in phases if s <= k < e)
                # the object itself sits between the fingers during descend/lift; only the counter matters here
                plan.reason = f"hand {100 * c:.1f} cm above counter during {name} (need {100 * need:.1f})"
                return plan
        if self.cfg["policy"].get("novel_obstacles", False) and clearance_obstacles:
            margin = float(g.get("generic_obstacle_margin_m", 0.025))
            for k in range(0, len(plan.arm), 3):
                pts = self.kin.hand_points(side, plan.arm[k], plan.hand[k])
                for obstacle in clearance_obstacles:
                    q = np.asarray(obstacle, float).reshape(-1)
                    if len(q) < 2:
                        continue
                    radius = float(q[2]) if len(q) > 2 else 0.025
                    top_z = float(q[3]) if len(q) > 3 else zc + self.height
                    near = np.linalg.norm(pts[:, :2] - q[:2], axis=1) < radius + margin
                    if np.any(near & (pts[:, 2] < top_z + margin)):
                        name = next(n for n, s, e in phases if s <= k < e)
                        plan.reason = f"novel obstacle clearance during {name}"
                        return plan
        drop = np.zeros(len(plan.arm), bool)
        for name, s_, e in phases:
            drop[s_:e] |= name in ("open",)
        source_phases = {"descend", "close", "lift_check", "lift"} if source_bin is not None else set()
        for k in range(0, len(plan.arm), 3):
            name = next(n for n, s_, e in phases if s_ <= k < e)
            if drop[k]:
                continue
            if name in source_phases:
                if self.hits_bin_wall(side, plan.arm[k], plan.hand[k], allowed_inside=source_bin):
                    plan.reason = f"hand intersects the {source_bin} bin wall during {name}"
                    return plan
                continue
            c = self.bin_clearance(side, plan.arm[k], plan.hand[k])
            if c < 0.003:
                plan.reason = f"hand {100 * c:.1f} cm above a bin rim during {name}"
                return plan
        nboxes = self.neutral_boxes()   # perception.neutral_obstacles: [] when off, no work
        if nboxes:
            nmargin = float((cfg["perception"].get("neutral_params") or {}).get("margin_m", 0.02))
            for k in range(0, len(plan.arm), 3):
                gap = self.neutral_clearance(side, plan.arm[k], plan.hand[k], nboxes)
                if gap < nmargin:
                    name = next(n for n, s_, e in phases if s_ <= k < e)
                    plan.reason = f"hand {100 * gap:.1f} cm from a neutral obstacle during {name} (need {100 * nmargin:.1f})"
                    if self._neutral_deadline is not None and self._neutral_fallback is None \
                            and not self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand):
                        self._neutral_fallback = copy.copy(plan)   # complete stock plan, kept for fail-open
                        self._neutral_fallback.ok, self._neutral_fallback.reason = True, ""
                    return plan
        plan.safety_clips = self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
        if plan.safety_clips:
            plan.reason = f"{plan.safety_clips} safety clips in dry run"
            return plan
        plan.ok = True
        return plan

    def dry_run_safety(self, arm_q, hand_q, arm, hand):
        f = SafetyFilter(self.spec, self.safety)
        f.reset(arm_q, hand_q, 0.0)
        for k in range(len(arm)):
            f.filter({"arm_q_target": arm[k], "hand_q_target": hand[k]}, (k + 1) * self.dt)
        return f.clipped_steps

    def go_to(self, arm_from, hand_from, arm_to, hand_to):
        """Joint-space move (both arms + hands) at the configured speeds, e.g. start pose -> home."""
        mo = self.cfg["motion"]
        a = time_scale([arm_from, arm_to], mo["arm_max_vel"], mo["arm_max_acc"], self.dt)
        h = self.hand_ramp(hand_from, hand_to)
        n = max(len(a), len(h))
        a = np.vstack([a, np.repeat(np.asarray(arm_to, float)[None], n - len(a), 0)]) if len(a) < n else a
        h = np.vstack([h, np.repeat(np.asarray(hand_to, float)[None], n - len(h), 0)]) if len(h) < n else h
        return a, h
