"""Bounded head-extrinsic and counter-plane fitting for the pre-motion calibration phase.

The public surface is deliberately small:

* :class:`PrimaryCalibrationFitter.fit` selects hand ICP, plane-only, or nominal.
* :class:`AsyncCalibrationPhase.advance` primes two still frames and runs ``fit`` off the control loop.
* :class:`CounterHeightTracker.observe` freshness-checks and robustly smooths later plane heights.

All inputs are copies owned by the caller.  No method mutates a shared Calibration or Scene object.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .hand_calib import ArmSurface, _rodrigues, _rot_deg


def _rotation_vector(R):
    """Principal rotation vector for a 3x3 rotation matrix."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    if angle < 1e-10:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:  # near pi; eigenvector is stable where the skew part is not
        _, vec = np.linalg.eigh((R + np.eye(3)) * 0.5)
        axis = vec[:, -1]
    else:
        axis /= n
    return axis * angle


def _rotation_between(a, b):
    """Smallest rotation taking unit vector ``a`` to unit vector ``b``."""
    a = np.asarray(a, float) / max(np.linalg.norm(a), 1e-12)
    b = np.asarray(b, float) / max(np.linalg.norm(b), 1e-12)
    cross = np.cross(a, b)
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-10:
        if dot > 0.0:
            return np.eye(3)
        axis = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return _rodrigues(np.pi * axis)
    return _rodrigues(np.arctan2(np.linalg.norm(cross), dot) * cross / np.linalg.norm(cross))


@dataclass
class PlaneFit:
    success: bool
    T: np.ndarray
    normal_camera: np.ndarray = field(default_factory=lambda: np.zeros(3))
    counter_z: float = float("nan")
    tilt_deg: float = float("inf")
    residual_m: float = float("inf")
    n_points: int = 0
    n_inliers: int = 0
    reason: str = ""
    point_camera: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class CalibrationFitResult:
    source: str                    # hand | plane | nominal
    T: np.ndarray
    counter_z: float
    hand_accepted: bool
    elapsed_ms: float
    checks: tuple[tuple[str, bool, str], ...] = ()
    plane: PlaneFit | None = None
    n_model: int = 0
    n_inliers: int = 0
    coverage: float = 0.0
    valid_frac: float = 0.0
    residual_m: float = float("inf")
    spread_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dt_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    drot_deg: float = 0.0
    reason: str = ""

    def summary(self):
        d = 1000.0 * self.dt_m
        spread = 100.0 * np.asarray(self.spread_m)
        return (f"source={self.source} dt=({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f})mm "
                f"drot={self.drot_deg:.2f}deg inliers={self.n_inliers} coverage={self.coverage:.2f} "
                f"valid={self.valid_frac:.2f} rms={self.residual_m * 1000.0:.2f}mm "
                f"spread_cm=({spread[0]:.1f},{spread[1]:.1f},{spread[2]:.1f}) "
                f"time={self.elapsed_ms:.1f}ms ({self.reason})")


class FrameGuard:
    """Reject frames that predate motion or disagree with a joint sample at least ``pair_s`` earlier."""

    def __init__(self, pair_s=0.1, joint_rad=0.02):
        self.pair_s = float(pair_s)
        self.joint_rad = float(joint_rad)
        self.previous = None

    def reset(self):
        self.previous = None

    def check(self, timestamp, joints, last_motion_time=None, require_pair=False):
        try:
            timestamp = float(timestamp)
            joints = np.asarray(joints, float).copy()
        except Exception:
            return False, "invalid frame metadata"
        if not np.isfinite(timestamp) or joints.ndim != 1 or not np.all(np.isfinite(joints)):
            return False, "invalid frame metadata"
        if last_motion_time is not None and timestamp <= float(last_motion_time) + 1e-9:
            return False, f"stale timestamp {timestamp:.3f} <= last motion {float(last_motion_time):.3f}"
        prev = self.previous
        if prev is not None and timestamp < prev[0] - 1e-9:
            self.previous = (timestamp, joints)
            return False, f"timestamp moved backwards {timestamp:.3f} < {prev[0]:.3f}"
        if prev is None:
            self.previous = (timestamp, joints)
            return (False, "primer frame") if require_pair else (True, "fresh")
        dt = timestamp - prev[0]
        if dt + 1e-9 < self.pair_s:
            return (False, f"frame pair only {dt:.3f}s apart") if require_pair else (True, "fresh")
        self.previous = (timestamp, joints)
        if prev[1].shape != joints.shape:
            return False, "joint vector shape changed"
        moved = float(np.max(np.abs(joints - prev[1])))
        if moved > self.joint_rad:
            return False, f"joints moved {moved:.4f}>{self.joint_rad:.4f}rad over {dt:.3f}s"
        return True, f"fresh/still pair dt={dt:.3f}s dq={moved:.4f}rad"


class PrimaryCalibrationFitter:
    """Full 6-DOF point-to-plane camera fit with a nominal-pose prior and guarded fallbacks."""

    def __init__(self, params, surface: ArmSurface | None = None, crosscheck=False):
        self.p = dict(params or {})
        self.surface = surface or ArmSurface()
        self.crosscheck = bool(crosscheck)

    @staticmethod
    def _deproject(K, depth_scale, u, v, depth):
        z = np.asarray(depth, float) * float(depth_scale)
        return np.c_[((u - K[0, 2]) / K[0, 0]) * z, ((v - K[1, 2]) / K[1, 1]) * z, z]

    def fit_plane(self, cal, cam, depth, T=None, counter_hint=None):
        """Robust counter plane in optical coordinates; ``T`` controls only crop/orientation and returned height."""
        T = np.asarray(cal._T(cam) if T is None else T, float)
        c = cal.cams[cam]
        fallback = PlaneFit(False, T.copy(), reason="no plane")
        if depth is None or np.ndim(depth) != 2:
            fallback.reason = "no depth"
            return fallback
        depth = np.asarray(depth)
        h, w = depth.shape
        stride = max(2, int(self.p.get("pixel_stride", 3)))
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        vv, uu = vv.ravel(), uu.ravel()
        d = depth[vv, uu].astype(float) * float(c.depth_scale)
        good = np.isfinite(d) & (d > 0.15) & (d < 1.8)
        if int(good.sum()) < int(self.p.get("plane_min_inliers", 500)):
            fallback.reason = "too few valid plane pixels"
            return fallback
        Q = self._deproject(c.K, c.depth_scale, uu[good], vv[good], depth[vv[good], uu[good]])
        P = Q @ T[:3, :3].T + T[:3, 3]
        hint = 0.01 if counter_hint is None else float(counter_hint)
        crop = ((P[:, 0] > 0.08) & (P[:, 0] < 0.72) & (np.abs(P[:, 1]) < 0.52) &
                (P[:, 2] > hint - 0.10) & (P[:, 2] < hint + 0.26))
        Q, P = Q[crop], P[crop]
        max_points = 6000
        if len(Q) > max_points:
            take = np.linspace(0, len(Q) - 1, max_points).astype(int)
            Q, P = Q[take], P[take]
        n = len(Q)
        minimum = int(self.p.get("plane_min_inliers", 500))
        if n < minimum:
            fallback.reason, fallback.n_points = "counter crop too small", n
            return fallback
        threshold = 0.008
        rng = np.random.default_rng(914)
        best = None
        for _ in range(64):
            q = Q[rng.choice(n, 3, replace=False)]
            normal = np.cross(q[1] - q[0], q[2] - q[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal /= norm
            distance = np.abs(Q @ normal - q[0] @ normal)
            inside = distance <= threshold
            count = int(inside.sum())
            if best is None or count > best[0]:
                best = (count, inside)
        if best is None or best[0] < minimum:
            fallback.reason, fallback.n_points = "plane RANSAC rejected", n
            fallback.n_inliers = 0 if best is None else best[0]
            return fallback
        inside = best[1]
        gates = (0.012, 0.008, 0.006, 0.004)
        for gate in gates:
            q = Q[inside]
            center = q.mean(axis=0)
            _, vectors = np.linalg.eigh((q - center).T @ (q - center))
            normal = vectors[:, 0]
            if (T[:3, :3] @ normal) @ np.array([0.0, 0.0, 1.0]) < 0.0:
                normal = -normal
            signed = Q @ normal - center @ normal
            med = float(np.median(signed))
            inside = np.abs(signed - med) <= gate
            if int(inside.sum()) < minimum:
                break
        ni = int(inside.sum())
        if ni < minimum:
            fallback.reason, fallback.n_points, fallback.n_inliers = "trimmed plane rejected", n, ni
            return fallback
        center = Q[inside].mean(axis=0)
        residual = float(np.sqrt(np.mean((Q[inside] @ normal - center @ normal) ** 2)))
        Rlevel = _rotation_between(T[:3, :3] @ normal, np.array([0.0, 0.0, 1.0]))
        corrected = T.copy()
        corrected[:3, :3] = Rlevel @ T[:3, :3]
        # Height is measured in the supplied camera frame. ``corrected`` is a separate plane-only fallback pose;
        # using it here would let the counter plane alter the hand-selected frame before the height estimate.
        z = float(np.median(P[inside, 2]))
        tilt = float(np.degrees(np.arccos(np.clip((T[:3, :3] @ normal)[2], -1.0, 1.0))))
        max_rms = float(self.p.get("plane_max_rms_m", 0.006))
        ok = np.isfinite(residual) and residual <= max_rms
        return PlaneFit(ok, corrected if ok else T.copy(), normal, z, tilt, residual, n, ni,
                        "ok" if ok else f"plane RMS {residual * 1000.0:.1f}>{max_rms * 1000.0:.1f}mm",
                        center.copy())

    @staticmethod
    def _visible(P, T, K, w, h, cell_px=4, tol=0.008):
        pc = (P - T[:3, 3]) @ T[:3, :3]
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]
            v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        ok = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        gw, gh = (w + cell_px - 1) // cell_px, (h + cell_px - 1) // cell_px
        cell = np.full(len(P), -1)
        cell[ok] = (v[ok] // cell_px).astype(int) * gw + (u[ok] // cell_px).astype(int)
        zmin = np.full(gw * gh, np.inf)
        np.minimum.at(zmin, cell[ok], z[ok])
        visible = ok.copy()
        visible[ok] = z[ok] <= zmin[cell[ok]] + tol
        return visible, np.isfinite(zmin).reshape(gh, gw)

    def _starts(self, T0, Q, tree, normals):
        """Score a broad deterministic start set, then optimize only the best three plus nominal."""
        tm = float(self.p.get("multistart_trans_m", 0.012))
        rm = np.radians(float(self.p.get("multistart_rot_deg", 2.0)))
        signs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
        raw = [(T0[:3, :3].copy(), T0[:3, 3].copy(), "nominal")]
        for sign in np.vstack([signs, -signs]):
            raw.append((T0[:3, :3] @ _rodrigues(rm * sign), T0[:3, 3] + tm * sign,
                        "multi " + "".join("+" if x > 0 else "-" for x in sign)))

        def score(item):
            R, t, _ = item
            pd = Q @ R.T + t
            dist, idx = tree.query(pd, distance_upper_bound=float(self.p.get("thr_start_m", 0.055)))
            keep = np.isfinite(dist)
            if keep.sum() < 6:
                return (0, np.inf)
            err = np.einsum("ij,ij->i", normals[idx[keep]], pd[keep] - tree.data[idx[keep]])
            return (int(keep.sum()), float(np.median(np.abs(err))))

        scored = [(score(item), item) for item in raw]
        ranked = sorted(scored[1:], key=lambda x: (-x[0][0], x[0][1]))[:3]
        return [raw[0]] + [item for _, item in ranked]

    def _icp(self, Q, M, Mn, T0):
        p, tree = self.p, cKDTree(M)
        best = None
        gates = np.geomspace(float(p.get("thr_start_m", 0.055)), float(p.get("thr_end_m", 0.008)),
                             max(2, int(p.get("max_iters", 10))))
        sigma_r = np.radians(float(p.get("prior_rot_deg", 3.0)))
        sigma_t = float(p.get("prior_trans_m", 0.020))
        sigma_d = float(p.get("depth_noise_m", 0.004))
        independent = float(p.get("decorrelated_samples", 50))
        for R0, t0, label in self._starts(T0, Q, tree, Mn):
            R, t = R0.copy(), t0.copy()
            for threshold in gates:
                Pd = Q @ R.T + t
                distance, idx0 = tree.query(Pd, distance_upper_bound=float(threshold))
                mask = np.isfinite(distance)
                if int(mask.sum()) < 6:
                    break
                idx = idx0[mask]
                error = np.einsum("ij,ij->i", Mn[idx], Pd[mask] - M[idx])
                nk = Mn[idx] @ R
                J = np.hstack([np.cross(Q[mask], nk), nk])
                data_scale = independent / (max(int(mask.sum()), 1) * sigma_d ** 2)
                H = data_scale * (J.T @ J)
                g = data_scale * (J.T @ error)
                prior = np.r_[_rotation_vector(T0[:3, :3].T @ R) / sigma_r,
                              (t - T0[:3, 3]) / sigma_t]
                Jp = np.zeros((6, 6))
                Jp[:3, :3] = np.eye(3) / sigma_r
                Jp[3:, 3:] = R / sigma_t
                H += Jp.T @ Jp
                g += Jp.T @ prior
                xi = -np.linalg.solve(H + 1e-8 * np.eye(6), g)
                if np.linalg.norm(xi[:3]) > np.radians(1.5):
                    xi[:3] *= np.radians(1.5) / np.linalg.norm(xi[:3])
                if np.linalg.norm(xi[3:]) > 0.012:
                    xi[3:] *= 0.012 / np.linalg.norm(xi[3:])
                t = R @ xi[3:] + t
                R = R @ _rodrigues(xi[:3])
            Pd = Q @ R.T + t
            distance, idx0 = tree.query(Pd, distance_upper_bound=float(p.get("thr_end_m", 0.008)))
            mask = np.isfinite(distance)
            idx = idx0[mask]
            err = (np.einsum("ij,ij->i", Mn[idx], Pd[mask] - M[idx]) if mask.any() else np.empty(0))
            rms = float(np.sqrt(np.mean(err ** 2))) if len(err) else float("inf")
            rank = (int(mask.sum()), -rms)
            if best is None or rank > best[0]:
                best = (rank, R, t, mask, idx, rms, label)
        return best[1:]

    def fit(self, cal, cam, depth, arm_q, hand_q, counter_nominal):
        started = time.perf_counter()
        p, T0 = self.p, np.asarray(cal._T(cam), float).copy()
        checks = []

        def check(name, ok, detail):
            checks.append((name, bool(ok), str(detail)))
            return bool(ok)

        plane = self.fit_plane(cal, cam, depth, T0, counter_nominal)
        check("plane available", plane.success,
              f"inliers={plane.n_inliers} rms={plane.residual_m * 1000.0:.2f}mm reason={plane.reason}")
        try:
            c = cal.cams[cam]
            if depth is None or np.ndim(depth) != 2:
                raise ValueError("no depth")
            depth = np.asarray(depth)
            h, w = depth.shape
            P, N = self.surface.posed(arm_q, hand_q)
            visible, cells = self._visible(P, T0, c.K, w, h)
            M, Mn = P[visible], N[visible]
            n_model = len(M)
            minimum_model = int(p.get("min_model_points", 150))
            if not check("model points", n_model >= minimum_model, f"{n_model}>={minimum_model}"):
                raise ValueError("arms not in view")
            cell_px = 4
            radius = max(1, int(np.ceil(float(p.get("search_px", 56)) / cell_px)))
            cells = cv2.dilate(cells.astype(np.uint8),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
            stride = max(1, int(p.get("pixel_stride", 3)))
            vv, uu = np.mgrid[0:h:stride, 0:w:stride]
            vv, uu = vv.ravel(), uu.ravel()
            d = depth[vv, uu].astype(float) * float(c.depth_scale)
            keep = cells[vv // cell_px, uu // cell_px].astype(bool) & np.isfinite(d) & (d > 0.15) & (d < 1.5)
            Q = self._deproject(c.K, c.depth_scale, uu[keep], vv[keep], depth[vv[keep], uu[keep]])
            if plane.success:
                q0 = plane.normal_camera @ plane.point_camera
                signed = Q @ plane.normal_camera - q0
            else:
                pp = Q @ T0[:3, :3].T + T0[:3, 3]
                signed = pp[:, 2] - (counter_nominal + float(p.get("fallback_counter_dz_m", 0.15)))
            Q = Q[np.abs(signed) > float(p.get("counter_clearance_m", 0.058))]
            tree0 = cKDTree(M)
            near, _ = tree0.query(Q @ T0[:3, :3].T + T0[:3, 3],
                                  distance_upper_bound=float(p.get("thr_start_m", 0.055)) + 0.025)
            Q = Q[np.isfinite(near)]
            if len(Q) > int(p.get("max_data_points", 1800)):
                Q = Q[np.linspace(0, len(Q) - 1, int(p.get("max_data_points", 1800))).astype(int)]
            min_inliers = int(p.get("min_inliers", 300))
            if not check("arm depth candidates", len(Q) >= min_inliers, f"{len(Q)}>={min_inliers}"):
                raise ValueError("too few arm depth candidates")
            R, t, mask, idx, residual, start_label = self._icp(Q, M, Mn, T0)
            n_inliers = int(mask.sum())
            fitted = Q[mask] @ R.T + t
            if n_inliers:
                found, _ = cKDTree(fitted).query(M, distance_upper_bound=2.0 * float(p.get("thr_end_m", 0.008)))
                coverage = float(np.isfinite(found).mean())
                spread = np.ptp(fitted, axis=0)
            else:
                coverage, spread = 0.0, np.zeros(3)
            pc = (M - t) @ R
            pu = np.rint(c.K[0, 0] * pc[:, 0] / pc[:, 2] + c.K[0, 2]).astype(int)
            pv = np.rint(c.K[1, 1] * pc[:, 1] / pc[:, 2] + c.K[1, 2]).astype(int)
            inb = (pc[:, 2] > 0) & (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
            dz = np.full(len(M), np.inf)
            dz[inb] = np.abs(depth[pv[inb], pu[inb]].astype(float) * float(c.depth_scale) - pc[inb, 2])
            valid_frac = float((dz < 0.030).mean())
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            dt, drot = t - T0[:3, 3], _rot_deg(T0[:3, :3].T @ R)
            spread_axes = int(np.sum(spread >= float(p.get("min_spread_m", 0.10))))
            accepted = all((
                check("inlier points", n_inliers >= min_inliers, f"{n_inliers}>={min_inliers}"),
                check("residual", residual <= float(p.get("max_residual_m", 0.004)),
                      f"{residual * 1000.0:.2f}<={float(p.get('max_residual_m', 0.004)) * 1000.0:.2f}mm"),
                check("coverage", coverage >= float(p.get("min_coverage", 0.55)),
                      f"{coverage:.3f}>={float(p.get('min_coverage', 0.55)):.3f}"),
                check("returned depth", valid_frac >= float(p.get("min_valid_frac", 0.35)),
                      f"{valid_frac:.3f}>={float(p.get('min_valid_frac', 0.35)):.3f}"),
                check("two-axis spread", spread_axes >= 2,
                      f"spans={np.round(100.0 * spread, 1).tolist()}cm axes={spread_axes}>=2"),
                check("translation cap", np.linalg.norm(dt) <= float(p.get("max_trans_m", 0.05)),
                      f"{np.linalg.norm(dt) * 1000.0:.1f}<={float(p.get('max_trans_m', 0.05)) * 1000.0:.0f}mm"),
                check("rotation cap", drot <= float(p.get("max_rot_deg", 6.0)),
                      f"{drot:.2f}<={float(p.get('max_rot_deg', 6.0)):.2f}deg"),
            ))
            hand_plane = self.fit_plane(cal, cam, depth, T, counter_nominal)
            if self.crosscheck:
                agree = hand_plane.success and hand_plane.tilt_deg <= float(p.get("plane_tilt_agree_deg", 1.5))
                accepted = check("plane tilt cross-check", agree,
                                 f"tilt={hand_plane.tilt_deg:.2f}<={float(p.get('plane_tilt_agree_deg', 1.5)):.2f}deg "
                                 f"plane={hand_plane.reason}") and accepted
            if accepted:
                return CalibrationFitResult("hand", T, hand_plane.counter_z if hand_plane.success else float("nan"),
                                            True, (time.perf_counter() - started) * 1000.0, tuple(checks), plane,
                                            n_model, n_inliers, coverage, valid_frac, residual, spread, dt, drot,
                                            f"accepted from {start_label}")
        except Exception as exc:
            check("hand solve", False, f"{type(exc).__name__}: {exc}")
            n_model = locals().get("n_model", 0)
            n_inliers = locals().get("n_inliers", 0)
            coverage = locals().get("coverage", 0.0)
            valid_frac = locals().get("valid_frac", 0.0)
            residual = locals().get("residual", float("inf"))
            spread = locals().get("spread", np.zeros(3))
            dt = locals().get("dt", np.zeros(3))
            drot = locals().get("drot", 0.0)
        if plane.success:
            check("fallback selection", True, "plane-only")
            return CalibrationFitResult("plane", plane.T, plane.counter_z, False,
                                        (time.perf_counter() - started) * 1000.0, tuple(checks), plane,
                                        n_model, n_inliers, coverage, valid_frac, residual, spread, dt, drot,
                                        "hand rejected; plane-only fallback")
        prior_z = float(counter_nominal) + float(p.get("fallback_counter_dz_m", 0.15))
        check("fallback selection", True, f"nominal extrinsic, counter_z prior={prior_z:.3f}")
        return CalibrationFitResult("nominal", T0, prior_z, False,
                                    (time.perf_counter() - started) * 1000.0, tuple(checks), plane,
                                    n_model, n_inliers, coverage, valid_frac, residual, spread, dt, drot,
                                    "hand and plane rejected; nominal fallback")

@dataclass
class PhaseUpdate:
    hold: bool
    result: CalibrationFitResult | None = None
    reason: str = ""


class AsyncCalibrationPhase:
    """Generation-guarded one-worker adapter used by the policy's hold-only calibration phase."""

    def __init__(self, fitter, params):
        self.fitter, self.params = fitter, dict(params or {})
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="primary-calib")
        self.lock = threading.Lock()
        self.generation = 0
        self.future = None
        self.started = 0.0
        self.guard = FrameGuard(self.params.get("stable_pair_s", 0.1), self.params.get("stable_joint_rad", 0.02))

    def reset(self):
        with self.lock:
            self.generation += 1
            self.future = None
            self.started = 0.0
            self.guard.reset()

    def advance(self, cal, cam, depth, arm_q, hand_q, timestamp, counter_nominal, last_motion_time=None):
        with self.lock:
            generation, future = self.generation, self.future
        if future is not None:
            if future.done():
                try:
                    token, result = future.result()
                except Exception as exc:
                    token, result = generation, exc
                with self.lock:
                    if future is self.future:
                        self.future = None
                if token != self.generation:
                    return PhaseUpdate(True, reason="discarded stale calibration generation")
                if isinstance(result, Exception):
                    return PhaseUpdate(False, reason=f"worker error: {type(result).__name__}: {result}")
                return PhaseUpdate(False, result, "fit complete")
            if time.perf_counter() - self.started > float(self.params.get("job_timeout_s", 1.5)):
                with self.lock:
                    if future is self.future:
                        self.generation += 1
                        self.future = None
                return PhaseUpdate(False, reason="fit timed out")
            return PhaseUpdate(True, reason="fit running")
        joints = np.r_[np.asarray(arm_q, float), np.asarray(hand_q, float)]
        ok, reason = self.guard.check(timestamp, joints, last_motion_time, require_pair=True)
        if not ok:
            return PhaseUpdate(True, reason=reason)
        args = (cal, cam, np.asarray(depth).copy(), np.asarray(arm_q, float).copy(),
                np.asarray(hand_q, float).copy(), float(counter_nominal))

        def work():
            return generation, self.fitter.fit(*args)

        with self.lock:
            self.started = time.perf_counter()
            self.future = self.executor.submit(work)
        return PhaseUpdate(True, reason="fit submitted")


@dataclass
class CounterUpdate:
    committed: bool
    counter_z: float | None
    accepted: bool
    reason: str
    n_samples: int
    tilt_deg: float = float("nan")
    residual_m: float = float("inf")


class CounterHeightTracker:
    """MAD-gated counter height tracker with stale-frame rejection and bounded smoothing."""

    def __init__(self, fitter, params, nominal_z):
        self.fitter, self.p, self.nominal_z = fitter, dict(params or {}), float(nominal_z)
        self.guard = FrameGuard(self.p.get("stale_pair_s", 0.1), self.p.get("stale_joint_rad", 0.02))
        self.reset()

    def reset(self):
        self.samples, self.tries, self.value = [], 0, None
        self.guard.reset()

    def observe(self, cal, cam, depth, T, timestamp, joints, last_motion_time=None):
        self.tries += 1
        ok, reason = self.guard.check(timestamp, joints, last_motion_time, require_pair=True)
        if not ok:
            return CounterUpdate(self.value is not None, self.value, False, reason, len(self.samples))
        plane = self.fitter.fit_plane(cal, cam, depth, T, self.value if self.value is not None else self.nominal_z)
        max_tilt = float(self.p.get("max_tilt_deg", 3.0))
        if not plane.success or plane.tilt_deg > max_tilt or not np.isfinite(plane.counter_z):
            why = plane.reason if not plane.success else f"plane tilt {plane.tilt_deg:.2f}>{max_tilt:.2f}deg"
            return CounterUpdate(self.value is not None, self.value, False, why, len(self.samples),
                                 plane.tilt_deg, plane.residual_m)
        z = float(plane.counter_z)
        if self.samples:
            med = float(np.median(self.samples))
            mad = float(np.median(np.abs(np.asarray(self.samples) - med)))
            gate = max(float(self.p.get("mad_floor_m", 0.004)),
                       float(self.p.get("mad_scale", 3.5)) * 1.4826 * mad)
            if len(self.samples) >= 2 and abs(z - med) > gate:
                return CounterUpdate(self.value is not None, self.value, False,
                                     f"MAD gate |{z:.4f}-{med:.4f}|>{gate:.4f}", len(self.samples),
                                     plane.tilt_deg, plane.residual_m)
        self.samples.append(z)
        self.samples = self.samples[-int(self.p.get("max_samples", 7)):]
        need = max(3, int(self.p.get("samples", 3)))
        if len(self.samples) < need:
            return CounterUpdate(False, None, True, f"consistent sample {len(self.samples)}/{need}", len(self.samples),
                                 plane.tilt_deg, plane.residual_m)
        target = float(np.median(self.samples))
        if self.value is None:
            self.value = target
        else:
            alpha = float(np.clip(self.p.get("smooth_alpha", 0.35), 0.0, 1.0))
            step = np.clip(alpha * (target - self.value), -float(self.p.get("max_update_m", 0.01)),
                           float(self.p.get("max_update_m", 0.01)))
            self.value += float(step)
        return CounterUpdate(True, self.value, True, "committed MAD median / bounded smooth", len(self.samples),
                             plane.tilt_deg, plane.residual_m)
