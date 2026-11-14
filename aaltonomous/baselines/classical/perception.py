"""Head RGB-D -> coloured objects in the pelvis frame. Shape-agnostic: an object is a coloured blob, its grasp
target is the centre of its top face.

    det = Perception(cfg, scene, calibration).detect(obs["head_rgb"], obs["head_depth"])
    # -> [Detection(color, top (3,) pelvis, n_px, n_depth, method, yaw), ...]

Per colour: HSV threshold (config.yaml perception.hsv) -> opening -> connected components. Per blob: erode, take
the pixels with valid depth, deproject them (calibration.yaml intrinsics + T_pelvis_optical), keep the top face
(points within `top_band` of a robust max height) and use the median of those as the top centre. With too few valid
depth pixels (holes, or no depth stream) the blob centroid ray is intersected with the plane at the known object
height instead (method "plane"). Blobs over the bins, off the mat, or at a wrong height are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import time

import cv2
import numpy as np

from .bin_vote import bin_geometry
from .shape_id import classify_top_face   # the ONE shape decision; swap this import to change the classifier
from .shape_2d import decide_shape2d, shape2d_features
from .shape_id import classify_top_face_diag   # diagnostics only (perception.shape_diag): never decides anything
from .hue_guard import AMBIGUOUS_CONF, Cand, classify_hue, judge, top_face_hsv


def colour_confidence(hsv, blob, ranges_by_class, color, min_area_px=120):
    """Cheap 0..1 confidence that ``blob`` is an object of ``color``.

    The score combines class-mask purity, HSV threshold margin, V brightness,
    and a brightness-aware penalty near the real orange bin tape (H 2..7,
    V about 119).  Absolute brightness is not scored: shadow is not a colour error.  Hue distance is circular, so stock red at H=0 is correctly
    treated as being inside the wrapped red range.
    """
    hsv = np.asarray(hsv, dtype=np.uint8)
    use = np.asarray(blob, dtype=bool)
    px = hsv[use]
    if not len(px) or color not in ranges_by_class:
        return 0.0

    inside = np.zeros(len(px), bool)
    sv_margin = np.zeros(len(px), float)
    hue_allowed = np.zeros(180, bool)
    for lo, hi in ranges_by_class[color]:
        lo, hi = np.asarray(lo), np.asarray(hi)
        hit = np.all((px >= lo) & (px <= hi), axis=1)
        inside |= hit
        # The meaningful S/V rejection edges are the lower limits.  Treating
        # 255 as an uncertainty edge would incorrectly punish bright top faces.
        margin = np.minimum(np.clip((px[:, 1].astype(float) - float(lo[1])) / 30.0, 0.0, 1.0),
                            np.clip((px[:, 2].astype(float) - float(lo[2])) / 60.0, 0.0, 1.0))
        sv_margin = np.maximum(sv_margin, np.where(hit, margin, 0.0))
        hlo, hhi = max(0, int(lo[0])), min(179, int(hi[0]))
        if hlo <= hhi:
            hue_allowed[hlo:hhi + 1] = True
    purity = float(inside.mean())
    support = float(np.clip(len(px) / max(float(min_area_px), 1.0), 0.0, 1.0))

    def distance_to(mask):
        """Circular hue distance lookup; fixed 180x180 work, not blob-size squared."""
        target = np.flatnonzero(mask)
        if not len(target):
            return np.full(180, 180.0)
        h = np.arange(180)[:, None]
        delta = np.abs(h - target[None, :])
        return np.minimum(delta, 180 - delta).min(axis=1).astype(float)

    h = px[:, 0].astype(int)
    own_margin = distance_to(~hue_allowed)[h]
    hue_factor = float(np.clip(np.median(own_margin[inside]) / 3.0, 0.0, 1.0)) if inside.any() else 0.0
    sv_factor = float(np.median(sv_margin[inside])) if inside.any() else 0.0

    # Also soften a class whose pixels sit within three hue units of another
    # class.  This currently matters most around red/orange vs yellow.
    neighbour = np.zeros(180, bool)
    for other, ranges in ranges_by_class.items():
        if other == color:
            continue
        for lo, hi in ranges:
            hlo, hhi = max(0, int(lo[0])), min(179, int(hi[0]))
            if hlo <= hhi:
                neighbour[hlo:hhi + 1] = True
    neighbour_distance = distance_to(neighbour)[h]
    neighbour_factor = float(np.clip(np.median(neighbour_distance[inside]) / 3.0, 0.0, 1.0)) if inside.any() else 0.0

    tape = np.zeros(180, bool)
    tape[2:8] = True
    tape_proximity = np.clip(1.0 - distance_to(tape)[h] / 4.0, 0.0, 1.0)
    tape_dimness = np.clip((200.0 - px[:, 2].astype(float)) / 100.0, 0.0, 1.0)
    tape_factor = 1.0 - float(np.mean(tape_proximity * tape_dimness))
    return float(np.clip(support * purity * hue_factor * sv_factor * neighbour_factor * tape_factor,
                         0.0, 1.0))


@dataclass
class Detection:
    color: str
    top: np.ndarray            # top-face centre, pelvis frame (m)
    n_px: int
    n_depth: int
    method: str                # "depth" | "plane"
    yaw: float | None = None   # in-plane orientation of the top face (rad), info only
    uv: tuple = field(default=(0.0, 0.0))
    shape: str | None = None   # "cube" | "cylinder" | "triangle" | None (unknown)
    apex_yaw: float | None = None  # triangle only: pelvis-frame heading (rad, x fwd, y left) centre -> any one vertex (mod 120 deg)
    apex_conf: float = 0.0     # confidence of apex_yaw, 0..1 (from perception.triangle_pose)
    shape_conf: float = 0.0    # shape_id confidence (0..1) behind `shape`
    apex_gate: tuple | None = None  # grasp.posture_apex_gate only: (apex_conf, max vertex angle deg) of a "triangle" top
    shape_path: str | None = None  # shape_id path ("metric" | "blob" | "abstain_*" | "off"), only filled when grasp.close_posture_by_shape
    generic_required: bool = False  # grasp.generic_fallback: label absent/below confidence threshold
    generic_width: float | None = None  # accepted proposal's closing-axis width (m); None means abstain
    generic_confidence: float = 0.0
    shape_diag: dict | None = None  # perception.shape_diag only: source, n_top, ratio_cloud, ratio_blob, path, label,
                                    # conf, near_bin, dz_top (top z - counter z, m), top_xy (list of [x, y], pelvis)
    color_conf: float = 1.0  # policy.place_gate only; 1 keeps synthetic/legacy detections confidently compatible
    hsv_med: tuple | None = None  # perception.hue_guard only: (H, S, V) median of the brightest 35 pct by V of the object pixels
    sat_med: float | None = None  # perception.color_gate.sat_floor only: median HSV S (0-255) over the blob mask

    def __repr__(self):
        x, y, z = self.top
        return (f"{self.color:<6} top=({x:+.3f},{y:+.3f},{z:+.3f}) px={self.n_px} depth_px={self.n_depth} "
                f"[{self.method}]")


def perturb_extrinsics(T, err):
    """T_pelvis_optical with an injected error [dpitch_deg, dyaw_deg, dx, dy, dz]: the camera is rotated about its own
    position by Rz(dyaw) @ Ry(dpitch) in the PELVIS frame (positive pitch = nose down, x fwd / z up) and translated by
    (dx, dy, dz) m. Object points deprojected with the result move accordingly (a -7 deg pitch error at 0.6 m range
    puts objects ~7 cm off)."""
    dp, dy, dx, dyy, dz = (float(v) for v in err)
    a, b = np.radians(dp), np.radians(dy)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    out = np.array(T, float).copy()
    out[:3, :3] = Rz @ Ry @ out[:3, :3]
    out[:3, 3] += (dx, dyy, dz)
    return out


@dataclass
class BinCalibration:
    """Result of one episode's depth-only alignment to the fixed bin geometry."""
    success: bool
    valid_frame: bool
    confidence: float
    xy_offset: np.ndarray
    n_points: int
    residual_m: float
    elapsed_ms: float
    reason: str


@dataclass
class PitchCalibration:
    """Result of one episode's optional camera-tilt calibration."""
    success: bool
    valid_frame: bool
    pitch_deg: float
    n_points: int
    residual_m: float
    elapsed_ms: float
    reason: str
    mode: str = "bin_rim"
    roll_deg: float = 0.0
    camera_height_m: float = np.nan
    plane_height_m: float = np.nan
    n_inliers: int = 0
    T: object = field(default=None, repr=False, compare=False)  # corrected extrinsic when success


class Perception:
    def __init__(self, cfg, scene, calibration, log=None):
        self.cfg = cfg["perception"]
        self.scene = scene
        self.cal = calibration
        self.cam = self.cfg["camera"]
        # perception.principal_point (Oliver lane 3): "file" = calibration.yaml K (stock, right for the real D435i);
        # "center" = cx, cy at the image centre and fx = fy = mean, which is what Isaac renders: IsaacLab drops
        # aperture offsets ("c_x and c_y will be half of width and height", utils/sensors.py).
        # Perception gets its own shallow copy of the calibration, so every path (pixel_to_pelvis, project, the ray
        # fallbacks) uses the same K and nothing outside perception changes.
        if self.cfg.get("principal_point", "file") == "center":
            import copy
            import dataclasses
            cam = calibration.cams[self.cam]
            K = np.array(cam.K, float).copy()
            f = 0.5 * (K[0, 0] + K[1, 1])
            K[0, 0] = K[1, 1] = f
            K[0, 2], K[1, 2] = cam.width / 2.0, cam.height / 2.0
            self.cal = copy.copy(calibration)
            self.cal.cams = dict(calibration.cams)
            self.cal.cams[self.cam] = dataclasses.replace(cam, K=K)
        # perception.inject_cam_error (TEST ONLY): the head T_pelvis_optical perception uses is perturbed, so the object
        # estimate is wrong the way it is on a mis-calibrated real robot. Own copy, nothing outside perception changes.
        inj = self.cfg.get("inject_cam_error")
        if inj is not None:
            import copy
            import dataclasses
            cam = self.cal.cams[self.cam]
            self.cal = copy.copy(self.cal)
            self.cal.cams = dict(self.cal.cams)
            self.cal.cams[self.cam] = dataclasses.replace(cam, T_pelvis_optical=perturb_extrinsics(cam.T_pelvis_optical, inj))
        # what reset_calibration() restores: AFTER every construction step above (principal_point copy), so a reset never
        # reverts perception to the file K. perception.hand_calibration swaps self.cal for a corrected COPY per episode.
        self.cal_nominal = self.cal
        scene.counter_z_prior = (cfg.get("scene") or {}).get("counter_z_prior")   # None = nominal
        self.ah = (cfg.get("calib") or {}).get("plane") or {}   # calib.plane.apply_height (FLAG, default OFF)
        self._ah_est, self._ah_tries = [], 0
        self.size = float(self.cfg.get("object_size") or scene.size)
        self.height = float(self.cfg.get("object_height") or scene.object_height())
        # Pelvis-frame xy correction added to every detection. The configured value is the fail-safe fallback.
        self.base_xy_offset = np.asarray(self.cfg.get("xy_offset") or [0.0, 0.0], float)
        self.xy_offset = self.base_xy_offset.copy()
        self.calibration = None
        # false = off, true = the original bin-rim fit, "table" = assumption-free counter-plane fit.
        self.online_pitch_cal = self.cfg.get("online_pitch_cal", False)
        self.pitch_calibration = None
        self.pitch_T = None
        # opt-in shape / triangle pose (all default OFF = stock behaviour)
        grasp = cfg.get("grasp", {})
        self.generic_fallback = bool(grasp.get("generic_fallback", False))
        self.generic_shape_min_conf = float(grasp.get("generic_shape_min_conf", 0.5))
        self.generic_min_confidence = float(grasp.get("generic_min_confidence", 0.5))
        self.generic_max_width = float(grasp.get("generic_max_width", 0.060))
        self.classify_shape = bool(self.cfg.get("classify_shape", False)) or self.generic_fallback
        self.want_path = bool(grasp.get("close_posture_by_shape", False)) or self.generic_fallback
        self._last_path = None
        # grasp.posture_apex_gate (default off): prism-top check features for "triangle" detections (apex_gate.py)
        self.want_apex = self.want_path and bool(cfg.get("grasp", {}).get("posture_apex_gate", False))
        self.triangle_pose = bool(self.cfg.get("triangle_pose", False))
        # #7 near-bin guard: abstain on shape when the top face may be cut by the bin exclusion (needs classify_shape)
        self.near_bin_guard = bool(self.cfg.get("near_bin_guard", False))
        # perception.y_affine [a, b] (23-grasp-failure §11b/§13): perceived - true y = a*y + b (pelvis) -> subtract it.
        # Applied after xy_offset in both the depth and the plane path; null = off. Do not combine with a self-
        # calibration that already absorbs the constant b without re-fitting.
        ya = self.cfg.get("y_affine")
        self.y_affine = None if ya is None else (float(ya[0]), float(ya[1]))
        # passive shape diagnostics: Detection.shape_diag + one log line per detection (default OFF, no action change)
        self.shape_diag = bool(self.cfg.get("shape_diag", False))
        # no-blob shape (Oliver f857054): no metric top face -> abstain instead of the 2D-blob ratio (default OFF)
        self.no_blob_shape = bool(self.cfg.get("no_blob_shape", False))
        # r17 shape_2d_v2 (default OFF): yellow 2D cube/cylinder from the blob top edge where the shape would be None / blob
        self.shape_2d_v2 = bool(self.cfg.get("shape_2d_v2", False))
        self.shape_2d_params = dict(self.cfg.get("shape_2d_v2_params") or {})
        # real colour (hf_perception 2026-11-14): one HSV set for sim + the real head cam; split touching objects
        hs = self.cfg.get("hsv_set", "stock")
        self.hsv = self.cfg["hsv_robust_v2"] if hs == "robust_v2" else self.cfg["hsv_robust"] if hs == "robust" else self.cfg["hsv"]
        self.split_blobs = bool(self.cfg.get("split_blobs", False))
        self.place_gate = bool((cfg.get("policy") or {}).get("place_gate", False))
        # perception.hue_guard (default OFF): abstain on orange/yellow hues near the red|yellow boundary (hue_guard.py)
        hg = self.cfg.get("hue_guard") or {}
        self.hue_guard = bool(hg.get("enabled", False))
        self.hue_guard_min_gap_deg = float(hg.get("min_gap_deg", 6.0))
        self.hue_guard_red_min_s = float(hg.get("red_min_s", 110.0))
        # Per-episode colour k-means always starts/falls back at the measured
        # robust ranges, regardless of the separately selectable stock set.
        self.colour_kmeans = bool(self.cfg.get("colour_kmeans", False))
        self._colour_kmeans_base = self.cfg.get("hsv_robust")
        if self.colour_kmeans:
            self.hsv = self._colour_kmeans_base
        self._colour_kmeans_done = False
        self._colour_kmeans_tries = 0
        self.colour_kmeans_info = None
        self.log = log if log is not None else print
        # perception.color_gate (FLAG, default OFF): post-filter at the end of detect(), see _color_gate
        self.color_gate = bool((self.cfg.get("color_gate") or {}).get("enabled", False))
        self.color_gate_cfg = dict(self.cfg.get("color_gate") or {})
        self.color_gate_degraded = False
        self.fk_mask = bool(self.color_gate_cfg.get("fk_mask", False))   # perception.color_gate.fk_mask (FLAG)
        self._arm_surface = None   # lazily built FK surface for fk_mask (own instance: never shares B1's lock)
        self.fk_mask_px = 0        # pixels the last fk mask removed (diagnostics)
        self.sat_floor = self.color_gate and bool(self.color_gate_cfg.get("sat_floor", False))   # color_gate.sat_floor (FLAG)
        self.live_bins = bool(self.color_gate_cfg.get("live_bins", False))   # perception.color_gate.live_bins (FLAG)
        self.bin_shift_src = None  # live_bins: object with .bin_shift {bin: pelvis dxy} (the policy's planner, bin.vote)
        self.geometry_runtime = False   # bin.geometry_runtime flag, set by policy.py: every bin consumer follows bin_shift_src.bin_shift
        self.bin_vote_on = False   # bin.vote flag, set by policy.py: bin footprints follow bin_shift_src.bin_shift

    def reset_calibration(self):
        """Restore the configured fallback at the start of every episode."""
        self.scene.counter_z_runtime = None
        self._ah_est, self._ah_tries = [], 0
        self.xy_offset = self.base_xy_offset.copy()
        self.calibration = None
        self.pitch_calibration = None
        self.pitch_T = None
        if self.colour_kmeans:
            self.hsv = self._colour_kmeans_base
            self._colour_kmeans_done = False
            self._colour_kmeans_tries = 0
            self.colour_kmeans_info = None
        self.cal = self.cal_nominal   # B1: a hand-calibrated copy never outlives its episode

    def _T(self):
        """Episode-corrected camera pose, or the untouched nominal calibration."""
        T = self.pitch_T
        return T if T is not None else self.cal._T(self.cam)

    def projection_snapshot(self):
        """Read-only T_pelvis_optical for all consumers of one copied frame."""
        T = self.current_T()
        T.setflags(write=False)
        return T

    def _pixel_to_pelvis(self, u, v, depth_mm, transform_snapshot=None):
        # Keep the disabled/fallback path byte-for-byte identical to Calibration.pixel_to_pelvis.
        if transform_snapshot is not None:
            T = np.asarray(transform_snapshot, float)
            p = self.cal.deproject(self.cam, u, v, depth_mm)
            return p @ T[:3, :3].T + T[:3, 3]
        T = self.pitch_T  # read once: a worker thread may swap it in
        if T is None:
            return self.cal.pixel_to_pelvis(self.cam, u, v, depth_mm)
        p = self.cal.deproject(self.cam, u, v, depth_mm)
        return p @ T[:3, :3].T + T[:3, 3]

    def _project(self, p_pelvis):
        # Keep the disabled/fallback path byte-for-byte identical to Calibration.project.
        T = self.pitch_T
        if T is None:
            return self.cal.project(self.cam, p_pelvis)
        p = (np.asarray(p_pelvis, float) - T[:3, 3]) @ T[:3, :3]
        K = self.cal.cams[self.cam].K
        z = p[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * p[..., 0] / z + K[0, 2]
            v = K[1, 1] * p[..., 1] / z + K[1, 2]
        return np.stack([u, v, z], axis=-1)

    @staticmethod
    def _optical_rx(deg):
        a = np.radians(float(deg))
        c, s = np.cos(a), np.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    @staticmethod
    def _optical_rz(deg):
        a = np.radians(float(deg))
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _pitch_mode(self):
        return "table" if self.online_pitch_cal == "table" else "bin_rim"

    def pitch_calibration_failure(self, reason, elapsed_ms=0.0):
        """Latch the nominal extrinsic after an unexpected optional-calibration error."""
        return self.commit_pitch(self.make_pitch_failure(reason, elapsed_ms))

    def make_pitch_failure(self, reason, elapsed_ms=0.0):
        """Pure: a failed result (nominal extrinsic) without touching any state."""
        return PitchCalibration(False, True, 0.0, 0, np.inf, float(elapsed_ms), str(reason), mode=self._pitch_mode())

    def commit_pitch(self, out):
        """Install a result: the extrinsic first, then the record (single attribute stores, atomic under the GIL)."""
        self.pitch_T = None if not out.success else np.asarray(out.T, float).copy()
        self.pitch_calibration = out
        return out

    def pitch_frame_valid(self, rgb, depth):
        """Cheap main-thread check that a frame is worth starting the (once-per-episode) fit on."""
        if depth is None or np.ndim(depth) != 2:
            return False
        if rgb is None or np.ndim(rgb) != 3 or rgb.shape[:2] != depth.shape or rgb.shape[2] < 3:
            return False
        c = self.cfg.get("table_pitch_calibration" if self.online_pitch_cal == "table" else "pitch_calibration") or {}
        dlo, dhi = self.cfg["depth_range_mm"]
        return float(((depth >= dlo) & (depth <= dhi)).mean()) >= float(c.get("min_valid_frac", self.cfg["min_valid_frac"]))

    def compute_pitch(self, rgb, depth):
        """Pure fit (no state written): safe to run in a worker thread. Result carries the extrinsic in ``.T``."""
        if self.online_pitch_cal == "table":
            return self._calibrate_table(rgb, depth)
        return self._calibrate_bin_rims(rgb, depth)

    def calibrate_pitch(self, rgb, depth):
        """Synchronous fit + install (tests / tools). The policy uses compute_pitch in a background thread."""
        return self.commit_pitch(self.compute_pitch(rgb, depth))

    def _calibrate_bin_rims(self, rgb, depth):
        """Fit ``T_nominal @ Rx(dtheta)`` from the orange bins' far top rim.

        The top-most orange pixel in each image column is a far-rim candidate. Rays through the dominant horizontal
        edge are intersected with the known rim plane, then a bounded 1-D search minimizes a trimmed RMS distance to
        the three nominal far-rim segments. The fit uses RGB geometry; valid depth is required only to avoid latching
        on an incomplete startup frame.
        """
        started = time.perf_counter()
        c = self.cfg.get("pitch_calibration") or {}

        def result(success, valid, pitch, n_points, residual, reason, T=None):
            return PitchCalibration(bool(success), bool(valid), float(pitch), int(n_points), float(residual),
                                    (time.perf_counter() - started) * 1000.0, str(reason), mode="bin_rim",
                                    n_inliers=int(n_points), T=T)

        if not self.online_pitch_cal:
            return result(False, True, 0.0, 0, np.inf, "disabled")
        if depth is None or np.ndim(depth) != 2:
            return result(False, False, 0.0, 0, np.inf, "no depth frame")
        dlo, dhi = self.cfg["depth_range_mm"]
        valid_depth = (depth >= dlo) & (depth <= dhi)
        if valid_depth.mean() < float(c.get("min_valid_frac", self.cfg["min_valid_frac"])):
            return result(False, False, 0.0, 0, np.inf, "insufficient valid depth")
        if rgb is None or np.ndim(rgb) != 3 or rgb.shape[:2] != depth.shape or rgb.shape[2] < 3:
            return result(False, False, 0.0, 0, np.inf, "invalid RGB frame")

        nominal = np.asarray(self.cal._T(self.cam), float)
        K = self.cal.cams[self.cam].K
        b = self.scene.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2.0
        rim_z = self.scene.counter_z + float(b["wall_height"])
        rims = []
        for center in b["centers"].values():
            center = np.asarray(center, float)
            # +mat-y is farther from the pelvis; mat-x spans each rim left-to-right.
            rims.append(self.scene.mat_to_pelvis(
                [[center[0] - half[0], center[1] + half[1], float(b["wall_height"])],
                 [center[0] + half[0], center[1] + half[1], float(b["wall_height"])]]))
        rims = np.asarray(rims, float)

        lo_deg, hi_deg = (float(x) for x in c.get("search_deg", [-12.0, 12.0]))
        coarse_step = float(c.get("coarse_step_deg", 0.25))
        coarse = np.arange(lo_deg, hi_deg + 0.5 * coarse_step, coarse_step)

        def pose(deg):
            T = nominal.copy()
            T[:3, :3] = nominal[:3, :3] @ self._optical_rx(deg)
            return T

        def project_with(T, points):
            q = (np.asarray(points, float) - T[:3, 3]) @ T[:3, :3]
            z = q[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = K[0, 0] * q[..., 0] / z + K[0, 2]
                v = K[1, 1] * q[..., 1] / z + K[1, 2]
            return np.stack([u, v, z], axis=-1)

        # Restrict orange-edge extraction to the image envelope swept by the possible far rims.
        height, width = depth.shape
        allowed_u = np.zeros(width, bool)
        bin_allowed_u = np.zeros((len(rims), width), bool)
        v_limits = []
        for deg in coarse:
            uvz = project_with(pose(deg), rims)
            for bin_i, seg in enumerate(uvz):
                if np.all(seg[:, 2] > 0):
                    ua, ub = sorted(seg[:, 0])
                    ia, ib = max(0, int(np.floor(ua)) - 2), min(width, int(np.ceil(ub)) + 3)
                    if ia < ib:
                        allowed_u[ia:ib] = True
                        bin_allowed_u[bin_i, ia:ib] = True
                    v_limits.extend(seg[:, 1].tolist())
        finite_v = np.asarray(v_limits, float)
        finite_v = finite_v[np.isfinite(finite_v)]
        if not allowed_u.any() or not len(finite_v):
            return result(False, True, 0.0, 0, np.inf, "nominal rims outside image")
        pad = int(c.get("roi_pad_px", 8))
        va = max(0, int(np.floor(finite_v.min())) - pad)
        vb = min(height, int(np.ceil(finite_v.max())) + pad + 1)

        pix = np.asarray(rgb[..., :3], float)
        r, g, blue = pix[..., 0], pix[..., 1], pix[..., 2]
        eps = 1.0
        orange = ((r >= float(c.get("orange_min_r", 55.0))) &
                  (r >= float(c.get("orange_rg_ratio", 1.15)) * (g + eps)) &
                  (g >= float(c.get("orange_gb_ratio", 1.10)) * (blue + eps)) &
                  (g / (r + eps) >= float(c.get("orange_g_over_r_min", 0.20))) &
                  (g / (r + eps) <= float(c.get("orange_g_over_r_max", 0.82))) &
                  (blue / (r + eps) <= float(c.get("orange_b_over_r_max", 0.60))))
        roi = np.zeros_like(orange)
        roi[va:vb, allowed_u] = True
        orange &= roi
        has = orange.any(axis=0)
        columns = np.flatnonzero(has)
        if len(columns):
            rows = np.argmax(orange[:, columns], axis=0)
            # Three coplanar far rims form the dominant near-horizontal top edge. Reject isolated orange objects.
            hist = np.bincount(rows, minlength=height)
            band = max(0, int(c.get("edge_band_px", 3)))
            smooth = np.convolve(hist, np.ones(2 * band + 1, int), mode="same")
            peak = int(np.argmax(smooth))
            edge = np.abs(rows - peak) <= band
            columns, rows = columns[edge], rows[edge]
        min_points = int(c.get("min_rim_points", 80))
        if len(columns) < min_points:
            return result(False, True, 0.0, len(columns), np.inf, "too few far-rim points")
        # A single orange object or a subset of the boxes can otherwise produce a deceptively good 1-D fit.
        # The known bin spans are disjoint in this view, so require edge support in every swept span before fitting.
        bin_support = np.sum(bin_allowed_u[:, columns], axis=1)
        min_bin_points = int(c.get("min_bin_rim_points", 20))
        if np.any(bin_support < min_bin_points):
            return result(False, True, 0.0, len(columns), np.inf,
                          f"not all bins found; support={bin_support.tolist()}")
        max_points = int(c.get("max_rim_points", 640))
        if len(columns) > max_points:
            take = np.linspace(0, len(columns) - 1, max_points).astype(int)
            columns, rows = columns[take], rows[take]

        rays_opt = np.stack([(columns - K[0, 2]) / K[0, 0],
                             (rows - K[1, 2]) / K[1, 1], np.ones(len(columns))], axis=1)
        seg_a, seg_b = rims[:, 0, :2], rims[:, 1, :2]
        seg_d = seg_b - seg_a
        seg_d2 = np.sum(seg_d * seg_d, axis=1)
        trim_fraction = float(np.clip(c.get("trim_fraction", 0.75), 0.1, 1.0))
        trim_n = max(min_points, int(np.ceil(trim_fraction * len(columns))))
        trim_n = min(trim_n, len(columns))

        def rim_distances(deg):
            T = pose(deg)
            rays = rays_opt @ T[:3, :3].T
            good = rays[:, 2] < -1e-8
            scale = np.full(len(rays), np.nan)
            scale[good] = (rim_z - T[2, 3]) / rays[good, 2]
            good &= np.isfinite(scale) & (scale > 0.0)
            if good.sum() < trim_n:
                return None
            points = T[:2, 3] + scale[good, None] * rays[good, :2]
            delta = points[:, None, :] - seg_a[None, :, :]
            along = np.clip(np.sum(delta * seg_d[None, :, :], axis=2) / seg_d2[None, :], 0.0, 1.0)
            closest = seg_a[None, :, :] + along[..., None] * seg_d[None, :, :]
            return np.sqrt(np.sum((points[:, None, :] - closest) ** 2, axis=2))

        def score(deg):
            distance_matrix = rim_distances(deg)
            if distance_matrix is None:
                return np.inf
            distances = np.min(distance_matrix, axis=1)
            kept = np.partition(distances, trim_n - 1)[:trim_n]
            return float(np.sqrt(np.mean(kept * kept)))

        scores = np.asarray([score(deg) for deg in coarse])
        if not np.isfinite(scores).any():
            return result(False, True, 0.0, len(columns), np.inf, "no forward rim intersections")
        estimate = float(coarse[int(np.nanargmin(scores))])
        fine_step = float(c.get("fine_step_deg", 0.02))
        fine = np.arange(max(lo_deg, estimate - coarse_step), min(hi_deg, estimate + coarse_step) +
                         0.5 * fine_step, fine_step)
        fine_scores = np.asarray([score(deg) for deg in fine])
        best = int(np.nanargmin(fine_scores))
        estimate, residual = float(fine[best]), float(fine_scores[best])
        max_residual = float(c.get("max_residual_m", 0.05))
        if not np.isfinite(residual) or residual > max_residual:
            return result(False, True, 0.0, len(columns), residual,
                          f"high residual; rejected {estimate:+.2f} deg")
        distance_matrix = rim_distances(estimate)
        nearest = np.argmin(distance_matrix, axis=1)
        nearest_distance = distance_matrix[np.arange(len(distance_matrix)), nearest]
        fit_support = np.bincount(nearest[nearest_distance <= max_residual], minlength=len(rims))
        if np.any(fit_support < min_bin_points):
            return result(False, True, 0.0, len(columns), residual,
                          f"not all bins fit; support={fit_support.tolist()}; rejected {estimate:+.2f} deg")
        max_abs_pitch = float(c.get("max_abs_pitch_deg", 20.0))
        if abs(estimate) > max_abs_pitch:
            return result(False, True, 0.0, len(columns), residual,
                          f"pitch outside guard; rejected {estimate:+.2f} deg")
        corrected = pose(estimate)
        return result(True, True, estimate, len(columns), residual,
                      f"accepted; support={fit_support.tolist()}", corrected)

    def _bin_footprint_mask(self, rgb, shape, c, pad):
        """uint8 mask of the bin boxes (footprint + margin, counter to rim) projected through the NOMINAL camera,
        plus orange pixels: the bins must not enter the counter-plane fit (they are not the plane)."""
        import cv2
        mask = np.zeros(shape, np.uint8)
        b = self.scene.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2.0 + float(c.get("bin_mask_margin_m", 0.02))
        wall = float(b["wall_height"])
        # the unknown pitch error shifts the projection: union hulls over a small sweep around nominal
        sweep = float(c.get("bin_mask_pitch_sweep_deg", 4.0))
        nominal = np.asarray(self.cal._T(self.cam), float)
        K = self.cal.cams[self.cam].K
        for deg in (0.0, -sweep, sweep):
            T = nominal.copy()
            T[:3, :3] = nominal[:3, :3] @ self._optical_rx(deg)
            for cx, cy in b["centers"].values():
                corners = np.array([[cx + sx * half[0], cy + sy * half[1], z]
                                    for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, wall)])
                q = (self.scene.mat_to_pelvis(corners) - T[:3, 3]) @ T[:3, :3]
                if np.any(q[:, 2] <= 1e-6):
                    continue
                uv = np.stack([K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2], K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]], axis=1)
                cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        orange = cv2.inRange(hsv, np.array([8, 100, 70], np.uint8), np.array([18, 255, 255], np.uint8))
        mask |= (orange > 0).astype(np.uint8)
        if pad:
            mask = cv2.dilate(mask, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
        return mask

    def _calibrate_table(self, rgb, depth):
        """Fit the counter plane without using the nominal positions of the bins.

        Candidate pixels are the central lower image (the side and bottom bands where the hands enter are omitted),
        with every configured coloured-object blob removed. RANSAC rejects remaining object/hand depth before a
        total-least-squares refit on counter inliers. The optical-frame plane determines pitch, roll, and camera
        height above the known counter plane.
        """
        started = time.perf_counter()
        c = self.cfg.get("table_pitch_calibration") or {}

        def result(success, valid, pitch, roll, camera_height, n_points, n_inliers, residual, reason, T=None):
            return PitchCalibration(bool(success), bool(valid), float(pitch), int(n_points), float(residual),
                                    (time.perf_counter() - started) * 1000.0, str(reason), mode="table",
                                    roll_deg=float(roll), camera_height_m=float(camera_height),
                                    plane_height_m=float(self.scene.counter_z) if success else np.nan,
                                    n_inliers=int(n_inliers), T=T)

        if self.online_pitch_cal != "table":
            return result(False, True, 0.0, 0.0, np.nan, 0, 0, np.inf, "disabled")
        if depth is None or np.ndim(depth) != 2:
            return result(False, False, 0.0, 0.0, np.nan, 0, 0, np.inf, "no depth frame")
        if rgb is None or np.ndim(rgb) != 3 or rgb.shape[:2] != depth.shape or rgb.shape[2] < 3:
            return result(False, False, 0.0, 0.0, np.nan, 0, 0, np.inf, "invalid RGB frame")

        dlo, dhi = self.cfg["depth_range_mm"]
        valid_depth = (depth >= dlo) & (depth <= dhi)
        if valid_depth.mean() < float(c.get("min_valid_frac", self.cfg["min_valid_frac"])):
            return result(False, False, 0.0, 0.0, np.nan, 0, 0, np.inf, "insufficient valid depth")

        height, width = depth.shape
        # This central lower crop is independent of scene/bin geometry.  Besides selecting the exposed mat/counter,
        # the side and bottom exclusions conservatively keep the robot's hands out of the candidate cloud.
        roi = np.zeros((height, width), bool)
        u0 = int(np.clip(float(c.get("roi_left_frac", 0.12)) * width, 0, width))
        u1 = int(np.clip(float(c.get("roi_right_frac", 0.88)) * width, 0, width))
        v0 = int(np.clip(float(c.get("roi_top_frac", 0.55)) * height, 0, height))
        v1 = int(np.clip(float(c.get("roi_bottom_frac", 0.92)) * height, 0, height))
        if u0 >= u1 or v0 >= v1:
            return result(False, True, 0.0, 0.0, np.nan, 0, 0, np.inf, "empty table ROI")
        roi[v0:v1, u0:u1] = True

        # Remove all known coloured blobs before subsampling.  Dilation also removes their mixed-depth edge pixels.
        coloured = np.zeros((height, width), np.uint8)
        for mask in self.masks(rgb).values():
            coloured |= mask
        pad = max(0, int(c.get("object_mask_pad_px", 5)))
        if pad:
            coloured = cv2.dilate(coloured, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
        stride = max(1, int(c.get("pixel_stride", 3)))
        min_inliers = int(c.get("min_inliers", 800))

        def sample(mask):
            v_, u_ = np.nonzero((valid_depth & roi & (mask == 0))[::stride, ::stride])
            return v_ * stride, u_ * stride

        vv, uu = sample(coloured | self._bin_footprint_mask(rgb, (height, width), c, pad))
        if len(uu) < min_inliers:
            # The footprint hulls are projected through the BELIEVED pose swept only +-4 deg; a larger error (-7.5 deg in the
            # field) slides them over the whole counter ROI. Fall back to the orange/colour mask alone: RANSAC rejects any
            # bin leakage and the inlier-ratio / RMS gates still apply.
            vv, uu = sample(coloured)
        n_points = len(uu)
        if n_points < min_inliers:
            return result(False, True, 0.0, 0.0, np.nan, n_points, 0, np.inf, "too few table points")
        # Evenly retain a fixed-size, image-wide sample. This bounds every RANSAC iteration independently of image
        # resolution and preserves the spatial distribution better than taking the first raster-order pixels.
        max_points = max(min_inliers, int(c.get("max_points", 6000)))
        if n_points > max_points:
            keep = np.linspace(0, n_points - 1, max_points).astype(int)
            vv, uu = vv[keep], uu[keep]
            n_points = len(uu)
        points = self.cal.deproject(self.cam, uu, vv, depth[vv, uu].astype(float))

        threshold = float(c.get("ransac_inlier_m", 0.008))
        rng = np.random.default_rng(int(c.get("ransac_seed", 0)))
        best = None
        for _ in range(max(1, int(c.get("ransac_iterations", 96)))):
            sample = points[rng.choice(n_points, 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            normal /= norm
            offset = -float(normal @ sample[0])
            distance = np.abs(points @ normal + offset)
            inliers = distance <= threshold
            count = int(inliers.sum())
            rms = float(np.sqrt(np.mean(distance[inliers] ** 2))) if count else np.inf
            if best is None or count > best[0] or (count == best[0] and rms < best[1]):
                best = (count, rms, inliers)
        if best is None or best[0] < min_inliers:
            count = 0 if best is None else best[0]
            return result(False, True, 0.0, 0.0, np.nan, n_points, count, np.inf,
                          f"too few plane inliers ({count} < {min_inliers})")

        # Total least squares, with one reclassification/refit pass to remove RANSAC sample bias.
        inliers = best[2]
        expected = np.asarray(self.cal._T(self.cam), float)[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        for _ in range(2):
            q = points[inliers]
            center = q.mean(axis=0)
            _, vectors = np.linalg.eigh((q - center).T @ (q - center))
            normal = vectors[:, 0]
            if normal @ expected < 0.0:
                normal = -normal
            offset = -float(normal @ center)
            distance = np.abs(points @ normal + offset)
            inliers = distance <= threshold
            if int(inliers.sum()) < min_inliers:
                break
        n_inliers = int(inliers.sum())
        if n_inliers < min_inliers:
            return result(False, True, 0.0, 0.0, np.nan, n_points, n_inliers, np.inf,
                          f"too few refined inliers ({n_inliers} < {min_inliers})")
        inlier_ratio = n_inliers / float(n_points)
        if inlier_ratio < float(c.get("min_inlier_ratio", 0.60)):
            return result(False, True, 0.0, 0.0, np.nan, n_points, n_inliers, np.inf,
                          f"low plane inlier ratio ({inlier_ratio:.2f})")
        residual = float(np.sqrt(np.mean((points[inliers] @ normal + offset) ** 2)))

        # Solve Q = Rz(droll) Rx(dpitch), Q n_measured = n_expected.  The small-magnitude pitch branch is unique
        # inside the safety gate; roll is then the signed angle between the two normals in optical xy.
        yz = float(np.hypot(normal[1], normal[2]))
        if yz < 1e-9:
            return result(False, True, 0.0, 0.0, np.nan, n_points, n_inliers, residual,
                          "degenerate plane normal")
        phi = float(np.arctan2(normal[1], normal[2]))
        alpha = float(np.arccos(np.clip(expected[2] / yz, -1.0, 1.0)))

        def wrap(angle):
            return (angle + np.pi) % (2.0 * np.pi) - np.pi

        pitch_rad = min((wrap(phi + alpha), wrap(phi - alpha)), key=abs)
        pitched = self._optical_rx(np.degrees(pitch_rad)) @ normal
        roll_rad = wrap(float(np.arctan2(expected[1], expected[0]) - np.arctan2(pitched[1], pitched[0])))
        pitch, roll = float(np.degrees(pitch_rad)), float(np.degrees(roll_rad))
        correction = self._optical_rz(roll) @ self._optical_rx(pitch)
        nominal = np.asarray(self.cal._T(self.cam), float)
        corrected = nominal.copy()
        corrected[:3, :3] = nominal[:3, :3] @ correction
        # With an upward unit normal, the fitted plane offset is the optical-origin distance to the counter. The
        # plane cannot observe camera x/y, and z is confounded with the (uncalibrated) counter height and depth scale,
        # so only pitch and roll are applied; the height delta is reported, never written to the extrinsic.
        camera_height = float(offset)
        nominal_height = float(nominal[2, 3]) - float(self.scene.counter_z)
        height_delta = camera_height - nominal_height

        max_rms = float(c.get("max_rms_m", 0.005))
        max_pitch = float(c.get("max_abs_pitch_deg", 15.0))
        max_roll = float(c.get("max_abs_roll_deg", 5.0))
        if not np.isfinite(residual) or residual > max_rms:
            return result(False, True, pitch, roll, camera_height, n_points, n_inliers, residual,
                          f"high plane RMS; height delta={height_delta * 1000.0:+.1f} mm (not applied)")
        if abs(pitch) > max_pitch:
            return result(False, True, pitch, roll, camera_height, n_points, n_inliers, residual,
                          f"pitch outside guard; height delta={height_delta * 1000.0:+.1f} mm (not applied)")
        if abs(roll) > max_roll:
            return result(False, True, pitch, roll, camera_height, n_points, n_inliers, residual,
                          f"roll outside guard; height delta={height_delta * 1000.0:+.1f} mm (not applied)")
        if not np.isfinite(camera_height) or camera_height <= 0.0:
            return result(False, True, pitch, roll, camera_height, n_points, n_inliers, residual,
                          "invalid camera height")
        return result(True, True, pitch, roll, camera_height, n_points, n_inliers, residual,
                      f"accepted; height delta={height_delta * 1000.0:+.1f} mm (not applied)", corrected)

    def current_T(self):
        """T_pelvis_optical perception uses right now. perception.online_pitch_cal (timi/pitch-cal) keeps its corrected
        pose in self.pitch_T (None = nominal); hand calibration starts from whatever is current, so it chains AFTER it."""
        T = getattr(self, "pitch_T", None)
        return np.array(T if T is not None else self.cal._T(self.cam), float)

    def set_camera_pose(self, T):
        """perception.hand_calibration: use T as this camera's T_pelvis_optical until the next reset, on a copy of the
        calibration (the shared object stays nominal). With online_pitch_cal merged, pitch_T (which wins in _T /
        _pixel_to_pelvis / _project) is set too, so every path sees the refined pose."""
        from .hand_calib import corrected_calibration
        self.cal = corrected_calibration(self.cal, self.cam, T)
        if hasattr(self, "pitch_T"):
            self.pitch_T = np.array(T, float)

    def calibration_failure(self, reason, elapsed_ms=0.0):
        """Latch a safe failed result after an unexpected calibration error."""
        self.xy_offset = self.base_xy_offset.copy()
        self.calibration = BinCalibration(False, True, 0.0, self.xy_offset.copy(), 0, np.inf,
                                          float(elapsed_ms), str(reason))
        return self.calibration

    def _counter_plane(self, depth, c, z0=None, transform_snapshot=None):
        """Pelvis points of `depth` (stride-subsampled) and the counter plane z = ax + by + c fitted to them.
        (P, coef, None) or (None, None, reason). Shared by calibrate_bins and the bin.vote rim vote."""
        dlo, dhi = self.cfg["depth_range_mm"]
        valid = (depth >= dlo) & (depth <= dhi)
        if valid.mean() < float(c.get("min_valid_frac", self.cfg["min_valid_frac"])):
            return (None, None, "insufficient valid depth")

        stride = max(1, int(c.get("pixel_stride", 3)))
        vv, uu = np.nonzero(valid[::stride, ::stride])
        vv, uu = vv * stride, uu * stride
        P = (self._pixel_to_pelvis(uu, vv, depth[vv, uu].astype(float)) if transform_snapshot is None else
             self._pixel_to_pelvis(uu, vv, depth[vv, uu].astype(float), transform_snapshot))
        sc = self.scene

        # Fit z = ax + by + c to counter points. Bins and objects are positive outliers and are trimmed iteratively.
        pm = sc.pelvis_to_mat(P)
        mat = np.asarray(sc.cfg["counter"]["mat_size"], float)
        counter = ((pm[:, 0] >= -0.03) & (pm[:, 0] <= mat[0] + 0.03) &
                   (pm[:, 1] >= -0.03) & (pm[:, 1] <= mat[1] + 0.03) &
                   (np.abs(P[:, 2] - (sc.counter_z if z0 is None else z0)) < float(c.get("plane_z_window", 0.06))))
        plane_points = P[counter]
        if len(plane_points) < int(c.get("min_plane_points", 500)):
            return (None, None, "counter plane not visible")
        A = np.c_[plane_points[:, :2], np.ones(len(plane_points))]
        keep = np.ones(len(A), bool)
        coef = np.array([0.0, 0.0, sc.counter_z if z0 is None else z0])
        plane_tol = float(c.get("plane_inlier_m", 0.006))
        for _ in range(3):
            if keep.sum() < int(c.get("min_plane_points", 500)):
                break
            coef = np.linalg.lstsq(A[keep], plane_points[keep, 2], rcond=None)[0]
            residual = plane_points[:, 2] - A @ coef
            center = np.median(residual)
            keep = np.abs(residual - center) <= plane_tol
        if keep.sum() < int(c.get("min_plane_points", 500)):
            return (None, None, "counter plane fit rejected")
        return P, coef, None

    def _plane_height(self, depth, a):
        """(z at the mat centre, tilt deg, n inliers, None) from the existing counter-plane fit, or (nan, tilt, n, why)."""
        nan = float("nan")
        if depth is None or np.ndim(depth) != 2 or np.size(depth) == 0:
            return nan, nan, 0, "no depth"
        depth = np.nan_to_num(np.asarray(depth, float), nan=0.0, posinf=0.0, neginf=0.0)
        nom, sc = self.scene.nominal_counter_z, self.scene
        c = {"plane_z_window": float(a.get("window_m", 0.40)), "min_plane_points": int(a.get("min_inliers", 500))}
        xy = sc.mat_to_pelvis([*(np.asarray(sc.cfg["counter"]["mat_size"], float) / 2), 0.0])[:2]
        z = nom
        for _ in range(2):          # pass 1 coarse (wide window), pass 2 tight around pass 1's plane
            P, coef, why = self._counter_plane(depth, c, z0=z)
            if why:
                return nan, nan, 0, why
            z = float(coef @ np.r_[xy, 1.0])
            c["plane_z_window"] = 0.03
        tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
        n = int((np.abs(P[:, 2] - np.c_[P[:, :2], np.ones(len(P))] @ coef) <= 0.006).sum())
        if n < int(a.get("min_inliers", 500)):
            return nan, tilt, n, f"few inliers {n}"
        if tilt > float(a.get("max_tilt_deg", 8.0)):
            return z, tilt, n, f"tilt {tilt:.1f} deg"
        return z, tilt, n, None

    def update_counter_height(self, depth):
        """calib.plane.apply_height: one look -> scene.counter_z_runtime = median of the first N valid estimates, clamped.
        Inert (no state touched) when the flag is off."""
        a = self.ah
        n_looks = int(a.get("looks", 3))
        if not a.get("apply_height") or len(self._ah_est) >= n_looks or self._ah_tries >= 3 * n_looks:
            return
        self._ah_tries += 1
        nom = self.scene.nominal_counter_z
        try:
            z, tilt, n, why = self._plane_height(depth, a)
        except Exception as exc:
            z, tilt, n, why = float("nan"), float("nan"), 0, f"{type(exc).__name__}"
        if why is None and np.isfinite(z):
            self._ah_est.append(z)
        est = float(np.median(self._ah_est)) if self._ah_est else float("nan")
        lim = float(a.get("max_dz_m", 0.20))
        applied = float(np.clip(est, nom - lim, nom + lim)) if self._ah_est else nom
        self.scene.counter_z_runtime = applied if self._ah_est else None
        self.log(f"[apply_height] est={est:.4f} applied={applied:.4f} n={len(self._ah_est)} tilt_deg={tilt:.2f} "
                 f"fallback={why or ('clamped' if applied != est else 'none')}")

    def calibrate_bins(self, depth):
        """Estimate a single XY correction by aligning elevated depth points to the fixed bin walls.

        A plane is fitted to the counter first, so camera z translation and small roll/pitch errors do not turn the
        counter into false wall points. Only a bounded translation is optimized: pose translation and rotation are
        intentionally not separated because they are nearly degenerate at this working distance.
        """
        started = time.perf_counter()
        c = self.cfg.get("self_calibration") or {}

        def result(success, valid, confidence, offset, n_points, residual, reason):
            out = BinCalibration(success, valid, float(confidence), np.asarray(offset, float), int(n_points),
                                 float(residual), (time.perf_counter() - started) * 1000.0, reason)
            self.calibration = out
            return out

        if not c.get("enabled", False):
            return result(False, True, 0.0, self.xy_offset.copy(), 0, np.inf, "disabled")
        if depth is None or np.ndim(depth) != 2:
            return result(False, False, 0.0, self.xy_offset.copy(), 0, np.inf, "no depth frame")
        P, coef, why = self._counter_plane(depth, c)
        if why == "insufficient valid depth":
            return result(False, False, 0.0, self.xy_offset.copy(), 0, np.inf, why)
        if why:
            return result(False, True, 0.0, self.xy_offset.copy(), 0, np.inf, why)
        sc = self.scene

        height = P[:, 2] - (np.c_[P[:, :2], np.ones(len(P))] @ coef)
        b = sc.cfg["bins"]
        gate = float(c.get("max_offset_m", 0.03))
        half_mat = np.asarray(b["outer_size"], float) / 2
        around = np.zeros(len(P), bool)
        centers = []
        for ctr_mat in b["centers"].values():
            ctr = sc.mat_to_pelvis([ctr_mat[0], ctr_mat[1], 0.0])[:2]
            centers.append(ctr)
            around |= np.all(np.abs(P[:, :2] - ctr) <= half_mat[::-1] + gate + 0.01, axis=1)
        wall = (around & (height >= float(c.get("min_wall_height_m", 0.008))) &
                (height <= float(c.get("max_wall_height_m", float(b["wall_height"]) + 0.015))))
        Q = P[wall, :2]
        max_points = int(c.get("max_points", 6000))
        if len(Q) > max_points:
            Q = Q[np.linspace(0, len(Q) - 1, max_points).astype(int)]
        min_points = int(c.get("min_wall_points", 180))
        if len(Q) < min_points:
            return result(False, True, 0.0, self.xy_offset.copy(), len(Q), np.inf, "too few bin wall points")

        # Rasterized distance-to-edge field makes the bounded search deterministic and fast on CPU.
        resolution = float(c.get("grid_resolution_m", 0.001))
        margin = gate + 0.02
        outer_half = half_mat[::-1]
        inner_half = outer_half - float(b["wall_thickness"])
        lo = np.min(np.asarray(centers) - outer_half - margin, axis=0)
        hi = np.max(np.asarray(centers) + outer_half + margin, axis=0)
        shape = np.ceil((hi - lo) / resolution).astype(int) + 1
        import cv2
        edge_image = np.full((shape[1], shape[0]), 255, np.uint8)

        def pixel(xy):
            ij = np.round((np.asarray(xy) - lo) / resolution).astype(int)
            return int(ij[0]), int(ij[1])

        for ctr in centers:
            for half in (outer_half, inner_half):
                x0, y0 = pixel(ctr - half)
                x1, y1 = pixel(ctr + half)
                cv2.rectangle(edge_image, (x0, y0), (x1, y1), 0, 1)
        distance = cv2.distanceTransform(edge_image, cv2.DIST_L2, 5) * resolution
        truncate = float(c.get("score_truncate_m", 0.012))

        def distances(offset):
            ij = np.round((Q + offset - lo) / resolution).astype(int)
            inside = ((ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) &
                      (ij[:, 1] >= 0) & (ij[:, 1] < shape[1]))
            d = np.full(len(Q), truncate)
            d[inside] = distance[ij[inside, 1], ij[inside, 0]]
            return d

        def search(center, span, step):
            values = np.arange(-span, span + 0.5 * step, step)
            best_score, best = np.inf, None
            for dx in values:
                for dy in values:
                    off = center + [dx, dy]
                    if np.linalg.norm(off) > gate:
                        continue
                    d = np.minimum(distances(off), truncate)
                    score = float(np.mean(d * d))
                    if score < best_score:
                        best_score, best = score, off.copy()
            return best

        coarse_step = float(c.get("coarse_step_m", 0.003))
        estimate = search(np.zeros(2), gate, coarse_step)
        if estimate is None:
            return result(False, True, 0.0, self.xy_offset.copy(), len(Q), np.inf, "search outside gate")
        estimate = search(estimate, coarse_step, float(c.get("fine_step_m", 0.0005)))
        if estimate is None or np.linalg.norm(estimate) > gate:
            return result(False, True, 0.0, self.xy_offset.copy(), len(Q), np.inf, "offset outside gate")

        d = distances(estimate)
        inlier_m = float(c.get("inlier_m", 0.006))
        inlier_ratio = float(np.mean(d <= inlier_m))
        residual = float(np.median(d))

        # Require constraints normal to both wall directions; one visible line cannot determine a 2-D shift safely.
        def axis_support(axis):
            normal, tangent = axis, 1 - axis
            supported = np.zeros(len(Q), bool)
            for ctr in centers:
                for half in (outer_half, inner_half):
                    for sign in (-1, 1):
                        edge = ctr[normal] + sign * half[normal]
                        normal_ok = np.abs(Q[:, normal] + estimate[normal] - edge) <= inlier_m
                        tangent_ok = np.abs(Q[:, tangent] + estimate[tangent] - ctr[tangent]) <= half[tangent] + inlier_m
                        supported |= normal_ok & tangent_ok
            return int(supported.sum())

        support_x, support_y = axis_support(0), axis_support(1)
        support_min = int(c.get("min_axis_points", 35))
        point_factor = min(1.0, len(Q) / max(2.0 * min_points, 1.0))
        residual_factor = max(0.0, 1.0 - residual / max(inlier_m, 1e-9))
        axis_factor = min(1.0, min(support_x, support_y) / max(2.0 * support_min, 1.0))
        confidence = float(np.clip(inlier_ratio * point_factor * residual_factor * axis_factor, 0.0, 1.0))
        confidence_min = float(c.get("min_confidence", 0.30))
        if support_x < support_min or support_y < support_min:
            return result(False, True, confidence, self.xy_offset.copy(), len(Q), residual,
                          f"weak axis support x={support_x} y={support_y}; rejected estimate "
                          f"({1000 * estimate[0]:+.1f},{1000 * estimate[1]:+.1f}) mm")
        if inlier_ratio < float(c.get("min_inlier_ratio", 0.35)) or confidence < confidence_min:
            return result(False, True, confidence, self.xy_offset.copy(), len(Q), residual,
                          f"low confidence (inliers={inlier_ratio:.2f}); rejected estimate "
                          f"({1000 * estimate[0]:+.1f},{1000 * estimate[1]:+.1f}) mm")

        self.xy_offset = estimate.copy()
        return result(True, True, confidence, estimate, len(Q), residual,
                      f"accepted; support x={support_x} y={support_y}, inliers={inlier_ratio:.2f}")

    def masks(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        return self._masks_hsv(hsv)

    def _masks_hsv(self, hsv, ranges=None):
        import cv2
        out = {}
        for color, color_ranges in (ranges or self.hsv).items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in color_ranges:
                m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            k = int(self.cfg["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            out[color] = m
        return out

    def _fit_colour_kmeans(self, hsv, keep):
        """One bounded k=3 attempt on robust object-blob pixels per episode."""
        import cv2
        from .colour_kmeans import fit_hue_ranges

        self._colour_kmeans_tries += 1
        pixels = []
        try:
            for m in self._masks_hsv(hsv, self._colour_kmeans_base).values():
                m = m & keep
                n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                for i in range(1, n):
                    if int(stats[i, cv2.CC_STAT_AREA]) >= int(self.cfg["min_area_px"]):
                        pixels.append(hsv[lab == i])
            px = np.concatenate(pixels, axis=0) if pixels else np.empty((0, 3), np.uint8)
            self.hsv, self.colour_kmeans_info = fit_hue_ranges(px, self._colour_kmeans_base)
            info = self.colour_kmeans_info
            if info["status"] == "adapted":
                self._colour_kmeans_done = True
            self.log(f"[colour_kmeans] {info['status']} reason={info['reason']} "
                     f"centres={info['centres']} sizes={info['sizes']} ranges={info['ranges']}")
        except Exception as exc:
            self.hsv = self._colour_kmeans_base
            self.colour_kmeans_info = {"status": "fallback", "reason": f"{type(exc).__name__}: {exc}",
                                       "centres": [], "sizes": [], "purities": [], "ranges": {}}
            self.log(f"[colour_kmeans] fallback: {type(exc).__name__}: {exc}")
        if self._colour_kmeans_tries >= 3:
            self._colour_kmeans_done = True  # give up after the 3rd attempt; robust ranges stay

    def detect(self, rgb, depth, debug=False, include_bins=False, arm_clear=True, robot_q=None):
        """Detect coloured objects.

        ``include_bins`` keeps pixels over the known bin footprints long enough for
        :meth:`_reject` to label them ``in_bin:<name>``.  They are still returned
        only through the rejected half of the ``debug=True`` result, so existing
        callers cannot accidentally pick an already sorted object.  ``arm_clear``: the
        policy says no arm is in view (looks happen at home); colour k-means only fits then.
        """
        import cv2
        c = self.cfg
        dets, rejected = [], []
        self.color_gate_degraded = False
        self.update_counter_height(depth)    # calib.plane.apply_height (inert when off)
        keep = self._keep_mask(depth, exclude_bins=not include_bins)
        if self.fk_mask and robot_q is not None:   # the robot's own arms / hands are never objects
            arm = self._fk_arm_mask(keep.shape, *robot_q)
            if arm is not None:
                keep[arm] = 0
        # Preserve the old path exactly when both new consumers are off.  When
        # either is on, share one RGB->HSV conversion.
        need_hsv = self.place_gate or self.colour_kmeans or self.hue_guard or self.sat_floor
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV) if need_hsv else None
        if self.colour_kmeans and not self._colour_kmeans_done and arm_clear:
            kkeep = self._keep_mask(depth, exclude_bins=True) if include_bins else keep  # never fit on bin pixels
            self._fit_colour_kmeans(hsv, kkeep)
        masks = self._masks_hsv(hsv) if hsv is not None else self.masks(rgb)
        items = self._hue_guard_items(masks, hsv) if self.hue_guard else masks.items()
        for color, m in items:
            m = m & keep  # drop bin / counter pixels BEFORE grouping, so an object touching a same-colour bin survives
            k = int(c["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < c["min_area_px"]:
                    continue
                blob = (lab == i).astype(np.uint8)
                if self.split_blobs:
                    for part, p_area, p_cent in self.split_blob(blob, area, cent[i], depth):
                        self._detect_blob(color, part, p_area, p_cent, depth, include_bins, dets, rejected,
                                          hsv if (self.place_gate or self.hue_guard or self.sat_floor) else None)
                    continue
                self._detect_blob(color, blob, area, cent[i], depth, include_bins, dets, rejected,
                                  hsv if (self.place_gate or self.hue_guard or self.sat_floor) else None)
        if self.hue_guard:
            dets, rejected = self._hue_guard(dets, rejected)
        if self.color_gate:
            dets = self._color_gate(dets, rejected)
        return (dets, rejected) if debug else dets

    def _hue_guard_items(self, masks, hsv):
        """perception.hue_guard layer 3: red and yellow pixels form one component per object, so an object whose side
        faces fall in the red range and whose top is yellow is a single candidate. Its colour is the class of the top
        face hue (brightest ~35 % by V, re-classified on the active set); a hue in the band keeps the majority colour so
        layer 1 can abstain on it. Blue and the other colours pass through unchanged."""
        import cv2
        ry = (masks["red"] | masks["yellow"]).astype(np.uint8)
        n, lab, _, _ = cv2.connectedComponentsWithStats(ry, connectivity=8)
        items = []
        for i in range(1, n):
            comp = lab == i
            top = top_face_hsv(hsv[comp])
            label = classify_hue(top[0], self.hsv)
            if label is None:
                label = "red" if (masks["red"][comp] > 0).sum() >= (masks["yellow"][comp] > 0).sum() else "yellow"
            items.append((label, comp.astype(np.uint8) * 255))
        return items + [(c, m) for c, m in masks.items() if c not in ("red", "yellow")]

    def _hue_guard(self, dets, rejected):
        """perception.hue_guard: abstain (or resolve by frame evidence) on orange/yellow hues near the boundary."""
        idx = [i for i, d in enumerate(dets) if d.hsv_med is not None]
        cands = [Cand(dets[i].color, *dets[i].hsv_med) for i in idx]
        try:
            verdict = judge(cands, self.hsv, self.hue_guard_min_gap_deg, self.hue_guard_red_min_s)
        except Exception as exc:   # fail closed: nothing hue-judged is placed
            verdict = [(None, f"error:{type(exc).__name__}")] * len(idx)
            self.log(f"[hue_guard] error {type(exc).__name__}: {exc}")
        by_idx = dict(zip(idx, verdict))
        keep, moved, n_amb, n_relabel, reasons = [], [], 0, 0, {}
        for i, d in enumerate(dets):
            if i not in by_idx:
                keep.append(d)
                continue
            label, reason = by_idx[i]
            if reason is None:
                if label != d.color:
                    n_relabel += 1
                    d = replace(d, color=label)
                keep.append(d)
                continue
            n_amb += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            amb = replace(d, color_conf=AMBIGUOUS_CONF)
            if self.place_gate:   # the place gate (colour_min > 0.3) keeps it off the bins; it stays on the table
                keep.append(amb)
            else:
                moved.append((amb, "hue_guard:low_s_red" if reason == "low_s_red" else "hue_guard:ambiguous"))
        self.log(f"[hue_guard] cands={len(cands)} ambiguous={n_amb} relabelled={n_relabel} "
                 f"moved_rejected={len(moved)} reasons={reasons}")
        return keep, rejected + moved

    def _detect_blob(self, color, blob, area, centroid, depth, include_bins, dets, rejected, hsv=None):
        """One colour blob -> Detection (depth top face, else plane) -> dets or rejected (the stock per-blob body)."""
        import cv2
        c = self.cfg
        dlo, dhi = c["depth_range_mm"]
        e = int(c["erode_px"])
        core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
        if core.sum() < 0.25 * area:
            core = blob
        v, u = np.nonzero(core)
        d = depth[v, u].astype(float) if depth is not None else np.zeros(len(u))
        ok = (d >= dlo) & (d <= dhi)
        det = None
        if ok.sum() >= c["min_valid_depth_px"]:
            det = self._from_depth(color, u[ok], v[ok], d[ok], area, blob, depth)
        if det is None:
            det = self._from_plane(color, centroid, area, int(ok.sum()))
            if det is not None:
                det.shape, conf, path = self._shape_conf(None, blob)
                if self.want_path:
                    det.shape_conf, det.shape_path = conf, self._last_path
                if self.generic_fallback:
                    self._apply_generic_fallback(det, conf, blob, depth)
                if self.shape_diag:
                    det.shape_diag = self._diag(det, "plane", None, blob, conf, path, None)
        if det is not None and self.shape_2d_v2:
            if color == "yellow":
                self._apply_shape2d(det, blob)
            elif color == "orange":
                self._log_orange_candidate(blob)
        if det is not None and hsv is not None and self.hue_guard:
            det.hsv_med = top_face_hsv(hsv[blob > 0])   # layer 3: colour from the top face, not the whole blob
        if det is not None and hsv is not None and self.sat_floor:
            det.sat_med = float(np.median(hsv[blob > 0][:, 1]))   # color_gate.sat_floor: shadow / sheen has low S
        if det is not None and hsv is not None and self.place_gate:
            try:
                det.color_conf = colour_confidence(hsv, blob, self.hsv, color, c["min_area_px"])
            except Exception as exc:  # never raise into act()
                det.color_conf = 0.0
                self.log(f"[gate] colour_confidence failed for {color}: {type(exc).__name__}: {exc}")
        why = self._reject(det, bins_first=include_bins)
        if self.shape_diag and det is not None and det.shape_diag is not None:
            self._log_diag(det, why)
        if why:
            rejected.append((det, why))
        else:
            dets.append(det)

    def expected_area_px(self, blob, centroid, depth):
        """Pixel area of ONE object (split.object_size) at this blob's distance: area_coef * (fx * size / Z)^2.
        Z = median valid depth of the blob, else the optical depth of the centroid ray at the object-top plane,
        else None -> split.fallback_area_px."""
        s = self.cfg.get("split") or {}
        size, coef = float(s.get("object_size") or self.size), float(s.get("area_coef", 1.2))   # null -> scene objects.size
        fallback = float(s.get("fallback_area_px", 3000))
        if self.cal is None or self.cam not in getattr(self.cal, "cams", {}):
            return fallback
        K = self.cal.cams[self.cam].K
        z = None
        if depth is not None:
            dlo, dhi = self.cfg["depth_range_mm"]
            d = depth[blob > 0].astype(float)
            d = d[(d >= dlo) & (d <= dhi)]
            if len(d) >= int(self.cfg["min_valid_depth_px"]):
                z = float(np.median(d)) / (1.0 / float(getattr(self.cal.cams[self.cam], "depth_scale", 0.001)))   # mm -> m via calibration depth_scale (0.001 -> /1000.0 exactly)
        if z is None:
            try:
                T = self._T()
                ray = T[:3, :3] @ np.array([(centroid[0] - K[0, 2]) / K[0, 0], (centroid[1] - K[1, 2]) / K[1, 1], 1.0])
                if ray[2] < -1e-6:
                    z = float((self.scene.counter_z + self.height - T[2, 3]) / ray[2])   # optical z (ray_opt z = 1)
            except Exception:
                z = None
        if z is None or not np.isfinite(z) or z <= 0:
            return fallback
        return coef * (K[0, 0] * size / z) ** 2

    def split_blob(self, blob, area, centroid, depth):
        """[(mask, area, centroid)]: the blob itself, or one part per touching object. Only blobs larger than
        split.area_factor x the expected single-object area are tried. Seeds = connected components of the distance
        transform above peak_frac * max (tried in order); every blob pixel goes to its nearest seed; accepted only if
        every part has >= min_part_frac x the expected area. No such neck but an elongated blob (aspect >= min_aspect,
        e.g. two cubes side by side): k = round(area / expected) equal-area slices along the major axis. Otherwise
        the blob is returned unchanged."""
        import cv2
        s = self.cfg.get("split") or {}
        whole = [(blob, area, centroid)]
        exp = self.expected_area_px(blob, centroid, depth)
        if area <= float(s.get("area_factor", 1.5)) * exp:
            return whole
        x, y, w, h = cv2.boundingRect(blob)
        crop = np.ascontiguousarray(blob[y:y + h, x:x + w])
        dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
        dmax = float(dist.max())
        min_part = float(s.get("min_part_frac", 0.3)) * exp
        for frac in s.get("peak_frac", [0.5, 0.6, 0.7, 0.8]):
            n, lab = cv2.connectedComponents((dist >= frac * dmax).astype(np.uint8), connectivity=8)
            if n < 3:
                continue
            # at most as many seeds as objects the blob can hold (strongest peaks), decided BEFORE any per-seed work
            k_max = min(int(s.get("max_parts", 3)), max(1, int(round(area / exp))))
            if k_max < 2:
                continue
            if n - 1 > k_max:
                strength = np.zeros(n)
                np.maximum.at(strength, lab.ravel(), dist.ravel())
                keep = np.argsort(-strength[1:], kind="stable")[:k_max] + 1
                lab = np.where(np.isin(lab, keep), lab, 0)
            # single pass: every pixel -> its nearest seed (labels of the zero components of the seed-free image)
            _, owner = cv2.distanceTransformWithLabels((lab == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_CCOMP)
            ids = [int(v) for v in np.unique(owner[lab > 0])]
            parts = [((owner == j) & (crop > 0)) for j in ids]
            if len(parts) < 2 or min(int(p.sum()) for p in parts) < min_part:
                continue
            return self._parts_out(blob, parts, x, y, w, h)
        # No neck (e.g. two cubes side by side = one rectangle): an elongated blob is cut along its major axis into
        # k = round(area / expected) equal-area slices.
        vv, uu = np.nonzero(crop)
        pts = np.c_[uu, vv].astype(float)
        evals, evecs = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
        if evals[0] <= 0 or np.sqrt(evals[1] / evals[0]) < float(s.get("min_aspect", 1.4)):
            return whole
        k = int(np.clip(round(area / exp), 2, int(s.get("max_parts", 3))))
        proj = (pts - pts.mean(0)) @ evecs[:, 1]
        edges = np.quantile(proj, np.linspace(0, 1, k + 1)[1:-1])
        idx = np.searchsorted(edges, proj)
        parts = []
        for j in range(k):
            p = np.zeros_like(crop, bool)
            p[vv[idx == j], uu[idx == j]] = True
            parts.append(p)
        if min(int(p.sum()) for p in parts) < min_part:
            return whole
        return self._parts_out(blob, parts, x, y, w, h)

    @staticmethod
    def _parts_out(blob, parts, x, y, w, h):
        out = []
        for p in parts:
            full = np.zeros_like(blob)
            full[y:y + h, x:x + w] = p
            vv, uu = np.nonzero(p)
            out.append((full, int(p.sum()), np.array([x + uu.mean(), y + vv.mean()])))
        return out

    def bin_geometry(self, gate=False):
        """Runtime bin geometry (bin_vote.bin_geometry). Shift source = the policy's bin_shift (bin.vote), taken when
        bin.geometry_runtime is on; flag off: only under bin.vote (gate=False) / color_gate.live_bins (gate=True) as before."""
        on = self.geometry_runtime or (self.live_bins if gate else self.bin_vote_on)
        shift = dict(getattr(self.bin_shift_src, "bin_shift", None) or {}) if on else {}
        return bin_geometry(self.scene, shift)

    def _bin_centres_mat(self):
        """{bin: mat-frame centre}: nominal scene.yaml centres + the runtime shift (see bin_geometry)."""
        return {n: g["centre_mat"] for n, g in self.bin_geometry().items()}

    def _keep_mask(self, depth, exclude_bins=True):
        """uint8 mask of pixels that may belong to an object: no valid depth (unknown), or a 3D point at object
        height over the mat and outside every bin footprint."""
        c, sc = self.cfg, self.scene
        shape = depth.shape if depth is not None else (480, 640)
        keep = self._bin_hulls_mask(shape) if exclude_bins else np.full(shape, 255, np.uint8)
        if depth is None:
            return keep
        dlo, dhi = c["depth_range_mm"]
        d = depth.astype(float)
        valid = (d >= dlo) & (d <= dhi)
        if valid.mean() < c["min_valid_frac"]:
            return keep  # (almost) no depth stream: only the projected bin hulls can be removed
        keep[:] = 255    # with depth: holes stay "unknown" (kept); valid pixels are judged in 3D
        vv, uu = np.nonzero(valid)
        P = self._pixel_to_pelvis(uu, vv, d[vv, uu])
        h = (P[:, 2] - sc.counter_z) / self.height
        pm = sc.pelvis_to_mat(P)
        # Verification must also see a same-colour object stacked above the
        # 6 cm rim.  _reject() will discard elevated blobs outside bin
        # footprints, while bin blobs remain labelled ``in_bin:*``.
        upper = c["height_range"][1] if exclude_bins else max(c["height_range"][1], 4.0)
        ok = (h >= 0.25) & (h <= upper)
        if exclude_bins:
            half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
            for ctr in self._bin_centres_mat().values():
                ok &= ~np.all(np.abs(pm[:, :2] - ctr) <= half, axis=1)
        keep[vv[~ok], uu[~ok]] = 0
        return keep

    def _bin_hulls_mask(self, shape):
        """255 everywhere except inside the image hull of each bin box (footprint + margin, counter to rim)."""
        import cv2
        sc = self.scene
        keep = np.full(shape, 255, np.uint8)
        b = sc.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + self.cfg["bin_pixel_margin"]
        for cx, cy in self._bin_centres_mat().values():
            corners = np.array([[cx + sx * half[0], cy + sy * half[1], z] for sx in (-1, 1) for sy in (-1, 1)
                                for z in (0.0, float(b["wall_height"]))])
            uvz = self._project(sc.mat_to_pelvis(corners))
            if np.any(uvz[:, 2] <= 0):
                continue
            hull = cv2.convexHull(np.round(uvz[:, :2]).astype(np.int32))
            cv2.fillConvexPoly(keep, hull, 0)
        return keep

    def _near_bin(self, top):
        """True if any top-face point (pelvis frame) lies within a bin footprint + bin_pixel_margin + object size.
        Same 3D test (mat frame) as the bin exclusion in the keep mask: there the top face may have been cut by a
        straight line in xy, which turns a cube into a 'cylinder'; such a face must not be classified."""
        sc = self.scene
        pm = sc.pelvis_to_mat(np.asarray(top, float))
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + self.cfg["bin_pixel_margin"] + self.size
        return any(np.any(np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1))
                   for ctr in self._bin_centres_mat().values())

    def _y_affine(self, xy):
        """y_corrected = y - (a*y + b): the measured perceived-minus-true error model in the pelvis frame."""
        a, b = self.y_affine
        out = np.array(xy, float)
        out[1] = out[1] - (a * out[1] + b)
        return out

    def _shape_of(self, top_xy, blob, near_bin=False):
        """Shape label, or None (flags off / abstain). Delegates to `classify_top_face` and decides nothing itself;
        `near_bin` (top face possibly cut by the bin exclusion) only forces an abstain."""
        return self._shape_conf(top_xy, blob, near_bin)[0]

    def _shape_conf(self, top_xy, blob, near_bin=False):
        """(label, conf, path): path is "off" / "abstain_near_bin" when decided here, else None (= the seam's)."""
        self._last_path = None
        if not (self.classify_shape or self.triangle_pose):
            self._last_path = "off"
            return None, 0.0, "off"
        if near_bin:
            self._last_path = "abstain_near_bin"
            return None, 0.0, "abstain_near_bin"
        shape = None if blob is None else blob.shape
        if self.want_path or self.shape_2d_v2:
            label, conf, info = classify_top_face_diag(top_xy, blob, shape, use_blob=not self.no_blob_shape)
            self._last_path = info.get("path")
        else:
            label, conf = classify_top_face(top_xy, blob, shape, use_blob=not self.no_blob_shape)
        return label, float(conf), None

    def _diag(self, det, source, top_xy, blob, conf, path, near_bin):
        """Passive shape diagnostics (perception.shape_diag). Reads only; never changes det's fields."""
        try:
            _, _, info = classify_top_face_diag(top_xy, blob, None if blob is None else blob.shape,
                                                use_blob=not self.no_blob_shape)
        except Exception as e:   # diagnostics must never break detection
            info = {"path": f"diag_error:{e!r}", "n_top": 0, "ratio_cloud": None, "ratio_blob": None}
        if path is None:          # the seam decided: report the default classifier's path
            path = info["path"]
            if source == "plane" and path == "abstain_few_points":
                path = "abstain_plane"
        top_list = None
        if top_xy is not None:
            top_list = np.asarray(top_xy, float).reshape(-1, 2).tolist()
        return {"source": source, "n_top": int(info["n_top"]), "ratio_cloud": info["ratio_cloud"],
                "ratio_blob": info["ratio_blob"], "path": path, "label": det.shape, "conf": float(conf),
                "near_bin": near_bin, "dz_top": float(det.top[2] - self.scene.counter_z), "top_xy": top_list}

    def _log_diag(self, det, why=None):
        g = det.shape_diag

        def f(x, n=3):
            return "None" if x is None else f"{x:.{n}f}"
        self.log(f"[shape_diag] {det.color} src={g['source']} n_top={g['n_top']} r_cloud={f(g['ratio_cloud'])} "
                 f"r_blob={f(g['ratio_blob'])} path={g['path']} label={g['label']} conf={g['conf']:.2f} "
                 f"near_bin={g['near_bin']} dz_top={f(g['dz_top'], 4)}" + (f" rej={why}" if why else ""))

    def _from_depth(self, color, u, v, d, area, blob=None, frame_depth=None):
        P = self._pixel_to_pelvis(u, v, d)
        z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
        top = P[P[:, 2] >= z_top - self.cfg["top_band"]]
        if len(top) < 5:
            return None
        # pixels sample the top face unevenly (perspective): weight each by its footprint ~ depth^2
        w = (d[P[:, 2] >= z_top - self.cfg["top_band"]] / 1000.0) ** 2
        xy = (top[:, :2] * w[:, None]).sum(0) / w.sum()
        z = float(np.median(top[:, 2]))
        near_bin = self.near_bin_guard and self.classify_shape and self._near_bin(top)   # near-bin guard: abstain on shape only
        shape, conf, path = self._shape_conf(top[:, :2], blob, near_bin)
        apex_yaw, apex_conf = None, 0.0
        if shape == "triangle" and self.triangle_pose:
            from .prism_pose import robust_center, triangle_pose
            tp = triangle_pose(top[:, :2], altitude=self.size)
            if tp is not None:
                apex_yaw, apex_conf = float(tp[1]), float(tp[2])
            rc = robust_center(top[:, :2])
            if rc is not None:
                xy = rc  # unweighted, trimmed: the weighted mean is biased by the triangle's uneven pixel sampling
        xy = xy + self.xy_offset
        if self.y_affine is not None:
            xy = self._y_affine(xy)
        det = Detection(color, np.array([xy[0], xy[1], z]), area, len(u), "depth", _yaw(top[:, :2]),
                        (float(np.median(u)), float(np.median(v))))
        det.shape = shape
        if self.want_path:
            det.shape_conf, det.shape_path = conf, self._last_path
        if apex_yaw is not None:
            det.apex_yaw, det.apex_conf = apex_yaw, apex_conf
        if self.want_apex and shape == "triangle":
            from .apex_gate import apex_gate_features
            det.apex_gate = apex_gate_features(top[:, :2], altitude=self.size)
        if self.generic_fallback:   # flag off: no call, no work
            self._apply_generic_fallback(det, conf, blob, frame_depth)
        if self.shape_diag:
            nb = bool(near_bin) if (self.classify_shape and self.near_bin_guard) else _safe_near_bin(self, top)
            det.shape_diag = self._diag(det, "depth", top[:, :2], blob, conf, path, nb)
        return det

    def _apply_shape2d(self, det, blob):
        """perception.shape_2d_v2, yellow only. A confident metric shape is kept (disagreement logged). Where the shape is
        None from too few top-face points, or came from the 2D-blob ratio, the top-edge rule decides (None = abstain).
        Off / near-bin / border-clipped / classify_shape off: untouched."""
        path = self._last_path if det.shape_path is None else det.shape_path
        p = self.shape_2d_params
        try:
            f = shape2d_features(blob, p.get("approx_eps_frac", 0.02))
            new = decide_shape2d(f, p.get("quad_rms_max", 1.77), p.get("n_vert_min", 8), p.get("cube_n_vert_max", 6))
        except Exception as e:   # never break detection
            self.log(f"[shape2d] color={det.color} error {type(e).__name__}: {e}")
            return

        def g(x):
            return "None" if x is None else f"{x:.3f}"
        if path == "metric":
            note = f" metric={det.shape} DISAGREE" if new is not None and new != det.shape else f" metric={det.shape}"
            self.log(f"[shape2d] color={det.color} quad_rms={g(f['quad_rms'])} n_vert={f['n_vert']} "
                     f"axis_ratio={g(f['axis_ratio'])} -> {new}{note} (kept)")
            return
        if path not in ("blob", "abstain_few_points"):
            self.log(f"[shape2d] color={det.color} quad_rms={g(f['quad_rms'])} n_vert={f['n_vert']} "
                     f"axis_ratio={g(f['axis_ratio'])} -> {new} (path={path}, kept {det.shape})")
            return
        old = det.shape
        det.shape = new
        det.shape_conf = 0.0 if new is None else 0.6   # ponytail: fixed conf; real conf needs the measured margin
        self.log(f"[shape2d] color={det.color} quad_rms={g(f['quad_rms'])} n_vert={f['n_vert']} "
                 f"axis_ratio={g(f['axis_ratio'])} -> {new} (was {old}, path={path})")

    def _log_orange_candidate(self, blob):
        """Log only (prism vs cube candidate: topQ = top-edge quad_rms, hu2 = Hu moment 2). Never sets a shape."""
        try:
            import cv2
            f = shape2d_features(blob, self.shape_2d_params.get("approx_eps_frac", 0.02))
            hu2 = float(cv2.HuMoments(cv2.moments((np.asarray(blob) > 0).astype(np.uint8), True))[1][0])
            q = f["quad_rms"]
            self.log(f"[shape2d] color=orange candidate topQ={'None' if q is None else f'{q:.3f}'} hu2={hu2:.5f} -> unchanged")
        except Exception as e:
            self.log(f"[shape2d] color=orange error {type(e).__name__}: {e}")

    def _apply_generic_fallback(self, det, shape_conf, blob, depth):
        """Attach an accepted generic proposal, or mark the detection so policy abstains.

        This is only called on the opt-in path. Any malformed calibration/depth
        degrades to leaving the object in place instead of terminating a look.
        """
        if not self.generic_fallback or det is None:
            return
        known = det.shape in ("cube", "cylinder", "triangle")
        det.generic_required = (not known) or float(shape_conf) < self.generic_shape_min_conf
        if not det.generic_required or depth is None or blob is None:
            return
        try:
            from .generic_grasp import propose_grasp
            camera = self.cal.cams[self.cam]
            raw_depth = np.asarray(depth)
            dlo, dhi = self.cfg["depth_range_mm"]
            valid_depth = np.isfinite(raw_depth) & (raw_depth >= dlo) & (raw_depth <= dhi)
            depth_m = np.where(valid_depth, raw_depth, 0.0).astype(np.float64) * float(camera.depth_scale)
            # Keep the original RGB blob for the proposal's truncation check:
            # invalid edge depth must not hide that the object touches the frame.
            proposal = propose_grasp(blob, depth_m, camera.K, self._T())
        except Exception:
            return
        if proposal is None:
            return
        det.generic_confidence = float(proposal.confidence)
        if (proposal.confidence < self.generic_min_confidence or proposal.width <= 0.005 or
                proposal.width > self.generic_max_width):
            return
        centre = np.asarray(proposal.centre, float).copy()
        centre[:2] += self.xy_offset
        if self.y_affine is not None:
            centre[:2] = self._y_affine(centre[:2])
        det.top = centre
        det.yaw = float(proposal.yaw)
        det.generic_width = float(proposal.width)
        # A low-confidence known label must never select a shape-specific bin.
        det.shape = None
        det.shape_path = "generic"

    def _from_plane(self, color, uv, area, n_depth):
        """Centroid ray of the blob intersected with the horizontal plane at the object's top."""
        T = self._T()
        K = self.cal.cams[self.cam].K
        ray_opt = np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
        ray = T[:3, :3] @ ray_opt
        o = T[:3, 3]
        z_top = self.scene.counter_z + self.height
        # the blob centroid lies between top face and front face: aim a bit below the top
        z_aim = z_top - 0.25 * self.height
        if ray[2] >= -1e-6:
            return Detection(color, np.array([np.nan, np.nan, np.nan]), area, n_depth, "plane")
        s = (z_aim - o[2]) / ray[2]
        p = o + s * ray
        p[:2] += self.xy_offset
        if self.y_affine is not None:
            p[:2] = self._y_affine(p[:2])
        return Detection(color, np.array([p[0], p[1], z_top]), area, n_depth, "plane", None, (float(uv[0]), float(uv[1])))

    def _reject(self, det, bins_first=False):
        """Rejection reason or None. Stock order: height, off_mat, in_bin. `bins_first` (#6 verification look) checks
        the bin footprints before the height so a stacked / rim object is labelled in_bin:<name>, not height."""
        if det is None or not np.all(np.isfinite(det.top)):
            return "no_3d"
        c, sc = self.cfg, self.scene
        h = (det.top[2] - sc.counter_z) / self.height
        lo, hi = c["height_range"]
        if not bins_first and not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        pm = sc.pelvis_to_mat(det.top)
        W, D = sc.cfg["counter"]["mat_size"]
        m = c["counter_margin"]
        if not (-m <= pm[0] <= W + m and -m <= pm[1] <= D + m):
            return "off_mat"
        bm = c["bin_exclusion_margin"]
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + bm
        for name, ctr in self._bin_centres_mat().items():
            if np.all(np.abs(pm[:2] - ctr) <= half):
                return f"in_bin:{name}"
        if bins_first and not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        return None

    def _fk_arm_mask(self, shape, arm_q, hand_q=None):
        """perception.color_gate.fk_mask: bool image mask of the FK-posed arm + hand surface points projected with the
        CURRENT camera pose (pitch-cal / B1 corrected), dilated by fk_dilate_px. None on any failure (= no mask)."""
        import cv2
        try:
            if self._arm_surface is None:
                from .hand_calib import ArmSurface
                self._arm_surface = ArmSurface()
            P, _ = self._arm_surface.posed(arm_q, hand_q)
            uvz = self._project(P)
            h, w = shape
            ok = np.all(np.isfinite(uvz), axis=1) & (uvz[:, 2] > 0)
            u, v = np.round(uvz[ok, 0]).astype(int), np.round(uvz[ok, 1]).astype(int)
            inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            m = np.zeros(shape, np.uint8)
            m[v[inb], u[inb]] = 1
            r = int(self.color_gate_cfg.get("fk_dilate_px", 15))
            if r > 0:
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            self.fk_mask_px = int(m.sum())
            return m.astype(bool)
        except Exception as e:   # never raises into detect(): no mask is the stock behaviour
            self.log(f"[color_gate] fk mask failed ({type(e).__name__}: {e}); unmasked")
            return None

    def _color_gate(self, dets, rejected):
        """perception.color_gate (FLAG): post-filter of the accepted detections. Rejected ones are appended to
        `rejected` as (det, "color_gate:<why>"). More than max_cands survivors -> degraded (the policy's DEGRADED tier
        orders them); NOTHING is truncated here, so an unplannable large blob can never hide a real object (Codex C1)."""
        g = self.color_gate_cfg
        kept, n_in, reasons = [], 0, {}
        for det in dets:
            n_in += 1
            why = self._color_gate_why(det, g)
            if why:
                rejected.append((det, why))
                reasons[why] = reasons.get(why, 0) + 1
            else:
                kept.append(det)
        self.log(f"[color_gate] in={n_in} out={len(kept)} reasons={{"
                 + ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items())) + "}")   # diagnostics only
        if len(kept) > int(g["max_cands"]):
            self.color_gate_degraded = True
            self.log(f"[color_gate] degraded: {len(kept)} candidates > max_cands {g['max_cands']}; all passed on, "
                     f"largest first")
            kept = sorted(kept, key=lambda d: d.n_px, reverse=True)
        return [self._color_gate_shape(d) for d in kept]

    def _color_gate_why(self, det, g):
        """Rules 1 (area floor scaled by distance), 1b (area ceiling: rim / tape / merged blob), 2b (positive workspace:
        only the counter in front of the bin row) and 2 (bin zone). None = keep."""
        if g.get("sat_floor") and self._color_gate_sat_low(det, g):
            return "color_gate:sat_floor"
        Z = self._color_gate_depth(det.top)
        if Z is None:
            if det.n_px < g["min_px_no_z"]:
                return "color_gate:speckle_no_z"
            if g.get("max_px_no_z") is not None and det.n_px > g["max_px_no_z"]:
                return "color_gate:too_big_no_z"
        else:
            K = self.cal.cams[self.cam].K
            expected = float(K[0, 0]) * float(K[1, 1]) * (self.size / Z) ** 2   # one object's top face, px
            if det.n_px < g["area_frac"] * expected:
                return "color_gate:speckle"
            if g.get("max_area_frac") is not None and det.n_px > g["max_area_frac"] * expected:
                return "color_gate:too_big"
        return self._color_gate_workspace(det, g) or self._color_gate_bin(det, g)

    def _color_gate_sat_low(self, det, g):
        """perception.color_gate.sat_floor: a blue (purple) or yellow blob whose median S is below its floor is a
        shadow or sheen. Other colours (red, orange) and blobs without a measured S are never rejected here."""
        name = "blue" if det.color == "purple" else det.color
        floor = (g.get("sat_floor_params") or {}).get(name)
        return floor is not None and det.sat_med is not None and det.sat_med < float(floor)

    def _color_gate_workspace(self, det, g):
        """Rule 2b: keep only pelvis x in ws_x, |y| <= ws_abs_y, and in front of the bin row's near edge (nominal or
        bin.vote-shifted footprint, never the near_margin hedge) minus ws_edge_m. Bin walls / rims project BEHIND it."""
        ws_x = g.get("ws_x")
        if ws_x is None:
            return None
        x, y = float(det.top[0]), float(det.top[1])
        near_mat_y = min(float(lo[1]) for lo, _ in self._color_gate_boxes(g, hedge=False))
        near_x = float(self.scene.mat_to_pelvis(np.array([0.0, near_mat_y]))[0])
        if not (ws_x[0] <= x <= min(ws_x[1], near_x - float(g.get("ws_edge_m", 0.01)))) or abs(y) > g["ws_abs_y"]:
            return "color_gate:workspace"
        return None

    def _color_gate_depth(self, top):
        """Camera-frame depth Z (optical axis, m) of a pelvis-frame point, or None (no extrinsics / not in front)."""
        try:
            z = float(self._project(np.asarray(top, float))[2])
        except ValueError:   # calibration has no extrinsics yet
            return None
        return z if np.isfinite(z) and z > 0 else None

    def _color_gate_bin(self, det, g):
        """Rule 2, 3D in the mat frame (nominal footprint, no margin). Inside a bin -> in_bin. Within edge_m outside
        one -> kept only as a depth top face edge_h above the counter (a bin rim), else bin_edge. Else keep."""
        sc = self.scene
        pm = sc.pelvis_to_mat(det.top)[:2]
        near_edge = False
        for lo, hi in self._color_gate_boxes(g):
            outside = np.maximum(lo - pm, 0.0) + np.maximum(pm - hi, 0.0)
            gap = float(np.hypot(*outside))
            if gap == 0.0:
                return "color_gate:in_bin"
            near_edge = near_edge or gap <= float(g["edge_m"])
        # no depth = cannot tell a rim from an object: keep (Codex H2); depth: keep only an object-height top face
        if near_edge and det.method == "depth" and not self._color_gate_rim_ok(det, g):
            return "color_gate:bin_edge"
        return None

    def _color_gate_boxes(self, g, hedge=True):
        """[(lo, hi)] mat-frame footprint of every bin. Flag live_bins off: nominal scene.yaml. On: the policy's live
        estimate (bin.vote shift; a calibrate_bins success already moves the DETECTIONS onto the nominal bins), and with
        no estimate at all the nominal box grown near_margin_m towards the robot (mat -y = pelvis -x, the drop_2 shift)."""
        sc = self.scene
        shift = dict(getattr(self.bin_shift_src, "bin_shift", None) or {}) if self.live_bins else {}
        estimate = bool(shift) or (self.calibration is not None and self.calibration.success)
        out = []
        for g_ in self.bin_geometry(gate=True).values():
            lo, hi = g_["outer_lo"], g_["outer_hi"]
            if hedge and self.live_bins and not estimate:
                lo = lo - np.array([0.0, float(g.get("near_margin_m", 0.10))])
            out.append((lo, hi))
        return out

    def _color_gate_rim_ok(self, det, g):
        h = float(det.top[2] - self.scene.counter_z)
        lo, hi = g["edge_h"]
        return det.method == "depth" and det.n_depth > 0 and lo <= h <= hi

    def _color_gate_shape(self, det):
        """Rule 4: a triangle that is not a depth top face (2D blob) has no trusted shape: unknown."""
        if det.shape == "triangle" and det.method != "depth":
            diag = dict(det.shape_diag, label=None, gate="color_gate:no_depth_triangle") if det.shape_diag else None
            return replace(det, shape=None, shape_conf=0.0, shape_path="abstain_color_gate", shape_diag=diag,
                           apex_yaw=None, apex_conf=0.0, apex_gate=None)
        return det


def _safe_near_bin(perc, top):
    """Diagnostics only: the near-bin test even when the guard is off (classify_shape false)."""
    try:
        return bool(perc._near_bin(top))
    except Exception:
        return None


def _yaw(xy):
    """Principal direction of the top-face points, folded to [-pi/4, pi/4) (square symmetry). Info only."""
    if len(xy) < 10:
        return None
    import cv2
    (_, _), (_, _), ang = cv2.minAreaRect(xy.astype(np.float32))
    a = np.radians(ang)
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
