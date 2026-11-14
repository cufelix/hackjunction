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

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import threading
from collections import Counter
import time

import cv2
import numpy as np

from .bin_vote import bin_geometry
from .prism_orientation import prism_orientation
from .shape_id import classify_top_face   # the ONE shape decision; swap this import to change the classifier
from .shape_2d import decide_shape2d, shape2d_features
from .shape_id import classify_top_face_diag   # diagnostics only (perception.shape_diag): never decides anything
from .hue_guard import AMBIGUOUS_CONF, Cand, classify_hue, judge, top_face_hsv
from . import bare_table


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
    vertex_yaw: float | None = None  # triangle only: blob rectangle's pelvis-frame vertex line (mod 120 deg)
    shape_conf: float = 0.0    # shape_id confidence (0..1) behind `shape`
    apex_gate: tuple | None = None  # grasp.posture_apex_gate only: (apex_conf, max vertex angle deg) of a "triangle" top
    shape_path: str | None = None  # shape_id path; filled for shape-specific posture/cage planning
    generic_required: bool = False  # grasp.generic_fallback: label absent/below confidence threshold
    generic_width: float | None = None  # accepted proposal's closing-axis width (m); None means abstain
    generic_confidence: float = 0.0
    shape_diag: dict | None = None  # perception.shape_diag only: source, n_top, ratio_cloud, ratio_blob, path, label,
                                    # conf, near_bin, dz_top (top z - counter z, m), top_xy (list of [x, y], pelvis)
    color_conf: float = 1.0  # policy.place_gate only; 1 keeps synthetic/legacy detections confidently compatible
    hsv_med: tuple | None = None  # perception.hue_guard only: (H, S, V) median of the brightest 35 pct by V of the object pixels
    sat_med: float | None = None  # perception.color_gate.sat_floor only: median HSV S (0-255) over the blob mask
    hue_mixed: bool = False       # H1: a post-split part still contains both red and yellow source-mask pixels
    prism_orientation: str | None = None  # perception.prism_orientation only: "standing" | "lying" | None (unsure)
    ridge_yaw: float | None = None  # lying prism: pelvis-frame heading (rad, mod pi) of the ridge = long axis
    prism_xy: tuple | None = None   # lying prism: ridge-strip centre (pelvis xy, = footprint centroid by symmetry)
    prism_conf: float = 0.0         # smallest normalised cue margin of the orientation decision (0..1)
    novel: bool = False  # novel objects only: built from a depth blob, never a known colour/shape detection
    label: str | None = None  # novel only: query phrase / label
    footprint: tuple | None = None  # novel only: (major, minor) extents of the blob in m
    height_m: float | None = None  # novel only: top height above the counter plane in m
    ungraspable: bool = False  # novel only: too wide / flat / unsupported -> the policy skips it

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


@dataclass(frozen=True)
class FrameTransform:
    """One immutable camera transform used by every projection/deprojection of one RGB-D frame."""
    T: np.ndarray = field(repr=False, compare=False)
    version: int = 0


@dataclass(frozen=True)
class CounterHeightSample:
    """Pure result of one counter-plane height fit; committing it is a separate main-thread operation."""
    z: float
    tilt_deg: float
    n_inliers: int
    reason: str | None


class Perception:
    def __init__(self, cfg, scene, calibration, log=None):
        self._root_cfg = cfg
        self.cfg = cfg["perception"]
        self.profile = cfg.get("profile", "none")
        self.scene = scene
        self.cal = calibration
        self.cam = self.cfg["camera"]
        # perception.principal_point (Oliver lane 3): "file" = calibration.yaml K (stock, right for the real D435i);
        # "center" = cx, cy at the image centre and fx = fy = mean, which is what Isaac renders: IsaacLab drops
        # aperture offsets ("c_x and c_y will be half of width and height", utils/sensors.py).
        # Perception gets its own shallow copy of the calibration, so every path (pixel_to_pelvis, project, the ray
        # fallbacks) uses the same K and nothing outside perception changes.
        centred_sim = bool(self.cfg.get("sim_center_principal_point", False)) and self.profile == "sim"
        explicit_center = self.cfg.get("principal_point", "file") == "center"
        if explicit_center or centred_sim:
            import copy
            import dataclasses
            cam = calibration.cams[self.cam]
            K = np.array(cam.K, float).copy()
            f = 0.5 * (K[0, 0] + K[1, 1])
            K[0, 0] = K[1, 1] = f
            K[0, 2], K[1, 2] = (cam.width - 1) / 2.0, (cam.height - 1) / 2.0
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
        calib_cfg = cfg.get("calib") or {}
        self.counter_after_hand = bool(calib_cfg.get("counter_after_hand", False))
        self.refit_per_look = bool(calib_cfg.get("refit_per_look", False))
        self.counter_tracker = None       # installed by policy with the same fitter used by the primary phase
        self.camera_pose_selected = False
        # calib.wide_counter_window (FLAG, default OFF): estimation window nominal -wide_lo .. +wide_hi, runtime height
        # estimated even without apply_height, mat z follows the runtime counter (see COUNTERZ.md)
        self.wide = bool(((cfg.get("calib") or {}).get("wide_counter_window", False)))
        self.wide_lo = float(self.ah.get("wide_lo_m", 0.08))
        self.wide_hi = float(self.ah.get("wide_hi_m", 0.22))
        if self.wide:
            self.scene.mat_z_follows_counter = True
        self.height_est = bool(self.ah.get("apply_height")) or self.wide   # a runtime counter height is estimated
        self._ah_done = not self.height_est
        self._ah_committed = None
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
        self._transform_lock = threading.Lock()
        self._transform_version = 0
        self._frame_transform = ContextVar(f"perception_frame_{id(self)}", default=None)
        # opt-in shape / triangle pose (all default OFF = stock behaviour)
        grasp = cfg.get("grasp", {})
        self.generic_fallback = bool(grasp.get("generic_fallback", False))
        self.generic_shape_min_conf = float(grasp.get("generic_shape_min_conf", 0.5))
        self.generic_min_confidence = float(grasp.get("generic_min_confidence", 0.5))
        self.generic_max_width = float(grasp.get("generic_max_width", 0.060))
        self.classify_shape = bool(self.cfg.get("classify_shape", False)) or self.generic_fallback
        self.want_path = (bool(grasp.get("close_posture_by_shape", False))
                          or bool(grasp.get("cage_mode", False)) or self.generic_fallback)
        self._last_path = None
        # grasp.posture_apex_gate (default off): prism-top check features for "triangle" detections (apex_gate.py)
        self.want_apex = self.want_path and bool(cfg.get("grasp", {}).get("posture_apex_gate", False))
        self.triangle_pose = bool(self.cfg.get("triangle_pose", False))
        # perception.prism_orientation (default off): standing vs lying triangular prism (prism_orientation.py)
        self.prism_orientation = bool(self.cfg.get("prism_orientation", False))
        self.prism_params = dict(self.cfg.get("prism_orientation_params") or {})
        self.tri_vertex_align = bool(grasp.get("tri_vertex_align", False))
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
        self.split_by_solidity = bool(self.cfg.get("split_by_solidity", False))
        self.split_solidity_cfg = dict(self.cfg.get("split_by_solidity_params") or {})
        self.tape_value_cut = bool(self.cfg.get("tape_value_cut", False))
        self.tape_value_cfg = dict(self.cfg.get("tape_value_cut_params") or {})
        self.gate_below_bins = bool(self.cfg.get("gate_below_bins", False))
        self.gate_below_cfg = dict(self.cfg.get("gate_below_bins_params") or {})
        self._below_bin_row = None
        self._image_bin_gate_active = False
        self.reject_log = bool(self.cfg.get("reject_log", False))
        self.bins_from_image = bool(self.cfg.get("bins_from_image", False))
        self.bins_img_cfg = dict(self.cfg.get("bins_from_image_params") or {})
        self.bin_rims = None          # last RimResult (bins_from_image); diagnostics / tests
        self.bin_rim_hint = {}        # {bin: (dx, dy)} pelvis shift estimated from the rim line (param hint)
        self._below_bin_margin = float(self.gate_below_cfg.get("margin_px", 3))
        self.reject_tally = {}
        self.still_vote = bool(self.cfg.get("still_vote", False))
        self.still_vote_cfg = dict(self.cfg.get("still_vote_params") or {})
        self._still_frames = []
        self.flood_guard = float(self.cfg.get("flood_guard", 0.0) or 0.0)   # perception.flood_guard (FLAG, default OFF)
        self.place_gate = bool((cfg.get("policy") or {}).get("place_gate", False))
        # perception.hue_guard (default OFF): abstain on orange/yellow hues near the red|yellow boundary (hue_guard.py)
        hg = self.cfg.get("hue_guard") or {}
        self.hue_guard = bool(hg.get("enabled", False))
        self.hue_guard_min_gap_deg = float(hg.get("min_gap_deg", 6.0))
        self.hue_guard_red_min_s = float(hg.get("red_min_s", 110.0))
        self.topface_mode = self.cfg.get("topface_mode", "top35")
        self.topface_min_px = int(self.cfg.get("topface_min_px", 1024))
        # bare / glossy table (all default OFF): glare_reject, reflection_reject, local_bg (bare_table.py)
        self.glare_cfg = dict(self.cfg.get("glare_reject") or {})
        self.glare_reject = bool(self.glare_cfg.get("enabled", False))
        self.reflect_cfg = dict(self.cfg.get("reflection_reject") or {})
        self.reflection_reject = bool(self.reflect_cfg.get("enabled", False))
        self.local_bg_cfg = dict(self.cfg.get("local_bg") or {})
        self.local_bg = bool(self.local_bg_cfg.get("enabled", False))
        self.bare_stats = {"glare_px": 0, "reflection": 0, "local_bg": 0}   # last detect(): diagnostics only
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
        self.fk_silhouette = self.fk_mask and bool(self.color_gate_cfg.get("fk_silhouette", False))
        self._arm_surface = None   # lazily built FK surface for fk_mask (own instance: never shares B1's lock)
        self._arm_surface_error = None
        self._last_arm_mask = None
        self.fk_mask_px = 0        # pixels the last fk mask removed (diagnostics)
        self.sat_floor_adaptive = bool(self.cfg.get("sat_floor_adaptive", False))
        self.sat_floor_fixed = self.color_gate and bool(self.color_gate_cfg.get("sat_floor", False))
        self.sat_floor = self.color_gate and (self.sat_floor_fixed or self.sat_floor_adaptive)
        self._sat_floors = dict(self.color_gate_cfg.get("sat_floor_params") or {})
        self.area_floor_geometric = self.color_gate and bool(self.cfg.get("area_floor_geometric", False))
        self.partial_ok_near_hand = bool(self.cfg.get("partial_ok_near_hand", False))
        self.partial_near_hand_cfg = dict(self.cfg.get("partial_ok_near_hand_params") or {})
        self.live_bins = bool(self.color_gate_cfg.get("live_bins", False))   # perception.color_gate.live_bins (FLAG)
        self.bin_shift_src = None  # live_bins: object with .bin_shift {bin: pelvis dxy} (the policy's planner, bin.vote)
        self.geometry_runtime = False   # bin.geometry_runtime flag, set by policy.py: every bin consumer follows bin_shift_src.bin_shift
        self.bin_vote_on = False   # bin.vote flag, set by policy.py: bin footprints follow bin_shift_src.bin_shift
        # P0 depth recovery: imported/executed only on this opt-in path and only on scheduled detect() updates, never
        # on ordinary control ticks.  The real runner already executes scheduled look work in its worker thread.
        self.depth_blob_recover = bool(self.cfg.get("depth_blob_recover", False))
        self.depth_recover_cfg = dict(self.cfg.get("depth_blob_recover_params") or {})
        self.depth_recover_ms = 0.0
        if self.depth_blob_recover:
            # Loading Pinocchio, the URDF, and the surface asset can take tens
            # of milliseconds.  Do it during policy construction, never on the
            # first scheduled perception/control tick.
            try:
                from .hand_calib import ArmSurface
                self._arm_surface = ArmSurface()
            except Exception as exc:
                self._arm_surface_error = exc
                self.log(f"[depth_recover] FK mask unavailable ({type(exc).__name__}: {exc}); recovery will abstain")
        # R22 novel objects.  Keeping the switch here (rather than merely in
        # the policy) makes the disabled detect() path instruction-for-
        # instruction identical through the existing return below.
        self.novel_objects = bool(self.cfg.get("novel_objects", False))

    def reset_calibration(self):
        """Restore the configured fallback at the start of every episode."""
        self.scene.counter_z_runtime = None
        self._ah_est, self._ah_tries = [], 0
        self._ah_done = not self.height_est
        self._ah_committed = None
        self.camera_pose_selected = False
        if self.counter_tracker is not None:
            self.counter_tracker.reset()
        self.xy_offset = self.base_xy_offset.copy()
        self.calibration = None
        self.pitch_calibration = None
        with self._transform_lock:
            self.pitch_T = None
            self.cal = self.cal_nominal   # B1: a hand-calibrated copy never outlives its episode
            self._transform_version += 1
        if self.colour_kmeans:
            self.hsv = self._colour_kmeans_base
            self._colour_kmeans_done = False
            self._colour_kmeans_tries = 0
            self.colour_kmeans_info = None
        self._still_frames = []
        self.bin_rim_hint = {}        # a rim hint must not survive into the next episode

    def frame_transform(self):
        """Snapshot the current camera pose/version for one complete RGB-D frame."""
        with self._transform_lock:
            T = np.array(self.pitch_T if self.pitch_T is not None else self.cal._T(self.cam), float, copy=True)
            version = self._transform_version
        T.setflags(write=False)
        return FrameTransform(T, version)

    @contextmanager
    def use_frame_transform(self, frame=None):
        """Make one immutable transform visible to all nested frame consumers in this thread."""
        snap = self.frame_transform() if frame is None else frame
        token = self._frame_transform.set(snap)
        try:
            yield snap
        finally:
            self._frame_transform.reset(token)

    def _T(self, frame=None):
        """Episode-corrected camera pose, or the immutable transform bound to the current frame."""
        snap = frame if frame is not None else self._frame_transform.get()
        if snap is not None:
            return snap.T
        T = self.pitch_T
        return T if T is not None else self.cal._T(self.cam)

    def projection_snapshot(self):
        """Read-only T_pelvis_optical for all consumers of one copied frame."""
        T = self.current_T()
        T.setflags(write=False)
        return T

    def _pixel_to_pelvis(self, u, v, depth_mm, frame=None, transform_snapshot=None):
        # Keep the disabled/fallback path byte-for-byte identical to Calibration.pixel_to_pelvis.
        if frame is not None and not isinstance(frame, FrameTransform):   # a raw 4x4 here would be silently transposed
            raise TypeError(f"frame must be a FrameTransform, got {type(frame).__name__}; pass transform_snapshot=")
        if transform_snapshot is not None:   # fixvote H4 explicit per-frame pose
            T = np.asarray(transform_snapshot, float)
            p = self.cal.deproject(self.cam, u, v, depth_mm)
            return p @ T[:3, :3].T + T[:3, 3]
        snap = frame if frame is not None else self._frame_transform.get()
        T = snap.T if snap is not None else self.pitch_T
        if T is None and snap is None:
            return self.cal.pixel_to_pelvis(self.cam, u, v, depth_mm)
        if T is None:
            T = self.cal._T(self.cam)
        p = self.cal.deproject(self.cam, u, v, depth_mm)
        return p @ T[:3, :3].T + T[:3, 3]

    def _project(self, p_pelvis, frame=None):
        # Keep the disabled/fallback path byte-for-byte identical to Calibration.project.
        if frame is not None and not isinstance(frame, FrameTransform):   # a raw 4x4 here would be silently transposed
            raise TypeError(f"frame must be a FrameTransform, got {type(frame).__name__}; pass transform_snapshot=")
        snap = frame if frame is not None else self._frame_transform.get()
        T = snap.T if snap is not None else self.pitch_T
        if T is None and snap is None:
            return self.cal.project(self.cam, p_pelvis)
        if T is None:
            T = self.cal._T(self.cam)
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
        """Install one transform/version atomically; active frame snapshots remain immutable."""
        with self._transform_lock:
            self.pitch_T = None if not out.success else np.asarray(out.T, float).copy()
            self.pitch_calibration = out
            self._transform_version += 1
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
        frame = self._frame_transform.get()
        return np.array(frame.T if frame is not None else self.frame_transform().T, float)

    def set_camera_pose(self, T):
        """perception.hand_calibration: use T as this camera's T_pelvis_optical until the next reset, on a copy of the
        calibration (the shared object stays nominal). With online_pitch_cal merged, pitch_T (which wins in _T /
        _pixel_to_pelvis / _project) is set too, so every path sees the refined pose."""
        from .hand_calib import corrected_calibration
        with self._transform_lock:
            self.cal = corrected_calibration(self.cal, self.cam, T)
            if hasattr(self, "pitch_T"):
                self.pitch_T = np.array(T, float)
            self._transform_version += 1
        self.camera_pose_selected = True

    def calibration_failure(self, reason, elapsed_ms=0.0):
        """Latch a safe failed result after an unexpected calibration error."""
        self.xy_offset = self.base_xy_offset.copy()
        self.calibration = BinCalibration(False, True, 0.0, self.xy_offset.copy(), 0, np.inf,
                                          float(elapsed_ms), str(reason))
        return self.calibration

    def _counter_plane(self, depth, c, z0=None, transform_snapshot=None, wide_seed=False):
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
             self._pixel_to_pelvis(uu, vv, depth[vv, uu].astype(float), transform_snapshot=transform_snapshot))
        sc = self.scene

        # Fit z = ax + by + c to counter points. Bins and objects are positive outliers and are trimmed iteratively.
        pm = sc.pelvis_to_mat(P)
        mat = np.asarray(sc.cfg["counter"]["mat_size"], float)
        if self.wide and z0 is None and (wide_seed or sc.counter_z_runtime is None):
            # calib.wide_counter_window: no runtime height yet -> seed the +-window from the dominant z layer inside
            # nominal -wide_lo .. +wide_hi over the mat (bins / objects are minor layers)
            z0 = self._wide_seed(P, pm, mat, int(c.get("min_plane_points", 500)))
            if z0 is None:
                return (None, None, "counter plane not visible")
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

    def counter_z_prior(self):
        """calib.wide_counter_window: configured counter z prior (scene.counter_z_prior, else calib.plane.counter_z_prior_m,
        pelvis frame), clamped to the window; None if neither is set."""
        v = getattr(self.scene, "counter_z_prior", None)
        if v is None:
            v = self.ah.get("counter_z_prior_m")
        if v is None or not np.isfinite(float(v)):
            return None
        nom = self.scene.nominal_counter_z
        return float(np.clip(float(v), nom - self.wide_lo, nom + self.wide_hi))

    def _wide_seed(self, P, pm, mat, min_points):
        """calib.wide_counter_window: z of the counter layer = the lowest well-populated peak of the 1 cm z histogram of
        mat points inside nominal -wide_lo .. +wide_hi (bin rims / objects are smaller layers ABOVE the counter). None if
        no layer has min_points."""
        nom = self.scene.nominal_counter_z
        lo, hi = nom - self.wide_lo, nom + self.wide_hi
        on = ((pm[:, 0] >= -0.03) & (pm[:, 0] <= mat[0] + 0.03) & (pm[:, 1] >= -0.03) & (pm[:, 1] <= mat[1] + 0.03)
              & (P[:, 2] >= lo) & (P[:, 2] <= hi))
        if int(on.sum()) < min_points:
            return None
        z = P[on, 2]
        edges = np.arange(lo, hi + 0.01, 0.01)
        h = np.histogram(z, bins=edges)[0].astype(float)
        smooth = h + np.r_[0.0, h[:-1]] + np.r_[h[1:], 0.0]    # +-1 cm
        top = smooth.max()
        if top < min_points:
            return None
        for i in np.nonzero(smooth >= 0.6 * top)[0]:           # lowest peak of comparable mass
            if i == 0 or smooth[i] >= smooth[i - 1]:
                if i == len(smooth) - 1 or smooth[i] >= smooth[i + 1]:
                    sel = z[(z >= edges[max(i - 1, 0)]) & (z <= edges[min(i + 2, len(edges) - 1)])]
                    return float(np.median(sel))
        return None

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
        if self.wide:   # calib.wide_counter_window: pass 1 seeds z0 from the nominal -wide_lo .. +wide_hi histogram
            c["plane_z_window"] = 0.03
            P, coef, why = self._counter_plane(depth, c, wide_seed=True)
            if why:
                return nan, nan, 0, why
            z = float(coef @ np.r_[xy, 1.0])
            if not nom - self.wide_lo <= z <= nom + self.wide_hi:
                return z, nan, 0, f"z {z - nom:+.3f} outside the window"
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

    def measure_counter_height(self, depth, frame=None):
        """Pure counter-plane fit for a copied frame; safe to run in a background worker."""
        a = self.ah
        try:
            if frame is None:
                z, tilt, n, why = self._plane_height(depth, a)
            else:
                with self.use_frame_transform(frame):
                    z, tilt, n, why = self._plane_height(depth, a)
        except Exception as exc:
            z, tilt, n, why = float("nan"), float("nan"), 0, f"{type(exc).__name__}"
        return CounterHeightSample(float(z), float(tilt), int(n), why)

    def height_fallback(self):
        """Height committed when no consistent measurement exists: the configured prior under calib.wide_counter_window
        (never the nominal scene value, counterz), else None = nominal."""
        return self.counter_z_prior() if self.wide else None

    def accept_counter_height_sample(self, sample, commit_scene=True):
        """Apply the N-sample spread/MAD gate and latch at most one runtime height per episode."""
        a = self.ah
        if not self.height_est or self._ah_done:
            return self._ah_done, self._ah_committed
        self._ah_tries += 1
        n_looks = max(1, int(a.get("looks", 3)))
        if sample.reason is None and np.isfinite(sample.z):
            self._ah_est.append(float(sample.z))
        window = np.asarray(self._ah_est[-n_looks:], float)
        est = float(np.median(window)) if len(window) else float("nan")
        spread = float(np.ptp(window)) if len(window) else float("inf")
        mad = float(np.median(np.abs(window - est))) if len(window) else float("inf")
        consistent = (len(window) == n_looks and spread <= float(a.get("max_spread_m", 0.015))
                      and mad <= float(a.get("max_mad_m", 0.006)))
        applied = None
        fallback = sample.reason or "collecting"
        if consistent:
            nom, lim = self.scene.nominal_counter_z, float(a.get("max_dz_m", 0.20))
            lo, hi = (nom - self.wide_lo, nom + self.wide_hi) if self.wide else (nom - lim, nom + lim)   # counterz window
            applied = float(np.clip(est, lo, hi))
            self._ah_done, self._ah_committed = True, applied
            if commit_scene:
                self.scene.counter_z_runtime = applied
            fallback = "clamped" if applied != est else "none"
        elif len(window) == n_looks:
            fallback = f"inconsistent(spread={spread:.4f},mad={mad:.4f})"
        if not self._ah_done and self._ah_tries >= 3 * n_looks:
            self._ah_done, self._ah_committed = True, self.height_fallback()
            if self._ah_committed is not None and commit_scene:
                self.scene.counter_z_runtime = self._ah_committed
            fallback = "sample_gate_exhausted"
        shown = self.scene.nominal_counter_z if applied is None else applied
        self.log(f"[apply_height] est={est:.4f} applied={shown:.4f} n={len(window)}/{n_looks} "
                 f"spread={spread:.4f} mad={mad:.4f} tilt_deg={sample.tilt_deg:.2f} fallback={fallback}")
        return self._ah_done, self._ah_committed

    def update_counter_height(self, depth, frame=None):
        """Synchronous compatibility helper; policy preflight uses measure/accept in a worker + main thread.
        calib.counter_after_hand (handcal): height only after the camera pose is selected, via the shared tracker."""
        if self.counter_after_hand:
            if not self.camera_pose_selected or self.counter_tracker is None or frame is None:
                return None
            if self.counter_tracker.value is not None and not self.refit_per_look:
                return None
            try:
                update = self.counter_tracker.observe(
                    self.cal, self.cam, depth, self.current_T(), frame["timestamp"], frame["joints"],
                    frame.get("last_motion_time"))
            except Exception as exc:
                self.log(f"[counter_after_hand] rejected: {type(exc).__name__}: {exc}")
                return None
            if update.committed and update.counter_z is not None:
                self.scene.counter_z_runtime = float(update.counter_z)
            self.log(f"[counter_after_hand] accepted={int(update.accepted)} committed={int(update.committed)} "
                     f"z={update.counter_z} n={update.n_samples} tilt_deg={update.tilt_deg:.2f} "
                     f"rms_mm={update.residual_m * 1000.0:.2f} ({update.reason})")
            return update
        if not self.height_est or self._ah_done:
            return self._ah_committed
        out = self.accept_counter_height_sample(self.measure_counter_height(depth))[1]
        if self.wide and not self._ah_est:   # counterz: unmeasured look -> the configured prior (never the nominal z)
            prior = self.counter_z_prior()
            if prior is not None:
                self.scene.counter_z_runtime = prior
        return out

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

    def _counter_image_mask(self, shape):
        """Projected object-free near-mat band, used only by adaptive saturation sampling."""
        mat = np.asarray(self.scene.cfg["counter"]["mat_size"], float)
        object_near = float(self.scene.cfg["layout"]["region_y"][0]) - 0.5 * self.size
        far = max(0.02, object_near - 0.005)
        corners = self.scene.mat_to_pelvis(np.array([[0.0, 0.0, 0.0], [mat[0], 0.0, 0.0],
                                                     [mat[0], far, 0.0], [0.0, far, 0.0]]))
        uvz = self._project(corners)
        if not np.all(np.isfinite(uvz)) or np.any(uvz[:, 2] <= 0):
            return np.ones(shape, bool)
        poly = np.round(uvz[:, :2]).astype(np.int32)
        out = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(out, poly, 1)
        return out.astype(bool)

    def _adaptive_saturation_masks(self, hsv, keep):
        """Blue/purple mask with a counter-relative S threshold and the original H/V bands unchanged."""
        p = self.cfg.get("sat_floor_adaptive_params") or {}
        blue = self.hsv.get("blue") or []
        # The projected strip is deliberately in front of the object region,
        # so sample all of it. Excluding purple-hued counter pixels would hide
        # exactly the background competition this adaptive floor must measure.
        bg = (keep > 0) & self._counter_image_mask(hsv.shape[:2])
        sat = hsv[..., 1][bg]
        if len(sat):
            hist = np.bincount(sat, minlength=256)
            rank = int(np.ceil(float(p.get("bg_percentile", 90.0)) * hist.sum() / 100.0))
            bg_p = int(np.searchsorted(np.cumsum(hist), max(rank, 1), side="left"))
        else:
            bg_p = 0
        floor = float(np.clip(max(float(p.get("abs_min", 45.0)),
                                  bg_p + float(p.get("margin", 10.0))), 0.0, 255.0))
        self._sat_floors = (dict(self.color_gate_cfg.get("sat_floor_params") or {})
                            if self.sat_floor_fixed else {})
        self._sat_floors["blue"] = floor
        ranges = {name: list(vals) for name, vals in self.hsv.items()}
        ranges["blue"] = [([int(lo[0]), int(np.ceil(floor)), int(lo[2])], list(hi))
                           for lo, hi in blue]
        return self._masks_hsv(hsv, ranges)

    def _topface_hsv(self, px):
        """Configured hue vote; top35 remains the exact legacy path."""
        if self.topface_mode != "whole_blob":
            return top_face_hsv(px)
        px = np.asarray(px).reshape(-1, 3)
        top_n = max(1, int(np.ceil(0.35 * len(px))))
        if top_n < self.topface_min_px:
            med = np.median(px, axis=0)
            return float(med[0]), float(med[1]), float(med[2])
        return top_face_hsv(px)

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

    def detect(self, rgb, depth, debug=False, include_bins=False, arm_clear=True, robot_q=None, frame=None,
               calibration_frame=None, still_frame=False):
        """Detect one frame under one immutable transform/version when online calibration is enabled."""
        if not self.online_pitch_cal and frame is None:
            return self._detect_frame(rgb, depth, debug, include_bins, arm_clear, robot_q, calibration_frame,
                                      still_frame)
        with self.use_frame_transform(frame):
            return self._detect_frame(rgb, depth, debug, include_bins, arm_clear, robot_q, calibration_frame,
                                      still_frame)

    def _detect_frame(self, rgb, depth, debug=False, include_bins=False, arm_clear=True, robot_q=None,
                      calibration_frame=None, still_frame=False):
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
        self.bare_stats = {"glare_px": 0, "reflection": 0, "local_bg": 0}
        self.update_counter_height(depth, calibration_frame)    # inert when its flags are off
        self._image_bin_gate_active = False
        self._below_bin_row = None
        self._incomplete_rim = False
        self.bin_rim_hint = {}        # a missed frame must not keep the previous rim shift
        early_hsv = None
        rim_tape = None
        trusted = False
        if self.bins_from_image:
            self.bin_rims, trusted = None, self._live_bins_trusted()
            early_hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
            rims = self._detect_rims(early_hsv)
            if rims is not None:
                rim_tape = rims.tape if rims.tape is not None and rims.tape.any() else None
                if trusted or (not self.bins_img_cfg.get("with_depth", True) and self._depth_ok(depth)):
                    self._crosscheck_rims(rims)           # depth / runtime bin poses win; the image only cross-checks
                elif rims.row_limit is not None:
                    self._below_bin_row, self._below_bin_margin = rims.row_limit, 0.0
                    self._image_bin_gate_active = True
        if self.gate_below_bins and not self._image_bin_gate_active and not self._live_bins_trusted():
            if early_hsv is None:
                early_hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
            self._below_bin_row = self._observed_bin_bottom(early_hsv)
            self._below_bin_margin = float(self.gate_below_cfg.get("margin_px", 3))
            self._image_bin_gate_active = self._below_bin_row is not None
        # A partial rim set is not a bin row: publishing it turned off every 3D bin test, and the nominal
        # footprint does not contain a bin that has moved farther (a valid depth point there is a pick, -1).
        # Abstain above the nominal near edge — the closest the bins can be — and do not guess a shift.
        if (self._incomplete_rim and not self._image_bin_gate_active and not trusted and early_hsv is not None):
            near = self._nominal_near_row()
            if near is not None:
                self._below_bin_row = np.full(early_hsv.shape[1], near, np.float64)
                self._below_bin_margin = 0.0
                self._image_bin_gate_active = True
        keep = self._keep_mask(depth, exclude_bins=not include_bins)
        blocked_by_image_gate = None
        keep_before_image_gate = None
        if rim_tape is not None:
            keep[rim_tape > 0] = 0                          # rim tape is never an object
            if debug or self.reject_log:
                for area, cen in self.bin_rims.tape_blobs:
                    rejected.append((self._blob_placeholder("orange", area, cen), "rim_tape"))
        if self._image_bin_gate_active:
            margin = self._below_bin_margin
            below = np.arange(keep.shape[0])[:, None] > self._below_bin_row[None, :] + margin
            keep_before_image_gate = keep.astype(bool)
            blocked_by_image_gate = ~below
            keep[~below] = 0
        self._last_arm_mask = None
        arm = None
        arm_boundary = None
        if (self.fk_mask or self.depth_blob_recover) and robot_q is not None:   # the robot's own arms / hands are never objects
            arm = self._fk_arm_mask(keep.shape, *robot_q)
            if arm is not None:
                self._last_arm_mask = arm
                if self.fk_mask:   # depth_blob_recover alone must not FK-mask the colour blobs
                    keep[arm] = 0
                if self.partial_ok_near_hand:
                    r = max(1, int(self.partial_near_hand_cfg.get("boundary_px", 2)))
                    outer = cv2.dilate(arm.astype(np.uint8),
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
                    arm_boundary = (outer > 0) & ~arm
        depth_fk_failed = self.depth_blob_recover and robot_q is not None and arm is None
        # Preserve the old path exactly when both new consumers are off.  When
        # either is on, share one RGB->HSV conversion.
        need_hsv = (self.place_gate or self.colour_kmeans or self.hue_guard or self.sat_floor or self.sat_floor_adaptive
                    or self.depth_blob_recover or self.tape_value_cut or self.split_by_solidity or self.gate_below_bins
                    or self.glare_reject or self.reflection_reject or self.local_bg or self.bins_from_image)
        hsv = early_hsv if early_hsv is not None else (
            cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV) if need_hsv else None)
        colour_records = [] if self.depth_blob_recover else None
        if self.colour_kmeans and not self._colour_kmeans_done and arm_clear:
            kkeep = self._keep_mask(depth, exclude_bins=True) if include_bins else keep  # never fit on bin pixels
            self._fit_colour_kmeans(hsv, kkeep)
        if self.sat_floor_adaptive:
            masks = self._adaptive_saturation_masks(hsv, keep)
        else:
            masks = self._masks_hsv(hsv) if hsv is not None else self.masks(rgb)
        if self.glare_reject:   # baretable
            masks = self._glare_filter(masks, hsv, depth)
        all_masks = None
        if self.local_bg:       # baretable
            all_masks = np.zeros(keep.shape, np.uint8)
            for m_ in masks.values():
                all_masks |= (m_ > 0).astype(np.uint8)
        if self.flood_guard > 0.0 and any(float(m_.mean()) / 255.0 > self.flood_guard for m_ in masks.values()):
            self.log(f"[flood_guard] one colour class covers > {self.flood_guard:.0%} of the frame: no colour information, "
                     f"no detections")
            return ([], []) if debug else []
        if self.hue_guard:
            # H1: the red|yellow union is grouped and split before any colour is
            # assigned.  Each resulting part is then classified and guarded
            # solely from its own source pixels.
            for color, blob, area, centroid, mixed, tent in self._hue_guard_parts(masks, hsv, keep, depth,
                                                                                  arm_boundary):
                before = len(dets)
                self._detect_blob(color, blob, area, centroid, depth, include_bins, dets, rejected, hsv,
                                  hue_mixed=mixed, tentative=tent)
                if colour_records is not None and len(dets) > before:
                    colour_records.append((dets[-1], blob))
            items = ((color, mask) for color, mask in masks.items() if color not in ("red", "yellow"))
        else:
            items = masks.items()
        for color, m in items:
            if blocked_by_image_gate is not None and (debug or self.reject_log):
                self._append_fully_masked_rejections(
                    color, m & keep_before_image_gate, ~blocked_by_image_gate, "gate_below_bins", rejected)
            m = m & keep  # drop bin / counter pixels BEFORE grouping, so an object touching a same-colour bin survives
            k = int(c["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            skip = (self._reflection_skip(lab, stats, c["min_area_px"], hsv, depth)
                    if self.reflection_reject and n > 2 else ())
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
                ww, hh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
                win = np.s_[y0:y0 + hh, x0:x0 + ww]
                tentative = bool(arm_boundary is not None and
                                  np.any(arm_boundary[win] & (lab[win] == i)))
                floor = (int(self.partial_near_hand_cfg.get("min_area_px", 30))
                         if tentative else int(c["min_area_px"]))
                if area < floor:
                    if self.reject_log:
                        rejected.append((self._blob_placeholder(color, area, cent[i]), "min_area"))
                    continue
                if i in skip:
                    continue
                blob = (lab == i).astype(np.uint8)
                if all_masks is not None and self._local_bg_reject(blob, hsv, all_masks):   # baretable
                    continue
                parts = [(blob, area, cent[i])]
                if self.split_by_solidity:
                    lo, hi = self.split_solidity_cfg.get("area_px", [520, 15500])
                    if area < int(lo) or area > int(hi):
                        rejected.append((self._blob_placeholder(color, area, cent[i]),
                                         "split_by_solidity:area_low" if area < int(lo) else "split_by_solidity:area_high"))
                        continue
                    if not self._compact_single(blob):
                        parts = self.split_blob(blob, area, cent[i], depth, force=True,
                                                min_part_px=int(self.split_solidity_cfg.get("min_part_px", 250)))
                    elif self.split_blobs:
                        parts = self.split_blob(blob, area, cent[i], depth)
                elif self.split_blobs:
                    parts = self.split_blob(blob, area, cent[i], depth)
                # gatefix relabels split parts from their own pixels only under its flag (r22 keeps the merged label)
                relabel = self.split_by_solidity and len(parts) > 1
                for part, p_area, p_cent in parts:
                    part_color = self._part_colour(color, part, hsv) if relabel else color
                    if self.tape_value_cut and self._tape_like(part_color, part, hsv):
                        rejected.append((self._blob_placeholder(part_color, p_area, p_cent), "tape_value_cut"))
                        continue
                    part_tentative = tentative if part is blob else bool(
                        arm_boundary is not None and np.any(arm_boundary[win] & (part[win] > 0)))
                    before = len(dets)
                    self._detect_blob(part_color, part, p_area, p_cent, depth, include_bins, dets, rejected,
                                      hsv if (self.place_gate or self.hue_guard or self.sat_floor) else None,
                                      tentative=part_tentative)
                    if colour_records is not None and len(dets) > before:
                        colour_records.append((dets[-1], part))
        if self.depth_blob_recover and not depth_fk_failed:
            try:
                from .depth_recover import recover_depth_blobs
                recovered, depth_rejected, self.depth_recover_ms, stats = recover_depth_blobs(
                    self, rgb, depth, hsv, colour_records, arm)
                dets.extend(recovered)
                rejected.extend(depth_rejected)
                self.log(f"[depth_recover] components={stats['components']} matched={stats['matched']} "
                         f"unknown={stats['unknown']} accepted={stats['accepted']} plane={stats['plane']} "
                         f"elapsed_ms={self.depth_recover_ms:.2f}")
            except Exception as exc:   # optional recall path fails closed; the colour detections still pass through
                self.depth_recover_ms = 0.0
                self.log(f"[depth_recover] failed ({type(exc).__name__}: {exc}); colour-only")
        elif depth_fk_failed:
            self.depth_recover_ms = 0.0
            self.log("[depth_recover] skipped: FK arm/hand mask unavailable; colour-only")
        if self.hue_guard:
            dets, rejected = self._hue_guard(dets, rejected)
        if self.color_gate:
            dets = self._color_gate(dets, rejected)
        if self.reject_log:
            self._record_rejections(rejected)
        if self.still_vote and still_frame:
            dets = self._still_vote_frame(dets)
            if dets is None:
                return (None, rejected) if debug else None
        if self.novel_objects:
            try:
                from .novel_objects import augment_detections
                dets = augment_detections(self, rgb, depth, dets, robot_q=robot_q)
            except Exception as exc:
                # Novel perception is optional.  A bad frame/classifier must
                # retain the known, already safety-gated detections.
                self.log(f"[novel] perception failed closed: {type(exc).__name__}: {exc}")
        return (dets, rejected) if debug else dets

    def _blob_placeholder(self, color, area, centroid):
        """Small diagnostic-only detection for a candidate rejected before localisation."""
        det = self._from_plane(color, centroid, area, 0)
        return det if det is not None else Detection(color, np.full(3, np.nan), area, 0, "plane",
                                                     uv=(float(centroid[0]), float(centroid[1])))

    def _append_fully_masked_rejections(self, color, mask, survivor_mask, reason, rejected):
        """Record pre-gate components only when no candidate-sized portion survives the gate."""
        k = int(self.cfg["open_px"])
        m = np.asarray(mask, np.uint8)
        if k > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        surviving = np.bincount(labels[np.asarray(survivor_mask, bool)], minlength=n)
        area_floor = int(self.cfg["min_area_px"])
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= area_floor and int(surviving[i]) < area_floor:
                rejected.append((self._blob_placeholder(color, area, centroids[i]), reason))

    def _part_colour(self, fallback, blob, hsv):
        """Re-label a split part from its own pixels instead of inheriting the merged component's label."""
        if hsv is None or not np.any(blob):
            return fallback
        label = classify_hue(top_face_hsv(hsv[blob > 0])[0], self.hsv)
        return fallback if label is None else label

    def _compact_single(self, blob):
        """True when solidity and hull aspect both look like one object."""
        cnts, _ = cv2.findContours(np.asarray(blob, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return True
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(np.count_nonzero(blob)) / max(hull_area, 1.0)
        _, (w, h), _ = cv2.minAreaRect(hull)
        aspect = max(float(w), float(h)) / max(min(float(w), float(h)), 1e-6)
        return (solidity >= float(self.split_solidity_cfg.get("solidity", 0.82)) and
                aspect <= float(self.split_solidity_cfg.get("max_aspect", 2.0)))

    def _tape_like(self, color, blob, hsv):
        """Thin warm candidate without the configured bright saturated core. Hue never decides rejection."""
        if color not in ("red", "orange") or hsv is None:
            return False
        vv, uu = np.nonzero(blob)
        if len(uu) < 3:
            return True
        p = self.tape_value_cfg
        _, (w, h), _ = cv2.minAreaRect(np.c_[uu, vv].astype(np.float32))
        short, long = min(float(w), float(h)), max(float(w), float(h))
        thin = (long / max(short, 1e-6) >= float(p.get("min_aspect", 3.0)) or
                short <= float(p.get("max_thickness_px", 18)))
        px = hsv[blob > 0]
        core = (px[:, 1] >= float(p.get("min_s", 140))) & (px[:, 2] >= float(p.get("min_v", 140)))
        weak = float(core.mean()) < float(p.get("min_core_frac", 0.20))
        return bool(thin and weak)

    def _depth_ok(self, depth):
        if depth is None:
            return False
        dlo, dhi = self.cfg["depth_range_mm"]
        d = np.asarray(depth)
        return float(((d >= dlo) & (d <= dhi)).mean()) >= self.cfg["min_valid_frac"]

    def _nominal_bin_px(self):
        """[(name, centre_u, width_px, near_row)] of the nominal bins (near wall at counter height), by name."""
        out = []
        for n, g in bin_geometry(self.scene, {}).items():
            cx, cy = g["centre_mat"]
            h = np.asarray(self.scene.cfg["bins"]["outer_size"], float) / 2
            pts = np.array([[cx - h[0], cy - h[1], 0.0], [cx + h[0], cy - h[1], 0.0], [cx, cy - h[1], 0.0]])
            uvz = self._project(self.scene.mat_to_pelvis(pts))
            if np.any(uvz[:, 2] <= 0):
                continue
            out.append((n, float(uvz[2, 0]), float(abs(uvz[1, 0] - uvz[0, 0])), float(uvz[2, 1])))
        return out

    def _detect_rims(self, hsv):
        """bins_from_image: find the taped bin rims in the picture (rim_detect), name them, set the optional hint."""
        from . import rim_detect
        try:
            nom = self._nominal_bin_px()
        except ValueError:      # no extrinsics: nominal projection unavailable, detect without the width hint
            nom = []
        w_nom = float(np.median([w for _n, _u, w, _r in nom])) if nom else None
        res = rim_detect.detect_bin_rims(hsv, self.bins_img_cfg.get("detect"), expect_w_px=w_nom)
        if not res.bins:
            return None
        for rb in res.bins:
            if nom:
                uc = 0.5 * (rb.u0 + rb.u1)
                rb.name = min(nom, key=lambda t: abs(t[1] - uc))[0]
        params = dict(rim_detect.DEFAULTS)
        params.update(self.bins_img_cfg.get("detect") or {})
        names = [rb.name for rb in res.bins]
        # min_bins, and a unique cover of the nominal names.  Anything less is not published: detect()
        # abstains above the nominal near edge instead of treating the partial row as the whole frame.
        cover = len(res.bins) >= int(params["min_bins"]) and (
            not nom or (len(names) == len(set(names)) and set(names) == {t[0] for t in nom}))
        if not cover:
            self._incomplete_rim = True
            return None
        self.bin_rims = res
        self.bin_rim_hint = self._rim_hint(res) if self.bins_img_cfg.get("hint", False) else {}
        return res

    def _rim_hint(self, res):
        """Bin-distance estimate from each front-rim line: ray through the rim-line pixel at the bin centre column
        intersected with the counter plane (scene.counter_z_prior if present) gives the near edge's pelvis x; the
        hint is (dx, 0) against the nominal near edge.  Bins clipped by the image or with a ragged line are skipped."""
        sc = self.scene
        z_plane = float(getattr(sc, "counter_z_prior", None) or sc.counter_z) + float(
            self.bins_img_cfg.get("rim_z_m", 0.0))
        T = self._T()
        K = self.cal.cams[self.cam].K
        out = {}
        max_dx = float(self.bins_img_cfg.get("hint_max_m", 0.25))
        for rb in res.bins:
            if rb.name is None or rb.clipped or rb.resid_px > float(self.bins_img_cfg.get("hint_max_resid_px", 6.0)):
                continue
            u = 0.5 * (rb.u0 + rb.u1)
            d_cam = np.array([(u - K[0, 2]) / K[0, 0], (float(rb.v_at(u)) - K[1, 2]) / K[1, 1], 1.0])
            d = T[:3, :3] @ d_cam
            if d[2] >= -1e-6:
                continue
            p = T[:3, 3] + d * ((z_plane - T[2, 3]) / d[2])
            g = bin_geometry(self.scene, {}).get(rb.name)
            if g is None:
                continue
            half = float(np.asarray(sc.cfg["bins"]["outer_size"], float)[1]) / 2
            nominal_near_x = float(sc.mat_to_pelvis(np.array([g["centre_mat"][0], g["centre_mat"][1] - half, 0.0]))[0])
            dx = float(p[0]) - nominal_near_x
            if abs(dx) <= max_dx:
                out[rb.name] = (dx, 0.0)
        names = tuple(self.scene.bin_names)
        if not out or not names:
            return {}
        # Clipped side rims are skipped, so a uniform shift often yields a centre-only hint.  Applying that
        # to the centre alone moves one box and leaves the others.  Every nominal bin is shifted by the
        # median dx unless each bin already has its own hint.
        if set(names) <= set(out):
            return {n: out[n] for n in names}
        dx = float(np.median([v[0] for v in out.values()]))
        return {n: (dx, 0.0) for n in names}

    def _crosscheck_rims(self, res):
        """Runtime bin poses are trusted: keep them, only log where the image rim disagrees with the projected one."""
        try:
            nom = {n: r for n, _u, _w, r in self._nominal_bin_px()}
        except ValueError:
            return
        for rb in res.bins:
            if rb.name in nom:
                dv = float(rb.v_at(0.5 * (rb.u0 + rb.u1))) - nom[rb.name]
                if abs(dv) > float(self.bins_img_cfg.get("crosscheck_px", 12.0)):
                    self.log(f"[bins_from_image] {rb.name}: rim line {dv:+.0f} px from the trusted projected rim")

    def _rim_column_unknown(self, u):
        """True when the 3D bin test must still run for this image column.

        The image gate replaces that test only where ``row_limit`` is a measured rim row.  The -1e6 sentinel
        (and a column off the image) means there is no rim, so guessing that the pixel is "below the bin"
        would accept an object that is already inside one.  Flag off: the gate is inactive and this is True,
        which is the stock 3D test.
        """
        if not self._image_bin_gate_active:
            return True
        return bool(self._rim_columns_unknown(np.asarray([np.nan if u is None else u], float))[0])

    def _rim_columns_unknown(self, u):
        row = self._below_bin_row
        u = np.asarray(u, float)
        if not self._image_bin_gate_active or row is None:
            return np.ones(u.shape, bool)
        finite = np.isfinite(u)
        ui = np.clip(np.where(finite, np.round(u), 0), 0, len(row) - 1).astype(int)
        return ~finite | (u < 0) | (u >= len(row)) | (row[ui] <= -1e5)

    def _nominal_near_row(self):
        """Image row of the nominal near edge (the closest the bin row can be). None if it does not project."""
        try:
            nom = self._nominal_bin_px()
        except ValueError:
            return None
        rows = [r for _n, _u, _w, r in nom]
        return float(np.median(rows)) if rows else None

    def _live_bins_trusted(self):
        """The image gate is unnecessary only when all live bin positions have trusted evidence."""
        if not self.live_bins:
            return False
        if self.calibration is not None and getattr(self.calibration, "success", False):
            return True
        src = self.bin_shift_src
        shift = dict(getattr(src, "bin_shift", None) or {})
        conf = dict(getattr(src, "bin_conf", None) or {})
        names = tuple(self.scene.bin_names)
        floor = float(self.gate_below_cfg.get("trusted_bin_conf", 0.5))
        return bool(names) and all(n in shift and float(conf.get(n, 0.0)) >= floor for n in names)

    def _observed_bin_bottom(self, hsv):
        """Lowest observed bin-mass row by column, with the median supported column as fallback."""
        p = self.gate_below_cfg
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        warm = (((h <= int(p.get("h_max", 17))) | (h >= 170)) &
                (s >= int(p.get("s_min", 77))) & (v >= int(p.get("v_min", 50))))
        bright_object = (s >= int(p.get("object_s_min", 140))) & (v >= int(p.get("object_v_min", 150)))
        m = (warm & ~bright_object).astype(np.uint8)
        k = int(p.get("open_px", 5))
        if k > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m, bool)
        min_comp = int(p.get("min_component_px", 2000))
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_comp:
                keep |= lab == i
        if int(keep.sum()) < int(p.get("min_mass_px", 18000)):
            return None
        supported = keep.any(axis=0)
        rows = np.full(keep.shape[1], np.nan)
        rows[supported] = keep.shape[0] - 1 - np.argmax(keep[::-1, supported], axis=0)
        fallback = float(np.nanmedian(rows))
        rows[~np.isfinite(rows)] = fallback
        return rows

    @staticmethod
    def _reject_stage(reason):
        if reason.startswith("height "):
            return "height"
        if reason.startswith("in_bin:"):
            return "in_bin"
        return reason

    def _record_rejections(self, rejected):
        frame = {}
        for det, reason in rejected:
            color = getattr(det, "color", "unknown")
            stage = self._reject_stage(str(reason))
            self.reject_tally.setdefault(color, Counter())[stage] += 1
            frame.setdefault(color, Counter())[stage] += 1
        payload = " ".join(f"{color}=" + ",".join(f"{stage}:{n}" for stage, n in sorted(counts.items()))
                           for color, counts in sorted(frame.items()))
        self.log(f"[reject_log] {payload or 'none'}")

    def rejection_table(self):
        """Plain dictionaries suitable for diagnostics tools and JSON."""
        return {color: dict(counts) for color, counts in sorted(self.reject_tally.items())}

    def _still_vote_frame(self, dets):
        """Consume one still frame; return a voted list only when the configured window is complete."""
        n = int(self.still_vote_cfg.get("n_frames", 3))
        k = int(self.still_vote_cfg.get("min_hits", 2))
        radius = float(self.still_vote_cfg.get("match_m", 0.02))
        self._still_frames.append(list(dets))
        if len(self._still_frames) < n:
            return None
        frames, self._still_frames = self._still_frames[:n], []
        tracks = []
        for fi, frame in enumerate(frames):
            for det in frame:
                best, best_d = None, radius
                for track in tracks:
                    if track["color"] != det.color or fi in track["frames"]:
                        continue
                    centre = np.median([d.top for d in track["dets"]], axis=0)
                    dist = float(np.linalg.norm(det.top[:2] - centre[:2]))
                    if dist < best_d:
                        best, best_d = track, dist
                if best is None:
                    best = {"color": det.color, "frames": set(), "dets": []}
                    tracks.append(best)
                best["frames"].add(fi)
                best["dets"].append(det)
        voted = []
        for track in tracks:
            if len(track["frames"]) < k:
                continue
            ds = track["dets"]
            rep = max(ds, key=lambda d: d.n_px)
            top = np.median([d.top for d in ds], axis=0)
            uv = tuple(np.median([d.uv for d in ds], axis=0))
            voted.append(replace(rep, top=np.asarray(top, float), uv=uv,
                                 n_px=int(np.median([d.n_px for d in ds]))))
        return voted

    def _counter_heights(self, ys, xs, depth, n=12):
        """Median object-heights above the counter of up to n valid-depth pixels, None = no valid depth / any failure."""
        if depth is None or np.ndim(depth) != 2 or not len(ys):
            return None
        try:
            dlo, dhi = self.cfg["depth_range_mm"]
            sel = np.linspace(0, len(ys) - 1, min(n * 4, len(ys))).astype(int)
            d = np.asarray(depth)[ys[sel], xs[sel]].astype(float)
            ok = (d >= dlo) & (d <= dhi)
            if not ok.any():
                return None
            P = self._pixel_to_pelvis(xs[sel][ok][:n], ys[sel][ok][:n], d[ok][:n])
            return float(np.median((P[:, 2] - self.scene.counter_z) / self.height))
        except Exception:
            return None

    def _glare_filter(self, masks, hsv, depth):
        """perception.glare_reject: remove small specular highlights on the counter plane (and their low-S halo) from the
        colour masks. Fails open: any error returns the masks unchanged."""
        g = self.glare_cfg
        try:
            hmax = float(g.get("plane_h_max", 0.5))

            def on_plane(ys, xs):
                h = self._counter_heights(ys, xs, depth)
                return h is None or h < hmax
            kill = bare_table.glare_kill_mask(hsv, int(g.get("v_min", 245)), int(g.get("s_max", 40)),
                                              int(g.get("max_px", 900)), int(g.get("dilate_px", 6)),
                                              int(g.get("halo_s_max", 65)), on_plane)
        except Exception as exc:
            self.log(f"[glare_reject] error {type(exc).__name__}: {exc}")
            return masks
        if kill is None:
            return masks
        self.bare_stats["glare_px"] = int(kill.sum())
        keep = (kill == 0).astype(np.uint8) * 255
        return {c: m & keep for c, m in masks.items()}

    def _reflection_skip(self, lab, stats, min_area, hsv, depth):
        """perception.reflection_reject: component ids (of one colour) that mirror a stronger blob above them."""
        try:
            ids = [i for i in range(1, len(stats)) if stats[i, cv2.CC_STAT_AREA] >= min_area]
            if len(ids) < 2:
                return ()
            r = self.reflect_cfg
            p = {"max_gap_px": float(r.get("max_gap_px", 40)), "dx_frac": float(r.get("dx_frac", 0.6)),
                 "width_ratio": tuple(r.get("width_ratio", (0.5, 1.8))), "s_ratio": float(r.get("s_ratio", 0.9)),
                 "min_iou": float(r.get("min_iou", 0.45)), "h_max": float(r.get("h_max", 0.5))}
            blobs = []
            for i in ids:
                x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                blobs.append((lab[y:y + h, x:x + w] == i, x, y))
            bad = bare_table.reflection_indices(blobs, hsv, p, lambda ys, xs: self._counter_heights(ys, xs, depth))
            self.bare_stats["reflection"] += len(bad)
            return {ids[k] for k in bad}
        except Exception as exc:
            self.log(f"[reflection_reject] error {type(exc).__name__}: {exc}")
            return ()

    def _local_bg_reject(self, blob, hsv, all_masks):
        """perception.local_bg: blob whose median H/S equals its surrounding ring (the table) is not an object."""
        try:
            l = self.local_bg_cfg
            if bare_table.low_contrast_with_ring(hsv, blob, all_masks, int(l.get("ring_px", 10)),
                                                 float(l.get("dh", 5)), float(l.get("ds", 25))):
                self.bare_stats["local_bg"] += 1
                return True
        except Exception as exc:
            self.log(f"[local_bg] error {type(exc).__name__}: {exc}")
        return False

    # ------------------------------------------------------------------ grasp.pregrasp_relocalise
    def relocalise(self, rgb, depth, prior, robot_q=None, gate_m=0.06, occluded_frac=0.75, hidden_frac=0.6,
                   min_visible_frac=0.15, arm_dilate_px=6):
        """Re-detect `prior` (a first-look Detection) near its predicted position, with the hand over it.

        Returns {"kind": "found" | "gone" | "hidden" | "unsupported", ...}. "found": top (3,), yaw (or None), n_px,
        occluded, ratio (visible / first-look area). "gone": nothing of that colour inside the gate although the
        predicted footprint is in view (the object is not there). "hidden": the footprint is under the robot's own
        arm / hand (or only a sliver is visible): no verdict, the caller keeps its plan.

        Same pipeline as detect() (HSV class of the prior only, depth keep mask, FK arm mask, _detect_blob), so the
        estimate carries the same xy_offset / y_affine as the first look. A blob cut by the hand is completed with
        the known object footprint: the visible top points are re-intersected with a plane fitted to them (depth
        noise along the ray drops out), the side that is not cut gives the true edge, the cut side is not trusted."""
        import cv2
        color = prior.color
        if depth is None or color not in self.hsv:
            return {"kind": "unsupported"}
        c = self.cfg
        keep = self._keep_mask(depth, exclude_bins=True)
        arm = None
        if robot_q is not None:
            arm = self._fk_arm_mask(keep.shape, *robot_q)
            if arm is not None:
                if arm_dilate_px > 0:
                    k = 2 * int(arm_dilate_px) + 1
                    arm = cv2.dilate(arm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
                keep[arm] = 0
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        m = self._masks_hsv(hsv, {color: self.hsv[color]})[color] & keep
        k = int(c["open_px"])
        if k > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        prior_px = max(float(prior.n_px), 1.0)
        min_px = max(20, int(0.12 * prior_px))
        pred = np.asarray(prior.top[:2], float)
        frags = []   # (det, blob, area, centroid) of every colour fragment inside the gate
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_px:
                continue
            blob = (lab == i).astype(np.uint8)
            parts = self.split_blob(blob, area, cent[i], depth) if self.split_blobs else [(blob, area, cent[i])]
            for part, p_area, p_cent in parts:
                dets, rej = [], []
                self._detect_blob(color, part, p_area, p_cent, depth, False, dets, rej, None)
                for d in dets:
                    if np.linalg.norm(d.top[:2] - pred) <= gate_m:
                        frags.append((d, part, p_area, p_cent))
        if not frags:
            return {"kind": "hidden" if self._footprint_covered(prior, arm, hidden_frac) else "gone"}
        seed = min(frags, key=lambda f: np.linalg.norm(f[0].top[:2] - pred))
        blob, area = seed[1].copy(), seed[2]
        for f in frags:   # fragments of one object cut by the hand: same colour, within one object size of the seed
            if f is seed:
                continue
            if np.linalg.norm(f[0].top[:2] - seed[0].top[:2]) <= 1.2 * self.size and area + f[2] <= 1.3 * prior_px:
                blob |= f[1]
                area += f[2]
        det = seed[0]
        if len(frags) > 1 and area != seed[2]:
            dets, rej = [], []
            ys, xs = np.nonzero(blob)
            self._detect_blob(color, blob, area, (float(xs.mean()), float(ys.mean())), depth, False, dets, rej, None)
            if not dets:
                return {"kind": "gone"}
            det = dets[0]
        ratio = float(area) / prior_px
        if ratio < min_visible_frac and self._footprint_covered(prior, arm, hidden_frac):
            return {"kind": "hidden", "ratio": ratio}
        touches = False
        cut = None
        if arm is not None:
            near_arm = cv2.dilate(arm.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            cut = (blob > 0) & near_arm
            touches = bool(cut.any())
        occluded = ratio < occluded_frac or touches
        top, yaw = det.top.copy(), det.yaw
        if occluded and det.method == "depth" and det.shape != "triangle" and prior.shape != "triangle":
            fit = self._complete_top_face(blob, cut, depth, prior)
            if fit is not None:
                top[:2], yaw = fit, None   # yaw of a cut face is not observable: the planner keeps the first-look yaw
        return {"kind": "found", "top": top, "yaw": yaw, "n_px": int(area), "occluded": bool(occluded),
                "ratio": ratio, "det": det}

    def _footprint_covered(self, prior, arm, frac):
        """True when the predicted object footprint (corners + centre at the top plane) lies mostly under the arm
        mask or outside the image: the camera cannot see the object, so absence is no evidence."""
        if arm is None:
            return False
        h = self.size / 2
        pts = np.array([[dx, dy] for dx in (-h, 0.0, h) for dy in (-h, 0.0, h)]) + np.asarray(prior.top[:2], float)
        P = np.c_[pts, np.full(len(pts), prior.top[2])]
        uvz = self._project(P)
        H, W = arm.shape
        hid = 0
        for u, v, z in uvz:
            if not (np.isfinite(u) and np.isfinite(v)) or z <= 0:
                hid += 1
                continue
            ui, vi = int(round(u)), int(round(v))
            hid += int(not (0 <= ui < W and 0 <= vi < H) or bool(arm[vi, ui]))
        return hid / len(uvz) >= frac

    def _complete_top_face(self, blob, cut, depth, prior):
        """Centre xy of an object whose top face is partly hidden. Visible top points -> plane fit -> their rays
        re-intersected with that plane -> extents along the object axes (first-look yaw). An axis whose visible
        extent is a full object width gives the centre directly. A short axis is cut on one side: the uncut side is a
        true edge (centre = that edge -/+ size/2); if the cut cannot be told apart, the centre inside the feasible
        interval nearest the first-look position (no correction beyond what the visible part proves)."""
        import cv2
        c = self.cfg
        e = int(c["erode_px"])
        core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
        if core.sum() < 0.25 * blob.sum():
            core = blob
        v, u = np.nonzero(core)
        d = depth[v, u].astype(float)
        dlo, dhi = c["depth_range_mm"]
        ok = (d >= dlo) & (d <= dhi)
        if ok.sum() < 30:
            return None
        u, v, d = u[ok], v[ok], d[ok]
        P = self._pixel_to_pelvis(u, v, d)
        z_top = np.percentile(P[:, 2], c["top_percentile"])
        top = P[:, 2] >= z_top - c["top_band"]
        if top.sum() < 20:
            return None
        Pt, ut, vt = P[top], u[top], v[top]
        A = np.c_[Pt[:, 0], Pt[:, 1], np.ones(len(Pt))]
        try:
            (a, b, c0), *_ = np.linalg.lstsq(A, Pt[:, 2], rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not (abs(a) < 0.1 and abs(b) < 0.1):    # a tilted fit on a sliver: horizontal plane at the median top z
            a, b, c0 = 0.0, 0.0, float(np.median(Pt[:, 2]))
        T = self._T()
        K = self.cal.cams[self.cam].K
        ray = np.c_[(ut - K[0, 2]) / K[0, 0], (vt - K[1, 2]) / K[1, 1], np.ones(len(ut))] @ T[:3, :3].T
        o = T[:3, 3]
        den = ray[:, 2] - a * ray[:, 0] - b * ray[:, 1]
        good = den < -1e-6
        if good.sum() < 20:
            return None
        s = (c0 + a * o[0] + b * o[1] - o[2]) / den[good]
        xy = (o + s[:, None] * ray[good])[:, :2] + self.xy_offset
        if self.y_affine is not None:
            xy = np.array([self._y_affine(p) for p in xy])
        yaw = prior.yaw if prior.yaw is not None else 0.0
        cs, sn = np.cos(yaw), np.sin(yaw)
        R = np.array([[cs, sn], [-sn, cs]])          # world -> object axes
        q = xy @ R.T
        pc = np.asarray(prior.top[:2], float) @ R.T
        cut_q = None
        if cut is not None and cut[vt[good], ut[good]].any():
            # top points next to the arm mask: where the object is cut
            cq = q[cut[vt[good], ut[good]]]
            cut_q = cq.mean(0)
        half = self.size / 2
        tol = 0.012
        out = np.zeros(2)
        for ax in range(2):
            lo, hi = np.percentile(q[:, ax], [1, 99])
            if hi - lo >= self.size - tol:
                out[ax] = 0.5 * (lo + hi)
                continue
            lo_c, hi_c = hi - half, lo + half        # feasible centre interval given the visible extent
            if lo_c > hi_c:
                out[ax] = 0.5 * (lo + hi)
                continue
            mid = 0.5 * (lo + hi)
            if cut_q is not None and abs(cut_q[ax] - mid) > max(0.1 * (hi - lo), 0.001):
                out[ax] = lo + half if cut_q[ax] > mid else hi - half   # cut on the high side -> the low edge is true
            elif hi - lo < 0.35 * self.size:
                return None   # a sliver of the top face and no way to tell which side is cut: not observable
            else:
                out[ax] = float(np.clip(pc[ax], lo_c, hi_c))
        return out @ R   # object axes -> world (R orthonormal)

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
            top = self._topface_hsv(hsv[comp])
            label = classify_hue(top[0], self.hsv)
            if label is None:
                label = "red" if (masks["red"][comp] > 0).sum() >= (masks["yellow"][comp] > 0).sum() else "yellow"
            items.append((label, comp.astype(np.uint8) * 255))
        return items + [(c, m) for c, m in masks.items() if c not in ("red", "yellow")]

    def _hue_guard_parts(self, masks, hsv, keep, depth, arm_boundary=None):
        """H1 red/yellow candidates, split before label assignment.

        A part retaining pixels from both disjoint source masks is tagged mixed;
        `_hue_guard` abstains it even when its brightest pixels look decisive.
        """
        import cv2
        red = masks["red"] & keep
        yellow = masks["yellow"] & keep
        union = (red | yellow).astype(np.uint8)
        k = int(self.cfg["open_px"])
        if k > 0:
            union = cv2.morphologyEx(union, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(union, connectivity=8)
        out = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            blob = (lab == i).astype(np.uint8)
            # recall partial_ok_near_hand: an FK-boundary fragment gets the lower floor and is marked tentative
            tentative = bool(arm_boundary is not None and np.any(arm_boundary & (blob > 0)))
            floor = (int(self.partial_near_hand_cfg.get("min_area_px", 30)) if tentative
                     else int(self.cfg["min_area_px"]))
            if area < floor:
                continue
            parts = self.split_blob(blob, area, cent[i], depth) if self.split_blobs else [(blob, area, cent[i])]
            for part, p_area, p_cent in parts:
                use = part > 0
                part_tentative = bool(arm_boundary is not None and np.any(arm_boundary & use))
                top = self._topface_hsv(hsv[use])   # top35 = legacy; recall topface_mode whole_blob fallback
                label = classify_hue(top[0], self.hsv)
                nr, ny = int(np.count_nonzero(red[use])), int(np.count_nonzero(yellow[use]))
                if label not in ("red", "yellow"):
                    label = "red" if nr >= ny else "yellow"
                out.append((label, part, p_area, p_cent, nr > 0 and ny > 0, part_tentative))
        return out

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
            if d.hue_mixed:
                label, reason = None, "mixed"
            if reason is None:
                if label != d.color:
                    n_relabel += 1
                    old = d
                    d = replace(d, color=label)
                    self._copy_tentative(old, d)
                keep.append(d)
                continue
            n_amb += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            amb = replace(d, color_conf=AMBIGUOUS_CONF)
            self._copy_tentative(d, amb)
            if self.place_gate:   # the place gate (colour_min > 0.3) keeps it off the bins; it stays on the table
                keep.append(amb)
            else:
                moved.append((amb, "hue_guard:low_s_red" if reason == "low_s_red" else "hue_guard:ambiguous"))
        self.log(f"[hue_guard] cands={len(cands)} ambiguous={n_amb} relabelled={n_relabel} "
                 f"moved_rejected={len(moved)} reasons={reasons}")
        return keep, rejected + moved

    def _detect_blob(self, color, blob, area, centroid, depth, include_bins, dets, rejected, hsv=None,
                     hue_mixed=False, tentative=False):
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
        if det is not None and self.tri_vertex_align and det.shape == "triangle":
            # The pose estimator is preferred by the planner. This rectangle fallback keeps the feature useful when
            # only a standing triangle label and colour blob survive depth filtering.
            det.vertex_yaw = self._blob_vertex_yaw(blob, float(det.top[2]))
        if det is not None and hsv is not None and self.hue_guard:
            det.hsv_med = self._topface_hsv(hsv[blob > 0])   # configured top-face / small-blob hue vote
            det.hue_mixed = bool(hue_mixed)
        if det is not None and hsv is not None and self.sat_floor:
            det.sat_med = float(np.median(hsv[blob > 0][:, 1]))   # color_gate.sat_floor: shadow / sheen has low S
        if det is not None and hsv is not None and self.place_gate:
            try:
                det.color_conf = colour_confidence(hsv, blob, self.hsv, color, c["min_area_px"])
            except Exception as exc:  # never raise into act()
                det.color_conf = 0.0
                self.log(f"[gate] colour_confidence failed for {color}: {type(exc).__name__}: {exc}")
        if det is not None and tentative:
            det.tentative = True
            det.tentative_reason = "fk_mask_boundary"
        why = self._reject(det, bins_first=include_bins)
        if self.shape_diag and det is not None and det.shape_diag is not None:
            self._log_diag(det, why)
        if why:
            rejected.append((det, why))
        else:
            dets.append(det)

    def _blob_vertex_yaw(self, blob, z):
        """Pelvis-frame direction of a triangle vertex line from the blob's minimum-area rectangle, modulo 120 deg.

        For a triangle, the rectangle's shorter axis is an undirected altitude/vertex line. Two points on that pixel
        axis are ray-intersected with the horizontal object-top plane before measuring yaw, so camera tilt does not
        turn an image angle into a robot-frame angle. None is the safe fallback for a malformed/tiny blob.
        """
        try:
            import cv2
            m = (np.asarray(blob) > 0).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                return None
            (u, v), (w, h), angle_deg = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
            if min(float(w), float(h)) <= 1.0:
                return None
            # OpenCV's angle follows the width axis. Select the rectangle's shorter axis: a triangle altitude.
            a = np.radians(float(angle_deg) + (0.0 if w <= h else 90.0))
            duv = 20.0 * np.array([np.cos(a), np.sin(a)])
            T, K = self._T(), self.cal.cams[self.cam].K

            def hit(uv):
                ray_opt = np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
                ray = T[:3, :3] @ ray_opt
                if abs(ray[2]) <= 1e-9:
                    return None
                return T[:3, 3] + ((z - T[2, 3]) / ray[2]) * ray

            p0, p1 = hit(np.array([u, v]) - duv), hit(np.array([u, v]) + duv)
            if p0 is None or p1 is None or not np.all(np.isfinite([p0, p1])):
                return None
            d = p1[:2] - p0[:2]
            if np.linalg.norm(d) <= 1e-9:
                return None
            return float(np.arctan2(d[1], d[0]) % (2.0 * np.pi / 3.0))
        except Exception:
            return None

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

    def split_blob(self, blob, area, centroid, depth, force=False, min_part_px=None):
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
        if not force and area <= float(s.get("area_factor", 1.5)) * exp:
            return whole
        x, y, w, h = cv2.boundingRect(blob)
        crop = np.ascontiguousarray(blob[y:y + h, x:x + w])
        dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
        dmax = float(dist.max())
        min_part = (float(min_part_px) if force and min_part_px is not None
                    else float(s.get("min_part_frac", 0.3)) * exp)
        for frac in s.get("peak_frac", [0.5, 0.6, 0.7, 0.8]):
            n, lab = cv2.connectedComponents((dist >= frac * dmax).astype(np.uint8), connectivity=8)
            if n < 3:
                continue
            # at most as many seeds as objects the blob can hold (strongest peaks), decided BEFORE any per-seed work
            k_max = min(int(s.get("max_parts", 3)), max(2 if force else 1, int(round(area / exp))))
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
        if evals[0] <= 0 or (not force and np.sqrt(evals[1] / evals[0]) < float(s.get("min_aspect", 1.4))):
            return whole
        if force:
            # Deterministic coordinate k-means. Farthest projections give stable seeds and two compact parts even
            # when a diagonal contact has no clean distance-transform neck.
            axis = evecs[:, 1]
            proj = (pts - pts.mean(0)) @ axis
            centres = np.array([pts[np.argmin(proj)], pts[np.argmax(proj)]], float)
            idx = np.zeros(len(pts), int)
            for _ in range(8):
                idx = np.argmin(((pts[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)
                if any(not np.any(idx == j) for j in range(2)):
                    return whole
                new = np.array([pts[idx == j].mean(axis=0) for j in range(2)])
                if np.allclose(new, centres):
                    break
                centres = new
            parts = []
            for j in range(2):
                p = np.zeros_like(crop, bool)
                p[vv[idx == j], uu[idx == j]] = True
                parts.append(p)
            if min(int(p.sum()) for p in parts) < min_part:
                return whole
            return self._parts_out(blob, parts, x, y, w, h)
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
        if not shift and self.bin_rim_hint and self.bins_from_image and self.bins_img_cfg.get("hint", False):
            shift = dict(self.bin_rim_hint)   # image rim estimate, only without any runtime estimate
        return bin_geometry(self.scene, shift)

    def _bin_centres_mat(self):
        """{bin: mat-frame centre}: nominal scene.yaml centres + the runtime shift (see bin_geometry)."""
        return {n: g["centre_mat"] for n, g in self.bin_geometry().items()}

    def _keep_mask(self, depth, exclude_bins=True):
        """uint8 mask of pixels that may belong to an object: no valid depth (unknown), or a 3D point at object
        height over the mat and outside every bin footprint."""
        c, sc = self.cfg, self.scene
        shape = depth.shape if depth is not None else (480, 640)
        keep = (self._bin_hulls_mask(shape) if exclude_bins and not self._image_bin_gate_active
                else np.full(shape, 255, np.uint8))
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
        if exclude_bins and not self._image_bin_gate_active:
            half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
            for ctr in self._bin_centres_mat().values():
                ok &= ~np.all(np.abs(pm[:, :2] - ctr) <= half, axis=1)
        elif exclude_bins:
            # Image gate is on, but a sentinel row_limit (-1e6) means this column has no rim.  Keep the 3D
            # footprint there; every other column was already limited by the rim line.
            unknown = self._rim_columns_unknown(uu)
            if np.any(unknown):
                half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
                inside = np.zeros(len(uu), bool)
                for ctr in self._bin_centres_mat().values():
                    inside |= np.all(np.abs(pm[:, :2] - ctr) <= half, axis=1)
                ok &= ~(unknown & inside)
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
        if self.prism_orientation:
            self._apply_prism_orientation(det, P, top)
        if self.generic_fallback:   # flag off: no call, no work
            self._apply_generic_fallback(det, conf, blob, frame_depth)
        if self.shape_diag:
            nb = bool(near_bin) if (self.classify_shape and self.near_bin_guard) else _safe_near_bin(self, top)
            det.shape_diag = self._diag(det, "depth", top[:, :2], blob, conf, path, nb)
        return det

    def _apply_prism_orientation(self, det, P, top):
        """perception.prism_orientation: sets det.prism_orientation (+ ridge_yaw / prism_xy / prism_conf for a lying prism).
        A lying prism is a triangle prism whatever the rectangular top face said: shape -> "triangle", path "lying" (so
        the demo posture, which is for standing prisms, is not chosen). Unsure / error: det untouched."""
        try:
            p = self.prism_params
            length = float(p.get("length_m") or self.height)
            dims = self.scene.object_dims("triangle")
            r = prism_orientation(P, top, self.scene.counter_z, 3.0 * dims["apothem"], dims["side"], length,
                                  float(self.cfg["top_band"]), float(self.cfg["top_percentile"]),
                                  int(self.cfg["min_valid_depth_px"]), float(p.get("fill_min", 0.75)))
        except Exception as exc:   # never raise into act()
            self.log(f"[prism] orientation failed: {type(exc).__name__}: {exc}")
            return
        if r is None:
            return
        det.prism_orientation, det.prism_conf = r["orientation"], r["conf"]
        if r["orientation"] != "lying":
            return
        xy = np.asarray(r["xy"], float) + self.xy_offset
        if self.y_affine is not None:
            xy = self._y_affine(xy)
        det.ridge_yaw, det.prism_xy = r["ridge_yaw"], (float(xy[0]), float(xy[1]))
        det.shape, det.shape_conf, det.shape_path = "triangle", r["conf"], "lying"

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
        z_aim = z_top - float(self.cfg.get("aim_z_frac", 0.25)) * self.height
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
        u = None if det.uv is None else det.uv[0]
        if self._rim_column_unknown(u):
            for name, ctr in self._bin_centres_mat().items():
                if np.all(np.abs(pm[:2] - ctr) <= half):
                    return f"in_bin:{name}"
        if bins_first and not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        return None

    def _fk_arm_mask(self, shape, arm_q, hand_q=None):
        """FK arm/hand mask from filled per-link silhouettes at the current camera pose.

        Surface samples are 5 mm apart, so marking their projected pixels alone leaves coloured robot pixels between
        samples.  A convex hull per link fills those gaps without bridging the two arms (or separate fingers), then
        the flag-param ``fk_dilate_px`` covers the rendered/real RGB-to-FK edge residual.  None on any failure keeps
        the historical fail-open behaviour of this optional gate.
        """
        import cv2
        try:
            if self._arm_surface_error is not None:
                return None
            if self._arm_surface is None:
                from .hand_calib import ArmSurface
                self._arm_surface = ArmSurface()
            h, w = shape
            m = np.zeros(shape, np.uint8)
            r = int(self.color_gate_cfg.get("fk_dilate_px", 0))
            fill = self.fk_silhouette or r > 0   # gatefix fk_silhouette, or fixfk's dilation-coupled per-link fill
            for P, _ in self._arm_surface.posed_links(arm_q, hand_q):
                uvz = self._project(P)
                ok = np.all(np.isfinite(uvz), axis=1) & (uvz[:, 2] > 0)
                uv = np.round(uvz[ok, :2]).astype(np.int32)
                uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
                if fill and len(uv) >= 3:
                    cv2.fillConvexPoly(m, cv2.convexHull(uv), 1)
                elif len(uv):
                    m[uv[:, 1], uv[:, 0]] = 1
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
        if self.sat_floor and self._color_gate_sat_low(det, g):
            return "color_gate:sat_floor"
        Z = self._color_gate_depth(det.top)
        if Z is None:
            if not getattr(det, "tentative", False) and det.n_px < self._visible_expected_area(det, float(g["min_px_no_z"])):
                return "color_gate:speckle_no_z"
            if g.get("max_px_no_z") is not None and det.n_px > g["max_px_no_z"]:
                return "color_gate:too_big_no_z"
        elif self.area_floor_geometric:
            p = self.cfg.get("area_floor_geometric_params") or {}
            prism = self._projected_top_area_px(det.top, self.size * self.size / np.sqrt(3.0))
            cube = self._projected_top_area_px(det.top, self.size * self.size)
            if prism is None or cube is None:
                if not getattr(det, "tentative", False) and det.n_px < g["min_px_no_z"]:
                    return "color_gate:speckle_no_z"
                if g.get("max_px_no_z") is not None and det.n_px > g["max_px_no_z"]:
                    return "color_gate:too_big_no_z"
            else:
                if not getattr(det, "tentative", False) and det.n_px < float(p.get("min_frac", 0.35)) * prism:
                    return "color_gate:speckle"
                if det.n_px > float(p.get("max_factor", 2.0)) * cube:
                    return "color_gate:too_big"
        else:
            K = self.cal.cams[self.cam].K
            expected = float(K[0, 0]) * float(K[1, 1]) * (self.size / Z) ** 2   # one object's top face, px
            speckle_expected = self._visible_expected_area(det, expected)
            if not getattr(det, "tentative", False) and det.n_px < g["area_frac"] * speckle_expected:
                return "color_gate:speckle"
            if g.get("max_area_frac") is not None and det.n_px > g["max_area_frac"] * expected:
                return "color_gate:too_big"
        return self._color_gate_workspace(det, g) or self._color_gate_bin(det, g)

    def _visible_expected_area(self, det, expected):
        """Expected object pixels outside the dilated arm silhouette; masked pixels are not missing evidence."""
        if not self.fk_silhouette or self._last_arm_mask is None:
            return expected
        side = max(1.0, float(np.sqrt(expected)))
        u, v = det.uv
        h, w = self._last_arm_mask.shape
        x0, x1 = max(0, int(np.floor(u - side / 2))), min(w, int(np.ceil(u + side / 2)))
        y0, y1 = max(0, int(np.floor(v - side / 2))), min(h, int(np.ceil(v + side / 2)))
        if x1 <= x0 or y1 <= y0:
            return expected
        visible = 1.0 - float(self._last_arm_mask[y0:y1, x0:x1].mean())
        return expected * max(visible, 0.05)

    def _color_gate_sat_low(self, det, g):
        """perception.color_gate.sat_floor: a blue (purple) or yellow blob whose median S is below its floor is a
        shadow or sheen. Other colours (red, orange) and blobs without a measured S are never rejected here."""
        name = "blue" if det.color == "purple" else det.color
        if name == "blue" and self.sat_floor_adaptive:
            floor = self._sat_floors.get("blue")
        else:
            floor = (g.get("sat_floor_params") or {}).get(name) if self.sat_floor_fixed else None
        return floor is not None and det.sat_med is not None and det.sat_med < float(floor)

    @staticmethod
    def _copy_tentative(source, target):
        """Preserve opt-in dynamic metadata across dataclass replacements without changing legacy Detection bytes."""
        if getattr(source, "tentative", False):
            target.tentative = True
            target.tentative_reason = getattr(source, "tentative_reason", "fk_mask_boundary")
        return target

    def _projected_top_area_px(self, top, area_m2):
        """Local pinhole area of a horizontal top-face patch at ``top`` (row-aware via optical Z)."""
        try:
            T = self._T()
            p = (np.asarray(top, float) - T[:3, 3]) @ T[:3, :3]
            z = float(p[2])
            if not np.isfinite(z) or z <= 0:
                return None
            n = np.array([0.0, 0.0, 1.0]) @ T[:3, :3]
            K = self.cal.cams[self.cam].K
            out = float(K[0, 0]) * float(K[1, 1]) * abs(float(np.dot(n, p))) * float(area_m2) / z ** 3
            return out if np.isfinite(out) and out > 0 else None
        except (KeyError, TypeError, ValueError):
            return None

    def _color_gate_workspace(self, det, g):
        """Rule 2b: keep only pelvis x in ws_x, |y| <= ws_abs_y, and in front of the bin row's near edge (nominal or
        bin.vote-shifted footprint, never the near_margin hedge) minus ws_edge_m. Bin walls / rims project BEHIND it."""
        ws_x = g.get("ws_x")
        if ws_x is None:
            return None
        if self.wide and self.scene.counter_z_runtime is None:
            return None   # calib.wide_counter_window: counter height unmeasured and no prior -> the xy are unreliable: fail OPEN
        x, y = float(det.top[0]), float(det.top[1])
        if self._image_bin_gate_active:
            return "color_gate:workspace" if x < ws_x[0] or abs(y) > g["ws_abs_y"] else None
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
        u = None if det.uv is None else det.uv[0]
        if self._image_bin_gate_active and not self._rim_column_unknown(u):
            return None
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
        """[(lo, hi)] mat-frame footprint of every bin, with a per-bin confidence hedge.

        A confident bin vote uses its tight live footprint.  Missing, prior-only, and disagreeing votes use the union
        of the nominal footprint and its full allowed displacement towards the robot (mat -y = pelvis -x).  Trust is
        keyed only by ``planner.bin_conf[name]``: a populated ``bin_shift`` is not evidence by itself (M1).
        """
        sc = self.scene
        src = self.bin_shift_src
        conf = dict(getattr(src, "bin_conf", None) or {}) if self.live_bins else {}
        nominal = bin_geometry(sc)
        out = []
        for name, g_ in self.bin_geometry(gate=True).items():
            lo, hi = g_["outer_lo"], g_["outer_hi"]
            trusted = float(conf.get(name, 0.0)) >= 1.0
            if hedge and self.live_bins and not trusted:
                margin = float(g.get("near_margin_m", 0.0))
                nlo, nhi = nominal[name]["outer_lo"], nominal[name]["outer_hi"]
                # A shift paired with prior/disagree/missing confidence is not evidence.  Ignore it and sweep the
                # nominal footprint through the full allowed near displacement; only CONF_AGREE may tighten around
                # a live shifted box.
                lo = nlo - np.array([0.0, margin])
                hi = nhi
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
            out = replace(det, shape=None, shape_conf=0.0, shape_path="abstain_color_gate", shape_diag=diag,
                          apex_yaw=None, apex_conf=0.0, apex_gate=None)
            return self._copy_tentative(det, out)
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
