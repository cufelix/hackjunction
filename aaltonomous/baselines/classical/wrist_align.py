"""grasp.wrist_align (FLAG, default off): correct the last centimetres of a pick from the grasping hand's wrist camera.

The head camera's extrinsic error (1-3 cm on the real robot) moves the whole pick; the wrist camera rides on the hand,
so at the pregrasp pose (hand above the object, before the final descent) the object's image position tells where it
really is relative to the hand. The policy holds there for `hold_ticks`, then:

  1. segments the target colour in `<side>_wrist_rgb` with perception's HSV set (perception.hsv_set),
  2. projects the expected object centre (head detection, mid-height) into the wrist camera
     (FK of the camera's parent link x T_link_optical, with lens distortion),
  3. picks the target-colour blob nearest to it and intersects its centroid ray with the horizontal plane through the
     object centre (known height) -> observed pelvis xy -> correction = observed - expected,
  4. accepts the correction only if it is consistent: blob area within area_ratio x the expected area, blob not cut
     by the image border, |correction| <= max_corr_m, exactly one target-colour blob inside that radius and no
     other-colour blob nearer the expected position. Otherwise the original plan continues unchanged.

The observed xy is then refined with the object model: a cylinder (diameter = object size, height = object height) at
xy projects (FK, mount, distortion) to a convex silhouette whose area centroid must equal the blob centroid; 3 Gauss-
Newton steps from the ray estimate. That removes the perspective bias of the plain ray (the camera sees the object from
above/behind: moving it away shows more top face, ~30 % gain error at 18 cm). Cube / prism silhouettes are close to the
cylinder's (centrally symmetric or nearly), a few mm at most.

Camera model: calibration.yaml `cameras.<side>_wrist` when its `T_link_optical` is set (parent_link, intrinsics, dist;
the organisers' v2 bundle, or our own hand-eye measurement). Otherwise only with `use_fallback`:
`grasp.wrist_align_cfg.fallback_camera.<side>` (the sim preset as read from unitree_sim_isaaclab). That preset model
does NOT reproduce the sim's rendered wrist images (box G frames 2026-11-14: bins ~250 px off), so it is off by default:
without a calibrated mount every pick logs a skip and runs exactly as with the flag off.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def quat_wxyz_to_R(q):
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass
class WristCam:
    side: str
    parent: str                 # URDF frame the camera is fixed to
    T_link_optical: np.ndarray  # 4x4, optical frame (z fwd, x right, y down) in the parent frame
    K: np.ndarray               # 3x3
    dist: np.ndarray            # OpenCV (k1, k2, p1, p2, k3)
    width: int
    height: int
    source: str                 # "calibration" | "config"


@dataclass
class WristResult:
    ok: bool
    reason: str
    corr: np.ndarray = field(default_factory=lambda: np.zeros(2))   # pelvis xy correction (m) to add to det.top
    uv_exp: tuple | None = None
    uv_obs: tuple | None = None
    area: int = 0
    area_exp: float = 0.0

    def summary(self):
        f = lambda uv: "n/a" if uv is None else f"({uv[0]:.0f},{uv[1]:.0f})"
        return (f"{'ok' if self.ok else 'reject'} ({self.reason}) corr=({1000 * self.corr[0]:+.1f},"
                f"{1000 * self.corr[1]:+.1f}) mm uv_exp={f(self.uv_exp)} uv_obs={f(self.uv_obs)} "
                f"area={self.area}/{self.area_exp:.0f}px")


def load_camera(side, cal_cfg, wa_cfg):
    """Wrist camera model: calibration bundle if it has the mount, else the configured fallback."""
    c = ((cal_cfg or {}).get("cameras") or {}).get(f"{side}_wrist") or {}
    if c.get("T_link_optical") is not None:
        i = c["intrinsics"]
        T = np.asarray(c["T_link_optical"], float)
        parent, src = c.get("parent_link", f"{side}_wrist_yaw_link"), "calibration"
    else:
        fb = (wa_cfg.get("fallback_camera") or {}).get(side) if wa_cfg.get("use_fallback", False) else None
        if fb is None:
            return None
        i = fb["intrinsics"]
        T = np.eye(4)
        T[:3, :3] = quat_wxyz_to_R(fb["quat_wxyz"])
        T[:3, 3] = np.asarray(fb["pos"], float)
        parent, src = fb["parent_link"], "config"
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], float)
    dist = np.zeros(5)
    d = np.asarray(i.get("dist") or [], float)
    dist[:min(5, len(d))] = d[:5]
    return WristCam(side, parent, T, K, dist, int(i.get("width", 640)), int(i.get("height", 480)), src)


class WristAligner:
    def __init__(self, cfg, cal_cfg, kin, perception, scene):
        self.cfg = cfg
        self.wa = cfg["grasp"].get("wrist_align_cfg") or {}
        self.kin, self.perception, self.scene = kin, perception, scene
        self.cams = {s: load_camera(s, cal_cfg, self.wa) for s in ("left", "right")}
        self.height = float(perception.height)
        self._fid = {}
        self._nlim = {}

    # ------------------------------------------------------------------ geometry
    def cam_pose(self, side, arm_q):
        """T_pelvis_optical of the side's wrist camera for the full arm vector (14,)."""
        cam = self.cams[side]
        fid = self._fid.get(cam.parent)
        if fid is None:
            fid = self._fid[cam.parent] = self.kin.model.getFrameId(cam.parent)
        self.kin._fk(np.asarray(arm_q, float))
        M = self.kin.data.oMf[fid]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = M.rotation, M.translation
        return T @ cam.T_link_optical

    @staticmethod
    def _to_cam(T, p):
        return (np.asarray(p, float) - T[:3, 3]) @ T[:3, :3]

    def project(self, side, T, p):
        """Pelvis points (n,3) -> (uv (n,2) with distortion, z_optical (n,))."""
        import cv2
        cam = self.cams[side]
        pc = self._to_cam(T, np.atleast_2d(p))
        uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.dist)
        return uv.reshape(-1, 2), pc[:, 2]

    def fov_ok(self, side, T, p):
        """Whether pelvis point p lies inside the camera's field of view. Checked before projectPoints: a strong
        barrel polynomial (sim/real Dex3 wrist k1 ~ -0.4) folds back outside the image and maps far points inside.
        Limit = the undistorted radius where the radial model reaches the image corner, or folds, whichever first."""
        cam = self.cams[side]
        lim = self._nlim.get(side)
        if lim is None:
            k1, k2, _, _, k3 = cam.dist
            r = np.linspace(0.0, 4.0, 4001)
            rd = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
            fold = np.nonzero(np.diff(rd) <= 0)[0]
            r_fold = r[fold[0]] if len(fold) else r[-1]
            corner = max(np.hypot((u - cam.K[0, 2]) / cam.K[0, 0], (v - cam.K[1, 2]) / cam.K[1, 1])
                         for u in (0, cam.width) for v in (0, cam.height))
            reach = np.nonzero(rd >= corner)[0]
            lim = self._nlim[side] = min(r_fold, r[reach[0]] if len(reach) else r[-1])
        pc = self._to_cam(T, np.asarray(p, float)[None])[0]
        return bool(pc[2] > 0.03 and np.hypot(pc[0], pc[1]) / pc[2] <= lim)

    def ray_plane(self, side, T, uv, z_plane):
        """Pixel (distorted) -> pelvis point on the plane z = z_plane, or None."""
        import cv2
        cam = self.cams[side]
        n = cv2.undistortPoints(np.array([[uv]], float), cam.K, cam.dist).reshape(2)
        d = T[:3, :3] @ np.array([n[0], n[1], 1.0])
        if d[2] > -1e-6:
            return None
        t = (z_plane - T[2, 3]) / d[2]
        return None if t <= 0 else T[:3, 3] + t * d

    def _centre(self, det):
        return np.asarray(det.top, float) - np.array([0.0, 0.0, self.height / 2])

    def silhouette(self, side, T, centre):
        """(area px, area centroid (2,)) of the cylinder model's image at `centre` (mid-height), or (0, None)."""
        import cv2
        r, h = float(self.perception.size) / 2, self.height
        a = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ring = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(a)], 1)
        pts = np.vstack([centre + ring + [0, 0, h / 2], centre + ring - [0, 0, h / 2]])
        uv, z = self.project(side, T, pts)
        if np.any(z <= 0.01):
            return 0.0, None
        hull = cv2.convexHull(uv.astype(np.float32))
        m = cv2.moments(hull)
        if m["m00"] <= 1e-9:
            return 0.0, None
        return float(m["m00"]), np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])

    def expected_area(self, side, T, centre):
        return self.silhouette(side, T, centre)[0]

    def refine(self, side, T, centre_z, xy0, uv_obs, iters=3, h=0.001):
        """Pelvis xy whose model silhouette centroid equals uv_obs (Gauss-Newton from xy0), or xy0 if it fails."""
        xy, uv_obs = np.asarray(xy0, float).copy(), np.asarray(uv_obs, float)
        for _ in range(iters):
            _, c = self.silhouette(side, T, np.r_[xy, centre_z])
            cx = self.silhouette(side, T, np.r_[xy + [h, 0.0], centre_z])[1]
            cy = self.silhouette(side, T, np.r_[xy + [0.0, h], centre_z])[1]
            if c is None or cx is None or cy is None:
                return np.asarray(xy0, float)
            J = np.stack([(cx - c) / h, (cy - c) / h], 1)
            step = np.linalg.lstsq(J, uv_obs - c, rcond=None)[0]
            if not np.all(np.isfinite(step)) or np.linalg.norm(step) > 0.05:
                return np.asarray(xy0, float)
            xy = xy + step
            if np.linalg.norm(step) < 1e-4:
                break
        return xy

    def in_view(self, side, arm_q, det):
        """Whether the expected object centre projects into the wrist image (with view_margin_px) at arm_q."""
        cam = self.cams.get(side)
        if cam is None:
            return False, "no wrist camera model (calibration.yaml T_link_optical is null, use_fallback off)"
        T = self.cam_pose(side, arm_q)
        if not self.fov_ok(side, T, self._centre(det)):
            pc = self._to_cam(T, self._centre(det)[None])[0]
            return False, f"object outside the wrist camera's view (optical {np.round(pc, 3).tolist()} m)"
        uv, z = self.project(side, T, self._centre(det)[None])
        m = float(self.wa.get("view_margin_px", 20))
        u, v = uv[0]
        if z[0] <= 0.03:
            return False, f"object behind the camera (z={z[0]:+.3f} m)"
        if not (m <= u < cam.width - m and m <= v < cam.height - m):
            return False, f"object outside the wrist image (uv=({u:.0f},{v:.0f}))"
        return True, f"uv=({u:.0f},{v:.0f})"

    # ------------------------------------------------------------------ measurement
    def measure(self, rgb, side, arm_q, det):
        """WristResult for detection `det` (head-camera estimate) from the side's wrist image at arm_q."""
        import cv2
        cam = self.cams.get(side)
        if cam is None:
            return WristResult(False, "no camera model")
        if rgb is None or np.ndim(rgb) != 3 or not np.any(rgb):
            return WristResult(False, "no wrist frame")
        rgb = np.asarray(rgb)
        if rgb.shape[:2] != (cam.height, cam.width):
            return WristResult(False, f"wrist frame {rgb.shape[1]}x{rgb.shape[0]} != calibration "
                                      f"{cam.width}x{cam.height}")
        wa = self.wa
        T = self.cam_pose(side, arm_q)
        centre = self._centre(det)
        if not self.fov_ok(side, T, centre):
            return WristResult(False, "object outside the wrist camera's view")
        uv_e, z_e = self.project(side, T, centre[None])
        uv_e = (float(uv_e[0, 0]), float(uv_e[0, 1]))
        if z_e[0] <= 0.03:
            return WristResult(False, "object behind the wrist camera", uv_exp=uv_e)
        area_exp = self.expected_area(side, T, centre)
        max_corr = float(wa.get("max_corr_m", 0.03))
        min_area = int(wa.get("min_area_px", 60))
        border = int(wa.get("border_px", 3))
        H, W = rgb.shape[:2]
        blobs = []   # (|corr|, colour, corr, uv, area, touches_border)
        for color, m in self.perception.masks(rgb).items():
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                uv = (float(cent[i, 0]), float(cent[i, 1]))
                hit = self.ray_plane(side, T, uv, centre[2])
                if hit is None:
                    continue
                corr = hit[:2] - centre[:2]
                x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                         cv2.CC_STAT_HEIGHT))
                edge = x <= border or y <= border or x + w >= W - border or y + h >= H - border
                blobs.append((float(np.linalg.norm(corr)), color, corr, uv, area, edge))
        mine = sorted((b for b in blobs if b[1] == det.color), key=lambda b: b[0])
        if not mine or mine[0][0] > max_corr:
            near = f", nearest {1000 * mine[0][0]:.0f} mm" if mine else ""
            return WristResult(False, f"no {det.color} blob within {1000 * max_corr:.0f} mm{near}", uv_exp=uv_e,
                               area_exp=area_exp)
        d, _, corr, uv, area, edge = mine[0]
        if bool(wa.get("model_refine", True)):   # perspective-exact: silhouette centroid of the object model
            corr = self.refine(side, T, centre[2], centre[:2] + corr, uv) - centre[:2]
            if np.linalg.norm(corr) > max_corr:
                return WristResult(False, f"refined correction {1000 * np.linalg.norm(corr):.0f} mm > "
                                          f"{1000 * max_corr:.0f} mm", np.zeros(2), uv_e, uv, area, area_exp)
        res = WristResult(False, "", np.asarray(corr, float), uv_e, uv, area, area_exp)
        if len(mine) > 1 and mine[1][0] <= max_corr:
            res.reason = f"ambiguous: {len(mine)} {det.color} blobs, 2nd at {1000 * mine[1][0]:.0f} mm"
            return res
        other = [b for b in blobs if b[1] != det.color and b[0] < d]
        if other:
            res.reason = f"colour mismatch: {other[0][1]} blob nearer ({1000 * other[0][0]:.0f} mm)"
            return res
        if edge:
            res.reason = "blob cut by the image border"
            return res
        lo, hi = (float(v) for v in wa.get("area_ratio", [0.4, 1.8]))
        if area_exp <= 0 or not (lo * area_exp <= area <= hi * area_exp):
            res.reason = f"area {area} px outside [{lo:.2f}, {hi:.2f}] x {area_exp:.0f} px"
            return res
        res.ok, res.reason = True, "consistent"
        res.corr = np.asarray(corr, float) * float(wa.get("gain", 1.0))
        return res
