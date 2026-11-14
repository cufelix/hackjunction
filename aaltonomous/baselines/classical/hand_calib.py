"""Per-episode head-camera pose refinement from the robot's own arms (flag perception.hand_calibration.enabled).

The arms are known from the joint encoders + FK; the head camera pose is not (sim jitter +-1 cm / +-1 deg, real
robot tens of pixels). So: pose the arm surface model (points sampled from the Unitree link meshes, shipped as
assets/arm_surface.npz) with the MEASURED arm_q / hand_q, project it with the nominal T_pelvis_optical to find the
image region where the forearms / hands should be, take the depth pixels around it, and solve a small 6-DOF
correction of T_pelvis_optical by robust point-to-plane ICP (data pixel -> nearest visible model point).

Gates (any fails -> nominal extrinsics, reason logged): enough model points in view, enough inliers, low inlier
RMS, model coverage, a well-conditioned problem (all 6 directions observed) and a capped correction
(max_trans_m / max_rot_deg). Pure numpy + scipy cKDTree + pinocchio FK.

    hc = HandCalibrator(cfg["perception"]["hand_calibration"])
    res = hc.solve(cal, "head", depth_mm, arm_q, hand_q)   # res.T = corrected T_pelvis_optical (or nominal)

Rebuild the surface asset from the URDF meshes (not in this repo; Unitree xr_teleoperate assets/g1/meshes):
    python -m aaltonomous.baselines.classical.hand_calib --build <meshes_dir>
"""
from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aaltonomous.interface.safety import DEFAULT_URDF, JointSpec

HERE = Path(__file__).resolve().parent
ASSET = HERE / "assets" / "arm_surface.npz"
_LINKS = ("shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
          "hand_palm_link", "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link", "hand_index_0_link",
          "hand_index_1_link", "hand_middle_0_link", "hand_middle_1_link")
ARM_LINKS = tuple(f"{s}_{l}" for s in ("left", "right") for l in _LINKS)

DEFAULTS = dict(
    enabled=False,
    max_trans_m=0.03,         # cap on the camera position change (pelvis frame)
    max_rot_deg=3.0,          # cap on the camera rotation change
    min_model_points=150,     # visible model points (after self-occlusion) needed to try at all
    min_inliers=300,          # matched depth pixels after the last ICP round
    min_coverage=0.6,         # fraction of visible model points that got a matched data point nearby
    max_residual_m=0.0025,    # RMS point-to-plane residual of the inliers
    min_eig=2e-4,             # smallest eigenvalue of the normalised 6x6 information matrix (observability)
    search_px=40,             # data window: dilation of the predicted arm silhouette
    pixel_stride=3,
    max_data_points=1500,     # depth points used by ICP (uniform subsample; runtime ~ this)
    vis_cell_px=4,            # z-buffer cell for model self-occlusion
    vis_tol_m=0.008,
    max_iters=14,
    thr_start_m=0.04,         # correspondence gate, shrinks geometrically to thr_end_m
    thr_end_m=0.008,
    huber_m=0.003,
    depth_range_mm=[150, 1500],
    max_tries=10,             # policy: give up after this many still frames without a valid result
    still_rad=0.01,           # policy: max arm joint change between two frames to count as still
)


# ---------------------------------------------------------------------------------------------- surface model
def read_stl(path):
    """Binary (or ASCII) STL -> triangles (n, 3, 3) float64."""
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0] if len(b) >= 84 else 0
    if len(b) == 84 + 50 * n:
        rec = np.frombuffer(b, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n,
                            offset=84)
        return rec["v"].astype(float)
    vals = [list(map(float, l.split()[1:4])) for l in b.decode(errors="ignore").splitlines()
            if l.strip().startswith("vertex")]
    return np.asarray(vals, float).reshape(-1, 3, 3)


def sample_mesh(tris, spacing, rng, oversample=8):
    """Area-uniform surface samples + face normals, thinned to one point per `spacing` voxel."""
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    ok = area > 1e-12
    tris, cr, area, e1, e2 = tris[ok], cr[ok], area[ok], e1[ok], e2[ok]
    nrm = cr / np.linalg.norm(cr, axis=1, keepdims=True)
    k = max(int(oversample * area.sum() / spacing ** 2), 50)
    idx = rng.choice(len(tris), size=k, p=area / area.sum())
    a, b = rng.random(k), rng.random(k)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    p = tris[idx, 0] + a[:, None] * e1[idx] + b[:, None] * e2[idx]
    if spacing <= 0:
        return p, nrm[idx]
    _, keep = np.unique(np.floor(p / spacing).astype(np.int64), axis=0, return_index=True)
    return p[keep], nrm[idx][keep]


def build_surface_asset(mesh_dir, out=ASSET, spacing=0.005, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {}
    for link in ARM_LINKS:
        p, n = sample_mesh(read_stl(Path(mesh_dir) / f"{link}.STL"), spacing, rng)
        arrays[f"p/{link}"] = p.astype(np.float32)
        arrays[f"n/{link}"] = n.astype(np.float32)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spacing=np.float32(spacing), **arrays)
    return arrays


class ArmSurface:
    """Link-frame surface points of both arms (shoulder_yaw .. fingertips), posed by FK in the pelvis frame.
    Legs / waist at 0 = the stiff pose calibration.py and the safety filter assume (URDF visual origins are 0)."""

    def __init__(self, asset=ASSET, urdf=DEFAULT_URDF, spec: JointSpec | None = None, links=None):
        import pinocchio as pin
        self.pin = pin
        spec = spec or JointSpec()
        if isinstance(asset, dict):
            z, self.spacing = asset, float(asset.get("spacing", 0.003))
        else:
            z = np.load(asset)
            self.spacing = float(z["spacing"])
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        iq = lambda n: self.model.joints[self.model.getJointId(n)].idx_q  # noqa: E731
        self.arm_iq = np.array([iq(n) for n in spec.arm_names])
        self.hand_iq = np.array([iq(n) for n in spec.hand_names])
        self.frames, self.pts, self.nrm = [], [], []
        for link in links or ARM_LINKS:
            if f"p/{link}" not in z:
                continue
            self.frames.append(self.model.getFrameId(link))
            self.pts.append(np.asarray(z[f"p/{link}"], float))
            self.nrm.append(np.asarray(z[f"n/{link}"], float))

    def posed(self, arm_q, hand_q=None):
        """-> points (N, 3), normals (N, 3) in the pelvis frame."""
        self.q[self.arm_iq] = np.asarray(arm_q, float)
        if hand_q is not None:
            self.q[self.hand_iq] = np.asarray(hand_q, float)
        self.pin.framesForwardKinematics(self.model, self.data, self.q)
        P, N = [], []
        for f, p, n in zip(self.frames, self.pts, self.nrm):
            M = self.data.oMf[f]
            P.append(p @ M.rotation.T + M.translation)
            N.append(n @ M.rotation.T)
        return np.concatenate(P), np.concatenate(N)


# ---------------------------------------------------------------------------------------------- solver
@dataclass
class HeadCalibResult:
    success: bool
    valid_frame: bool          # False = try again on a later frame (no depth, arms not in view, solver error)
    T: np.ndarray              # T_pelvis_optical to use (nominal on failure)
    dt_m: np.ndarray = field(default_factory=lambda: np.zeros(3))   # camera position change, pelvis frame
    drot_deg: float = 0.0
    n_model: int = 0
    n_inliers: int = 0
    coverage: float = 0.0
    residual_m: float = float("inf")
    residual0_m: float = float("inf")   # same metric at the nominal pose (before)
    min_eig: float = 0.0
    elapsed_ms: float = 0.0
    reason: str = ""

    def summary(self):
        d = self.dt_m * 1000
        return (f"dt=({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f}) mm drot={self.drot_deg:.2f} deg, inliers={self.n_inliers}, "
                f"model={self.n_model}, coverage={self.coverage:.2f}, rms={self.residual0_m * 1000:.1f}->"
                f"{self.residual_m * 1000:.1f} mm, min_eig={self.min_eig:.1e}, {self.elapsed_ms:.1f} ms ({self.reason})")


def _rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


class HandCalibrator:
    def __init__(self, params=None, surface: ArmSurface | None = None):
        p = dict(DEFAULTS)
        p.update(params or {})
        self.p = p
        self._surface = surface

    @property
    def surface(self):
        if self._surface is None:
            self._surface = ArmSurface()
        return self._surface

    def visible_model(self, P, T, K, w, h):
        """Model points the camera at T should see: in the image, in front, front-most in their z-buffer cell."""
        p = self.p
        pc = (P - T[:3, 3]) @ T[:3, :3]
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]
            v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        ok = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        c = int(p["vis_cell_px"])
        gw, gh = (w + c - 1) // c, (h + c - 1) // c
        cell = np.full(len(P), -1)
        cell[ok] = (v[ok] // c).astype(int) * gw + (u[ok] // c).astype(int)
        zmin = np.full(gw * gh, np.inf)
        np.minimum.at(zmin, cell[ok], z[ok])
        vis = ok.copy()
        vis[ok] = z[ok] <= zmin[cell[ok]] + p["vis_tol_m"]
        occ = np.isfinite(zmin).reshape(gh, gw)
        zr = (float(z[vis].min()), float(z[vis].max())) if vis.any() else (0.0, 0.0)
        return vis, occ, zr

    def solve(self, cal, cam, depth, arm_q, hand_q=None):
        t0 = time.perf_counter()
        p = self.p
        T0 = None

        def out(success, valid, reason, T=None, **kw):
            T = T0 if T is None else T
            return HeadCalibResult(success, valid, None if T is None else np.array(T, float),
                                   elapsed_ms=(time.perf_counter() - t0) * 1e3, reason=reason, **kw)

        try:
            c = cal.cams[cam]
            T0 = np.array(cal._T(cam), float)
            if depth is None:
                return out(False, False, "no depth")
            from scipy.spatial import cKDTree
            import cv2
            depth = np.asarray(depth)
            if depth.ndim != 2:
                return out(False, False, f"bad depth shape {depth.shape}")
            h, w = depth.shape
            P, N = self.surface.posed(arm_q, hand_q)
            vis, occ, (zlo, zhi) = self.visible_model(P, T0, c.K, w, h)
            n_model = int(vis.sum())
            if n_model < p["min_model_points"]:
                return out(False, False, f"arms not in view ({n_model} model points)", n_model=n_model)
            M, Mn = P[vis], N[vis]
            # data window: predicted silhouette (z-buffer cells) dilated by search_px
            cpx = int(p["vis_cell_px"])
            r = max(1, int(np.ceil(p["search_px"] / cpx)))
            occ = cv2.dilate(occ.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            s = int(p["pixel_stride"])
            vv, uu = np.mgrid[0:h:s, 0:w:s]
            vv, uu = vv.ravel(), uu.ravel()
            dmm = depth[vv, uu].astype(float) * (c.depth_scale / 0.001)
            lo, hi = p["depth_range_mm"]
            lo = max(lo, (zlo - 0.15) * 1000)
            hi = min(hi, (zhi + 0.15) * 1000)
            keep = occ[vv // cpx, uu // cpx].astype(bool) & np.isfinite(dmm) & (dmm > lo) & (dmm < hi)
            if keep.sum() < p["min_inliers"]:
                return out(False, False, f"too few depth pixels near the arms ({int(keep.sum())})", n_model=n_model)
            Q = cal.deproject(cam, uu[keep], vv[keep], dmm[keep])          # optical frame (m)
            tree = cKDTree(M)
            R, t = T0[:3, :3].copy(), T0[:3, 3].copy()
            # drop pixels nowhere near the arm model (table, objects), then subsample evenly
            d0, _ = tree.query(Q @ R.T + t, distance_upper_bound=p["thr_start_m"])
            Q = Q[np.isfinite(d0)]
            if len(Q) < p["min_inliers"]:
                return out(False, True, f"too few depth points near the arm model ({len(Q)})", n_model=n_model)
            if len(Q) > p["max_data_points"]:
                Q = Q[np.linspace(0, len(Q) - 1, int(p["max_data_points"])).astype(int)]

            def residuals(R, t, thr):
                Pd = Q @ R.T + t
                d, idx = tree.query(Pd, distance_upper_bound=thr)
                m = np.isfinite(d)
                idx = idx[m]
                e = np.einsum("ij,ij->i", Mn[idx], Pd[m] - M[idx])
                return m, idx, e

            m, idx, e = residuals(R, t, p["thr_end_m"])
            res0 = float(np.sqrt(np.mean(e ** 2))) if m.sum() else float("inf")
            thr = p["thr_start_m"]
            decay = (p["thr_end_m"] / p["thr_start_m"]) ** (1.0 / max(1, p["max_iters"] // 2))
            for _ in range(int(p["max_iters"])):
                m, idx, e = residuals(R, t, thr)
                if m.sum() < 6:
                    return out(False, True, "ICP lost all correspondences", n_model=n_model)
                nk = Mn[idx] @ R                                          # model normals in the camera frame
                J = np.hstack([np.cross(Q[m], nk), nk])
                a = np.abs(e)
                wgt = np.where(a <= p["huber_m"], 1.0, p["huber_m"] / np.maximum(a, 1e-12))
                H = J.T @ (J * wgt[:, None])
                g = J.T @ (wgt * e)
                xi = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
                t = R @ xi[3:] + t
                R = R @ _rodrigues(xi[:3])
                done = thr <= p["thr_end_m"] and np.linalg.norm(xi) < 1e-5
                thr = max(p["thr_end_m"], thr * decay)
                if done:
                    break
            m, idx, e = residuals(R, t, p["thr_end_m"])
            n_in = int(m.sum())
            res = float(np.sqrt(np.mean(e ** 2))) if n_in else float("inf")
            # coverage: fraction of the visible model surface that has an inlier depth point near it
            if n_in:
                dm, _ = cKDTree(Q[m] @ R.T + t).query(M, distance_upper_bound=2 * p["thr_end_m"])
                cov = float(np.isfinite(dm).mean())
            else:
                cov = 0.0
            # observability: information matrix with rotations scaled by the mean range (metres at the scene)
            if n_in >= 6:
                nk = Mn[idx] @ R
                J = np.hstack([np.cross(Q[m], nk) / max(np.linalg.norm(Q[m], axis=1).mean(), 1e-6), nk])
                min_eig = float(np.linalg.eigvalsh(J.T @ J / n_in)[0])
            else:
                min_eig = 0.0
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            dt = t - T0[:3, 3]
            drot = _rot_deg(T0[:3, :3].T @ R)
            kw = dict(dt_m=dt, drot_deg=drot, n_model=n_model, n_inliers=n_in, coverage=cov, residual_m=res,
                      residual0_m=res0, min_eig=min_eig)
            why = []
            if n_in < p["min_inliers"]:
                why.append(f"inliers {n_in}<{p['min_inliers']}")
            if cov < p["min_coverage"]:
                why.append(f"coverage {cov:.2f}<{p['min_coverage']}")
            if res > p["max_residual_m"]:
                why.append(f"rms {res * 1000:.1f}>{p['max_residual_m'] * 1000:.1f} mm")
            if min_eig < p["min_eig"]:
                why.append(f"ill-conditioned (min_eig {min_eig:.1e})")
            if np.linalg.norm(dt) > p["max_trans_m"]:
                why.append(f"|dt| {np.linalg.norm(dt) * 1000:.1f}>{p['max_trans_m'] * 1000:.0f} mm")
            if drot > p["max_rot_deg"]:
                why.append(f"drot {drot:.2f}>{p['max_rot_deg']} deg")
            if why:
                return out(False, True, "rejected: " + ", ".join(why), **kw)
            return out(True, True, "ok", T=T, **kw)
        except Exception as exc:   # never let calibration end an episode
            return out(False, False, f"{type(exc).__name__}: {exc}")


def corrected_calibration(cal, cam, T):
    """Shallow copy of `cal` whose `cam` has T_pelvis_optical = T. The shared object is never mutated."""
    import copy
    new = copy.copy(cal)
    new.cams = dict(cal.cams)
    new.cams[cam] = copy.copy(cal.cams[cam])
    new.cams[cam].T_pelvis_optical = np.array(T, float)
    return new


# ---------------------------------------------------------------------------------------------- synthetic data
def render_depth(P, T, K, w=640, h=480, spacing=0.003, background=None):
    """Synthetic depth (uint16 mm) of surface points P (pelvis frame) seen from T_pelvis_optical = T: z-buffer of
    square splats sized to cover the sample spacing. `background`: (h, w) depth in mm drawn behind (e.g. a table)."""
    pc = (P - T[:3, 3]) @ T[:3, :3]
    z = pc[:, 2]
    ok = z > 0.05
    pc, z = pc[ok], z[ok]
    u = np.round(K[0, 0] * pc[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[:, 1] / z + K[1, 2]).astype(int)
    zb = np.full(h * w, np.inf) if background is None else np.asarray(background, float).ravel() / 1000
    rad = np.maximum(1, np.ceil(0.6 * spacing * K[0, 0] / z)).astype(int)
    for rr in np.unique(rad):
        sel = rad == rr
        for du in range(-rr + 1, rr):
            for dv in range(-rr + 1, rr):
                uu, vv = u[sel] + du, v[sel] + dv
                inn = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                np.minimum.at(zb, vv[inn] * w + uu[inn], z[sel][inn])
    zb[~np.isfinite(zb)] = 0
    return np.round(zb * 1000).reshape(h, w).astype(np.uint16)


def table_depth(T, K, z_table=0.01, x_min=0.14, w=640, h=480):
    """Depth (mm, float, inf = no hit) of the counter plane z = z_table (pelvis frame, x > x_min) seen from T."""
    vv, uu = np.mgrid[0:h, 0:w]
    ray = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    rp = ray @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z_table - T[2, 3]) / rp[..., 2]
    x = T[0, 3] + s * rp[..., 0]
    return np.where((s > 0) & (x > x_min), s * 1000, np.inf)


def perturb(T, dtrans, drot_deg_xyz):
    """T_pelvis_optical moved by dtrans (pelvis frame, m) and rotated by small xyz angles (deg, camera frame)."""
    T2 = np.array(T, float)
    T2[:3, :3] = T[:3, :3] @ _rodrigues(np.radians(np.asarray(drot_deg_xyz, float)))
    T2[:3, 3] = T[:3, 3] + np.asarray(dtrans, float)
    return T2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", metavar="MESH_DIR", help="sample the arm link STLs into assets/arm_surface.npz")
    ap.add_argument("--spacing", type=float, default=0.005)
    a = ap.parse_args(argv)
    if a.build:
        arr = build_surface_asset(a.build, spacing=a.spacing)
        print(f"wrote {ASSET}: {sum(len(v) for k, v in arr.items() if k.startswith('p/'))} points")


if __name__ == "__main__":
    main()
