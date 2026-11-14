"""Synthetic tests for prism_pose (triangle pose, robust centre; shape decisions are tested in test_shape_id.py). Run:
    python -m pytest aaltonomous/baselines/classical/test_prism_pose.py -q
    python -m aaltonomous.baselines.classical.test_prism_pose        # prints the full error table (about 1 min)
Yaw errors are measured modulo 120 deg (equilateral symmetry): 'flip' = error > 30 deg, i.e. the wrong vertex wedge.
"""
import cv2
import numpy as np

from .prism_pose import (TRIANGLE_MIN_POINTS, robust_center, triangle_pose)

SIDE = 0.045 / (np.sqrt(3) / 2)          # 0.052 m
ALT = 0.045


def make_cloud(rng, n_kept, hole, sigma, apex=None, centre=None):
    """Uniform points in an equilateral triangle (apex towards `apex`), then uneven holes: drop probability grows
    linearly along a random direction (mean hole fraction `hole`, max 0.95), then gaussian noise sigma (m).
    Returns (xy kept, true centre, true apex angle). Exactly n_kept points survive."""
    apex = rng.uniform(0, 2 * np.pi) if apex is None else apex
    centre = rng.uniform(-0.3, 0.3, 2) if centre is None else np.asarray(centre)
    n0 = int(round(1.5 * n_kept / max(1 - hole, 0.05))) + 20
    R = ALT * 2 / 3
    V = R * np.stack([np.cos(apex + 2 * np.pi * np.arange(3) / 3), np.sin(apex + 2 * np.pi * np.arange(3) / 3)], -1)
    u = rng.random((n0, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    p = V[0] + u[:, :1] * (V[1] - V[0]) + u[:, 1:] * (V[2] - V[0])
    if hole > 0:
        d = rng.uniform(0, 2 * np.pi)
        t = p @ np.array([np.cos(d), np.sin(d)])
        t = (t - t.min()) / (np.ptp(t) + 1e-12)
        keep = rng.random(n0) >= np.clip(2 * hole * t, 0, 0.95)
        p = p[keep]
    p = p[rng.permutation(len(p))[:n_kept]]          # exactly n_kept survive (when enough were generated)
    p = p + rng.normal(0, sigma, p.shape)
    return p + centre, centre, apex


def err_mod120(a, b):
    d = (a - b + np.pi / 3) % (2 * np.pi / 3) - np.pi / 3
    return abs(np.degrees(d))


def baseline(xy):
    """Current approach: mean centre; yaw = direction centroid -> minAreaRect centre (points at the apex side)."""
    c = xy.mean(0)
    (rx, ry), _, _ = cv2.minAreaRect(xy.astype(np.float32))
    return c, np.arctan2(ry - c[1], rx - c[0]) % (2 * np.pi)


def trial(rng, n, hole, sigma):
    xy, c, a = make_cloud(rng, n, hole, sigma)
    r = triangle_pose(xy, min_points=10)
    bc, by = baseline(xy)
    ours = (np.linalg.norm(r[0] - c) * 1e3, err_mod120(r[1], a), r[2]) if r else (np.nan, np.nan, 0.0)
    return ours + (np.linalg.norm(bc - c) * 1e3, err_mod120(by, a))


def stats(rng, n, hole, trials=100, sigma_max=0.002):
    R = np.array([trial(rng, n, hole, rng.uniform(0, sigma_max)) for _ in range(trials)])
    f = lambda e: (np.percentile(e, 50), np.percentile(e, 90), e.max(), np.mean(e > 30))
    return {"ours": f(R[:, 1]), "base": f(R[:, 4]), "c_ours": np.median(R[:, 0]), "c_base": np.median(R[:, 3]),
            "conf": np.median(R[:, 2])}


NS = (20, 30, 40, 60, 100, 200, 500, 1500)
HOLES = (0.0, 0.1, 0.2, 0.3)


def print_table(trials=100):
    rng = np.random.default_rng(0)
    print("yaw err deg (mod 120) p50/p90/max | flips(>30) | centre err mm p50 ; ours vs baseline(mean+minAreaRect)")
    print("noise sigma U(0,2) mm, N = points KEPT after holes")
    for h in HOLES:
        print(f"-- hole fraction {h:.1f}")
        for n in NS:
            s = stats(rng, n, h, trials)
            o, b = s["ours"], s["base"]
            print(f"N={n:5d} ours {o[0]:5.1f}/{o[1]:5.1f}/{o[2]:5.1f} flip {o[3]:.2f} c {s['c_ours']:.2f} | "
                  f"base {b[0]:5.1f}/{b[1]:5.1f}/{b[2]:5.1f} flip {b[3]:.2f} c {s['c_base']:.2f}")
    print("min N for ours p90 < 10 deg:")
    for h in HOLES:
        ok = [n for n in NS if stats(np.random.default_rng(1), n, h, trials)["ours"][1] < 10]
        print(f"  hole {h:.1f}: {ok[0] if ok else '>' + str(NS[-1])}")


# ---- pytest ---------------------------------------------------------------------------------------------------------
def test_clean_triangle_exact_any_apex():
    rng = np.random.default_rng(1)
    for apex in np.radians(np.arange(0, 360, 17.0)):
        xy, c, a = make_cloud(rng, 400, 0.0, 0.0, apex=apex, centre=(0.1, -0.2))
        cen, yaw, conf = triangle_pose(xy)
        assert err_mod120(yaw, a) < 1.5
        assert np.linalg.norm(cen - c) < 1.5e-3
        assert 0 <= yaw < 2 * np.pi and conf > 0.7


def test_yaw_is_vertex_nearest_prefer():
    rng = np.random.default_rng(2)
    xy, c, a = make_cloud(rng, 400, 0.0, 0.0, apex=np.pi / 2 + np.radians(100))   # true vertices: 190, 310, 70 deg
    _, yaw, _ = triangle_pose(xy, prefer_yaw=np.pi / 2)
    assert abs(np.degrees(yaw) - 70) < 2.0
    _, yaw, _ = triangle_pose(xy, prefer_yaw=np.radians(200))
    assert abs(np.degrees(yaw) - 190) < 2.0


def test_min_points_gate():
    rng = np.random.default_rng(3)
    xy, _, _ = make_cloud(rng, TRIANGLE_MIN_POINTS - 1, 0.0, 0.001)
    assert triangle_pose(xy[:TRIANGLE_MIN_POINTS - 1]) is None
    assert triangle_pose(xy, min_points=10) is not None
    assert triangle_pose(np.zeros((0, 2))) is None


def test_stress_holes_noise():
    rng = np.random.default_rng(4)
    s = stats(rng, 200, 0.2, trials=60)
    assert s["ours"][1] < 10 and s["ours"][3] <= 0.02          # p90 < 10 deg, flips rare
    assert s["c_ours"] < 2.0                                    # median centre error < 2 mm
    assert s["ours"][1] <= s["base"][1]


def test_beats_baseline_centre_with_uneven_holes():
    rng = np.random.default_rng(5)
    s = stats(rng, 300, 0.3, trials=60)
    assert s["c_ours"] <= s["c_base"] + 0.1


def test_robust_center_ignores_outliers():
    rng = np.random.default_rng(6)
    xy = rng.normal(0, 0.003, (200, 2))
    xy[:20] += 0.05
    assert np.linalg.norm(robust_center(xy)) < 0.0015
    assert np.linalg.norm(xy.mean(0)) > 0.003
    assert robust_center(xy[:3]) is None


def _wrong_shape_cloud(kind, side, n=400, seed=0, rot=0.4):
    rng = np.random.default_rng(seed)
    if kind == "cube":
        p = rng.uniform(-side / 2, side / 2, (n, 2))
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        return p @ R.T
    a, r = rng.uniform(0, 2 * np.pi, n), side / 2 * np.sqrt(rng.uniform(0, 1, n))   # cylinder, diameter = side
    return np.stack([r * np.cos(a), r * np.sin(a)], -1)


def test_wrong_shape_gives_low_triangle_confidence():
    """A cube / cylinder wrongly labelled triangle must end in conf < the planner's yaw_min_conf (0.5), whatever
    its size (size term alone is fooled by a 30 mm cube; the fill term is not)."""
    for kind in ("cube", "cylinder"):
        for side in (0.03, 0.045, 0.06):
            for seed in range(3):
                tp = triangle_pose(_wrong_shape_cloud(kind, side, seed=seed))
                assert tp is not None and tp[2] < 0.5, (kind, side, seed, tp[2])


def test_altitude_param_follows_scene_size():
    rng = np.random.default_rng(3)
    xy, c, a = make_cloud(rng, 400, 0.0, 0.0, apex=0.3)
    xy50 = c + (xy - c) * (0.065 / ALT)
    assert triangle_pose(xy50, altitude=0.065)[2] > 0.7
    assert triangle_pose(xy50)[2] < 0.5          # default 45 mm expectation rejects a 65 mm part


if __name__ == "__main__":
    print_table()
