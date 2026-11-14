"""grasp.wrist_align: pregrasp xy correction from the wrist camera. Run:
    python -m unittest aaltonomous.baselines.classical.test_wrist_align -v

Synthetic wrist frames come from the mock ray renderer through a TEST mount (a camera fixed to the wrist that looks at
the object from behind/above at the pregrasp pose), written into a copy of calibration.yaml, i.e. the
"calibration has the mount" path. The shipped fallback (sim asset mount) is checked separately.
"""
import copy
import hashlib
import unittest
from unittest import mock

import numpy as np
import yaml

from aaltonomous.interface.calibration import HERE as CAL_DIR, Calibration

from .config import load_config
from .mock_world import RenderMockBackend, Renderer, make_mock_env
from .perception import Detection
from .policy import ClassicalPolicy
from .wrist_align import WristAligner, load_camera

REF = {"left": (0.30, 0.10), "right": (0.30, -0.10)}


def _wa_cfg(**over):
    cfg = load_config(overlay="none", overrides={"grasp.wrist_align": True})
    cfg["grasp"]["wrist_align_cfg"].update(over)
    return cfg


def _look_at(pos, target):
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = pos
    return T


_CAL = None


def test_calibration():
    """calibration.yaml + a wrist mount per side (T_link_optical in <side>_wrist_yaw_link): the camera sits 6 cm behind
    and 4 cm above the grasp point at a reference pregrasp and looks at the object centre. Intrinsics: the yaml's."""
    global _CAL
    if _CAL is not None:
        return _CAL
    cfg = load_config(overlay="none")
    pol = ClassicalPolicy(config=cfg, log=None)
    with open(CAL_DIR / "calibration.yaml") as f:
        raw = yaml.safe_load(f)
    H = pol.planner.height
    for side, xy in REF.items():
        top = np.r_[xy, pol.scene.counter_z + H]
        p = pol.planner._plan_side_hints(Detection("red", top, 500, 300, "depth"), side, side, pol.home.copy(),
                                         pol.post["open"].copy())
        assert p.ok, p.reason
        e = next(e for n, s, e in p.phases if n == "to_pregrasp")
        q = p.arm[e - 1]
        g, _ = pol.kin.grasp_pose(side, q)
        pol.kin._fk(q)
        M = pol.kin.data.oMf[pol.kin.model.getFrameId(f"{side}_wrist_yaw_link")]
        Tw = np.eye(4)
        Tw[:3, :3], Tw[:3, 3] = M.rotation, M.translation
        Tc = _look_at(g + [-0.06, 0.0, 0.04], top - [0, 0, H / 2])
        c = raw["cameras"][f"{side}_wrist"]
        c["parent_link"] = f"{side}_wrist_yaw_link"
        c["T_link_optical"] = (np.linalg.inv(Tw) @ Tc).tolist()
    _CAL = Calibration(raw)
    return _CAL


def _setup(cfg=None):
    cfg = cfg or _wa_cfg()
    cal = test_calibration()
    pol = ClassicalPolicy(config=cfg, calibration=cal, log=None)
    return pol


def _pregrasp(pol, side, xy):
    """(det, q at the end of to_pregrasp) for a red object whose head estimate is at xy."""
    top = np.r_[xy, pol.scene.counter_z + pol.planner.height]
    det = Detection("red", top, 500, 300, "depth")
    p = pol.planner._plan_side_hints(det, side, side, pol.home.copy(), pol.post["open"].copy())
    assert p.ok, p.reason
    e = next(e for n, s, e in p.phases if n == "to_pregrasp")
    return det, p.arm[e - 1], p


def _render(pol, side, q, objects):
    T = pol.wrist.cam_pose(side, q)
    rgb, _ = Renderer(pol.scene, pol.cal, cam=f"{side}_wrist", T_pelvis_optical=T).render(objects)
    return rgb


def _obj(pol, xy, color="red", shape="cylinder"):
    return {"color": color, "shape": shape, "yaw": 0.3, "size": pol.scene.size, "height": pol.planner.height,
            "pos": np.r_[xy, pol.scene.counter_z + pol.planner.height / 2]}


class TestMeasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pol = _setup()

    def test_camera_comes_from_the_calibration_when_it_has_the_mount(self):
        self.assertEqual(self.pol.wrist.cams["left"].source, "calibration")
        self.assertEqual(self.pol.wrist.cams["left"].K[0, 0], 400.0)
        stock = Calibration.load()   # shipped bundle: T_link_optical null -> no model unless use_fallback
        self.assertIsNone(load_camera("right", stock.cfg, _wa_cfg()["grasp"]["wrist_align_cfg"]))
        cam = load_camera("right", stock.cfg, _wa_cfg(use_fallback=True)["grasp"]["wrist_align_cfg"])
        self.assertEqual((cam.source, cam.parent), ("config", "right_wrist_yaw_link"))
        self.assertAlmostEqual(cam.K[0, 0], 418.429)
        self.assertAlmostEqual(cam.dist[0], -0.382201)

    def test_offset_blob_gives_the_offset(self):
        pol = self.pol
        for side, xy in REF.items():
            det, q, _ = _pregrasp(pol, side, xy)
            for off in ((0.012, 0.0), (0.0, -0.015), (-0.01, 0.02), (0.02, 0.015), (0.0, 0.0)):
                for shape in ("cylinder", "cube"):
                    rgb = _render(pol, side, q, [_obj(pol, np.add(xy, off), shape=shape)])
                    r = pol.wrist.measure(rgb, side, q, det)
                    self.assertTrue(r.ok, (side, off, shape, r.summary()))
                    self.assertLess(r.area / r.area_exp, 1.6, r.summary())   # merged pair >= 2.1
                    np.testing.assert_allclose(r.corr, off, atol=0.003, err_msg=f"{side} {off} {shape} {r.summary()}")

    def test_no_blob_no_correction(self):
        pol = self.pol
        det, q, _ = _pregrasp(pol, "left", REF["left"])
        for objs in ([], [_obj(pol, REF["left"], color="blue")]):
            r = pol.wrist.measure(_render(pol, "left", q, objs), "left", q, det)
            self.assertFalse(r.ok, r.summary())
            np.testing.assert_array_equal(r.corr, [0.0, 0.0])
            self.assertIn("no red blob", r.reason)
        r = pol.wrist.measure(np.zeros((480, 640, 3), np.uint8), "left", q, det)   # camera without a frame
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "no wrist frame")

    def test_inconsistent_frames_are_refused(self):
        pol = self.pol
        xy = np.asarray(REF["left"])
        det, q, _ = _pregrasp(pol, "left", xy)
        far = pol.wrist.measure(_render(pol, "left", q, [_obj(pol, xy + [0.045, 0.0])]), "left", q, det)
        self.assertFalse(far.ok)                                                     # > max_corr_m (3 cm)
        two = pol.wrist.measure(_render(pol, "left", q, [_obj(pol, xy + [0.0, 0.024]), _obj(pol, xy - [0.0, 0.024])]),
                                "left", q, det)
        self.assertFalse(two.ok)
        self.assertIn("ambiguous", two.reason)
        wide = WristAligner(_wa_cfg(max_corr_m=0.06), pol.cal.cfg, pol.kin, pol.perception, pol.scene)
        other = wide.measure(_render(pol, "left", q, [_obj(pol, xy + [0.0, -0.012], color="yellow"),
                                                      _obj(pol, xy + [0.0, 0.035])]), "left", q, det)
        self.assertFalse(other.ok)
        self.assertIn("colour mismatch", other.reason)
        small = WristAligner(_wa_cfg(area_ratio=[3.0, 5.0]), pol.cal.cfg, pol.kin, pol.perception, pol.scene)
        r = small.measure(_render(pol, "left", q, [_obj(pol, xy)]), "left", q, det)
        self.assertFalse(r.ok)
        self.assertIn("area", r.reason)

    def test_in_view(self):
        pol = self.pol
        det, q, _ = _pregrasp(pol, "left", REF["left"])
        self.assertTrue(pol.wrist.in_view("left", q, det)[0])
        self.assertFalse(pol.wrist.in_view("left", pol.home, det)[0])


class WristMock(RenderMockBackend):
    """Mock backend that also renders the wrist images (test mount) whenever the commanded arm is still."""

    def __init__(self, *a, wrist=None, **kw):
        super().__init__(*a, **kw)
        self.wrist, self._last_cmd, self._wcache = wrist, None, (None, None)

    def images(self):
        out = super().images()
        still = self._last_cmd is not None and np.array_equal(self._last_cmd, self.arm_cmd)
        self._last_cmd = np.array(self.arm_cmd, copy=True)
        if still:
            key = (tuple(np.round(self.arm_q, 4)), tuple(np.round(np.concatenate([o["pos"] for o in self.objects]), 4)))
            if self._wcache[0] != key:
                imgs = {s: _render(self.wrist, s, self.arm_q, self.objects) for s in ("left", "right")}
                self._wcache = (key, imgs)
            out["left_wrist_rgb"] = self._wcache[1]["left"].copy()
            out["right_wrist_rgb"] = self._wcache[1]["right"].copy()
        return out


def _episode(cfg, seed, cal=None, wrist_backend=False):
    """Run one mock episode; returns (actions sha256, info, policy, env)."""
    env = make_mock_env(seed=seed, cfg=cfg)
    pol = ClassicalPolicy(config=copy.deepcopy(cfg), calibration=cal, log=None, safety=env.filter.cfg)
    if wrist_backend:
        b = env.backend
        env.backend = WristMock(env.scene, Calibration.load(), b.kin, b.post, seed=seed, wrist=pol)
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    h = hashlib.sha256()
    while True:
        a = pol.act(obs)
        h.update(np.ascontiguousarray(a["arm_q_target"], np.float64).tobytes())
        h.update(np.ascontiguousarray(a["hand_q_target"], np.float64).tobytes())
        obs, info = env.step(a)
        if info["episode_over"]:
            break
    return h.hexdigest(), info, pol, env


class TestPolicy(unittest.TestCase):
    def test_flag_off_never_builds_the_aligner(self):
        pol = ClassicalPolicy(config=load_config(overlay="submission"), log=None)
        self.assertIsNone(pol.wrist)
        with mock.patch("aaltonomous.baselines.classical.wrist_align.WristAligner.measure",
                        side_effect=AssertionError("measured with the flag off")):
            _episode(load_config(overlay="none"), seed=0)

    def test_flag_on_without_a_view_is_identical_to_off(self):
        """Shipped calibration (no wrist mount): no model -> every pick logs a skip, actions byte-identical to the flag
        off. Same with the sim fallback model, which does not see the object at the pregrasp pose."""
        off = load_config(overlay="none")
        h_off, _, _, _ = _episode(off, seed=1)
        for fb in (False, True):
            on = load_config(overlay="none", overrides={"grasp.wrist_align": True,
                                                        "grasp.wrist_align_cfg.use_fallback": fb})
            h_on, info, pol, _ = _episode(on, seed=1)
            self.assertEqual(h_on, h_off, fb)
            self.assertIsNotNone(pol.wrist)
            self.assertEqual(pol.wrist_results, [])

    def test_fov_guard_rejects_far_points_under_strong_distortion(self):
        pol = ClassicalPolicy(config=_wa_cfg(use_fallback=True), log=None)
        T = np.eye(4)
        w = pol.wrist
        self.assertTrue(w.fov_ok("left", T, [0.0, 0.0, 0.3]))
        self.assertTrue(w.fov_ok("left", T, [0.15, 0.1, 0.3]))
        self.assertFalse(w.fov_ok("left", T, [0.18, -0.15, 0.05]))   # the m1 case: projectPoints gave uv ~ -6e5
        self.assertFalse(w.fov_ok("left", T, [0.0, 0.0, -0.3]))

    def test_head_bias_is_corrected_end_to_end(self):
        """Head detections biased by (+15, -10) mm: with the wrist mount the picks are re-aimed onto the true object."""
        bias = np.array([0.015, -0.010])
        cfg = _wa_cfg()
        cfg["perception"]["xy_offset"] = bias.tolist()
        truth = []
        orig = WristMock.reset_scene

        def grab(self, layout, instruction):
            orig(self, layout, instruction)
            truth[:] = [(o["color"], np.array(o["pos"][:2])) for o in self.objects]
        with mock.patch.object(WristMock, "reset_scene", grab):
            _, info, pol, env = _episode(cfg, seed=0, cal=test_calibration(), wrist_backend=True)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
        applied = [r for r in pol.wrist_results if r[4]]
        self.assertTrue(applied, [r[3].summary() for r in pol.wrist_results])
        for color, side, top, res, _ in applied:
            true_xy = min((xy for c, xy in truth if c == color), key=lambda xy: np.linalg.norm(xy - top[:2]))
            before, after = np.linalg.norm(top[:2] - true_xy), np.linalg.norm(top[:2] + res.corr - true_xy)
            self.assertGreater(before, 0.010, res.summary())
            self.assertLess(after, 0.004, (side, before, after, res.summary()))
        out = env.backend.outcomes(pol.mapping)
        self.assertNotIn("wrong", out)
        self.assertGreaterEqual(out.count("correct"), 1)

if __name__ == "__main__":
    unittest.main()
