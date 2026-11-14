"""Apply the novel-eval overlay to a COPY of the stock sim tree: python3 apply_overlay.py <tree>.
Exact-string edits of aalto_task/{layout,scene_cfg,hooks}.py; fails loudly if the stock files differ."""
import sys
from pathlib import Path

tree = Path(sys.argv[1])
T = tree / "aaltonomous/sim/aalto_task"


def sub(path, old, new, count=1):
    s = path.read_text()
    if "novel_assets" in s and old in new and new in s:
        return
    if s.count(old) != count:
        sys.exit(f"{path.name}: expected {count}x of {old[:60]!r}, found {s.count(old)}")
    path.write_text(s.replace(old, new))


# ---------------------------------------------------------------- layout.py
L = T / "layout.py"
if "novel_assets" not in L.read_text():
    sub(L, "from aaltonomous.scene.scene import Scene\n", "from aaltonomous.scene.scene import Scene\nfrom . import novel_assets as NA\n")
    sub(L, '''    o = sc.cfg["objects"]
    return [f"obj_{c}_{s}_{k}" for s in o["shapes"] for c in o["colors"] for k in range(int(o["copies_per_type"]))]''',
        '''    o = sc.cfg["objects"]
    return [f"obj_{c}_{s}_{k}" for s in o["shapes"] for c in o["colors"] for k in range(int(o["copies_per_type"]))] + NA.NAMES''')
    sub(L, '''def park_pose(sc: Scene, i, n):
    """Pelvis-frame pose for parked object i of n: on the floor, in a row behind the robot."""
    y = (i - (n - 1) / 2) * 0.10
    z = -PELVIS_W[2] + sc.object_height() / 2 + 0.002''',
        '''def park_pose(sc: Scene, i, n, name=None):
    """Pelvis-frame pose for parked object i of n: on the floor, in a row behind the robot."""
    y = -0.52 + 0.09 * (i % 8)                 # parked on the floor UNDER the counter (hidden from the demo camera)
    z = -PELVIS_W[2] + (NA.rest_z(name) if name in NA.SPECIAL else sc.object_height() / 2 + 0.002)''')
    sub(L, '''        m = sc.cube_spawn_mat(ob)
        if "z" in ob:''', '''        m = sc.cube_spawn_mat(ob)
        special = pick in NA.SPECIAL
        if special:
            m[2] = NA.rest_z(pick)
        if "z" in ob:''')
    sub(L, '''        poses[pick] = (p, yaw_quat_xyzw(mat_yaw_to_pelvis(ob.get("yaw", 0.0))))''',
        '''        poses[pick] = (p, yaw_quat_xyzw(float(ob.get("yaw", 0.0)) if special else mat_yaw_to_pelvis(ob.get("yaw", 0.0))))''')
    sub(L, "poses[n] = park_pose(sc, names.index(n), len(names))", "poses[n] = park_pose(sc, names.index(n), len(names), n)")
    sub(L, "    return np.r_[PARK_X, y, z], yaw_quat_xyzw(0.0)", "    return np.r_[0.22 + 0.10 * (i // 8), y, z], yaw_quat_xyzw(0.0)")

# ---------------------------------------------------------------- scene_cfg.py
S = T / "scene_cfg.py"
if "novel_assets" not in S.read_text():
    sub(S, "from .layout import ARM_HOME, PELVIS_W, assign, object_names, parse_name\n",
        "from .layout import ARM_HOME, PELVIS_W, assign, object_names, parse_name\nfrom . import novel_assets as NA\n")
    sub(S, '''def _rgb(name):
    return tuple(c / 255.0 for c in SC.colors[name])''', '''def _rgb(name):
    if name in NA.COLORS:
        return NA.COLORS[name]
    return tuple(c / 255.0 for c in SC.colors[name])''')
    sub(S, '''        color, shape, _ = parse_name(n)
        p, q = poses[n]
        out[n] = RigidObjectCfg(''', '''        color, shape, _ = parse_name(n)
        p, q = poses[n]
        if n in NA.SPECIAL:
            out[n] = NA.special_cfg(n, ENV, _w(p), tuple(float(v) for v in q), rotmat_to_quat_xyzw)
            continue
        out[n] = RigidObjectCfg(''')
    sub(S, '''        "right_wrist_camera": CameraPresets.right_dex3_wrist_camera(),
    }''', '''        "right_wrist_camera": CameraPresets.right_dex3_wrist_camera(),
        **({"demo_camera": NA.demo_camera_cfg(ENV, rotmat_to_quat_xyzw)} if os.environ.get("NOVEL_CAM") == "1" else {}),
    }''')
    sub(S, "from pathlib import Path\n", "import os\nfrom pathlib import Path\n")

# ---------------------------------------------------------------- hooks.py
H = T / "hooks.py"
if "_rec_dir" not in H.read_text():
    sub(H, "        self.step_dt = float(env.step_dt)\n", "        self.step_dt = float(env.step_dt)\n        self._rec_dir, self._rec_n, self._rec_last = None, 0, None\n")
    sub(H, '''        if cmd == "state":
            return {"ok": True, "objects": self.state(env), "layout": self.layout}''', '''        if cmd == "state":
            return {"ok": True, "objects": self.state(env), "layout": self.layout}
        if cmd == "rec_start":      # novel-eval: save demo-camera frames (jpg) of the next episode into req["dir"]
            os.makedirs(req["dir"], exist_ok=True)
            self._rec_dir, self._rec_n, self._rec_last = req["dir"], 0, None
            return {"ok": True, "has_cam": "demo_camera" in env.scene.keys()}
        if cmd == "rec_stop":
            n, self._rec_dir = self._rec_n, None
            return {"ok": True, "frames": n}''')
    sub(H, '''        if self.rgbd is not None:
            self._publish_rgbd(env)
''', '''        if self.rgbd is not None:
            self._publish_rgbd(env)
        if self._rec_dir is not None and "demo_camera" in env.scene.keys():
            try:
                self._rec_frame(env)
            except Exception:
                import traceback
                traceback.print_exc()
                self._rec_dir = None

    def _rec_frame(self, env):
        import cv2
        cam = env.scene["demo_camera"]
        n = int(env.common_step_counter)          # 15 Hz of sim time (step_dt 0.01 s): the camera frame id does not advance
        if self._rec_last is not None and n - self._rec_last < 7:
            return
        self._rec_last = n
        rgb = _t(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(self._rec_dir, f"{self._rec_n:06d}.jpg"), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        self._rec_n += 1
''')
print("overlay applied to", tree)
