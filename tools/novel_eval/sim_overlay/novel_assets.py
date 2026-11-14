"""OUR OWN extra objects for the novel-object evaluation. NOT part of the official evaluator.

Dropped into a COPY of the stock sim tree (see tools/novel_eval/make_novel_tree.sh); the stock tree and
aaltonomous/sim in the repo are never edited.

Object names follow the stock pattern obj_<colour>_<shape>_<k>, so the scene service, state() and the policy side
need no changes:
  * known shapes in NEW colours (green/white/purple/black): the printed STL meshes (cube, cylinder, triangle)
  * ball        sphere, 5 cm
  * pan         small pan: 7 cm disc (3 cm high) + 8 cm handle that floats 2 cm above the table (compound collider)
  * banana, mug, can, marker   YCB meshes from the standard NVIDIA Isaac asset server (Props/YCB/Axis_Aligned),
                downloaded once to $NOVEL_ASSETS, scaled to graspable size; convex-hull collision, visual = the
                original textured mesh
Layout entries for these shapes use `yaw` as the PELVIS yaw (rad, CCW from pelvis +x = away from the robot);
the stock shapes keep the stock mat-yaw convention. Positions are mat coordinates for all.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ASSETS = Path(os.environ.get("NOVEL_ASSETS", "/data/work/novel/assets/Axis_Aligned"))

COLORS = {                      # sRGB 0..1 for the primitive / STL objects
    "green": (0.10, 0.60, 0.20),
    "white": (0.92, 0.92, 0.92),
    "purple": (0.45, 0.18, 0.62),
    "black": (0.05, 0.05, 0.05),
}
STL_NOVEL = [("green", "cube"), ("green", "cylinder"), ("white", "cylinder"), ("white", "triangle"),
             ("purple", "triangle"), ("purple", "cube"), ("black", "cube"), ("black", "cylinder")]
BALLS = ["green", "white", "purple"]
PANS = ["black", "purple"]
# natural-colour YCB: shape -> (usd, scale, rot_x_deg, mass_kg, collision approximation)
YCB = {
    "banana": ("011_banana.usd", 0.55, 0.0, 0.05, "convexHull"),
    "mug": ("025_mug.usd", 0.65, 0.0, 0.12, "convexHull"),
    "can": ("005_tomato_soup_can.usd", 0.60, 90.0, 0.08, "convexHull"),   # authored lying; stand it up
    "marker": ("040_large_marker.usd", 1.00, 0.0, 0.03, "convexHull"),
}
BALL_R = 0.025
PAN_R, PAN_H, PAN_HL, PAN_HW, PAN_HT = 0.035, 0.030, 0.080, 0.016, 0.008

NOVEL_SHAPES = {"ball", "pan", *YCB}                       # shapes that are not one of the printed STL shapes


def names():
    n = [f"obj_{c}_{s}_0" for c, s in STL_NOVEL]
    n += [f"obj_{c}_ball_0" for c in BALLS] + [f"obj_{c}_pan_0" for c in PANS]
    n += [f"obj_natural_{s}_0" for s in YCB]
    return n


NAMES = names()
SPECIAL = {n for n in NAMES if n.split("_")[2] in NOVEL_SHAPES}      # not spawned by the STL path


def _shape(name):
    return name.split("_")[2]


def color_rgb(name_or_color):
    return COLORS.get(name_or_color)


def _ycb_mesh(shape):
    """Collision/bounds mesh of a YCB object after scale + rotation (trimesh, metres, bbox-centred)."""
    import trimesh
    from pxr import Usd, UsdGeom
    usd, scale, rx, _m, _a = YCB[shape]
    st = Usd.Stage.Open(str(ASSETS / usd))
    prim = next(p for p in st.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh = UsdGeom.Mesh(prim)
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)).T          # column vectors
    pts = np.asarray(mesh.GetPointsAttr().Get(), float)
    pts = (xf @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    cnt = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
    faces, o = [], 0
    for c in cnt:                                                                       # fan triangulation
        for k in range(1, c - 1):
            faces.append((idx[o], idx[o + k], idx[o + k + 1]))
        o += c
    t = np.radians(rx)
    R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    m = trimesh.Trimesh(vertices=(R @ (pts * scale).T).T, faces=np.asarray(faces), process=False)
    lo, hi = m.bounds
    m.apply_translation(-(lo + hi) / 2)
    return m


_REST_Z = {}


def rest_z(name):
    """Object centre height above the surface it rests on (m) when placed upright / flat (+ 2 mm drop)."""
    if name in _REST_Z:
        return _REST_Z[name]
    s = _shape(name)
    if s == "ball":
        z = BALL_R
    elif s == "pan":
        z = PAN_H / 2
    elif s in YCB:                                  # half heights of the scaled / rotated meshes (pxr must not be
        z = {"banana": 0.0204, "mug": 0.0303, "can": 0.0306, "marker": 0.0097}[s]   # imported before the Kit app starts)
    else:
        z = 0.0225
    _REST_Z[name] = z + 0.002
    return _REST_Z[name]


def demo_camera_pose():
    return (0.95, 0.70, 1.28), (0.38, 0.02, 0.90)


def special_cfg(name, env_ns, pos_w, quat_xyzw, rotmat_to_quat_xyzw):
    """RigidObjectCfg for a non-STL novel object."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    PanCfg, YcbCfg = _classes()
    s, color = _shape(name), name.split("_")[1]
    init = RigidObjectCfg.InitialStateCfg(pos=pos_w, rot=quat_xyzw)
    fr = dict(static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
    phys = sim_utils.RigidBodyMaterialCfg(friction_combine_mode="max", restitution_combine_mode="min", **fr)
    rigid = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0)
    coll = sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.004, rest_offset=0.0)
    path = f"{env_ns}/{name}"
    if s == "ball":
        spawn = sim_utils.SphereCfg(
            radius=BALL_R, rigid_props=rigid, mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
            collision_props=coll, physics_material=phys,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[color], roughness=0.5, metallic=0.0))
    elif s == "pan":
        spawn = PanCfg(rigid_props=rigid, mass_props=sim_utils.MassPropertiesCfg(mass=0.09), collision_props=coll,
                       physics_material=phys,
                       visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[color], roughness=0.4,
                                                                   metallic=0.3))
    else:
        usd, scale, rx, mass, approx = YCB[s]
        spawn = YcbCfg(shape=s, rigid_props=rigid, mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                       collision_props=coll, physics_material=phys, approximation=approx)
    return RigidObjectCfg(prim_path=path, init_state=init, spawn=spawn)


def _define_classes():
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    @configclass
    class PanCfg(sim_utils.SpawnerCfg):
        func: object = None
        rigid_props: object = None
        mass_props: object = None
        collision_props: object = None
        physics_material: object = None
        visual_material: object = None

        def __post_init__(self):
            self.func = spawn_pan

    @clone
    def spawn_pan(prim_path, cfg, translation=None, orientation=None, **kwargs):
        from isaaclab.sim import schemas
        from isaaclab.sim.utils import create_prim, get_current_stage
        stage = get_current_stage()
        create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
        common = dict(collision_props=cfg.collision_props, physics_material=cfg.physics_material,
                      visual_material=cfg.visual_material)
        body = sim_utils.CylinderCfg(radius=PAN_R, height=PAN_H, axis="Z", **common)
        body.func(prim_path + "/body", body, translation=(0.0, 0.0, 0.0))
        handle = sim_utils.CuboidCfg(size=(PAN_HL, PAN_HW, PAN_HT), **common)
        handle.func(prim_path + "/handle", handle, translation=(PAN_R - 0.005 + PAN_HL / 2, 0.0, PAN_H / 2 - 0.007))
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage)
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage)
        return stage.GetPrimAtPath(prim_path)

    @configclass
    class YcbCfg(sim_utils.SpawnerCfg):
        func: object = None
        shape: str = "mug"
        rigid_props: object = None
        mass_props: object = None
        collision_props: object = None
        physics_material: object = None
        approximation: str = "convexHull"

        def __post_init__(self):
            self.func = spawn_ycb

    @clone
    def spawn_ycb(prim_path, cfg, translation=None, orientation=None, **kwargs):
        """Physics proxy = hull of the scaled mesh (invisible); visual = the original textured USD, scaled."""
        from pxr import Gf, UsdGeom, UsdPhysics
        from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
        usd, scale, rx, _m, approx = YCB[cfg.shape]
        mesh = _ycb_mesh(cfg.shape)
        proxy = sim_utils.MeshCfg(rigid_props=cfg.rigid_props, mass_props=cfg.mass_props,
                                  collision_props=cfg.collision_props, physics_material=cfg.physics_material,
                                  visual_material=None)
        prim = _spawn_mesh_geom_from_mesh(prim_path, proxy, mesh, translation, orientation, **kwargs)
        stage = prim.GetStage()
        mp = stage.GetPrimAtPath(prim_path + "/geometry/mesh")
        UsdPhysics.MeshCollisionAPI.Apply(mp).GetApproximationAttr().Set(cfg.approximation)
        UsdGeom.Imageable(mp).MakeInvisible()
        vis = UsdGeom.Xform.Define(stage, prim_path + "/vis")
        lo, hi = _raw_bounds(cfg.shape)
        api = UsdGeom.XformCommonAPI(vis.GetPrim())
        api.SetScale(Gf.Vec3f(scale, scale, scale))
        api.SetRotate(Gf.Vec3f(rx, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        # the proxy mesh is re-centred on its bbox; shift the visual by the same offset (rotated + scaled)
        t = np.radians(rx)
        R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
        off = -(R @ ((lo + hi) / 2 * scale))
        api.SetTranslate(Gf.Vec3d(*[float(v) for v in off]))
        inner = stage.DefinePrim(prim_path + "/vis/inner", "Xform")
        inner.GetReferences().AddReference(str(ASSETS / usd))
        return prim

    return PanCfg, YcbCfg


def _raw_bounds(shape):
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(ASSETS / YCB[shape][0]))
    bb = UsdGeom.BBoxCache(0.0, ["default"]).ComputeWorldBound(st.GetDefaultPrim()).ComputeAlignedRange()
    return np.array(bb.GetMin(), float), np.array(bb.GetMax(), float)


_CLS = []


def _classes():
    """Spawner classes, built lazily: importing isaaclab before the Kit app is launched breaks the sim start."""
    if not _CLS:
        _CLS.extend(_define_classes())
    return _CLS


def demo_camera_cfg(env_ns, rotmat_to_quat_xyzw):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    pos, tgt = (np.asarray(v, float) for v in demo_camera_pose())
    f = tgt - pos
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    R = np.c_[right, down, f]
    return CameraCfg(
        prim_path=f"{env_ns}/DemoCam", update_period=1.0 / 15, height=450, width=800, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in pos), rot=rotmat_to_quat_xyzw(R), convention="ros"))
