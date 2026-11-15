"""Felix's VLM use-case scenes: lime, short teacup, knife props (AALTO_TOOLS=1) + run_pan variants (idempotent)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, "/data/work/oliver/pan")
from tools_patch_lib import Mesh, sub  # noqa: E402

TREE = Path("/data/aaltonomous-humanoid-pan/aaltonomous")
RUN = Path("/data/work/oliver/pan/tools/run_pan.py")
A = TREE / "scene/assets"

m = Mesh()                                   # lime: 50 mm sphere (lat-long, convex)
R, NU, NV = 25.0, 16, 10
pt = lambda i, j: (R * math.sin(math.pi * j / NV) * math.cos(2 * math.pi * i / NU),
                   R * math.sin(math.pi * j / NV) * math.sin(2 * math.pi * i / NU), R + R * math.cos(math.pi * j / NV))
for j in range(NV):
    for i in range(NU):
        a, b, c, d = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
        if j > 0:
            m.tris.append((a, c, b))
        if j < NV - 1:
            m.tris.append((a, d, c))
m.write(A / "object_lime.stl", "lime")

m = Mesh()                                   # teacup: 70 dia x 65 tall, open top, 3 mm wall, side handle along +x
SEG, RO, T, H, TB = 16, 35.0, 3.0, 65.0, 4.0
for i in range(SEG):
    a0, a1 = 2 * math.pi * i / SEG, 2 * math.pi * (i + 1) / SEG
    c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
    ri = RO - T
    m.hexa([(ri * c0, ri * s0, 0), (RO * c0, RO * s0, 0), (RO * c1, RO * s1, 0), (ri * c1, ri * s1, 0),
            (ri * c0, ri * s0, H), (RO * c0, RO * s0, H), (RO * c1, RO * s1, H), (ri * c1, ri * s1, H)])
m.hexa([(-ri, -ri, 0), (ri, -ri, 0), (ri, ri, 0), (-ri, ri, 0), (-ri, -ri, TB), (ri, -ri, TB), (ri, ri, TB), (-ri, ri, TB)]) if False else None
n = 24                                       # base disc
for i in range(n):
    a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
    p0, p1 = (RO * math.cos(a0), RO * math.sin(a0)), (RO * math.cos(a1), RO * math.sin(a1))
    m.tris += [((0, 0, TB), (p0[0], p0[1], TB), (p1[0], p1[1], TB)), ((0, 0, 0), (p1[0], p1[1], 0), (p0[0], p0[1], 0)),
               ((p0[0], p0[1], 0), (p1[0], p1[1], 0), (p1[0], p1[1], TB)), ((p0[0], p0[1], 0), (p1[0], p1[1], TB), (p0[0], p0[1], TB))]
m.box(RO - 2, RO + 22, -5, 5, 46, 54)        # handle: top bar, side bar, bottom bar (a closed loop)
m.box(RO + 14, RO + 22, -5, 5, 18, 54)
m.box(RO - 2, RO + 22, -5, 5, 18, 26)
m.write(A / "object_teacup.stl", "teacup")

m = Mesh()                                   # knife 210 mm: handle 100 x 25 x 20, flat blade 110 x 20 x 2
m.box(-105, -5, -12.5, 12.5, 0, 20)
m.box(-5, 105, -10, 10, 0, 2)
m.write(A / "object_knife.stl", "knife")

lay = TREE / "sim/aalto_task/layout.py"
sub(lay, '"obj_dgrey_hammer_0"]', '"obj_dgrey_hammer_0", "obj_green_lime_0", "obj_white_teacup_0", "obj_dgrey_knife_0"]')
cfg = TREE / "sim/aalto_task/scene_cfg.py"
sub(cfg, '"dgrey": (0.30, 0.30, 0.32)}', '"dgrey": (0.30, 0.30, 0.32), "white": (0.93, 0.93, 0.93)}')
sub(cfg, '"pliers": 0.12, "hammer": 0.10}', '"pliers": 0.12, "hammer": 0.10, "lime": 0.06, "teacup": 0.06, "knife": 0.05}')
sub(cfg, '("pan", "screwdriver", "wrench", "pliers", "hammer")', '("pan", "screwdriver", "wrench", "pliers", "hammer", "teacup", "knife")')
# lime colour: green H~70 (yellow ends at 60); the TOOL_RGB green is that already -> lime reuses "green"

sub(RUN, '"hammer": 0.0135}', '"hammer": 0.0135, "lime": 0.026, "teacup": 0.0335, "knife": 0.011}')
sub(RUN, '        tz = {', '        tz = {')
sub(RUN, '    "meat_right":', '''    # Felix's VLM use cases (screwdriver sideways along y; left-bin targets on +y for the left arm, right-bin on -y)
    "thirsty_left": (None, 0.0, [("tool", "white", "teacup", (0.30, 0.12), -90.0), ("tool", "green", "screwdriver", (0.255, -0.08), 90.0),
                                 ("tool", "dgrey", "hammer", (0.345, -0.10), 90.0), ("tool", "black", "pliers", (0.30, 0.28), 0.0)]),
    "tidy":         (None, 0.0, [("tool", "green", "screwdriver", (0.30, 0.12), 90.0), ("tool", "green", "lime", (0.30, -0.12), 0.0),
                                 ("tool", "white", "teacup", (0.33, -0.25), 90.0)]),
    "restraint":    (None, 0.0, [("tool", "dgrey", "knife", (0.255, 0.0), 90.0), ("tool", "green", "lime", (0.335, -0.15), 0.0),
                                 ("tool", "white", "teacup", (0.335, 0.15), -90.0)]),
    "meat_right":''')
sub(RUN, 'if any(t in o["name"] for t in ("screwdriver", "wrench", "pliers", "hammer")):',
    'if any(t in o["name"] for t in ("screwdriver", "wrench", "pliers", "hammer", "lime", "teacup", "knife")):')
print("patched")
