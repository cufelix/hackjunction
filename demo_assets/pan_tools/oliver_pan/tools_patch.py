"""Tool-table demo props for the pan sim (AALTO_TOOLS=1) + run_pan variants tools_left/tools_right (idempotent).
Meshes are procedural (own geometry), mm, lying flat on the counter (z 0..h), long axis along mesh +x."""
import math
from pathlib import Path

TREE = Path("/data/aaltonomous-humanoid-pan/aaltonomous")
RUN = Path("/data/work/oliver/pan/tools/run_pan.py")
ASSETS = TREE / "scene/assets"


def sub(p, old, new):
    s = p.read_text()
    if new in s:
        return
    assert old in s, (p, old[:70])
    p.write_text(s.replace(old, new, 1))


class Mesh:
    def __init__(self):
        self.tris = []

    def hexa(self, v):   # v[0..3] bottom ccw from above, v[4..7] top
        for a, b, c, d in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
            self.tris += [(v[a], v[b], v[c]), (v[a], v[c], v[d])]

    def box(self, x0, x1, y0, y1, z0, z1, yaw_deg=0.0, pivot=(0.0, 0.0)):
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        r = lambda x, y: (pivot[0] + (x - pivot[0]) * c - (y - pivot[1]) * s, pivot[1] + (x - pivot[0]) * s + (y - pivot[1]) * c)
        q = [r(x0, y0), r(x1, y0), r(x1, y1), r(x0, y1)]
        self.hexa([(x, y, z0) for x, y in q] + [(x, y, z1) for x, y in q])

    def xcyl(self, x0, x1, rad, zc, n=16):   # cylinder along x, lying on the counter
        ring = [(rad * math.cos(2 * math.pi * i / n), zc + rad * math.sin(2 * math.pi * i / n)) for i in range(n)]
        for i in range(n):
            (y0, z0), (y1, z1) = ring[i], ring[(i + 1) % n]
            self.tris += [((x0, 0, zc), (x0, y1, z1), (x0, y0, z0)), ((x1, 0, zc), (x1, y0, z0), (x1, y1, z1)),
                          ((x0, y0, z0), (x0, y1, z1), (x1, y1, z1)), ((x0, y0, z0), (x1, y1, z1), (x1, y0, z0))]

    def write(self, path, name):
        with open(path, "w") as f:
            f.write(f"solid {name}\n")
            for p, q, r in self.tris:
                u = [q[k] - p[k] for k in range(3)]
                v = [r[k] - p[k] for k in range(3)]
                n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
                ln = math.sqrt(sum(x * x for x in n)) or 1.0
                f.write("facet normal %g %g %g\n outer loop\n" % tuple(x / ln for x in n))
                for pt in (p, q, r):
                    f.write("  vertex %.4f %.4f %.4f\n" % tuple(pt))
                f.write(" endloop\nendfacet\n")
            f.write(f"endsolid {name}\n")


m = Mesh()                                   # screwdriver 180 mm: handle 100 x 30 dia, shaft 80 x 6 dia
m.xcyl(-90, 10, 15.0, 15.0)
m.xcyl(10, 90, 3.0, 15.0, n=8)
m.write(ASSETS / "object_screwdriver.stl", "screwdriver")
m = Mesh()                                   # wrench 150 mm, 8 mm flat: bar + open jaw at +x
m.box(-75, 45, -7, 7, 0, 8)
m.box(45, 75, -16, -4, 0, 8)
m.box(45, 75, 4, 16, 0, 8)
m.box(40, 50, -16, 16, 0, 8)
m.write(ASSETS / "object_wrench.stl", "wrench")
m = Mesh()                                   # pliers 140 mm, 13 mm high: jaws + two diverging handles
m.box(20, 70, -8, 8, 0, 13)
m.box(-70, 20, 2, 12, 0, 13, yaw_deg=-6, pivot=(20, 7))
m.box(-70, 20, -12, -2, 0, 13, yaw_deg=6, pivot=(20, -7))
m.write(ASSETS / "object_pliers.stl", "pliers")
m = Mesh()                                   # hammer 180 mm: handle 16 x 14, head 25 x 90 x 25 at +x
m.box(-90, 65, -8, 8, 0, 14)
m.box(65, 90, -45, 45, 0, 25)
m.write(ASSETS / "object_hammer.stl", "hammer")

lay = TREE / "sim/aalto_task/layout.py"
sub(lay, 'MEAT_NAME = "obj_brown_meat_0"',
    'MEAT_NAME = "obj_brown_meat_0"\nTOOL_NAMES = ["obj_green_screwdriver_0", "obj_silver_wrench_0", "obj_black_pliers_0", '
    '"obj_dgrey_hammer_0"]   # AALTO_TOOLS=1: tool-table demo')
sub(lay, "    return names + ([PAN_NAME] if pan_enabled() else []) + pancake + meat",
    '    tools = TOOL_NAMES if pan_enabled() and os.environ.get("AALTO_TOOLS", "0") == "1" else []\n'
    "    return names + ([PAN_NAME] if pan_enabled() else []) + pancake + meat + tools")

cfg = TREE / "sim/aalto_task/scene_cfg.py"
sub(cfg, "MEAT_RGB, MEAT_MASS = (0.24, 0.20, 0.18), 0.06",
    "MEAT_RGB, MEAT_MASS = (0.24, 0.20, 0.18), 0.06\n"
    "# tool table: no known HSV class (green H~75 > yellow's 60, greys/black low S)\n"
    'TOOL_RGB = {"green": (0.10, 0.50, 0.30), "silver": (0.62, 0.62, 0.65), "black": (0.10, 0.10, 0.11), '
    '"dgrey": (0.30, 0.30, 0.32)}\n'
    'TOOL_MASS = {"screwdriver": 0.06, "wrench": 0.10, "pliers": 0.12, "hammer": 0.10}')
sub(cfg, '    if name == "brown":\n        return MEAT_RGB',
    '    if name == "brown":\n        return MEAT_RGB\n    if name in TOOL_RGB:\n        return TOOL_RGB[name]')
sub(cfg, '{"pan": PAN_MASS, "pancake": PANCAKE_MASS, "meat": MEAT_MASS}.get(shape, OBJ_MASS)',
    '{"pan": PAN_MASS, "pancake": PANCAKE_MASS, "meat": MEAT_MASS, **TOOL_MASS}.get(shape, OBJ_MASS)')
sub(cfg, 'approximation="convexDecomposition" if shape == "pan" else "convexHull",',
    'approximation="convexDecomposition" if shape in ("pan", "screwdriver", "wrench", "pliers", "hammer") else "convexHull",')

# run_pan: tool variants (no pan in the layout), tool entries, final tool positions + bin
sub(RUN, '    "meat_right":', '''    # tool table (AALTO_TOOLS=1, no pan): screwdriver sideways (axis along y) on the side of the arm that serves the
    # target bin; distractors >= 5 cm apart, x 0.25..0.35 (free strip), long axes along y
    "tools_left":  (None, 0.0, [("tool", "green", "screwdriver", (0.30, 0.12), 90.0), ("tool", "silver", "wrench", (0.255, -0.08), 90.0),
                                ("tool", "dgrey", "hammer", (0.345, -0.10), 90.0), ("tool", "black", "pliers", (0.30, 0.28), 0.0)]),
    "tools_right": (None, 0.0, [("tool", "green", "screwdriver", (0.30, -0.12), -90.0), ("tool", "silver", "wrench", (0.255, 0.08), -90.0),
                                ("tool", "dgrey", "hammer", (0.345, 0.10), -90.0), ("tool", "black", "pliers", (0.30, -0.28), 0.0)]),
    "meat_right":''')
sub(RUN, '    lay.append(entry("grey", "pan", (px, py), theta, PAN_Z_MAT))\n',
    '    if VARIANTS[name][0] is None:   # tool table: no pan\n'
    '        tz = {"screwdriver": 0.016, "wrench": 0.005, "pliers": 0.0075, "hammer": 0.0135}   # bbox half height + 1 mm\n'
    '        return [entry(c, s, xy, yaw, tz[s]) for _, c, s, xy, yaw in others]\n'
    '    lay.append(entry("grey", "pan", (px, py), theta, PAN_Z_MAT))\n')
sub(RUN, "    (px, py), theta, others = VARIANTS[name]\n",
    "    pos, theta, others = VARIANTS[name]\n    px, py = pos if pos is not None else (0.0, 0.0)\n")
sub(RUN, '            v = verdict(poll.rows, env.scene)\n',
    '            v = verdict(poll.rows, env.scene)\n'
    '            if VARIANTS[vname][0] is None:   # tool table: where did each tool end up\n'
    '                fin, bins = {}, {k: env.scene.mat_to_pelvis(np.array([c[0], c[1], 0.0]))\n'
    '                                 for k, c in env.scene.cfg["bins"]["centers"].items()}\n'
    '                for o in sc.call(cmd="state")["objects"]:\n'
    '                    if any(t in o["name"] for t in ("screwdriver", "wrench", "pliers", "hammer")):\n'
    '                        p = o["pos"]\n'
    '                        inb = [k for k, c in bins.items() if abs(p[0] - c[0]) < 0.105 and abs(p[1] - c[1]) < 0.105]\n'
    '                        fin[o["name"]] = {"pos": [round(float(x), 3) for x in p], "bin": inb[0] if inb else None}\n'
    '                v = {"tools": fin, "verdict": "tools"}\n'
    '                print(f"{vname:11s} tools={json.dumps(fin)}", flush=True)\n')
print("patched")
