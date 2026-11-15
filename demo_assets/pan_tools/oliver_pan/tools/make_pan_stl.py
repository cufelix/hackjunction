#!/usr/bin/env python3
"""Pan-like novel object for the sim pan test: a disc (the pan) + a box handle, ASCII STL in mm (the scene loader
scales by 0.001 and centres z on the bbox). Mesh frame: disc centre at the origin, handle along +x.

    python3 make_pan_stl.py out/object_pan.stl

Dims (mm): disc R=100, thickness 30 (z 0..30); handle x 95..215, width 25 (y +-12.5), z 10..30 (top flush with the
disc top, 1 cm clear under it so fingers can wrap). Two closed shells; the sim uses convexDecomposition.
"""
import math
import sys

import os
if os.environ.get("PANCAKE"):     # demo prop: a 4 cm radius, 5 mm thick disc, no handle
    R, H = 40.0, 5.0
    HX0 = None
elif os.environ.get("PAN_SMALL"):   # toy pan: disc r 6 cm x 2.5 cm, handle 8 cm beyond the rim, 2 cm wide
    R, H = 60.0, 25.0
    HX0, HX1, HW, HZ0, HZ1 = 55.0, 140.0, 20.0, 14.0, 28.0   # bar 14 mm clear of the counter (angled-up handle)
else:
    R, H = 100.0, 30.0
    HX0, HX1, HW, HZ0, HZ1 = 95.0, 215.0, 25.0, 10.0, 30.0
SEG = 48


def tri(f, a, b, c):
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    ln = math.sqrt(sum(x * x for x in n)) or 1.0
    f.write("facet normal %g %g %g\n outer loop\n" % tuple(x / ln for x in n))
    for p in (a, b, c):
        f.write("  vertex %.4f %.4f %.4f\n" % p)
    f.write(" endloop\nendfacet\n")


def disc(f):
    ring = [(R * math.cos(2 * math.pi * i / SEG), R * math.sin(2 * math.pi * i / SEG)) for i in range(SEG)]
    for i in range(SEG):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % SEG]
        tri(f, (0.0, 0.0, H), (x0, y0, H), (x1, y1, H))          # top (ccw from above)
        tri(f, (0.0, 0.0, 0.0), (x1, y1, 0.0), (x0, y0, 0.0))    # bottom
        tri(f, (x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, H))        # side
        tri(f, (x0, y0, 0.0), (x1, y1, H), (x0, y0, H))


def box(f, x0, x1, y0, y1, z0, z1):
    v = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    for a, b, c, d in quads:
        tri(f, v[a], v[b], v[c])
        tri(f, v[a], v[c], v[d])


def main(path):
    with open(path, "w") as f:
        f.write("solid pan\n")
        disc(f)
        if HX0 is not None:
            box(f, HX0, HX1, -HW / 2, HW / 2, HZ0, HZ1)
        f.write("endsolid pan\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "object_pan.stl")
