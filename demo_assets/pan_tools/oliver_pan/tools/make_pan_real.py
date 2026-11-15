#!/usr/bin/env python3
"""Realistic frying pan for the demo video, procedural (own geometry, no third-party asset/licence). ASCII STL in mm.

    python3 make_pan_real.py out/object_pan.stl

Mesh frame: pan centre at the origin, counter at z=0, handle along +x.
  body   flared skillet: base r 70 (14 cm), rim r 90 (18 cm), wall height 40, wall 3 thick, base 4 thick
  handle 150 long x 22 wide x 8 thick, angled up 10 deg, root at the rim (x 86, z 30); bar underside >= 26 mm
         above the counter everywhere, ~39 mm at its middle (grasp zone)
Every part is its own closed convex shell (base disc, 24 wall segments, neck, handle) so PhysX
convexDecomposition recovers them nearly exactly: the bowl stays open (a pancake can lie inside), and the gap
under the handle stays free for the fingers. 27 shells <= the 32 default hull budget.
"""
import math
import os
import sys

SEG = 24
R_RIM = float(os.environ.get("PAN_RIM_R", "90"))              # 90 = 18 cm skillet; 75 = 15 cm egg pan
R_BASE = round(R_RIM * 70.0 / 90.0, 1)
H_WALL, T_WALL, T_BASE = 40.0, 3.0, 4.0
H_LEN, H_W, H_T, H_DEG, H_X0, H_Z0 = 150.0, 22.0, 8.0, 10.0, R_RIM - 4.0, 30.0


def tri(f, a, b, c):
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    ln = math.sqrt(sum(x * x for x in n)) or 1.0
    f.write("facet normal %g %g %g\n outer loop\n" % tuple(x / ln for x in n))
    for p in (a, b, c):
        f.write("  vertex %.4f %.4f %.4f\n" % tuple(p))
    f.write(" endloop\nendfacet\n")


def hexa(f, v):
    """8 corners: v[0..3] bottom ring, v[4..7] top ring, same winding (ccw from above)."""
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for a, b, c, d in quads:
        tri(f, v[a], v[b], v[c])
        tri(f, v[a], v[c], v[d])


def base(f, n=48):
    ring = [(R_BASE * math.cos(2 * math.pi * i / n), R_BASE * math.sin(2 * math.pi * i / n)) for i in range(n)]
    for i in range(n):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % n]
        tri(f, (0, 0, T_BASE), (x0, y0, T_BASE), (x1, y1, T_BASE))
        tri(f, (0, 0, 0), (x1, y1, 0), (x0, y0, 0))
        tri(f, (x0, y0, 0), (x1, y1, 0), (x1, y1, T_BASE))
        tri(f, (x0, y0, 0), (x1, y1, T_BASE), (x0, y0, T_BASE))


def walls(f):
    for i in range(SEG):
        a0, a1 = 2 * math.pi * i / SEG, 2 * math.pi * (i + 1) / SEG
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        ro_b, ri_b, ro_t, ri_t = R_BASE, R_BASE - T_WALL, R_RIM, R_RIM - T_WALL
        hexa(f, [(ri_b * c0, ri_b * s0, 0), (ro_b * c0, ro_b * s0, 0), (ro_b * c1, ro_b * s1, 0), (ri_b * c1, ri_b * s1, 0),
                 (ri_t * c0, ri_t * s0, H_WALL), (ro_t * c0, ro_t * s0, H_WALL), (ro_t * c1, ro_t * s1, H_WALL),
                 (ri_t * c1, ri_t * s1, H_WALL)])


def box(f, x0, x1, y0, y1, z0, z1, pitch_deg=0.0, pivot=(0.0, 0.0)):
    th = math.radians(pitch_deg)

    def rot(x, y, z):   # rotate about the y axis through pivot (x,z): +pitch raises +x
        dx, dz = x - pivot[0], z - pivot[1]
        return (pivot[0] + dx * math.cos(th) - dz * math.sin(th), y, pivot[1] + dx * math.sin(th) + dz * math.cos(th))
    hexa(f, [rot(x0, y0, z0), rot(x1, y0, z0), rot(x1, y1, z0), rot(x0, y1, z0),
             rot(x0, y0, z1), rot(x1, y0, z1), rot(x1, y1, z1), rot(x0, y1, z1)])


def main(path):
    with open(path, "w") as f:
        f.write("solid pan\n")
        base(f)
        walls(f)
        box(f, R_RIM - 8.0, H_X0 + 8.0, -14.0, 14.0, H_Z0 - 8.0, H_Z0 + 4.0)            # neck / rivet plate
        box(f, H_X0, H_X0 + H_LEN, -H_W / 2, H_W / 2, H_Z0 - H_T / 2, H_Z0 + H_T / 2,
            pitch_deg=H_DEG, pivot=(H_X0, H_Z0))                                            # handle, 10 deg up
        f.write("endsolid pan\n")
    mid = H_LEN / 2
    print(f"wrote {path}; handle underside at mid {H_Z0 - H_T / 2 + mid * math.sin(math.radians(H_DEG)):.1f} mm, "
          f"tip top {H_Z0 + H_T / 2 + H_LEN * math.sin(math.radians(H_DEG)):.1f} mm, "
          f"span x {-R_RIM:.0f}..{H_X0 + H_LEN * math.cos(math.radians(H_DEG)):.0f} mm")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "object_pan.stl")
