import math
from pathlib import Path

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


