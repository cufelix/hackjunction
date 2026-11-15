import json, math, sys
def R(q):
    x, y, z, w = q
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
def loc(row, p):
    M = R(row["quat"]); d = [p[i]-row["pos"][i] for i in range(3)]
    return [sum(M[r][c]*d[r] for r in range(3)) for c in range(3)]   # R^T d
H = (0.0975, 0.0, 0.004)   # handle centre in the pan frame (disc centre origin, bbox-centred z)
for f in sys.argv[1:]:
    rows = [r for r in json.load(open(f)) if r["pos"][0] > 0]
    print("==", f.split("/")[-1], "rows", len(rows))
    for side in ("left", "right"):
        best = None
        for r in rows:
            b = r["bodies"]; a = loc(r, b[f"{side}_hand_index_1_link"]); t = loc(r, b[f"{side}_hand_thumb_2_link"])
            mid = [(a[i]+t[i])/2 for i in range(3)]
            d = math.dist(mid, H)
            if best is None or d < best[0]:
                best = (d, r)
        d, r = best
        if d > 0.15:
            print(f"  {side}: never near (min {d:.3f} m)"); continue
        M = R(r["quat"]); ax = [M[0][0], M[1][0], M[2][0]]
        print(f"  {side}: t={r['t']} pinch-mid->handle {d*1000:.0f} mm | pan pos {[round(v,3) for v in r['pos']]} handle axis(pelvis) {[round(v,2) for v in ax]}")
        for k in ("palm_link", "thumb_2_link", "index_1_link", "middle_1_link"):
            l = loc(r, r["bodies"][f"{side}_hand_{k}"])
            print(f"     {k:14s} rel handle (along, across, up) mm: {[round((l[i]-H[i])*1000) for i in range(3)]}")
