"""
Runs one optimization dialogue on the proximity task. Each `propose(...)` call is
one agent turn: a labeled transformation + its candidate implementation. The harness
returns typed feedback; the ledger keeps episodic memory and best-so-far.
"""

import math
from harness import evaluate, Ledger
from task_proximity import TASK, haversine, R_METERS

REF_LAT = 37.80
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(REF_LAT))

ledger = Ledger()
baseline_cache = {}
_it = 0


def propose(label, fn, src=""):
    global _it
    _it += 1
    fn._source = src
    a = evaluate(TASK, fn, label, baseline_cache)
    a.iteration = _it
    ledger.record(a, source=src)
    sp = f"  (speedup {a.speedup:.2f}x)" if a.speedup else ""
    print(f"--- iter {_it}: {label}")
    print(f"    FEEDBACK [{a.feedback.value}]{sp}: {a.detail}\n")
    return a


# ITER 1 — transformation: LATITUDE BOUNDING-BOX PREFILTER
# Provably safe: great-circle distance >= meridional distance, so if the N-S
# distance alone exceeds R the pair cannot be within R. Cheap gate before haversine.
def cand_bbox(points):
    pairs = set()
    n = len(points)
    lat_gate = R_METERS / M_PER_DEG_LAT
    for a in range(n):
        ia, la, ka = points[a]
        for b in range(a + 1, n):
            ib, lb, kb = points[b]
            if abs(la - lb) > lat_gate:
                continue
            if haversine(la, ka, lb, kb) <= R_METERS:
                pairs.add((min(ia, ib), max(ia, ib)))
    return pairs


# ITER 2 — transformation: SPATIAL GRID, SAME-CELL ONLY (intentionally greedy)
# Buckets points into ~R-sized cells and only compares within a cell. This MISSES
# pairs that straddle a cell boundary -> should be rejected by the oracle.
def cand_grid_samecell(points):
    pairs = set()
    cell_lat = R_METERS / M_PER_DEG_LAT
    cell_lon = R_METERS / M_PER_DEG_LON
    buckets = {}
    for (i, lat, lon) in points:
        key = (int(lat // cell_lat), int(lon // cell_lon))
        buckets.setdefault(key, []).append((i, lat, lon))
    for cell in buckets.values():
        for x in range(len(cell)):
            ia, la, ka = cell[x]
            for y in range(x + 1, len(cell)):
                ib, lb, kb = cell[y]
                if haversine(la, ka, lb, kb) <= R_METERS:
                    pairs.add((min(ia, ib), max(ia, ib)))
    return pairs


# ITER 3 — transformation: SPATIAL GRID, 3x3 NEIGHBOURHOOD
# Cell size = R in each axis, so any pair within R lives in the same or an adjacent
# cell. Comparing each cell against its 3x3 neighbourhood is correct. Dedup via i<j.
def cand_grid_3x3(points):
    pairs = set()
    cell_lat = R_METERS / M_PER_DEG_LAT
    cell_lon = R_METERS / M_PER_DEG_LON
    buckets = {}
    for (i, lat, lon) in points:
        key = (int(lat // cell_lat), int(lon // cell_lon))
        buckets.setdefault(key, []).append((i, lat, lon))
    for (cx, cy), cell in buckets.items():
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neigh.extend(buckets.get((cx + dx, cy + dy), ()))
        for (ia, la, ka) in cell:
            for (ib, lb, kb) in neigh:
                if ia < ib and haversine(la, ka, lb, kb) <= R_METERS:
                    pairs.add((ia, ib))
    return pairs


# ITER 4 — transformation: GRID 3x3 + NUMPY-VECTORISED DISTANCE
# Same correct neighbourhood structure, but distances within each cell-vs-neighbourhood
# batch are computed vectorised instead of one haversine call at a time.
def cand_grid_numpy(points):
    import numpy as np
    pairs = set()
    cell_lat = R_METERS / M_PER_DEG_LAT
    cell_lon = R_METERS / M_PER_DEG_LON
    EARTH = 6_371_000.0
    buckets = {}
    for (i, lat, lon) in points:
        key = (int(lat // cell_lat), int(lon // cell_lon))
        buckets.setdefault(key, []).append((i, lat, lon))
    for (cx, cy), cell in buckets.items():
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neigh.extend(buckets.get((cx + dx, cy + dy), ()))
        if not cell or not neigh:
            continue
        ci = np.fromiter((p[0] for p in cell), dtype=np.int64, count=len(cell))
        cla = np.radians(np.fromiter((p[1] for p in cell), dtype=np.float64, count=len(cell)))
        clo = np.radians(np.fromiter((p[2] for p in cell), dtype=np.float64, count=len(cell)))
        ni = np.fromiter((p[0] for p in neigh), dtype=np.int64, count=len(neigh))
        nla = np.radians(np.fromiter((p[1] for p in neigh), dtype=np.float64, count=len(neigh)))
        nlo = np.radians(np.fromiter((p[2] for p in neigh), dtype=np.float64, count=len(neigh)))
        dlat = nla[None, :] - cla[:, None]
        dlon = nlo[None, :] - clo[:, None]
        a = np.sin(dlat / 2) ** 2 + np.cos(cla)[:, None] * np.cos(nla)[None, :] * np.sin(dlon / 2) ** 2
        dist = 2 * EARTH * np.arcsin(np.sqrt(a))
        mask = (dist <= R_METERS) & (ci[:, None] < ni[None, :])
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows.tolist(), cols.tolist()):
            pairs.add((int(ci[r]), int(ni[c])))
    return pairs


if __name__ == "__main__":
    print("=" * 70)
    print(f"TASK: {TASK.name}   (R = {int(R_METERS)} m, N = 2500 benchmark points)")
    print("=" * 70 + "\n")
    propose("bbox: latitude prefilter before haversine", cand_bbox)
    propose("grid: bucket by ~R cells, compare SAME cell only", cand_grid_samecell)
    propose("grid: bucket by ~R cells, compare 3x3 neighbourhood", cand_grid_3x3)
    propose("grid 3x3 + numpy-vectorised haversine per batch", cand_grid_numpy)
    print("=" * 70)
    print("LEDGER (the episodic memory replayed to the agent each turn):")
    print(ledger.as_feedback_log())
    print("=" * 70)
