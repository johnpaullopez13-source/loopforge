"""
Demo target: "all user pairs within R meters" — the canonical hot loop in a
proximity app. Baseline is brute-force O(N^2) haversine. The oracle is exact
set-equality on the returned pairs, so any optimization that drops or invents a
pair is rejected as INCORRECT (the whole point of RQ7).
"""

import math
import random
from harness import Task

R_METERS = 500.0
EARTH_M = 6_371_000.0


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_M * math.asin(math.sqrt(a))


def gen_inputs():
    """First case is the big benchmark case; the rest are smaller correctness cases
    (including edge cases) so the oracle has teeth."""
    cases = []
    # big clustered case ~ a slice of the Bay Area
    rng = random.Random(7)
    pts = []
    for _ in range(2500):
        lat = 37.80 + rng.uniform(-0.03, 0.03)
        lon = -122.27 + rng.uniform(-0.03, 0.03)
        pts.append((len(pts), lat, lon))
    cases.append(pts)
    # smaller cases + edge cases
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        pts = [(i, 37.8 + rng.uniform(-0.05, 0.05), -122.27 + rng.uniform(-0.05, 0.05))
               for i in range(300)]
        cases.append(pts)
    cases.append([])                       # empty
    cases.append([(0, 37.8, -122.27)])     # single point, no pairs
    cases.append([(0, 37.8, -122.27), (1, 37.8, -122.27)])  # coincident -> one pair
    return cases


def reference(points):
    """Trusted O(N^2) implementation. Returns a set of (i, j) with i < j within R."""
    pairs = set()
    n = len(points)
    for a in range(n):
        ia, la, ka = points[a]
        for b in range(a + 1, n):
            ib, lb, kb = points[b]
            if haversine(la, ka, lb, kb) <= R_METERS:
                pairs.add((min(ia, ib), max(ia, ib)))
    return pairs


def equal(ref_set, cand_set):
    return set(ref_set) == set(cand_set)


TASK = Task(
    name="proximity_pairs_within_R",
    reference=reference,
    gen_inputs=gen_inputs,
    equal=equal,
    banned=("import reference", "from reference"),  # no calling the oracle
    bench_trials=5,
    baseline_trials=2,
)
