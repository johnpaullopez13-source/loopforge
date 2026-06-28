"""
Target: count unordered index pairs (i, j), i < j, with a[i] + a[j] == target.
Baseline is brute-force O(N^2) double scan. Oracle is exact integer equality on
the returned count, so any miscount (self-pair, duplicate over-count, asymmetric
pair) is rejected as INCORRECT.
"""

import random
from harness import Task


def reference(case):
    a, target = case
    n = len(a)
    count = 0
    for i in range(n):
        ai = a[i]
        for j in range(i + 1, n):
            if ai + a[j] == target:
                count += 1
    return count


def gen_inputs():
    cases = []

    # big adversarial benchmark case (first = largest)
    rng = random.Random(20260623)
    target = 100
    a = [50] * 1500                        # self-pair trap: 50+50==target (x == target-x)
    for _ in range(2000):
        k = rng.randint(1, 49)             # heavy duplicates -> many cross pairs
        a += [k, target - k]               # k + (100-k) == target
    a += [0, target] * 250                 # asymmetric pairing: 0+100==target
    a += [rng.randint(200, 10_000) for _ in range(500)]  # noise: complements absent
    rng.shuffle(a)
    cases.append((a, target))

    # smaller randomized cases
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        a = [rng.randint(-20, 20) for _ in range(200)]
        cases.append((a, 0))

    # edge cases
    cases.append(([], 100))                # empty
    cases.append(([50], 100))              # single element
    cases.append(([50, 50], 100))          # one self-pair
    cases.append(([1, 2, 3], 7))           # odd target, no pair
    cases.append(([3, 3, 3, 3], 6))        # all self-complement -> 6
    cases.append(([0, 100, 0, 100], 100))  # asymmetric, multiplicity 4
    return cases


def equal(ref_out, cand_out):
    return ref_out == cand_out


TASK = Task(
    name="twosum_pair_count",
    reference=reference,
    gen_inputs=gen_inputs,
    equal=equal,
    banned=("import task_twosum", "from task_twosum"),
    bench_trials=5,
    baseline_trials=2,
)
