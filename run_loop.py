"""
Live LoopForge loop on the twosum target. One propose(...) per turn.
"""

from harness import evaluate, Ledger
from task_twosum import TASK

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


# ===== ITER 1 — running-count hash pass (O(n^2) -> O(n)) =====
def cand_hash(case):
    a, target = case
    seen = {}
    count = 0
    for x in a:
        count += seen.get(target - x, 0)
        seen[x] = seen.get(x, 0) + 1
    return count


# ===== ITER 2 — array tally (restructure: dict -> fixed count array) =====
def cand_array(case):
    a, target = case
    if not a:
        return 0
    lo = min(a)
    hi = max(a)
    size = hi - lo + 1
    counts = [0] * size
    count = 0
    for x in a:
        c = target - x                 # complement we need to have seen already
        idx = c - lo
        if 0 <= idx < size:
            count += counts[idx]
        counts[x - lo] += 1
    return count


if __name__ == "__main__":
    print("=" * 70)
    print(f"TASK: {TASK.name}")
    print("=" * 70 + "\n")
    propose("hash: running-count single pass", cand_hash)
    propose("array: fixed count-array tally", cand_array)
    print("=" * 70)
    print("LEDGER:")
    print(ledger.as_feedback_log())
    print("=" * 70)
