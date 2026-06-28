"""
LoopForge harness — the "environment" half of a ComPilot-style optimization loop.

ComPilot delegates correctness + code generation to a compiler (Tiramisu) and
feeds the LLM agent typed empirical feedback. There is no polyhedral checker for
general application code, so this harness substitutes the strongest practical
oracle: differential testing of a candidate against a trusted reference across a
battery of inputs (plus any task-specific invariants). This is the weaker-guarantee
regime ComPilot flags in RQ7 — so the oracle is the most important thing to get right.

The agent never asserts that its own code is correct or fast. The harness decides,
and emits one of five feedback categories (mapping 1:1 onto ComPilot's):

    MALFORMED       <-> Invalid Schedule    (didn't apply / wrong shape / banned import)
    INCORRECT       <-> Illegal Schedule    (oracle mismatch: changes behavior)
    NOT_APPLICABLE  <-> Solver Failure      (precondition not met; nothing to do)
    RUNTIME_ERROR   <-> Compiler Crash      (raised during execution)
    SUCCESS         <-> Successful Execution (correct; reports measured speedup)
"""

from __future__ import annotations

import time
import statistics
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Feedback(str, Enum):
    MALFORMED = "MALFORMED"
    INCORRECT = "INCORRECT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    SUCCESS = "SUCCESS"


@dataclass
class Task:
    """A pluggable optimization target.

    reference:  trusted, correct implementation. Used as the correctness oracle
                AND as the performance baseline (speedup = baseline_time / candidate_time).
    gen_inputs: returns a list of input cases. More + nastier inputs = stronger oracle.
                The FIRST case should be the largest (used for benchmarking).
    equal:      semantic equality of two outputs (set-equality, tolerance, etc.).
    invariants: optional extra checks run on candidate output (raise to reject).
    banned:     substrings forbidden in a candidate's source (e.g. cheating shortcuts).
    """
    name: str
    reference: Callable[[Any], Any]
    gen_inputs: Callable[[], list]
    equal: Callable[[Any, Any], bool]
    invariants: Callable[[Any, Any], None] = lambda inp, out: None
    banned: tuple[str, ...] = ()
    bench_trials: int = 5
    baseline_trials: int = 2


@dataclass
class Attempt:
    iteration: int
    label: str
    feedback: Feedback
    detail: str
    speedup: Optional[float] = None


@dataclass
class Ledger:
    """Episodic memory — every attempt + outcome, replayed to the agent each turn."""
    attempts: list[Attempt] = field(default_factory=list)
    best_speedup: float = 1.0
    best_label: Optional[str] = None
    best_source: Optional[str] = None

    def record(self, a: Attempt, source: Optional[str] = None):
        self.attempts.append(a)
        if a.feedback is Feedback.SUCCESS and a.speedup and a.speedup > self.best_speedup:
            self.best_speedup, self.best_label, self.best_source = a.speedup, a.label, source

    def as_feedback_log(self) -> str:
        if not self.attempts:
            return "(no attempts yet)"
        lines = []
        for a in self.attempts:
            tag = f"[{a.feedback.value}]"
            sp = f" speedup={a.speedup:.2f}x" if a.speedup else ""
            lines.append(f"  iter {a.iteration}: {a.label} -> {tag}{sp}\n      {a.detail}")
        lines.append(f"  >> best so far: {self.best_label or 'baseline'} @ {self.best_speedup:.2f}x")
        return "\n".join(lines)


def _median_time(fn: Callable[[Any], Any], inp: Any, trials: int) -> float:
    # one warm-up, then median of `trials` runs
    fn(inp)
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(inp)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def evaluate(task: Task, candidate: Callable[[Any], Any], label: str,
             baseline_time_cache: dict) -> Attempt:
    """Run the two-stage check: cheap validity -> oracle -> benchmark.
    Mirrors ComPilot's pre-filter-then-compiler design so we never benchmark
    something that's wrong or broken."""
    inputs = task.gen_inputs()
    bench_input = inputs[0]

    # Stage 0: cheap validity pre-filter (banned patterns / callability)
    src = getattr(candidate, "_source", "")
    for b in task.banned:
        if b in src:
            return Attempt(0, label, Feedback.MALFORMED, f"uses banned construct: {b!r}")
    if not callable(candidate):
        return Attempt(0, label, Feedback.MALFORMED, "candidate is not callable")

    # Stage 1: correctness oracle (differential testing across the battery)
    for i, inp in enumerate(inputs):
        try:
            got = candidate(inp)
        except Exception:
            tb = traceback.format_exc().strip().splitlines()[-1]
            return Attempt(0, label, Feedback.RUNTIME_ERROR, f"raised on input #{i}: {tb}")
        ref = task.reference(inp)
        if not task.equal(ref, got):
            n_ref = len(ref) if hasattr(ref, "__len__") else "?"
            n_got = len(got) if hasattr(got, "__len__") else "?"
            return Attempt(0, label, Feedback.INCORRECT,
                           f"output differs from reference on input #{i} "
                           f"(reference size={n_ref}, candidate size={n_got})")
        try:
            task.invariants(inp, got)
        except Exception as e:
            return Attempt(0, label, Feedback.INCORRECT, f"invariant violated on input #{i}: {e}")

    # Stage 2: benchmark (correct -> measure speedup vs cached baseline)
    key = id(task)
    if key not in baseline_time_cache:
        baseline_time_cache[key] = _median_time(task.reference, bench_input, task.baseline_trials)
    base = baseline_time_cache[key]
    cand_t = _median_time(candidate, bench_input, task.bench_trials)
    speedup = base / cand_t if cand_t > 0 else float("inf")
    return Attempt(0, label, Feedback.SUCCESS,
                   f"correct on all {len(inputs)} inputs; "
                   f"baseline={base*1e3:.1f}ms candidate={cand_t*1e3:.1f}ms",
                   speedup=speedup)
