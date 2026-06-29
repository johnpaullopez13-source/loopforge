# LoopForge — a Loop Engineering harness

An independent, general-purpose take on the closed-loop LLM optimization approach
introduced in ComPilot (Merouani et al., PACT 2025) — for everyday optimization work
instead of polyhedral loop nests. The relationship is conceptual, not code-derived:
ComPilot is a compiler-specific system built on Tiramisu's polyhedral solvers, whereas
LoopForge is a general-purpose, in-process Python harness. **LoopForge is not affiliated
with the ComPilot project.** ComPilot's real contribution
isn't loop transformations — it's a discipline: an LLM agent proposes *structured,
labeled transformations*, an environment *validates and measures* them, and *typed
empirical feedback* grounds the next move. That discipline transfers to any task
with a correctness oracle and a stopwatch.

## What maps to what

| ComPilot | LoopForge |
| --- | --- |
| LLM agent (off-the-shelf) | Claude, via `AGENT_PROMPT.md` (chat or Claude Code) |
| Tiramisu compiler | `harness.py` — the environment driver |
| Polyhedral legality check | Correctness oracle: differential testing vs a trusted reference + invariants |
| Speedup measurement | Warm-up + median-of-trials benchmark, relative to the reference |
| 5 feedback types | `MALFORMED / INCORRECT / NOT_APPLICABLE / RUNTIME_ERROR / SUCCESS` |
| `<schedule>` tags | `<transform label="...">` block: reasoning + drop-in candidate |
| Dialogue = episodic memory | `Ledger` — every attempt + verdict, replayed each turn |
| comp_IDs / anonymization | Target spec (file:function); optional symbol anonymization |
| Multi-run K, iterations T | Configurable; the prompt bakes in diminishing-returns discipline |
| Push past premature quits | The prompt's "push once more" rule |

## The one honest caveat (ComPilot RQ7)

ComPilot gets *formal* correctness from polyhedral dependence analysis. General app
code has no such checker, so LoopForge substitutes the strongest practical thing:
differential testing against a reference across a battery of inputs plus invariants.
That is the weaker-guarantee regime RQ7 warns about — output comparison can pass a
transformation that is subtly wrong on inputs you didn't test. **So the oracle is the
most important file you write.** Make `gen_inputs()` adversarial: edge cases, empty,
degenerate, boundary values, large random. In the demo, iteration 1 looked provably
correct and was off by one pair — only a fat input battery caught it.

Where a real transformation engine exists, use it instead of LLM-written code and you
recover ComPilot's stronger guarantees: e.g. SQL (rewrite validated by `EXPLAIN
ANALYZE` for cost + result-set equality for correctness) is a near-perfect fit.

## Quickstart

Requirements: **Python 3** (tested on 3.13). `harness.py` and the two-sum loop
(`run_loop.py`) use only the standard library. The proximity demo's final
iteration imports **numpy** — that is the only external dependency. **There is no
`requirements.txt` in this repo**, so install numpy directly:

```
pip install numpy
```

(If numpy is absent the demo still runs, but its last iteration reports
`RUNTIME_ERROR` instead of a speedup.)

Run the proximity demo — four pre-baked transformation attempts:

```
python run_demo.py        # on Windows you can also use:  py run_demo.py
```

It optimizes a brute-force O(N^2) "users within R meters" function. The oracle
rejects two attempts (a subtly-wrong latitude prefilter that drops one pair, then
a greedy grid that drops boundary-straddling pairs), then accepts a correct
3x3-grid win, then a larger numpy-vectorised win. The exact speedup multipliers
are hardware-dependent — they are `baseline_time / candidate_time` measured on
your own CPU.

Run the live two-sum loop (the worked example below):

```
python run_loop.py
```

## LLM-driven loop (the intended pattern)

`run_demo.py` and `run_loop.py` replay pre-written candidates. The loop is meant
to be driven by an LLM acting as the **architect** (proposer): give the model
`AGENT_PROMPT.md` as its system prompt, then on each turn send it the target
source, the baseline time, and the running ledger. The model replies with a
`<transform>` block; you extract the candidate, run it through `evaluate()`
(passing its source via `src=...` so the `banned` check fires), record the typed
verdict in the `Ledger`, and feed that verdict back to the model. Repeat until
the model emits `no_further_transformations`, a turn cap is hit, or the
best-so-far stalls.

This repo does not ship a runnable driver for that loop — `evaluate()` and the
`Ledger` are the building blocks; wire them to whichever LLM client you use.

> **Safety:** a candidate returned by a model is arbitrary code. If you evaluate
> it in-process, do so only in a trusted environment, on a target you control.

## Defining a target (the `Task` schema)

A target is a `Task` (defined in `harness.py`). You supply a trusted reference, an
input battery, and an equality test; the harness does the rest. The complete field
list, with defaults exactly as they appear in `harness.py`:

| field | type | required | meaning |
| --- | --- | --- | --- |
| `name` | `str` | yes | label for the target |
| `reference` | `callable(input) -> output` | yes | the trusted implementation — used as BOTH the correctness oracle and the speed baseline |
| `gen_inputs` | `callable() -> list` | yes | the input battery; **`inputs[0]` is the largest** (used for benchmarking), the rest are correctness cases. Make it adversarial. |
| `equal` | `callable(ref_out, cand_out) -> bool` | yes | semantic equality of two outputs |
| `invariants` | `callable(input, output) -> None` | no — default no-op | extra checks; raise to reject a candidate |
| `banned` | `tuple[str, ...]` | no — default `()` | source substrings forbidden in a candidate (e.g. importing the oracle) |
| `bench_trials` | `int` | no — default `5` | median-of-N timed runs for a candidate |
| `baseline_trials` | `int` | no — default `2` | median-of-N timed runs for the reference |

> Note: the `banned` check only runs when a candidate's source is supplied — i.e.
> `propose(label, fn, src=...)`. If `src` is omitted, no substring check is performed.

A candidate is a drop-in replacement for `reference` with the same signature. The
harness checks it in three stages (see `evaluate()` in `harness.py`): a banned-
substring / callable pre-filter → the correctness oracle over every input in the
battery → a benchmark on `inputs[0]`. It returns one of the five verdicts above.

Worked example (`task_twosum.py` — count index pairs `i < j` with `a[i] + a[j] == target`):

```python
from harness import Task

def reference(case):
    a, target = case
    n = len(a); count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] + a[j] == target:
                count += 1
    return count

def gen_inputs():
    # inputs[0] is the big benchmark case; the rest are adversarial correctness
    # cases (self-pairs, duplicates, empty, single element, asymmetric pairs).
    ...

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
```

Then drive the loop (see `run_loop.py`): each agent turn becomes a
`propose(label, candidate_fn)` call, which runs `evaluate()` and records the
verdict + speedup in the `Ledger`. There is no CLI binary — you run the driver
script directly (`python run_loop.py`) and read the printed verdicts.

## Wiring into the architect/executor workflow

This is the loop you already run by hand, made explicit:

1. **Architect (this chat / a planning Claude):** holds `AGENT_PROMPT.md`, sees the
   target + ledger, emits the next `<transform>` candidate.
2. **Executor (Claude Code):** drops the candidate into the repo, runs `harness.py`
   against the task, captures the verdict.
3. The verdict + updated ledger go back to the architect. Repeat.

Keep the architect and executor as separate Claude instances exactly as you do now;
LoopForge just standardizes the message format between them so feedback is typed and
the best-so-far is never lost.

## Good first targets
- A Supabase/PostGIS proximity or "nearby" query (oracle = result-set equality, perf
  = `EXPLAIN ANALYZE` cost or wall time). Strongest fit — a real legality-ish checker.
- A deterministic game/physics simulation step (oracle = identical state from a
  fixed seed; perf = frame time).
- A transaction categorizer or record-matching pass (oracle = identical output on a
  fixed input set; perf = wall time over a batch).

## References

Merouani, M., Kara Bernou, I., and Baghdadi, R. "Agentic Auto-Scheduling: An
Experimental Study of LLM-Guided Loop Optimization." PACT 2025. arXiv:2511.00592.
https://arxiv.org/abs/2511.00592
