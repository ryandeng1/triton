# Global dependence control for explicit loop pipelining

The current local compiler produces incorrect output with loop `num_stages=3`
for a one-CTA, in-place global-memory recurrence. This is a correctness control,
not a performance benchmark or a claim about an independently built upstream
release. No compiler changes were made for this experiment.

```python
cols = tl.arange(0, 128)
for t in tl.range(0, T, num_stages=STAGES):
    value = tl.load(X + t * 128 + cols)
    tl.store(X + (t + 1) * 128 + cols, value + 1)
```

The full `(T+1) × 128` allocation is initialized. Row zero contains `arange(128)`;
all remaining rows initially contain -1000. One logical block executes the loop,
with no other observers or programs. The tensors have the same blocked layout
throughout. Each column has a sequential dependence from the previous iteration;
there are no hidden per-lane scalars, cross-CTA races or out-of-bounds accesses.
The expected row `r` is `arange(128) + r`.

| Loop stages | Async-copy sites in final TTGIR | Shared bytes | Exact checks |
| --- | ---: | ---: | --- |
| 1 | 0 | 0 | Pass T=0,1,2,4,17 |
| 2 | 2 | 512 | Pass T=0,1,2,4,17 |
| 3 | 3 | 1024 | Pass T=0,1; fail T=2,4,17 |

At T=2, stages3 returns first column `[0, 1, -999]` instead of `[0, 1, 2]`.
The generated TTGIR copies rows zero **and one** into shared memory in the
prologue, before any loop store. Thus it consumes row one's initial value rather
than the value written by iteration zero. Stages2 copies the next row after the
current store and passes these checks. That does not prove stages2 safe for all
alias patterns.

This supports adding global-memory dependence reasoning before generalizing
automatic prefetch. An explicit scheduling request does not itself prove a
load independent of intervening stores. Alias-group disjointness or immutable
input facts could license selected load motion; this in-place recurrence must
retain its dependence or decline that motion. Argument-root separation alone
cannot infer disjoint regions within this single allocation.

`results.json` contains every full-tensor check and diagnostic first column;
compressed TTIR/TTGIR/PTX show the transformed schedules. `provenance.json`
identifies the native library, source and relevant compiler snapshots.

Reproduce on sm103, using an unused output directory:

```sh
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/pipeline_alias_probe.py --run --output /tmp/pipeline-alias-new
```

This is a retained reproducer. A native guard and regression test have not been
implemented in this experiment.
