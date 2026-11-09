# Integrated pipeline profitability selection

Implemented an opt-in runtime selector in
[`pipeline.py`](../../../../python/triton/experimental/semantic_contracts/pipeline.py),
exposed through `triton.experimental.semantic_contracts.native.compile_pipeline`.
The native dependence analysis establishes legality. Runtime measurements select
depth; measurements never substitute for a memory proof.

## Policy and API

For one kernel specialization and launch configuration, the selector compiles the
same-contract baseline and depths 2/3/4. Optional `tune_licm=True` independently
includes LICM at every depth, including a depth0 LICM control. It deduplicates
identical binaries and records register/shared-memory resources. Known resource
exhaustion discards a candidate; other compilation and execution failures propagate.

Nine shuffled rounds screen candidates. A frozen winner must beat baseline by
at least 2% in another nine-round confirmation. Otherwise baseline remains selected.
The plan caches this choice in process; calls check argument metadata, scalar
specialization, device, launch grid and full-allocation alias relationships, then
launch exactly once. They do not benchmark or reset outputs.

The native inference pass also skips loops with constant-proven zero/one trip,
using widened integer arithmetic for signed and unsigned bounds. This is a cheap
heuristic about lookahead, not a claim that all single-trip staging is slower.
Dynamic bounds still go through the ordinary candidate machinery. Explicit source
stage/schedule annotations are honored; this API tunes inferred pipelines only.

Example for a kernel whose only tensor arguments are input0 and output1:

```python
from triton.experimental.semantic_contracts.native import compile_pipeline

plan = compile_pipeline(
    kernel, args, grid=grid,
    alias_groups={0: 0, 1: 1}, mutable_args=(1,),
    replay_safe=True,  # Full-overwrite, independent repeated invocations.
    options={"num_warps": 4, "num_stages": 1},
)
plan(*args)
print(plan.report()["selected"])
```

All accesses must stay within argument backing allocations. `mutable_args` must
include every allocation written, including state and scratch; calibration requires
exclusive use. Calibration snapshots and restores their **entire backing storage**.
Default `replay_safe=False` restores state before every measured invocation too,
and includes those reset copies in timing. The explicit full-overwrite contract
allows kernel-only timing. Calls on the returned plan never restore state.
Cold calibration is rejected during CUDA graph capture; cached calls support capture.

## Independently tuned comparison

B300, four timed workloads × four original launch configurations, plus three
boundary workloads at their first configuration. All source bodies unchanged;
LICM disabled for the performance study. The runtime API selected each plan's
depth. The harness then independently tuned baseline and selected-plan launch
configurations over the same domain, froze both choices, and confirmed them on
three fresh input seeds. Configuration tuning is in the experiment harness, not
part of the fixed-launch runtime API.

| Workload | Runtime choice at selected launch | Speedup against separately tuned baseline |
|---|---|---:|
| Log-softmax backward M256/K8192 FP32 | stages3, block1024/w4 | **1.115–1.130×** |
| Rotary T128/Q32/K8/D128 FP32 | stages4, w2 | **1.853–1.861×** |
| Rotary T1024/Q32/K8/D128 BF16 | stages4, w1 | **1.524–1.537×** |
| Resize 2×64×64×256 → 128×512 | baseline, tile64×2/w4 | **0.999–1.001×** |

The corresponding baseline choices were log-softmax block1024/w8, both rotary
cases w4, and the same resize tile. BF16 rotary w4/w8 produced identical depth
candidates and skipped calibration; w1/w2 produced real deeper pipelines.
Resize selected pipelines at some slower launch configurations, but retained
baseline at its fastest configuration. The overall comparison therefore avoids
mistaking recovery from a poor launch configuration for an optimization gain.

All **200 variant validations and 305 bitwise output comparisons passed**.
Every candidate and guarded selected path was checked with independent FP64
mathematical references, input integrity and allocation guards. All 19 plans
preserved caller backing storage across calibration. Cached plans accepted fresh
same-metadata allocations without reselection. Boundary coverage includes masks,
strided position IDs and single-height resize.

## Cost, scope and validation

Measured calibration cost was 9–159 ms per non-duplicate plan, excluding compile
time. Compilation ranged from 5 ms with cached code to 1.18 s in this run; the run
mixed existing and new compile-cache entries. These are cold-plan costs, excluded
from the reported steady-state gains. A plan must be reused to amortize them.

Steady-state timings use CUDA graphs (15 shuffled rounds, 32 invocations per
graph) and exclude host metadata checks, allocation, compilation and calibration.
They do not measure uncaptured end-to-end dispatch latency. Cached decisions can
become stale under different inputs, cache conditions or GPU load; the margin
and confirmation are not a universal no-regression guarantee. This is one GPU,
three kernel families and a bounded configuration domain, not a corpus-wide result.
AITER RMSNorm's explicit source stages are outside the selector's scope.

Native library held fixed during the experiment:
`04888937bce1bd18866aa87971e0b93e331d35df5cc0488508301996389b17a8`.
`make` rebuilt the compiler. The pipeline lit file passes, including seven new
trip-count cases. Seventeen new runtime tests cover state restoration, cached
single updates, graph capture, guards, selection margins, failures, deduplication,
independent LICM and resource fallback. Twenty-eight existing pipeline/dependence
runtime cases also pass.

- [Plan](plan.json), [environment/source hashes](environment.json),
  [summary](summary.json), [independent review](review.json).
- Per-workload JSON retains raw calibration/confirmation timings, configuration
  choices, resource counts and full validation records; `code/` retains TTIR,
  TTGIR and PTX for every compiled candidate.
- [Experiment driver](../../pipeline_profitability_experiment.py).
- [Prior slowdown controls](../pipeline_slowdown_final/README.md).
