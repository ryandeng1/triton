The native private-output pass now relaxes unused-result i32 additions across
multiple sites and nested `scf.for` loops without carried state. All updates must
target one private global output allocation; ordinary loads must be disjoint.
Integer modular addition makes the final result independent of ordering. Floating
loops/multiple sites remain unsupported because rounding can expose ordering.

The unchanged FlagGems persistent embedding frequency counter was extracted from
the recorded Cambricon backend revision and compiled directly on B300. This is a
counter experiment, not full embedding backward or NVIDIA-provider performance.
Output reset is included; allocation, producer generation and Python guards are
excluded. Source extraction is AST checked. The manual control only adds relaxed
ordering; native TTIR differs from baseline only in atomic ordering and contracts.

`results.json` contains 114 paired measurements: three sizes, uniform/hot-32
indices, blocks 256/512, persistent and full grids, and three seeds. Each case has
15 shuffled rounds of 32 CUDA-graph repetitions. Best-tested native speedups are
1.76/1.54/1.61× for uniform indices and 1.34/1.05/1.003× for hot-32 indices.
Native/manual relaxed timings agree within 0.03% geometrically. `summary.json`
records aggregate results; `smoke.json` and `code/` retain legality/code evidence.

Correctness checks cover exact counts, input changes, invalid/padding/tail masks,
guard storage and overflow. Compiler/runtime tests additionally cover multiple
sites, nested loops, dynamic bounds/steps, zero trips, Gluon and rejected effects.
Build and validation logs plus the exact harness/compiler source are retained.

From the Triton checkout with its source build installed:

```sh
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/private_atomic_loop_experiment.py --source experiment_notes/semantic_candidate_experiments/results/private_atomic_loop_final/sources/flaggems_cambricon_embedding.py --output /tmp/private-atomic-loop-rerun
```
