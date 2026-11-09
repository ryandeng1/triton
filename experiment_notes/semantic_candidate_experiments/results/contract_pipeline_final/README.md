# Contract-licensed recurrent pipelining

On the NVIDIA B300, the compiler now recovers the earlier source-pipelining
oracle without changing the pinned AITER kernel body. The caller supplies
input/output footprint separation; the compiler uses it to move future loads
across current stores. A calibrated selector chooses among baseline, contract
LICM, automatic two-stage pipelining, and both.

| Case | Selected variant | Tuned baseline μs | Selected μs | Fresh-seed speedup range |
| --- | --- | ---: | ---: | ---: |
| decode16 | baseline | 6.25 | 6.25 | 1.000–1.000× |
| upstream1024 | licm | 760.14 | 670.08 | 1.134–1.135× |
| decode64 | licm | 32.92 | 30.39 | 1.082–1.085× |
| upstream63 | pipeline | 45.92 | 38.02 | 1.207–1.210× |
| upstream500 | pipeline | 418.62 | 343.96 | 1.214–1.218× |
| upstream2048 | baseline | 1269.16 | 1269.16 | 1.000–1.000× |

Each variant independently screens four launch configurations. Selection is
frozen before three fresh input seeds and 13 shuffled timing rounds per seed;
the baseline is retained below a predeclared 2% screening gain. Baseline-selected
rows execute that same kernel, so their 1.000× ratio is identity, not an independent
measurement of zero overhead. This is per-shape calibration, not held-out-shape
prediction, a static resource heuristic, or a general runtime autotuner.
Calibration took 4.2–10.3 seconds per case in this run and is excluded from the
table. All timings include the same state reset on every invocation, all timestep
outputs and final-state updates, but exclude compilation, allocation, host guards
and dispatch. The reference is the best tested configuration of this kernel,
not a strongest-provider algorithm or end-to-end model.

All four variants share the same argument contracts and unchanged source body;
contract load reuse is disabled uniformly. The no-transform contracted baseline
must exactly match no-contract PTX. Complete input/output backing-storage
separation is checked before launch. All 24 pipeline configurations generate
async copies. All 204 validation records pass an independent float64 recurrence
oracle and same-configuration bitwise output/state comparison, including mixed
variable lengths, no initial state and all-empty T=0. Inputs and output guards
remain intact. The native library stayed unchanged during measurement.

The T1024 LICM result agrees with the previous study: about 1.134× versus original.
The earlier 0.819× ratio was **pipeline plus LICM versus LICM**, not LICM versus
original. Combining the transforms still loses: about 0.929× original at T1024
and 1.175× original at T500, below the 1.214–1.218× pipeline-only choice.

The semantic information establishes legality; it does not determine a profitable
schedule. Keeping separate choices matters: pipeline wins T63/T500, LICM wins
T1024/larger decode, and both optimizations are declined for short decode/T2048.
The safety/provenance limits are documented in the [guard evidence](../pipeline_guard_final/README.md).

[summary.json](summary.json) and per-case reports retain all screens and fresh
confirmations, configurations, resources and validation. `code/` retains compressed
TTIR/TTGIR/PTX for each independently tuned variant. `provenance.json`, source,
license and compressed measured/oracle harnesses preserve the experiment.
The source preparation had already removed provider autotune/heuristic decorators;
only the original kernel body is claimed unchanged.

Reproduce from the checkout using an unused output directory:

```sh
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/contract_pipeline_experiment.py --output /tmp/triton-contract-pipeline-rerun
```
