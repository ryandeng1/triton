# Recurrent loop-local pipelining oracle

Two-stage loop pipelining improves the pinned AITER T500/D60 recurrent kernel after bounded launch tuning, but regresses T1024/D128 and single-token decode. This is a source scheduling experiment, not an implementation of automatic contract-licensed prefetch. The compiler library was unchanged throughout.

| Workload | Original / LICM / pipeline / both, µs¹ | Pipeline / original speedup² | Both / LICM speedup² |
|---|---:|---:|---:|
| B2, T500, H4/HV4, D60 | 419.25 / 396.77 / 345.08 / 356.45 | 1.215× (1.214–1.216) | 1.113× (1.113–1.114) |
| B2, T1024, H4/HV8, D128 | 758.96 / 669.46 / 785.46 / 817.18 | 0.966× (0.966–0.966) | 0.819× (0.819–0.820) |
| 16 sequences × one token, H8/HV8, D128 | 6.277 / 6.313 / 6.422 / 6.468 | 0.978× (0.975–0.980) | 0.996× (0.976–0.996) |

¹ First fresh confirmation's median. ² Median and range of three fresh seeds' paired speedups. Larger is better. Each variant independently selects among the same four configurations. Every invocation includes identical state reset, every timestep output and the indexed final-state update. Allocation, compilation, oracle and tuning are excluded. These are complete scalar-gating recurrent-operator timings, not a complete model block.

At T500, pure pipelining is fastest: adding LICM to it costs about 3.3%. The combined variant still improves original by 1.174–1.176×. The result therefore does not establish that these optimizations compose profitably.

## Controls and validation

The extracted original body is AST-identical to AITER revision `ef0a5a26b8cb38ea1308f827985e36cde964a6ab`. An inverse-edit audit verifies the pipeline clone changes only `range(T)` to `tl.range(T, num_stages=2)`. The loop-local stage count remains two in every configuration. Kernel-launch `num_stages` is a separate control.

LICM uses two alias groups: all read-only inputs versus output/state allocations. Full backing-storage disjointness is checked. With the new contract load-reuse pass enabled, alias contracts also remove the final `h0_indices` reread; explicit immutability of only `A_log` and `dt_bias` cannot justify that elimination. [The diagnostic](load_reuse_diagnostic.json) retains this difference: nine versus ten TTGIR loads. We uniformly set `enable_contract_load_reuse=False` in the factorial experiment. Alias-only and explicit-immutable variants then have exactly identical PTX at every compiled configuration; only the alias version is timed.

The retained independent FP64 PyTorch recurrence checks all outputs and state slots. All 120 validation records pass, and all variants are bitwise identical to the same-configuration original. Maximum absolute reference errors are 0.000122 for output and 3.42e-7 for state. Boundaries cover D60 masks, grouped heads, mixed variable lengths including an empty sequence, negative state indices, no initial state, and an all-empty T=0 launch. Input tensors and output guard regions remain unchanged. CUDA graph captures reset the state before each invocation; resets are included in every timing.

## Selection and generated code

The first screen held the historical original's best configuration fixed. All three pipeline comparisons were negative. A follow-up screened the predeclared `(BV, warps, launch stages)` set `(8,1,3), (16,1,2), (32,2,2), (64,4,1)` with five randomized rounds per configuration. T500 selected `(8,1,3)` for both pipeline variants and `(32,2,2)` for original/LICM. T1024 selected `(8,1,3)` throughout; decode selected `(16,1,2)`. Three new payload seeds then confirmed these fixed selections with thirteen randomized rounds each. No confirmation timings were used to retune.

The transformations are real: selected T500 pipeline TTGIR contains fourteen static async-copy operations (ten with LICM), compared with zero in the corresponding original. Shared memory rises from 256 bytes in the selected original to 724/692 bytes in the selected pipeline variants. Their registers fall from 114/116 to 68/70 as tile size changes; no selected variant spills. Compressed TTGIR/PTX is retained for both the initial fixed configurations and final selections.

Resource counts alone do not explain the negatives. At T1024, LICM-to-combined changes registers from 108 to 97, yet time rises about 22%; shared memory changes 512→1076 bytes and no spills appear. Pipelining introduces async commits, waits and additional synchronization, while the state recurrence remains serialized. These are plausible costs, not a measured bottleneck attribution: no hardware-counter profile was taken. The T500 configuration dependence shows why a single fixed-configuration failure was insufficient to reject the scheduling opportunity.

## Artifacts and reproduction

[final_summary.json](final_summary.json) contains the final aggregates. The older `summary.json` records only the initial fixed screen; `*.bounded_screen.json` and `*.confirmation.json` contain selection and fresh raw timings. Source/provenance, CPU-only compilation manifests and the frozen native-library hash are retained. Compressed snapshots preserve the exact harness used by the initial screen and the bounded screen; the current companion script matches confirmation metadata. The bundle is approximately 610 KB. Existing experiment bundles were not changed.

With the pinned AITER source cache already present, run from the Triton checkout using `/home/ryandeng/tpaa/.venv-semantic/bin/python`:

```sh
python experiment_notes/semantic_candidate_experiments/recurrent_pipeline_experiment.py --run
python experiment_notes/semantic_candidate_experiments/recurrent_pipeline_experiment.py --precompile
python experiment_notes/semantic_candidate_experiments/recurrent_pipeline_experiment.py --screen
python experiment_notes/semantic_candidate_experiments/recurrent_pipeline_experiment.py --confirm-screen
```

GPU runs require exclusive ownership. The pinned long-T inputs come from upstream correctness tests, which establishes supported workloads but not production prevalence. This experiment does not compare alternative recurrent algorithms, larger tuning searches or AMD dispatch performance. The disjoint input/output allocations make this particular read-ahead experiment legal; no automatic scheduling proof was added.
