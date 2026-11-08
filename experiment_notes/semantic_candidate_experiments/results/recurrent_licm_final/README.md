# Recurrent immutable-load gate

Explicit immutable contracts improve three of six independently tuned recurrent cases on the NVIDIA B300, including an authentic upstream long-sequence test. Three cases regress slightly. Alias-only contracts left the kernel unchanged in the frozen baseline compiler; the post-build result below shows the added pointer proof. The current residency guard accepts the negative cases too.

Every timing includes identical state reset before **each invocation**, every timestep output, and the final indexed state update. Allocation, compilation and tuning are excluded. This measures the complete scalar-gating recurrent operator; convolution, gated RMSNorm and the model block are outside scope.

| Workload | Original / immutable µs, first confirmation | Immutable speedup, median of 3 seeds | Seed range | Manual speedup |
|---|---:|---:|---:|---:|
| Decode, 16 sequences × 1 token, 8 heads, D128 | 6.248 / 6.290 | 0.994× | 0.980–0.995× | 0.993× |
| Upstream test, B2 × T1024, H4/HV8, D128 | 759.17 / 669.11 | 1.134× | 1.134–1.135× | 1.134× |
| Decode, 64 sequences × 1 token, 12 heads, D128 | 32.651 / 30.097 | 1.085× | 1.083–1.089× | 1.085× |
| Upstream test, B1 × T63, H1/HV1, D64 | 45.514 / 46.408 | 0.981× | 0.980–0.984× | 0.983× |
| Upstream test, B2 × T500, H4/HV4, D60 | 418.76 / 396.64 | 1.056× | 1.055–1.057× | 1.142× |
| Upstream test, B4 × T2048, H4/HV8, D64 | 1267.23 / 1280.93 | 0.989× | 0.989–0.990× | 0.989× |

These are paired, randomized timing ratios after each variant independently chooses among `(BV, warps, stages) = (8,1,3), (16,1,2), (32,2,2), (64,4,1)`. The first choice is upstream's CUDA configuration. Selection used three screening rounds; confirmation used a new payload seed and eleven rounds. Two further seeds each used thirteen rounds with configurations fixed. Raw timings, selection, validation and resource metrics are in the per-case JSON files; [summary.json](summary.json) contains the three-seed aggregates. Alias-only ratios stayed approximately 1.0×.

The D128/T1024 gain survives independent tuning, and native/manual choose the upstream configuration. Registers increase from 96 to 108, while the small grid fits resident capacity. The large decode winner instead reduces registers from 209 to 168 at BV16. Thus even a one-iteration loop can change code generation and scheduling; that gain is not evidence of repeated-load amortization. The T2048 negative has 64 registers in both variants. Neither register count nor the current occupancy heuristic predicts all outcomes.

For T500, manual hoisting outperforms native LICM. Saved TTGIR confirms **both** already hoist `exp(A_log)` and its negation. At BV16 both use 106 registers, 256 shared bytes and no spills. Manual supplies explicit zero for inactive `T>0` masked loads; native leaves inactive values undefined, and IR ordering differs. The remaining gap is downstream code generation, not a missing dependent-expression hoist. We preserved the benchmark source and report this limit rather than attributing the extra manual gain to more semantic information.

## Workload and correctness audit

The exact pinned AITER function has [upstream tests](https://github.com/ROCm/aiter/blob/ef0a5a26b8cb38ea1308f827985e36cde964a6ab/op_tests/test_gated_delta_rule.py#L453-L556) covering T63–2048, D60/64/128, grouped heads, FP32/FP16/BF16, normalized QK and initial state. Our long-T cases come from those tests. This establishes supported inputs, not production frequency. Mixed precision follows the test: FP32 Q/K and gating parameters, with BF16 values/beta in the two BF16 long cases; output dtype follows Q as in the upstream wrapper.

The pinned decode benchmark supplies `cu_seqlens=arange(batch+1)`, so each sequence has one token. However, its legacy three-kernel branch supplies per-channel gating arrays to this scalar-gating kernel and ignores its returned output before RMSNorm. That branch is **not** used as a validated equivalent-work baseline. Our decode shapes use its dimensions and BF16 inputs, with the scalar-gating API's correct shapes. Speculative fused-KDA benchmarks invoke a different kernel; our short mixed-varlen cases are correctness diagnostics only. [The retained source excerpts](upstream_workload_evidence.txt) and [provenance](provenance.json) document these distinctions.

A separate FP64 PyTorch recurrence checks every output and every state slot, including unchanged guard slots. All variants bitwise match the same-configuration original and pass the independent reference tolerances. Across 200 validation records, the largest absolute FP64-reference error is 0.000122 for outputs and 7.59e-7 for state. Initial state, grouped heads, non-power-of-two D60, mixed variable lengths with an empty sequence and negative state index, and no-initial-state execution are covered. Readonly inputs remain unchanged. Fresh allocations establish disjointness; the harness owns immutable parameters throughout each invocation.

The original kernel body is AST-identical to the pinned source. An inverse-edit audit verifies that the manual clone changes only two guarded parameter loads and their dependent negative exponential. Provider autotune/heuristic decorators are removed and flags bound explicitly; runtime `T` remains unspecialized. Native library hashes remained frozen throughout measurement. The helper's raw optimized binary is measured even when reporting its policy decision.

A separate survey correction remains relevant: Liger's plain fast backward path already hoists its scalar gradient. The prior general-path plain specialization should not be presented as a production opportunity.

## Post-build alias proof

The bounded same-root `scf.for` pointer analysis now lets alias-region contracts establish the same no-clobber fact. On all six cases, alias-only and immutable variants have **exactly equal PTX and resources** at matching configurations, with both parameter loads outside the timestep loop. Alias variants contain no immutable attribute, and full backing-storage checks establish input/output disjointness. Fresh FP64 output/state validation passes. Alias/immutable timing ratios are 0.99942–1.00029 over thirteen randomized rounds; the positive and negative performance outcomes remain. [The separate post-build report](postbuild_alias_validation.json) records the new compiler/library hashes, so these results are distinct from the frozen three-seed study.

The proof follows an advancing pointer only when its loop backedge derives from that exact carried pointer, then resolves the initial allocation root. This does not make the advancing address loop-invariant. Unknown/root-changing/cross-carried/select/while paths decline; lit and runtime validation are recorded by the parent experiment.

## Reproduce

```sh
python3 experiment_notes/semantic_candidate_experiments/recurrent_licm_experiment.py --prepare
/home/ryandeng/tpaa/.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/recurrent_licm_experiment.py --run
/home/ryandeng/tpaa/.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/recurrent_licm_experiment.py --confirm-fixed
```

After rebuilding the pointer-analysis change, run `--postbuild-alias` against the retained fixed-configuration study. GPU runs require exclusive device ownership. Source downloads stay under `/tmp`; artifacts are below 1 MB. The retained AITER license applies to the extracted source. The source/compiler revisions and GPU details are recorded, so future pointer-provenance changes must be reported as a separate post-build validation.
