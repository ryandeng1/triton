# Real-kernel race-freedom experiments, 2026-11-07

[Start here](README.md) · [Consolidated findings](findings_overview.md).
This report covers the native compiler experiments; later source prototypes
and their results are summarized in the consolidated overview.

**No improvement over the best tested attention configuration was established.**
Forcing one QK buffer gives a repeatable 1.205× gain on one fixed configuration,
but that configuration is substantially slower than the best unchanged baseline.
The Aref scheduler produced no code changes in the tested GEMM and attention
cases; the packing check on attention was also unchanged.
[Machine-readable summary](summary_11_07.json).

A subsequent [matched implementation comparison and semantic ablation](semantic_attention_findings_11_07.md)
reproduces the faster FA4/Gluon attention kernels and isolates identity-rescaling
elision. Those are source-level experiments with separate numerical-contract
limits, not improvements from the compiler flags evaluated below.

A subsequent [LayerNorm backward ownership experiment](layernorm_ownership_findings_11_07.md)
targets private partial-gradient reductions and removes competing writers. It
includes BF16 inputs/outputs with matched FP32 partial buffers, separately
accounting for accumulation precision and the ownership transformation.

All GPU results use an NVIDIA B300 SXM6 PC, 148 SMs, CUDA target sm_103.
The kernels are unchanged Tritonbench sources. The harnesses bypass autotuning
and ordinary JIT argument specialization, so these are controlled compiler
comparisons, not end-to-end provider benchmarks. No Tritonbench files changed.

The compiler now exposes two default-off experiments:

- `enable_aref_release_scheduling` runs checked Aref release scheduling after
  Aref insertion/cleanup and before Aref lowering erases ownership metadata.
  Added diagnostics explain ownership rejection and motion boundaries;
  recognition and motion rules remain unchanged.
- `enable_tmem_single_buffer` applies the existing source policy
  `tt.disallow_acc_multi_buffer` to marked warp-specialized `scf.for` loops
  **before latency inference**, so scheduling and allocation agree about depth.
  It skips a marked loop if that loop or a nested loop has explicit latencies.
  It does not weaken alias, generation, or asynchronous-completion checks.

The single-buffer comparison disables Aref scheduling, explicit TMEM release
scheduling, and packing; only the new policy changes between the two variants.

The original scheduling experiments produced these results:

| Unchanged source | Coverage | Compiler result |
| --- | --- | --- |
| Persistent FP16 GEMM, epilogue subtile 2 and 4 | Three shapes × three seeds × two subtiles = 18 cases | Off/on PTX identical |
| Persistent tensorwise FP8 GEMM | Three shapes × three seeds = 9 cases | Off/on PTX identical; provider tile 32 uses no TMEM |
| Nonpersistent BF16 attention | B2 H8 N4096 D128, BM128 BN64, three seeds | Off/Aref/packing TTGIR and PTX identical |

Evidence: [FP16 subtile 2](real_kernel_results/gemm-subtile2/results.json),
[subtile 4](real_kernel_results/gemm-subtile4/results.json),
[FP8](real_kernel_results/fp8-tensorwise-compat/results.json), and
[attention](real_kernel_results/attention-bn64-fixed/results.json).
The FP8 harness records `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1` for both variants
because the unchanged source reads PyTorch's fixed `ScalingType` enum.
All cases passed numerical and guard checks. Timing differences between identical
binaries establish no compiler speedup; attention timings were skipped.

Generated GEMM and attention IR places relevant releases immediately after the
last TMEM access. There is no independent pure suffix for a late release mover
to cross. Unsupported cyclic ownership also limits early Aref recognition.
Broadening recognition alone would not create useful scheduling slack here.

Single buffering uses `proton_blackwell_ws_fused_attention.py::_attn_fwd` with
BF16 inputs/outputs, noncausal attention, B2 H8 N4096 D128 unless stated otherwise.
Despite its filename, the device function contains no profiler instrumentation.
Timings below are the median of three seed medians, in microseconds; normal
validation runs use 15 shuffled interleaved rounds, 32 kernel executions per
graph replay, and at
least 200 ms warmup.

| Configuration | Baseline | Single buffer | Interpretation |
| --- | ---: | ---: | --- |
| BM128 BN64, stages 2, warps 4, cap 80 | 670.84 | 556.39 | 1.206×; TMEM reservation 512→256 |
| Independent repeat of that configuration | 670.84 | 556.51 | 1.205×; same PTX as first run |
| Same tile, uncapped registers | 672.19 | 733.56 | Slower despite smaller TMEM reservation |
| Same cap-80 tile, B1 H1 (32 CTAs) | 166.22 | 183.80 | Slower; identical binaries to the large grid |
| BM128 BN128, stages 2, warps 4, cap 80 | 505.13 | 648.19 | Slower; reservation remains 512 |
| BM128 BN64, stages 2, warps 8, cap 48 | 1932.40 | 1637.40 | Paired gain on an uncompetitive configuration |
| Best baseline: BM128 BN128, stages 2, warps 8, cap 80; independent repeat | 130.87 | 198.65 | Single buffering regresses the best configuration |

Results: [initial cap 80](real_kernel_results/attention-single-cap80/results.json),
[repeat](real_kernel_results/attention-single-cap80-repeat/results.json),
[uncapped](real_kernel_results/attention-single-uncapped/results.json),
[small grid](real_kernel_results/attention-single-cap80-smallgrid/results.json),
[BN128](real_kernel_results/attention-single-bn128/results.json),
[cap 48](real_kernel_results/attention-single-cap48-w8/results.json), and
[best-config repeat](real_kernel_results/attention-screen-best-repeat/results.json).

The [provider-grid screen](real_kernel_results/attention-config-screen/sweep.json)
covered BM={64,128}, BN={64,128}, stages={2,3,4}, warps={4,8}, cap 80:
20 configurations measured successfully and four exceeded shared-memory limits
for both variants. Screening used one seed, five rounds, and 16 kernel executions
per graph replay. Table warp counts are compiler settings; warp specialization
can add warps to the actual launch.
Its best baseline was 131.15 us versus 198.61 us for single buffering;
the independent repeat above confirmed the result. The best screen and repeat
have identical PTX for each variant after the final guard rebuild.

For cap-80 BN64, QK depth changes from two to one while O, P, and double-buffered
K/V retain their shapes. PTX allocates 256 rather than 512 TMEM columns. Shared
memory remains 99,920 bytes and launch metadata reports 12 warps; register
redistribution instructions are unchanged. `n_regs=80` and `n_spills=152` match.
Here `n_spills` means CUDA local-memory bytes divided by four, not spill instruction
count. Warp specialization's `setmaxnreg` means `80 × threads` is not an occupancy
calculation. Occupancy was not measured. The controls are consistent with a
resource/overlap tradeoff, but do not isolate its hardware mechanism.

The [single-buffer TTGIR](real_kernel_results/attention-single-cap80/single/kernel.ttgir)
retains QK full waits, consumer loads/releases, producer empty waits, alternating
phases, and the final drain handshake. P reuse still waits for PV MMA completion;
O correction and final output loads retain their waits. The
[PTX](real_kernel_results/attention-single-cap80/single/kernel.ptx) retains all
three TMEM load-wait and three store-wait sites before publication. Fewer static
MMA/commit sites reflect the smaller prefetch prologue, not lost completion edges.

Correctness checks include an FP32 attention reference with TF32 disabled, LSE,
guard regions, exact intervariant outputs, and fresh outputs poisoned with NaNs.
BN128 sequences [128](real_kernel_results/attention-single-bn128-loop1/results.json),
[256](real_kernel_results/attention-single-bn128-loop2/results.json), and
[512](real_kernel_results/attention-single-bn128-loop4/results.json) test one,
two, and four loop iterations across three seeds, including graph replay.
All pass. These tests support the inspected protocol; they are not a general
proof for arbitrary kernels. Source hashes, patches, native-library hashes,
compiler logs, IR, PTX, cubins, and later harness snapshots accompany results.

`make` and `ninja triton-opt` succeeded. All five focused lit tests passed after
the explicit-latency guard: latency assignment, automatic warp specialization,
Aref TMEM insertion, NVWS ops, and TMEM interleaving.
[Final test log](real_kernel_results/2026-11-07-build/single-buffer-explicit-latency-lit.log).
Earlier blocked [environment](real_kernel_results/environment.json),
[build](real_kernel_results/build.log), and
[preflight](real_kernel_results/preflight_artifacts/results.json) reports remain
historical records; they do not describe the final working environment.

Reproduce from the Triton repository root with the configured environment and
a datacenter Blackwell GPU. Every output directory must be new. Rebuild native
changes with `PATH="../.venv-triton-ir/bin:$PATH" make` before testing.

```bash
../.venv-triton-ir/bin/python experiment_notes/aref_gemm_experiment.py --output /tmp/aref-subtile2
../.venv-triton-ir/bin/python experiment_notes/aref_gemm_experiment.py --epilogue-subtile 4 --output /tmp/aref-subtile4
../.venv-triton-ir/bin/python experiment_notes/aref_gemm_experiment.py --kernel fp8-tensorwise --output /tmp/aref-fp8
../.venv-triton-ir/bin/python experiment_notes/aref_attention_experiment.py --maxnreg 80 --output /tmp/attention-existing
../.venv-triton-ir/bin/python experiment_notes/aref_attention_experiment.py --optimization single-buffer --maxnreg 80 --rounds 15 --output /tmp/attention-single
../.venv-triton-ir/bin/python experiment_notes/aref_attention_experiment.py --optimization single-buffer --maxnreg 80 --block-n 128 --num-warps 8 --rounds 15 --output /tmp/attention-best
../.venv-triton-ir/bin/python experiment_notes/attention_buffer_sweep.py --output /tmp/attention-screen
```

For controls, omit `--maxnreg` for uncapped registers, add `--batch 1 --heads 1`
for the small grid, or use `--block-n 128` and `--sequence 128`, `256`, or `512`
in separate invocations for the short loops. `--compile-only` collects compiler
evidence without GPU correctness or performance claims.

The next useful step is to find a **best-config baseline actually limited by
TMEM reservation**, then reduce storage while preserving its profitable overlap.
QK/P reuse is a candidate, but a single fixed P slot cannot simply alias either
half of a ping-pong QK allocation. Couple physical slot selection to generation,
and prove PV's asynchronous P read completes before the corresponding QK slot
is overwritten. Preserve producer/consumer waits, initialization, wraparound,
and drain behavior. Compare against the best provider configuration and profile
resource limits before promoting a policy. Global forced single buffering is
not justified by these results.
