# Faster attention implementations and semantic ablations, 2026-11-07

[Start here](README.md) · [Consolidated findings](findings_overview.md).

The matched attention experiment reproduces a substantial implementation gap:
FlashAttention-4 is 1.59× faster than the best previously screened Tritonbench
configuration, and the maintained Gluon example is 1.36× faster. Isolating one
operation gives a smaller, useful result: skipping an identity accumulator
rescale improves FA4 by about 10%, and a manual Gluon port by about 5% under
the same deferred-rescaling policy. This is an experiment in kernel semantics,
not a new automatic compiler optimization.

[Machine-readable summary](semantic_attention_summary_11_07.json).

## Matched workload and baseline

The primary workload is BF16 noncausal attention, B=2, H=8, N=4096, D=128,
scale=1/sqrt(D), on the NVIDIA B300 SXM6 PC (148 SMs, sm_103). Each provider
receives the same contiguous BHND Q/K/V allocations and writes preallocated
output and log-sum-exp. FA4 receives BNHD strided views without copying data.
Its natural-log LSE is converted to base two for validation. Timings exclude
Python, compilation, and allocation, and include all GPU work in the warmed
forward call.

[Harness](semantic_attention_experiment.py),
[matched comparison](real_kernel_results/semantic_attention_comparison/results.json).
The Tritonbench device function is unchanged
`proton_blackwell_ws_fused_attention.py::_attn_fwd.fn`, with BM128/BN128,
stages=2, warps=8, maxnreg=80. The register cap matches the provider. The
[earlier configuration screen](real_kernel_results/attention-config-screen/sweep.json)
covered its 24 advertised tile/stage/warp combinations. This run uses ordinary
JIT argument specialization; its PTX instruction body matches the earlier
AST baseline, with only the LSE pointer alignment declaration changed.
The AST control has indistinguishable timing. Every experimental compiler
optimization flag is disabled.

| Implementation | Median time, µs | Tritonbench time / provider time |
| --- | ---: | ---: |
| Tritonbench, ordinary JIT | 143.47 | 1.00× |
| Earlier unspecialized AST control | 143.66 | 1.00× |
| Maintained Gluon example, selected two-CTA configuration | 105.28 | 1.36× |
| FA4, pinned official source | 90.11 | 1.59× |

These are medians of three seed medians, each with 15 shuffled interleaved
rounds and 32 launches per CUDA graph replay, after at least 200 ms warmup.
Clocks are not locked. Absolute times vary between runs; comparisons use
providers measured together. The short smoke run is not the performance result.
The original Gluon source is `python/examples/gluon/01-attention-forward.py`.
FA4 is pinned to `7c53542eab6a1c4fe760d1101b8d57d6307d1dbe` in the separate
`../flash-attention-reference` checkout and `.venv-fa4-reference` environment.

The baseline [Nsight report](semantic_attention_results/baseline-profile-details.txt)
reports one resident CTA limited by registers and shared memory, with no
eligible warp on about 55% of scheduler cycles. This is a diagnostic first-launch
profile, not a timing baseline or proof of the entire performance gap.

## Identity rescaling: a measured semantic effect

FA4 sometimes retains the previous softmax shift until the maximum increases
by more than eight base-two exponent units. If the shift is unchanged,
`alpha = exp2(old_shift - new_shift) = 1`. The correction partition can then
leave its accumulator in tensor memory instead of loading, multiplying, and
storing it. The softmax identity permits different shifts in real arithmetic;
changing the shift policy changes floating-point rounding.

The isolated FA4 variant changes only `should_rescale` from its ballot-derived
predicate to `True`. It retains the threshold-eight policy, selected configuration,
scale values, and synchronization code. This reverse ablation measures the
benefit of an optimization already present in FA4.

| Same-policy comparison | Always correct, µs | Skip identity correction, µs | Speedup |
| --- | ---: | ---: | ---: |
| FA4, verified artifact repeat | 103.23 | 94.22 | 1.096× |
| Gluon port, first run | 112.74 | 106.86 | 1.055× |
| Gluon port, independent repeat | 113.45 | 108.02 | 1.050× |

Results: [FA4 pair with verified assembly](real_kernel_results/semantic_attention_fa4_correction_verified/results.json),
[initial FA4 pair](real_kernel_results/semantic_attention_fa4_correction/results.json),
[Gluon pair and context](real_kernel_results/semantic_attention_gluon_correction/results.json),
[independent controls](real_kernel_results/semantic_attention_gluon_correction_repeat/results.json).
All three seeds have exactly equal output and LSE within each same-policy pair,
including repeated launches, poisoned fresh outputs, and graph replay.

The [Gluon port](gluon_attention_rescale_ablation.py) uses a valid logical tensor
reduction to a uniform scalar over the correction region. Its branch encloses
only accumulator loads, multiplies, and stores; channel waits and arrivals stay
outside. The [compiled audit](real_kernel_results/semantic_attention_gluon_correction/compiled_protocol_audit.json)
finds identical normalized TTGIR wait/arrival/commit sequences. It retains the
same 512 TMEM columns, 230161 shared-memory bytes, 16 launch warps, 128 registers,
and 20 local-memory words reported as `n_spills`. The reduction introduces
intra-region synchronization, so the optimization has a cost as well as a benefit.

The initial FA4 timing run accidentally saved artifact path strings instead of
assembly bytes; those records are explicitly marked invalid as code evidence.
The final normal-input N4096 repeat retains verified PTX and ELF cubins and
reproduces the paired gain. Its source snapshots and configuration agree with
the initial run. The [verified assembly audit](real_kernel_results/semantic_attention_fa4_correction_verified/compiled_protocol_audit.json)
finds two guarded correction regions, each containing eight TMEM loads,
64 packed multiplies, eight TMEM stores, and a fence. Their release operations
remain after the branch targets. Static synchronization counts and resources
match between the two variants.

Two controls constrain the conclusion:

- With the original threshold-zero policy, identity skipping is slightly slower:
  113.23 → 113.85 µs. Adding this branch universally is not justified.
- The existing two-CTA Gluon default is already 109.82 µs in the independent
  repeat, versus 108.02 µs for the one-CTA port. Its advantage over that default
  is small, varying by seed. The repeat's Tritonbench and FA4 controls are
  153.58 and 94.21 µs respectively. The 5% paired gain does not establish a
  large gain over the strongest existing Gluon implementation.

FA4's selected configuration on this GPU has `ex2_emu_freq=0`. Polynomial
exponential emulation is not the explanation for these B300 results.

## Typed score/probability reuse

The other [ablation](gluon_attention_semantic_ablation.py) retains the arithmetic
and pipeline but compares reused FP32 score storage with separately allocated
BF16 probability storage. Both modes add the same explicit score/P dependencies
to channel waits. The unchanged example is kept as a distinct reference.

| Head dimension, BM256/BN64, one CTA | Separate P, µs | Reused P, µs | Reserved TMEM columns, separate → reuse |
| --- | ---: | ---: | --- |
| D128 | 121.19 | 120.10 | 512 → 512 |
| D64 | 94.12 | 94.09 | 512 → 256 |

Results: [D128](real_kernel_results/semantic_attention_storage_d128/results.json),
[D64](real_kernel_results/semantic_attention_storage_d64/results.json).
Outputs and LSE match exactly. The D64 reservation reduction produces no
meaningful speedup with this schedule. Persistent grid size, register allocation,
and shared memory also constrain residency; a smaller TMEM reservation does
not by itself create more useful overlap.

There is a concrete capacity boundary: at D128/BM256/BN128 with one CTA,
the separate-P variant needs 640 columns and is rejected against the 512-column
hardware limit before launch. The reused-P version runs with 512 columns.
The [capacity probe](real_kernel_results/semantic_attention_capacity_bm256_bn128/results.json)
retains IR/PTX/cubin and the resource error. This establishes an enabled tile,
not a measured same-configuration speedup over a kernel that cannot launch.

The [protocol notes](semantic_attention_protocol.md) explain the separate compiler
opportunity: typed physical subview reuse across loop generations. It requires
score consumption, probability publication, asynchronous MMA A-read lifetime,
the next score overwrite, and tail/persistent-tile transitions to be proved.
PTX specifically orders a prior MMA's A reads before a following MMA's D writes
under the documented same-warp/CTA-group conditions. Arbitrary asynchronous
issue order is insufficient. This can preserve overlap without waiting for
the entire PV computation to finish.

## Accuracy limits and validation

The FP32 reference disables TF32 and computes chunked attention. Output checks
use `atol=0.01, rtol=0.03`; base-two LSE checks use `atol=0.01, rtol=0.001`.
The main comparisons pass across three random seeds, including allocation
guards and fresh outputs initialized to NaNs. Multiplying Q/K/V by two also
passes the unchanged tolerances for all six correction variants at
[N512](real_kernel_results/semantic_attention_correction_stress_s2_n512/results.json)
and [N4096](real_kernel_results/semantic_attention_correction_stress_s2_n4096/results.json),
across three seeds. These short runs validate correctness; their timings are
not performance evidence.

At input scale four and N512, the
[diagnostic run](real_kernel_results/semantic_attention_correction_stress_s4_n512_diagnostic/results.json)
fails the original tolerance: both FA4 variants fail on the same 3 of 1048576
output elements, and both threshold-eight Gluon variants fail on the same 18.
The threshold-zero Gluon variants pass. Every skip/always pair remains exactly
equal, and LSE/guard checks pass. Tolerances were not relaxed.

Thus the identity-elision step preserves the tested results, while changing
the softmax-shift policy can exceed the original accuracy contract. The combined
Gluon change is not suitable as a default semantics-preserving compiler rewrite.
Any numerical-policy annotation needs an explicit accepted error contract and
further validation for its intended input distribution.

## Compiler direction

There are two distinct semantic problems:

1. **Value-dependent removal of memory updates.** Under an explicit permitted
   softmax-shift policy, propagate the fact `alpha == 1` into the correction
   partition, prove the accumulator ownership and effects, and remove only
   the identity read/modify/write. Preserve the synchronization protocol.
   Profitability depends on how often the predicate holds and the cost of
   making it uniform. The threshold-zero control rules out an unconditional
   compiler policy that inserts this branch everywhere.
2. **Typed asynchronous lifetime reuse.** Recognize a bounded repeated QK/P
   protocol rather than weakening the existing packing pass's guards. This
   is principally a tile-capacity opportunity in these measurements; the
   D64 result does not support claiming a throughput gain from halving TMEM.

The first combines algebraic/value semantics with race-free ownership. The
second combines physical alias/lifetime analysis with operand-specific hardware
ordering. Race freedom alone establishes neither optimization. No native compiler
code or Tritonbench source was changed in this follow-up experiment.

## Reproduction

From the Triton repository root, using the prepared isolated environment:

```bash
../.venv-fa4-reference/bin/python experiment_notes/semantic_attention_experiment.py --variants triton,triton-ast,fa4,gluon-default --output /tmp/attention-matched
../.venv-fa4-reference/bin/python experiment_notes/semantic_attention_experiment.py --variants fa4,fa4-always-correct --output /tmp/attention-fa4-correction
../.venv-fa4-reference/bin/python experiment_notes/semantic_attention_experiment.py --variants triton,fa4,gluon-default,gluon-rescale0,gluon-rescale0-skip,gluon-rescale8,gluon-rescale8-skip --gluon-block-m 256 --gluon-block-n 128 --output /tmp/attention-gluon-correction
../.venv-fa4-reference/bin/python experiment_notes/semantic_attention_experiment.py --variants gluon-reuse,gluon-separate --gluon-block-m 256 --gluon-block-n 64 --output /tmp/attention-storage-d128
../.venv-fa4-reference/bin/python experiment_notes/semantic_attention_experiment.py --variants gluon-reuse,gluon-separate --head-dim 64 --gluon-block-m 256 --gluon-block-n 64 --output /tmp/attention-storage-d64
```

Every output directory must be new. Results include inputs/strides, tolerances,
source snapshots, repository revisions/patches, native-library hashes, resources,
errors, and per-round timings. The FA4 loader and source variant are under
`../fa4-force-correction-ablation`; they leave the pinned reference untouched.
No native rebuild is required for these Python-only experiments.
