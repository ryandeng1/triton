# Consolidated semantic optimization findings

Updated 2026-11-07. [File index](README.md).

The objective is to use ownership, race freedom, asynchronous completion,
physical storage lifetimes, and value properties to improve Triton kernels,
especially kernels already present in Tritonbench. The experiments now contain
both negative compiler results and a promising real-kernel source rewrite.
They do not yet establish a general automatic compiler optimization that speeds
up the strongest Tritonbench implementations across operators.

The clearest next target is the private partial-gradient reduction in LayerNorm
backward. Reassigning work so each partial buffer has one writer gives substantial
gains while retaining the original launch grid and final reduction. The native
compiler work required to discover and perform that rewrite remains outstanding.

## Results at a glance

| Experiment | Result | What the evidence establishes |
| --- | --- | --- |
| Earlier explicit TMEM release scheduling | Recorded 1.01–1.08× recovery from deliberately delayed releases | Controlled scheduling opportunity; historical evidence, not an unchanged real-kernel gain |
| Earlier partial TMEM lifetime packing | Recorded 512→256 columns and about 1.05–1.09× on larger controlled workloads | Storage reuse can matter when it changes a limiting resource; historical evidence |
| Aref scheduling on real FP16/FP8 GEMM and BF16 attention | Compared binaries unchanged | No compiler speedup established on these cases |
| Forced attention single buffering | 1.205× on a slow fixed configuration, but 130.87→198.65 µs on the best tested baseline | The local win does not survive a competitive baseline |
| Matched BF16 attention implementations | Tritonbench 143.47 µs; Gluon 105.28; FA4 90.11 | Faster implementations exist; the full gap was not attributed to one semantic property |
| Identity accumulator rescale elision | FA4 1.096×; manual Gluon port 1.050× under the same threshold-eight policy | A measured semantic effect, with policy and profitability limits |
| Typed attention score/probability reuse | D64 reservation 512→256, time 94.12→94.09 µs | No meaningful speed gain there; another tile becomes launchable by staying within capacity |
| LayerNorm ownership, BF16 with FP32 partials on both sides | About 1.9–2.0× at 4096×1024; 1.78× at 4096×4096 | A source transformation of one registered Tritonbench provider; not a completed compiler pass or fastest-provider claim |

All November 7 GPU results use an NVIDIA B300 SXM6 PC, 148 SMs, sm_103.
Confirmatory performance reports generally use three seeds, 15 shuffled
interleaved rounds per seed, 32 invocations per CUDA graph replay, and at least
200 ms warmup. Tables use the median of seed medians. Clocks are not locked;
compare variants measured together, rather than absolute times from different
runs. Screens, stress cases, and profiler runs have different purposes and are
not interchangeable with confirmation timings.

## Earlier compiler work and the real-kernel outcome

The [November 4 writeup](writeup_11_04.md) records three experimental passes:
explicit TMEM release scheduling, Aref release scheduling, and partial-lifetime
TMEM packing. The work also corrected completion-related memory effects so
ordinary optimizations could not reuse a load from an earlier buffer lifetime.
The recorded Aref validation covers 96 GPU cases; packing covers 28 GPU cases
and 10 focused compiler test files. The passes were experimental and outside
the default pipeline. Their controlled performance figures are historical:
the raw paths in [that summary](summary_11_04.json) are absent in this workspace.
The retained prose rounded the packing range to 1.06–1.09×; the JSON records
1.0467–1.0927× across the larger-grid measurements. A single wave of CTAs showed
no material packing gain. The packing implementation was limited to one CTA
and two straight-line warp partitions.

The subsequent [real-kernel experiments](writeup_11_07.md) tested unchanged
Tritonbench device sources. Aref off/on PTX was identical for 18 FP16 GEMM cases
(two epilogue subtiles, three shapes, three seeds) and nine FP8 GEMM cases.
The tested BF16 attention's off/Aref/packing TTGIR and PTX were also identical.
Relevant releases were already immediately after the last TMEM access; the
late scheduler had no useful independent work to move them across. Restricted
recognition of cyclic ownership was another limit, but recognizing more cases
would not itself create scheduling slack.

Forcing one attention QK buffer reduced reservation from 512 to 256 columns
and improved a fixed configuration from 670.84 to 556.51 µs in an independent
repeat. However, tuning the unchanged baseline found 130.87 µs, and forcing
single buffering there regressed it to 198.65 µs. Uncapped-register and small-grid
controls also regressed. Reducing memory use can sacrifice useful overlap, and
smaller TMEM reservation need not improve occupancy when other resources limit it.

Native changes exposed the Aref scheduling option and a default-off
`enable_tmem_single_buffer` policy before latency inference, with an explicit
latency guard. The recorded build and five focused lit checks passed. These are
implemented compiler experiments; their real-kernel results do not justify
enabling them globally. Evidence and commands are in the
[compiler report](writeup_11_07.md), including the
[best-configuration repeat](real_kernel_results/attention-screen-best-repeat/results.json).

## Attention: what faster implementations taught us

The matched workload is noncausal BF16 attention, B2/H8/N4096/D128, with common
Q/K/V inputs and output plus log-sum-exp validation. The
[matched comparison](real_kernel_results/semantic_attention_comparison/results.json)
reproduces 1.36× for the maintained Gluon example and 1.59× for pinned official
FlashAttention-4 relative to the best previously screened Tritonbench
configuration. Experimental compiler flags are disabled. These are controlled
device-call comparisons, not application-level speedups or a broad survey of
frameworks or agent-generated kernels.

One isolated opportunity is the softmax accumulator correction. When the old
and new softmax shifts are equal, `alpha = exp2(old_shift - new_shift) = 1`.
The correction can skip loading, multiplying, and storing the accumulator while
retaining its synchronization protocol. FA4 already implements this; forcing
the correction back on is a reverse ablation of an existing optimization.

At the same threshold-eight shift policy, the verified FA4 pair improves
103.23→94.22 µs (1.096×), and an independent manual Gluon repeat improves
113.45→108.02 µs (1.050×). Each same-policy pair has exactly equal outputs and
LSE across the recorded checks. These gains explain one component, not the
whole implementation gap. At the original threshold-zero policy, inserting
identity skipping is slightly slower, 113.23→113.85 µs. The existing two-CTA
Gluon control is already 109.82 µs, close to the 108.02 µs port.

An initial FA4 run saved artifact path strings instead of assembly bytes; it
remains marked invalid as compiled-code evidence. The reported verified repeat
retains actual PTX/cubin and reproduces the gain. Artifact provenance matters
separately from timing and numerical validation.

The shift policy is a separate numerical choice. Normal inputs and a scale-two
stress test pass the selected checks: output `atol=0.01, rtol=0.03`, and base-two
LSE `atol=0.01, rtol=0.001`. These are the experiment's checks, not a universal
attention accuracy contract. At scale four and N512, both FA4
threshold-eight variants fail the same three output elements, and both Gluon
threshold-eight variants fail the same 18; threshold-zero Gluon passes.
Same-policy skip/always pairs remain exactly equal. No tolerance was relaxed.
Thus identity elision and changing the shift policy must not be conflated into
an unconditional semantics-preserving compiler optimization.

The storage experiment compares reused FP32 score storage with separate BF16
probability storage. At D64, reuse halves reserved TMEM but gives essentially
identical runtime. At D128/BM256/BN128, separate storage needs 640 columns and
cannot launch against the 512-column limit; reuse fits in 512. That is a tile
capacity benefit, not a measured speedup over the unlaunchable configuration.
Correct reuse requires proof of physical subviews, loop generations, asynchronous
MMA operand-read completion, and initialization/drain behavior. The
[attention findings](semantic_attention_findings_11_07.md) and
[protocol](semantic_attention_protocol.md) preserve those distinctions and link
the verified assembly, numerical failures, controls, and reproduction commands.

## LayerNorm: the ownership transformation

This target is an actual Tritonbench provider. In
[`layer_norm/operator.py`](../../tritonbench/tritonbench/operators/layer_norm/operator.py),
registered `triton_layer_norm` calls `tutorial.layer_norm`. Its backward uses
`_layer_norm_bwd_dx_fused` and `_layer_norm_bwd_dwdb` from
[`tutorial.py`](../../tritonbench/tritonbench/operators/layer_norm/tutorial.py).
The name reflects its tutorial origin; it is not a separately invented benchmark.
M4096/N1024 and M4096/N4096 are in the default shape sweep.

Tritonbench also registers other LayerNorm implementations. We directly invoke
the tutorial-derived backward kernels in a controlled harness, tune launch
parameters, and use common saved statistics. The reported gain is over that
provider's best tested numerically eligible configuration. It is not a stock
runner measurement or a demonstrated 1.8–2× gain over the fastest registered
LayerNorm provider.

The original launches one program per input row. Each computes that row's DX,
then contributes to DW/DB partial buffer `row % G` under a lock. Multiple
programs contend for each partial buffer. The rewrite gives one program ownership
of each partial buffer and all rows contributing to it:

| Partial buffer, for eight rows and G=2 | Original | Ownership rewrite |
| --- | --- | --- |
| 0 | Programs 0, 2, 4, 6 take turns updating it under a lock | Program 0 computes rows 0, 2, 4, 6 |
| 1 | Programs 1, 3, 5, 7 take turns updating it under a lock | Program 1 computes rows 1, 3, 5, 7 |

Each owner keeps partial sums in registers, writes each row's DX, and stores
DW/DB partials once. The second reduction stays unchanged. With M4096/G256,
each owner handles 16 rows, reducing source-level partial stores by 16× and
eliminating intermediate partial loads, atomics, and lock contention. It also
reuses weights. Fewer active programs and longer-lived registers are potential
costs, so the benefit is shape-dependent and cannot be attributed solely to CAS.

The headline prototype retains the original M-program grid and argument list;
programs with IDs at least G do no work. It retains the caller's reset and
preserves final lock/count values. This is therefore a possible kernel-body
rewrite without changing the host launch grid, within a fixed scratch policy.
Compiled audits verify no atomics and an unchanged final reduction. Profiling
supports contention as a contributor but is not used as performance timing.

Legality requires complete row coverage, unique writers, nonaliasing buffers,
immutable inputs, a one-dimensional launch, fresh private zeroed lock/count
state, and no intermediate observers. Every row r maps uniquely to owner r%G
and loop iteration r//G. Kernel completion still orders the final reduction.
The owner preserves per-row rounding to the chosen scratch dtype and chooses
increasing row order, one permitted original serialization under those
preconditions. This does not promise equality with every original lock order
or permit arbitrary floating-point reassociation. The
[ownership protocol](layernorm_ownership_protocol.md) gives the full proof.

## LayerNorm measurements and BF16 corrections

All following times include the complete DX/DW/DB backward computation,
required reset, and final reductions; allocation and forward are excluded.
The BF16 rows give both original and owner FP32 partial buffers, while keeping
inputs, weights, and all output gradients BF16.

| Inputs/outputs; partial dtype | M×N | Best eligible original (µs) | Same-grid owner (µs) | Speedup | Native PyTorch (µs) | Quack (µs) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BF16; FP32 | 4096×1024 | 47.95 | 23.75 | 2.02× | 23.16 | 30.94 |
| BF16; FP32 | 4096×4096 | 77.18 | 43.44 | 1.78× | 56.52 | 42.96 |
| FP32; FP32 | 4096×1024 | 46.63 | 24.77 | 1.88× | 22.43 | 29.06 |
| FP32; FP32 | 4096×4096 | 76.22 | 54.42 | 1.40× | 82.23 | 48.91 |
| FP16; FP16 | 1151×8192 | 34.98 | 19.08 | 1.83× | 33.61 | 39.03 |

An earlier passing BF16 N1024 confirmation measured 1.93×; the practical result
is approximately a twofold gain with run-to-run variation. At fixed G256/W4,
BF16 N4096 improves 78.14→46.08 µs (1.70×), so its gain does not require changing
those tuning parameters. The owner does not win against every reference: native
PyTorch is slightly faster at BF16 N1024, and Quack is close at BF16 N4096 and
faster at FP32 N4096. The FP16 test shape beats both references.

Three distinct numerical findings must remain separate:

1. **BF16 partial-buffer precision.** The stock wrapper stores partials in the
   input dtype. Repeated BF16 rounding makes both original and owner fail our
   selected 0.01 absolute, zero-relative check. Even with one row per partial,
   product rounding gives 292/1024 DW mismatches and maximum difference 0.0625;
   DB is exact. This remains with the corrected FP64 oracle. The check was
   borrowed from the tutorial's FP16 test, not an established universal BF16
   training contract. Calling this a demonstrated original-kernel bug was too
   strong. FP32 scratch is a separate allocation/precision change applied to
   both sides, not a consequence of ownership alone.
2. **An error in our reference.** At N4096, seed 2, DW column 204, the FP32
   reference landed on 6.203125 and rounded down to BF16 6.1875. FP64 evaluation
   gives approximately 6.203126535 and rounds up to 6.21875, matching all four
   diagnosed implementations. The explicit FP64 oracle preserves common FP32
   saved statistics and uses direct nearest-even output rounding to avoid an
   intermediate-FP32 double-rounding hazard. Tolerances were unchanged.
3. **A remaining original-configuration rejection.** With FP32 scratch and the
   corrected oracle, original G256/W8 at BF16 N4096 still fails one DW element
   by 0.03125 in seed 2 in both confirmations. This is consistent with reduction
   order crossing a BF16 boundary; its index was not recorded, so it is not
   established to be column 204. It remains excluded from the headline baseline;
   G512/W8 is the fastest eligible original. The ownership variants and the other
   eligible configurations pass every recorded check. This is not proof of a
   race or of unacceptable training accuracy.

All five boundary cases pass across three seeds and four validation phases,
including M<G, masked columns, a single wrapped row, and uneven multi-iteration
loops. Guards, fresh poisoned outputs, final lock/count state, exact paired
outputs, compiler artifacts, and an autograd cross-check support the result.
Guard/poison coverage differs for native functional and Quack internal buffers;
the [detailed findings](layernorm_ownership_findings_11_07.md) state those limits.
They also link every selected performance report, rejected run, numerical
diagnostic, and artifact audit. No training convergence claim is made.

## What this means for compiler work

The results do not show a fundamental inability to improve Triton kernels.
They show that a narrow release mover often has nothing useful to move, that
storage reduction must preserve profitable overlap, and that competing resources
can prevent a smaller allocation from speeding up a kernel. Faster reference
implementations and the LayerNorm rewrite demonstrate useful transformations
outside the earlier passes' scope.

Race freedom is part of the argument, but the useful facts are more specific:
exclusive ownership and no intermediate observers for LayerNorm; an identity
value plus accumulator effects for rescale elision; physical regions, generations,
and asynchronous operand completion for storage reuse. Fully ordering accesses
does not by itself make storage reuse legal: A.write → B.write → A.read would
destroy A if A and B shared storage before A's read completed.

The proposed next implementation is a bounded, opt-in compiler transformation
for the private lock/count partial-reduction protocol. It should obtain explicit
launch, nonaliasing, and initial-state facts; reject unsupported effects or
observers; emit the same-grid owner loop; and preserve the original final-state
stores and chosen numerical policy. Compare the transformed original with the
source prototype, then test a second kernel with the same protocol before
generalizing. Broader representative BF16 shapes and matched full-gradient
providers are still needed before making a strongest-provider claim.

The implemented native passes and their tests are documented in
[the compiler report](writeup_11_07.md). The attention ablations and LayerNorm
ownership rewrite are Python/source prototypes; neither is an automatic native
pass. Their detailed reports contain reproduction commands and source/native
hashes. Native changes require a rebuild under the repository instructions;
Python-only experiments and this documentation consolidation do not. Failed
and superseded artifacts are preserved as history, not silently promoted into
successful performance evidence.
