# LayerNorm backward ownership experiment, 2026-11-07

See the [experiment index](README.md) and [consolidated findings](findings_overview.md)
for the compiler, attention, and LayerNorm results together.

LayerNorm backward provides a useful ownership optimization target. Reassigning
each partial-gradient group to one program removes competing writers and permits
register accumulation. Within each matched scratch-precision policy, the
prototype preserves the original stage-one signature,
launch grid, lock/count reset, final lock/count values, and final reduction.
It is a source prototype of a possible compiler transformation; no automatic
native compiler pass has been implemented.

The source is Tritonbench's
[`layer_norm/tutorial.py`](../../tritonbench/tritonbench/operators/layer_norm/tutorial.py).
This is an actual registered Tritonbench provider: in
[`layer_norm/operator.py`](../../tritonbench/tritonbench/operators/layer_norm/operator.py),
`@register_benchmark()` registers `triton_layer_norm`, which calls
`tutorial.layer_norm(...)`. Its backward launches `_layer_norm_bwd_dx_fused`
and `_layer_norm_bwd_dwdb`; the experimental baseline imports those exact
device functions. With the default M=4096 and N=512*i for i=2..31, both
4096×1024 and 4096×4096 belong to Tritonbench's default shape sweep.

The filename reflects the provider's tutorial origin. Tritonbench also
registers `triton_fused_layer_norm` from `fused_triton.py` and other providers.
Our reported gain is over the tutorial-derived provider's backward kernels in
a controlled harness, with tuned launch parameters and matched FP32 partial
buffers for BF16. It is not a stock Tritonbench runner result or evidence of
the same gain over its fastest LayerNorm provider.

All experiments measure the complete backward operation producing DX, DW, and
DB, excluding allocation and forward. Tritonbench sources remain unchanged.
The ownership implementation, harness, and proof are linked below.

## BF16 and the earlier accuracy explanation

Calling this an “issue with the original kernel” was too strong. The original
tutorial wrapper allocates partial DW/DB buffers in the input dtype. With BF16
inputs this rounds each partial update to BF16. Our original and modified kernels
both failed the selected `atol=0.01, rtol=0` check. That tolerance and input
distribution were borrowed from the tutorial's FP16 test; they are not an
established universal BF16 training accuracy requirement. The failure therefore
does not establish that the original kernel is buggy or unsuitable for training.

There is a concrete precision effect: even with one input row per group, the
FP32 `dy * xhat` product is rounded to BF16 before the final reduction. In the
[initial BF16 screen](layernorm_ownership_results/screen-bf16-m4096-n1024/results.json),
G=M=4096 has DW maximum absolute difference 0.0625 and 292/1024 elements outside
the selected tolerance, while DB matches exactly. Summing in FP32 inside each
owner but still writing BF16 partials does not eliminate this final partial-store
rounding. DX passes throughout that screen.
The [FP64-reference recheck](layernorm_ownership_results/check-bf16-inputscratch-fp64ref-m4096-n1024/results.json)
confirms the same one-row-per-group DW result and rejects all nine tested
Triton variants using BF16 scratch; native PyTorch and Quack pass. Thus the
partial-buffer precision effect remains after correcting the reference.

The matched precision experiment instead gives **both** the original locked
kernel and the ownership kernel FP32 partial buffers, retaining BF16 inputs,
weights, and all output gradients. The original device arithmetic already
supports these buffer types; the final reduction remains its original source.
This allocation policy is a separate change from ownership. Claims about the
same caller apply within the matched FP32-scratch comparison, not to the stock
wrapper's BF16-scratch allocation.

A separate reference error occurred in seed 2 at M4096/N4096 after this change.
All implementations, including native PyTorch and Quack, disagreed with our FP32
reference at one DW element. At column 204:

| Evaluation | Unrounded DW | BF16 result |
| --- | ---: | ---: |
| Original FP32 reference | 6.203125 | 6.1875 |
| FP64 sum of the same FP32 row products | 6.203126586766416 | 6.21875 |
| FP64 derivative evaluation with common FP32 saved statistics | 6.203126535480063 | 6.21875 |
| Original, ownership, native PyTorch, Quack | — | 6.21875 |

The FP32 reference landed exactly on the midpoint and rounded to the wrong
neighbor relative to both higher-precision evaluations. Four launches
of every implementation pass against both higher-precision references at the
unchanged tolerance. The [failed run](layernorm_ownership_results/confirm-bf16-f32scratch-m4096-n4096/results.json)
and [diagnostic](layernorm_ownership_results/diagnose-bf16-f32scratch-seed2-m4096-n4096/results.json)
are preserved. Confirmation uses an explicitly recorded FP64 reference;
saved mean/rstd remain FP32 for every implementation. This fixes the reference,
not the kernel or the tolerance. The new oracle explicitly rounds directly to
the nearest even BF16/FP16 value: a generic FP64-to-BF16 cast can double-round
through FP32 for values extremely close to a midpoint. That separate casting
hazard does not affect the diagnosed column, which is over three FP32 ULPs above
the midpoint, but the oracle handles it for other elements.

The first FP64-reference N4096 confirmation then exposed one remaining
order-sensitive result: original G256/W8 passed before and after timing but
failed the fresh post-timing launch for seed 2 at one DW element (0.03125
difference). All other gradients, guards, and lock/count checks passed. This is
consistent with nondeterministic FP32 summation crossing a BF16 rounding
boundary; the report does not identify the column, so it is not established to
be the earlier column 204. The remaining eight configurations, including all ownership variants,
passed all recorded checks. G256/W8 remains excluded from the
headline N4096 baseline across the confirmation history, even if a later run
passes. The final repeat also rejects G256/W8 at seed 2, this time in the
post-timing check. This is a rejection under the selected strict check, not evidence of a
memory race or of unacceptable training accuracy.

## Whole-backward performance

All timings use one NVIDIA B300 SXM6 PC, 148 SMs, sm_103. Reported confirmation
timings are the median of three seed medians. Each seed uses 15 shuffled, interleaved
rounds, 32 complete backward invocations per CUDA graph replay, and at least
200 ms warmup. Required lock/count reset and final reductions are timed.
Provider-specific output allocation and common saved-statistics generation are
outside timing. Native PyTorch uses functional `native_layer_norm_backward`;
its `.out` overload was excluded because its fallback adds output copies.
Quack 0.6.5 includes its full DX/partial-gradient kernel, partial reductions, and
output casts. All providers return all three gradients.

| Inputs/outputs; partial buffers | Shape M×N | Best eligible original (µs) | Same-grid owner (µs) | Speedup | Native PyTorch (µs) | Quack (µs) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BF16; FP32 partials | 4096×1024 | 47.95 | 23.75 | 2.02× | 23.16 | 30.94 |
| BF16; FP32 partials | 4096×4096 | 77.18 | 43.44 | 1.78× | 56.52 | 42.96 |
| FP32; FP32 partials | 4096×1024 | 46.63 | 24.77 | 1.88× | 22.43 | 29.06 |
| FP32; FP32 partials | 4096×4096 | 76.22 | 54.42 | 1.40× | 82.23 | 48.91 |
| FP16; FP16 partials | 1151×8192 | 34.98 | 19.08 | 1.83× | 33.61 | 39.03 |

Each selected table entry passes every recorded check across all three seeds.
BF16 N4096 excludes the rejected G256/W8 original; G512/W8 is the fastest
eligible original. The entire report is explicitly marked as containing a
rejection, rather than reporting every configuration as passing. At fixed
G256/W4, the same BF16 N4096 repeat improves 78.14→46.08 µs (1.70×),
showing a gain without changing either tuning parameter.

The earlier fully passing BF16 N1024 confirmation measured 46.17→23.95 µs
(1.93×); the FP64-reference confirmation above measures 47.95→23.75 µs
(2.02×). The device kernels are unchanged by the oracle update. The practical
claim is approximately a twofold gain, with run-to-run variation. BF16 N4096
improves 77.18→43.44 µs (1.78×) in the final repeat, closely matching the
first FP64-reference run’s 77.23→43.39 µs for those same valid configurations.

BF16 N1024 remains slightly slower than native PyTorch, and BF16 N4096 is
close to Quack. FP32 N4096 is slower than Quack; the FP16 test shape beats both
references. These are gains over the existing Tritonbench implementation,
not a universal claim of fastest LayerNorm backward.

Reports: [BF16 N1024](layernorm_ownership_results/confirm-bf16-f32scratch-fp64ref-m4096-n1024/results.json),
[BF16 N4096 repeat](layernorm_ownership_results/repeat-bf16-f32scratch-fp64ref-m4096-n4096/results.json),
[FP32 N1024](layernorm_ownership_results/same-grid-f32-m4096-n1024/results.json),
[FP32 N4096](layernorm_ownership_results/same-grid-f32-m4096-n4096/results.json),
[FP16](layernorm_ownership_results/confirm-fp16-m1151-n8192/results.json).

The original was screened over group counts and warp counts before independent
confirmation. FP32 screens cover G={64,128,256,512,1024,2048,4096}, W={1,2,4,8};
BF16 with FP32 scratch covers G={128,256,512,1024}, W={2,4,8}; the expanded FP16
original screen covers G={64,96,128,256,512,1151}, W={1,2,4,8}. These are the best
tested original configurations, not a claim of exhaustive tuning. FP16 G64 was
rejected numerically and excluded from performance selection.

The reduced-grid owner removes inactive launches and the reset, and is measured
separately. The headline variant deliberately retains the original M-program
grid and reset: only programs p<G do useful work, processing rows p+kG.
At matching G/W, reduced-grid and same-grid outputs agree exactly in the
recorded confirmations. Numerical comparisons against the locked kernel permit
its nondeterministic lock acquisition order.

## Why this relates to race freedom

The original launches one program per input row. Each program computes its
own DX row, then contributes DW/DB to partial buffer `row % G`, where G is the
number of partial-gradient groups. Programs sharing a group acquire its lock,
read/add/write the partials, and release the lock. The rewrite instead gives
each group to one owner program p, which computes rows `p, p+G, p+2G, ...`.
For eight rows and two groups:

| Partial buffer | Original programs updating it | Owner's row loop |
| --- | --- | --- |
| 0 | Programs 0, 2, 4, 6 compete for its lock | Program 0 computes rows 0, 2, 4, 6 |
| 1 | Programs 1, 3, 5, 7 compete for its lock | Program 1 computes rows 1, 3, 5, 7 |

Each owner keeps its two partial-gradient vectors in registers and stores them
once after its row loop. Each row still uses parallel computation across
columns, and DX is written as that row is processed. The unchanged second
kernel combines the groups. For M=4096/G=256, each owner processes 16 rows, so
the logical partial-store count falls by 16×; intermediate partial loads also
disappear. This counts source memory requests, not measured DRAM traffic.

The measured same-grid version still launches M programs: owners p<G do the
work, and the remaining programs exit. It retains the caller's lock/count
reset and preserves their final values. It therefore demonstrates a gain
without changing the caller's launch grid. Fewer active programs and longer
register lifetimes can reduce parallelism, so this work redistribution needs
a profitability decision as well as a correctness proof.

The useful fact is **exclusive ownership of private reduction state**, together
with complete row coverage and absence of intermediate observers. Each row r
has exactly one owner r%G and one iteration r//G. DX rows are disjoint, each
partial DW/DB row has one writer, and the unchanged final reduction runs after
stage one completes. This makes the inter-program lock unnecessary and permits
keeping intermediate partials in registers.

The ordinary race freedom of the locked original alone would not justify this
rewrite. The compiler also needs nonaliasing, immutable inputs, a one-dimensional
launch domain, freshly zeroed private lock/count storage, and no concurrent
observer. The owner preserves rounding after each row to the selected scratch
dtype and chooses increasing row order, one serialization allowed by the private
lock protocol. This is not permission for arbitrary floating-point reassociation.
The [protocol](layernorm_ownership_protocol.md) states these assumptions and
the proof in detail.

Compiled FP32 audits show three static atomic instruction sites becoming zero,
partial accumulators carried in registers, W loaded before the owner loop, and
byte-identical final reduction through cubin. The same-grid version retains the
original argument order and compiled types, reads the logical grid size, and
preserves final Count/Lock state. It uses no hidden lane/warp scalars.

Nsight Compute supports contention as one contributor. For FP32 M4096/N1024,
G256/W4, the original records 36,690 global-atomic set accesses; the owner records
zero. The original's noncontended G=M=4096 control records 12,288 (=3M).
At G256, 8,448 successful protocol atomics are required, so the higher observed
access count is consistent with retries. The counter measures accesses, not a
direct dynamic CAS instruction count. The original G256 profile reports 32.4
of 55.7 average warp cycles between issued instructions waiting at CTA barriers
(58.2%), and reaches 8.47% of DRAM
throughput. Profiling changes clocks and execution conditions; profiler durations
are not used as benchmark results.

Ownership simultaneously removes atomics, repeated scratch traffic and some
barriers, reuses weights, changes scheduling, and increases register lifetime.
The gain cannot be assigned entirely to lock contention. The same-grid control
does establish that changing the host grid or removing the reset is unnecessary
for the observed improvement.

## Validation and scope

Confirmation validates DX/DW/DB before and after timing. Preallocated outputs
are additionally checked after fresh NaN poisoning at both points. Guards cover
Triton buffers and Quack outputs; native functional outputs and Quack's internal
partials are unguarded. Final Triton lock/count state is checked. The independent
analytic oracle is also cross-checked with FP32
autograd. Reference precision is explicit in each report; historical runs retain
their original FP32 oracle. The selected tolerance remains 0.01 absolute, zero
relative. Unrounded reference errors are recorded as well.

All five boundary cases pass for the locked, reduced-grid owner, and same-grid
owner across three seeds and four validation phases: FP32 M1/N1/G4;
FP32 M17/N33/G64; BF16 with FP32 scratch M257/N1000/G256;
FP16 with input-dtype scratch M257/N1000/G256; and FP32 M513/N4097/G128.
They exercise scalar rows, M<G, masked columns, a single wrapped row, inactive
programs, and uneven multi-iteration loops. Their one-round/two-replay timings
are correctness controls only. [Boundary purposes](layernorm_ownership_protocol.md)
and per-case report links appear in the machine-readable summary.

This establishes a promising real-kernel source transformation on the tested
shapes and distributions, not general LayerNorm correctness or a training
convergence result. It does not establish a speedup on every dtype/shape or over
every optimized library. M4096 matches the tutorial benchmark's row count;
N1024 and N4096 exercise different row widths. The FP16 M1151/N8192 case is the
tutorial's unit-test shape. No claim is made about the entire Tritonbench suite.

## Compiler next step and reproduction

The next compiler step is a bounded, opt-in recognition of this private
lock/count partial-reduction protocol, with explicit launch/nonaliasing and
initial-state facts. Reconstruct the same-grid owner loop while retaining the
original signature and final-state stores. Reject unmatched effects, aliases,
nonzero entry counts, other grid dimensions, or external observers. Validate
the transformed original against this source prototype, then test a second
kernel with the same reduction protocol before attempting generalization.

This experiment changes Python only. No native rebuild is necessary for these
new files. Existing native experimental flags are all explicitly disabled.
`enable_fp_fusion=False` matches the ordinary default of this local checkout.
Benchmark report directories include source snapshots/hashes, repository revisions
and patches, native-library hash, compiler artifacts, and GPU state.

From the Triton repository root, with the existing reference environment:

```bash
../.venv-fa4-reference/bin/python experiment_notes/layernorm_ownership_experiment.py \
  --dtype bf16 --scratch-dtype fp32 --reference-precision fp64 \
  --rows 4096 --cols 4096 \
  --configs locked:128:8,locked:256:4,locked:256:8,locked:512:8,owner-round:256:8,owner-same-grid:256:4,owner-same-grid:256:8,torch-native,quack \
  --seeds 0,1,2 --rounds 15 --graph-repeats 32 --warmup-ms 200 \
  --autograd-check --screen --output /tmp/layernorm-bf16-fp32scratch-fp64ref
```

The output directory must be new. Supporting files:

- [Machine-readable summary](layernorm_ownership_summary_11_07.json)
- [Ownership kernels](layernorm_ownership_kernels.py), [comparison harness](layernorm_ownership_experiment.py), [reference adapters](layernorm_reference_adapters.py)
- [Ownership proof and compiler preconditions](layernorm_ownership_protocol.md)
- [FP32 N1024 compiled audit](layernorm_ownership_results/same-grid-f32-m4096-n1024/compiled_same_grid_audit.json), [FP32 N4096 compiled audit](layernorm_ownership_results/same-grid-f32-m4096-n4096/compiled_same_grid_audit.json)
- [BF16 N1024 compiled audit](layernorm_ownership_results/confirm-bf16-f32scratch-fp64ref-m4096-n1024/compiled_same_grid_audit.json), [BF16 N4096 audit with recorded original-kernel rejection](layernorm_ownership_results/confirm-bf16-f32scratch-fp64ref-m4096-n4096/compiled_same_grid_audit.json)
- [BF16 N4096 repeat audit](layernorm_ownership_results/repeat-bf16-f32scratch-fp64ref-m4096-n4096/compiled_same_grid_audit.json), [boundary validation audit](layernorm_ownership_results/boundary_validation_audit.json)
- [Direct-rounding CPU checks](layernorm_ownership_results/reference_rounding_cpu_audit.json)
- [Stage-one profile driver](layernorm_profile_probe.py), [BF16 reference diagnostic](layernorm_reference_diagnostic.py)
