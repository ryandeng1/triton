# LayerNorm backward ownership experiment

This experiment changes work ownership in the unchanged mathematical LayerNorm
backward computation. It does not implement a native compiler pass. The source
baseline is [`tutorial.py::_layer_norm_bwd_dx_fused`](../../tritonbench/tritonbench/operators/layer_norm/tutorial.py),
whose SHA256 at the start of the experiment is
`58716180088be8a05e3e21fb2bab2db2bc5e7780fd3dd7e680e682a82ecdd048`.
The prototype is [`layernorm_ownership_kernels.py`](layernorm_ownership_kernels.py),
and measurements are produced by
[`layernorm_ownership_experiment.py`](layernorm_ownership_experiment.py).

## Original protocol and observable outputs

The original stage one launches M programs. Program r computes one row of DX
and contributes to DW/DB partial row `r % G`, where G is `GROUP_SIZE_M`.
Programs sharing that partial row serialize through an atomic-CAS lock. A
separate count flag distinguishes the first writer from later writers: the
first writes its contribution without reading uninitialized partial storage;
later writers load the stored partial and add their contribution. The program
then publishes its writes and releases the lock.

The caller must reset both locks and first-writer flags before every backward
invocation, including every captured graph iteration. A stale count flag can
cause accumulation into a previous invocation's gradients. The experimental
timing includes this required reset for the locked variant.

The semantic boundary used here returns DX, final DW, and final DB. Lock words,
count flags, and intermediate partial gradients are private implementation
storage. They have no observers during stage one. The final reduction is their
only consumer and executes after stage one has completed in the same stream.
This boundary is stronger than a claim that arbitrary callers of the original
device function may observe only those three outputs.

## Preconditions

The proof applies to positive M, N, and G with the launch domains and buffer
sizes enforced by the harness. X, DY, and DX are contiguous M-by-N tensors;
W has N elements; Mean and Rstd have M elements; partial DW/DB have G-by-N
elements; final DW/DB have N elements. Pointer/index arithmetic must remain
within its represented integer range. The tested sizes satisfy this condition.

Every writable buffer is disjoint from every other writable buffer and from
the read-only inputs. Inputs remain immutable throughout the two-stage operation.
No concurrent kernel, host access, or callback observes intermediate state.
In particular, scratch cannot alias X, DY, W, saved statistics, or DX. Otherwise
removing intermediate scratch stores could change later input loads, even when
each individual program has exclusive write ownership of its partial buffer.

The reduced-grid owner JIT requires exactly `min(M, G)` programs. Its public
helper checks shapes and contiguity; disjointness is a caller precondition.
The additional same-grid variant described below retains M programs in a
one-dimensional grid. The experiment allocates distinct guarded buffers.
Standalone callers must establish the same preconditions rather than relying
on numerical tests to prove them.

## Coverage, uniqueness, and lifetime proof

Owner program p processes rows `p, p+G, p+2G, ...` below M. Euclidean division
gives every valid row r a unique pair `(p, k) = (r % G, r // G)`. Therefore:

1. Every original row contribution is computed exactly once.
2. No two programs write the same DX row. Column masks restrict every load/store
   to `0 <= col < N`, and padded reduction lanes are explicitly zeroed.
3. Every active partial-gradient row p has exactly one writer, owner p. Its
   contributions stay in that program's registers until its final partial store.
4. The first row exists for every launched owner because `p < min(M, G)`.
   Initializing the accumulator from that first contribution avoids reading
   poisoned or uninitialized scratch and avoids introducing an add-zero step.
5. The final reduction reads exactly `min(M, G)` partial rows. Inactive rows when
   M < G need not be initialized and are never read.

Per-program registers hold each partial value until the program's final store.
There are no intermediate consumers that require earlier publication. The
ordered kernel boundary completes all producer stores before the unchanged
`_layer_norm_bwd_dwdb` reads them. No inter-program lock or barrier is needed
inside the owner kernel. Ordinary intra-program reduction synchronization
remains the compiler's responsibility and is retained in generated code.

Program IDs and loop/control scalars are uniform within logical Triton programs.
The prototype does not use physical lane/warp IDs to introduce hidden scalar
variation.

## Numerical contract

In the tutorial, W is converted to FP32 immediately after loading. Consequently
`(dy * xhat).to(w.dtype)` produces an FP32 contribution even when inputs and
partial buffers are BF16. The contribution must not be rounded to BF16 before
adding the prior partial. The baseline performs FP32 addition followed by
rounding at each partial-buffer store.

The `owner-round` variant follows the same recurrence: it initializes from the
first FP32 contribution and rounds to the scratch dtype; every later step adds
the new FP32 contribution to the promoted prior scratch value, then rounds again.
It does this in increasing row order within each group. The original lock
acquisition order is nondeterministic, so a particular baseline run can differ
bitwise. Under the private-state and no-other-communication preconditions above,
the owner chooses a serialization that the original lock protocol permits.
This is a refinement of the available serializations, not a general license to
reassociate floating-point expressions. Matching this source-level recurrence
still requires compatible floating-point compilation semantics and testing of
the generated code.

The `owner-fp32` variant accumulates in FP32 until its final scratch store. When
scratch is BF16/FP16 this changes the rounding policy and is reported separately.
When scratch is FP32 the two policies coincide; the smoke artifacts confirm
byte-identical generated code. No numerical tolerance is widened to admit an
otherwise failing variant.

The harness independently selects partial-buffer allocation precision with
`--scratch-dtype input|fp32`. The `input` choice reproduces the tutorial wrapper's
allocation dtype. The `fp32` choice allocates both partial buffers in FP32 for
both the original locked kernel and the owner kernel, while keeping BF16 inputs
and BF16 DX/DW/DB outputs when `--dtype bf16` is used. No device-source change is
needed: original contributions are already FP32, partial pointer types determine
load/store precision, and the unchanged final reduction accepts FP32 partials
and rounds only its final output stores to BF16.

Changing BF16 partial buffers to FP32 is a separate allocation and numerical
policy from changing ownership. It doubles their storage and changes the
partial-sum rounding history. A valid ownership comparison holds that choice
fixed in both variants. Such a comparison does not establish a kernel-only
rewrite of the stock BF16 wrapper's BF16 allocations. The kernel-only scope is
the matched pair with the same chosen buffer types, launch, and reset policy.
With FP32 partials, `owner-round` already accumulates in FP32, so treating
`owner-fp32` as a distinct precision improvement would be misleading.

The initial BF16-input/BF16-scratch cases failed the experiment's fixed
`atol=0.01, rtol=0` check. This threshold and input distribution were adopted
from the tutorial's FP16 unit-test setup; failure under that imposed BF16 check
does not by itself establish an application bug or violation of an existing
BF16 application contract. Those failed variants are excluded from accepted
performance comparisons, and the threshold is not relaxed to accommodate them.

## Reference precision

`--reference-precision fp32|fp64` selects the independent analytic backward
oracle, with FP32 remaining the default. Every provider receives the same
FP32 saved mean and reciprocal standard deviation in either mode. FP64 oracle
evaluation promotes those same statistics and the already-quantized input
values; it does not recompute higher-precision saved statistics or change the
device kernels.

Assertions compare each output against the oracle rounded to that output's
dtype. The report also records errors against the unrounded oracle, its dtype,
and the explicit reference-precision option. FP64 changes reference accuracy,
not the tolerance or the tested kernels' arithmetic policy. Historical failed
runs remain associated with the oracle that they used.

A [BF16 midpoint diagnostic](layernorm_ownership_results/diagnose-bf16-f32scratch-seed2-m4096-n4096/results.json)
illustrates why this distinction matters. For one DW column, the FP32 oracle
computed exactly 6.203125, the midpoint between BF16 values 6.1875 and 6.21875,
and rounded down. Both FP64 accumulation of FP32 row products and full FP64
backward evaluation using the common FP32 statistics give approximately
6.2031265 and round up. The original locked kernel, owner, PyTorch, and Quack
all returned 6.21875. The resulting 0.03125 mismatch against the rounded FP32
oracle exceeded 0.01 even though the oracle's unrounded reduction error was
only about 1.5e-6. Rechecking with a documented higher-precision oracle is not
equivalent to widening tolerance. It also does not turn the adopted BF16
tolerance into a universally validated application contract.

The experiment explicitly sets `enable_fp_fusion=False`, matching the ordinary
default in this local Triton checkout. It does not change that local default as
an experimental treatment.

## What the transformation removes

The reduced-grid rewrite removes lock acquisition/release, count checks and
updates, repeated partial-buffer loads/stores, and the required lock/count
initialization launch. It also loads W once per owner rather than once per
input-row program. For
`A = min(M, G)` active groups, the logical partial-store volume changes from
`2*M*N` elements to `2*A*N`; partial loads of `2*(M-A)*N` elements disappear.
This counts source-level memory requests, not DRAM traffic: caches can satisfy
many original requests without accessing DRAM.

These changes come together with a smaller launch grid, sequential row work
inside each owner, and longer-lived register accumulators. They can reduce
available parallelism or increase register pressure. A whole-operation gain
must not be attributed entirely to atomic-CAS contention. Separate stage
profiles and the noncontended control help distinguish these effects.

## Noncontended control and compiler feasibility

When `M <= G`, `r % G = r` over the launch domain. Original and owner stage one
then both have M programs, each processes exactly one row, and each partial
buffer has one writer. No reduction-order change is needed. This is a useful
control for lock/count overhead with matching work distribution and partial
storage traffic.

Even in this case, simply deleting atomics is not justified by race freedom of
the original kernel. A compiler needs the launch bound, exclusive/private lock
ownership, initial zero count flags, and absence of observable synchronization
with external work. If lock/count contents are observable after the kernel,
the transformation must preserve their final values: active counts become one,
inactive counts stay zero, and locks end at zero. The current prototype can
omit those writes because that storage is internal to the operation boundary.

The original stage-one function has no M argument. A compiler needs the launch
domain as well as the ownership and initial-state facts; arbitrary pointer
arguments and `program_id % G` alone do not prove them. The same-grid variant
below obtains the first-axis launch extent through the logical builtin
`tl.num_programs(0)`, without adding an argument. It still relies on the
one-dimensional launch and private, freshly initialized scratch contract.

For `M > G`, the prototype additionally changes the launch from M programs to
G programs and inserts the row loop. Automating that transformation requires
coordinated launch rewriting and proof of row coverage, permitted serialization,
and absence of other cross-program effects. It is a broader transformation than
barrier elimination or an atomic peephole. A candidate implementation should
start with a bounded recognized protocol or explicit ownership/launch metadata.
The existing TMEM/Aref passes do not implement this transformation.

## Same-grid kernel-only prototype

`_layer_norm_bwd_owner_same_grid` keeps the original stage-one argument list,
including B, Lock, and eps, and retains the original M-program launch. It reads
M with `tl.num_programs(0)` and runs the owner loop only when
`tl.program_id(0) < G`. The active programs are exactly `0..min(M,G)-1`;
remaining programs are still launched but perform no writes. This uses valid
logical-program builtins and uniform control flow, not lane- or warp-varying
hardware scalars.

Each active program calls the same scratch-rounding owner body. It then stores
one to its count word and zero to its lock word. With the unchanged caller
reset, inactive counts remain zero and all locks finish at zero, matching the
original post-kernel lock/count state. These final ordinary stores are sufficient
because nobody observes or modifies scratch during the operation. They do not
replace synchronization that an external observer could legitimately require;
such observers are excluded by the private-state precondition.

This prototype retains the original lock/count reset launch and the unchanged
final-reduction source. A speedup therefore cannot be credited to removing the
reset launch. It still combines atomic/synchronization removal with
register-resident partial accumulation and weight reuse, and pays for the
original number of launched programs, including inactive ones. When FP32
partial buffers are selected for BF16 inputs, this statement compares the two
kernels under that common allocation policy; it does not claim that the
unmodified tutorial wrapper already allocates FP32 partial buffers.

Unlike the reduced-grid prototype, this variant is expressible as a kernel-body
rewrite without a new M parameter or modified caller grid. The compiler would
still need a recognized private reduction protocol, nonaliasing/effect proof,
zero-entry count facts, and an accepted serialization interpretation. Arbitrary
nonzero entry counts are unsupported because the original would add into prior
scratch values. A general guarded transformation also needs to establish the
one-dimensional launch domain: additional Y/Z grid dimensions would duplicate
the original row effects and are outside this experiment's contract.

## Compiled smoke audit

The [smoke run](layernorm_ownership_results/smoke-f32-m4096-n1024/results.json)
uses FP32 M4096/N1024/G256, four warps, and is not a tuned performance result.
Its [compiled audit](layernorm_ownership_results/smoke-f32-m4096-n1024/compiled_ownership_audit.json)
establishes:

- Locked stage one contains one static CAS site and two exchange sites; owner
  stage one contains no atomic instructions.
- Owner TTGIR carries DW/DB accumulators through a row loop, loads W before
  the loop, and stores each partial only after the loop. It still writes DX
  for each row. The final reduction is unchanged byte-for-byte through cubin.
- FP32 `owner-round` and `owner-fp32` have byte-identical TTIR, TTGIR, LLVM IR,
  PTX, and cubin for both stages. A timing difference between those two smoke
  variants is measurement variation, not a precision-mode speedup.
- Stage-one registers increase from 48 to 68; both variants report zero local
  memory words and 16 shared-memory bytes. Static `bar.sync` sites change from
  nine to seven. These static counts do not measure dynamic lock retries or
  executed barrier counts through the owner's row loop.

No GPU work or native compiler edits were performed for this protocol audit.

## Same-grid compiled validation

The completed same-grid runs have separate CPU artifact audits for
[FP32 M4096/N1024](layernorm_ownership_results/same-grid-f32-m4096-n1024/compiled_same_grid_audit.json)
and [FP32 M4096/N4096](layernorm_ownership_results/same-grid-f32-m4096-n4096/compiled_same_grid_audit.json).
Each audit verifies all recorded source/artifact hashes and all three seeds at
pre-run, freshly poisoned pre-run, post-timing, and freshly poisoned post-timing
validation points.

At matching eight-warps configurations, the same-grid and reduced-grid owners
have exactly equal DX, DW, and DB at every validation point. Four-warp same-grid
also passes the ordinary numerical and lock/count checks, but the N1024 run
contains no four-warp reduced-grid owner for a corresponding exact-pair claim.

The source argument order and compiled argument types/constexprs match the
original stage one. The original grid and lock/count reset remain. TTGIR obtains
M through `tt.get_num_programs`, guards the complete body with `pid < G`, and
places count/lock stores after the row loop and both final partial stores. PTX
has the corresponding inactive-program exit and `%nctaid.x` extent read, with
no atomic instructions. All compared final-reduction artifacts are byte-identical
through cubin.

This order describes source/IR operations; it does not promise that a count
store becomes visible after every other warp's partial store to an observer
while the kernel executes. The proof only requires post-kernel visibility under
the private-scratch contract. An external polling consumer remains unsupported.

## Boundary validation targets

The following cases exercise different proof obligations. This table describes
their purpose; execution status and numerical outcomes belong to their separate
result records.

| M / N / G | Policy | Obligation exercised |
| --- | --- | --- |
| 1 / 1 / 4 | FP32 | Single scalar row, zero variance, first contribution, final reduction of one active group, inactive count words remain zero |
| 17 / 33 / 64 | FP32 | M below G, masked column tail, non-power-of-two row stride, final reduction ignores uninitialized inactive partial rows |
| 257 / 1000 / 256 | BF16 with FP32 partials | Exactly one group wraps to a second row; the extra original-grid program is inactive, and column-tail masks remain necessary |
| 257 / 1000 / 256 | FP16 with input-precision partials | The same coverage boundary while retaining actual per-update scratch rounding rather than FP32-only accumulators |
| 513 / 4097 / 128 | FP32 | Uneven multiple row iterations, many inactive original-grid programs, and a large padded reduction tile with one valid column beyond 4096 |
