# Semantic optimization findings

Consolidated 2026-11-08. Semantic information can improve Triton kernels when
it establishes a missing dependence, ownership, or value fact and enables a
profitable transformation. Reconstructing the compiler's own buffer protocol
often adds no useful information: releases may already follow the last use,
and smaller buffers can reduce overlap. The results do not establish a
fundamental limitation of Triton or a general benefit from assuming race freedom.

All GPU measurements below used an NVIDIA B300. Speedups compare the tested
variants and configurations; they are not model-level, strongest-provider, or
cross-device claims. Later confirmation runs used shuffled, interleaved CUDA
graphs and common physical output buffers, with unlocked clocks. Compare paired
runs rather than absolute timings across experiments.

| Experiment | Finding | Qualification |
| --- | --- | --- |
| Explicit TMEM release scheduling | 1.01–1.08× recovery | Controlled kernels with deliberately delayed releases. |
| Partial-lifetime TMEM packing | 512→256 columns; 1.047–1.093× on larger grids | No material gain at one CTA wave; restricted one-CTA/two-partition prototype. |
| Aref scheduling on real GEMM/attention | No changed binaries in the tested FP16/FP8 GEMMs or BF16 attention | Relevant releases already followed their last accesses; more recognition alone creates no scheduling slack. |
| Attention single buffering | Best baseline regressed 130.87→198.65 μs | A 1.205× gain on a slow fixed configuration disappeared after baseline tuning. Resource/overlap tradeoff; mechanism not isolated. |
| Faster attention implementations | Tritonbench 143.47 μs; Gluon 105.28; FA4 90.11 | Matched BF16 B2/H8/N4096/D128. The whole gap was not attributed to one semantic property. |
| Identity accumulator rescale elision | FA4 1.096×; manual Gluon port 1.050× | Same threshold-eight policy and exactly equal paired outputs/LSE. Threshold-zero elision was slightly slower; threshold-eight failed scale-four accuracy stress checks. |
| Typed score/probability storage reuse | D64 reservation 512→256; runtime 94.12→94.09 μs | Capacity rather than throughput benefit: another tile fits 512 columns instead of requiring 640. |
| LayerNorm backward, native private ownership | BF16 2.05×/1.59× at 4096×1024/4096; FP32 1.77×/1.42× | Matched FP32 scratch, original grid/reset/final reduction, tuned locked baseline and fixed candidate configurations. Torch native is faster at width 1024. Exact permitted-serialization checks pass; separate BF16 mathematical accuracy limits remain. |
| Shared global-memory analysis and redundant-load reuse | LayerNorm weight loads 2→1; main cases within about 1%; spilling boundary 1.061–1.073× | 15 paired on/off records with identical source/contracts/ownership TTIR. FP32 width4096 slightly regresses. Reuses the same analysis as LICM; no broad new speedup claim. |
| Persistent AITER RMSNorm, native immutable-input contract | Forward 1.11–1.48×; backward 1.05–1.06× | Large-shape regression now profiled: occupancy falls while DRAM traffic barely changes. Resource guard declines measured losses, but also misses a held-out 17% gain. AMD-origin kernels compiled on NVIDIA. |
| FlagGems CTC, native immutable input across CTA barriers | Complete training-forward 1.021–1.024× at T1024/N64/U128; 1.047–1.053× at T256/N256/U64 | 139 source-study records and 51 rebuilt-native confirmations. Full saved alpha and loss checked; barriers retained. Most other cases gain 0–3%. Backward is excluded. |
| AITER recurrent delta rule, native alias-derived LICM | About 1.134× at upstream T1024/D128; 1.09× at larger decode; 1.056× at T500/D60 | Three of six cases improve; three are neutral/slower. Includes state reset, every timestep output and final state. New pointer provenance makes alias-only and immutable PTX identical. Supported long-T tests do not establish production prevalence. |
| Transformer Engine MoE unpermute backward | Native hoist 1.19×, 1.11×, approximately 1.00× | Best tested configuration per variant after bounded tuning; manual and contract variants matched machine code. |
| Producer-unique embedding backward | Source specialization 1.16–1.26×; frequency-scaled 1.41–2.18× | Includes dense zeroing and final copy. These historical timings used finite, non-subnormal BF16 gradients and predate the automatic proof pipeline below. |
| Unchanged FlagGems indexed updates, native Triton pass | Unique RMW 1.15–1.25×; initialized-zero 1.15–1.28× | Correct native transformation, but the ordering control below recovers essentially all of this gain while retaining atomics. Reset/count/conversion included. Gluon correctness also validated. |
| Embedding atomic-ordering control | Original→relaxed atomics 1.186× geometric mean; relaxed→native RMW 0.996×; relaxed→native zero 1.006× | 36 paired measurements. Changing only `sem="relaxed"` removes two CTA barriers and permits no-return reductions. Earlier attribution to uniqueness was too strong. |
| Native private-output ordering relaxation | 1.192× geometric mean with unique indices; 1.179× with fourfold repeated indices | 72 measurements; atomics retained, unchanged source, native/manual relaxed timings within 0.3%. Includes reset/count/conversion; no uniqueness or immutable-index contracts. |
| Four-way atomic ordering control | Original→relaxed 1.189×; original→acquire 1.150×; release→relaxed 1.070× | 36 paired cases. Acquire still uses `atom` and beats release's `red`; release also emits a GPU fence. Pure CTA-barrier cost remains unisolated. |
| Persistent FlagGems integer frequency counter | Uniform indices 1.54–1.76×; hot-32 indices 1.003–1.34× | 114 measurements, best tested block/grid per variant including nonpersistent launches. Native loop relaxation matches manual relaxed within 0.03% geometrically. Reset+count only; Cambricon-origin body on B300. |
| Segmented-unique FlagGems scatter-add | Ordinary RMW 0.65–0.91× consumer ratio; known-zero stores about 1.00–1.03× | 96 screening and 26 confirmation cases. With topk producer included, known-zero gains stay below 0.5% on tuned shapes. Baseline already uses relaxed atomics; native expansion was not implemented. |
| Complete coverage and direct output writes | 1.40–2.30× over the previous unique-index implementation | Shared-output consumer ablation, including required initialization/conversion; bounded tuning over block sizes 128/256. Full permutations need one write launch; unique subsets need zeroing plus write. |
| Real embedding backward vs Inductor, frequency scaling enabled | 1.69–14.61× GPU speedup over the faster tested mode | Output-only graphs include arange/randperm generation. Inductor default and max-autotune-no-cudagraphs retain atomic-update/count work. External CUDA graphs give all variants the same timing protocol. |
| Same Inductor comparison, frequency scaling disabled | 0.58–1.05× GPU ratio; no general win | Inductor is often faster. Eliminating an atomic alone does not establish a profitable optimization; layout, fusion, launch count, and host overhead still matter. |
| Inductor internal CUDA-graph replay, frequency scaling enabled | No ordinary-call win: semantic FX takes 114–182 μs vs Inductor 104–173 μs | Three full-coverage producer graphs, three seeds; replay confirmed active. The current Python interpreter prevents the device savings from becoming an end-to-end improvement against this stronger baseline. |
| Mamba affine scan | 0.87–0.93× at representative state dimensions 4096/8192 | Gains of roughly 5–11× occurred only in small-state, low-parallelism diagnostics. Expanded tuning remained slower; some tiles exhausted shared memory. Reassociation required an explicit numerical allowance. |
| Uniform segment reduction | Cached host dispatch/discovery fell about 31.3→24.5 and 34.7→26.4 μs | Upstream already chose the specialized GPU kernel; the generic/uniform device ratio is not a new optimization. |
| TorchRec TBE immutable metadata | No improvement; identical PTX/executable code | Tested upstream and persistent grids. Immutable data does not make iteration-varying addresses invariant. |

The LayerNorm rewrite assigns each partial buffer and its contributing rows to
one owner, retaining two partial vectors in registers and removing locks and
intermediate partial traffic. The reconstructed harness retains the exact original
Tritonbench kernel AST, launch grid, reset and final reduction. It checks every
partial, DX, final gradient, count and lock against an independently evaluated
permitted original serialization, including first assignment and original
new-plus-old order. Fresh source validation covers 96 tuning configurations,
21 confirmations and 15 boundaries; source BF16 gains reproduce at 2.02×/1.83×.
The subsequent native study has 30 records and preserves that serialization.
Some BF16 cases fail the separate historical
absolute-0.01 FP64 mathematical diagnostic in both versions; none was dropped or
given a wider tolerance. FP32 cases pass. Program refinement does not establish
universal training accuracy. Historical timings remain separately identified.

Attention identity elision assumes finite scores/accumulators; changing the
softmax shift policy is a separate transformation. Storage reuse additionally
needs physical-region, generation, initialization, and asynchronous
operand-completion proofs. Ordering accesses
alone does not preserve a value overwritten before its final read.

The implemented [semantic-contract frontend](../python/triton/experimental/semantic_contracts/backend.py)
now connects producer facts to optimization:

- It derives index uniqueness from producers such as `randperm`, propagates it
  through supported transformations, and invalidates it on aliasing writes or
  unknown effects. It never infers uniqueness from example tensor contents.
- Fresh output allocations establish alias groups; native LICM derives input
  read invariance from memory effects without a manually asserted immutable flag.
- Uniqueness plus bounds and cardinality proves complete destination coverage.
  The unique writer removes FP32 scratch/conversion, and complete coverage removes
  initialization: **one launch for full permutations, two for unique subsets**.
  Padding owners explicitly write zero. Unknown or mutated indices retain the
  original fallback. Floating-point edge cases are covered.
- The same proof now specializes real `aten.embedding_dense_backward`, including
  graphs captured from `torch.func.grad(F.embedding)`. It preserves PyTorch's
  non-flushing arithmetic separately from the owned operator's atomic arithmetic.
  Unsupported shapes/layouts/devices or unaudited PyTorch revisions fall back to
  the original framework operator.
- RMSNorm calibration selected hoisting at 4096×1024 (4.72→3.51 μs) and baseline
  at 31×137 and 16384×4096. These are first-use variant-calibration timings.
  That FX implementation passed **198 runtime tests** and **78 seeded benchmark
  measurements** at its recorded revision. Its native validation passed 14
  compiler tests; the subsequent native phase below adds compiler code and tests.

This remains a fixed-metadata FX prototype with two owned operators and a bounded
real ATen consumer on NVIDIA CUDA. It has no autograd formulas for the owned ops
or general `torch.compile`/torch-mlir integration. General attention transformations remain unimplemented. A separate bounded
native LayerNorm ownership proof is now implemented, as described below. Propagating
uniqueness into Inductor before embedding decomposition could remove frequency
counting and avoid a separate FX interpreter. The following experiment instead
tests the compiler opportunity directly in Triton/Gluon. No TLX CLC or multi-CTA
work-queue experiment was run.

The new [native indexed-update pass](../lib/Dialect/Triton/Transforms/SpecializeIndexedUpdates.cpp)
consumes argument contracts through a [guarded launch interface](../python/triton/experimental/semantic_contracts/native.py).
The caller supplies an immutable, globally unique bounded index prefix, disjoint
private initialized output, and exact launch dimensions. The compiler derives
destination injectivity from the actual address expression and masks, checks
integer overflow and memory effects, and replaces eligible f32 atomics with
masked load/add/store. Optional positive-zero initialization removes the load.
Explicit FTZ addition preserves the existing atomic arithmetic, including signed
zero and subnormal behavior; NaN payloads are not guaranteed. Kernel source bodies
remain unchanged. This is an opt-in CUDA pass, also connected to Gluon lowering.

The native experiment establishes uniqueness using a `randperm` producer; the
low-level interface takes explicit caller contracts rather than automatically
discovering that producer. It supports one CTA per logical program and a bounded
scalar-index row-update pattern. Unsupported IR or missing contracts retains
atomics. Runtime guards check the bound tensor, tracked mutations, metadata,
storage overlap and launch geometry; raw device writes and the initialization
promise remain caller obligations. Duplicate-index embedding workloads do not
qualify. Counting kernels, loops, arbitrary scatter expressions, and launch
elimination are outside this pass.

The original six shape/mode groups gained **1.185× geometrically averaged** from
unique RMW and **1.195×** with initialized zero, matching manual rewrites within
1%. A subsequent 36-case control changed only the original atomic's ordering to
`relaxed`, retaining atomics and all reset/count/conversion work. That recovered
**1.186×** on its own. Relative to this stronger baseline, native RMW measured
**0.996×** and native zero **1.006×**. Thus the original result establishes legality
of atomic removal, but does not establish meaningful performance benefit from
uniqueness. The ordering change removes two block barriers and enables no-return
`red` instructions. A subsequent four-way control measured original→acquire
**1.150×**, original→release **1.112×**, and original→relaxed **1.189×** over 36
matched cases. Acquire retains `atom` yet beats release's `red`; SASS shows release
also adds `MEMBAR.ALL.GPU`. Thus instruction family alone does not explain the
gain, and these controls still do not isolate pure CTA-barrier latency. The larger
historical frequency-scaled gains also removed counting work.

The second native candidate was unchanged FlagGems `scatter_add_2d_kernel`, using
row-wise topk/sort to establish uniqueness within each row while allowing repeated
index values across rows. Manual load/add/store regressed roughly **10–55%** at
the four tuned shapes. Known-zero stores gave at most **3%** consumer improvement
and less than **0.5%** with topk included. Both variants included initialization;
BLOCK 32/64/128/256 and LOOP 1/2/4 were screened. Exact outputs, unwritten positions,
changed inputs/producers, and an additional full-width sort control passed.
This baseline already uses relaxed atomics. The performance gate failed, so no
segmented compiler analysis or public API was added.

That weaker ordering proof is now implemented as a separate native
`triton-relax-private-atomics` pass, exposed through
[`compile_private_atomic`](../python/triton/experimental/semantic_contracts/native.py).
The caller promises initialized output and no access from outside the invocation.
The compiler derives the absence of an observable synchronization role: ordinary
reads are disjoint from its private output, atomic old values are unused, and no
other writes, calls, or unknown effects are allowed. It keeps the atomic and changes
only its ordering. No uniqueness, immutable-index, bound, grid, or zero facts are
required by the native proof. Input contents and objects may change between
guarded launches. Both native passes now reject private/called helper functions,
whose callers could introduce effects outside the inspected body.

On unchanged plain/scaled FlagGems kernels, 72 measurements show **1.15–1.26×**
speedups: geometric means **1.192×** for unique indices and **1.179×** for shuffled
fourfold repeats. Native and manual relaxed variants match within **0.3%**. All
retain atomics; the transformed TTIR body changes only ordering, and representative
PTX replaces two barriers plus `atom.acq_rel` with barrier-free `red.relaxed`.
Duplicate correctness uses exact small-integer contributions; ordinary BF16
gradient checks use a derived FP32 summation-error bound against FP64 references,
allowing valid atomic serialization differences. Fourfold repeats are a controlled
collision workload, not a claim about every training-index distribution.

The new extension accepts multiple **i32 additions** to one private output,
including nested `scf.for` loops without carried state. Addition modulo 2^32
preserves the final result independently of ordering. Floating-point loops and
multiple sites remain unsupported: acquire/release ordering can constrain their
observable rounding outcomes across locations. Dynamic loop bounds, zero trips,
collisions, masks, overflow, and whole-function rejection are tested in Triton and
Gluon. Ordinary input accesses must be defined, as in the original program; no
immutability between invocations is required.

On the unchanged FlagGems persistent frequency counter, **114 measurements** across
three sizes, two collision distributions, block sizes 256/512 and persistent/full
grids give best-tested speedups **1.76×, 1.54×, 1.61×** for uniform indices. With
only 32 hot destinations they fall to **1.34×, 1.05×, 1.003×**: contention limits
the opportunity. Native/manual relaxed timings agree within **0.03%** geometrically.
These include output reset and counting only, not complete embedding backward.
The Cambricon-origin source body was compiled directly on B300; this is not a
comparison against FlagGems' NVIDIA dispatch or a tuned histogram library.

The RMSNorm profitability follow-up covers **22 configurations × three seeds**,
plus eight Nsight Compute profiles. At 16384×4096/grid4736, hoisting changes
achieved occupancy **40.51%→29.04%** and eligible warps/scheduler **0.65→0.38**,
while DRAM reads remain approximately **134.23 MB**. Repeated weights were already
cached. Grid592 avoids the occupancy loss and the regression. A predeclared guard
declines spills or reduced residency unless all launched CTAs fit the optimized
resident capacity. It rejects all held-out >2% losses but also declines a 17% gain.
[`compile_immutable_licm`](../python/triton/experimental/semantic_contracts/native.py)
implements this heuristic after compiling both variants, where actual resource
use is available. It is separate from FX timing calibration, requires explicit
immutable-input contracts, and cannot guarantee profitability.

The pinned FlagGems/Liger/AITER/Transformer Engine AST survey finds **311**
default-order, syntactically unused atomic-add sites (**274** after exact-body
deduplication), including **75** inside loops. It also finds **250** invariant
load/loop pairs with a store in the loop (**233** after deduplication), checking
address, mask and `other`, not just pointer roots. These are source candidates,
including platform-specific code and helpers, rather than proven legal or
profitable transformations. The Liger general-path scalar-load miss is not a
production plain-backward opportunity: its fast plain path already hoists it.

Two analysis gaps are now closed in native Triton LICM. Explicit immutable global
inputs may move across existing CTA barriers, retaining masks, entry guards and
all barriers. This recovers the manual CTC hoist on unchanged training-forward
source. Alias-only contracts still conservatively reject barrier-bearing loops.
Pointer provenance now follows bounded `scf.for` backedges derived from their own
carried pointer, including nested loops. This proves AITER's ordinary stores
cannot clobber its invariant parameters: alias-only and immutable binaries match
exactly on all six post-build cases. Advancing addresses remain variant;
root-changing/cross-carried pointers and unknown operations are rejected.

AITER also exposes the current profitability limits. The residency heuristic
accepts all three negative cases. At T500, both native and manual code hoist the
loads and dependent negative exponential, but manual gains about 14% versus
native's 6%. Masked inactive values and downstream IR scheduling differ; this is
not a missing dependent-expression hoist. The long-T workloads are exact upstream
test shapes; a malformed legacy decode benchmark branch was not used as an
equivalent baseline, and chunked prefill remains a stronger algorithmic candidate.

The new opt-in [private-reduction ownership pass](../lib/Dialect/Triton/Transforms/PrivateReductionOwnership.cpp)
recognizes IR roles rather than kernel names. Private initialized disjoint outputs,
zeroed locks/counts and bound scalar/grid facts permit it to prove disjoint rows
and assign owners to a two-FP32-partial lock/count protocol. It preserves the
original grid and one correlated per-group serialization, with no floating-point
reassociation license. Missing facts, overflow, observers and unsupported IR keep
the original kernel. The [guarded wrapper](../python/triton/experimental/semantic_contracts/native.py)
checks metadata, scalar values, storage overlap and launch geometry; initialization
and privacy remain caller promises. This is a bounded prototype, not a general
reduction or profitability analysis. Normal scalar arguments retain their original
specialization policy; exact-value facts are used for proof only. Native backward
measures 49.69→24.19 μs and 80.22→50.39 μs on BF16 widths 1024/4096. The manual
owner remained faster on some configurations: that native build reloaded an
invariant weight vector after the peeled first row and kept pointer setup live longer.
At the irregular 513×4097 boundary, native regresses 32.99→35.90 μs with spills
rising from 2 to 170. The load-reuse follow-up below removes the duplicated weight
read. A profitability guard for ownership remains unimplemented.

The shared [GlobalMemoryEffects analysis](../include/triton/Analysis/GlobalMemoryEffects.h)
now centralizes function-scoped argument provenance, alias groups, immutable
facts and ordinary memory-effect summaries. LICM uses it with its existing
legality checks. A second consumer in [LoopAwareCSE](../lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp)
reuses earlier global loads across disjoint stores within the same block. All
pointer, type/layout, property and operand checks remain exact, except a bounded
AND-mask implication when both loads omit `other`. This handles LICM's added
loop-entry mask without speculating reads or changing defined inactive values.
The analysis does not infer initialization, privacy or freedom to remove waits;
unknown effects stay conservative, and summaries have explicit invalidation rules.

The new consumer runs through existing Triton and Gluon CSE pipelines. Gluon
still does not run Triton LICM, and native ownership remains Triton CUDA-only.
`enable_contract_load_reuse=False` disables the new consumer for controlled
comparisons while retaining contracts and ownership rewriting. Fifteen complete
LayerNorm backward on/off records preserve bitwise selected-serialization, state
and guard checks. All remove one of two weight loads, but main-case timings stay
within about 1%; FP32 width4096 is slightly slower. The irregular spilling case
improves 6.1–7.3%, with spills 170→164. Fewer loads are not a general profitability
proof. No new TLX or torch-mlir integration was added; allocation/host guards remain
outside these GPU timings.

Current [shared-analysis validation](semantic_candidate_experiments/results/shared_memory_analysis_final/validation.json)
passed `make`, **308 runtime tests** and **14 compiler test files**, adding 26
runtime and 29 compiler cases. The [previous native validation](native_extensions_validation_11_08.json)
and its performance measurements retain their original source/library fingerprints.

Retained evidence and reproducers:

- November 4/7 historical records: [controlled compiler work](summary_11_04.json),
  [real-kernel compiler work](summary_11_07.json),
  [attention](semantic_attention_summary_11_07.json), and
  [LayerNorm](layernorm_ownership_summary_11_07.json). Their original harnesses
  and raw artifact trees were already absent before this cleanup; embedded
  paths are historical provenance, not working reproduction links.
- [Candidate benchmark summary](semantic_candidate_experiments/results/summary.json)
  and its canonical analysis/timing reports preserve measured configurations,
  negative results, numerical limits, and source/compiler fingerprints.
- [Final proof-pipeline evidence](semantic_candidate_experiments/results/proof_pipeline_final/summary.json)
  retains the earlier implementation's code, snapshots, and validation logs.
  [Coverage and framework-consumer results](semantic_candidate_experiments/results/coverage_final/summary.json)
  record the subsequent direct-write optimization, full-producer timings, and
  ordinary-call overhead. The Python interpreter erases GPU gains in several
  small/medium cases; device timings are not ordinary-call or model speedups.
  [Inductor controls](semantic_candidate_experiments/results/coverage_inductor_final/results.json)
  compare output-only graphs, with 24 successful compiler builds and 36 seeded
  measurements, and retain bounded generated-code excerpts. The subsequent
  [internal-replay control](semantic_candidate_experiments/results/coverage_inductor_final/replay_control.json)
  supersedes ordinary-call wins against modes without internal replay.
  Historical speedups must not be attributed to a later implementation.
- [Native indexed-update evidence](semantic_candidate_experiments/results/native_indexed_final/summary.json)
  records the compiler proof, validation and six aggregate comparisons;
  [raw results](semantic_candidate_experiments/results/native_indexed_final/results.json)
  retain all 36 seeded measurements, source/compiler hashes and upstream AST
  audits. Representative TTIR/TTGIR/PTX and test logs are retained alongside them.
- [Ordering-control evidence](semantic_candidate_experiments/results/native_atomic_ordering_final/results.json)
  and its [instruction analysis](semantic_candidate_experiments/results/native_atomic_ordering_final/synchronization_analysis.json)
  supersede the earlier attribution of the native speedup to uniqueness.
  [Segmented scatter evidence](semantic_candidate_experiments/results/segmented_final/summary.json)
  retains the negative result, tuning screens, producer-inclusive confirmation,
  pinned source, and compact reproduction artifacts.
- [Native private-atomic evidence](semantic_candidate_experiments/results/native_private_atomic_final/summary.json)
  retains 72 paired measurements, proof scope, numerical checks, source/compiler
  hashes, representative IR/PTX and build/test logs. Earlier records retain their
  original compiler fingerprints and are historical measurements.
- [Four-way ordering control](semantic_candidate_experiments/results/atomic_fourway_final/README.md),
  [integer loop extension](semantic_candidate_experiments/results/private_atomic_loop_final/summary.json),
  [LICM profitability](semantic_candidate_experiments/results/licm_profitability_final/findings.json),
  and [pinned prevalence survey](semantic_candidate_experiments/results/semantic_survey_final/summary.json)
  retain current measurements, profiler evidence, compiled follow-ups and
  reproduction commands. Source counts are not measured optimization coverage.
- [CTC source study](semantic_candidate_experiments/results/ctc_licm_final/findings.json)
  and [native confirmation](semantic_candidate_experiments/results/ctc_licm_native_final/findings.json)
  retain complete training-forward controls and immutable/barrier legality checks.
  [AITER recurrent evidence](semantic_candidate_experiments/results/recurrent_licm_final/README.md)
  separates frozen immutable results from rebuilt alias-only validation.
  [Reconstructed LayerNorm evidence](semantic_candidate_experiments/results/layernorm_ownership_final/README.md)
  retains the exact original source, independent serialization oracle, numerical
  limitations, fresh tuning and separate native follow-up.
- [Shared-analysis load-reuse control](semantic_candidate_experiments/results/shared_memory_analysis_final/layernorm/README.md)
  retains paired feature-on/off measurements, exact program/state checks, resource
  changes and reproduction commands. The first application establishes a second
  consumer of reusable memory facts; its small main-case changes are not a new
  general LayerNorm speedup.
- [Experiment scripts](semantic_candidate_experiments/) and pinned `vendor/`
  sources/licenses remain. RMSNorm, MoE, irregular-kernel, and Mamba harnesses
  expose `--help`; analyses can consume newly generated reports.

From the Triton checkout with the local source build installed, rerun the current
integration examples into an unused output directory:

```sh
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/proof_pipeline_experiment.py --output /tmp/triton-proof-rerun
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/native_indexed_update_experiment.py --mode confirm --output /tmp/triton-native-indexed-rerun
```

Earlier cleanup removed overlapping prose, superseded runs, historical generated-code
copies, and caches. Selected raw timing reports remain to audit aggregate
statistics; their old assembly paths may point to pruned files. See
[cleanup.json](cleanup.json) for the retention policy and size accounting.
