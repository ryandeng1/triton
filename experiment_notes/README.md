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
| LayerNorm backward owner-computes rewrite | BF16: about 1.9–2.0× at 4096×1024; 1.78× at 4096×4096 | Source prototype against the tuned tutorial-derived registered Tritonbench provider, with FP32 partials on both sides. Other providers sometimes match or beat it. FP32 gains were 1.88×/1.40×; FP16 1151×8192 gained 1.83×. |
| Persistent AITER RMSNorm, native immutable-input contract | Forward 1.11–1.48×; backward 1.05–1.06× | Forward 16384×4096 regressed about 9%. Registers increased 68→94; occupancy causality was not profiled. AMD-origin kernels were compiled on NVIDIA. |
| Transformer Engine MoE unpermute backward | Native hoist 1.19×, 1.11×, approximately 1.00× | Best tested configuration per variant after bounded tuning; manual and contract variants matched machine code. |
| Producer-unique embedding backward | Source specialization 1.16–1.26×; frequency-scaled 1.41–2.18× | Includes dense zeroing and final copy. These historical timings used finite, non-subnormal BF16 gradients and predate the automatic proof pipeline below. |
| Unchanged FlagGems indexed updates, native Triton pass | Unique RMW 1.15–1.25×; initialized-zero 1.15–1.28× | 36 seeded measurements on three shapes, plain/scaled modes, and block sizes 128/256. Reset, original frequency counting, and BF16 conversion retained; producer generation/allocations excluded. Gluon correctness also validated. |
| Complete coverage and direct output writes | 1.40–2.30× over the previous unique-index implementation | Shared-output consumer ablation, including required initialization/conversion; bounded tuning over block sizes 128/256. Full permutations need one write launch; unique subsets need zeroing plus write. |
| Real embedding backward vs Inductor, frequency scaling enabled | 1.69–14.61× GPU speedup over the faster tested mode | Output-only graphs include arange/randperm generation. Inductor default and max-autotune-no-cudagraphs retain atomic-update/count work. External CUDA graphs give all variants the same timing protocol. |
| Same Inductor comparison, frequency scaling disabled | 0.58–1.05× GPU ratio; no general win | Inductor is often faster. Eliminating an atomic alone does not establish a profitable optimization; layout, fusion, launch count, and host overhead still matter. |
| Inductor internal CUDA-graph replay, frequency scaling enabled | No ordinary-call win: semantic FX takes 114–182 μs vs Inductor 104–173 μs | Three full-coverage producer graphs, three seeds; replay confirmed active. The current Python interpreter prevents the device savings from becoming an end-to-end improvement against this stronger baseline. |
| Mamba affine scan | 0.87–0.93× at representative state dimensions 4096/8192 | Gains of roughly 5–11× occurred only in small-state, low-parallelism diagnostics. Expanded tuning remained slower; some tiles exhausted shared memory. Reassociation required an explicit numerical allowance. |
| Uniform segment reduction | Cached host dispatch/discovery fell about 31.3→24.5 and 34.7→26.4 μs | Upstream already chose the specialized GPU kernel; the generic/uniform device ratio is not a new optimization. |
| TorchRec TBE immutable metadata | No improvement; identical PTX/executable code | Tested upstream and persistent grids. Immutable data does not make iteration-varying addresses invariant. |

The LayerNorm rewrite assigns each partial buffer and its contributing rows to
one owner, retaining partial sums in registers and removing locks and intermediate
partial traffic. The headline variant retains the original launch grid, reset,
and final reduction. Its proof needs complete row coverage, disjoint buffers,
unchanged inputs, private initialized scratch, no intermediate observers, the
supported launch shape, and preserved final state. It selects a permitted
serialization; it does not authorize arbitrary floating-point reassociation.
BF16 scratch precision is a separate issue: FP32 scratch was applied to both
sides. An FP32 reference rounding error was corrected using FP64/direct rounding
without relaxing tolerances; one original configuration still failed and was
excluded. These checks do not establish a universal training-accuracy contract.

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
or general `torch.compile`/torch-mlir integration. Automatic LayerNorm ownership
rewriting and general attention transformations remain unimplemented. Propagating
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

Against the best tested block size per variant, the six shape/mode groups gain
**1.185× geometrically averaged** from unique RMW and **1.195×** with initialized
zero. Native results match manual rewrites within 1%; zero knowledge adds only
0.09–2.36% beyond unique RMW. This isolates a useful missing semantic fact in the
native compiler. The larger earlier frequency-scaled gains also removed counting
work, which this experiment deliberately retains. These are device timings, not
ordinary-call or model-level speedups. Current validation passes **215 runtime
tests**, including 17 new native GPU cases, and **14 compiler test files**,
including 32 new proof/negative cases. A next native extension is proving vector
index uniqueness in counting kernels; producer-fact attachment and profitability
still need broader integration.

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
- [Experiment scripts](semantic_candidate_experiments/) and pinned `vendor/`
  sources/licenses remain. RMSNorm, MoE, irregular-kernel, and Mamba harnesses
  expose `--help`; analyses can consume newly generated reports.

From the Triton checkout with the local source build installed, rerun the current
integration examples into an unused output directory:

```sh
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/proof_pipeline_experiment.py --output /tmp/triton-proof-rerun
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/native_indexed_update_experiment.py --mode confirm --output /tmp/triton-native-indexed-rerun
```

Cleanup removed overlapping prose, superseded runs, historical generated-code
copies, and caches. Selected raw timing reports remain to audit aggregate
statistics; their old assembly paths may point to pruned files. See
[cleanup.json](cleanup.json) for the retention policy and size accounting.
