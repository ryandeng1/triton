# Semantic-property source survey

Pinned source patterns are common enough to justify targeted work, but they do not establish optimization applicability. Among 766 physical Triton/Gluon `atomic_add` sites, 342 omit `sem`; 311 also discard the result. Deduplicating exact function bodies reduces that last count to 274. Most hits are in FlagGems, including backend forks. AITER and Transformer Engine already use explicit relaxed ordering at most atomic sites.

| Project | Python files | JIT definitions, raw / exact bodies | Atomic sites, raw / exact | Default + unused, raw / exact | Of those, inside source loops | LICM source pairs, raw / exact |
|---|---:|---:|---:|---:|---:|---:|
| FlagGems | 4,111 | 11,280 / 8,629 | 686 / 608 | 269 / 235 | 58 | 183 / 166 |
| Liger | 286 | 200 / 196 | 37 / 34 | 34 / 31 | 16 | 11 / 11 |
| AITER | 481 | 1,376 / 1,305 | 30 / 30 | 4 / 4 | 0 | 55 / 55 |
| Transformer Engine | 232 | 34 / 34 | 13 / 13 | 4 / 4 | 1 | 1 / 1 |

The scope is production-package Python at the roots in [provenance.json](provenance.json), including JIT helpers, backend forks and vendored code. FlagGems, AITER and TE reuse the previous experiment pins; Liger uses default-branch HEAD queried on 2026-11-08. This is not a synchronized latest-release census. No file failed to parse. Six cuTile/CuTe atomic calls are excluded.

The raw 311 default-unused sites include 91 generic and 7 NVIDIA-backend FlagGems sites; 171 FlagGems sites are in other backend directories. Liger has 11 generic and 23 Ascend sites. AITER's four include one Gluon timeout-error counter. Backend location is not a portability or performance guarantee. There are 75 directly loop-nested default-unused sites and 138 in functions with multiple direct atomic sites; those subsets overlap. Another 58 same-module calls to atomic helpers, 38 in loops, are audited separately without multiplying primitive counts.

The LICM screen finds 250 load/loop pairs across 137 source functions, or 233 pairs across 123 exact bodies. It checks transitive invariance of the actual address, mask and `other`, not just the pointer argument. Among 8,749 load/loop pairs in loops with a direct store, only 259 pass that operand check; six volatile/unknown-volatile cases and three without argument dependencies are excluded. An invariant-address read of mutable scratch still passes the source screen, so these 250 pairs are candidates for memory-effect and contract review, not 250 safe hoists. Source static loops may disappear after specialization.

## Compiled followup

Three unchanged upstream bodies were compiled for explicit CUDA sm90 specializations. No GPU was initialized or launched. [The compile report](compiled_shortlist/summary.json) records source/body/compiler hashes, specializations, contracts and complete representative TTGIR.

| Kernel / selected load | Baseline inside loop | Immutable contract | Alias-region contract |
|---|---:|---|---|
| Liger cross-entropy backward, scalar upstream gradient | 1 | Hoists with loop-entry mask | Hoists with loop-entry mask |
| FlagGems CTC forward, previous target label | 1 | Retained | Retained |
| AITER recurrent delta-rule update, `A_log` and `dt_bias` | 2 | Both hoist with loop-entry mask | Both retained |

CTC retains its original `ttg.barrier` in the timestep loop. Source inspection of `hasOnlyOrdinaryMemoryEffects` explains the rejection: barriers fall outside the currently accepted effects. AITER's alias-only case has loop-carried read/write pointers, which `getEntryArgument` deliberately does not trace; the whole-function effect summary therefore cannot establish no-clobber. These explanations follow the saved IR and current pass guards; no rejection diagnostic was emitted.

Thus the survey followup establishes three compiled kernels with four repeated invariant-load sites, and successful contract lowering in two kernels. It does **not** establish runtime caller contracts, correctness, or speedups for these new cases. Previous RMSNorm and TE-permutation experiments are identified at matching pins in [shortlist.json](shortlist.json), but their timings are not extrapolated to the survey.

## Atomic shortlist

The strongest first extension is FlagGems' Cambricon persistent embedding frequency-count kernel: a strided loop, an ordinary index load, and unused-result `atomic_add(counts + idx, 1)`. Its host creates fresh zeroed **int32** counts. [The complete unchanged function and host allocation evidence](evidence/flaggems_persistent_count_provenance.json) retain the Apache header, revision and AST/source hashes. Integer modular addition avoids the cross-location floating rounding correlations that complicate multiple floating atomics. The parent experiment owns its native compilation, correctness and timing evidence.

Liger's Ascend MoE router histogram is a second real int32 counter loop, with a scalar validity branch; its host also creates a fresh zeroed int32 count tensor. Nearby rank-allocation atomics consume the old value and are excluded. Other apparent counters are float32, communication/error flags or counters read back within the same invocation; [the reviewed shortlist and rejections](shortlist.json) explain these distinctions with pinned source links and small snippets.

## Reproduce and interpret

From the Triton repository:

```sh
python3 experiment_notes/semantic_candidate_experiments/semantic_prevalence_survey.py --fetch
/home/ryandeng/tpaa/.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/semantic_prevalence_survey.py --compile-shortlist-only
```

The first command downloads only production Python into `/tmp/semantic-survey-cache`, then performs the standard-library-only scan. The second requires the local experimental compiler and performs CPU compilation only. [Atomic sites](atomic_sites.json), [LICM candidates](licm_candidates.json), [the audit](scan_audit.json) and [summary](summary.json) contain compact machine-readable evidence. The small hand-reviewed shortlist and count extraction are retained separately.

Alias resolution covers imported Triton/Gluon module aliases, direct import aliases, simple function aliases and inferred tensor-member atomics; all primitive hits in this corpus resolved to module calls. The scanner distinguishes omitted, explicit `None`, explicit ordering and forwarded semantics, although the selected revisions contain only omitted or explicit strings. Return use is syntactic, not compiler SSA liveness; destination dtype is unknown unless separately verified in host code. Same-module wrapper calls are recorded; dynamic imports, transitive wrapper expansion, generated specializations and descriptor memory APIs are outside the scan.

Deduplication compares exact AST body+signature after stripping function names, decorators and docstrings and normalizing imported Triton aliases. It removes exact backend copies, not all semantically equivalent kernels or renamed clones. Counts measure checked-in source, never runtime prevalence or time saved.
