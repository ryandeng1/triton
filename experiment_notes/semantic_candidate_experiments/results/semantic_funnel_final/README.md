# FlagGems semantic funnel: inventory and one measured pilot

Pinned FlagGems `3f1f896fd90f5db42d167d64fcb9691326071bc2`. This closes the source inventory and demonstrates one independent family; the population legality/profitability survey remains open.

| Stage | Count / scope |
|---|---|
| Default-order unused atomic-add source sites |91 sites, 58 functions/58 exact bodies; 25 sites in loops |
| LICM source candidates |62 load-loop pairs, 60 loads, 35 functions/34 exact bodies |
| Combined source functions |93 |
| Static host paths |92/93 |
| Static ATen registration references |68/93; runtime backend dispatch unverified |
| Direct repository test references |75/93 |
| Direct repository benchmark references |45/93 |
| Reviewed contract / compiled source functions |1:unfold_backward; 92 unreviewed |
| JIT-specialized pilot |18 specializations, 54 compiled variants, 21 shape/dtype cases |
| Generic-signature control |3 specializations, 9 compiled variants, 21 shape/dtype cases |

Counts are different units: sites and load-loop pairs are not independent kernels. Exact-body dedup comes from the original survey. The [manifest](manifest.json.gz) retains every site/pair, caller edge, source hash, test/benchmark reference, registration and unresolved status; [functions.tsv](functions.tsv) is the readable function index. Missing static links, unreviewed contracts and unattempted compilation are not legality rejections.

The source graph resolves lexical helpers, class methods, known autograd apply-to-forward calls and warp-specialize callbacks. It does not prove branch feasibility, backend selection or dynamic dispatch. The one function without a caller path is `ops/histogram.py::histogram_kernel`: that file’s host wrappers use torch.bucketize/scatter_add instead. This is evidence to audit reachability, not a compiler failure.

The pilot uses the unchanged `ops/unfold_backward.py::_unfold_backward_kernel`. Its host wrapper creates a fresh zeroed fp32 output. The kernel reads the input and performs one unused-result atomic add. A private-output/disjoint-input contract therefore follows from that host allocation; no index uniqueness, immutable-input assertion or unordered-reduction permission was invented.

On NVIDIA B300, the JIT-specialized pilot measured **1.2997×** geometric-mean baseline/native speedup over the 21 exact upstream benchmark shape/dtype cases.
This timing includes output reset plus the extracted update kernel. It excludes final dtype conversion, allocation, Python/package dispatch and the full FlagGems benchmark harness; it is not a whole-operator speedup. Original warmup scalar values, pointer/scalar divisibility, constexprs and default launch options are preserved. Other experimental transformations are disabled consistently.

The native pass transformed all 18 JIT specializations. Baseline PTX uses `atom.global.gpu.acq_rel.add.f32` and 2 CTA barriers; native/manual use `red.global.gpu.relaxed.add.f32` with 0 CTA barriers. Native and the sem-only manually relaxed control have identical executable PTX for every specialization. This comparison combines ordering and atom/red selection; it does not isolate barrier latency.

All 21 cases passed exact small-integer checks and random-input checks against CPU float64 accumulation with per-destination gamma_(n-1)×sum(abs(x)) bounds; 12 cases contain collisions. Timing uses shared physical buffers, full backing-allocation disjointness checks, reset inside every captured invocation, 100 graph repetitions and 9 randomized-order rounds. Reset-only time is recorded separately and never subtracted.

The earlier generic-signature control measured 1.2953×. It forces scalars to runtime i32 and omits upstream alignment/value specialization; it is retained separately under `pilot/generic/` and is not pooled with `pilot/production/`.

The other 92 functions, including all 35 LICM functions, received no additional contract/compiler/performance review in this funnel run. Previous independent experiments are not reconciled into these counts. The selected successful pilot does not estimate the share of the 91 atomic sites or 62 LICM pairs that are legal, transformable or profitable.

Reports: [funnel summary](summary.json), [JIT-specialized pilot](pilot/production/summary.json), [generic control](pilot/generic/summary.json), [provenance](provenance.json). The native compiler library hash is unchanged between compilation and artifact finalization for both pilots.

Reproduce from the Triton checkout with `python3 experiment_notes/semantic_candidate_experiments/semantic_funnel_experiment.py --fetch` for the inventory, then the semantic virtual environment’s Python with `--production-pilot` for the measured JIT-specialized pilot. `--gpu-pilot` reproduces the generic control. GPU ownership must be coordinated. Pinned tests/benchmarks and archive SHA are in provenance; production measured harness, IR/PTX and source snapshots are retained. See `LICENSE.FlagGems` and `NOTICE.FlagGems`.
