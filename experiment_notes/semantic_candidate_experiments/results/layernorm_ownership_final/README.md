The native ownership pass transforms the unchanged Tritonbench LayerNorm backward
row kernel into a persistent reduction. Private storage, alias separation, exact
launch/scalar facts and the recognized lock protocol let it derive one owner per
partial-gradient group. The source rewrite was a performance bound; the native
experiment demonstrates that these contracts enable the compiler transformation.

Git history retained the old protocol and summaries, but never contained the old
harness or owner kernels. The registered tutorial source still matches its exact
historical SHA256. The reconstructed harness extracts original kernels with
identical ASTs and mechanically builds the source owner's row computation from
the original prefix. Historical timings remain separate from these fresh runs.

For group p, owner program p visits rows p, p+G, ... and carries two partial
vectors. The original M-program grid, signature, scratch precision, lock/count
reset, final bookkeeping state and final reduction remain. N and stride stay
normal scalar arguments in all three variants; native exact-value attributes
support proofs without replacing these arguments with constants.

On B300, the 96-configuration source screen selected a shortlist. Native
confirmation used three independent input seeds, 15 randomized paired rounds and
32 graph repetitions per round. Each entry uses the best tested configuration for
that variant. Times include reset and unchanged final reduction, and exclude
forward, allocation and host guards. All main cases have M=4096 and FP32 scratch.

| Input / N | Tuned original | Native | Speedup | Torch native |
|---|---:|---:|---:|---:|
| BF16 / 1024 | 49.69 us | 24.19 us | 2.05x | 23.09 us |
| BF16 / 4096 | 80.22 us | 50.39 us | 1.59x | 55.27 us |
| FP32 / 1024 | 48.35 us | 27.29 us | 1.77x | 22.07 us |
| FP32 / 4096 | 78.14 us | 54.86 us | 1.42x | 82.88 us |

Quack is unavailable. Torch is faster at width 1024 and slower at width 4096;
these results are not a claim against the best available LayerNorm implementation.
The old-build source-owner gains were 2.02x, 1.83x, 1.98x and 1.44x, respectively.
They are stored separately from the rebuilt native study.

All 30 native records (21 main, nine boundary) pass bitwise checks against an
independently evaluated permitted original serialization. An unchanged G=M
baseline captures exact original FP32 row contributions and DX. A separate
ordered-reduction kernel applies original new+old and scratch rounding in
increasing-row order. First contribution assignment performs no addition by zero.
The owner and native partials, DX and final gradients match this reference
bitwise. Guards, inactive partials, final counts/locks and unchanged inputs are
checked before and after timing. The source stage also covered FP16 scratch,
M<G, N=1 and non-power-of-two rows/widths; that broader source coverage does not
imply the initial native recognizer accepts every form.

The native proof requires initialized private writable allocations, zeroed
locks/counts, disjoint input reads, complete unique row-write coverage, bounded
signed-i32 affine address intermediates and no intermediate observers. It
recognizes the two-FP32-partial protocol by IR roles, without kernel or argument
names. G=1 and N=1 canonical forms can decline, and FP16/BF16 scratch is outside
native scope. Selecting a legal original serialization does not permit arbitrary
floating reassociation or changing scratch precision.

Profitability remains conditional. The FP32 513x4097 boundary regresses from
32.99 to 35.90 us, with stage-one spills increasing from 2 to 170. The pass has no
profitability guard. Native TTGIR also retains a second weight load before the
loop; manual ownership is faster on some configurations. These are recorded
limitations, not hidden rejected measurements.

The separate mathematical diagnostic uses FP64 derivative arithmetic with actual
saved FP32 statistics and direct nearest-even output rounding. Some fresh
standard-normal BF16 cases fail historical absolute tolerance 0.01 in original,
owner and native. Those failures remain without relaxing tolerance or excluding
configurations. FP32 cases pass. Exact refinement of original program arithmetic
does not establish a universal BF16 mathematical or training-accuracy guarantee.

`summary.json` contains aggregates and limits. `source_measurements.json.gz` and
`native_measurements.json.gz` preserve all raw records, timing samples and reports.
`native_logs.json.gz` and `source_logs.json.gz` retain stdout. Exact measured
harness snapshots, tutorial/license, generated owner and native source snapshots
are under `sources/`; `native_source_snapshots.json` records their hashes.
`source_code/` contains the recovered exact old-build screen artifacts;
`native_code/` contains each run's first requested configuration, with IR/PTX
compressed except directly readable FP32-width-1024 TTIR. `compaction.json` records original paths and hashes. These archives
contain fresh measurements only; `sources/historical_protocol.md` is provenance.

Reproduce source and native confirmations from the rebuilt Triton checkout:

```sh
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/layernorm_ownership_experiment.py --mode confirm --dtype float32 --rows 4096 --cols 1024 --groups 256 --warps 4 --seeds 11,12,13 --output /tmp/layernorm-source-rerun
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/layernorm_ownership_experiment.py --native --mode confirm --dtype float32 --rows 4096 --cols 1024 --groups 256 --warps 4 --seeds 11,12,13 --output /tmp/layernorm-native-rerun
```

Use fresh output directories. The current harness defaults to the retained
historical tutorial snapshot, with a sibling Tritonbench checkout as fallback;
`--tutorial` can override it. The measured harness snapshots precede only this
portable-path/docstring/summary-reporting cleanup. A CPU `--mode prepare` check
verified the new default and exact source AST extraction; GPU results above
retain the measured snapshots' fingerprints.
