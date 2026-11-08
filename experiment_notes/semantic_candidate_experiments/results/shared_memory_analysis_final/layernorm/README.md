The shared analysis now enables alias-aware load reuse on the unchanged native
LayerNorm ownership kernel. The paired control changes only
`enable_contract_load_reuse=False/True`. Source, scalar specialization, contracts,
launch and configuration match. Every record asserts identical ownership TTIR and
identical unannotated baseline PTX before comparing the transformed binaries.

All five cases remove the duplicate weight load: W-rooted static TTGIR loads
fall from two to one; total `tt.load` sites fall from ten to nine. LICM had added a
loop-entry conjunction to the second load's mask. The earlier mask covers all its
active lanes, and both loads omit `other`, so the bounded implication proof can
reuse the first result without adding or moving a memory access.

| Input / M×N | Groups/warps | Off / on μs | Median paired ratio | Three-seed range | Registers | Spills |
|---|---:|---:|---:|---:|---:|---:|
| bfloat16 / 4096×1024 | 256/4 | 24.093 / 24.006 | 1.0050× | 1.0026–1.0058× | 80→80 | 0→0 |
| bfloat16 / 4096×4096 | 256/8 | 49.848 / 49.850 | 1.0000× | 0.9978–1.0016× | 128→128 | 0→0 |
| float32 / 4096×1024 | 256/4 | 27.323 / 27.278 | 1.0016× | 1.0012–1.0016× | 72→72 | 0→0 |
| float32 / 4096×4096 | 256/4 | 54.537 / 54.911 | 0.9932× | 0.9932–0.9992× | 235→234 | 0→0 |
| float32 / 513×4097 | 128/4 | 35.566 / 33.281 | 1.0722× | 1.0612–1.0730× | 255→255 | 170→164 |

The four main cases are effectively neutral or change by less than 1%. The
irregular 513×4097 case improves 6.1–7.3% relative to prior native ownership, with
fewer spills. This does not establish a general profitability or spill guard.
The source-owner gap on some large cases remains. The test demonstrates reuse of
the same memory facts by two optimizations, with a modest measured performance
result rather than a new broad LayerNorm speedup claim.

These are B300 complete-backward GPU timings with the same physical outputs,
lock/count reset and unchanged final reduction. Three fresh input seeds 21/22/23
used 15 randomized paired rounds and 32 graph repetitions. Forward, compilation,
allocations and Python guards are excluded. The fixed configurations were chosen
before this control. Locked, manual-owner and Torch implementations are retained
as timing controls in the raw records, not retuned or claimed best available here.

All 15 records pass exact permitted-original-serialization checks for native off,
native on and manual ownership, before and after timing. The checks include
partials, DX, final gradients, counts/locks, inactive state, canaries and unchanged
inputs. The independent FP64 mathematical diagnostic remains separate: existing
BF16 absolute-0.01 failures remain reported, without widening tolerance or dropping
cases. Original source and the state oracle match the reconstructed ownership study.

`measurements.json.gz` preserves all raw records, paired samples, contracts and
validation reports. `code/` retains the exact measured TTIR/TTGIR/PTX for the four Triton variants and final
reduction at the first seed of each case. `source_snapshots.json` records exact
measured source hashes and compressed snapshots. Historical source/native bundles
were left unchanged.

Reproduce one fixed control from the rebuilt Triton checkout into a fresh path:

```sh
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/layernorm_ownership_experiment.py --reuse-control --mode confirm --dtype bfloat16 --rows 4096 --cols 4096 --groups 256 --warps 8 --seeds 21,22,23 --output /tmp/layernorm-shared-memory-rerun
```

Use the configurations in `summary.json` for the other four cases. Runs require
exclusive GPU ownership. The retained source/license applies to the extracted
Tritonbench tutorial.
