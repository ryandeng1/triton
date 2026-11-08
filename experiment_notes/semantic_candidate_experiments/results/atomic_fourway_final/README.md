# Four atomic-order controls

The default `acq_rel` → `relaxed` speedup repeats at **1.189×** across 36 matched configurations. Holding the instruction family fixed still produces a substantial difference: `acq_rel` → `acquire` is **1.150×** with ATOMG, and `release` → `relaxed` is **1.070×** with REDG. Acquire is faster than release despite retaining ATOMG. The atom-to-red change alone cannot explain the total gain.

| Paired time ratio, geometric mean over 36 cases | Ratio |
| --- | ---: |
| acq_rel / relaxed | 1.1891× |
| acq_rel / acquire | 1.1505× |
| acq_rel / release | 1.1115× |
| release / relaxed | 1.0698× |
| acquire / relaxed | 1.0336× |

Whole-operation medians in microseconds below use **the same block size 256** for all four modes, then the median of three seeds. Both block sizes and best-of-two summaries are in `results.json`; choosing each mode's best block gives a similar 1.1855× baseline/relaxed geometric mean.

| N × D; vocabulary | Variant | acq_rel | acquire | release | relaxed |
| --- | --- | ---: | ---: | ---: | ---: |
| 4,096 × 128; 8,192 | plain | 7.290 | 6.146 | 6.388 | 5.811 |
| 4,096 × 128; 8,192 | scaled | 12.798 | 11.627 | 11.716 | 11.083 |
| 16,384 × 256; 32,768 | plain | 22.455 | 20.034 | 20.356 | 19.587 |
| 16,384 × 256; 32,768 | scaled | 28.990 | 25.736 | 26.515 | 25.182 |
| 8,192 × 1,024; 16,384 | plain | 45.894 | 39.338 | 40.906 | 38.349 |
| 8,192 × 1,024; 16,384 | scaled | 54.079 | 46.804 | 48.900 | 44.427 |

PTX and B300 SASS agree on the following lowering. Atomic return values are unused in every source variant.

| Order | Atomic family | CTA barriers | Other observed SASS ordering |
| --- | --- | --- | --- |
| acq_rel | ATOMG | before and after | MEMBAR.ALL.GPU before atomic |
| acquire | ATOMG | after | no MEMBAR.ALL.GPU |
| release | REDG | before | MEMBAR.ALL.GPU before atomic |
| relaxed | REDG | none | no MEMBAR.ALL.GPU |

The SASS atomics use `.F32.FTZ.RN.STRONG.GPU`; ATOMG writes to RZ. Release-bearing paths also contain ERRBAR/CGAERRBAR. Representative plain kernels all use 12 registers, scaled kernels 14, with zero spills and shared memory. **These are lowering-path controls, not a measurement of pure CTA-barrier latency:** release versus relaxed also changes GPU ordering and emitted MEMBAR instructions; acquire versus relaxed changes atomic family and completion ordering. Instruction scheduling can change too.

Method: NVIDIA B300; three original shapes × plain/scaled × block sizes 128/256 × three seeds. Each of 36 cases compares all four variants in 15 randomized rounds of 32 CUDA-graph repetitions, after 200 ms warmup. Each variant shares the same scratch, count and output storage and includes scratch zero, unchanged count reset/kernel for scaled mode, update, and BF16 output copy. Allocation, randperm generation and host guards are outside timing. Clocks are unlocked. No separate update-only claim is made.

The original FlagGems functions are AST-identical to the pinned source; each control differs only in its function name and `sem` keyword. Baseline omits `sem` and uses the default acq_rel. Both `enable_indexed_update_specialization` and `enable_private_atomic_relaxation` are explicitly false, including in compiled metadata. Inputs are the same unique randperm subsets as the earlier ordering experiment. This isolates the order-control question; it is not a prevalence or collision-distribution study.

Validation passed for all 144 compiled/executed mode/configuration combinations: exact FP32 scratch and BF16 output, poisoned output, changed gradients, and an untimed signed-zero/subnormal probe using an explicit FP32-global-atomic FTZ reference. Timed gradients are ordinary finite BF16. The source and compiler hashes were unchanged across the run. Smoke also passed.

Artifacts: `results.json` has raw per-round measurements, hashes, flags, PTX summaries and representative SASS summaries; `code/` has both representative plain/scaled TTIR, TTGIR, PTX and SASS; `sources/` snapshots the harness and required source helpers; `smoke-results.json`, `smoke.log` and `confirm.log` preserve validation/run records. `SHA256SUMS` covers the retained bundle. The compiler binary itself is not copied.

Reproduce from the Triton repository root with the semantic venv:

```sh
/home/ryandeng/tpaa/.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/atomic_fourway_experiment.py --mode confirm --output /tmp/atomic-fourway-new-output
```
