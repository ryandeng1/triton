# Why automatic pipelining regressed

The fixed two-stage policy introduces shared-memory staging but provides almost
no useful overlap in the inspected loops. Three or four stages recover gains in
log-softmax and FP32 rotary. Resize and RMSNorm still prefer ordinary loads.
The existing alias contracts establish legality; choosing a profitable schedule
requires additional performance decisions.

## Controlled measurements

NVIDIA B300, unchanged native compiler
`e7e15b9337405e632e887ceae9c96d9ea85e38aa1e9c35dec937d7b148d06178`.
Same contracts, launch configuration, arithmetic and outputs within each row;
contract LICM and unrelated experimental transforms disabled. Stage sweeps use
the existing compiler option for FlagGems, and source loop annotations for RMSNorm.
Three fresh input seeds, 15 shuffled rounds per seed, 32 invocations per CUDA graph.

Speedups below are baseline time divided by candidate time. These are fixed-launch
causal controls, not a new exhaustive tuning or corpus-wide study.

| Kernel and fixed launch | Stages 2 | Stages 3 | Stages 4 |
|---|---:|---:|---:|
| Log-softmax backward, M256/K8192 FP32, block1024/w8 | 0.876–0.882× | **1.113–1.115×** | 1.101–1.103× |
| Rotary, T128/Q32/K8/D128 FP32, w2 | 0.871–0.874× | 1.497–1.506× | **1.921–1.930×** |
| Same rotary, w8 | 0.780–0.783× | 1.319–1.325× | **1.678–1.683×** |
| Bilinear resize, 2×64×64×256 → 128×512, tile64×2/w4 | 0.653–0.655× | 0.763–0.764× | 0.760× |
| RMSNorm forward 4096×1024, w4, one row per CTA | 0.736–0.767× | 0.734–0.760× | 0.687× |
| Same RMSNorm, persistent grid 4×SM | 0.862–0.878× | 0.871–0.896× | 0.867–0.888× |

Log-softmax and resize use their previously selected best baseline launches.
Rotary includes both the earlier pipeline choice (w2) and best baseline choice
(w8). Do not report the w2 speedup as independently tuned speedup. RMSNorm's
baseline here explicitly requests stage1; the upstream source requests stage2.
Its row-grid loop executes once per CTA, and its 4×SM grid executes 6–7 times.

All **126 variant validations and 204 bitwise output comparisons pass**, including
independent mathematical references, poisoned outputs, allocation guards and input
integrity checks. No native code or mathematical kernel bodies were changed.

## Mechanism

For the direct loads inspected, stage2 assigns latency1 and one shared slot.
Reverse stage-cluster order expands to:

```text
wait(copy i) → shared read i → compute/store i → reuse barrier
            → issue copy i+1 → pointer/control work → wait(copy i+1)
```

Copies overlap a few address/control instructions and other copy issues, but no
different iteration's floating-point work. The transformation also adds shared
reads, commits, and a final masked zero-fill/drain. Stage3/4 allocate two/three
slots and leave later copy groups pending while computing an intervening
iteration. Actual TTGIR/PTX confirms this; it is not inferred from the option name.

The selective log-softmax controls localize most stage2 loss to the output loop:
sum-only staging is about 0.96–0.97×, output-only about 0.90–0.91×. Both individual
loops lose. Explicit `.ca` caching does not rescue the result: stage2 worsens to
about 0.82–0.83× and stage3 falls to about 1.03×. Rotary/resize cache controls are
essentially neutral. Cache bypass alone is therefore insufficient as an explanation.

Resize retains four gathered neighbor streams, four copy commits, four shared
matrix reads and two CTA barriers per channel iteration for seven floating-point
operations per output. Deeper buffers add index-management instructions. On the
selected warm-cache workload, Nsight records **1.69× executed instructions** for
stage2, about **1.05 million additional shared-read wavefronts**, and only **1.2% more
L2 read sectors**. Register residency limits remain equal and achieved occupancy
does not decline. Stage3 improves overlap but executes about **1.96× baseline
instructions** and still loses in unprofiled graph timing. This supports excess
on-chip staging/synchronization work rather than an occupancy collapse or extra
off-chip traffic for this workload. Barrier-only cost was not isolated.

For RMSNorm with one row per CTA, no next useful iteration exists. Deeper stages
increase setup/staging cost; all lose. The persistent control also loses, so trip
count alone cannot decide profitability.

The earlier recurrent stage2 success has a different code-level explanation:
staging removes two q/k layout conversions and reduces repeated shared stores and
barriers. It also lacks useful interiteration floating-point overlap. This is a
code-based explanation, not an isolated causal measurement of layout savings.

## Implications and limits

The next scheduling policy should account for actual prefetch distance, trip
count, number/width of load streams, staging and layout-conversion costs, and
resource usage. A blanket stage2 choice misses real opportunities; simply raising
the default would still regress resize and RMSNorm. Preserve the ordinary-load
candidate and compare stage depths independently of LICM.

There are no spills in these controls; register pressure is not a general
explanation. This study covers representative launches from four families on one
GPU. It does not establish a universal depth rule or a broad applicability rate.

Nsight uses application replay, five warm launches before the measured launch,
uncontrolled caches/clocks, and explicit metrics. Counter results are diagnostic.
In particular, its resize stage3 timing reverses the graph comparison, so profiler
durations are **not** performance evidence. Ordinary global-load counters omit
some asynchronous-copy traffic; use L2/DRAM counters for traffic comparisons.

## Evidence

- [Measurement plan](controls/plan.json), [environment](controls/environment.json),
  per-case JSON, generated annotation audits and compressed TTIR/TTGIR/PTX in `controls/`.
- [Static mechanisms and exact source/PTX citations](static_mechanisms.json).
- [Recurrent comparison and qualifications](recurrent_mechanism.json).
- [Profile commands](profiles/commands.json), [counters](profiles/summary.json),
  raw CSV and [review](profile_review.json).
- Driver: [pipeline_slowdown_experiment.py](../../pipeline_slowdown_experiment.py).
  Controls: [pipeline_slowdown_controls.py](../../pipeline_slowdown_controls.py).
  Profiling: [pipeline_slowdown_profile.py](../../pipeline_slowdown_profile.py).

The preceding [cross-kernel study](../cross_kernel_pipeline_final/README.md) remains
the record of the independently tuned fixed-stage2 policy. These controls explain
and qualify its negative result; they do not replace its observations.
