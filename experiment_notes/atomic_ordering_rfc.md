# Draft RFC: make implicit atomic ordering visible

Local draft, 2026-11-09. This has not been submitted upstream.

An omitted `sem` on a reduction can add measurable ordering work. In our B300
embedding-backward controls, changing only atomic ordering from the Triton
default to `relaxed` improved whole-operation performance by **1.189×**
geometrically across 36 matched configurations, including reset and conversion.
This is about 15.9% less time. The proposed first step is an opt-in diagnostic
that asks authors to review implicit ordering, with no automatic rewrite and no
change to the language default.

## Why expose this choice?

Triton's [`tl.atomic_add`](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)
defaults to `acq_rel`. CUDA's legacy
[`atomicAdd`](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-programming-guide/index.html#atomic-functions)
has relaxed ordering. A direct source port can therefore preserve atomicity while
accidentally adding synchronization that its reduction does not need. This is a
review opportunity, not evidence that all implicit ordering is unnecessary.

In the locally installed PyTorch 2.14.1+cu130, Inductor's ordinary `atomic_add`
store codegen explicitly emits `sem='relaxed'` (`_inductor/codegen/triton.py`,
line 5093). Its scan and barrier helpers also use explicit acquire/release
atomics. For example, `exclusive_scan_decoupled_lookback` publishes an ordinary
store with an unused-result release exchange, then polls with acquire additions.
Discarding the atomic return value does not make publication ordering removable.

## Measured evidence

The [four-way experiment](semantic_candidate_experiments/results/atomic_fourway_final/README.md)
uses two pinned FlagGems embedding kernels, three shapes, two block sizes and
three seeds. Original kernel bodies are AST checked against upstream; controls
change only the function name and `sem`. Compiler atomic-specialization passes
are disabled. All variants use common physical buffers, randomized interleaving
and the same zero/count/update/conversion work. All 144 mode/configuration
executions pass their correctness checks.

| Comparison | Geometric time ratio | Observed B300 lowering |
| --- | ---: | --- |
| acq_rel / relaxed | 1.1891× | ATOMG + ordering → REDG without those barriers/fences |
| acq_rel / acquire | 1.1505× | Both ATOMG; release-side ordering removed |
| acq_rel / release | 1.1115× | ATOMG → REDG; acquire-side ordering removed |
| release / relaxed | 1.0698× | Both REDG; release-side ordering removed |
| acquire / relaxed | 1.0336× | ATOMG → REDG; acquire-side ordering removed |

The observed release paths include `MEMBAR.ALL.GPU`; acquire paths retain a
post-atomic CTA barrier. All atomic additions retain the same FP32 FTZ/RN
arithmetic. The four-way controls rule out instruction family as the sole
explanation. They do **not** isolate pure CTA-barrier latency: GPU ordering,
instruction selection and scheduling also differ.

The separate [native contract experiment](semantic_candidate_experiments/results/native_private_atomic_final/summary.json)
reports 1.192× with unique indices and 1.179× with fourfold repeated indices.
The [integer frequency-counter experiment](semantic_candidate_experiments/results/private_atomic_loop_final/summary.json)
reports 1.54–1.76× for uniform indices but only 1.003–1.34× with 32 hot
destinations. Contention can dominate any ordering savings. These are distinct
studies at their recorded compiler revisions; their measurements must not be
attributed to later builds.

A subsequent independent [FlagGems unfold-backward pilot](semantic_candidate_experiments/results/semantic_funnel_final/README.md)
measures 1.300× across 21 upstream benchmark inputs and 18 distinct actual JIT
specializations. The contract pass and manual relaxed control agree on executable
PTX. This measures reset plus accumulation, excluding allocation, final dtype
conversion and dispatch; it is not a complete operator benchmark. The generic
scalar-specialization control is retained separately. These small cases extend
the evidence to another source kernel, not to the full survey.

The pinned four-project [source survey](semantic_candidate_experiments/results/semantic_survey_final/summary.json)
finds 311 default-order, syntactically unused atomic-add sites, or 274 after exact
body deduplication. These include platform variants and helpers. They are
review candidates, not 311 legal or profitable optimizations. The ongoing
FlagGems funnel maps source sites to host paths, actual specializations and
benchmark inputs before making an applicability claim.

## Proposed diagnostic

Offer a disabled-by-default source diagnostic for an atomic-add call that omits
`sem` (including an explicit constant `None`) and whose result is unused. An
initial implementation can reuse source locations and the AST survey; a future
compiler diagnostic can use actual SSA liveness after specialization. Preserve
whether the ordering was implicit, because current TTIR records the resolved
ordering rather than the author's intent.

Suggested message:

```text
atomic_add uses implicit acq_rel ordering and its return value is unused.
Review whether this atomic publishes or observes other memory. Specify sem
explicitly to document the required ordering. Unused results alone do not
justify sem="relaxed".
```

Do not flag explicit `acq_rel` as an accidental default. Deduplicate diagnostics
by source site, provide an opt-in report rather than repeated JIT warnings, and
treat dynamic/forwarded semantics as unknown. Suggest no automatic fix. The
diagnostic is advisory even when the compiler cannot prove a relaxation.

Separately, a proof-based optimization may relax ordering when caller contracts
and whole-function analysis establish that synchronization is unobservable.
The existing prototype requires initialized invocation-private output, input
accesses disjoint from that output, unused atomic old values and no unsupported
effects. It supports a bounded straight-line FP32 case, plus modular i32
additions across multiple sites/loops. It does not derive privacy from a dead
atomic result. Floating-point loops and multiple sites need additional reasoning
about joint final outcomes; generic operator nondeterminism is not a substitute
for that proof.

## Acceptance criteria and remaining evidence

- Check default/explicit/dynamic ordering and live/dead results, including JIT
  helpers and specialization, without changing generated code in diagnostic mode.
- Include publication, polling and barrier examples as negative controls for any
  automatic optimization. Keep deterministic-mode and FP32 rounding/FTZ behavior
  explicit when adding operator-level contracts.
- Report the funnel separately: source sites, distinct functions, host-mapped
  paths, compiled specializations, proven transformations, correctness checks,
  and paired performance. Include rejected, unreviewed and unavailable cases.
- Repeat controlled reductions across collision distributions and at least a
  second architecture before claiming portable performance impact. Current
  clocks are unlocked; timings are GPU operations rather than model speedups.

An advisory diagnostic and documentation change can be reviewed using the
existing evidence. A general default change or a claim that every reduction
benefits would require substantially more evidence.
