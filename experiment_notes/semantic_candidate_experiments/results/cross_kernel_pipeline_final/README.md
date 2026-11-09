# Cross-kernel validation of contract-licensed pipelining

The fixed two-stage automatic-pipelining policy finds **no additional profitable
kernel family** in this study. The earlier recurrent-kernel gains therefore
remain the only demonstrated performance wins for this automatic-prefetch policy.
The compiler successfully transforms independent kernels, but the tuned resulting
schedules lose or are declined. This does not establish a fundamental limitation
of semantic contracts or rule out different schedules.

We tested five main entry kernels from four families: AITER RMSNorm forward and
complete backward, FlagGems log-softmax backward, out-of-place rotary embedding,
and bilinear resize. The backward RMSNorm timing also includes its unchanged
auxiliary dgamma reduction. There are 13 timed workloads, five numerical boundary
workloads and 65 configurations, all on NVIDIA B300. Every kernel body is unchanged;
provider decorators/dispatch are replaced by explicit experimental launches.

| Kernel/family | Automatic-pipeline speedup vs independently tuned contracts-only baseline | Interpretation |
| --- | ---: | --- |
| Log-softmax backward, two FP32 workloads | 0.879–0.882× | Both chunk loops transform; tuned baseline wins. |
| Log-softmax backward, BF16 | 0.943–0.952× | Transforms and loses. |
| Rotary embedding, FP32 | 0.840–0.843× | Q and K head loops transform; both outputs are timed. |
| Rotary embedding, two BF16 workloads | Approximately 1× | Selected configurations have identical PTX/cubin to baseline and no async copies. Transformed smaller-warp configurations lose. |
| Bilinear resize, three workloads | 0.656–0.854× | Transforms, but baseline wins on every workload. |
| RMSNorm forward/backward | Identical auto-option on/off code | Source already requests stages=2; these are existing-schedule controls. |

Ratios above are baseline time / pipeline time; values below one are slower.
Ranges span three fresh input seeds after configuration choices were frozen.
All nine timed workloads in the three newly inferred families retain the baseline
under the predeclared 2% selection margin. The BF16 rotary neutral cases are not
counted as successful pipelined kernels.

## Controls and attribution

Each case has five variants: uncontracted guarded source, contracts only,
contracts+LICM, contracts+automatic pipeline, and contracts+both. The last four
carry identical input/output alias groups. Each output allocation has a distinct
group, and full backing-storage separation is checked. Contract load reuse and
other experimental transformations are disabled. The new optimization denominator
is **contracts only**, avoiding attribution of restored source-requested pipelines
to automatic stage inference. Every variant is independently tuned over the same
four or six configurations, including RMSNorm's one-CTA-per-row competitor.

RMSNorm's source already requests two stages. In all 26 tested configurations,
`pipeline` equals `contracts` PTX and `both` equals `licm` PTX. Some calibration
winners are labelled `both` because separately measured identical binaries have
small timing differences; these are LICM choices, not joint-transform wins.
Compared with the independently tuned **uncontracted guarded** source, the frozen
selected choices are 0.947–0.959× and 0.960–0.962× for the two forward workloads,
0.930–0.935× for smaller backward, and 0.999–1.000× for larger backward. Thus the
within-contract LICM gains do not imply a new net win over the faster safe control.
This different baseline does not overwrite earlier historical RMSNorm results.

The three FlagGems families' contracts-only PTX equals uncontracted PTX in all
39 configurations. Automatic stages change code in 34/39 configurations: 13/13
log-softmax, 8/13 rotary and 13/13 resize. The five remaining rotary configurations
retain ordinary loads. This separates legal transformations from ineffective
options and unprofitable schedules.

## Generated code and limits

Log-softmax pipelines its read-only sum pass and its load/store output pass:
six static async-copy sites and four waits, with selected variants using 4–8 KB
shared memory versus 16–32 bytes in baseline. The first loop is already independent;
pipelining it is a scheduling-policy change, not missing-alias information.
FP32 rotary adds eight copy sites, four waits, shared memory 0→1 KB and six static
PTX CTA-barrier sites. Resize adds eight copy sites and uses 1–2 KB at selected
configurations. Registers do not consistently rise, and these candidates do not
spill. These observations make staging and synchronization overhead plausible;
we did not isolate their costs with profiling or establish a bandwidth bottleneck.

Candidate selection was purposeful, not a random corpus sample. Log-softmax is
FlagGems' **experimental_ops** implementation. Resize uses exact output tiles:
its modulo-indexed source may overlap output stores on partial tiles, which are
excluded. Configurations and stage count are bounded rather than exhaustive.
Timing excludes allocation, metadata generation, host dispatch, compilation and
calibration. It is not a strongest-provider or model-level comparison. The kernels
overwrite complete outputs, so timing requires no reset; validation poisons outputs.

Retained TE unpermute/segment reductions use source while loops; CTC has genuine
loop-carried alpha dependencies and barriers. These were inspected as exclusions,
not timed or added to the measured kernel denominator.

## Validation and reproduction

All **625 variant-validation records** pass complete independent mathematical
references, output guards and input-integrity checks. All **1,000 output comparisons**
are literally bitwise equal to their same-configuration contracts baseline.
Boundaries cover RMSNorm zero-trip CTAs/masked width, log-softmax masked chunks,
rotary padded dimension/strided positions, and single-height resize. The compiler
binary remains the same as the earlier recurrent experiment:
`e7e15b9337405e632e887ceae9c96d9ea85e38aa1e9c35dec937d7b148d06178`.
No native compiler changes were made for this study.

[findings.json](findings.json) records aggregate counts, ratios, exact-code controls
and resources. `main/` and `upsample/` retain predeclared plans, source audits,
complete tuning/confirmation records, logs, compressed measured harnesses and
TTIR/TTGIR/PTX. Pinned originals and licenses are in
[the source bundle](../../vendor/cross_pipeline/). `initial_launch_shape_failure/`
records a corrected host-launch tuple error before any benchmark kernel executed;
it is excluded from all reported counts.

From the Triton checkout, use fresh output directories:

```sh
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/cross_kernel_pipeline_experiment.py --adapters rms flaggems --output /tmp/cross-pipeline-main
../.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/cross_kernel_pipeline_experiment.py --adapters upsample --output /tmp/cross-pipeline-upsample
```
