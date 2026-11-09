# NVIDIA global-dependence guard

The compiler now declines software pipelining and automatic warp partitioning
when it cannot prove the required global-memory independence. The valid recurrence
`X[t+1,:] = X[t,:] + 1`, which previously failed at loop stages=3, now passes.
The guard applies to explicit stages, latencies and saved schedules as well as
newly inferred schedules. The separate descriptor-store path completes writes
before reusing staging storage or issuing the next potentially overlapping store.

Different `tt.experimental_alias_group` values promise disjoint **derived access
footprints throughout the invocation**, not merely unequal base pointers.
Missing/equal groups may alias. Explicit immutable input is another way to prove
read stability. The guard checks ordinary read/write and distinct store/store
conflicts, volatile accesses, unknown effects and source synchronization. It does
not invent initialization, privacy, or pointer-extent facts.

Compiler-created barriers and private tensor-descriptor rings are admitted only
through a pre-protocol proof plus fresh checks of global effects/provenance.
Arbitrary source protocols, mixed descriptor reads/writes, unresolved pointer
provenance and CLC scopes remain conservative fallbacks. This can reduce overlap
in existing kernels lacking usable facts. AMD's separate pipeliner is outside
this patch. Existing read-only inner matmul pipelines remain supported.

`contract_pipeline_stages=2` is an opt-in NVIDIA compiler option that infers
loop stages for supported innermost straight-line unit-step loops with explicit
memory facts. `enable_contract_licm` independently controls contract-based load
hoisting, leaving ordinary LICM enabled. Existing loop scheduling hints are
preserved as candidates and still checked for safety. Gluon does not run this
automatic inference path.

Final evidence is in [final_validation.json](final_validation.json): 33 lit files
and 319 distinct selected runtime cases pass on the same native library used
for the [performance experiment](../contract_pipeline_final/README.md).
`runtime_validation.json` and earlier logs record intermediate builds; they do
not identify the final binary. Positive fixtures now state their disjointness or
concrete effects, unsupported cases explicitly test fallback, and lower-only
test helpers retain mechanical protocol coverage without claiming legality.
