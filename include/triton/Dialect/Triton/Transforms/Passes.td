#ifndef TRITON_PASSES
#define TRITON_PASSES

include "mlir/Pass/PassBase.td"

def TritonCombineOps : Pass</*cli-arg*/"triton-combine", /*Op*/"mlir::ModuleOp"> {
  let summary = "combine ops";
  let description = [{
    This pass aims to optimize the five following patterns:
    - `dot(a, b, 0) + c => dot(a, b, c)`

    - `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))`

    - `select(cond, load(ptrs, broadcast(cond), ???), other) =>
         load(ptrs, broadcast(cond), other)`

    - `broadcast(constant) => reshaped_constant`
    - `torch.sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1)
       => dot(x,y,splat(0))`
  }];

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def TritonReorderBroadcast : Pass</*cli-arg*/"triton-reorder-broadcast", /*Op*/"mlir::ModuleOp"> {
  let summary = "Moves broadcast and splat after elementwise operations";
  let description = [{
    The purpose of this pass is to transform:
      - `elementwise(broadcast(a)) => broadcast(elementwise(a))`
      - `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`
    In the event of a match, the broadcast (or splat) operation is delayed
    and performed after the ElementWise operation.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonRewriteTensorDescriptorToPointer : Pass</*cli-arg*/"triton-rewrite-tensor-descriptor-to-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores of tensor descriptors into pointer load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_descriptor` into pointer semantics. After
    this pass, `tt.make_tensor_descriptor`  will disappear, and it generates logics to compute the pointer/mask/other
    for each load/store.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopUnroll : Pass</*cli-arg*/"triton-loop-unroll", /*Op*/"mlir::ModuleOp"> {
  let summary = "Loop unroller";
  let description = [{
    The pass unrolls a scf loop with tt.loop_unroll_factor attribute. The attribute specialises how many iterations
    the loop should be unrolled.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopInvariantCodeMotion : Pass</*cli-arg*/"triton-licm", /*Op*/"mlir::ModuleOp"> {
  let summary = "MLIR's LICM plus hoist load ops out of loops with masks.";
  let options = [
    Option<"enableMemoryContracts", "enable-memory-contracts", "bool",
           /*default*/"true",
           "Use immutable and alias contracts for load hoisting across writes">
  ];
  let description = [{
    This pass uses MLIR's LICM pass as base. Additionally, it hoists load ops
    out of loops that consist of pure/read-only ops. For scf.for loops, it
    generates a trip-count check.

    Experimental opt-in: a pointer argument with `tt.experimental_immutable =
    1 : i32` promises that the memory reached through that argument is
    initialized at call entry and remains unchanged through every alias for
    the entire call. This is stronger than a readonly pointer argument and
    must be established by the caller. It allows invariant ordinary loads
    across ordinary stores, retaining their masks and loop-entry guards.
    Constant initial integer comparisons are supported for after-region loads
    in scf.while when the before region contains only straight-line,
    speculatable, effect-free entry-test computation. Contract-based hoisting
    also requires a straight-line prefix before the load in the loop body;
    entering a loop does not imply reaching a load after a nested region.
    Unknown initial conditions remain unsupported. Volatile
    accesses and loops with assertions, calls, or other unknown effects are
    not eligible for this extension. An ordinary load rooted in an explicitly
    immutable global pointer argument may also move across existing CTA-scoped
    `ttg.barrier` operations: initialization at entry and invariance through all
    aliases exclude barrier-mediated publication of new input values. The
    barriers retain their positions and their ordering of all other accesses.
    This exception does not extend to shared/tensor inputs, warp-scoped barriers,
    async operations, or the weaker alias-only inference below. Original masks,
    loop-entry guards and straight-line-prefix checks still apply.

    A second opt-in derives read invariance from weaker caller facts. Argument
    attribute `tt.experimental_alias_group = N : i32`, with nonnegative N,
    identifies a may-alias group local to a function. Distinct groups promise
    disjoint actual access regions for the invocation, including every derived
    pointer access. Equal groups and missing/negative attributes may alias.
    This is a footprint contract, not merely distinct base addresses. Inputs
    may share a group when they alias; they need not be disjoint from each other.
    The pass summarizes ordinary loads/stores in the function and proves that
    no write may alias an annotated input. This is a no-clobber query, not an
    assertion that input memory is initialized at entry, and it does not add an
    immutable argument attribute. Original masks and loop-entry guards remain
    necessary. Unknown pointer provenance, volatile accesses, calls, atomics,
    async operations and other unknown effects conservatively prevent this
    inference. Callers must establish the region disjointness facts; this pass
    does not inspect runtime storage addresses.
    Pointer provenance also follows bounded scf.for recurrences whose backedge
    derives from the same carried pointer, including validated nested for results.
    A stable allocation root does not make an advancing address loop-invariant;
    ordinary LICM operand checks remain necessary. Root-changing recurrences,
    cross-carried pointers, selections and unknown pointer operations are rejected.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonLoopAwareCSE : Pass<"triton-loop-aware-cse", "mlir::ModuleOp"> {
  let summary = "CSE within loop bodies";

  let description = [{
    The `triton-loop-aware-cse` pass performs recursive common subexpression
    elimination within loop bodies. Unlike regular CSE, which is a single-pass
    greedy algorithm, this pass can recursively eliminate loop iteration
    arguments and subcomputations that always have the same value.

    For ordinary global loads with experimental argument contracts, also reuse
    an earlier load in the same block across disjoint ordinary stores. Shared
    GlobalMemoryEffects analysis proves whole-function no-clobber from alias
    groups, or accepts an explicit initialized-at-entry immutable contract.
    Calls, volatile accesses and unknown effects conservatively prevent this
    extension; CTA barriers are permitted only for explicit immutable global
    contents. No values are propagated across block/region boundaries.

    Pointer, result type/layout, cache properties, attributes and other operands
    must match. Masks normally match too; when both loads omit `other`, a later
    mask formed by conjunction with the earlier mask may reuse its value.
    Unspecified inactive lanes are refined without adding or speculating reads.
    This does not infer initialization or permit changing explicit masked values.
    The option below disables only this contract-based extension for ablation.
  }];
  let options = [
    Option<"enableContractLoadReuse", "enable-contract-load-reuse", "bool", "true",
           "Reuse ordinary global loads using scoped argument contracts">
  ];
}

def TritonSpecializeIndexedUpdates : Pass<"triton-specialize-indexed-updates", "mlir::ModuleOp"> {
  let summary = "Specialize private indexed atomic updates using producer contracts";
  let description = [{
    Experimental opt-in for CUDA f32 indexed additions. This pass derives global
    destination-address injectivity from the actual pointer expression, index
    producer facts, and a checked launch contract. It does not accept a direct
    assertion that an atomic is safe to remove.

    On an integer index pointer, `tt.experimental_unique_indices = N : i32`
    promises that the prefix [0,N) is initialized, immutable through every alias
    for the invocation, and globally unique. `tt.experimental_index_bound = V :
    i32` additionally bounds those values to [0,V). N and V must be positive.

    On an output pointer, `tt.experimental_private = 1 : i32` promises initialized
    storage ready for ordinary access at invocation entry, no outside accesses
    until invocation completion. It does not assert ownership by a program,
    address injectivity, or the absence of internal synchronization partners.
    Optional `tt.experimental_initial_zero = 1 : i32` promises physical positive
    zero at entry. The existing `tt.experimental_alias_group` contract must prove
    every ordinary read footprint disjoint from the output footprint.

    Output argument attributes `tt.experimental_grid_x`, `_grid_y`, `_grid_z`
    specify exact positive i32 launch dimensions. `tt.experimental_num_ctas = 1 :
    i32` promises one CTA per logical program; grid_z must equal one. The caller
    must enforce these dimensions whenever launching the compiled specialization.

    The bounded proof recognizes IndexArg[program_id(x)] * D + program_id(y) * B
    + make_range(0,B), including lossless casts and equivalent addptr grouping.
    D is a positive constant, grid_x <= N, and grid_y = ceil(D/B). Tail masks must
    imply column < D. Integer arithmetic is checked for overflow. Index loads
    must be scalar and unmasked. Only public kernels with no symbol uses in their
    module are supported; private or called helper functions are rejected.
    Loops, regions, calls, volatile accesses, inline assembly, unknown effects,
    additional writes, live atomic old values, and
    non-GPU scope are unsupported. All unproved cases remain unchanged.

    Under this closed private-memory proof there is no synchronization partner
    or observable publication effect, including for acquire/release atomics.
    Replace the atomic with a masked load, an explicit CUDA add.rn.ftz.f32, and
    a masked store. The initial-zero fact elides the load, retaining the explicit
    addition to preserve subnormal and signed-zero behavior. The pass preserves
    masks and tensor encodings and annotates successfully transformed functions
    with `tt.experimental_indexed_update_specialized = 1 : i32`.
  }];
  let options = [
    Option<"target", "target", "std::string", "\"\"", "Target architecture; only cuda is supported">
  ];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::triton::TritonDialect"];
}

def TritonRelaxPrivateAtomics : Pass<"triton-relax-private-atomics", "mlir::ModuleOp"> {
  let summary = "Relax atomic ordering for closed private-output reductions";
  let description = [{
    Experimental opt-in for CUDA f32 and i32 additions. Keep the atomic operation and
    change acquire, release, or acquire-release ordering to relaxed after proving
    that the ordering has no observable synchronization role. Already-relaxed
    atomics are unchanged and do not receive a specialization marker.

    On the global output pointer, `tt.experimental_private = 1 : i32` promises
    initialized storage ready at invocation entry and no outside accesses or
    observers until invocation completion. Its effects proof establishes that
    weakening internal ordering preserves observable results, rather than
    accepting that as another promise.
    The output additionally requires `tt.experimental_num_ctas = 1 : i32`.
    The existing `tt.experimental_alias_group` contracts must prove every
    ordinary read footprint disjoint from the output footprint. All accesses must remain
    within the contracted allocations. These are caller obligations, not facts
    inferred from race freedom or from observed example values.

    The bounded proof accepts a public kernel with no symbol uses in its module,
    with unused-result GPU-scope rank-one atomic additions. For f32, only one
    straight-line atomic site is supported. For i32, multiple sites and nested
    scf.for loops without iter_args are supported, provided every atomic adds
    to the same private output argument. Integer addition modulo 2^32 makes the
    final state independent of cross-location/iteration ordering, including
    overflow. Floating-point loops and multiple sites need a stronger proof
    because ordering can constrain their observable rounding results.
    Only ordinary nonvolatile loads from known disjoint argument roots and
    memory-effect-free leaf operations may accompany these additions. Other
    regions, calls, inline assembly, unknown effects, additional writes, ordinary
    output reads, used atomic results, non-GPU scopes, and called/private helper
    functions are unsupported. All sites are checked before any are changed.

    Other programs cannot observe an old atomic value or acquire an additional
    write from this function, and the private contract excludes external partners.
    Atomicity is preserved, including for repeated destination indices. The proof
    needs no uniqueness, immutable-index, value-bound, grid, or initial-zero facts.
    It does not promise a fixed floating-point summation order. The atomic's
    pointer, value, mask, type, and scope remain unchanged.

    Successfully transformed functions receive
    `tt.experimental_private_atomic_relaxed = 1 : i32`. Only CUDA with one CTA
    per logical program is supported. All unproved cases remain unchanged.
  }];
  let options = [
    Option<"target", "target", "std::string", "\"\"", "Target architecture; only cuda is supported">
  ];
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::scf::SCFDialect"];
}

def TritonPrivateReductionOwnership : Pass<"triton-private-reduction-ownership", "mlir::ModuleOp"> {
  let summary = "Assign owners to bounded private lock/count reductions";
  let description = [{
    Experimental CUDA-only transformation of a public, uncalled entry kernel.
    Recognize one row computation and disjoint row store followed by a private
    lock/count reduction: lock index program_id(x) % G, empty acquire-release
    CAS spin loop, count-controlled first assignment versus new+old additions
    to two rank-one FP32 partial buffers, stores, a CTA barrier and unlock.
    Recognition follows IR roles and addresses, not kernel or argument names.

    All four mutated pointer roots require tt.experimental_private = 1 : i32,
    meaning initialized storage and no outside access during the invocation.
    Their distinct nonnegative tt.experimental_alias_group values prove disjoint
    actual footprints, including all derived accesses. Ordinary input reads must
    belong to groups disjoint from every mutated root. Readonly inputs may alias
    one another. Calls, inline assembly, other control regions, extra writes and
    unknown effects are rejected; pure arithmetic/math reduce combiners are allowed.

    The lock/count argument requires tt.experimental_initial_zero = 1 : i32 for
    every accessed lock and count at every invocation, experimental_num_ctas = 1,
    and exact experimental_grid_x/y/z facts (y=z=1), all with the tt. prefix and
    i32 values. Optional tt.experimental_value = N : i32 on an i32 scalar argument
    promises its exact runtime value for this invocation. These values are used
    to prove address and induction bounds, not to replace original SSA arguments.
    The caller must enforce these facts; runtime example values are not proofs.

    Address analysis proves group*N+lane partials, disjoint row*stride+lane output
    writes, a common lane<N mask, and nonnegative, nonoverflowing i32 intermediates.
    The original one-dimensional grid remains unchanged. Programs below G become
    owners and process rows owner+k*G, with the first row peeled to initialize
    partials directly. Later additions preserve the original new+old operand order.
    Final partials, counts and locks are preserved. This selects a permitted
    per-group serialization; it does not authorize floating-point reassociation.

    The proof is bounded to canonical matching IR, including its modulo operation;
    simplifications such as G=1 may leave the original kernel unchanged. It does
    not promise profitability. Functions transformed successfully receive
    tt.experimental_private_reduction_owned = 1 : i32. The pass is opt-in, requires
    target=cuda and one CTA per logical program, and leaves unproved cases unchanged.
  }];
  let options = [
    Option<"target", "target", "std::string", "\"\"", "Target architecture; only cuda is supported">
  ];
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::scf::SCFDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif
