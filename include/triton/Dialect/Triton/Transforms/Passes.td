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
    not eligible for this extension.

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
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopAwareCSE : Pass<"triton-loop-aware-cse", "mlir::ModuleOp"> {
  let summary = "CSE within loop bodies";

  let description = [{
    The `triton-loop-aware-cse` pass performs recursive common subexpression
    elimination within loop bodies. Unlike regular CSE, which is a single-pass
    greedy algorithm, this pass can recursively eliminate loop iteration
    arguments and subcomputations that always have the same value.
  }];
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
    until invocation completion, and no synchronization/publication role for the
    storage. It does not assert ownership by a program or address injectivity.
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
    must be scalar and unmasked. Loops, regions, calls, volatile accesses, inline
    assembly, unknown effects, additional writes, live atomic old values, and
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

#endif
