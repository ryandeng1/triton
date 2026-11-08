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

#endif
