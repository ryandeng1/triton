// RUN: triton-opt %s -triton-loop-aware-cse -allow-unregistered-dialect | FileCheck %s
// RUN: triton-opt %s -triton-loop-aware-cse="enable-contract-load-reuse=false" -allow-unregistered-dialect | FileCheck %s --check-prefix=NO-CONTRACT-REUSE

// CHECK-LABEL: @loop_buffer_phase_args
tt.func @loop_buffer_phase_args(%arg0: i32) {
  %c2_i32 = arith.constant 2 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: [[LOOP_RES:%.*]]:3 = scf.for {{.*}} iter_args
  // CHECK-SAME: [[M2_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[M2_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[M1_PHASE:%arg[0-9]+]] = %c0_i32
  %0:10 = scf.for %arg1 = %c0_i32 to %arg0 step %c128_i32 iter_args(%arg2 = %c0_i32, %arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %c0_i32, %arg6 = %c0_i32, %arg7 = %c0_i32, %arg8 = %c0_i32, %arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %c0_i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
    %1 = arith.subi %arg0, %c128_i32 : i32
    %2 = arith.cmpi slt, %arg1, %1 : i32
    // CHECK: [[M1_PHASE_INCR:%.*]] = arith.xori [[M1_PHASE]], %c1_i32
    %3 = arith.xori %arg7, %c1_i32 : i32
    // CHECK: "index_phase_use"([[M2_INDEX]], [[M2_PHASE]], [[M1_PHASE_INCR]], [[M1_PHASE]])
    "index_phase_use"(%arg4, %arg5, %3, %arg8) : (i32, i32, i32, i32) -> ()
    %4 = arith.addi %arg4, %c1_i32 : i32
    %5 = arith.xori %arg5, %c1_i32 : i32
    %6 = arith.cmpi eq, %4, %c2_i32 : i32
    // CHECK: [[M2_INDEX_INCR:%.*]] = arith.select %{{.*}}, %c0_i32
    // CHECK-NEXT: [[M2_PHASE_INCR:%.*]] = arith.select %{{.*}}, %{{.*}}, [[M2_PHASE]]
    // CHECK-NOT: arith.select
    %7 = arith.select %6, %c0_i32, %4 : i32
    %8 = arith.select %6, %5, %arg5 : i32
    %9 = arith.xori %arg8, %c1_i32 : i32
    %10 = arith.xori %arg11, %c1_i32 : i32
    %11 = arith.xori %arg6, %c1_i32 : i32
    %12 = arith.addi %arg2, %c1_i32 : i32
    %13 = arith.xori %arg3, %c1_i32 : i32
    %14 = arith.cmpi eq, %12, %c2_i32 : i32
    %15 = arith.select %14, %c0_i32, %12 : i32
    %16 = arith.select %14, %13, %arg3 : i32
    // CHECK: "index_phase_use"([[M2_INDEX_INCR]], [[M2_PHASE_INCR]], [[M1_PHASE_INCR]],
    "index_phase_use"(%15, %16, %11, %2) : (i32, i32, i32, i1) -> ()
    %17 = arith.xori %arg10, %c1_i32 : i32
    // CHECK: "index_phase_use"([[M1_PHASE_INCR]], [[M1_PHASE]])
    "index_phase_use"(%17, %arg11) : (i32, i32) -> ()
    %18 = arith.xori %arg9, %c1_i32 : i32
    // CHECK: "index_phase_use"([[M1_PHASE_INCR]], [[M1_PHASE]])
    "index_phase_use"(%17, %arg11) : (i32, i32) -> ()
    scf.yield %15, %16, %7, %8, %11, %3, %9, %18, %17, %10 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
  }
  tt.return
}

// CHECK-LABEL: @invalid_cache_test
tt.func public @invalid_cache_test(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %0:4 = scf.for
  %0:4 = scf.for %arg2 = %c0_i32 to %arg0 step %arg1 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %c0_i32, %arg6 = %c0_i32) -> (i32, i32, i32, i32)  : i32 {

    %1 = arith.addi %arg5, %c1_i32 : i32
    %2 = arith.xori %arg6, %c1_i32 : i32
    %3 = arith.cmpi eq, %1, %c3_i32 : i32
    %4 = arith.select %3, %2, %arg6 : i32
    %5 = arith.select %3, %c1_i32, %1 : i32

    %6 = arith.addi %arg3, %c1_i32 : i32
    %7 = arith.xori %arg4, %c1_i32 : i32
    %8 = arith.cmpi eq, %6, %c3_i32 : i32
    %9 = arith.select %8, %c0_i32, %6 : i32
    %10 = arith.select %8, %7, %arg4 : i32

    scf.yield %9, %10, %5, %4 : i32, i32, i32, i32
  }
  tt.return %0#1, %0#3 : i32, i32
}

// CHECK-LABEL: @multiple_op_results
tt.func @multiple_op_results(%arg0: i32) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: %0:2 = scf.for
  %0:2 = scf.for %i = %c0_i32 to %arg0 step %c1_i32 iter_args(%a = %c0_i32, %b = %c0_i32) -> (i32, i32) : i32 {
    // CHECK-NEXT: %1:2 = {{.*}} %arg2, %arg3
    %1:2 = tt.elementwise_inline_asm "asm" {constraints = "=r,=r,r,r", pure = true, packed_element = 1 : i32} %a, %b : i32, i32 -> i32, i32
    // CHECK-NEXT: yield %1#0, %1#1 : i32, i32
    scf.yield %1#0, %1#1 : i32, i32
  }
  tt.return %0#0, %0#1 : i32, i32
}

// Contract-based load reuse supplements ordinary CSE. The store in every
// straight-line case prevents unannotated MLIR CSE from proving equivalence.

tt.func private @load_reuse_unknown_callee(!tt.ptr<f32>)

// Disjoint invocation footprints permit reuse across an ordinary store.
// NO-CONTRACT-REUSE-LABEL: @contract_load_reuse_alias(
// NO-CONTRACT-REUSE: tt.load
// NO-CONTRACT-REUSE: tt.store
// NO-CONTRACT-REUSE: tt.load
// NO-CONTRACT-REUSE: tt.return
// CHECK-LABEL: @contract_load_reuse_alias
// CHECK: %[[REUSED:.*]] = tt.load
// CHECK: tt.store
// CHECK-NOT: tt.load
// CHECK: tt.return %[[REUSED]], %[[REUSED]]
tt.func @contract_load_reuse_alias(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// An explicit immutable input supplies the stronger stable-content fact.
// NO-CONTRACT-REUSE-LABEL: @contract_load_reuse_immutable(
// NO-CONTRACT-REUSE: tt.load
// NO-CONTRACT-REUSE: tt.store
// NO-CONTRACT-REUSE: tt.load
// NO-CONTRACT-REUSE: tt.return
// CHECK-LABEL: @contract_load_reuse_immutable
// CHECK: %[[REUSED:.*]] = tt.load
// CHECK: tt.store
// CHECK-NOT: tt.load
// CHECK: tt.return %[[REUSED]], %[[REUSED]]
tt.func @contract_load_reuse_immutable(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %q: !tt.ptr<f32>) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Distinct SSA pointer arguments are not a disjointness proof.
// CHECK-LABEL: @contract_load_reuse_missing_contract
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_missing_contract(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Equal alias groups may refer to the same bytes.
// CHECK-LABEL: @contract_load_reuse_same_group
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_same_group(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Negative group IDs carry no disjointness fact.
// CHECK-LABEL: @contract_load_reuse_negative_group
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_negative_group(%p: !tt.ptr<f32> {tt.experimental_alias_group = -1 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Only i32 group attributes define the experimental contract.
// CHECK-LABEL: @contract_load_reuse_malformed_group
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_malformed_group(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i64}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Volatile reads remain distinct even with immutable contents.
// CHECK-LABEL: @contract_load_reuse_volatile
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_volatile(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %q: !tt.ptr<f32>) -> (f32, f32) {
  %first = tt.load %p {isVolatile = true} : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p {isVolatile = true} : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Only the explicitly immutable global case may cross a CTA barrier.
// CHECK-LABEL: @contract_load_reuse_immutable_cta
// CHECK: %[[FIRST:.*]] = tt.load
// CHECK: ttg.barrier all
// CHECK-NOT: tt.load
// CHECK: tt.return %[[FIRST]], %[[FIRST]]
tt.func @contract_load_reuse_immutable_cta(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %q: !tt.ptr<f32>) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  ttg.barrier all
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Only the explicitly immutable global case may cross a CTA barrier.
// CHECK-LABEL: @contract_load_reuse_alias_cta
// CHECK: %[[FIRST:.*]] = tt.load
// CHECK: ttg.barrier all
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_alias_cta(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  ttg.barrier all
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Only the explicitly immutable global case may cross a CTA barrier.
// CHECK-LABEL: @contract_load_reuse_immutable_warp
// CHECK: %[[FIRST:.*]] = tt.load
// CHECK: ttg.barrier warp local
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_immutable_warp(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %q: !tt.ptr<f32>) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  ttg.barrier warp local
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Unknown effects invalidate the bounded interval proof.
// CHECK-LABEL: @contract_load_reuse_unknown_effect
// CHECK: tt.load
// CHECK: "opaque_effect"
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_unknown_effect(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %q: !tt.ptr<f32>) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  "opaque_effect"() : () -> ()
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Asynchronous completion needs a separate protocol proof.
// CHECK-LABEL: @contract_load_reuse_async
// CHECK: tt.load
// CHECK: ttg.async_wait
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_async(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %q: !tt.ptr<f32>) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  ttg.async_wait {num = 0 : i32}
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Calls are not summarized as ordinary disjoint accesses.
// CHECK-LABEL: @contract_load_reuse_call
// CHECK: tt.load
// CHECK: tt.call
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_call(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %q: !tt.ptr<f32>) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  tt.call @load_reuse_unknown_callee(%q) : (!tt.ptr<f32>) -> ()
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// A common root does not establish equal addresses.
// CHECK-LABEL: @contract_load_reuse_different_pointer
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_different_pointer(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %offset: i32) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %shifted = tt.addptr %p, %offset : !tt.ptr<f32>, i32
  %second = tt.load %shifted : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// A narrower first mask cannot supply a later read of other elements.
// CHECK-LABEL: @contract_load_reuse_different_mask
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_different_mask(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %left: i1, %right: i1) -> (f32, f32) {
  %first = tt.load %p, %left : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p, %right : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Inactive values are part of load equivalence.
// CHECK-LABEL: @contract_load_reuse_different_other
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_different_other(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %mask: i1) -> (f32, f32) {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %first = tt.load %p, %mask, %zero : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p, %mask, %one : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Cache policy differences remain intact.
// CHECK-LABEL: @contract_load_reuse_different_cache
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_different_cache(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// Unproved selected pointers cannot inherit an argument contract.
// CHECK-LABEL: @contract_load_reuse_unknown_root
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_unknown_root(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %other: !tt.ptr<f32>, %choose: i1) -> (f32, f32) {
  %selected = arith.select %choose, %p, %other : !tt.ptr<f32>
  %first = tt.load %selected : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %selected : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// A recursively summarized branch may write a disjoint region.
// CHECK-LABEL: @contract_load_reuse_branch_store
// CHECK: %[[REUSED:.*]] = tt.load
// CHECK: tt.store
// CHECK-NOT: tt.load
// CHECK: tt.return %[[REUSED]], %[[REUSED]]
tt.func @contract_load_reuse_branch_store(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %condition: i1) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  scf.if %condition {
    tt.store %q, %first : !tt.ptr<f32>
  }
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// A write in either branch must invalidate the whole-function no-clobber proof.
// CHECK-LABEL: @contract_load_reuse_branch_clobber
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_branch_clobber(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %condition: i1) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32>
  scf.if %condition {
    tt.store %p, %first : !tt.ptr<f32>
  }
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// A loop containing only disjoint ordinary stores does not clobber the input.
// CHECK-LABEL: @contract_load_reuse_loop_store
// CHECK: %[[REUSED:.*]] = tt.load
// CHECK: tt.store
// CHECK-NOT: tt.load
// CHECK: tt.return %[[REUSED]], %[[REUSED]]
tt.func @contract_load_reuse_loop_store(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) -> (f32, f32) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %first = tt.load %p : !tt.ptr<f32>
  scf.for %i = %zero to %n step %one : i32 {
    %next = tt.addptr %q, %i : !tt.ptr<f32>, i32
    tt.store %next, %first : !tt.ptr<f32>
  }
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// CHECK-LABEL: @contract_load_reuse_sibling_blocks
// CHECK: scf.if
// CHECK: tt.load
// CHECK: tt.store
// CHECK: else
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.return
tt.func @contract_load_reuse_sibling_blocks(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %condition: i1) {
  scf.if %condition {
    %left = tt.load %p : !tt.ptr<f32>
    tt.store %q, %left : !tt.ptr<f32>
  } else {
    %right = tt.load %p : !tt.ptr<f32>
    tt.store %q, %right : !tt.ptr<f32>
  }
  tt.return
}

// CHECK-LABEL: @contract_load_reuse_zero_trip_boundary
// CHECK: scf.for
// CHECK: tt.load
// CHECK: tt.store
// CHECK: }
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.return
tt.func @contract_load_reuse_zero_trip_boundary(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  scf.for %i = %zero to %n step %one : i32 {
    %inside = tt.load %p : !tt.ptr<f32>
    tt.store %q, %inside : !tt.ptr<f32>
  }
  %outside = tt.load %p : !tt.ptr<f32>
  tt.store %q, %outside : !tt.ptr<f32>
  tt.return
}

// A later masked load with no `other` has unspecified inactive lanes. Its mask
// may be stronger when every active lane was already read by the earlier load.
// NO-CONTRACT-REUSE-LABEL: @contract_load_reuse_stronger_mask(
// NO-CONTRACT-REUSE: tt.load
// NO-CONTRACT-REUSE: tt.store
// NO-CONTRACT-REUSE: tt.load
// NO-CONTRACT-REUSE: tt.return
// CHECK-LABEL: @contract_load_reuse_stronger_mask
// CHECK: %[[FIRST:.*]] = tt.load
// CHECK: tt.store
// CHECK-NOT: tt.load
// CHECK: tt.return %[[FIRST]], %[[FIRST]]
tt.func @contract_load_reuse_stronger_mask(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %mask: i1, %extra: i1) -> (f32, f32) {
  %first = tt.load %p, %mask : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %narrow = arith.andi %mask, %extra : i1
  %second = tt.load %p, %narrow : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// CHECK-LABEL: @contract_load_reuse_weaker_mask_stays
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_weaker_mask_stays(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %mask: i1, %extra: i1) -> (f32, f32) {
  %narrow = arith.andi %mask, %extra : i1
  %first = tt.load %p, %narrow : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p, %mask : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// An explicit later `other` must still be produced in its newly inactive lanes.
// CHECK-LABEL: @contract_load_reuse_stronger_mask_explicit_other_stays
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_stronger_mask_explicit_other_stays(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %mask: i1, %extra: i1) -> (f32, f32) {
  %zero = arith.constant 0.0 : f32
  %first = tt.load %p, %mask, %zero : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %narrow = arith.andi %mask, %extra : i1
  %second = tt.load %p, %narrow, %zero : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// CHECK-LABEL: @contract_load_reuse_matching_other
// CHECK: %[[FIRST:.*]] = tt.load
// CHECK: tt.store
// CHECK-NOT: tt.load
// CHECK: tt.return %[[FIRST]], %[[FIRST]]
tt.func @contract_load_reuse_matching_other(%p: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %q: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %mask: i1) -> (f32, f32) {
  %zero = arith.constant 0.0 : f32
  %first = tt.load %p, %mask, %zero : !tt.ptr<f32>
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p, %mask, %zero : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}

// The new contract consumer is limited to ordinary global pointer accesses.
// CHECK-LABEL: @contract_load_reuse_nonglobal_stays
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_nonglobal_stays(%p: !tt.ptr<f32, "constant"> {tt.experimental_immutable = 1 : i32}, %q: !tt.ptr<f32>) -> (f32, f32) {
  %first = tt.load %p : !tt.ptr<f32, "constant">
  tt.store %q, %first : !tt.ptr<f32>
  %second = tt.load %p : !tt.ptr<f32, "constant">
  tt.return %first, %second : f32, f32
}

// Atomic synchronization is outside the stable ordinary-access interval proof.
// CHECK-LABEL: @contract_load_reuse_atomic_stays
// CHECK: tt.load
// CHECK: tt.atomic_rmw
// CHECK: tt.load
// CHECK: tt.return
tt.func @contract_load_reuse_atomic_stays(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %q: !tt.ptr<f32>) -> (f32, f32) {
  %true = arith.constant true
  %first = tt.load %p : !tt.ptr<f32>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %q, %first, %true : (!tt.ptr<f32>, f32, i1) -> f32
  %second = tt.load %p : !tt.ptr<f32>
  tt.return %first, %second : f32, f32
}
