// RUN: triton-opt --split-input-file %s -triton-licm | FileCheck %s

tt.func @hoist_load_without_mask(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // Check if the load is hoisted
  // CHECK-LABEL: hoist_load_without_mask
  // CHECK: %[[TRIP_COUNT_CMP:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SPLAT:.*]] = tt.splat %[[TRIP_COUNT_CMP]]
  // CHECK: %[[LOAD:.*]] = tt.load %[[_:.*]], %[[SPLAT]]
  // CHECK: arith.addf %[[LOAD]], %[[LOAD]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @hoist_two_loads_without_mask(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>, %arg6: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // CHECK-LABEL: hoist_two_loads_without_mask
  // CHECK: %[[TRIP_COUNT_CMP_1:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SPLAT_1:.*]] = tt.splat %[[TRIP_COUNT_CMP_1]]
  // CHECK: %[[LOAD_1:.*]] = tt.load %[[_:.*]], %[[SPLAT_1]]
  // CHECK: %[[TRIP_COUNT_CMP_2:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
  // CHECK: %[[SPLAT_2:.*]] = tt.splat %[[TRIP_COUNT_CMP_2]]
  // CHECK: %[[LOAD_2:.*]] = tt.load %[[_:.*]], %[[SPLAT_2]]
  // CHECK: arith.addf %[[LOAD_1]], %[[LOAD_2]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %1 = scf.for %arg8 = %arg3 to %arg4 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0 : tensor<1024x!tt.ptr<f32>>
    %3 = tt.load %arg6 : tensor<1024x!tt.ptr<f32>>
    %4 = arith.addf %2, %3 : tensor<1024xf32>
    %5 = arith.addf %arg7, %4 : tensor<1024xf32>
    scf.yield %5 : tensor<1024xf32>
  }
  tt.store %arg5, %1 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @hoist_load_with_mask(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // Check if the load is hoisted
  // CHECK-LABEL: hoist_load_with_mask
  // CHECK: %[[MASK:.*]] = arith.cmpi
  // CHECK: %[[TRIP_COUNT_CMP:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SPLAT:.*]] = tt.splat %[[TRIP_COUNT_CMP]]
  // CHECK: %[[AND:.*]] = arith.andi %[[SPLAT]], %[[MASK]]
  // CHECK: %[[LOAD:.*]] = tt.load %[[_:.*]], %[[AND]]
  // CHECK: arith.addf %[[LOAD]], %[[LOAD]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @cannot_hoist_with_print_in_loop(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: arith.addf
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %4 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @cannot_hoist_with_assert_in_loop(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: arith.addf
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %cmp = arith.cmpi sge, %arg4, %arg3 : i32
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    tt.assert %cmp, "cond must be true " : i1
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @cannot_hoist_with_store_in_loop(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>, %tmp: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: arith.addf
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    tt.store %tmp, %4, %0 : tensor<1024x!tt.ptr<f32>>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @hoist_cond_no_hoist_load_from_scf_while(%ptr: tensor<1024x!tt.ptr<f32>>, %arg1: i32, %arg2 : i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  // CHECK-LABEL: hoist_cond_no_hoist_load_from_scf_while
  // CHECK: %[[CST42:.*]] = arith.constant 42
  // CHECK: %[[ADD:.*]] = arith.addi %[[_:.*]], %[[CST42]]
  // CHECK: %[[COND:.*]] = arith.cmpi slt, %[[ADD]], %[[_:.*]]
  // CHECK: scf.while
  // CHECK: do
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: scf.yield
  %1 = scf.while (%arg0 = %cst) : (tensor<1024xf32>) -> (tensor<1024xf32>) {
    %cst_42 = arith.constant 42 : i32
    %add_42 = arith.addi %arg1, %cst_42 : i32
    %2 = arith.cmpi slt, %add_42, %arg2 : i32
    scf.condition(%2) %arg0 : tensor<1024xf32>
  } do {
  ^bb0(%arg0: tensor<1024xf32>):
    %3 = tt.load %ptr : tensor<1024x!tt.ptr<f32>>
    %4 = arith.addf %3, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %ptr, %1 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @do_not_hoist_from_for_with_licm_disabled(%lb: i32, %ub: i32, %step: i32, %a: i32, %b: i32) -> i32 {
  // CHECK-LABEL: do_not_hoist_from_for_with_licm_disabled
  // CHECK-NOT: arith.addi
  // CHECK: scf.for
  // CHECK: arith.addi
  %result = scf.for %i = %lb to %ub step %step iter_args(%value = %a) -> i32 : i32 {
    %sum = arith.addi %a, %b : i32
    scf.yield %sum : i32
  } {llvm.loop_annotation = #llvm.loop_annotation<licm = <disable = true>>}
  tt.return %result : i32
}

// -----

// The contract is about memory contents through all aliases, not just the
// absence of stores through %weights. The entry guard must survive hoisting.
// CHECK-LABEL: @immutable_with_output_store
// CHECK: %[[ENTRY:.*]] = arith.cmpi slt
// CHECK: %[[GUARD:.*]] = arith.andi %[[ENTRY]], %{{.*}}
// CHECK: tt.load %{{.*}}, %[[GUARD]]
// CHECK: scf.for
// CHECK-NOT: tt.load
// CHECK: tt.store
tt.func @immutable_with_output_store(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32, %mask: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights, %mask : !tt.ptr<f32>
    %dst = tt.addptr %out, %i : !tt.ptr<f32>, i32
    tt.store %dst, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @immutable_zero_is_not_a_contract
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
tt.func @immutable_zero_is_not_a_contract(%weights: !tt.ptr<f32> {tt.experimental_immutable = 0 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @immutable_volatile_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load %{{.*}} {isVolatile = true}
tt.func @immutable_volatile_stays(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights {isVolatile = true} : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @immutable_constant_entry_while
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: tt.load %{{.*}}, %[[TRUE]]
// CHECK: scf.while
// CHECK: do
// CHECK-NOT: tt.load
// CHECK: tt.store
tt.func @immutable_constant_entry_while(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %end = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c8 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %value = tt.load %weights : !tt.ptr<f32>
    %dst = tt.addptr %out, %i : !tt.ptr<f32>, i32
    tt.store %dst, %value : !tt.ptr<f32>
    %next = arith.addi %i, %c1 : i32
    scf.yield %next : i32
  }
  tt.return
}

// -----

// CHECK-LABEL: @immutable_unknown_entry_while
// CHECK-NOT: tt.load
// CHECK: scf.while
// CHECK: do
// CHECK: tt.load
tt.func @immutable_unknown_entry_while(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %end = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %n : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    %next = arith.addi %i, %c1 : i32
    scf.yield %next : i32
  }
  tt.return
}

// -----

// CHECK-LABEL: @immutable_zero_trip_while
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: tt.load %{{.*}}, %[[FALSE]]
// CHECK: scf.while
// CHECK: do
// CHECK-NOT: tt.load
tt.func @immutable_zero_trip_while(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %end = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c0 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    %next = arith.addi %i, %c1 : i32
    scf.yield %next : i32
  }
  tt.return
}

// -----

// CHECK-LABEL: @immutable_derived_tensor_pointer
// CHECK: tt.splat
// CHECK: tt.addptr
// CHECK: tt.load
// CHECK: scf.for
// CHECK-NOT: tt.load
// CHECK: tt.store
tt.func @immutable_derived_tensor_pointer(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: tensor<32x!tt.ptr<f32>>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %base = tt.splat %weights : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %value = tt.load %ptrs : tensor<32x!tt.ptr<f32>>
    tt.store %out, %value : tensor<32x!tt.ptr<f32>>
  }
  tt.return
}

// -----

// A contract on values must not speculate an address ahead of a device assert.
// CHECK-LABEL: @immutable_assert_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.assert
// CHECK: tt.load
tt.func @immutable_assert_stays(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32, %valid: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    tt.assert %valid, "address must be valid" : i1
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Unknown pointer provenance does not inherit the contract.
// CHECK-LABEL: @immutable_select_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
tt.func @immutable_select_stays(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %other: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %choose: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %ptr = arith.select %choose, %weights, %other : !tt.ptr<f32>
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %ptr : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// The caller provides disjoint access footprints. The pass proves the input
// is not modified; the caller asserts neither initialization nor immutability.
// CHECK-LABEL: @alias_groups_derive_invariant
// CHECK: %[[ENTRY:.*]] = arith.cmpi slt
// CHECK: %[[GUARD:.*]] = arith.andi %[[ENTRY]], %{{.*}}
// CHECK: tt.load %{{.*}}, %[[GUARD]]
// CHECK: scf.for
// CHECK-NOT: tt.load
// CHECK: tt.store
tt.func @alias_groups_derive_invariant(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32, %mask: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights, %mask : !tt.ptr<f32>
    %dst = tt.addptr %out, %i : !tt.ptr<f32>, i32
    tt.store %dst, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Read-only inputs may alias each other; only overlapping writes prevent reuse.
// CHECK-LABEL: @alias_groups_readonly_aliases
// CHECK: tt.load
// CHECK: tt.load
// CHECK: scf.for
// CHECK-NOT: tt.load
// CHECK: tt.store
tt.func @alias_groups_readonly_aliases(%a: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %b: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %x = tt.load %a : !tt.ptr<f32>
    %y = tt.load %b : !tt.ptr<f32>
    %sum = arith.addf %x, %y : f32
    tt.store %out, %sum : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @alias_groups_overlapping_write_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
tt.func @alias_groups_overlapping_write_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %alias: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %alias, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// An unannotated writer may overlap every annotated input region.
// CHECK-LABEL: @alias_groups_unknown_writer_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
tt.func @alias_groups_unknown_writer_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// A missing group on the read argument also supplies no disjointness fact.
// CHECK-LABEL: @alias_groups_unknown_reader_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
tt.func @alias_groups_unknown_reader_stays(%weights: !tt.ptr<f32>, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// The summary covers the function, including writes outside the candidate loop.
// CHECK-LABEL: @alias_groups_write_before_loop_stays
// CHECK: tt.store
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
tt.func @alias_groups_write_before_loop_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  tt.store %weights, %zero : !tt.ptr<f32>
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// A negative group ID supplies no disjointness fact.
// CHECK-LABEL: @alias_groups_negative_id_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
tt.func @alias_groups_negative_id_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = -1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @alias_groups_unknown_origin_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
tt.func @alias_groups_unknown_origin_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out1: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %out2: !tt.ptr<f32> {tt.experimental_alias_group = 2 : i32}, %n: i32, %choose: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %out = arith.select %choose, %out1, %out2 : !tt.ptr<f32>
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

tt.func private @unknown_effect(!tt.ptr<f32>)

// CHECK-LABEL: @alias_groups_unknown_call_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.call
// CHECK: tt.load
tt.func @alias_groups_unknown_call_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    tt.call @unknown_effect(%out) : (!tt.ptr<f32>) -> ()
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// The new inference path preserves the same zero-trip guard as explicit facts.
// CHECK-LABEL: @alias_groups_zero_trip_while
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: tt.load %{{.*}}, %[[FALSE]]
// CHECK: scf.while
// CHECK: do
// CHECK-NOT: tt.load
// CHECK: tt.store
tt.func @alias_groups_zero_trip_while(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %end = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c0 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    %next = arith.addi %i, %c1 : i32
    scf.yield %next : i32
  }
  tt.return
}

// -----

// CHECK-LABEL: @alias_groups_volatile_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load %{{.*}} {isVolatile = true}
tt.func @alias_groups_volatile_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights {isVolatile = true} : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Calls and publication/ordering effects are outside the bounded effect model.
// CHECK-LABEL: @alias_groups_atomic_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.atomic_rmw
// CHECK: tt.load
tt.func @alias_groups_atomic_stays(%weights: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<i32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %old = tt.atomic_rmw add, relaxed, gpu, %out, %c1 : (!tt.ptr<i32>, i32) -> i32
    %value = tt.load %weights : !tt.ptr<i32>
    tt.store %out, %value : !tt.ptr<i32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @alias_groups_assert_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.assert
// CHECK: tt.load
tt.func @alias_groups_assert_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32, %valid: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    tt.assert %valid, "address must be valid" : i1
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// A constant outer condition alone cannot bypass a nonterminating before region.
// CHECK-LABEL: @alias_groups_nonterminating_before_stays
// CHECK-NOT: tt.load
// CHECK: scf.while
// CHECK: scf.while
// CHECK: tt.load
tt.func @alias_groups_nonterminating_before_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) {
  %true = arith.constant true
  scf.while : () -> () {
    scf.while : () -> () {
      scf.condition(%true)
    } do {
      scf.yield
    }
    scf.condition(%true)
  } do {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    scf.yield
  }
  tt.return
}

// -----

// A positive for trip count does not make a load after a nested loop reachable.
// CHECK-LABEL: @alias_groups_nonterminating_for_prefix_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: scf.while
// CHECK: tt.load
tt.func @alias_groups_nonterminating_for_prefix_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    scf.while : () -> () {
      scf.condition(%true)
    } do {
      scf.yield
    }
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// The same prefix restriction applies to the after-region of a while loop.
// CHECK-LABEL: @alias_groups_nonterminating_after_prefix_stays
// CHECK-NOT: tt.load
// CHECK: scf.while
// CHECK: scf.while
// CHECK: tt.load
tt.func @alias_groups_nonterminating_after_prefix_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) {
  %true = arith.constant true
  scf.while : () -> () {
    scf.condition(%true)
  } do {
    scf.while : () -> () {
      scf.condition(%true)
    } do {
      scf.yield
    }
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    scf.yield
  }
  tt.return
}

// -----

// A disjoint parameter can hoist past advancing read/write streams. The stream
// load itself remains in the loop even though its allocation root is known.
// CHECK-LABEL: @alias_groups_pointer_streams
// CHECK: %[[ENTRY:.*]] = arith.cmpi slt
// CHECK: %[[WEIGHT:.*]] = tt.load %{{.*}}, %[[ENTRY]]
// CHECK: scf.for
// CHECK: %[[ELEMENT:.*]] = tt.load
// CHECK: arith.addf %[[ELEMENT]], %[[WEIGHT]]
// CHECK: tt.store
// CHECK: tt.addptr
// CHECK: scf.yield
tt.func @alias_groups_pointer_streams(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %input: !tt.ptr<f32> {tt.experimental_alias_group = 2 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %ends:2 = scf.for %i = %c0 to %n step %c1 iter_args(%r = %input, %w = %out) -> (!tt.ptr<f32>, !tt.ptr<f32>) : i32 {
    %element = tt.load %r : !tt.ptr<f32>
    %weight = tt.load %weights : !tt.ptr<f32>
    %value = arith.addf %element, %weight : f32
    tt.store %w, %value : !tt.ptr<f32>
    %rn = tt.addptr %r, %c1 : !tt.ptr<f32>, i32
    %wn = tt.addptr %w, %c1 : !tt.ptr<f32>, i32
    scf.yield %rn, %wn : !tt.ptr<f32>, !tt.ptr<f32>
  }
  tt.return
}

// -----

// Same-root proof does not override an actual alias with the written stream.
// CHECK-LABEL: @alias_groups_pointer_stream_alias_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
// CHECK: tt.store
tt.func @alias_groups_pointer_stream_alias_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %end = scf.for %i = %c0 to %n step %c1 iter_args(%w = %out) -> (!tt.ptr<f32>) : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %w, %value : !tt.ptr<f32>
    %next = tt.addptr %w, %c1 : !tt.ptr<f32>, i32
    scf.yield %next : !tt.ptr<f32>
  }
  tt.return
}

// -----

// The initial pointer's disjointness is insufficient when a later iteration
// writes to a pointer derived from the parameter allocation.
// CHECK-LABEL: @alias_groups_pointer_root_change_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
// CHECK: tt.store
tt.func @alias_groups_pointer_root_change_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %end = scf.for %i = %c0 to %n step %c1 iter_args(%w = %out) -> (!tt.ptr<f32>) : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %w, %value : !tt.ptr<f32>
    %changed = tt.addptr %weights, %c1 : !tt.ptr<f32>, i32
    scf.yield %changed : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Cross-carried pointers can exchange allocation roots. Do not confuse the
// presence of a recurrence with preservation of each individual root.
// CHECK-LABEL: @alias_groups_pointer_cross_carry_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
// CHECK: tt.store
tt.func @alias_groups_pointer_cross_carry_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %ends:2 = scf.for %i = %c0 to %n step %c1 iter_args(%w = %out, %other = %weights) -> (!tt.ptr<f32>, !tt.ptr<f32>) : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %w, %value : !tt.ptr<f32>
    scf.yield %other, %w : !tt.ptr<f32>, !tt.ptr<f32>
  }
  tt.return
}

// -----

// Conditional root changes remain unsupported, even if an initial pointer has
// a declared disjoint region.
// CHECK-LABEL: @alias_groups_pointer_select_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.load
// CHECK: tt.store
tt.func @alias_groups_pointer_select_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32, %cond: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %end = scf.for %i = %c0 to %n step %c1 iter_args(%w = %out) -> (!tt.ptr<f32>) : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %w, %value : !tt.ptr<f32>
    %next = tt.addptr %w, %c1 : !tt.ptr<f32>, i32
    %selected = arith.select %cond, %next, %weights : !tt.ptr<f32>
    scf.yield %selected : !tt.ptr<f32>
  }
  tt.return
}

// -----

// A validated loop result keeps its root for subsequent memory-effect analysis.
// CHECK-LABEL: @alias_groups_pointer_result
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: arith.cmpi slt
// CHECK: tt.load
// CHECK: scf.for
// CHECK-NOT: tt.load
// CHECK: tt.store
tt.func @alias_groups_pointer_result(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %end = scf.for %i = %c0 to %n step %c1 iter_args(%w = %out) -> (!tt.ptr<f32>) : i32 {
    %next = tt.addptr %w, %c1 : !tt.ptr<f32>, i32
    scf.yield %next : !tt.ptr<f32>
  }
  scf.for %j = %c0 to %n step %c1 : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    %dst = tt.addptr %end, %j : !tt.ptr<f32>, i32
    tt.store %dst, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// A nested result may advance the outer pointer, provided each inner backedge
// derives only from its own carried pointer and its init derives from the outer.
// CHECK-LABEL: @alias_groups_nested_pointer_stream
// CHECK: tt.load
// CHECK: scf.for
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK-NOT: tt.load
// CHECK: tt.store
tt.func @alias_groups_nested_pointer_stream(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %end = scf.for %i = %c0 to %n step %c1 iter_args(%outer = %out) -> (!tt.ptr<f32>) : i32 {
    %inner_end = scf.for %j = %c0 to %n step %c1 iter_args(%inner = %outer) -> (!tt.ptr<f32>) : i32 {
      %value = tt.load %weights : !tt.ptr<f32>
      tt.store %inner, %value : !tt.ptr<f32>
      %next = tt.addptr %inner, %c1 : !tt.ptr<f32>, i32
      scf.yield %next : !tt.ptr<f32>
    }
    scf.yield %inner_end : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Signed -1 < 1 would incorrectly enable a load on this unsigned zero-trip path.
// CHECK-LABEL: @alias_groups_unsigned_entry_guard
// CHECK: %[[ENTRY:.*]] = arith.cmpi ult
// CHECK: tt.load %{{.*}}, %[[ENTRY]]
// CHECK: scf.for unsigned
// CHECK-NOT: tt.load
// CHECK: tt.store
tt.func @alias_groups_unsigned_entry_guard(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) {
  %cneg1 = arith.constant -1 : i32
  %c1 = arith.constant 1 : i32
  scf.for unsigned %i = %cneg1 to %c1 step %c1 : i32 {
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Initialized immutable global memory is independent of the CTA publication
// protocol. Keep both barriers and combine the original mask with loop entry.
// CHECK-LABEL: @immutable_cta_barrier_tensor_mask
// CHECK: %[[ENTRY:.*]] = arith.cmpi slt
// CHECK: %[[SPLAT:.*]] = tt.splat %[[ENTRY]]
// CHECK: %[[MASK:.*]] = arith.andi %[[SPLAT]], %{{.*}}
// CHECK: %[[LOAD:.*]] = tt.load %{{.*}}, %[[MASK]]
// CHECK: arith.addf %[[LOAD]], %[[LOAD]]
// CHECK: scf.for
// CHECK: ttg.barrier all
// CHECK-NOT: tt.load
// CHECK: tt.store
// CHECK: ttg.barrier all
// CHECK-NOT: tt.load
// CHECK: tt.return
tt.func @immutable_cta_barrier_tensor_mask(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: tensor<32x!tt.ptr<f32>>, %n: i32, %mask: tensor<32xi1>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %offset = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %base = tt.splat %weights : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %offset : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  scf.for %i = %c0 to %n step %c1 : i32 {
    ttg.barrier all
    %value = tt.load %ptr, %mask : tensor<32x!tt.ptr<f32>>
    %twice = arith.addf %value, %value : tensor<32xf32>
    tt.store %out, %twice : tensor<32x!tt.ptr<f32>>
    ttg.barrier all
  }
  tt.return
}

// -----

// Even an immutable input must not be accessed on a zero-trip path.
// CHECK-LABEL: @immutable_cta_barrier_zero_trip_while
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: tt.load %{{.*}}, %[[FALSE]]
// CHECK: scf.while
// CHECK: do
// CHECK: ttg.barrier all
// CHECK-NOT: tt.load
// CHECK: tt.store
// CHECK: ttg.barrier all
tt.func @immutable_cta_barrier_zero_trip_while(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %end = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c0 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    ttg.barrier all
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    ttg.barrier all
    %next = arith.addi %i, %c1 : i32
    scf.yield %next : i32
  }
  tt.return
}

// -----

// Disjointness alone does not promise initialized-at-entry contents.
// CHECK-LABEL: @alias_only_cta_barrier_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: ttg.barrier all
// CHECK: tt.load
tt.func @alias_only_cta_barrier_stays(%weights: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    ttg.barrier all
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @mutable_cta_barrier_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: ttg.barrier all
// CHECK: tt.load
tt.func @mutable_cta_barrier_stays(%weights: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    ttg.barrier all
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// The new exception is restricted to global pointer arguments.
// CHECK-LABEL: @immutable_nonglobal_pointer_barrier_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: ttg.barrier all
// CHECK: tt.load
tt.func @immutable_nonglobal_pointer_barrier_stays(%weights: !tt.ptr<f32, "constant"> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    ttg.barrier all
    %value = tt.load %weights : !tt.ptr<f32, "constant">
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @immutable_warp_barrier_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: ttg.barrier warp local
// CHECK: tt.load
tt.func @immutable_warp_barrier_stays(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    ttg.barrier warp local
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @immutable_cta_barrier_volatile_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: ttg.barrier all
// CHECK: tt.load %{{.*}} {isVolatile = true}
tt.func @immutable_cta_barrier_volatile_stays(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    ttg.barrier all
    %value = tt.load %weights {isVolatile = true} : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Async completion protocols remain outside the ordinary-effect model.
// CHECK-LABEL: @immutable_cta_barrier_async_wait_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: ttg.async_wait
// CHECK: ttg.barrier all
// CHECK: tt.load
tt.func @immutable_cta_barrier_async_wait_stays(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    ttg.async_wait {num = 0 : i32}
    ttg.barrier all
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

tt.func private @barrier_unknown_effect(!tt.ptr<f32>)

// CHECK-LABEL: @immutable_cta_barrier_call_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.call
// CHECK: ttg.barrier all
// CHECK: tt.load
tt.func @immutable_cta_barrier_call_stays(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    tt.call @barrier_unknown_effect(%out) : (!tt.ptr<f32>) -> ()
    ttg.barrier all
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @immutable_cta_barrier_assert_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: tt.assert
// CHECK: ttg.barrier all
// CHECK: tt.load
tt.func @immutable_cta_barrier_assert_stays(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32, %valid: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    tt.assert %valid, "read must be reachable" : i1
    ttg.barrier all
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}

// -----

// A barrier does not make the later load reachable after a divergent region.
// CHECK-LABEL: @immutable_cta_barrier_nonreached_prefix_stays
// CHECK-NOT: tt.load
// CHECK: scf.for
// CHECK: scf.while
// CHECK: ttg.barrier all
// CHECK: tt.load
tt.func @immutable_cta_barrier_nonreached_prefix_stays(%weights: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %out: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  scf.for %i = %c0 to %n step %c1 : i32 {
    scf.while : () -> () {
      scf.condition(%true)
    } do {
      scf.yield
    }
    ttg.barrier all
    %value = tt.load %weights : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
  }
  tt.return
}
