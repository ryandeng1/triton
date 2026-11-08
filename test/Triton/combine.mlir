// RUN: triton-opt %s -canonicalize -triton-combine --split-input-file | FileCheck %s
// RUN: triton-opt %s --triton-specialize-indexed-updates="target=cuda" --split-input-file | FileCheck %s --check-prefix=INDEXED
// RUN: triton-opt %s --triton-specialize-indexed-updates --split-input-file | FileCheck %s --check-prefix=NO-TARGET
// RUN: triton-opt %s --triton-specialize-indexed-updates="target=hip" --split-input-file | FileCheck %s --check-prefix=OTHER-TARGET
// RUN: triton-opt %s --triton-relax-private-atomics="target=cuda" --split-input-file | FileCheck %s --check-prefix=PRIVATE
// RUN: triton-opt %s --triton-relax-private-atomics --split-input-file | FileCheck %s --check-prefix=PRIVATE-NO-TARGET
// RUN: triton-opt %s --triton-relax-private-atomics="target=hip" --split-input-file | FileCheck %s --check-prefix=PRIVATE-OTHER-TARGET

// We don't combine if the dot result is used by more than one op.
// CHECK-LABEL: @test_combine_dot_add_invalid_pattern
tt.func @test_combine_dot_add_invalid_pattern() -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    // CHECK-DAG: %[[d:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[e:.*]] = arith.constant dense<4.000000e+00> : tensor<128x128xf32>
    %a = arith.constant dense<1.0> : tensor<128x128xf32>
    %b = arith.constant dense<2.0> : tensor<128x128xf32>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d = arith.constant dense<3.0> : tensor<128x128xf32>
    %e = arith.constant dense<4.0> : tensor<128x128xf32>

    %dot_out = tt.dot %a, %b, %zero : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>

    // CHECK: arith.addf %{{.*}}, %[[d]] : tensor<128x128xf32>
    %res0 = arith.addf %dot_out, %d : tensor<128x128xf32>

    // CHECK-NEXT: arith.addf %{{.*}}, %[[e]]  : tensor<128x128xf32>
    %res1 = arith.addf %dot_out, %e : tensor<128x128xf32>

    tt.return %res0, %res1 : tensor<128x128xf32>, tensor<128x128xf32>
}


// CHECK-LABEL: @test_combine_dot_add_pattern
tt.func @test_combine_dot_add_pattern() -> (tensor<128x128xf32>) {
    // CHECK-DAG: %[[d:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[b:.*]] = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[a:.*]] = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %a = arith.constant dense<1.0> : tensor<128x128xf32>
    %b = arith.constant dense<2.0> : tensor<128x128xf32>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d = arith.constant dense<3.0> : tensor<128x128xf32>

    %dot_out = tt.dot %a, %b, %zero : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>

    // CHECK-NEXT: %[[res:.*]] = tt.dot %[[a]], %[[b]], %[[d]] : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
    // CHECK-NEXT: tt.return %[[res]] : tensor<128x128xf32>
    %res = arith.addf %dot_out, %d : tensor<128x128xf32>

    tt.return %res : tensor<128x128xf32>
}


// CHECK-LABEL: @test_combine_scale_dot_add_pattern
tt.func @test_combine_scale_dot_add_pattern() -> (tensor<128x128xf32>) {
    // CHECK-DAG: %[[a:.*]] = arith.constant dense<1.000000e+00> : tensor<128x128xf8E5M2>
    // CHECK-DAG: %[[sa:.*]] = arith.constant dense<1> : tensor<128x4xi8>
    // CHECK-DAG: %[[b:.*]] = arith.constant dense<2.000000e+00> : tensor<128x128xf8E5M2>
    // CHECK-DAG: %[[sb:.*]] = arith.constant dense<2> : tensor<128x4xi8>
    // CHECK-DAG: %[[d:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    %a = arith.constant dense<1.0> : tensor<128x128xf8E5M2>
    %sa = arith.constant dense<1> : tensor<128x4xi8>
    %b = arith.constant dense<2.0> : tensor<128x128xf8E5M2>
    %sb = arith.constant dense<2> : tensor<128x4xi8>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d = arith.constant dense<3.0> : tensor<128x128xf32>

    %dot_out = tt.dot_scaled %a scale %sa, %b scale %sb, %zero lhs = e5m2 rhs = e5m2 {fastMath = false}
      : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x128xf8E5M2>, tensor<128x4xi8> -> tensor<128x128xf32>

    // CHECK-NEXT: %[[res:.*]] = tt.dot_scaled %[[a]] scale %[[sa]], %[[b]] scale %[[sb]], %[[d]] lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x128xf8E5M2>, tensor<128x4xi8> -> tensor<128x128xf32>
    // CHECK-NEXT: tt.return %[[res]] : tensor<128x128xf32>
    %res = arith.addf %dot_out, %d : tensor<128x128xf32>
    tt.return %res : tensor<128x128xf32>
}


// CHECK-LABEL: @test_combine_dot_add_rev_pattern
tt.func @test_combine_dot_add_rev_pattern() -> (tensor<128x128xf32>) {
    // CHECK-DAG: %[[d:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[b:.*]] = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[a:.*]] = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %a = arith.constant dense<1.0> : tensor<128x128xf32>
    %b = arith.constant dense<2.0> : tensor<128x128xf32>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d = arith.constant dense<3.0> : tensor<128x128xf32>

    %dot_out = tt.dot %a, %b, %zero : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>

    // CHECK-NEXT: %[[res:.*]] = tt.dot %[[a]], %[[b]], %[[d]] : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
    // CHECK-NEXT: tt.return %[[res]] : tensor<128x128xf32>
    %res = arith.addf %d, %dot_out : tensor<128x128xf32>

    tt.return %res : tensor<128x128xf32>
}


// CHECK-LABEL: @test_combine_addptr_pattern
tt.func @test_combine_addptr_pattern(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 10 : i32
    %off1 = arith.constant 15 : i32

    // CHECK-NEXT: %[[cst:.*]] = arith.constant dense<25> : tensor<8xi32>

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    // CHECK-NEXT: %[[tmp0:.*]] = tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.splat %off0 : i32 -> tensor<8xi32>
    %idx1 = tt.splat %off1 : i32 -> tensor<8xi32>

    // CHECK-NEXT: %1 = tt.addptr %[[tmp0]], %[[cst]] : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>

    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_combine_addptr_pattern_discardableattrs
tt.func @test_combine_addptr_pattern_discardableattrs(%base: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %off0 = arith.constant 8 : i32
    %off1 = arith.constant 4 : i32
    // CHECK-NEXT: %[[cst:.*]] = arith.constant 12 : i32
    // CHECK-NEXT: %0 = tt.addptr %{{.*}}, %[[cst]] {tt.constancy = 8 : i32, tt.contiguity = 512 : i32, tt.divisibility = 16 : i32} : !tt.ptr<f32>, i32
    %ptr0 = tt.addptr %base, %off0 : !tt.ptr<f32>, i32
    %ptr1 = tt.addptr %ptr0, %off1 {tt.divisibility = 16 : i32, tt.constancy = 8 : i32, tt.contiguity = 512 : i32} : !tt.ptr<f32>, i32

    tt.return %ptr1 : !tt.ptr<f32>
}

// CHECK-LABEL: @test_combine_addptr_pattern_discardableattrs_disallowed
tt.func @test_combine_addptr_pattern_discardableattrs_disallowed(%base: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %off0 = arith.constant 8 : i32
    %off1 = arith.constant 4 : i32
    // CHECK-NEXT: %[[cst:.*]] = arith.constant 12 : i32
    // CHECK-NEXT: %0 = tt.addptr %{{.*}}, %[[cst]] {tt.divisibility = 16 : i32} : !tt.ptr<f32>, i32
    %ptr0 = tt.addptr %base, %off0 : !tt.ptr<f32>, i32
    %ptr1 = tt.addptr %ptr0, %off1 {tt.divisibility = 16 : i32, tt.disallowed = 8 : i32} : !tt.ptr<f32>, i32

    tt.return %ptr1 : !tt.ptr<f32>
}
// CHECK-LABEL: @test_combine_addptr_pattern_i64
tt.func @test_combine_addptr_pattern_i64(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 10 : i64
    %off1 = arith.constant dense<15> : tensor<8xi64>

    // CHECK-NEXT: %[[cst:.*]] = arith.constant dense<25> : tensor<8xi64>

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    // CHECK-NEXT: %[[tmp0:.*]] = tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.splat %off0 : i64 -> tensor<8xi64>

    // CHECK-NEXT: %1 = tt.addptr %[[tmp0]], %[[cst]] : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    %ptr1 = tt.addptr %ptr0, %off1 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>

    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_combine_addptr_pattern_scalar
tt.func @test_combine_addptr_pattern_scalar(%base: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %off0 = arith.constant 10 : i32
    %off1 = arith.constant 15 : i32

    // CHECK-NEXT: %[[cst:.*]] = arith.constant 25 : i32
    // CHECK-NEXT: %0 = tt.addptr %{{.*}}, %[[cst]] : !tt.ptr<f32>, i32
    %ptr0 = tt.addptr %base, %off0 : !tt.ptr<f32>, i32
    %ptr1 = tt.addptr %ptr0, %off1 : !tt.ptr<f32>, i32

    tt.return %ptr1 : !tt.ptr<f32>
}

// CHECK-LABEL: @test_not_combine_addptr_pattern_1
tt.func @test_not_combine_addptr_pattern_1(%base: !tt.ptr<f32>, %idx0: tensor<8xi32>) -> tensor<8x!tt.ptr<f32>> {
    %off1 = arith.constant 15 : i32

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %idx1 = tt.splat %off1 : i32 -> tensor<8xi32>

    // CHECK: tt.addptr
    // CHECK-NEXT: tt.addptr
    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_not_combine_addptr_pattern
tt.func @test_not_combine_addptr_pattern(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 10 : i16
    %off1 = arith.constant 15 : i32

    // CHECK-DAG: %[[cst:.*]] = arith.constant dense<10> : tensor<8xi16>
    // CHECK-DAG: %[[cst1:.*]] = arith.constant dense<15> : tensor<8xi32>

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.splat %off0 : i16 -> tensor<8xi16>
    %idx1 = tt.splat %off1 : i32 -> tensor<8xi32>

    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi16>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>

    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_not_combine_addptr_pattern_overflow
tt.func @test_not_combine_addptr_pattern_overflow(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 127 : i8
    %off1 = arith.constant 1 : i8

    // CHECK-DAG: %[[cst:.*]] = arith.constant dense<127> : tensor<8xi8>
    // CHECK-DAG: %[[cst1:.*]] = arith.constant dense<1> : tensor<8xi8>

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.splat %off0 : i8 -> tensor<8xi8>
    %idx1 = tt.splat %off1 : i8 -> tensor<8xi8>

    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi8>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi8>

    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_combine_select_masked_load_pattern
tt.func @test_combine_select_masked_load_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %cond: i1) -> (tensor<8xf32>, tensor<8xf32>) {
    %mask = tt.splat %cond : i1 -> tensor<8xi1>
    %false_val = arith.constant dense<0.0> : tensor<8xf32>

    // CHECK: %[[res1:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    %x = tt.load %ptr, %mask, %false_val : tensor<8x!tt.ptr<f32>>
    %0 = arith.select %cond, %x, %false_val : tensor<8xf32>

    // CHECK: %[[res2:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    %y = tt.load %ptr, %mask, %false_val : tensor<8x!tt.ptr<f32>>
    %1 = arith.select %cond, %y, %false_val : tensor<8xf32>

    // CHECK: tt.return %[[res1]], %[[res2]] : tensor<8xf32>, tensor<8xf32>
    tt.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// A volatile load with a single user is combined in place: the select is
// removed and the volatile load's `other` is replaced with the false value, so
// the read still happens exactly once.
// CHECK-LABEL: @test_combine_select_volatile_masked_load_single_use
tt.func @test_combine_select_volatile_masked_load_single_use(%ptr: tensor<8x!tt.ptr<f32>>, %cond: i1, %old_other: tensor<8xf32>, %false_value: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK: %[[MASK:.*]] = tt.splat %{{.*}} : i1 -> tensor<8xi1>
    // CHECK-NEXT: %[[LOAD:.*]] = tt.load %{{.*}}, %[[MASK]], %arg3 {isVolatile = true} : tensor<8x!tt.ptr<f32>>
    // CHECK-NOT: arith.select
    // CHECK-NEXT: tt.return %[[LOAD]] : tensor<8xf32>
    %mask = tt.splat %cond : i1 -> tensor<8xi1>
    %load = tt.load %ptr, %mask, %old_other {isVolatile = true} : tensor<8x!tt.ptr<f32>>
    %select = arith.select %cond, %load, %false_value : tensor<8xf32>
    tt.return %select : tensor<8xf32>
}

// A volatile load with more than one user is not combined: rewriting would
// build a second volatile load and read the memory twice.
// CHECK-LABEL: @test_not_combine_select_volatile_masked_load_multi_use
tt.func @test_not_combine_select_volatile_masked_load_multi_use(%ptr: tensor<8x!tt.ptr<f32>>, %cond: i1, %old_other: tensor<8xf32>, %false_value: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
    // CHECK: %[[LOAD:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} {isVolatile = true} : tensor<8x!tt.ptr<f32>>
    // CHECK-NEXT: %[[SELECT:.*]] = arith.select %{{.*}}, %[[LOAD]], %{{.*}} : tensor<8xf32>
    // CHECK-NEXT: tt.return %[[SELECT]], %[[LOAD]] : tensor<8xf32>, tensor<8xf32>
    %mask = tt.splat %cond : i1 -> tensor<8xi1>
    %load = tt.load %ptr, %mask, %old_other {isVolatile = true} : tensor<8x!tt.ptr<f32>>
    %select = arith.select %cond, %load, %false_value : tensor<8xf32>
    tt.return %select, %load : tensor<8xf32>, tensor<8xf32>
}

// A volatile load cannot be updated in place when the false value is defined
// after it (using it as the load's `other` would not dominate the load), and it
// cannot be duplicated either, so it is left unchanged.
// CHECK-LABEL: @test_not_combine_select_volatile_masked_load_false_after
tt.func @test_not_combine_select_volatile_masked_load_false_after(%ptr: tensor<8x!tt.ptr<f32>>, %cond: i1, %old_other: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK: %[[LOAD:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} {isVolatile = true} : tensor<8x!tt.ptr<f32>>
    // CHECK-NEXT: %[[FV:.*]] = arith.negf
    // CHECK-NEXT: %[[SELECT:.*]] = arith.select %{{.*}}, %[[LOAD]], %[[FV]] : tensor<8xf32>
    // CHECK-NEXT: tt.return %[[SELECT]] : tensor<8xf32>
    %mask = tt.splat %cond : i1 -> tensor<8xi1>
    %load = tt.load %ptr, %mask, %old_other {isVolatile = true} : tensor<8x!tt.ptr<f32>>
    %false_value = arith.negf %old_other : tensor<8xf32>
    %select = arith.select %cond, %load, %false_value : tensor<8xf32>
    tt.return %select : tensor<8xf32>
}

// A load whose false value is defined after it is left unchanged: the false
// value does not dominate the load, so it cannot become the load's operand, and
// combining is only done in place (never by creating a second load).
// CHECK-LABEL: @test_not_combine_select_masked_load_false_after
tt.func @test_not_combine_select_masked_load_false_after(%ptr: tensor<8x!tt.ptr<f32>>, %cond: i1, %old_other: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK: %[[LOAD:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    // CHECK-NEXT: %[[FV:.*]] = arith.negf
    // CHECK-NEXT: %[[SELECT:.*]] = arith.select %{{.*}}, %[[LOAD]], %[[FV]] : tensor<8xf32>
    // CHECK-NEXT: tt.return %[[SELECT]] : tensor<8xf32>
    %mask = tt.splat %cond : i1 -> tensor<8xi1>
    %load = tt.load %ptr, %mask, %old_other : tensor<8x!tt.ptr<f32>>
    %false_value = arith.negf %old_other : tensor<8xf32>
    %select = arith.select %cond, %load, %false_value : tensor<8xf32>
    tt.return %select : tensor<8xf32>
}

// When the false value depends on the load, the load necessarily has more than
// one user, so it is left unchanged: an in-place update is not possible and a
// fresh load would duplicate the access.
// CHECK-LABEL: @test_not_combine_select_masked_load_false_depends_on_load
tt.func @test_not_combine_select_masked_load_false_depends_on_load(%ptr: tensor<8x!tt.ptr<f32>>, %cond: i1, %old_other: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK: %[[LOAD:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    // CHECK-NEXT: %[[FV:.*]] = arith.negf %[[LOAD]]
    // CHECK-NEXT: %[[SELECT:.*]] = arith.select %{{.*}}, %[[LOAD]], %[[FV]] : tensor<8xf32>
    // CHECK-NEXT: tt.return %[[SELECT]] : tensor<8xf32>
    %mask = tt.splat %cond : i1 -> tensor<8xi1>
    %load = tt.load %ptr, %mask, %old_other : tensor<8x!tt.ptr<f32>>
    %false_value = arith.negf %load : tensor<8xf32>
    %select = arith.select %cond, %load, %false_value : tensor<8xf32>
    tt.return %select : tensor<8xf32>
}

// CHECK-LABEL: @test_combine_select_masked_load_fail_pattern
tt.func @test_combine_select_masked_load_fail_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %dummy_load: tensor<8xf32>, %dummy_broadcast: tensor<8xi1>, %cond0: i1, %cond1: i1) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %false_val = arith.constant dense<0.0> : tensor<8xf32>

    // Case 1: value at the "load" position is not an "op".  Select should not be canonicalized.
    // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    %0 = arith.select %cond0, %dummy_load, %false_val : tensor<8xf32>

    // Case 2: value at the "broadcast" position is not an "op".  Select should not be canonicalized.
    %real_load0 = tt.load %ptr, %dummy_broadcast, %false_val : tensor<8x!tt.ptr<f32>>
    // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    %1 = arith.select %cond0, %real_load0, %false_val : tensor<8xf32>

    // Case 3: condition of "broadcast" is not the same as the condition of "select".  Select should not be canonicalized.
    %cond0_ = tt.splat %cond0 : i1 -> tensor<8xi1>
    %real_load1 = tt.load %ptr, %cond0_, %false_val : tensor<8x!tt.ptr<f32>>
    // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    %2 = arith.select %cond1, %real_load1, %false_val : tensor<8xf32>

    tt.return %0, %1, %2 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_masked_load_pattern
tt.func @test_canonicalize_masked_load_pattern(%ptr: tensor<8x!tt.ptr<f32>>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %true_mask = arith.constant dense<true> : tensor<8xi1>
    %false_mask = arith.constant dense<false> : tensor<8xi1>
    %other_val = arith.constant dense<0.0> : tensor<8xf32>

    // true_mask with other
    // CHECK: %[[res1:.*]] = tt.load %{{.*}} : tensor<8x!tt.ptr<f32>>
    %x = tt.load %ptr, %true_mask : tensor<8x!tt.ptr<f32>>

    // true_mask without other
    // CHECK: %[[res2:.*]] = tt.load %{{.*}} : tensor<8x!tt.ptr<f32>>
    %y = tt.load %ptr, %true_mask, %other_val : tensor<8x!tt.ptr<f32>>

    // false_mask with other. It should become "other" (i.e., %y)
    %z = tt.load %ptr, %false_mask, %y : tensor<8x!tt.ptr<f32>>

    // CHECK: tt.return %[[res1]], %[[res2]], %[[res2]] : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
    tt.return %x, %y, %z: tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_masked_load_fail_pattern
tt.func @test_canonicalize_masked_load_fail_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %mask: tensor<8xi1>) -> (tensor<8xf32>, tensor<8xf32>) {
    %other_val = arith.constant dense<0.0> : tensor<8xf32>

    // Case: value at the "mask" position is not an "op".  Load should not be canonicalized.
    // CHECK: %[[res1:.*]] = tt.load %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    %x = tt.load %ptr, %mask : tensor<8x!tt.ptr<f32>>
    // CHECK: %[[res1:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    %y = tt.load %ptr, %mask, %other_val : tensor<8x!tt.ptr<f32>>

    tt.return %x, %y: tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_masked_store_pattern
tt.func @test_canonicalize_masked_store_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %val: tensor<8xf32>) {
    %true_mask = arith.constant dense<true> : tensor<8xi1>
    %false_mask = arith.constant dense<false> : tensor<8xi1>

    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    tt.store %ptr, %val, %true_mask : tensor<8x!tt.ptr<f32>>

    // The following store should disappear.
    // CHECK-NEXT: tt.return
    tt.store %ptr, %val, %false_mask : tensor<8x!tt.ptr<f32>>
    tt.return
}

// CHECK-LABEL: @test_canonicalize_masked_store_fail_pattern
tt.func @test_canonicalize_masked_store_fail_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %val: tensor<8xf32>, %mask: tensor<8xi1>) {
    // Case: value at the "mask" position is not an "op".  Store should not be canonicalized.
    // CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<f32>>
    tt.store %ptr, %val, %mask : tensor<8x!tt.ptr<f32>>
    tt.return
}

// CHECK-LABEL: @test_canonicalize_expand_dims
tt.func @test_canonicalize_expand_dims(%arg0: tensor<f32>, %arg1: tensor<1xf32>) -> (tensor<1x8xf32>, tensor<8x8xf32>) {
    %splat = tt.splat %arg0 : tensor<f32> -> tensor<8xf32>
    // CHECK: %{{.*}} = tt.splat %arg0 : tensor<f32> -> tensor<1x8xf32>
    %ed = tt.expand_dims %splat {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>

    // CHECK-NEXT: %[[ed2:.*]] = tt.expand_dims %arg1 {axis = 0 : i32} : tensor<1xf32> -> tensor<1x1xf32>
    // CHECK-NEXT: %{{.*}} = tt.broadcast %[[ed2]] : tensor<1x1xf32> -> tensor<8x8xf32>
    %bc = tt.broadcast %arg1 : tensor<1xf32> -> tensor<8xf32>
    %ed2 = tt.expand_dims %bc {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
    %bc2 = tt.broadcast %ed2 : tensor<1x8xf32> -> tensor<8x8xf32>

    tt.return %ed, %bc2 : tensor<1x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: @test_canonicalize_reshape
tt.func @test_canonicalize_reshape(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> (tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>) {
    %reshape0 = tt.reshape %arg0 : tensor<8xf32> -> tensor<2x4xf32>
    // CHECK: %{{.*}} = tt.reshape %arg0 : tensor<8xf32> -> tensor<4x2xf32>
    %reshape1 = tt.reshape %reshape0 : tensor<2x4xf32> -> tensor<4x2xf32>

    %splat = tt.splat %arg1 : tensor<f32> -> tensor<8xf32>
    // CHECK: %{{.*}} = tt.splat %arg1 : tensor<f32> -> tensor<2x2x2xf32>
    %reshape2 = tt.reshape %splat : tensor<8xf32> -> tensor<2x2x2xf32>

    %reshape3 = tt.reshape %arg0 : tensor<8xf32> -> tensor<8xf32>
    // CHECK: %{{.*}} = arith.addf %arg0, %arg0 : tensor<8xf32>
    %add = arith.addf %reshape3, %arg0 : tensor<8xf32>

    // CHECK: %{{.*}} = tt.reshape %arg0 : tensor<8xf32> -> tensor<2x2x2xf32>
    %view = tt.reshape %reshape0 : tensor<2x4xf32> -> tensor<2x2x2xf32>

    tt.return %reshape1, %reshape2, %add, %view : tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>
}

// CHECK-LABEL: @test_canonicalize_reshape_expand_dims
tt.func @test_canonicalize_reshape_expand_dims(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK-NEXT: tt.return %arg0 : tensor<8xf32>
    %0 = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
    %1 = tt.reshape %0 : tensor<1x8xf32> -> tensor<8xf32>
    tt.return %1 : tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_broadcast
tt.func @test_canonicalize_broadcast(%arg0: tensor<1x1x8xf32>, %arg1: tensor<f32>) -> (tensor<4x2x8xf32>, tensor<8x8xf32>, tensor<1x1x8xf32>) {
    %broadcast0 = tt.broadcast %arg0 : tensor<1x1x8xf32> -> tensor<1x2x8xf32>
    // CHECK: %{{.*}} = tt.broadcast %arg0 : tensor<1x1x8xf32> -> tensor<4x2x8xf32>
    %broadcast1 = tt.broadcast %broadcast0 : tensor<1x2x8xf32> -> tensor<4x2x8xf32>

    %splat = tt.splat %arg1 : tensor<f32> -> tensor<1x8xf32>
    // CHECK: %{{.*}} = tt.splat %arg1 : tensor<f32> -> tensor<8x8xf32>
    %broadcast2 = tt.broadcast %splat : tensor<1x8xf32> -> tensor<8x8xf32>

    %broadcast3 = tt.broadcast %arg0 : tensor<1x1x8xf32> -> tensor<1x1x8xf32>
    // CHECK: %{{.*}} = arith.addf %arg0, %arg0 : tensor<1x1x8xf32>
    %add = arith.addf %broadcast3, %arg0 : tensor<1x1x8xf32>

    tt.return %broadcast1, %broadcast2, %add : tensor<4x2x8xf32>, tensor<8x8xf32>, tensor<1x1x8xf32>
}

// CHECK-LABEL: @test_fold_views
tt.func @test_fold_views() -> (tensor<16x8xf32>, tensor<16x128xf32>, tensor<1x1x128xf32>) {
    %a = arith.constant dense<1.0> : tensor<1x128xf32>

    // CHECK-DAG: %{{.*}} = arith.constant dense<1.{{.*}}> : tensor<16x8xf32>
    %b = tt.reshape %a : tensor<1x128xf32> -> tensor<16x8xf32>

    // CHECK-DAG: %{{.*}} = arith.constant dense<1.{{.*}}> : tensor<16x128xf32>
    %c = tt.broadcast %a : tensor<1x128xf32> -> tensor<16x128xf32>

    // CHECK-DAG: %{{.*}} = arith.constant dense<1.{{.*}}> : tensor<1x1x128xf32>
    %d = tt.expand_dims %a {axis = 0: i32} : tensor<1x128xf32> -> tensor<1x1x128xf32>

    tt.return %b, %c, %d : tensor<16x8xf32>, tensor<16x128xf32>, tensor<1x1x128xf32>
}

// CHECK-LABEL: @test_nop_transpose
tt.func @test_nop_transpose(%arg0: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
    %a = tt.trans %arg0 {order = array<i32: 0, 1>} : tensor<2x4xf32> -> tensor<2x4xf32>
    // CHECK: tt.return %arg0
    tt.return %a : tensor<2x4xf32>
}

// CHECK-LABEL: @test_nested_transpose
tt.func @test_nested_transpose(%arg0: tensor<2x4x8xf32>) -> (tensor<8x2x4xf32>) {
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<2x4x8xf32> -> tensor<4x2x8xf32>
    %b = tt.trans %a {order = array<i32: 2, 1, 0>} : tensor<4x2x8xf32> -> tensor<8x2x4xf32>
    // CHECK: %[[res:.*]] = tt.trans %arg0 {order = array<i32: 2, 0, 1>}
    // CHECK: tt.return %[[res]]
    tt.return %b : tensor<8x2x4xf32>
}

// CHECK-LABEL: test_reshape_reduce
tt.func @test_reshape_reduce(%0: tensor<32x4x2xi32>) -> (i32, tensor<16xi32>) {
  // CHECK: tt.reshape %{{.+}} : tensor<32x4x2xi32> -> tensor<256xi32>
  %1 = tt.reshape %0 : tensor<32x4x2xi32> -> tensor<256xi32>
  %2 = "tt.reduce" (%1) ({
    ^bb0(%arg7: i32, %arg8: i32):
      %add = arith.addi %arg7, %arg8 : i32
      tt.reduce.return %add : i32
    }) {axis = 0 : i32} : (tensor<256xi32>) -> i32
  %3 = tt.histogram %1 : tensor<256xi32> -> tensor<16xi32>
  tt.return %2, %3 : i32, tensor<16xi32>
}

// CHECK-LABEL: test_rank_reduce_desc_load
tt.func @test_rank_reduce_desc_load(%0: !tt.tensordesc<1x128x64xf16>) -> (tensor<128x64xf16>) {
  %c0 = arith.constant 0 : i32
  // CHECK: %[[R:.+]] = tt.descriptor_load {{.*}} : !tt.tensordesc<1x128x64xf16> -> tensor<128x64xf16>
  // CHECK: tt.return %[[R]]
  %l = tt.descriptor_load %0[%c0, %c0, %c0] : !tt.tensordesc<1x128x64xf16> -> tensor<1x128x64xf16>
  %r = tt.reshape %l : tensor<1x128x64xf16> -> tensor<128x64xf16>
  tt.return %r :  tensor<128x64xf16>
}

// CHECK-LABEL: @test_combine_dot_add_no_fold_when_imprecise_allowed
tt.func @test_combine_dot_add_no_fold_when_imprecise_allowed() -> (tensor<128x128xf32>) {
    // CHECK-DAG: %[[D:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    %a    = arith.constant dense<1.0> : tensor<128x128xf32>
    %b    = arith.constant dense<2.0> : tensor<128x128xf32>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d    = arith.constant dense<3.0> : tensor<128x128xf32>

    %dot_out = tt.dot %a, %b, %zero {maxNumImpreciseAcc = 1 : i32}
               : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>

    // CHECK: arith.addf %{{.*}}, %[[D]] : tensor<128x128xf32>
    // CHECK-NEXT: tt.return %{{.*}} : tensor<128x128xf32>
    %res = arith.addf %dot_out, %d : tensor<128x128xf32>
    tt.return %res : tensor<128x128xf32>
}

// CHECK-LABEL: @test_combine_dot_add_fold_when_precise_required
tt.func @test_combine_dot_add_fold_when_precise_required() -> (tensor<128x128xf32>) {
    // CHECK-DAG: %[[D:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[B:.*]] = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[A:.*]] = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %a    = arith.constant dense<1.0> : tensor<128x128xf32>
    %b    = arith.constant dense<2.0> : tensor<128x128xf32>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d    = arith.constant dense<3.0> : tensor<128x128xf32>

    %dot_out = tt.dot %a, %b, %zero {maxNumImpreciseAcc = 0 : i32}
               : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>

    // CHECK-NEXT: %[[RES:.*]] = tt.dot %[[A]], %[[B]], %[[D]] : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
    // CHECK-NEXT: tt.return %[[RES]] : tensor<128x128xf32>
    %res = arith.addf %dot_out, %d : tensor<128x128xf32>
    tt.return %res : tensor<128x128xf32>
}

// CHECK-LABEL: @test_combine_broadcast_mul_reduce
tt.func @test_combine_broadcast_mul_reduce(%arg0: tensor<32x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<32x32xf32> {
    // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    // CHECK: %[[RES:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[CST]] : tensor<32x16xf32> * tensor<16x32xf32> -> tensor<32x32xf32>
    // CHECK: tt.return %[[RES]] : tensor<32x32xf32>
    %0 = tt.expand_dims %arg0 {axis = 2 : i32} : tensor<32x16xf32> -> tensor<32x16x1xf32>
    %1 = tt.broadcast %0 : tensor<32x16x1xf32> -> tensor<32x16x32xf32>
    %2 = tt.expand_dims %arg1 {axis = 0 : i32} : tensor<16x32xf32> -> tensor<1x16x32xf32>
    %3 = tt.broadcast %2 : tensor<1x16x32xf32> -> tensor<32x16x32xf32>
    %4 = arith.mulf %1, %3 : tensor<32x16x32xf32>
    %5 = "tt.reduce"(%4) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
        %6 = arith.addf %arg2, %arg3 : f32
        tt.reduce.return %6 : f32
    }) : (tensor<32x16x32xf32>) -> tensor<32x32xf32>
    tt.return %5 : tensor<32x32xf32>
}

// CHECK-LABEL: @test_combine_broadcast_mul_reduce_reshape
tt.func @test_combine_broadcast_mul_reduce_reshape(%arg0: tensor<32x16x1xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<32x32xf32> {
    // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    // CHECK: %[[LHS:.*]] = tt.reshape %arg0 : tensor<32x16x1xf32> -> tensor<32x16xf32>
    // CHECK: %[[RHS:.*]] = tt.reshape %arg1 : tensor<1x16x32xf32> -> tensor<16x32xf32>
    // CHECK: %[[RES:.*]] = tt.dot %[[LHS]], %[[RHS]], %[[CST]] : tensor<32x16xf32> * tensor<16x32xf32> -> tensor<32x32xf32>
    // CHECK: tt.return %[[RES]] : tensor<32x32xf32>
    %0 = tt.broadcast %arg0 : tensor<32x16x1xf32> -> tensor<32x16x32xf32>
    %1 = tt.broadcast %arg1 : tensor<1x16x32xf32> -> tensor<32x16x32xf32>
    %2 = arith.mulf %0, %1 : tensor<32x16x32xf32>
    %3 = "tt.reduce"(%2) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
        %4 = arith.addf %arg2, %arg3 : f32
        tt.reduce.return %4 : f32
    }) : (tensor<32x16x32xf32>) -> tensor<32x32xf32>
    tt.return %3 : tensor<32x32xf32>
}

// CHECK-LABEL: @test_combine_broadcast_mul_reduce_extra_broadcast
tt.func @test_combine_broadcast_mul_reduce_extra_broadcast(%arg0: tensor<32x1xf32>, %arg1: tensor<1x32xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: tt.dot
    // CHECK: tt.reduce
    %0 = tt.expand_dims %arg0 {axis = 2 : i32} : tensor<32x1xf32> -> tensor<32x1x1xf32>
    %1 = tt.broadcast %0 : tensor<32x1x1xf32> -> tensor<32x16x32xf32>
    %2 = tt.expand_dims %arg1 {axis = 0 : i32} : tensor<1x32xf32> -> tensor<1x1x32xf32>
    %3 = tt.broadcast %2 : tensor<1x1x32xf32> -> tensor<32x16x32xf32>
    %4 = arith.mulf %1, %3 : tensor<32x16x32xf32>
    %5 = "tt.reduce"(%4) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
        %6 = arith.addf %arg2, %arg3 : f32
        tt.reduce.return %6 : f32
    }) : (tensor<32x16x32xf32>) -> tensor<32x32xf32>
    tt.return %5 : tensor<32x32xf32>
}

// CHECK-LABEL: @test_combine_broadcast_mul_reduce_wrong_axis
tt.func @test_combine_broadcast_mul_reduce_wrong_axis(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: tt.dot
    // CHECK: tt.reduce
    %0 = tt.expand_dims %arg0 {axis = 2 : i32} : tensor<32x32xf32> -> tensor<32x32x1xf32>
    %1 = tt.broadcast %0 : tensor<32x32x1xf32> -> tensor<32x32x32xf32>
    %2 = tt.expand_dims %arg1 {axis = 0 : i32} : tensor<32x32xf32> -> tensor<1x32x32xf32>
    %3 = tt.broadcast %2 : tensor<1x32x32xf32> -> tensor<32x32x32xf32>
    %4 = arith.mulf %1, %3 : tensor<32x32x32xf32>
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
        %6 = arith.addf %arg2, %arg3 : f32
        tt.reduce.return %6 : f32
    }) : (tensor<32x32x32xf32>) -> tensor<32x32xf32>
    tt.return %5 : tensor<32x32xf32>
}

// CHECK-LABEL: @test_combine_broadcast_mul_reduce_higher_rank
tt.func @test_combine_broadcast_mul_reduce_higher_rank(%arg0: tensor<32x16x4xf32>, %arg1: tensor<16x32x4xf32>) -> tensor<32x32x4xf32> {
    // CHECK-NOT: tt.dot
    // CHECK: tt.reduce
    %0 = tt.expand_dims %arg0 {axis = 2 : i32} : tensor<32x16x4xf32> -> tensor<32x16x1x4xf32>
    %1 = tt.broadcast %0 : tensor<32x16x1x4xf32> -> tensor<32x16x32x4xf32>
    %2 = tt.expand_dims %arg1 {axis = 0 : i32} : tensor<16x32x4xf32> -> tensor<1x16x32x4xf32>
    %3 = tt.broadcast %2 : tensor<1x16x32x4xf32> -> tensor<32x16x32x4xf32>
    %4 = arith.mulf %1, %3 : tensor<32x16x32x4xf32>
    %5 = "tt.reduce"(%4) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
        %6 = arith.addf %arg2, %arg3 : f32
        tt.reduce.return %6 : f32
    }) : (tensor<32x16x32x4xf32>) -> tensor<32x32x4xf32>
    tt.return %5 : tensor<32x32x4xf32>
}

// CHECK-LABEL: @test_combine_broadcast_mul_reduce_multiple_results
tt.func @test_combine_broadcast_mul_reduce_multiple_results(%arg0: tensor<32x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: tt.dot
    // CHECK: tt.reduce
    %0 = tt.expand_dims %arg0 {axis = 2 : i32} : tensor<32x16xf32> -> tensor<32x16x1xf32>
    %1 = tt.broadcast %0 : tensor<32x16x1xf32> -> tensor<32x16x32xf32>
    %2 = tt.expand_dims %arg1 {axis = 0 : i32} : tensor<16x32xf32> -> tensor<1x16x32xf32>
    %3 = tt.broadcast %2 : tensor<1x16x32xf32> -> tensor<32x16x32xf32>
    %4 = arith.mulf %1, %3 : tensor<32x16x32xf32>
    %5:2 = "tt.reduce"(%4, %4) <{axis = 1 : i32}> ({
    ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
        %6 = arith.addf %a0, %b0 : f32
        tt.reduce.return %6, %6 : f32, f32
    }) : (tensor<32x16x32xf32>, tensor<32x16x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>)
    tt.return %5#0 : tensor<32x32xf32>
}

// Native indexed updates: contracts describe producer values and invocation
// footprints. The pass must derive address injectivity from the actual mapping.
// Unsupported effects, domains, masks, arithmetic, and launches retain atomics.

// -----

// INDEXED-LABEL: @indexed_update_unique_zero
// INDEXED-SAME: tt.experimental_indexed_update_specialized = 1 : i32
// INDEXED: tt.load
// INDEXED-NOT: tt.load
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: add.rn.ftz.f32
// INDEXED: tt.store
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: tt.return
// NO-TARGET-LABEL: @indexed_update_unique_zero
// NO-TARGET: tt.atomic_rmw
// OTHER-TARGET-LABEL: @indexed_update_unique_zero
// OTHER-TARGET: tt.atomic_rmw
tt.func public @indexed_update_unique_zero(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_unique_nonzero
// INDEXED-SAME: tt.experimental_indexed_update_specialized = 1 : i32
// INDEXED: tt.load
// INDEXED: tt.load
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: add.rn.ftz.f32
// INDEXED: tt.store
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_unique_nonzero(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_combined_offset
// INDEXED-SAME: tt.experimental_indexed_update_specialized = 1 : i32
// INDEXED: tt.load
// INDEXED-NOT: tt.load
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: add.rn.ftz.f32
// INDEXED: tt.store
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_combined_offset(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rows = tt.splat %row : i32 -> tensor<8xi32>
  %offsets = arith.addi %rows, %cols : tensor<8xi32>
  %outs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %outs, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_lossless_index_truncation
// INDEXED-SAME: tt.experimental_indexed_update_specialized = 1 : i32
// INDEXED: tt.load
// INDEXED-NOT: tt.load
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: add.rn.ftz.f32
// INDEXED: tt.store
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_lossless_index_truncation(%indices: !tt.ptr<i64> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i64>, i32
  %loaded = tt.load %iptr : !tt.ptr<i64>
  %index = arith.trunci %loaded : i64 to i32
  %row = arith.muli %index, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_wide_index
// INDEXED-SAME: tt.experimental_indexed_update_specialized = 1 : i32
// INDEXED: tt.load
// INDEXED-NOT: tt.load
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: add.rn.ftz.f32
// INDEXED: tt.store
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_wide_index(%indices: !tt.ptr<i64> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i64>, i32
  %loaded = tt.load %iptr : !tt.ptr<i64>
  %cd64 = arith.constant 13 : i64
  %row = arith.muli %loaded, %cd64 : i64
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %widecols = arith.extsi %cols : tensor<8xi32> to tensor<8xi64>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i64
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %widecols : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_full_tiles
// INDEXED-SAME: tt.experimental_indexed_update_specialized = 1 : i32
// INDEXED: tt.load
// INDEXED-NOT: tt.load
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: add.rn.ftz.f32
// INDEXED: tt.store
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_full_tiles(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 16 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_missing_unique
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_missing_unique(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_missing_private
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_missing_private(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_missing_alias_group
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_missing_alias_group(%indices: !tt.ptr<i32> {tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_aliasing_index
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_aliasing_index(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 1 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_short_unique_prefix
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_short_unique_prefix(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 8 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_missing_grid_axis
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_missing_grid_axis(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_duplicated_grid_axis
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_duplicated_grid_axis(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 2 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_incorrect_tail_mask
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_incorrect_tail_mask(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi sle, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_local_range_mask
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_local_range_mask(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %range, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_or_mask_not_a_guard
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_or_mask_not_a_guard(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %all = arith.constant dense<true> : tensor<8xi1>
  %mask = arith.ori %tail, %all : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_overlapping_column_tiles
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_overlapping_column_tiles(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 7 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_wrapping_row_product
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_wrapping_row_product(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 2147483647 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_truncation_loses_uniqueness
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_truncation_loses_uniqueness(%indices: !tt.ptr<i64> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 131072 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i64>, i32
  %loaded = tt.load %iptr : !tt.ptr<i64>
  %index = arith.trunci %loaded : i64 to i16
  %rowindex = arith.extsi %index : i16 to i32
  %row = arith.muli %rowindex, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_repeated_index_position
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_repeated_index_position(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %c0 : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_masked_index_load
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_masked_index_load(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %index_active = arith.cmpi slt, %x, %c1 : i32
  %loaded = tt.load %iptr, %index_active, %c0 : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_volatile_index_load
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_volatile_index_load(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr {isVolatile = true} : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_used_atomic_result
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_used_atomic_result(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) -> tensor<8xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return %old : tensor<8xf32>
}

// -----

// INDEXED-LABEL: @indexed_update_other_write
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_other_write(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.store %ptrs, %values, %mask : tensor<8x!tt.ptr<f32>>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_read_may_alias_output
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_read_may_alias_output(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}, %alias: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %read = tt.load %alias : !tt.ptr<f32>
  %readvalues = tt.splat %read : f32 -> tensor<8xf32>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %readvalues, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_unknown_call
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func private @indexed_unknown_effect()
tt.func public @indexed_update_unknown_call(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  tt.call @indexed_unknown_effect() : () -> ()
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_loop_repeats_site
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_loop_repeats_site(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  scf.for %i = %c0 to %c8 step %c1 : i32 {
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  }
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_system_scope
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_system_scope(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, sys, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_missing_cta_contract
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_missing_cta_contract(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_multiple_ctas
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
tt.func public @indexed_update_multiple_ctas(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 2 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_module_multiple_ctas
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw
// INDEXED: tt.return
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @indexed_update_module_multiple_ctas(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}
}

// -----

// INDEXED-LABEL: @indexed_update_gluon_layout
// INDEXED-SAME: tt.experimental_indexed_update_specialized = 1 : i32
// INDEXED: tt.load
// INDEXED-NOT: tt.load
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: add.rn.ftz.f32
// INDEXED: tt.store
// INDEXED-NOT: tt.atomic_rmw
// INDEXED: tt.return
// NO-TARGET-LABEL: @indexed_update_gluon_layout
// NO-TARGET: tt.atomic_rmw
// OTHER-TARGET-LABEL: @indexed_update_gluon_layout
// OTHER-TARGET: tt.atomic_rmw
#indexed_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @indexed_update_gluon_layout(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32, #indexed_blocked>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #indexed_blocked>
  %bases = tt.splat %base : i32 -> tensor<8xi32, #indexed_blocked>
  %cols = arith.addi %bases, %range : tensor<8xi32, #indexed_blocked>
  %ds = tt.splat %cd : i32 -> tensor<8xi32, #indexed_blocked>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #indexed_blocked>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>, #indexed_blocked>, tensor<8xi32, #indexed_blocked>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32, #indexed_blocked>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1, #indexed_blocked>
  %mask = arith.andi %tail, %valids : tensor<8xi1, #indexed_blocked>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>, #indexed_blocked>, tensor<8xf32, #indexed_blocked>, tensor<8xi1, #indexed_blocked>) -> tensor<8xf32, #indexed_blocked>
  tt.return
}
}

// Private atomic relaxation preserves colliding updates and needs no uniqueness,
// index bounds, launch dimensions, or zero-initialization attributes.

// -----

// PRIVATE-LABEL: @private_atomic_acq_rel
// PRIVATE-SAME: tt.experimental_private_atomic_relaxed = 1 : i32
// PRIVATE: tt.atomic_rmw fadd, relaxed, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
// PRIVATE-NO-TARGET-LABEL: @private_atomic_acq_rel
// PRIVATE-NO-TARGET: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-OTHER-TARGET-LABEL: @private_atomic_acq_rel
// PRIVATE-OTHER-TARGET: tt.atomic_rmw fadd, acq_rel, gpu
tt.func public @private_atomic_acq_rel(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_acquire
// PRIVATE-SAME: tt.experimental_private_atomic_relaxed = 1 : i32
// PRIVATE: tt.atomic_rmw fadd, relaxed, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_acquire(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acquire, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_release
// PRIVATE-SAME: tt.experimental_private_atomic_relaxed = 1 : i32
// PRIVATE: tt.atomic_rmw fadd, relaxed, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_release(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, release, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_gluon
// PRIVATE-SAME: tt.experimental_private_atomic_relaxed = 1 : i32
// PRIVATE: tt.atomic_rmw fadd, relaxed, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
#private_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @private_atomic_gluon(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32, #private_blocked>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #private_blocked>
  %mask = arith.constant dense<true> : tensor<8xi1, #private_blocked>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>, #private_blocked>, tensor<8xf32, #private_blocked>, tensor<8xi1, #private_blocked>) -> tensor<8xf32, #private_blocked>
  tt.return
}
}

// -----

// PRIVATE-LABEL: @private_atomic_already_relaxed
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, relaxed, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_already_relaxed(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_missing_private
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_missing_private(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_missing_input_alias
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_missing_input_alias(%src: !tt.ptr<f32> , %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_missing_output_alias
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_missing_output_alias(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_aliasing_input
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_aliasing_input(%src: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_missing_num_ctas
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_missing_num_ctas(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_multiple_ctas
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_multiple_ctas(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 2 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_module_multiple_ctas
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
module attributes {"ttg.num-ctas" = 2 : i32} {
tt.func public @private_atomic_module_multiple_ctas(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}
}

// -----

// PRIVATE-LABEL: @private_atomic_malformed_cta_contract
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_malformed_cta_contract(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = "one"}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_output_read
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_output_read(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %out : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_extra_store
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_extra_store(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  tt.store %src, %value : !tt.ptr<f32>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_two_updates
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_two_updates(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  %other = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_used_result
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_used_result(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) -> tensor<8xf32> {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return %old : tensor<8xf32>
}

// -----

// PRIVATE-LABEL: @private_atomic_volatile_load
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_volatile_load(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src {isVolatile = true} : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_unknown_call
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func private @private_unknown_effect()
tt.func public @private_atomic_unknown_call(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  tt.call @private_unknown_effect() : () -> ()
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_pure_asm
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_pure_asm(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %unused_asm = tt.elementwise_inline_asm "mov.b32 $0, $1;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %values : tensor<8xf32> -> tensor<8xf32>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_loop
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_loop(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %c4 step %c1 : i32 {
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  }
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_system_scope
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, sys
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_system_scope(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, sys, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// PRIVATE-LABEL: @private_atomic_unsupported_type
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_unsupported_type(%src: !tt.ptr<f64> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f64> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f64>
  %values = tt.splat %value : f64 -> tensor<8xf64>
  %ptrs = tt.splat %out : !tt.ptr<f64> -> tensor<8x!tt.ptr<f64>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f64>>, tensor<8xf64>, tensor<8xi1>) -> tensor<8xf64>
  tt.return
}
// -----

// PRIVATE-LABEL: @private_atomic_private_helper
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func private @private_atomic_private_helper(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}


// -----

// PRIVATE-LABEL: @private_atomic_called_public
// PRIVATE-NOT: tt.experimental_private_atomic_relaxed
// PRIVATE: tt.atomic_rmw fadd, acq_rel, gpu
// PRIVATE-NOT: tt.store
// PRIVATE: tt.return
tt.func public @private_atomic_called_public(%src: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_num_ctas = 1 : i32}) {
  %value = tt.load %src : !tt.ptr<f32>
  %values = tt.splat %value : f32 -> tensor<8xf32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}


tt.func public @private_atomic_calling_entry(%src: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  tt.call @private_atomic_called_public(%src, %out) : (!tt.ptr<f32>, !tt.ptr<f32>) -> ()
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_private_helper
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw fadd, acq_rel, gpu
// INDEXED: tt.return
tt.func private @indexed_update_private_helper(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

// -----

// INDEXED-LABEL: @indexed_update_called_public_helper
// INDEXED-NOT: tt.experimental_indexed_update_specialized
// INDEXED: tt.atomic_rmw fadd, acq_rel, gpu
// INDEXED: tt.return
tt.func public @indexed_update_called_public_helper(%indices: !tt.ptr<i32> {tt.experimental_alias_group = 0 : i32, tt.experimental_unique_indices = 16 : i32, tt.experimental_index_bound = 32 : i32}, %out: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32, tt.experimental_private = 1 : i32, tt.experimental_grid_x = 16 : i32, tt.experimental_grid_y = 2 : i32, tt.experimental_grid_z = 1 : i32, tt.experimental_num_ctas = 1 : i32, tt.experimental_initial_zero = 1 : i32}) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %cd = arith.constant 13 : i32
  %true = arith.constant true
  %values = arith.constant dense<1.0> : tensor<8xf32>
  %x = tt.get_program_id x : i32
  %y = tt.get_program_id y : i32
  %iptr = tt.addptr %indices, %x : !tt.ptr<i32>, i32
  %loaded = tt.load %iptr : !tt.ptr<i32>
  %row = arith.muli %loaded, %cd : i32
  %base = arith.muli %y, %c8 : i32
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %bases = tt.splat %base : i32 -> tensor<8xi32>
  %cols = arith.addi %bases, %range : tensor<8xi32>
  %ds = tt.splat %cd : i32 -> tensor<8xi32>
  %rowptr = tt.addptr %out, %row : !tt.ptr<f32>, i32
  %rowptrs = tt.splat %rowptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ptrs = tt.addptr %rowptrs, %cols : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %tail = arith.cmpi slt, %cols, %ds : tensor<8xi32>
  %valid = arith.cmpi sge, %x, %c0 : i32
  %valids = tt.splat %valid : i1 -> tensor<8xi1>
  %mask = arith.andi %tail, %valids : tensor<8xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %values, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}

tt.func public @indexed_update_calling_entry(%indices: !tt.ptr<i32>, %out: !tt.ptr<f32>) {
  tt.call @indexed_update_called_public_helper(%indices, %out) : (!tt.ptr<i32>, !tt.ptr<f32>) -> ()
  tt.return
}
