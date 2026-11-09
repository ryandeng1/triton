// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies="num-stages=3 contract-pipeline-stages=2" -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | FileCheck %s --check-prefix=CONTRACT
// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies="num-stages=3 contract-pipeline-stages=2" | FileCheck %s --check-prefix=TRIP
// RUN: triton-opt %s -split-input-file -tritongpu-pipeline | FileCheck %s --check-prefix=RAW
// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// CHECK-LABEL: tt.func @load_two_users
  tt.func @load_two_users(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_0 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: scf.for
    // CHECK:   ttg.async_wait {{.*}} {num = 1 : i32}
    // CHECK:   tt.dot
    // CHECK:   tt.dot
    // CHECK:   ttg.async_copy_global_to_local
    // CHECK:   scf.yield
    // CHECK: ttg.async_wait {num = 0 : i32}

    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = ttg.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = ttg.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = ttg.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared, #smem>
      %25 = ttg.memdesc_trans %24 {order=array<i32: 1,0>} : !ttg.memdesc<64x16xf16, #shared, #smem> -> !ttg.memdesc<16x64xf16, #shared1, #smem>
      %26 = ttg.local_load %25 : !ttg.memdesc<16x64xf16, #shared1, #smem> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %21, %27 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %17#0, %17#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----

// CHECK-NOT:  ttg.convert_layout {{.*}} : tensor<32x64xf32, #shared> -> tensor<32x64xf32, #shared1>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.get_program_id y : i32
    %3 = tt.load %arg3 : !tt.ptr<i64>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.splat %1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.addi %5, %4 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %8 = tt.splat %3 : i64 -> tensor<64x1xi64, #blocked>
    %9 = arith.extsi %7 : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
    %10 = arith.addi %8, %9 : tensor<64x1xi64, #blocked>
    %11 = arith.extsi %arg5 : i32 to i64
    %12 = tt.splat %11 : i64 -> tensor<64x1xi64, #blocked>
    %13 = arith.muli %10, %12 : tensor<64x1xi64, #blocked>
    %14 = arith.muli %2, %arg5 : i32
    %15 = arith.extsi %14 : i32 to i64
    %16 = tt.splat %15 : i64 -> tensor<64x1xi64, #blocked>
    %17 = arith.addi %13, %16 : tensor<64x1xi64, #blocked>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.expand_dims %18 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %21 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %22 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked>
    %23 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %24 = arith.muli %20, %22 : tensor<1x64xi32, #blocked>
    %25 = arith.muli %21, %23 : tensor<1x64xi32, #blocked1>
    %26 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %27 = arith.extsi %24 : tensor<1x64xi32, #blocked> to tensor<1x64xi64, #blocked>
    %28 = arith.extsi %25 : tensor<1x64xi32, #blocked1> to tensor<1x64xi64, #blocked1>
    %29 = tt.broadcast %27 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %30 = arith.addi %26, %29 : tensor<64x64xi64, #blocked>
    %31 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %33 = tt.splat %3 : i64 -> tensor<32x1xi64, #blocked1>
    %34 = arith.extsi %32 : tensor<32x1xi32, #blocked1> to tensor<32x1xi64, #blocked1>
    %35 = arith.addi %33, %34 : tensor<32x1xi64, #blocked1>
    %36 = tt.splat %11 : i64 -> tensor<32x1xi64, #blocked1>
    %37 = arith.muli %35, %36 : tensor<32x1xi64, #blocked1>
    %38 = tt.splat %15 : i64 -> tensor<32x1xi64, #blocked1>
    %39 = arith.addi %37, %38 : tensor<32x1xi64, #blocked1>
    %40 = tt.broadcast %39 : tensor<32x1xi64, #blocked1> -> tensor<32x64xi64, #blocked1>
    %41 = tt.broadcast %28 : tensor<1x64xi64, #blocked1> -> tensor<32x64xi64, #blocked1>
    %42 = arith.addi %40, %41 : tensor<32x64xi64, #blocked1>
    %43 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %44 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %45 = tt.expand_dims %43 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %46 = tt.expand_dims %44 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %47 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked1>
    %48 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked>
    %49 = arith.muli %45, %47 : tensor<1x32xi32, #blocked1>
    %50 = arith.muli %46, %48 : tensor<1x32xi32, #blocked>
    %51 = tt.broadcast %39 : tensor<32x1xi64, #blocked1> -> tensor<32x32xi64, #blocked1>
    %52 = arith.extsi %49 : tensor<1x32xi32, #blocked1> to tensor<1x32xi64, #blocked1>
    %53 = arith.extsi %50 : tensor<1x32xi32, #blocked> to tensor<1x32xi64, #blocked>
    %54 = tt.broadcast %52 : tensor<1x32xi64, #blocked1> -> tensor<32x32xi64, #blocked1>
    %55 = arith.addi %51, %54 : tensor<32x32xi64, #blocked1>
    %56 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %57 = tt.addptr %56, %30 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi64, #blocked>
    %58 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked1>
    %59 = tt.addptr %58, %42 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi64, #blocked1>
    %60 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %61 = tt.addptr %60, %55 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi64, #blocked1>
    %62 = tt.load %57 : tensor<64x64x!tt.ptr<f32>, #blocked>
    %63 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c32_i32 iter_args(%arg7 = %cst) -> (tensor<64x32xf32, #mma>)  : i32 {
      %70 = tt.load %59 : tensor<32x64x!tt.ptr<f32>, #blocked1>
      %71 = ttg.convert_layout %62 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %72 = ttg.local_alloc %70 : (tensor<32x64xf32, #blocked1>) -> !ttg.memdesc<32x64xf32, #shared, #smem>
      %73 = ttg.memdesc_trans %72 {order=array<i32: 1,0>} : !ttg.memdesc<32x64xf32, #shared, #smem> -> !ttg.memdesc<64x32xf32, #shared1, #smem>
      %74 = ttg.local_load %73 : !ttg.memdesc<64x32xf32, #shared1, #smem> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %75 = tt.dot %71, %74, %cst, inputPrecision = tf32 : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      %76 = tt.load %61 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %77 = ttg.convert_layout %75 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %78 = ttg.convert_layout %76 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %79 = tt.dot %77, %78, %arg7, inputPrecision = tf32 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      scf.yield %79 : tensor<64x32xf32, #mma>
    }
    %64 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %65 = tt.broadcast %53 : tensor<1x32xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %66 = arith.addi %64, %65 : tensor<64x32xi64, #blocked>
    %67 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked>
    %68 = tt.addptr %67, %66 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi64, #blocked>
    %69 = ttg.convert_layout %63 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #blocked>
    tt.store %68, %69 : tensor<64x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
} // end module

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
//   CHECK-LABEL: @matmul_tma
//     CHECK-DAG:   ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #{{.+}}, #smem, mutable>
//     CHECK-DAG:   ttg.local_alloc : () -> !ttg.memdesc<3x64x256xf16, #{{.+}}, #smem, mutable>
//     CHECK-DAG:   ttg.local_alloc : () -> !ttg.memdesc<3x1xi64, #{{.+}}, #smem, mutable>
// CHECK-COUNT-3:   ttng.init_barrier
// CHECK-COUNT-4:   ttng.async_tma_copy_global_to_local
//         CHECK:   scf.for
//         CHECK:     ttng.wait_barrier
//     CHECK-NOT:     ttng.wait_barrier
// CHECK-COUNT-2:     ttng.async_tma_copy_global_to_local
//         CHECK:     scf.yield
  tt.func public @matmul_tma(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x256xf16, #shared>) -> tensor<128x256xf32, #mma> {
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0:2 = scf.for %arg3 = %c0_i32 to %c256_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
      %1 = tt.descriptor_load %arg0[%c0_i32, %arg5] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %2 = ttg.local_alloc %1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %3 = tt.descriptor_load %arg1[%arg5, %c0_i32] : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
      %4 = ttg.local_alloc %3 : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %5 = ttng.warp_group_dot %2, %4, %arg4 { inputPrecision = 0 : i32 } : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %6 = arith.addi %arg5, %c64_i32 : i32
      scf.yield %5, %6 : tensor<128x256xf32, #mma>, i32
    }
    tt.return %0#0 : tensor<128x256xf32, #mma>
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @guard_same_root
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.return
  tt.func @guard_same_root(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c128 = arith.constant 128 : i32
    %one = arith.constant dense<1.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %offset = arith.muli %i, %c128 : i32
      %base = tt.addptr %input, %offset : !tt.ptr<f32>, i32
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %ptr = tt.addptr %splat, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %x = tt.load %ptr : tensor<128x!tt.ptr<f32>, #guard>
      %next = arith.addi %offset, %c128 : i32
      %out = tt.addptr %input, %next : !tt.ptr<f32>, i32
      %os = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %op = tt.addptr %os, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %y = arith.addf %x, %one : tensor<128xf32, #guard>
      tt.store %op, %y : tensor<128x!tt.ptr<f32>, #guard>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @guard_missing_groups
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.return
  tt.func @guard_missing_groups(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c128 = arith.constant 128 : i32
    %one = arith.constant dense<1.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %offset = arith.muli %i, %c128 : i32
      %base = tt.addptr %input, %offset : !tt.ptr<f32>, i32
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %ptr = tt.addptr %splat, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %x = tt.load %ptr : tensor<128x!tt.ptr<f32>, #guard>
      %next = arith.addi %offset, %c128 : i32
      %out = tt.addptr %output, %next : !tt.ptr<f32>, i32
      %os = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %op = tt.addptr %os, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %y = arith.addf %x, %one : tensor<128xf32, #guard>
      tt.store %op, %y : tensor<128x!tt.ptr<f32>, #guard>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @guard_equal_groups
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.return
  tt.func @guard_equal_groups(%input: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %output: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c128 = arith.constant 128 : i32
    %one = arith.constant dense<1.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %offset = arith.muli %i, %c128 : i32
      %base = tt.addptr %input, %offset : !tt.ptr<f32>, i32
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %ptr = tt.addptr %splat, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %x = tt.load %ptr : tensor<128x!tt.ptr<f32>, #guard>
      %next = arith.addi %offset, %c128 : i32
      %out = tt.addptr %output, %next : !tt.ptr<f32>, i32
      %os = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %op = tt.addptr %os, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %y = arith.addf %x, %one : tensor<128xf32, #guard>
      tt.store %op, %y : tensor<128x!tt.ptr<f32>, #guard>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @guard_disjoint_groups
// CHECK: ttg.async_copy_global_to_local
// CHECK: tt.return
  tt.func @guard_disjoint_groups(%input: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %output: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c128 = arith.constant 128 : i32
    %one = arith.constant dense<1.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %offset = arith.muli %i, %c128 : i32
      %base = tt.addptr %input, %offset : !tt.ptr<f32>, i32
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %ptr = tt.addptr %splat, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %x = tt.load %ptr : tensor<128x!tt.ptr<f32>, #guard>
      %next = arith.addi %offset, %c128 : i32
      %out = tt.addptr %output, %next : !tt.ptr<f32>, i32
      %os = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %op = tt.addptr %os, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %y = arith.addf %x, %one : tensor<128xf32, #guard>
      tt.store %op, %y : tensor<128x!tt.ptr<f32>, #guard>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @guard_immutable
// CHECK: ttg.async_copy_global_to_local
// CHECK: tt.return
  tt.func @guard_immutable(%input: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %output: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c128 = arith.constant 128 : i32
    %one = arith.constant dense<1.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %offset = arith.muli %i, %c128 : i32
      %base = tt.addptr %input, %offset : !tt.ptr<f32>, i32
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %ptr = tt.addptr %splat, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %x = tt.load %ptr : tensor<128x!tt.ptr<f32>, #guard>
      %next = arith.addi %offset, %c128 : i32
      %out = tt.addptr %output, %next : !tt.ptr<f32>, i32
      %os = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %op = tt.addptr %os, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %y = arith.addf %x, %one : tensor<128xf32, #guard>
      tt.store %op, %y : tensor<128x!tt.ptr<f32>, #guard>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @guard_volatile
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.return
  tt.func @guard_volatile(%input: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %output: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c128 = arith.constant 128 : i32
    %one = arith.constant dense<1.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %offset = arith.muli %i, %c128 : i32
      %base = tt.addptr %input, %offset : !tt.ptr<f32>, i32
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %ptr = tt.addptr %splat, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %x = tt.load %ptr {isVolatile = true} : tensor<128x!tt.ptr<f32>, #guard>
      %next = arith.addi %offset, %c128 : i32
      %out = tt.addptr %output, %next : !tt.ptr<f32>, i32
      %os = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %op = tt.addptr %os, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %y = arith.addf %x, %one : tensor<128xf32, #guard>
      tt.store %op, %y : tensor<128x!tt.ptr<f32>, #guard>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @guard_two_alias_stores
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.return
  tt.func @guard_two_alias_stores(%input: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %output: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c128 = arith.constant 128 : i32
    %one = arith.constant dense<1.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %offset = arith.muli %i, %c128 : i32
      %base = tt.addptr %input, %offset : !tt.ptr<f32>, i32
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %ptr = tt.addptr %splat, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %x = tt.load %ptr : tensor<128x!tt.ptr<f32>, #guard>
      %next = arith.addi %offset, %c128 : i32
      %out = tt.addptr %output, %next : !tt.ptr<f32>, i32
      %os = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %op = tt.addptr %os, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %y = arith.addf %x, %one : tensor<128xf32, #guard>
      tt.store %op, %y : tensor<128x!tt.ptr<f32>, #guard>
      tt.store %op, %x : tensor<128x!tt.ptr<f32>, #guard>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @guard_auto_disjoint
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.return
// CONTRACT-LABEL: @guard_auto_disjoint
// CONTRACT: ttg.async_copy_global_to_local
// CONTRACT: tt.return
  tt.func @guard_auto_disjoint(%input: !tt.ptr<f32> {tt.experimental_alias_group = 0 : i32}, %output: !tt.ptr<f32> {tt.experimental_alias_group = 1 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c128 = arith.constant 128 : i32
    %one = arith.constant dense<1.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %offset = arith.muli %i, %c128 : i32
      %base = tt.addptr %input, %offset : !tt.ptr<f32>, i32
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %ptr = tt.addptr %splat, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %x = tt.load %ptr : tensor<128x!tt.ptr<f32>, #guard>
      %next = arith.addi %offset, %c128 : i32
      %out = tt.addptr %output, %next : !tt.ptr<f32>, i32
      %os = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
      %op = tt.addptr %os, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
      %y = arith.addf %x, %one : tensor<128xf32, #guard>
      tt.store %op, %y : tensor<128x!tt.ptr<f32>, #guard>
    }
    tt.return
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @guard_raw_schedule
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.return
// RAW-LABEL: @guard_raw_schedule
// RAW-NOT: ttg.async_copy_global_to_local
// RAW-NOT: loop.stage
// RAW: tt.return
  tt.func @guard_raw_schedule(%p: tensor<128x!tt.ptr<f32>, #guard>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %one = arith.constant dense<1.0> : tensor<128xf32, #guard>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %x = tt.load %p {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #guard>
      %y = arith.addf %x, %one {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #guard>
      tt.store %p, %y {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x!tt.ptr<f32>, #guard>
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32}
    tt.return
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Even without a local store, a raw barrier can publish global data from another
// region. An existing serialized schedule is not a pre-protocol memory proof.
// CHECK-LABEL: @guard_raw_publication
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: ttng.wait_barrier
// CHECK: tt.return
// RAW-LABEL: @guard_raw_publication
// RAW-NOT: ttg.async_copy_global_to_local
// RAW-NOT: loop.stage
// RAW: ttng.wait_barrier
// RAW: tt.return
  tt.func @guard_raw_publication(%p: tensor<128x!tt.ptr<f32>, #guard>, %barrier: !ttg.memdesc<1xi64, #bar, #ttg.shared_memory, mutable>, %n: i32) -> tensor<128xf32, #guard> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      ttng.wait_barrier %barrier, %c0 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1xi64, #bar, #ttg.shared_memory, mutable>
      %x = tt.load %p {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #guard>
      %y = arith.addf %x, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #guard>
      scf.yield %y : tensor<128xf32, #guard>
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32}
    tt.return %result : tensor<128xf32, #guard>
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @unknown_global_effect(!tt.ptr<f32>)
// CHECK-LABEL: @guard_unknown_call
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.call
// CHECK: tt.return
  tt.func @guard_unknown_call(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %n: i32) -> tensor<128xf32, #guard> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
    %ptrs = tt.addptr %base, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      %x = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #guard>
      tt.call @unknown_global_effect(%p) : (!tt.ptr<f32>) -> ()
      %y = arith.addf %acc, %x : tensor<128xf32, #guard>
      scf.yield %y : tensor<128xf32, #guard>
    } {tt.num_stages = 3 : i32}
    tt.return %result : tensor<128xf32, #guard>
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// A stale internal proof never licenses newly introduced arbitrary global
// descriptor writes. Only the compiler's proven private scratch is exempt.
// CHECK-LABEL: @guard_marked_external_descriptor_create
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: ttng.tensormap_create
// CHECK: tt.return
// RAW-LABEL: @guard_marked_external_descriptor_create
// RAW-NOT: ttg.async_copy_global_to_local
// RAW-NOT: loop.stage
// RAW: ttng.tensormap_create
// RAW: tt.return
  tt.func @guard_marked_external_descriptor_create(%p: tensor<128x!tt.ptr<f32>, #guard>, %desc: !tt.ptr<i8>, %source: !tt.ptr<i16>, %n: i32) -> tensor<128xf32, #guard> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c256 = arith.constant 256 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      %x = tt.load %p {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #guard>
      ttng.tensormap_create %desc, %source, [%c256], [%n], [], [%c1] {elem_type = 3 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : (!tt.ptr<i8>, !tt.ptr<i16>, i32, i32, i32) -> ()
      %y = arith.addf %x, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #guard>
      scf.yield %y : tensor<128xf32, #guard>
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32, ttg.global_pipeline_preprotocol_proof}
    tt.return %result : tensor<128xf32, #guard>
  }

// CHECK-LABEL: @guard_marked_external_descriptor_fence
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: ttng.tensormap_fenceproxy_acquire
// CHECK: tt.return
// RAW-LABEL: @guard_marked_external_descriptor_fence
// RAW-NOT: ttg.async_copy_global_to_local
// RAW-NOT: loop.stage
// RAW: ttng.tensormap_fenceproxy_acquire
// RAW: tt.return
  tt.func @guard_marked_external_descriptor_fence(%p: tensor<128x!tt.ptr<f32>, #guard>, %desc: !tt.ptr<i8>, %n: i32) -> tensor<128xf32, #guard> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      %x = tt.load %p {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #guard>
      ttng.tensormap_fenceproxy_acquire %desc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.ptr<i8>
      %y = arith.addf %x, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #guard>
      scf.yield %y : tensor<128xf32, #guard>
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32, ttg.global_pipeline_preprotocol_proof}
    tt.return %result : tensor<128xf32, #guard>
  }
}

// -----

#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// A pre-protocol proof cannot override volatility introduced on an async load.
// CHECK-LABEL: @guard_marked_volatile_async
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: scf.for
// CHECK: ttg.async_copy_global_to_local
// CHECK-SAME: isVolatile = true
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: tt.return
// RAW-LABEL: @guard_marked_volatile_async
// RAW-NOT: ttg.async_copy_global_to_local
// RAW: scf.for
// RAW-NOT: loop.stage
// RAW: ttg.async_copy_global_to_local
// RAW-SAME: isVolatile = true
// RAW-NOT: ttg.async_copy_global_to_local
// RAW-NOT: loop.stage
// RAW: tt.return
  tt.func @guard_marked_volatile_async(%p: tensor<128x!tt.ptr<f32>, #guard>, %n: i32) -> tensor<128xf32, #guard> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      %copy = ttg.async_copy_global_to_local %p, %buffer {isVolatile = true, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #guard> -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
      %commit = ttg.async_commit_group tokens %copy {loop.cluster = 1 : i32, loop.stage = 0 : i32}
      %wait = ttg.async_wait %commit {num = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %x = ttg.local_load %buffer token %wait {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #guard>
      %y = arith.addf %x, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xf32, #guard>
      scf.yield %y : tensor<128xf32, #guard>
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32, ttg.global_pipeline_preprotocol_proof}
    tt.return %result : tensor<128xf32, #guard>
  }
}

// -----

// Inspect inference before canonicalization folds the zero/single-trip loops.
#guard = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// TRIP-LABEL: @contract_trip_zero
// TRIP: scf.for
// TRIP-NOT: tt.experimental_contract_pipeline
// TRIP-NOT: tt.num_stages
// TRIP: tt.return
  tt.func @contract_trip_zero(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}) -> tensor<128xf32, #guard> {
    %lower = arith.constant 0 : i32
    %upper = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
    %ptrs = tt.addptr %base, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
    %result = scf.for %i = %lower to %upper step %one iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      %x = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #guard>
      %sum = arith.addf %acc, %x : tensor<128xf32, #guard>
      scf.yield %sum : tensor<128xf32, #guard>
    }
    tt.return %result : tensor<128xf32, #guard>
  }

// TRIP-LABEL: @contract_trip_one_negative_bounds
// TRIP: scf.for
// TRIP-NOT: tt.experimental_contract_pipeline
// TRIP-NOT: tt.num_stages
// TRIP: tt.return
  tt.func @contract_trip_one_negative_bounds(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}) -> tensor<128xf32, #guard> {
    %lower = arith.constant -3 : i32
    %upper = arith.constant -2 : i32
    %one = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
    %ptrs = tt.addptr %base, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
    %result = scf.for %i = %lower to %upper step %one iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      %x = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #guard>
      %sum = arith.addf %acc, %x : tensor<128xf32, #guard>
      scf.yield %sum : tensor<128xf32, #guard>
    }
    tt.return %result : tensor<128xf32, #guard>
  }

// TRIP-LABEL: @contract_trip_two
// TRIP: scf.for
// TRIP: tt.experimental_contract_pipeline = 2 : i32
// TRIP: tt.return
  tt.func @contract_trip_two(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}) -> tensor<128xf32, #guard> {
    %lower = arith.constant 0 : i32
    %upper = arith.constant 2 : i32
    %one = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
    %ptrs = tt.addptr %base, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
    %result = scf.for %i = %lower to %upper step %one iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      %x = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #guard>
      %sum = arith.addf %acc, %x : tensor<128xf32, #guard>
      scf.yield %sum : tensor<128xf32, #guard>
    }
    tt.return %result : tensor<128xf32, #guard>
  }

// TRIP-LABEL: @contract_trip_signed_max_zero
// TRIP: scf.for
// TRIP-NOT: tt.experimental_contract_pipeline
// TRIP-NOT: tt.num_stages
// TRIP: tt.return
  tt.func @contract_trip_signed_max_zero(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}) -> tensor<128xf32, #guard> {
    %lower = arith.constant 9223372036854775807 : i64
    %upper = arith.constant 9223372036854775807 : i64
    %one = arith.constant 1 : i64
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
    %ptrs = tt.addptr %base, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
    %result = scf.for %i = %lower to %upper step %one iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i64 {
      %x = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #guard>
      %sum = arith.addf %acc, %x : tensor<128xf32, #guard>
      scf.yield %sum : tensor<128xf32, #guard>
    }
    tt.return %result : tensor<128xf32, #guard>
  }

// TRIP-LABEL: @contract_trip_unsigned_max_one
// TRIP: scf.for
// TRIP-NOT: tt.experimental_contract_pipeline
// TRIP-NOT: tt.num_stages
// TRIP: tt.return
  tt.func @contract_trip_unsigned_max_one(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}) -> tensor<128xf32, #guard> {
    %lower = arith.constant -2 : i64
    %upper = arith.constant -1 : i64
    %one = arith.constant 1 : i64
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
    %ptrs = tt.addptr %base, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
    %result = scf.for unsigned %i = %lower to %upper step %one iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i64 {
      %x = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #guard>
      %sum = arith.addf %acc, %x : tensor<128xf32, #guard>
      scf.yield %sum : tensor<128xf32, #guard>
    }
    tt.return %result : tensor<128xf32, #guard>
  }

// TRIP-LABEL: @contract_trip_dynamic
// TRIP: scf.for
// TRIP: tt.experimental_contract_pipeline = 2 : i32
// TRIP: tt.return
  tt.func @contract_trip_dynamic(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}, %upper: i32) -> tensor<128xf32, #guard> {
    %lower = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
    %ptrs = tt.addptr %base, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
    %result = scf.for %i = %lower to %upper step %one iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      %x = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #guard>
      %sum = arith.addf %acc, %x : tensor<128xf32, #guard>
      scf.yield %sum : tensor<128xf32, #guard>
    }
    tt.return %result : tensor<128xf32, #guard>
  }

// TRIP-LABEL: @contract_trip_explicit_one
// TRIP: scf.for
// TRIP-NOT: tt.experimental_contract_pipeline
// TRIP: tt.num_stages = 3 : i32
// TRIP: tt.return
  tt.func @contract_trip_explicit_one(%p: !tt.ptr<f32> {tt.experimental_immutable = 1 : i32}) -> tensor<128xf32, #guard> {
    %lower = arith.constant 0 : i32
    %upper = arith.constant 1 : i32
    %one = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32, #guard>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #guard>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #guard>
    %ptrs = tt.addptr %base, %cols : tensor<128x!tt.ptr<f32>, #guard>, tensor<128xi32, #guard>
    %result = scf.for %i = %lower to %upper step %one iter_args(%acc = %zero) -> tensor<128xf32, #guard> : i32 {
      %x = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #guard>
      %sum = arith.addf %acc, %x : tensor<128xf32, #guard>
      scf.yield %sum : tensor<128xf32, #guard>
    } {tt.num_stages = 3 : i32}
    tt.return %result : tensor<128xf32, #guard>
  }
}
