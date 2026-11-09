// RUN: triton-opt %s -pass-pipeline='builtin.module(convert-triton-to-tritongpu{num-warps=4 target=cuda:100},tritongpu-coalesce,tritongpu-accelerate-matmul,tritongpu-remove-layout-conversions,tritongpu-optimize-dot-operands,cse,tritongpu-fuse-nested-loops,canonicalize,tritongpu-optimize-accumulator-init,tritongpu-hoist-tmem-alloc,tritongpu-assign-latencies,tritongpu-schedule-loops,tritongpu-pipeline,triton-nvidia-gpu-remove-tmem-tokens,canonicalize)' | FileCheck %s --check-prefixes=FALLBACK,BLACKWELL
// RUN: triton-opt %s -pass-pipeline='builtin.module(convert-triton-to-tritongpu{num-warps=4 target=cuda:90 },tritongpu-coalesce,tritongpu-accelerate-matmul,tritongpu-remove-layout-conversions,tritongpu-optimize-dot-operands,cse,tritongpu-fuse-nested-loops,canonicalize,tritongpu-optimize-accumulator-init,canonicalize,tritongpu-combine-tensor-select-and-if,tritongpu-assign-latencies,tritongpu-schedule-loops,tritongpu-pipeline,canonicalize)' | FileCheck %s --check-prefixes=FALLBACK,HOPPER
// RUN: triton-opt %s -pass-pipeline='builtin.module(convert-triton-to-tritongpu{num-warps=4 target=cuda:100},tritongpu-coalesce,tritongpu-accelerate-matmul,tritongpu-remove-layout-conversions,tritongpu-optimize-dot-operands,cse,canonicalize,tritongpu-optimize-accumulator-init,tritongpu-hoist-tmem-alloc,tritongpu-assign-latencies,tritongpu-schedule-loops,tritongpu-pipeline,triton-nvidia-gpu-remove-tmem-tokens,canonicalize)' | FileCheck %s --check-prefixes=INNER,INNER-BLACKWELL
// RUN: triton-opt %s -pass-pipeline='builtin.module(convert-triton-to-tritongpu{num-warps=4 target=cuda:90 },tritongpu-coalesce,tritongpu-accelerate-matmul,tritongpu-remove-layout-conversions,tritongpu-optimize-dot-operands,cse,canonicalize,tritongpu-optimize-accumulator-init,canonicalize,tritongpu-combine-tensor-select-and-if,tritongpu-assign-latencies,tritongpu-schedule-loops,tritongpu-pipeline,canonicalize)' | FileCheck %s --check-prefixes=INNER,INNER-HOPPER

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

// Flattening puts descriptor reads and writes into one loop. Without backing
// storage contracts, that mixed loop must keep its original global order.
// FALLBACK-LABEL: @matmul_kernel_tma_persistent
// FALLBACK-NOT: async_tma_copy
// FALLBACK: scf.for
// FALLBACK: tt.descriptor_load
// FALLBACK: tt.descriptor_load
// BLACKWELL: ttng.tc_gen5_mma
// BLACKWELL-NOT: ttng.tc_gen5_commit
// HOPPER: ttng.warp_group_dot
// HOPPER-NOT: isAsync = true
// FALLBACK: tt.descriptor_store
// FALLBACK-NOT: async_tma_copy
// FALLBACK: tt.return

// Without flattening, the outer loop is declined, while its read-only inner
// loop remains eligible for TMA/MMA pipelining. Declining the outer loop must
// not erase the inner loop's independently valid schedule.
// INNER-LABEL: @matmul_kernel_tma_persistent
// INNER: scf.for
// INNER: ttng.async_tma_copy_global_to_local
// INNER: scf.for
// INNER: ttng.wait_barrier
// INNER-BLACKWELL: ttng.tc_gen5_mma
// INNER-HOPPER: ttng.warp_group_dot {{.*}} isAsync = true
// INNER: tt.descriptor_store
// INNER: tt.return
tt.func public @matmul_kernel_tma_persistent(%arg0: !tt.ptr<i8, "descriptor">, %arg1: !tt.ptr<i8, "descriptor">, %arg2: !tt.ptr<i8, "descriptor">, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
  %c63_i32 = arith.constant 63 : i32
  %c127_i32 = arith.constant 127 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c8_i32 = arith.constant 8 : i32
  %c132_i32 = arith.constant 132 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.addi %arg3, %c127_i32 : i32
  %2 = arith.divsi %1, %c128_i32 : i32
  %3 = arith.addi %arg4, %c127_i32 : i32
  %4 = arith.divsi %3, %c128_i32 : i32
  %5 = arith.addi %arg5, %c63_i32 : i32
  %6 = arith.divsi %5, %c64_i32 : i32
  %7 = arith.muli %2, %4 : i32
  %8 = arith.subi %0, %c132_i32 : i32
  %9 = arith.muli %4, %c8_i32 : i32


  %10 = scf.for %arg6 = %0 to %7 step %c132_i32 iter_args(%arg7 = %8) -> (i32)  : i32 {
    %11 = arith.divsi %arg6, %9 : i32
    %12 = arith.muli %11, %c8_i32 : i32
    %13 = arith.subi %2, %12 : i32
    %14 = arith.minsi %13, %c8_i32 : i32
    %15 = arith.remsi %arg6, %14 : i32
    %16 = arith.addi %12, %15 : i32
    %17 = arith.remsi %arg6, %9 : i32
    %18 = arith.divsi %17, %14 : i32
    %19 = arith.muli %16, %c128_i32 : i32
    %20 = arith.muli %18, %c128_i32 : i32
    %21 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %cst) -> (tensor<128x128xf32>)  : i32 {
      %35 = arith.muli %arg8, %c64_i32 : i32
      %36 = ttng.reinterpret_tensor_descriptor %arg0 : !tt.ptr<i8, "descriptor"> to !tt.tensordesc<128x64xf16, #shared>
      %37 = tt.descriptor_load %36[%19, %35] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16>
      %38 = ttng.reinterpret_tensor_descriptor %arg1 : !tt.ptr<i8, "descriptor"> to !tt.tensordesc<128x64xf16, #shared>
      %39 = tt.descriptor_load %38[%20, %35] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16>

      %40 = tt.trans %39 {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %41 = tt.dot %37, %40, %arg9, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
      scf.yield %41 : tensor<128x128xf32>
    }

    %22 = arith.addi %arg7, %c132_i32 : i32
    %23 = arith.divsi %22, %9 : i32
    %24 = arith.muli %23, %c8_i32 : i32
    %25 = arith.subi %2, %24 : i32
    %26 = arith.minsi %25, %c8_i32 : i32
    %27 = arith.remsi %22, %26 : i32
    %28 = arith.addi %24, %27 : i32
    %29 = arith.remsi %22, %9 : i32
    %30 = arith.divsi %29, %26 : i32
    %31 = arith.muli %28, %c128_i32 : i32
    %32 = arith.muli %30, %c128_i32 : i32
    %33 = arith.truncf %21 : tensor<128x128xf32> to tensor<128x128xf16>
    %34 = ttng.reinterpret_tensor_descriptor %arg2 : !tt.ptr<i8, "descriptor"> to !tt.tensordesc<128x128xf16, #shared>
    tt.descriptor_store %34[%31, %32], %33 : !tt.tensordesc<128x128xf16, #shared>, tensor<128x128xf16>
    scf.yield %22 : i32
  } {tt.flatten}
  tt.return
}
