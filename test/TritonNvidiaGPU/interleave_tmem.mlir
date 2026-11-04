// RUN: triton-opt %s --triton-nvidia-interleave-tmem --allow-unregistered-dialect | FileCheck %s
// RUN: triton-opt %s --triton-nvidia-gpu-schedule-tmem-releases --allow-unregistered-dialect | FileCheck %s --check-prefix=RELEASE
// RUN: triton-opt %s --triton-nvidia-gpu-pack-tmem --cse --allow-unregistered-dialect | FileCheck %s --check-prefix=PACK

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#linear64 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#linear128 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

#release_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#release_linear32 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 0]], block = []}>
#release_scalar = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>


#pack_smem_a = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#pack_smem_b = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {

tt.func public @sink_load(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                          %arg1: tensor<128x128xf16, #blocked>,
                          %arg2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>)
                          -> (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x128xf16, #blocked>) {

  // CHECK: ttg.local_alloc
  // CHECK: ttng.tmem_load
  // CHECK: ttg.convert_layout
  // CHECK: arith.truncf
  %subslice0 = ttng.tmem_subslice %arg0 {offset = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
  %subtile0 = ttng.tmem_load %subslice0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #linear64>
  %outLHS = ttg.convert_layout %subtile0 : tensor<128x64xf32, #linear64> -> tensor<128x64xf32, #blocked>
  %subslice1 = ttng.tmem_subslice %arg0 {offset = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
  %subtile1 = ttng.tmem_load %subslice1 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #linear64>
  %outRHS = ttg.convert_layout %subtile1 : tensor<128x64xf32, #linear64> -> tensor<128x64xf32, #blocked>

  // CHECK: ttng.tmem_load
  // CHECK: ttg.convert_layout
  // CHECK: ttng.tmem_store
  // CHECK: arith.truncf
  %4 = ttg.local_alloc %arg1 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
  %5 = arith.truncf %outLHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

  %true = arith.constant true
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear128>
  ttng.tmem_store %cst, %arg2, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %6 = arith.truncf %outRHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

  // CHECK: ttng.tmem_load
  // CHECK: ttg.convert_layout
  // CHECK: "unknow_may_side_effect"() : () -> ()
  // CHECK: arith.truncf
  %7 = ttng.tmem_load %arg2 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  %8 = ttg.convert_layout %7 : tensor<128x128xf32, #linear128> -> tensor<128x128xf32, #blocked>
  "unknow_may_side_effect"() : () -> ()
  %9 = arith.truncf %8 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>

  ttg.local_dealloc %4 : !ttg.memdesc<128x128xf16, #shared, #smem>
  tt.return %5, %6, %9 : tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x128xf16, #blocked>
}

// CHECK-LABEL: @interleave_load_store_ws
tt.func @interleave_load_store_ws() {
  %0 = ttng.tmem_alloc : () -> (!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>)
  ttg.warp_specialize(%0)
  default{
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0(%arg0: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(8) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %alpha = arith.constant dense<0.5> : tensor<128x64xf32, #linear64>
    %true = arith.constant true

    // CHECK: scf.for
    scf.for %i = %c0 to %c32 step %c1 : i32 {
      // CHECK: memdesc_index
      %cur_acc = ttg.memdesc_index %arg0[%i] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: [[S0:%.+]] = ttng.tmem_subslice %{{.+}} {offset = 0 : i32}
      // CHECK-NEXT: [[S1:%.+]] = ttng.tmem_subslice %{{.+}} {offset = 64 : i32}

      // CHECK-NEXT: [[L0:%.+]] = ttng.tmem_load [[S0]]
      // CHECK-NEXT: [[M0:%.+]] = arith.mulf [[L0]]
      // CHECK-NEXT: ttng.tmem_store [[M0]], [[S0]]
      %slice0 = ttng.tmem_subslice %cur_acc {offset = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
      %val0 = ttng.tmem_load %slice0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #linear64>
      %mul0 = arith.mulf %val0, %alpha : tensor<128x64xf32, #linear64>

      // CHECK-NEXT: [[L1:%.+]] = ttng.tmem_load [[S1]]
      // CHECK-NEXT: [[M1:%.+]] = arith.mulf [[L1]]
      // CHECK-NEXT: ttng.tmem_store [[M1]], [[S1]]
      %slice1 = ttng.tmem_subslice %cur_acc {offset = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
      %val1 = ttng.tmem_load %slice1 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #linear64>
      %mul1 = arith.mulf %val1, %alpha : tensor<128x64xf32, #linear64>

      ttng.tmem_store %mul0, %slice0, %true : tensor<128x64xf32, #linear64> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
      ttng.tmem_store %mul1, %slice1, %true : tensor<128x64xf32, #linear64> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>

    }
    ttg.warp_return
  } : (!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
  tt.return
}

// CHECK-LABEL: @arrive_barrier
tt.func @arrive_barrier(%arg0: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %cst = arith.constant dense<0.0> : tensor<128x128xf32, #linear128>

  // CHECK-COUNT-2: ttng.tmem_alloc
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %noalias_alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: tmem_store
  // CHECK-NEXT: tmem_load
  %0 = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.tmem_store %cst, %noalias_alloc, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: arrive_barrier
  ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  "user"(%0) : (tensor<128x128xf32, #linear128>) -> ()
  tt.return
}

// CHECK-LABEL: @sink_alloc_op
tt.func @sink_alloc_op(%arg0: tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %true = arith.constant true

  %alloc0 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %subview0 = ttg.memdesc_index %alloc0[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK: [[ALLOC1:%.+]] = ttng.tmem_alloc
  %alloc1 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK: [[SUBVIEW1:%.+]] = ttg.memdesc_index [[ALLOC1]]
  %subview1 = ttg.memdesc_index %alloc1[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: tmem_store %arg0, [[SUBVIEW1]]
  ttng.tmem_store %arg0, %subview1, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[ALLOC0:%.+]] = ttng.tmem_alloc
  // CHECK: [[SUBVIEW0:%.+]] = ttg.memdesc_index [[ALLOC0]]
  // CHECK-NEXT: tmem_store %arg0, [[SUBVIEW0]]
  ttng.tmem_store %arg0, %subview0, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}

// Native release scheduling regression cases.


// RELEASE-LABEL: @release_correlated_parity
// RELEASE-COUNT-2: ttng.tmem_load
// RELEASE-NOT: ttng.tmem_load
// RELEASE: ttng.arrive_barrier
// RELEASE-COUNT-6: ttng.tmem_load
// RELEASE-NOT: ttng.tmem_load
// RELEASE: tt.return

tt.func @release_correlated_parity(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %iteration: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %pool = ttng.tmem_alloc : () -> !ttg.memdesc<128x512xf32, #release_tmem, #ttng.tensor_memory, mutable>
  %release16 = ttng.tmem_subslice %pool {offset = 208 : i32} : !ttg.memdesc<128x512xf32, #release_tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x16xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  %release32 = ttng.tmem_subslice %pool {offset = 224 : i32} : !ttg.memdesc<128x512xf32, #release_tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  %parity = arith.andi %iteration, %c1 : i32
  %odd = arith.cmpi ne, %parity, %c0 : i32
  %even = arith.cmpi eq, %parity, %c0 : i32
  %acc = scf.if %odd -> (!ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>) {
    %right = ttng.tmem_subslice %pool {offset = 208 : i32} : !ttg.memdesc<128x512xf32, #release_tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %right : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  } else {
    %left = ttng.tmem_subslice %pool {offset = 0 : i32} : !ttg.memdesc<128x512xf32, #release_tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %left : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  }
  %part0 = scf.if %even -> (!ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>) {
    %hi = ttng.tmem_subslice %acc {offset = 224 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %hi : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  } else {
    %lo = ttng.tmem_subslice %acc {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %lo : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  }
  %read0 = ttng.tmem_load %part0 : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> tensor<128x32xf32, #release_linear32>
  %part1 = scf.if %even -> (!ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>) {
    %hi = ttng.tmem_subslice %acc {offset = 192 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %hi : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  } else {
    %lo = ttng.tmem_subslice %acc {offset = 32 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %lo : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  }
  %read1 = ttng.tmem_load %part1 : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> tensor<128x32xf32, #release_linear32>
  %part2 = scf.if %even -> (!ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>) {
    %hi = ttng.tmem_subslice %acc {offset = 160 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %hi : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  } else {
    %lo = ttng.tmem_subslice %acc {offset = 64 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %lo : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  }
  %read2 = ttng.tmem_load %part2 : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> tensor<128x32xf32, #release_linear32>
  %part3 = scf.if %even -> (!ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>) {
    %hi = ttng.tmem_subslice %acc {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %hi : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  } else {
    %lo = ttng.tmem_subslice %acc {offset = 96 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %lo : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  }
  %read3 = ttng.tmem_load %part3 : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> tensor<128x32xf32, #release_linear32>
  %part4 = scf.if %even -> (!ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>) {
    %hi = ttng.tmem_subslice %acc {offset = 96 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %hi : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  } else {
    %lo = ttng.tmem_subslice %acc {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %lo : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  }
  %read4 = ttng.tmem_load %part4 : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> tensor<128x32xf32, #release_linear32>
  %part5 = scf.if %even -> (!ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>) {
    %hi = ttng.tmem_subslice %acc {offset = 64 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %hi : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  } else {
    %lo = ttng.tmem_subslice %acc {offset = 160 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %lo : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  }
  %read5 = ttng.tmem_load %part5 : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> tensor<128x32xf32, #release_linear32>
  %part6 = scf.if %even -> (!ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>) {
    %hi = ttng.tmem_subslice %acc {offset = 32 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %hi : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  } else {
    %lo = ttng.tmem_subslice %acc {offset = 192 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %lo : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  }
  %read6 = ttng.tmem_load %part6 : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> tensor<128x32xf32, #release_linear32>
  %part7 = scf.if %even -> (!ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>) {
    %hi = ttng.tmem_subslice %acc {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %hi : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  } else {
    %lo = ttng.tmem_subslice %acc {offset = 224 : i32} : !ttg.memdesc<128x256xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
    scf.yield %lo : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  }
  %read7 = ttng.tmem_load %part7 : !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512> -> tensor<128x32xf32, #release_linear32>
  ttng.arrive_barrier %bar, 1 release %release16, %release32 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x16xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>, !ttg.memdesc<128x32xf32, #release_tmem, #ttng.tensor_memory, mutable, 128x512>
  tt.return
}


// RELEASE-LABEL: @release_distinct_allocations
// RELEASE: ttng.tmem_load
// RELEASE-NEXT: ttng.arrive_barrier
// RELEASE-NEXT: ttng.tmem_load
// RELEASE-NEXT: ttng.tmem_store
tt.func @release_distinct_allocations(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.tmem_store %o, %out, %true : tensor<128x128xf32, #linear128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// RELEASE-LABEL: @release_no_contract
// RELEASE-COUNT-2: ttng.tmem_load
// RELEASE-NEXT: ttng.arrive_barrier
tt.func @release_no_contract(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}


// RELEASE-LABEL: @release_actual_alias
// RELEASE-COUNT-2: ttng.tmem_load
// RELEASE-NEXT: ttng.arrive_barrier
tt.func @release_actual_alias(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  %again = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// RELEASE-LABEL: @release_unknown_provenance
// RELEASE-COUNT-2: ttng.tmem_load
// RELEASE-NEXT: ttng.arrive_barrier
tt.func @release_unknown_provenance(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %unknown: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %unknown : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// RELEASE-LABEL: @release_boundary_wait
// RELEASE: ttng.wait_barrier
// RELEASE-NEXT: ttng.arrive_barrier
// RELEASE-NEXT: ttng.tmem_load
tt.func @release_boundary_wait(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.wait_barrier %bar, %c0, %true : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// RELEASE-LABEL: @release_boundary_arrive
// RELEASE: ttng.arrive_barrier
// RELEASE-NEXT: ttng.arrive_barrier
// RELEASE-NEXT: ttng.tmem_load
tt.func @release_boundary_arrive(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// RELEASE-LABEL: @release_boundary_cta_barrier
// RELEASE: ttg.barrier local
// RELEASE-NEXT: ttng.arrive_barrier
// RELEASE-NEXT: ttng.tmem_load
tt.func @release_boundary_cta_barrier(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttg.barrier local
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// RELEASE-LABEL: @release_boundary_commit
// RELEASE: ttng.tc_gen5_commit
// RELEASE-NEXT: ttng.arrive_barrier
// RELEASE-NEXT: ttng.tmem_load
tt.func @release_boundary_commit(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.tc_gen5_commit %bar : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// RELEASE-LABEL: @release_boundary_unknown_effect
// RELEASE: "test.unknown_effect"
// RELEASE-NEXT: ttng.arrive_barrier
// RELEASE-NEXT: ttng.tmem_load
tt.func @release_boundary_unknown_effect(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  "test.unknown_effect"() : () -> ()
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// RELEASE-LABEL: @release_boundary_pure_loop
// RELEASE: scf.for
// RELEASE: scf.yield
// RELEASE: }
// RELEASE-NEXT: ttng.arrive_barrier
// RELEASE-NEXT: ttng.tmem_load
tt.func @release_boundary_pure_loop(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  %value = scf.for %i = %c0 to %c1 step %c1 iter_args(%carry = %c0) -> (i32) : i32 {
    %next = arith.addi %carry, %c1 : i32
    scf.yield %next : i32
  }
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// RELEASE-LABEL: @release_barrier_storage_alias
// RELEASE: ttg.local_store
// RELEASE-NEXT: ttng.arrive_barrier
// RELEASE-NEXT: ttng.tmem_load
tt.func @release_barrier_storage_alias(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %bits: tensor<1xi64, #release_scalar>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttg.local_store %bits, %bar : tensor<1xi64, #release_scalar> -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}

tt.func private @release_call_target() {
  tt.return
}

// RELEASE-LABEL: @release_boundary_call
// RELEASE: tt.call
// RELEASE-NEXT: ttng.arrive_barrier
// RELEASE-NEXT: ttng.tmem_load
tt.func @release_boundary_call(%bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) {
  %alpha = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %out = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_load %alpha : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  tt.call @release_call_target() : () -> ()
  %o = ttng.tmem_load %out : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
  ttng.arrive_barrier %bar, 1 release %alpha : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}


// Partial lifetime storage reuse with unchanged synchronization.

// PACK-LABEL: @pack_partial_high
// PACK: [[ROOT:%.+]] = ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: [[REUSE:%.+]] = ttng.tmem_subslice [[ROOT]] {{.*}}offset = 128 : i32
// PACK: ttg.warp_specialize
// PACK: ttng.wait_barrier {{.*}} deps
// PACK: ttng.tmem_load
// PACK: ttng.arrive_barrier
// PACK: ttng.tmem_load
// PACK: ttng.wait_barrier {{.*}} deps
// PACK: ttng.tmem_load
tt.func @pack_partial_high(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_partial_low
// PACK: [[ROOT:%.+]] = ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: [[REUSE:%.+]] = ttng.tmem_subslice [[ROOT]] {{.*}}offset = 0 : i32
// PACK: ttg.warp_specialize
// PACK: ttng.wait_barrier {{.*}} deps
// PACK: ttng.tmem_load
// PACK: ttng.arrive_barrier
// PACK: ttng.tmem_load
// PACK: ttng.wait_barrier {{.*}} deps
// PACK: ttng.tmem_load
tt.func @pack_partial_low(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_reversed_allocations
// PACK: [[ROOT:%.+]] = ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: [[REUSE:%.+]] = ttng.tmem_subslice [[ROOT]] {{.*}}offset = 128 : i32
// PACK: ttg.warp_specialize
// PACK: ttng.wait_barrier {{.*}} deps
// PACK: ttng.tmem_load
// PACK: ttng.arrive_barrier
// PACK: ttng.tmem_load
// PACK: ttng.wait_barrier {{.*}} deps
// PACK: ttng.tmem_load
tt.func @pack_reversed_allocations(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_wrong_phase
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_wrong_phase(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %one_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_missing_release
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_missing_release(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_wrong_count
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_wrong_count(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 2 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_predicated_mma
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_predicated_mma(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %false_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_accumulating_mma
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_accumulating_mma(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %true_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_synchronous_mma
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_synchronous_mma(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_ordered_interleaved_values
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_ordered_interleaved_values(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_future_overlap_read
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_future_overlap_read(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %lateval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %lateval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

// PACK-LABEL: @pack_second_release
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_second_release(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}


// PACK-LABEL: @pack_partial_64_columns
// PACK: [[ROOT64:%.+]] = ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttng.tmem_subslice [[ROOT64]] {{.*}}offset = 192 : i32
// PACK: ttg.warp_specialize
// PACK: ttng.wait_barrier {{.*}} deps
// PACK: ttng.tmem_load
// PACK: ttng.arrive_barrier
// PACK-COUNT-3: ttng.tmem_load
// PACK: ttng.wait_barrier {{.*}} deps
// PACK: ttng.tmem_load
tt.func @pack_partial_64_columns(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x64xf16, #pack_smem_b, #smem>) -> (tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>) {
  %c0 = arith.constant 0 : i32
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:5 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 192 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x64xf32, #linear64>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low0 = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval0 = ttng.tmem_load %low0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x64xf32, #linear64>
    %low1 = ttng.tmem_subslice %a {offset = 64 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval1 = ttng.tmem_load %low1 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x64xf32, #linear64>
    %low2 = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval2 = ttng.tmem_load %low2 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x64xf32, #linear64>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear64>
    ttg.warp_yield %highval, %lowval0, %lowval1, %lowval2, %bval : tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x64xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x64xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x64xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2, %results#3, %results#4 : tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>, tensor<128x64xf32, #linear64>
}


// PACK-LABEL: @pack_assigned_barrier_offset
// PACK-COUNT-2: ttng.tmem_alloc
// PACK-NOT: ttng.tmem_alloc
// PACK: ttg.warp_specialize
tt.func @pack_assigned_barrier_offset(%x: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %wa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %wb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
  %b = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %fulla = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %fullb = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fulla, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %fullb, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %results:3 = ttg.warp_specialize(%x, %wa, %wb, %a, %b, %fulla, %empty, %fullb)
  default {
    ttng.wait_barrier %fulla, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %high = ttng.tmem_subslice %a {offset = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %highval = ttng.tmem_load %high : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.arrive_barrier %empty, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %low = ttng.tmem_subslice %a {offset = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %lowval = ttng.tmem_load %low : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #linear128>
    ttng.wait_barrier %fullb, %c0 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    %bval = ttng.tmem_load %b : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear128>
    ttg.warp_yield %highval, %lowval, %bval : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
  }
  partition0(%px: !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, %pwa: !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, %pwb: !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, %pa: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, %pb: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %pfulla: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pempty: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %pfullb: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) num_warps(1) {
    %true_p = arith.constant true
    %false_p = arith.constant false
    %zero_p = arith.constant 0 : i32
    %one_p = arith.constant 1 : i32
    ttng.tc_gen5_mma %px, %pwa, %pa, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.wait_barrier %pempty, %zero_p : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttng.tc_gen5_mma %px, %pwb, %pb, %false_p, %true_p {is_async} : !ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %pfullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x32xf16, #pack_smem_a, #smem>, !ttg.memdesc<32x256xf16, #pack_smem_b, #smem>, !ttg.memdesc<32x128xf16, #pack_smem_b, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> (tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>)
  ttng.inval_barrier %fulla : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %fullb : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return %results#0, %results#1, %results#2 : tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>, tensor<128x128xf32, #linear128>
}

}
