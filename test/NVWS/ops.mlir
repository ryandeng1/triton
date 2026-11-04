// RUN: triton-opt --split-input-file %s | FileCheck %s
// RUN: triton-opt --split-input-file --allow-unregistered-dialect %s --nvws-schedule-aref-releases | FileCheck %s --check-prefix=OWN

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_create_single
  // CHECK: nvws.aref.create
  tt.func @aref_create_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    tt.return
  }

}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_get
  // CHECK: nvws.aref.get.enter
  // CHECK: nvws.aref.get.exit
  tt.func @aref_get(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %1:3 = nvws.aref.get.enter %0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
    nvws.aref.get.exit %0[%c0_i32], %1#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>, !ttg.async.token
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put
  // CHECK: nvws.aref.put.enter
  // CHECK: nvws.aref.put.exit
  tt.func @aref_put(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %1:3 = nvws.aref.put.enter %0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
    nvws.aref.put.exit %0[%c0_i32], %1#2 [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>, !ttg.async.token
    tt.return
  }
}

// -----

// CHECK-LABEL: @warp_group_nothing
tt.func @warp_group_nothing() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  tt.return
}

// CHECK-LABEL: @warp_1_partition
tt.func @warp_1_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT:  num_warps(4) {
  partition0  num_warps(4) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}

// CHECK-LABEL: @warp_2_partition
tt.func @warp_2_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT: partition0  num_warps(8) {
  partition0  num_warps(8) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition1 num_warps(4) {
  partition1 num_warps(4) {
  // CHECK-NEXT:   nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}

// CHECK-LABEL: @warp_group_results
tt.func @warp_group_results(%x: i32, %y: f32) -> (i32, f32) {
  // CHECK-NEXT: [[R:%.*]]:2 = nvws.warp_group
  %r:2 = nvws.warp_group
  // CHECK-NEXT: partition0 num_warps(4) {
  partition0 num_warps(4) {
  // CHECK-NEXT: nvws.warp_group.yield %{{.*}}, %{{.*}} : i32, f32
    nvws.warp_group.yield %x, %y : i32, f32
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition1 num_warps(4) {
  partition1 num_warps(4) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: } -> (i32, f32)
  } -> (i32, f32)
  // CHECK-NEXT: tt.return [[R]]#0, [[R]]#1
  tt.return %r#0, %r#1 : i32, f32
}

// CHECK-LABEL: @token_producer_consumer
tt.func @token_producer_consumer() {

  // CHECK: nvws.create_token
  // CHECK: nvws.producer_acquire
  // CHECK: nvws.producer_commit
  // CHECK: nvws.consumer_wait
  // CHECK: nvws.consumer_release

  %0 = nvws.create_token {loadType = 1 : i32, numBuffers = 3 : i32} : tensor<3x!nvws.token>

  %c0_i32 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
  %false = arith.constant {async_task_id = dense<0> : vector<1xi32>} false

  nvws.producer_acquire %0, %c0_i32, %false {async_task_id = dense<0> : vector<1xi32>} : tensor<3x!nvws.token>, i32, i1
  nvws.producer_commit %0, %c0_i32 {async_task_id = dense<0> : vector<1xi32>} : tensor<3x!nvws.token>, i32
  nvws.consumer_wait %0, %c0_i32, %false {async_task_id = dense<1> : vector<1xi32>} : tensor<3x!nvws.token>, i32, i1
  nvws.consumer_release %0, %c0_i32 {async_task_id = dense<1> : vector<1xi32>} : tensor<3x!nvws.token>, i32
  tt.return
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @aref_phi
  tt.func @aref_phi() -> i32 {
    // CHECK: %[[LOCAL:.*]] = arith.constant {ttg.partition = array<i32: 3>} 1 : i32
    %local = arith.constant {ttg.partition = array<i32: 3>} 1 : i32
    // CHECK: %[[REMOTE:.*]] = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 2 : i32
    %remote = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 2 : i32
    // CHECK: %[[SELECTED:.*]] = nvws.aref.phi %[[LOCAL]], %[[REMOTE]] {ttg.partition = array<i32: 0, 1, 2, 3>} : i32
    %selected = nvws.aref.phi %local, %remote {ttg.partition = array<i32: 0, 1, 2, 3>} : i32
    // CHECK: tt.return %[[SELECTED]] : i32
    tt.return %selected : i32
  }
}

// -----

// Ownership scheduling must not confuse a buffer's last use with exclusive
// publication. Only a separately held, closed channel protects crossed effects.
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#ownership_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem_alt = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  tt.func private @ownership_escape(!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>)
  tt.func private @ownership_escape_token(!ttg.async.token)

  // CHECK-LABEL: @owned_release_before_other_store
  // OWN-LABEL: @owned_release_before_other_store
  tt.func @owned_release_before_other_store() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: nvws.aref.get.exit
    // OWN-NEXT: ttng.tmem_store [[READ]]
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_multiple_channels
  // OWN-LABEL: @ownership_reject_multiple_channels
  tt.func @ownership_reject_multiple_channels() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %balias = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_escape
  // OWN-LABEL: @ownership_reject_escape
  tt.func @ownership_reject_escape() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.call @ownership_escape(%bp) {ttg.partition = array<i32: 1>} : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @owned_release_with_get_writer
  // OWN-LABEL: @owned_release_with_get_writer
  tt.func @owned_release_with_get_writer() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: nvws.aref.get.exit
    // OWN-NEXT: ttng.tmem_store [[READ]]
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %bg, %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_wrong_token
  // OWN-LABEL: @ownership_reject_wrong_token
  tt.func @ownership_reject_wrong_token() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_wrong_stage
  // OWN-LABEL: @ownership_reject_wrong_stage
  tt.func @ownership_reject_wrong_stage() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b[%c1], %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_ended_scope
  // OWN-LABEL: @ownership_reject_ended_scope
  tt.func @ownership_reject_ended_scope() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_assigned_offset
  // OWN-LABEL: @ownership_reject_assigned_offset
  tt.func @ownership_reject_assigned_offset() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_partition_mismatch
  // OWN-LABEL: @ownership_reject_partition_mismatch
  tt.func @ownership_reject_partition_mismatch() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_schedule_mismatch
  // OWN-LABEL: @ownership_reject_schedule_mismatch
  tt.func @ownership_reject_schedule_mismatch() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_multiple_producers
  // OWN-LABEL: @ownership_reject_multiple_producers
  tt.func @ownership_reject_multiple_producers() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp2, %bpt2 = nvws.aref.put.enter %b {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %bp2, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %b, %bpt2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_buffer_type
  // OWN-LABEL: @ownership_reject_buffer_type
  tt.func @ownership_reject_buffer_type() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %bwrong = nvws.aref.buffer %b, %bpt {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_enter_encoding
  // OWN-LABEL: @ownership_reject_enter_encoding
  tt.func @ownership_reject_enter_encoding() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem_alt, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem_alt, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Multi-buffer generations use independent producer/consumer enter counts.
  // CHECK-LABEL: @owned_release_depth_two
  // OWN-LABEL: @owned_release_depth_two
  tt.func @owned_release_depth_two() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: nvws.aref.get.exit
    // OWN-NEXT: ttng.tmem_store [[READ]]
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @owned_release_carried_depth_three
  // OWN-LABEL: @owned_release_carried_depth_three
  tt.func @owned_release_carried_depth_three() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %active, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: nvws.aref.get.exit
    // OWN-NEXT: ttng.tmem_store [[READ]]
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @owned_release_carried_mma_producer
  // OWN-LABEL: @owned_release_carried_mma_producer
  tt.func @owned_release_carried_mma_producer(%lhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, %rhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>) {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %storetoken = ttng.tmem_store %zero, %active[], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %mmatoken = ttng.tc_gen5_mma %lhs, %rhs, %active[%storetoken], %true, %true {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: nvws.aref.get.exit
    // OWN-NEXT: ttng.tmem_store [[READ]]
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @owned_mma_consumer_release
  // OWN-LABEL: @owned_mma_consumer_release
  tt.func @owned_mma_consumer_release(%lhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, %rhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>) {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]], {{%.*}} = ttng.tmem_load
    %read, %readtoken = ttng.tmem_load %ag[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: {{%.*}} = ttng.tc_gen5_mma
    %mmatoken = ttng.tc_gen5_mma %lhs, %rhs, %ag[%readtoken], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // OWN-NEXT: nvws.aref.get.exit
    // OWN-NEXT: ttng.tmem_store [[READ]]
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_missing_drain
  // OWN-LABEL: @ownership_reject_missing_drain
  tt.func @ownership_reject_missing_drain() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %active, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_reused_initial_token
  // OWN-LABEL: @ownership_reject_reused_initial_token
  tt.func @ownership_reject_reused_initial_token() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %active, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %apt : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_foreign_yield
  // OWN-LABEL: @ownership_reject_foreign_yield
  tt.func @ownership_reject_foreign_yield() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %active, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %bgt : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_mma_without_completion
  // OWN-LABEL: @ownership_reject_mma_without_completion
  tt.func @ownership_reject_mma_without_completion(%lhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, %rhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>) {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %storetoken = ttng.tmem_store %zero, %active[], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %mmatoken = ttng.tc_gen5_mma %lhs, %rhs, %active[%storetoken], %true, %true {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_escaping_access_token
  // OWN-LABEL: @ownership_reject_escaping_access_token
  tt.func @ownership_reject_escaping_access_token(%lhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, %rhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>) {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %storetoken = ttng.tmem_store %zero, %active[], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %mmatoken = ttng.tc_gen5_mma %lhs, %rhs, %active[%storetoken], %true, %true {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.call @ownership_escape_token(%mmatoken) {ttg.partition = array<i32: 0>} : (!ttg.async.token) -> ()
    nvws.aref.put.exit %a, %current [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_external_access_dependency
  // OWN-LABEL: @ownership_reject_external_access_dependency
  tt.func @ownership_reject_external_access_dependency(%lhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, %rhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>) {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %storetoken = ttng.tmem_store %zero, %active[], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %mmatoken = ttng.tc_gen5_mma %lhs, %rhs, %active[%current], %true, %true {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_crossing_mma_issue
  // OWN-LABEL: @ownership_reject_crossing_mma_issue
  tt.func @ownership_reject_crossing_mma_issue(%lhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, %rhs: !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, %raw: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    scf.for %i = %c0 to %c1 step %c1 : i32 {
    %ap, %apt = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    ttng.tmem_store %zero, %ap, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %apt [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]], {{%.*}} = ttng.tmem_load
    %read, %readtoken = ttng.tmem_load %ag[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: {{%.*}} = ttng.tc_gen5_mma
    %mmatoken = ttng.tc_gen5_mma %lhs, %rhs, %ag[%readtoken], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: {{%.*}} = ttng.tc_gen5_mma
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %othermma = ttng.tc_gen5_mma %lhs, %rhs, %raw[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #ownership_shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_carried_owner
  // OWN-LABEL: @ownership_reject_carried_owner
  tt.func @ownership_reject_carried_owner() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %active, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_missing_carried_owner
  // OWN-LABEL: @ownership_reject_missing_carried_owner
  tt.func @ownership_reject_missing_carried_owner() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %active, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_carried_physical_type
  // OWN-LABEL: @ownership_reject_carried_physical_type
  tt.func @ownership_reject_carried_physical_type() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 3x128x128>
    ttng.tmem_store %zero, %active, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 3x128x128>
    nvws.aref.put.exit %a, %current [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @ownership_reject_explicit_phase
  // OWN-LABEL: @ownership_reject_explicit_phase
  tt.func @ownership_reject_explicit_phase() {
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %aroot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %broot = ttng.tmem_alloc : () -> !ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a = nvws.aref.create %aroot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %b = nvws.aref.create %broot : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %ap, %apt = nvws.aref.put.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %final = scf.for %i = %c0 to %c1 step %c1 iter_args(%current = %apt) -> (!ttg.async.token) : i32 {
    %active = nvws.aref.buffer %a, %current {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %active, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.put.exit %a, %current [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bp, %bpt = nvws.aref.put.enter %b {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %ag, %agt = nvws.aref.get.enter %a {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // OWN: [[READ:%.*]] = ttng.tmem_load
    %read = ttng.tmem_load %ag {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // OWN-NEXT: ttng.tmem_store [[READ]]
    // OWN-NEXT: nvws.aref.get.exit
    ttng.tmem_store %read, %bp, %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.aref.get.exit %a, %agt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %b, %bpt [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %bg, %bgt = nvws.aref.get.enter %b {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    %result = ttng.tmem_load %bg {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %b, %bgt [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %nextbuf, %next = nvws.aref.put.enter %a[%c0, %c0] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    scf.yield %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %a, %final [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %drainbuf, %drain = nvws.aref.get.enter %a : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %a, %drain [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }

}
