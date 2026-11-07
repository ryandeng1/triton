# Attention score/P storage ablation

The experiment changes where the softmax probabilities P are stored while preserving the attention arithmetic, two-query-subtile schedule, and barrier counts. It is an experiment module, not a compiler optimization.

Source: [`python/examples/gluon/01-attention-forward.py`](../python/examples/gluon/01-attention-forward.py), revision `a35b27e3f595c828e9fa9b6c43810bbba6a8202e`, SHA256 `4cbb58c1824e9afd9b496e39dee173ae3a9d44e971db76c48aca3158490e6a24` (verified against the original file). The clone is [`gluon_attention_semantic_ablation.py`](gluon_attention_semantic_ablation.py). `KernelConfig.SEPARATE_P=False` reuses score storage; `True` allocates independent P buffers. `attention_forward(..., return_kernel=True)` returns output, base-2 log-sum-exp, and the compiled kernel.

## Storage and numerical contract

The original `_borrow_s_as_p` reinterprets FP32 score storage as packed BF16/FP16 and takes the first `BLOCK_N` logical columns. With full 128-row subtiles, P occupies the first `BLOCK_N / 2` physical 32-bit columns. Alpha occupies physical column `BLOCK_N / 2`; epilogue statistics use the next two columns. These views are disjoint. The separate-buffer variant preserves the same P element type, logical shape, and `TensorMemoryLayout`; it does not remove padding or shrink an MMA instruction footprint.

The allocation switch is at clone lines 969–975. Both variants have identical reductions, casts, exponentials, scaling, accumulation, and output stores. Output and log-sum-exp should therefore match exactly between the two variants; comparison with another attention implementation still requires its numerical tolerance contract. No conditional or approximate rescaling was introduced.

For `BLOCK_M=256`, one CTA, BF16/FP16, the two subtiles each have 128 rows. The expected live full-row column counts are:

| Head dimension | BLOCK_N | Reuse | Separate P | Rounded reservation |
| --- | --- | --- | --- | --- |
| 64 | 64 | 256 | 320 | 256 versus 512 |
| 128 | 64 | 384 | 448 | 512 versus 512 |
| 128 | 128 | 512 | 640 | Separate P exceeds the 512-column budget |

Compiled metadata is authoritative. `BLOCK_M=128` uses 64-row subtiles and may permit physical row packing; the full-row counts above do not apply. The default persistent launch uses `OCCUPANCY=1`, which can hide a smaller reservation. Register redistribution or shared-memory use may also prevent two resident CTAs. Fewer reserved columns alone do not establish a performance gain.

## Ordering argument

For each query subtile and KV epoch:

1. MMA publishes completed QK through the score channel. Softmax acquires that generation and loads the score subtiles before writing alpha or P (clone lines 672–701).
2. Softmax publishes P by arriving at the score channel's empty barrier after the P stores. The MMA producer acquires that generation before PV reads P (lines 567–587). P's scalar row reduction remains after publication, as in the original.
3. The following QK may overwrite reused P storage while the previous PV is still computing. This relies on the specific hardware ordering of an MMA's TMEM A reads before a following MMA's D writes, with the same warp and CTA-group size. It does **not** assume arbitrary asynchronous instructions complete in issue order. See [PTX ISA §9.7.18.6.2](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions).
4. The next QK completion notification also covers prior issued MMA work. Softmax waits for this generation before the next P overwrite. At the tail, the final PV signals both score barriers (lines 592–595), and both softmax epilogues acquire those generations before storing statistics (lines 752–760). This covers the last P readers and the transition to the next persistent tile. See [PTX completion rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-mbarrier-completion).

Cross-partition loads/stores still require the compiler's usual TMEM waits and thread-synchronization fences; source ordering alone does not replace them. See [PTX inter-thread synchronization](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-non-pipelined-diff-thread).

The clone adds score and P descriptors to the relevant channel-wait dependencies in **both** variants (lines 131–153). This preserves remote writes and asynchronous read lifetimes in the compiler effect model; it adds no barrier arrivals. [`WaitBarrierOp::getEffects`](../lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp) at line 468 describes why dependencies must survive aliasing and CSE. The unchanged original lacks these explicit dependency annotations, so the clone is distinguished from that reference in measurements.

## Missing compiler proof

The current [`PackTMem.cpp`](../lib/Dialect/TritonNvidiaGPU/Transforms/PackTMem.cpp) recognizer accepts two straight-line partitions (line 91), full-row FP32 allocations (line 158), and a restricted shared-memory-input MMA protocol (line 253). It excludes this example's loop generations, mixed element types, TMEM P stores and MMA reads, and six partitions.

A bounded extension would recognize this repeated QK/P protocol, prove physical subview compatibility, and establish `QK completion → score consumption → P publication → PV A consumption → next QK overwrite`, including tail and persistent-tile boundaries. Operand-specific hardware ordering can preserve the existing overlap without forcing a wait for the entire PV result. This requires lifetime and value-preservation proofs in addition to race freedom; ordered writes must not destroy a still-live score, probability, or statistic.

GPU correctness, resource metadata, and timings are recorded separately by the comparison harness. This document is a source/protocol audit, not evidence of a measured speedup.

## Deferred rescaling and exact identity correction

[`gluon_attention_rescale_ablation.py`](gluon_attention_rescale_ablation.py) is a separate clone. `RESCALE_THRESHOLD=0` retains the original running-maximum update. At threshold eight, a row retains its previous maximum unless the candidate maximum exceeds it by more than eight base-2 exponent units. P, the accumulated output, and the row sum consistently use that retained scale. In real arithmetic the normalized result is unchanged and each exponential is at most 256; changing the scale changes floating-point rounding, including the BF16 P conversion. The original and threshold-eight versions therefore require a tolerance comparison.

`SKIP_IDENTITY_CORRECTION=True` reduces `(alpha != 1)` over the complete logical row tensor with `gl.sum` and branches on the resulting uniform scalar. Only the O loads, multiplies, and stores are conditional. Every channel wait and arrival remains unconditional, including publication of the O buffer for the following PV. For finite accumulator values, skipping multiplication by one preserves the accumulator. Compare threshold-eight always-correct versus threshold-eight skip-correct with exact output and log-sum-exp equality; this pair isolates the identity transformation from the changed softmax scaling.

The wrapper restricts the new behavior to noncausal BF16 attention, a positive finite softmax scale, and a finite threshold in `[0, 8]`. The experiment assumes finite score and accumulator intermediates. Identity skipping additionally requires one CTA: a reduction across multiple CTAs may insert cluster synchronization that is incompatible with the other warp-specialized partitions. This restriction avoids asserting uniform control flow across a cluster without a protocol proof. The original P-storage clone is unchanged by this follow-up.

The first matched GPU run used B2/H8/N4096/D128 BF16, scale `1/sqrt(128)`, BM256/BN128, one CTA, and three seeds. Median seed medians were 112.45 µs at threshold zero, 112.74 µs at threshold eight with unconditional correction, and 106.86 µs with identity skipping: **1.055×** for the exact threshold-eight pair. Outputs and log-sum-exp matched exactly before and after graph timing, including fresh launches; FP32-reference and guard checks passed. [Recorded results](real_kernel_results/semantic_attention_gluon_correction/results.json).

The [compiled audit](real_kernel_results/semantic_attention_gluon_correction/compiled_protocol_audit.json) confirms:

- The skip variant's [TTGIR](real_kernel_results/semantic_attention_gluon_correction/seed_0/gluon-rescale8-skip/kernel.ttgir), lines 224–244 and 256–276, reduces `tensor<128xi32>` to a scalar and branches around O read/multiply/write. Output arrivals remain outside both branches. The complete wait/arrive/commit operation sequences match the unconditional variant after removing source locations.
- Its [PTX](real_kernel_results/semantic_attention_gluon_correction/seed_0/gluon-rescale8-skip/kernel.ptx), lines 681–699 and 1243–1253, uses shared scratch and 128-thread synchronization to combine all four correction warps before branching. The reduction adds two static `barrier.sync` instructions; mbarrier/commit counts remain unchanged. TMEM instructions are skipped dynamically, so static load/store counts do not decrease.
- Both threshold-eight variants report identical resources: 512 TMEM columns, 230161 shared bytes, 16 warps, one CTA, 128 registers, and 20 spills.

This establishes a benefit from value-dependent elimination of an identity update while preserving the ownership protocol. Race freedom alone does not prove the value identity, and thresholded softmax scaling is a separate numerical transformation. These measurements do not establish an advantage over the best two-CTA Gluon configuration.
