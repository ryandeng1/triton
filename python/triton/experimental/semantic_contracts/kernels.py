"""CUDA kernel consumers for the experimental semantic-contract graph backend.

The backend, rather than callers of these low-level JIT functions, must establish
the specialization facts. In particular, UNIQUE_INDICES means injectivity of all
active destination addresses across the complete update grid. It does not mean
uniqueness within one tile. KNOWN_ZERO additionally requires initialized +0
destination elements with no intermediate observers or concurrent mutation.

The owned operator's floating-point updates deliberately match ordinary NVIDIA
global FP32 atomic addition's round-to-nearest, flush-to-zero behavior. This
avoids the previous prototype's finite/non-subnormal input restriction. The
unique writer can instead use non-flushing addition when required by its
consumer's arithmetic contract. NaNs retain IEEE NaN semantics; their payload
bits are not part of the owned operator contract.

Reference: CUDA Programming Guide, Floating-Point Computation, "Atomic Functions
Denormals Behavior":
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html

These functions contain no runtime inspection of index values to prove a fact.
The graph backend must use the atomic variant when no producer proof is present.
"""

import triton
import triton.language as tl


@triton.jit
def zero_buffer(Ptr, SIZE, BLOCK: tl.constexpr):
    """Initialize a numeric output, scratch, or counting buffer to +0."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Ptr + offsets, 0, mask=offsets < SIZE)


@triton.jit
def embedding_count(Indices, Counts, N, V, PADDING_IDX: tl.constexpr, BLOCK_N: tl.constexpr):
    """Count valid occurrences; Counts must start at zero and N must fit int32."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    active = offsets < N
    indices = tl.load(Indices + offsets, mask=active, other=0).to(tl.int64)
    valid = active & (indices >= 0) & (indices < V) & (indices != PADDING_IDX)
    # Avoid arithmetic on an invalid, potentially extreme sentinel index.
    safe_indices = tl.where(valid, indices, 0)
    tl.atomic_add(Counts + safe_indices, 1, mask=valid, sem="relaxed")


@triton.jit
def _global_atomic_add_value(old, contribution):
    """The arithmetic of global atomicAdd(float), without its memory effect.

    This is a pure per-element operation. It does not expose physical lane IDs
    or introduce hidden scalar variation into Triton's logical block model.
    An explicit rounding modifier also prevents contraction or +0 elimination.
    """
    return tl.inline_asm_elementwise(
        "add.rn.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[old, contribution],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def embedding_update(
    Grad,
    Indices,
    Counts,
    Scratch,
    N,
    D,
    V,
    PADDING_IDX: tl.constexpr,
    SCALE_BY_FREQ: tl.constexpr,
    UNIQUE_INDICES: tl.constexpr,
    KNOWN_ZERO: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Lower contiguous row scatter-add with optional occurrence normalization.

    Grad is [N,D], Indices is [N], and Scratch is contiguous FP32 [V,D].
    The grid is (N, ceil_div(D,BLOCK_D)). Counts is initialized int32 [V] when
    SCALE_BY_FREQ and not UNIQUE_INDICES; otherwise it may be None.

    Scratch is private to this logical operator and has no other simultaneous
    readers/writers. The atomic path may therefore use relaxed ordering. The
    backend additionally checks shapes/strides and integer address bounds before
    treating index uniqueness as destination-address injectivity.
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    index = tl.load(Indices + row, mask=row < N, other=0).to(tl.int64)
    valid_row = (row < N) & (index >= 0) & (index < V) & (index != PADDING_IDX)
    valid_column = columns < D
    contribution = tl.load(Grad + row * D + columns, mask=(row < N) & valid_column, other=0).to(tl.float32)
    safe_index = tl.where(valid_row, index, 0)
    if SCALE_BY_FREQ and not UNIQUE_INDICES:
        count = tl.load(Counts + safe_index, mask=valid_row, other=1).to(tl.float32)
        contribution = contribution / count
    destination = Scratch + safe_index * D + columns
    active = valid_row & valid_column
    if UNIQUE_INDICES:
        if KNOWN_ZERO:
            old = tl.full((BLOCK_D, ), 0.0, tl.float32)
        else:
            old = tl.load(destination, mask=active, other=0.0)
        updated = _global_atomic_add_value(old, contribution)
        tl.store(destination, updated, mask=active)
    else:
        tl.atomic_add(destination, contribution, mask=active, sem="relaxed")


@triton.jit
def embedding_unique_write(
    Grad,
    Indices,
    Out,
    N,
    D,
    V,
    PADDING_IDX: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPLETE_COVERAGE: tl.constexpr,
    ATOMIC_FTZ: tl.constexpr,
):
    """Write single-contribution rows directly into their final output dtype.

    The caller proves unique active destination addresses across the entire grid,
    contiguous Grad[N,D]/Indices[N]/Out[V,D], bounded integer address arithmetic,
    disjoint private output storage, and no intermediate observers. Each output
    element has a *logical* initial value +0; this kernel never reads Out.

    Without COMPLETE_COVERAGE, the caller must first zero Out so untouched and
    padding rows have their specified values. With COMPLETE_COVERAGE, indices
    are a permutation of [0,V), and the grid visits every row/column. Physical
    initialization can then be omitted: the unique owner of a padding row writes
    +0 explicitly, and every other element receives its single contribution.

    ATOMIC_FTZ selects the owned operator's FP32 global-atomic arithmetic.
    Otherwise addition explicitly preserves subnormals. Both paths add logical
    +0 before the final dtype conversion, preserving signed-zero behavior rather
    than treating the operation as an unrestricted copy. Unique valid indices
    also make each non-padding occurrence count one, including frequency mode.
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    index = tl.load(Indices + row, mask=row < N, other=0).to(tl.int64)
    valid_row = (row < N) & (index >= 0) & (index < V)
    valid_column = columns < D
    contributes = valid_row & (index != PADDING_IDX)
    contribution = tl.load(Grad + row * D + columns, mask=contributes & valid_column, other=0).to(tl.float32)
    logical_zero = tl.full((BLOCK_D, ), 0.0, tl.float32)
    if ATOMIC_FTZ:
        updated = _global_atomic_add_value(logical_zero, contribution)
    else:
        updated = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[logical_zero, contribution],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    if COMPLETE_COVERAGE:
        updated = tl.where(index == PADDING_IDX, logical_zero, updated)
        active = valid_row & valid_column
    else:
        active = contributes & valid_column
    # Invalid extreme indices never participate in pointer multiplication.
    safe_index = tl.where(valid_row, index, 0)
    tl.store(Out + safe_index * D + columns, updated, mask=active)


@triton.jit
def convert_output(Scratch, Out, SIZE, BLOCK: tl.constexpr):
    """Cast the complete initialized scratch, including untouched output rows."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Scratch + offsets, mask=offsets < SIZE, other=0.0)
    tl.store(Out + offsets, values, mask=offsets < SIZE)


@triton.jit
def rmsnorm_persistent(X, Y, W, Rsigma, M, N, EPS, BLOCK: tl.constexpr):
    """Contiguous RMSNorm with intentionally invariant W loads inside the loop.

    X/Y are [M,N], W is [N], and Rsigma is FP32 [M]. All dimensions and EPS
    are scalar arguments. W is argument 2 for the native alias-contract path.
    No hoist or special immutable variant is encoded in the source.
    """
    first_row = tl.program_id(0)
    step = tl.num_programs(0)
    columns = tl.arange(0, BLOCK)
    valid_column = columns < N
    for row in tl.range(first_row, M, step, num_stages=2):
        row_offset = row.to(tl.int64) * N + columns
        values = tl.load(X + row_offset, mask=valid_column, other=0.0).to(tl.float32)
        weights = tl.load(W + columns, mask=valid_column, other=0.0).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(values * values, 0) / N + EPS)
        tl.store(Rsigma + row, inverse_rms)
        normalized = values * inverse_rms * weights
        tl.store(Y + row_offset, normalized, mask=valid_column)
