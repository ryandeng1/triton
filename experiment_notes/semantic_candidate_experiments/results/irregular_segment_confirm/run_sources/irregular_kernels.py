# Extracted from FlagGems, Mamba, and TorchRec; licenses and pinned originals in vendor/irregular.
# Extraction removes framework/autotuning decorators only; original function bodies are preserved.
import torch
import triton
import triton.language as tl

# Experimental source specializations below are not upstream compiler passes.
# Globally unique indices, fresh zero scratch, no intermediate observers, and
# finite ordinary (non-subnormal) gradients license the embedding specialization.
@triton.jit
def embedding_unique_kernel(grad_output_ptr, indices_ptr, grad_weight_ptr,
                            num_weights, padding_idx, BLOCK_D: tl.constexpr,
                            EMBED_DIM: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM
    idx = tl.load(indices_ptr + pid_n)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
    go = tl.load(grad_output_ptr + pid_n * EMBED_DIM + offs_d,
                 mask=mask_d, other=0).to(tl.float32)
    # Fresh +0 initialization and a sole writer let us eliminate the old-value
    # load. IEEE round-to-nearest +0 + -0 is +0, so normalize signed zero.
    # All other finite, non-subnormal FP32 values are unchanged by +0.
    # Atomic f32 subnormal flushing is outside this experiment's input domain.
    ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    updated = tl.where(go == 0.0, 0.0, go)
    tl.store(ptrs, updated, mask=mask_d & valid)


@triton.jit
def affine_combine(a_left, b_left, a_right, b_right):
    return a_right * a_left, a_right * b_left + b_right


@triton.jit
def state_passing_scan(states_ptr, out_ptr, final_ptr, da_ptr, init_ptr,
                       seq_ptr, DIM: tl.constexpr, CHUNKS: tl.constexpr,
                       HEADS: tl.constexpr, BLOCK_D: tl.constexpr,
                       BLOCK_C: tl.constexpr, HAS_INIT: tl.constexpr,
                       HAS_SEQ: tl.constexpr):
    # Affine composition changes floating evaluation order. This is a
    # numerical-contract prototype, not an exact semantics-preserving rewrite.
    block_d = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // HEADS
    head = bh % HEADS
    c = tl.arange(0, BLOCK_C)
    d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    a = tl.exp(tl.load(da_ptr + (batch * CHUNKS + c) * HEADS + head,
                       mask=c < CHUNKS, other=0).to(tl.float32))
    if HAS_SEQ:
        cur = tl.load(seq_ptr + batch * CHUNKS + c, mask=c < CHUNKS, other=0)
        prev = tl.load(seq_ptr + batch * CHUNKS + c - 1,
                       mask=(c > 0) & (c < CHUNKS), other=0)
        a = tl.where(cur == prev, a, 0.0)
    b = tl.load(states_ptr + ((batch * CHUNKS + c[:, None]) * HEADS + head) * DIM + d[None, :],
                mask=(c[:, None] < CHUNKS) & (d[None, :] < DIM), other=0).to(tl.float32)
    aa = tl.broadcast_to(a[:, None], (BLOCK_C, BLOCK_D))
    a_scan, b_scan = tl.associative_scan((aa, b), 0, affine_combine)
    if HAS_INIT:
        initial = tl.load(init_ptr + bh * DIM + d, mask=d < DIM, other=0).to(tl.float32)
    else:
        initial = tl.full((BLOCK_D,), 0, tl.float32)
    result = a_scan * initial[None, :] + b_scan
    tl.store(out_ptr + ((batch * CHUNKS + c[:, None] + 1) * HEADS + head) * DIM + d[None, :],
             result, mask=(c[:, None] + 1 < CHUNKS) & (d[None, :] < DIM))
    tl.store(out_ptr + (batch * CHUNKS * HEADS + head) * DIM + d,
             initial, mask=d < DIM)
    final = tl.sum(tl.where(c[:, None] == CHUNKS - 1, result, 0), 0)
    tl.store(final_ptr + bh * DIM + d, final, mask=d < DIM)

tle = tl  # Extracted functions only use tle.program_id.
_UNIFORM_FAST_PATH_MIN_NUMEL = 1 << 20
_UNIFORM_LENGTHS_CACHE = {}
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


@triton.jit
def _embedding_dense_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    idx = tl.load(indices_ptr + pid_n)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0).to(tl.float32)

    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask)


@triton.jit
def _embedding_dense_backward_count_kernel(
    indices_ptr,
    counts_ptr,
    N,
    num_weights,
    padding_idx,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    idx = tl.load(indices_ptr + offs, mask=mask, other=0).to(tl.int32)
    valid = mask & (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
    tl.atomic_add(counts_ptr + idx, 1, mask=valid)


@triton.jit
def _embedding_dense_backward_kernel_scale_by_freq(
    grad_output_ptr,
    indices_ptr,
    counts_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    idx = tl.load(indices_ptr + pid_n).to(tl.int32)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    # go = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)
    go = tl.load(go_ptrs, mask=mask_d, other=0.0)

    # cnt = tl.load(counts_ptr + idx, mask=valid, other=1).to(tl.float32)
    cnt = tl.load(counts_ptr + idx, mask=valid, other=1)
    go = go / cnt

    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask)


@triton.jit
def _mul_combine(a, b):
    return a * b


def _all_lengths_equal(lengths, value):
    cache_key = (
        lengths.device.type,
        lengths.data_ptr(),
        tuple(lengths.shape),
        getattr(lengths, "_version", None),
        value,
    )
    is_equal = _UNIFORM_LENGTHS_CACHE.get(cache_key)
    if is_equal is None:
        is_equal = torch.all(lengths.detach().cpu() == value).item()
        if len(_UNIFORM_LENGTHS_CACHE) > 128:
            _UNIFORM_LENGTHS_CACHE.clear()
        _UNIFORM_LENGTHS_CACHE[cache_key] = is_equal
    return is_equal


def _check_index_tensor(data, index_tensor, name, axis):
    if index_tensor.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise NotImplementedError(f"segment_reduce(): {name} must be int32 or int64.")
    if index_tensor.device != data.device:
        raise RuntimeError(
            f"segment_reduce(): Expected data and {name} on the same device."
        )
    if data.dim() < index_tensor.dim():
        raise RuntimeError(
            f"segment_reduce(): Expected data.dim() >= {name}.dim(), got "
            f"{data.dim()} and {index_tensor.dim()}."
        )
    if axis != index_tensor.dim() - 1:
        raise RuntimeError(
            f"segment_reduce(): Expected axis to be the last dimension of {name} "
            f"but got {axis}."
        )


def _validate_lengths(data, lengths, axis, unsafe):
    _check_index_tensor(data, lengths, "lengths", axis)
    if unsafe:
        return
    lengths_cpu = lengths.detach().cpu()
    if torch.any(lengths_cpu < 0).item():
        raise RuntimeError("lengths contains negative value!")
    valid_lengths = torch.all(lengths_cpu.sum(dim=-1) == data.size(axis)).item()
    if not valid_lengths:
        raise RuntimeError(
            "segment_reduce(): Expected all rows of lengths along axis to sum to "
            "data.size(lengths.dim()-1) when !unsafe."
        )


def _get_uniform_segment_length(data, lengths, axis):
    if data.numel() < _UNIFORM_FAST_PATH_MIN_NUMEL:
        return None
    if tuple(lengths.shape[:-1]) != tuple(data.shape[:axis]):
        return None
    segment_count = lengths.shape[-1]
    if segment_count <= 0:
        return None
    data_size_axis = data.shape[axis]
    if data_size_axis % segment_count != 0:
        return None
    segment_length = data_size_axis // segment_count
    if segment_length <= 0:
        return None

    if _all_lengths_equal(lengths, segment_length):
        return segment_length
    return None


@triton.jit
def _segment_reduce_uniform_inner1_forward_kernel(
    data,
    output,
    total_rows,
    segment_count,
    segment_length,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < total_rows
    col_mask = cols < segment_length
    mask = row_mask & col_mask

    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    data_offsets = outer_idx * data_size_axis + dim_idx * segment_length + cols

    if IS_SUM or IS_MEAN:
        values = tl.load(data + data_offsets, mask=mask, other=0.0).to(compute_dtype)
        result = tl.sum(values, axis=1)
        if IS_MEAN:
            result = result / segment_length
    elif IS_PROD:
        values = tl.load(data + data_offsets, mask=mask, other=1.0).to(compute_dtype)
        result = tl.reduce(values, axis=1, combine_fn=_mul_combine)
    elif IS_MAX:
        values = tl.load(data + data_offsets, mask=mask, other=float("-inf")).to(
            compute_dtype
        )
        nan_mask = (values != values) & mask
        has_nan = tl.sum(nan_mask.to(tl.int32), axis=1) > 0
        nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=1)
        result = tl.max(tl.where(mask & ~nan_mask, values, float("-inf")), axis=1)
        result = tl.where(has_nan, nan_value, result)
    elif IS_MIN:
        values = tl.load(data + data_offsets, mask=mask, other=float("inf")).to(
            compute_dtype
        )
        nan_mask = (values != values) & mask
        has_nan = tl.sum(nan_mask.to(tl.int32), axis=1) > 0
        nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=1)
        result = tl.min(tl.where(mask & ~nan_mask, values, float("inf")), axis=1)
        result = tl.where(has_nan, nan_value, result)

    output_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(output + output_offsets, result, mask=output_offsets < total_rows)


@triton.jit
def _segment_reduce_uniform_forward_kernel(
    data,
    output,
    total_rows,
    segment_count,
    segment_length,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    row_mask = rows < total_rows
    k_mask = k_offsets < inner_size
    mask = row_mask & k_mask

    outer_idx = rows // segment_count
    dim_idx = rows - outer_idx * segment_count
    segment_start = dim_idx * segment_length
    base_offsets = (
        outer_idx * data_size_axis * inner_size + segment_start * inner_size + k_offsets
    )

    if IS_MAX:
        acc = tl.full((BLOCK_M, BLOCK_K), float("-inf"), dtype=compute_dtype)
    elif IS_MIN:
        acc = tl.full((BLOCK_M, BLOCK_K), float("inf"), dtype=compute_dtype)
    elif IS_PROD:
        acc = tl.full((BLOCK_M, BLOCK_K), 1.0, dtype=compute_dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)

    has_nan = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int1)
    nan_value = tl.zeros((BLOCK_M, BLOCK_K), dtype=compute_dtype)

    pos = 0
    while pos < segment_length:
        data_offsets = base_offsets + pos * inner_size
        if IS_SUM or IS_MEAN:
            values = tl.load(data + data_offsets, mask=mask, other=0.0).to(
                compute_dtype
            )
            acc += values
        elif IS_PROD:
            values = tl.load(data + data_offsets, mask=mask, other=1.0).to(
                compute_dtype
            )
            acc *= values
        elif IS_MAX:
            values = tl.load(data + data_offsets, mask=mask, other=float("-inf")).to(
                compute_dtype
            )
            nan_mask = (values != values) & mask
            has_nan |= nan_mask
            nan_value = tl.where(nan_mask, values, nan_value)
            acc = tl.maximum(acc, tl.where(mask & ~nan_mask, values, float("-inf")))
        elif IS_MIN:
            values = tl.load(data + data_offsets, mask=mask, other=float("inf")).to(
                compute_dtype
            )
            nan_mask = (values != values) & mask
            has_nan |= nan_mask
            nan_value = tl.where(nan_mask, values, nan_value)
            acc = tl.minimum(acc, tl.where(mask & ~nan_mask, values, float("inf")))
        pos += 1

    if IS_MEAN:
        acc = acc / segment_length
    if IS_MAX or IS_MIN:
        acc = tl.where(has_nan, nan_value, acc)

    output_offsets = rows * inner_size + k_offsets
    tl.store(output + output_offsets, acc, mask=mask)


@triton.jit
def _lengths_to_offsets_kernel(
    lengths,
    offsets,
    outer_count,
    segment_count,
):
    pid = tle.program_id(0)
    acc = tl.full((), 0, dtype=tl.int64)
    base_lengths = pid * segment_count
    base_offsets = pid * (segment_count + 1)
    tl.store(offsets + base_offsets, acc)

    idx = 0
    while idx < segment_count:
        length = tl.load(lengths + base_lengths + idx)
        acc += length
        tl.store(offsets + base_offsets + idx + 1, acc)
        idx += 1


@triton.jit
def _segment_reduce_forward_kernel(
    data,
    offsets,
    output,
    segment_count,
    inner_size,
    data_size_axis,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_MAX: tl.constexpr,
    IS_MIN: tl.constexpr,
    IS_PROD: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    INITIAL_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    data_dtype = data.dtype.element_ty
    compute_dtype = tl.float64 if data_dtype is tl.float64 else tl.float32

    inner_idx = pid % inner_size
    row_idx = pid // inner_size
    dim_idx = row_idx % segment_count
    outer_idx = row_idx // segment_count

    offsets_base = outer_idx * (segment_count + 1) + dim_idx
    segment_start = tl.load(offsets + offsets_base)
    segment_end = tl.load(offsets + offsets_base + 1)
    segment_length = segment_end - segment_start

    acc = tl.full((), INITIAL_VALUE, dtype=compute_dtype)
    if IS_PROD:
        pos = segment_start
        while pos < segment_end:
            data_offset = (
                outer_idx * data_size_axis * inner_size + pos * inner_size + inner_idx
            )
            value = tl.load(data + data_offset).to(compute_dtype)
            acc *= value
            pos += 1
    else:
        pos = segment_start
        while pos < segment_end:
            segment_offsets = pos + tl.arange(0, BLOCK_SIZE)
            mask = segment_offsets < segment_end
            data_offsets = (
                outer_idx * data_size_axis * inner_size
                + segment_offsets * inner_size
                + inner_idx
            )

            if IS_SUM or IS_MEAN:
                values = tl.load(data + data_offsets, mask=mask, other=0.0).to(
                    compute_dtype
                )
                acc += tl.sum(tl.where(mask, values, 0.0), axis=0)
            elif IS_MAX:
                values = tl.load(
                    data + data_offsets, mask=mask, other=float("-inf")
                ).to(compute_dtype)
                nan_mask = (values != values) & mask
                has_nan = tl.sum(nan_mask.to(tl.int32), axis=0) > 0
                nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=0)
                chunk = tl.max(
                    tl.where(mask & ~nan_mask, values, float("-inf")), axis=0
                )
                chunk = tl.where(has_nan, nan_value, chunk)
                acc = tl.where(has_nan, chunk, tl.maximum(acc, chunk))
            elif IS_MIN:
                values = tl.load(data + data_offsets, mask=mask, other=float("inf")).to(
                    compute_dtype
                )
                nan_mask = (values != values) & mask
                has_nan = tl.sum(nan_mask.to(tl.int32), axis=0) > 0
                nan_value = tl.sum(tl.where(nan_mask, values, 0.0), axis=0)
                chunk = tl.min(tl.where(mask & ~nan_mask, values, float("inf")), axis=0)
                chunk = tl.where(has_nan, nan_value, chunk)
                acc = tl.where(has_nan, chunk, tl.minimum(acc, chunk))
            pos += BLOCK_SIZE

    if IS_MEAN:
        acc_is_nan = acc != acc
        nan_value = acc / acc
        if not HAS_INITIAL:
            acc = tl.where(segment_length == 0, nan_value, acc)
        acc = tl.where((segment_length > 0) & ~acc_is_nan, acc / segment_length, acc)

    tl.store(output + pid, acc)


@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr, out_ptr, final_states_ptr, dA_cs_ptr, initstates_ptr, seq_idx_ptr,
    # Matrix dimensions
    dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head, stride_final_states_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_initstates_batch, stride_initstates_head, stride_initstates_dim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_b * stride_initstates_batch + pid_h * stride_initstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    else:
        initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    seq_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk

# TorchRec BSD-style license: see vendor/irregular/torchrec_LICENSE.
# Keep the original optional TLX dependency; only USE_CLC=False is exercised.
try:
    import triton.language.extra.tlx as tlx
except ImportError:
    pass

@triton.jit
def _get_weight_ptr(
    weight_ptrs,
    weight_chunk_starts: tl.constexpr,
    logical_offset,
):
    weight_ptr = weight_ptrs[0] + logical_offset
    for chunk in tl.static_range(1, len(weight_chunk_starts)):
        chunk_start = tl.full((), weight_chunk_starts[chunk], tl.int64)
        weight_ptr = tl.where(
            logical_offset >= chunk_start,
            weight_ptrs[chunk] + logical_offset - chunk_start,
            weight_ptr,
        )
    return weight_ptr


@triton.jit
def _get_weight_row_start_ptr(
    weight_ptrs,
    weight_chunk_starts: tl.constexpr,
    logical_row_start,
):
    if len(weight_chunk_starts) == 8:
        chunk_start_4 = tl.full((), weight_chunk_starts[4], tl.int64)
        if logical_row_start >= chunk_start_4:
            chunk_start_6 = tl.full((), weight_chunk_starts[6], tl.int64)
            if logical_row_start >= chunk_start_6:
                chunk_start_7 = tl.full((), weight_chunk_starts[7], tl.int64)
                if logical_row_start >= chunk_start_7:
                    row_start_ptr = weight_ptrs[7] + logical_row_start - chunk_start_7
                else:
                    row_start_ptr = weight_ptrs[6] + logical_row_start - chunk_start_6
            else:
                chunk_start_5 = tl.full((), weight_chunk_starts[5], tl.int64)
                if logical_row_start >= chunk_start_5:
                    row_start_ptr = weight_ptrs[5] + logical_row_start - chunk_start_5
                else:
                    row_start_ptr = weight_ptrs[4] + logical_row_start - chunk_start_4
        else:
            chunk_start_2 = tl.full((), weight_chunk_starts[2], tl.int64)
            if logical_row_start >= chunk_start_2:
                chunk_start_3 = tl.full((), weight_chunk_starts[3], tl.int64)
                if logical_row_start >= chunk_start_3:
                    row_start_ptr = weight_ptrs[3] + logical_row_start - chunk_start_3
                else:
                    row_start_ptr = weight_ptrs[2] + logical_row_start - chunk_start_2
            else:
                chunk_start_1 = tl.full((), weight_chunk_starts[1], tl.int64)
                if logical_row_start >= chunk_start_1:
                    row_start_ptr = weight_ptrs[1] + logical_row_start - chunk_start_1
                else:
                    row_start_ptr = weight_ptrs[0] + logical_row_start
    else:
        row_start_ptr = _get_weight_ptr(
            weight_ptrs,
            weight_chunk_starts,
            logical_row_start,
        )
    return row_start_ptr


@triton.jit
def _get_weight_row_ptrs_for_update(
    weight_ptrs,
    weight_chunk_starts: tl.constexpr,
    split_weight_row_starts: tl.constexpr,
    logical_row_start,
    col_offsets,
    mask,
):
    is_split_row = False
    for split_row in tl.static_range(0, len(split_weight_row_starts)):
        split_row_start = tl.full((), split_weight_row_starts[split_row], tl.int64)
        is_split_row |= logical_row_start == split_row_start

    if is_split_row:
        row_ptrs = _get_weight_ptr(
            weight_ptrs,
            weight_chunk_starts,
            logical_row_start + col_offsets,
        )
    else:
        row_start_ptr = _get_weight_row_start_ptr(
            weight_ptrs,
            weight_chunk_starts,
            logical_row_start,
        )
        row_ptrs = row_start_ptr + col_offsets
    return row_ptrs


@triton.jit
def _stochastic_rounding_store(
    ptr,
    val,
    mask,
    seed,
    offset,
):
    """
    Store FP32 values to FP16 using stochastic rounding.

    Probabilistically rounds each element to one of the two nearest FP16
    values with probability proportional to proximity:
        P(round to ceil) = (val - floor_fp16) / ULP

    This gives unbiased rounding, critical for convergence with FP16 weights.

    Unlike the CUDA TBE approach (noise + __float2half_rz truncation),
    we use an explicit probability-based method because Triton lacks a
    round-toward-zero FP16 conversion intrinsic.
    """
    val_fp16 = val.to(tl.float16)
    val_rne = val_fp16.to(tl.float32)

    error = val - val_rne

    val_bits = val_fp16.to(tl.int16, bitcast=True)
    is_neg = val_bits < 0

    adjacent_bits = tl.where(
        error > 0,
        tl.where(is_neg, val_bits - 1, val_bits + 1),
        tl.where(is_neg, val_bits + 1, val_bits - 1),
    ).to(tl.int16)
    adjacent = adjacent_bits.to(tl.float16, bitcast=True)
    adjacent_f32 = adjacent.to(tl.float32)

    ulp = tl.abs(adjacent_f32 - val_rne)
    abs_error = tl.abs(error)

    rand = tl.rand(seed, offset)
    result = tl.where(rand * ulp < abs_error, adjacent, val_fp16)

    tl.store(ptr, result, mask=mask)


@triton.jit
def triton_tbe_backward_short_run_unweighted(
    dout_ptr,
    weight_ptrs,
    weight_chunk_starts: tl.constexpr,
    split_weight_row_starts: tl.constexpr,
    infos_sorted_ptr,
    sorted_linear_indices_run_ptr,
    sorted_linear_indices_cumulative_run_lengths_ptr,
    short_run_ids_ptr,
    table_offsets_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    feature_table_map_ptr,
    hash_size_cumsum_ptr,
    momentum_ptr,
    rows_cumsum_ptr,
    sorted_linear_indices_num_runs_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    B_offsets_ptr,
    total_embedding_dim: tl.constexpr,
    B,
    learning_rate,
    eps,
    optimizer: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    info_B_num_bits,
    info_B_mask,
    USE_CLC: tl.constexpr,
    ENABLE_TRITON_TBE_OPTIMIZATIONS: tl.constexpr,
    STOCHASTIC_ROUNDING: tl.constexpr,
    stochastic_rounding_seed,
    vbe: tl.constexpr = False,
    BUFFER_SIZE: tl.constexpr = 2,
):
    """Backward kernel for short runs only. Each program handles one short run."""
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Gather width, tuned per target in TbeBackwardConfig. Each buffered row
    # keeps BLOCK_SIZE 64-bit addresses live, so it trades memory-level
    # parallelism against register footprint and hence occupancy.
    buffer_size: tl.constexpr = BUFFER_SIZE if ENABLE_TRITON_TBE_OPTIMIZATIONS else 16
    buffer_offsets = tl.arange(0, buffer_size)

    if USE_CLC:
        clc_phase_producer = 1
        clc_phase_consumer = 0
        clc_context = tlx.clc_create_context(1)

        tile_id = tl.program_id(0)
        num_tiles = tl.num_programs(0)
        num_short_runs = tl.load(sorted_linear_indices_num_runs_ptr)
        c_runs = num_short_runs // num_tiles
        r_runs = num_short_runs - c_runs * num_tiles
    else:
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)

        num_short_runs = tl.load(sorted_linear_indices_num_runs_ptr)

        c = num_short_runs // num_programs
        r = num_short_runs - c * num_programs

        local_id = c * pid + tl.maximum(pid - num_programs + r, 0)
        local_id_end = local_id + c + (pid >= num_programs - r)

        # Early exit for excess programs when grid > num_short_runs
        if local_id >= local_id_end:
            return

    has_more_tile = True
    while has_more_tile:
        if USE_CLC:
            tlx.clc_producer(clc_context, clc_phase_producer)
            clc_phase_producer ^= 1
            local_id = c_runs * tile_id + tl.maximum(tile_id - num_tiles + r_runs, 0)
            local_id_end = local_id + c_runs + (tile_id >= num_tiles - r_runs)

        while local_id < local_id_end:
            run_id = tl.load(short_run_ids_ptr + local_id)

            linear_index = tl.load(sorted_linear_indices_run_ptr + run_id)
            segment_start = tl.load(
                sorted_linear_indices_cumulative_run_lengths_ptr + run_id
            )
            segment_end = tl.load(
                sorted_linear_indices_cumulative_run_lengths_ptr + run_id + 1
            )

            info_start = tl.load(infos_sorted_ptr + segment_start).to(tl.uint32)
            t = (info_start >> info_B_num_bits).to(tl.int32)

            table_idx = tl.load(feature_table_map_ptr + t)
            table_offset = tl.load(table_offsets_ptr + table_idx)
            embedding_dim = tl.load(embedding_dims_ptr + t)
            mask = col_offsets < embedding_dim

            grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            sz = (segment_end - segment_start) // buffer_size
            endn = segment_start + sz * buffer_size

            for idx in tl.range(
                segment_start,
                endn,
                buffer_size,
                flatten=True,
                loop_unroll_factor=2,
            ):
                info = tl.load(infos_sorted_ptr + idx + buffer_offsets).to(tl.uint32)
                b = (info & info_B_mask).to(tl.int64)
                t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
                if vbe:
                    b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                    dout_row_start_ptr = dout_ptr + tl.load(
                        row_output_offsets_ptr + b_t
                    )
                else:
                    embedding_offset_per_entry = tl.load(
                        embedding_offsets_ptr + t_per_entry
                    )
                    dout_row_start_ptr = (
                        dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                    )
                dout_row_ptrs = dout_row_start_ptr[:, None] + col_offsets[None, :]
                grad_buffer = tl.load(dout_row_ptrs, mask=mask[None, :], other=0).to(
                    tl.float32
                )
                grad += tl.sum(grad_buffer, axis=0)

            for idx in range(endn, segment_end):
                info = tl.load(infos_sorted_ptr + idx).to(tl.uint32)
                b = (info & info_B_mask).to(tl.int64)
                t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
                if vbe:
                    b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                    dout_row_start_ptr = dout_ptr + tl.load(
                        row_output_offsets_ptr + b_t
                    )
                else:
                    embedding_offset_per_entry = tl.load(
                        embedding_offsets_ptr + t_per_entry
                    )
                    dout_row_start_ptr = (
                        dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                    )
                dout_row_ptrs = dout_row_start_ptr + col_offsets
                dout_row = tl.load(dout_row_ptrs, mask=mask, other=0).to(tl.float32)
                grad += dout_row

            grad_original = grad.to(tl.float32)

            index_offset = tl.load(hash_size_cumsum_ptr + t)
            row_idx = linear_index - index_offset
            logical_row_start = table_offset + row_idx * embedding_dim
            row_ptrs = _get_weight_row_ptrs_for_update(
                weight_ptrs,
                weight_chunk_starts,
                split_weight_row_starts,
                logical_row_start,
                col_offsets,
                mask,
            )
            row = tl.load(row_ptrs, mask=col_offsets < embedding_dim, other=0).to(
                tl.float32
            )

            row_update = row - learning_rate * grad_original

            if optimizer == 1:
                row_offset = tl.load(rows_cumsum_ptr + table_idx)
                momentum_idx = row_offset + row_idx

                grad_square = grad_original * grad_original
                grad_square_average = tl.sum(grad_square) / embedding_dim

                momentum = tl.load(momentum_ptr + momentum_idx)
                momentum_new = momentum + grad_square_average
                tl.store(momentum_ptr + momentum_idx, momentum_new)

                adaptive_learning_rate = learning_rate / (tl.sqrt(momentum_new) + eps)

                row_update = row - adaptive_learning_rate * grad_original

            if STOCHASTIC_ROUNDING:
                sr_offset = local_id * BLOCK_SIZE + col_offsets
                _stochastic_rounding_store(
                    row_ptrs, row_update, mask, stochastic_rounding_seed, sr_offset
                )
            else:
                tl.store(row_ptrs, row_update, mask=mask)

            local_id += 1

        if USE_CLC:
            tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
            clc_phase_consumer ^= 1
            has_more_tile = tile_id != -1
        else:
            has_more_tile = False



# Additional original wrapper helpers for exact uniform discovery flow.
_SUPPORTED_REDUCES = ("sum", "mean", "max", "min", "prod")
_SUPPORTED_DATA_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

def _wrap_axis(axis, ndim):
    if ndim == 0:
        raise IndexError(
            "segment_reduce(): input tensor must have at least one dimension."
        )
    if axis < -ndim or axis >= ndim:
        raise IndexError(
            f"segment_reduce(): axis {axis} is out of bounds for tensor of dimension {ndim}."
        )
    return axis % ndim


def _check_reduce_and_dtype(data, reduce):
    if reduce not in _SUPPORTED_REDUCES:
        raise RuntimeError(
            "segment_reduce(): reduce must be one of 'sum', 'mean', 'max', 'min', or 'prod'."
        )
    if data.dtype not in _SUPPORTED_DATA_DTYPES:
        raise NotImplementedError(f'"segment_reduce" not implemented for {data.dtype}.')


def _is_unit_lengths(data, lengths, axis):
    if tuple(lengths.shape[:-1]) != tuple(data.shape[:axis]):
        return False
    if lengths.shape[-1] != data.shape[axis]:
        return False
    return _all_lengths_equal(lengths, 1)

