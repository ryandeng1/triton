# Extracted from pinned FlagGems sources in this directory under Apache-2.0.
# Kernel signatures and bodies are unchanged. Provider dispatch decorators are
# omitted so all variants can independently use the same bounded configurations.
import triton
import triton.language as tl

@triton.jit
def _log_softmax_bwd_kernel(
    grad_ptr,  # pointer to grad_output (dL/dy)
    y_logsm_ptr,  # pointer to output of log_softmax (y = log_softmax(x))
    grad_in_ptr,  # pointer to grad_input (dL/dx)
    M,  # number of rows (product of all dims except reduction dim)
    K,  # length of the reduction dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * K
    offs = tl.arange(0, BLOCK_SIZE)

    # First pass: compute s = sum_j grad_output[j] along dim
    s = tl.zeros((), dtype=tl.float32)
    num_chunks = tl.cdiv(K, BLOCK_SIZE)
    for chunk in range(0, num_chunks):
        cols = chunk * BLOCK_SIZE + offs
        mask = cols < K
        go_chunk = tl.load(grad_ptr + row_start + cols, mask=mask, other=0.0)
        go32 = go_chunk.to(tl.float32)
        s += tl.sum(go32, axis=0)

    # Second pass: grad_input = grad_output - exp(output) * s
    for chunk in range(0, num_chunks):
        cols = chunk * BLOCK_SIZE + offs
        mask = cols < K
        go_chunk = tl.load(grad_ptr + row_start + cols, mask=mask, other=0.0)
        go32 = go_chunk.to(tl.float32)
        y_chunk = tl.load(y_logsm_ptr + row_start + cols, mask=mask, other=0.0)
        y32 = y_chunk.to(tl.float32)
        sm = tl.exp(y32)
        gi32 = go32 - sm * s
        gi = gi32.to(go_chunk.dtype)
        tl.store(grad_in_ptr + row_start + cols, gi, mask=mask)

@triton.jit
def apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = tl.program_id(0).to(tl.int64)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM
    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    oq_ptr += s_id * oq_stride_s
    q_ptr += s_id * q_stride_s

    for off_h in range(0, NUM_Q_HEADS):
        ordered_cols = off_h * q_stride_h + (ordered_block * q_stride_d)
        rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)
        output_offs = off_h * oq_stride_h + (ordered_block * oq_stride_d)

        q = tl.load(q_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        y = q * cos + rotated_q * sin
        tl.store(oq_ptr + output_offs, y, mask=mask)

    ok_ptr += s_id * ok_stride_s
    k_ptr += s_id * k_stride_s

    for off_h in range(0, NUM_K_HEADS):
        ordered_cols = off_h * k_stride_h + (ordered_block * k_stride_d)
        rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)
        output_offs = off_h * ok_stride_h + (ordered_block * ok_stride_d)

        k = tl.load(k_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        y = k * cos + rotated_k * sin
        tl.store(ok_ptr + output_offs, y, mask=mask)

