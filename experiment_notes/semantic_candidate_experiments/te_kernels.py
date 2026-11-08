# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from NVIDIA/TransformerEngine at c1c2d4d2fa008386f247c2fec56a998e27134e0a.
# Full original source and license are retained under vendor/.
# Changes: extract helpers/kernel without provider autotuning; add a clone that
# moves only merging_prob_off/merging_prob before the hidden-dimension loop.
# The baseline function body and all row-map producer bodies are unchanged.

import triton
import triton.language as tl
from triton.language import core
from triton.language.standard import _log2

get_int_dtype = triton.constexpr_function(core.get_int_dtype)

@triton.jit
def _compare_and_swap(x, indices, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * (2**i), 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    z = tl.reshape(indices, shape)

    mask = tl.arange(0, 2)[None, :, None]

    l_value = tl.reshape(tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape), x.shape).to(
        x.dtype
    )
    r_value = tl.reshape(tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape), x.shape).to(
        x.dtype
    )

    l_indice = tl.reshape(tl.broadcast_to(tl.sum(z * (1 - mask), 1)[:, None, :], shape), x.shape)
    r_indice = tl.reshape(tl.broadcast_to(tl.sum(z * mask, 1)[:, None, :], shape), x.shape)

    idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)

    il_value = l_value.to(idtype, bitcast=True)
    ir_value = r_value.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    flag1 = tl.where(((l_value > r_value) ^ flip) != 0, il_value ^ ir_value, tl.zeros_like(ix))
    ret = ix ^ flag1
    flag2 = tl.where(((l_value > r_value) ^ flip) != 0, l_indice ^ r_indice, tl.zeros_like(ix))
    ind = indices ^ flag2

    return ret.to(x.dtype, bitcast=True), ind


@triton.jit
def _bitonic_merge(x, indices, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    if order == 2:
        shape: tl.constexpr = [n_outer * (2 ** (n_dims - 1 - stage)), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = tl.full(x.shape, value=order, dtype=tl.int32)
    for i in tl.static_range(stage):
        x, indices = _compare_and_swap(x, indices, flip, i + (n_dims - stage), n_dims)
    return x, indices


@triton.jit
def _argsort(x, indices, n_dims: tl.constexpr):
    for i in tl.static_range(1, n_dims + 1):
        x, indices = _bitonic_merge(x, indices, i, 2 if i < n_dims else 1, n_dims)
    return x, indices


@triton.jit
def _row_id_map_pass_1_kernel(
    # input pointers
    routing_map_ptr,
    # sizes
    num_tokens,
    # strides
    stride_routing_map_token,
    stride_routing_map_expert,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # output pointers
    row_id_map_ptr,
    workspace_ptr,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_token_mask = tl.load(
        routing_map_ptr + pid_m * stride_routing_map_expert + offset * stride_routing_map_token,
        mask=(offset < num_tokens),
        other=0,
    ).to(tl.int32)
    row_id_within_token_block = tl.cumsum(expert_token_mask) * expert_token_mask
    tl.store(
        row_id_map_ptr + pid_m * stride_row_id_map_expert + offset * stride_row_id_map_token,
        row_id_within_token_block,
        mask=offset < num_tokens,
    )
    n_tokens_per_block = tl.sum(expert_token_mask)
    tl.store(workspace_ptr + pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n, n_tokens_per_block)


@triton.jit
def _row_id_map_pass_2_kernel(
    # pointers
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    num_out_tokens,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # metas
    WORKSPACE_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    chunk_idx = pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_id_within_token_block = tl.load(
        row_id_map_ptr + pid_m * stride_row_id_map_expert + offset * stride_row_id_map_token,
        mask=(offset < num_tokens),
        other=0,
    )

    workspace_off = tl.arange(0, WORKSPACE_LOAD_WIDTH)
    n_tokens_per_chunk = tl.load(workspace_ptr + workspace_off, mask=workspace_off < chunk_idx)
    row_id = tl.where(
        row_id_within_token_block == 0,
        -1,
        row_id_within_token_block + tl.sum(n_tokens_per_chunk) - 1,
    )
    # capacity limit: destinations at or past num_out_tokens are dropped, matching
    # the `idx >= num_out_tokens` branch of moe_permute_row_map in permutation.cu
    row_id = tl.where(row_id >= num_out_tokens, -1, row_id)
    tl.store(
        row_id_map_ptr + pid_m * stride_row_id_map_expert + offset * stride_row_id_map_token,
        row_id,
        mask=(offset < num_tokens),
    )


@triton.jit
def _row_id_map_pass_3_kernel(
    # pointers
    row_id_map_ptr,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # metas
    num_experts: tl.constexpr,
    LOAD_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_dims: tl.constexpr = _log2(LOAD_SIZE)
    off = tl.arange(0, LOAD_SIZE)
    row_id_map = tl.load(
        row_id_map_ptr + pid * stride_row_id_map_token + stride_row_id_map_expert * off,
        mask=off < num_experts,
        other=-1,
    )
    n_routed = tl.sum(tl.where(row_id_map != -1, 1, 0))
    indices = off
    sorted_map, indices = _argsort(row_id_map, indices, n_dims=n_dims)
    tl.store(
        row_id_map_ptr + pid * stride_row_id_map_token + off * stride_row_id_map_expert,
        sorted_map,
        mask=off < n_routed,
    )
    tl.store(
        row_id_map_ptr
        + pid * stride_row_id_map_token
        + (num_experts + off) * stride_row_id_map_expert,
        indices,
        mask=off < n_routed,
    )
    tl.store(
        row_id_map_ptr + pid * stride_row_id_map_token + num_experts * 2 * stride_row_id_map_expert,
        n_routed,
    )


@triton.jit
def _unpermute_bwd_with_merging_probs_kernel(
    # input pointers
    fwd_output_grad_ptr,
    fwd_input_ptr,
    merging_probs_ptr,
    row_id_map_ptr,
    pad_offsets_ptr,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_fwd_output_grad_token,
    stride_fwd_output_grad_hidden,
    stride_fwd_input_grad_token,
    stride_fwd_input_grad_hidden,
    stride_fwd_input_token,
    stride_fwd_input_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_merging_probs_grad_token,
    stride_merging_probs_grad_expert,
    # output pointers
    fwd_input_grad_ptr,
    merging_probs_grad_ptr,
    # metas
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    PROBS_LOAD_WIDTH: tl.constexpr,
    FUSION_UNPAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = fwd_output_grad_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)
    map_load_off = tl.arange(0, PROBS_LOAD_WIDTH)
    token_probs_grad_off = (
        pid * stride_merging_probs_grad_token + stride_merging_probs_grad_expert * map_load_off
    )
    tl.store(merging_probs_grad_ptr + token_probs_grad_off, 0.0, mask=map_load_off < num_experts)
    # Zeroing and routed stores are distributed across warps.
    # Synchronize before any warp enters the routed-write loop.
    tl.debug_barrier()
    n_routed = tl.load(
        row_id_map_ptr + pid * stride_row_id_map_token + num_experts * 2 * stride_row_id_map_expert
    )
    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr + pid * stride_row_id_map_token + idx * stride_row_id_map_expert
        ).to(tl.int64)
        expert_idx = tl.load(
            row_id_map_ptr
            + pid * stride_row_id_map_token
            + (num_experts + idx) * stride_row_id_map_expert
        )
        if FUSION_UNPAD:
            pad_off = tl.load(pad_offsets_ptr + expert_idx)
            dst_row = dst_row + pad_off
        prob_grad_accum = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        current_start = 0
        while current_start < hidden_size:
            current_offset = current_start + tl.arange(0, BLOCK_SIZE)
            mask = current_offset < hidden_size
            src_row = pid.to(tl.int64)
            input_off = (
                src_row * stride_fwd_output_grad_token
                + current_offset * stride_fwd_output_grad_hidden
            )
            inp = tl.load(fwd_output_grad_ptr + input_off, mask=mask)
            inp = inp.to(compute_type)
            merging_prob_off = (
                pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
            )
            merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
            output = inp * merging_prob
            output = output.to(data_type)
            output_off = (
                dst_row * stride_fwd_input_grad_token
                + current_offset * stride_fwd_input_grad_hidden
            )
            tl.store(fwd_input_grad_ptr + output_off, output, mask=mask)

            fwd_input_off = (
                dst_row * stride_fwd_input_token + current_offset * stride_fwd_input_hidden
            )
            fwd_input = tl.load(fwd_input_ptr + fwd_input_off, mask=mask)
            prob_grad_accum += fwd_input.to(compute_type) * inp
            current_start += BLOCK_SIZE
        probs_grad = tl.sum(prob_grad_accum).to(merging_probs_grad_ptr.dtype.element_ty)
        probs_grad_off = (
            pid * stride_merging_probs_grad_token + expert_idx * stride_merging_probs_grad_expert
        )
        tl.store(merging_probs_grad_ptr + probs_grad_off, probs_grad)


@triton.jit
def unpermute_bwd_probability_hoisted(
    # input pointers
    fwd_output_grad_ptr,
    fwd_input_ptr,
    merging_probs_ptr,
    row_id_map_ptr,
    pad_offsets_ptr,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_fwd_output_grad_token,
    stride_fwd_output_grad_hidden,
    stride_fwd_input_grad_token,
    stride_fwd_input_grad_hidden,
    stride_fwd_input_token,
    stride_fwd_input_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_merging_probs_grad_token,
    stride_merging_probs_grad_expert,
    # output pointers
    fwd_input_grad_ptr,
    merging_probs_grad_ptr,
    # metas
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    PROBS_LOAD_WIDTH: tl.constexpr,
    FUSION_UNPAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = fwd_output_grad_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)
    map_load_off = tl.arange(0, PROBS_LOAD_WIDTH)
    token_probs_grad_off = (
        pid * stride_merging_probs_grad_token + stride_merging_probs_grad_expert * map_load_off
    )
    tl.store(merging_probs_grad_ptr + token_probs_grad_off, 0.0, mask=map_load_off < num_experts)
    # Zeroing and routed stores are distributed across warps.
    # Synchronize before any warp enters the routed-write loop.
    tl.debug_barrier()
    n_routed = tl.load(
        row_id_map_ptr + pid * stride_row_id_map_token + num_experts * 2 * stride_row_id_map_expert
    )
    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr + pid * stride_row_id_map_token + idx * stride_row_id_map_expert
        ).to(tl.int64)
        expert_idx = tl.load(
            row_id_map_ptr
            + pid * stride_row_id_map_token
            + (num_experts + idx) * stride_row_id_map_expert
        )
        if FUSION_UNPAD:
            pad_off = tl.load(pad_offsets_ptr + expert_idx)
            dst_row = dst_row + pad_off
        merging_prob_off = (
            pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
        )
        merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
        prob_grad_accum = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        current_start = 0
        while current_start < hidden_size:
            current_offset = current_start + tl.arange(0, BLOCK_SIZE)
            mask = current_offset < hidden_size
            src_row = pid.to(tl.int64)
            input_off = (
                src_row * stride_fwd_output_grad_token
                + current_offset * stride_fwd_output_grad_hidden
            )
            inp = tl.load(fwd_output_grad_ptr + input_off, mask=mask)
            inp = inp.to(compute_type)
            output = inp * merging_prob
            output = output.to(data_type)
            output_off = (
                dst_row * stride_fwd_input_grad_token
                + current_offset * stride_fwd_input_grad_hidden
            )
            tl.store(fwd_input_grad_ptr + output_off, output, mask=mask)

            fwd_input_off = (
                dst_row * stride_fwd_input_token + current_offset * stride_fwd_input_hidden
            )
            fwd_input = tl.load(fwd_input_ptr + fwd_input_off, mask=mask)
            prob_grad_accum += fwd_input.to(compute_type) * inp
            current_start += BLOCK_SIZE
        probs_grad = tl.sum(prob_grad_accum).to(merging_probs_grad_ptr.dtype.element_ty)
        probs_grad_off = (
            pid * stride_merging_probs_grad_token + expert_idx * stride_merging_probs_grad_expert
        )
        tl.store(merging_probs_grad_ptr + probs_grad_off, probs_grad)
