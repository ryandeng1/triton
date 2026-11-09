# SPDX-License-Identifier: MIT
# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
# See AITer_LICENSE in this directory.
# Derived from ROCm/aiter at ef0a5a26b8cb38ea1308f827985e36cde964a6ab.
# Original body unchanged; manual clone moves two loads and dependent neg-exp.
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['T'])
def fused_sigmoid_gating_delta_rule_update_kernel(A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, o, h0_source, h0_indices, cu_seqlens, scale, T, B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, USE_QK_L2NORM_IN_KERNEL: tl.constexpr, IS_VARLEN: tl.constexpr):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_n, i_hv = (i_nh // HV, i_nh % HV)
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
        all = T
        T = eos - bos
    else:
        bos, eos = (i_n * T, i_n * T + T)
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(beta_x <= softplus_threshold, 1.0 / softplus_beta * tl.log(1.0 + tl.exp(beta_x)), x)
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-06)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-06)
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)

@triton.jit(do_not_specialize=['T'])
def recurrent_manual_hoist(A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, o, h0_source, h0_indices, cu_seqlens, scale, T, B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, USE_QK_L2NORM_IN_KERNEL: tl.constexpr, IS_VARLEN: tl.constexpr):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_n, i_hv = (i_nh // HV, i_nh % HV)
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
        all = T
        T = eos - bos
    else:
        bos, eos = (i_n * T, i_n * T + T)
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    b_A_log = tl.load(p_A_log, mask=T > 0).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias, mask=T > 0).to(tl.float32)
    negative_exp_A = -tl.exp(b_A_log)
    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(beta_x <= softplus_threshold, 1.0 / softplus_beta * tl.log(1.0 + tl.exp(beta_x)), x)
        b_g = negative_exp_A * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-06)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-06)
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)
