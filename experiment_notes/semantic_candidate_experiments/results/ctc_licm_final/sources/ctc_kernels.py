# Extracted from FlagGems, Apache-2.0; see ctc_loss.upstream.py.
import triton
import triton.language as tl
_debug_barrier = tl.debug_barrier

@triton.jit
def _logaddexp(a, b):
    max_ab = tl.maximum(a, b)
    min_ab = tl.minimum(a, b)
    return tl.where(max_ab == -float('inf'), -float('inf'), max_ab + tl.log(1.0 + tl.exp(min_ab - max_ab)))

@triton.jit
def _logaddexp3(a, b, c, use_c):
    c = tl.where(use_c, c, -float('inf'))
    max_abc = tl.maximum(tl.maximum(a, b), c)
    safe_max = tl.where(max_abc == -float('inf'), 0.0, max_abc)
    exp_sum = tl.exp(a - safe_max) + tl.exp(b - safe_max) + tl.exp(c - safe_max)
    return tl.where(max_abc == -float('inf'), -float('inf'), max_abc + tl.log(exp_sum))

@triton.jit
def _ctc_loss_forward_kernel(log_probs, targets, input_lengths, target_lengths, target_offsets, neg_log_likelihood, log_alpha, T: tl.constexpr, N: tl.constexpr, C: tl.constexpr, MAX_TARGET: tl.constexpr, STATE_COUNT_MAX: tl.constexpr, BLANK: tl.constexpr, TARGET_1D: tl.constexpr, BLOCK_S: tl.constexpr):
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)
    input_len = tl.load(input_lengths + batch)
    target_len = tl.load(target_lengths + batch)
    state_count = target_len * 2 + 1
    valid_state = states < state_count
    stored_state = states < STATE_COUNT_MAX
    is_blank_state = states % 2 == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)
    if TARGET_1D:
        target_base = tl.full((), 0, tl.int64)
        for prev_batch in tl.range(0, N):
            target_base += tl.load(target_lengths + prev_batch, mask=prev_batch < batch, other=0)
        target_origin = target_base
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index
    target_value = tl.load(target_ptrs, mask=target_mask, other=BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)
    t0_active = input_len > 0
    init_state = (states == 0) | (states == 1) & (target_len > 0)
    init_logp = tl.load(log_probs + batch * C + labels, mask=init_state & stored_state & t0_active, other=0.0).to(tl.float32)
    alpha = tl.where(init_state & valid_state & t0_active, init_logp, -float('inf'))
    tl.store(log_alpha + batch * T * STATE_COUNT_MAX + states, alpha, mask=stored_state)
    _debug_barrier()
    for t in tl.range(1, T):
        prev_base = log_alpha + batch * T * STATE_COUNT_MAX + (t - 1) * STATE_COUNT_MAX
        prev0 = tl.load(prev_base + states, mask=stored_state, other=-float('inf')).to(tl.float32)
        prev1 = tl.load(prev_base + tl.where(states > 0, states - 1, 0), mask=(states > 0) & stored_state, other=-float('inf')).to(tl.float32)
        prev2 = tl.load(prev_base + tl.where(states > 1, states - 2, 0), mask=(states > 1) & stored_state, other=-float('inf')).to(tl.float32)
        prev_target_index = tl.where(target_index > 0, target_index - 1, 0)
        prev_target_value = tl.load(targets + target_origin + prev_target_index, mask=target_mask & (target_index > 0), other=BLANK)
        skip_allowed = ~is_blank_state & (target_index > 0) & (target_value != prev_target_value)
        acc = _logaddexp3(prev0, prev1, prev2, skip_allowed)
        logp = tl.load(log_probs + t * N * C + batch * C + labels, mask=valid_state & (t < input_len), other=0.0).to(tl.float32)
        alpha = tl.where(valid_state & (t < input_len), acc + logp, -float('inf'))
        tl.store(log_alpha + batch * T * STATE_COUNT_MAX + t * STATE_COUNT_MAX + states, alpha, mask=stored_state)
        _debug_barrier()
    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float('inf'))
    else:
        _debug_barrier()
        final_base = log_alpha + batch * T * STATE_COUNT_MAX + (input_len - 1) * STATE_COUNT_MAX
        last = tl.load(final_base + state_count - 1).to(tl.float32)
        prev_last = tl.load(final_base + tl.where(target_len > 0, state_count - 2, 0), mask=target_len > 0, other=-float('inf')).to(tl.float32)
        log_likelihood = _logaddexp(last, prev_last)
        loss = -log_likelihood
    tl.store(neg_log_likelihood + batch, loss)

@triton.jit
def _ctc_loss_forward_kernel_manual_hoist(log_probs, targets, input_lengths, target_lengths, target_offsets, neg_log_likelihood, log_alpha, T: tl.constexpr, N: tl.constexpr, C: tl.constexpr, MAX_TARGET: tl.constexpr, STATE_COUNT_MAX: tl.constexpr, BLANK: tl.constexpr, TARGET_1D: tl.constexpr, BLOCK_S: tl.constexpr):
    batch = tl.program_id(0)
    states = tl.arange(0, BLOCK_S)
    input_len = tl.load(input_lengths + batch)
    target_len = tl.load(target_lengths + batch)
    state_count = target_len * 2 + 1
    valid_state = states < state_count
    stored_state = states < STATE_COUNT_MAX
    is_blank_state = states % 2 == 0
    target_index = (states - 1) // 2
    target_mask = (target_index >= 0) & (target_index < target_len)
    target_safe_index = tl.where(target_mask, target_index, 0)
    if TARGET_1D:
        target_base = tl.full((), 0, tl.int64)
        for prev_batch in tl.range(0, N):
            target_base += tl.load(target_lengths + prev_batch, mask=prev_batch < batch, other=0)
        target_origin = target_base
        target_ptrs = targets + target_origin + target_safe_index
    else:
        target_origin = batch * MAX_TARGET
        target_ptrs = targets + target_origin + target_safe_index
    target_value = tl.load(target_ptrs, mask=target_mask, other=BLANK)
    labels = tl.where(is_blank_state, BLANK, target_value)
    t0_active = input_len > 0
    init_state = (states == 0) | (states == 1) & (target_len > 0)
    init_logp = tl.load(log_probs + batch * C + labels, mask=init_state & stored_state & t0_active, other=0.0).to(tl.float32)
    alpha = tl.where(init_state & valid_state & t0_active, init_logp, -float('inf'))
    tl.store(log_alpha + batch * T * STATE_COUNT_MAX + states, alpha, mask=stored_state)
    _debug_barrier()
    prev_target_index = tl.where(target_index > 0, target_index - 1, 0)
    prev_target_value = tl.load(targets + target_origin + prev_target_index, mask=target_mask & (target_index > 0) & (T > 1), other=BLANK)
    skip_allowed = ~is_blank_state & (target_index > 0) & (target_value != prev_target_value)
    for t in tl.range(1, T):
        prev_base = log_alpha + batch * T * STATE_COUNT_MAX + (t - 1) * STATE_COUNT_MAX
        prev0 = tl.load(prev_base + states, mask=stored_state, other=-float('inf')).to(tl.float32)
        prev1 = tl.load(prev_base + tl.where(states > 0, states - 1, 0), mask=(states > 0) & stored_state, other=-float('inf')).to(tl.float32)
        prev2 = tl.load(prev_base + tl.where(states > 1, states - 2, 0), mask=(states > 1) & stored_state, other=-float('inf')).to(tl.float32)
        acc = _logaddexp3(prev0, prev1, prev2, skip_allowed)
        logp = tl.load(log_probs + t * N * C + batch * C + labels, mask=valid_state & (t < input_len), other=0.0).to(tl.float32)
        alpha = tl.where(valid_state & (t < input_len), acc + logp, -float('inf'))
        tl.store(log_alpha + batch * T * STATE_COUNT_MAX + t * STATE_COUNT_MAX + states, alpha, mask=stored_state)
        _debug_barrier()
    if input_len <= 0:
        loss = tl.where(target_len == 0, 0.0, float('inf'))
    else:
        _debug_barrier()
        final_base = log_alpha + batch * T * STATE_COUNT_MAX + (input_len - 1) * STATE_COUNT_MAX
        last = tl.load(final_base + state_count - 1).to(tl.float32)
        prev_last = tl.load(final_base + tl.where(target_len > 0, state_count - 2, 0), mask=target_len > 0, other=-float('inf')).to(tl.float32)
        log_likelihood = _logaddexp(last, prev_last)
        loss = -log_likelihood
    tl.store(neg_log_likelihood + batch, loss)
