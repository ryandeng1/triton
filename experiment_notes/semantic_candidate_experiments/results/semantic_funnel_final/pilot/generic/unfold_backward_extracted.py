# Extracted from the pinned Apache-2.0 FlagGems source; function body unchanged.
import triton
import triton.language as tl

@triton.jit
def _unfold_backward_kernel(grad_in_ptr, grad_out_ptr, numel_in, prod_after, L, size, step, D, inner_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel_in
    vals = tl.load(grad_in_ptr + offs, mask=mask, other=0)
    vals_f32 = tl.cast(vals, tl.float32)
    k = offs % size
    tmp1 = offs // size
    after_lin = tmp1 % prod_after
    tmp2 = offs // (prod_after * size)
    s = tmp2 % L
    before_lin = offs // inner_total
    pos = s * step + k
    out_id = (before_lin * D + pos) * prod_after + after_lin
    tl.atomic_add(grad_out_ptr + out_id, vals_f32, mask=mask)
