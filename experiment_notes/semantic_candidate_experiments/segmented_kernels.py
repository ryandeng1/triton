# Copyright 2026 FlagOS Contributors. Apache-2.0; see vendor/segmented/flag_LICENSE.
# Function signature/body copied unchanged; external dispatch decorators omitted.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "idx_ncols", "src_stride0", "out_ncols"])
def scatter_add_2d_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    N,
    idx_ncols,
    src_stride0,
    out_ncols,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)
    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N
        row = offsets // idx_ncols
        col = offsets % idx_ncols
        idx_offsets = row * idx_ncols + col
        src_offsets = row * src_stride0 + col
        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
        if DIM == 0:
            out_offsets = idx * out_ncols + col
        else:
            out_offsets = row * out_ncols + idx
        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0)
        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")
