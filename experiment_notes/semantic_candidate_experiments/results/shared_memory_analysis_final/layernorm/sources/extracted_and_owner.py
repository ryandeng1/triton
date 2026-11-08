# Extracted from Tritonbench tutorial.py; exact upstream file and license retained.
import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _layer_norm_bwd_dx_fused(DX, DY, DW, DB, X, W, B, Mean, Rstd, Lock, stride, N, eps, GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = dy.to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)

@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)

@triton.jit
def owner_row(DX, DY, X, Mean, Rstd, stride, N, row, cols, mask, w):
    X += row * stride
    DY += row * stride
    DX += row * stride
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = dy.to(w.dtype)
    return (partial_dw, partial_db)

@triton.jit
def owner_same_grid(DX, DY, DW, DB, X, W, B, Mean, Rstd, Lock, stride, N, eps,
                    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    owner = tl.program_id(0)
    rows = tl.num_programs(0)
    if owner < GROUP_SIZE_M:
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        first_dw, first_db = owner_row(DX, DY, X, Mean, Rstd, stride, N, owner, cols, mask, w)
        acc_dw = first_dw.to(DW.dtype.element_ty)
        acc_db = first_db.to(DB.dtype.element_ty)
        for row in range(owner + GROUP_SIZE_M, rows, GROUP_SIZE_M):
            new_dw, new_db = owner_row(DX, DY, X, Mean, Rstd, stride, N, row, cols, mask, w)
            acc_dw = (new_dw + acc_dw.to(tl.float32)).to(DW.dtype.element_ty)
            acc_db = (new_db + acc_db.to(tl.float32)).to(DB.dtype.element_ty)
        tl.store(DW + owner * N + cols, acc_dw, mask)
        tl.store(DB + owner * N + cols, acc_db, mask)
        tl.store(Lock + GROUP_SIZE_M + owner, 1)
        tl.store(Lock + owner, 0)
