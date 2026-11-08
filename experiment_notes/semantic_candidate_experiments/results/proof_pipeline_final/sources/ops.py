"""Functional CUDA operators used by the semantic-contract graph experiment.

These are explicit integration points, not replacements for arbitrary PyTorch
operators. Their outputs are fresh allocations and their inputs are not mutated.
The graph compiler can therefore combine producer proofs with these operators'
bounded memory footprints. Autograd formulas are not registered.
"""

import torch


@torch.library.custom_op("triton_semantics::rms_norm", mutates_args=())
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, num_warps: int,
              grid_multiplier: int) -> tuple[torch.Tensor, torch.Tensor]:
    from .backend import RMSPlan
    return RMSPlan(enabled=False, autotune=False).run(x, weight, eps, num_warps, grid_multiplier)


@_rms_norm.register_fake
def _rms_norm_fake(x, weight, eps, num_warps, grid_multiplier):
    if x.ndim != 2 or weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("rms_norm requires x[M,N] and weight[N]")
    return x.new_empty(x.shape), x.new_empty((x.shape[0], ), dtype=torch.float32)


@torch.library.custom_op("triton_semantics::embedding_backward", mutates_args=())
def _embedding_backward(grad: torch.Tensor, indices: torch.Tensor, num_weights: int, padding_idx: int,
                        scale_by_freq: bool, block_d: int) -> torch.Tensor:
    from .backend import EmbeddingPlan
    return EmbeddingPlan(unique=False).run(grad, indices, num_weights, padding_idx, scale_by_freq, block_d)


@_embedding_backward.register_fake
def _embedding_backward_fake(grad, indices, num_weights, padding_idx, scale_by_freq, block_d):
    if grad.ndim != 2 or indices.ndim != 1 or grad.shape[0] != indices.shape[0]:
        raise ValueError("embedding_backward requires grad[N,D] and indices[N]")
    return grad.new_empty((num_weights, grad.shape[1]))


def rms_norm(x, weight, eps=1e-6, num_warps=4, grid_multiplier=1):
    """Return fresh (normalized values, FP32 inverse RMS), for contiguous CUDA tensors."""
    return _rms_norm(x, weight, float(eps), int(num_warps), int(grid_multiplier))


def embedding_backward(grad, indices, num_weights, padding_idx=-1, scale_by_freq=False, block_d=128):
    """Sum gradients by row index into fresh storage; invalid/padding rows are dropped.

    Accumulation uses FP32 NVIDIA global-atomic arithmetic, followed by conversion
    to grad.dtype. Frequency scaling divides each contribution by its valid row
    count. The eager implementation always uses atomics. The graph compiler may
    specialize only when it derives global index uniqueness from producers.
    """
    return _embedding_backward(grad, indices, int(num_weights), int(padding_idx), bool(scale_by_freq), int(block_d))
