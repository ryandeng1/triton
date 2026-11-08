"""Bind explicit caller contracts to an unchanged Triton or Gluon kernel.

Unlike the FX frontend, this low-level interface does not discover a producer.
The caller establishes that the first ``index_count`` index elements are an
initialized, immutable, globally unique sequence in ``[0, index_bound)``. The
output allocation is initialized and private through kernel completion: it has
no concurrent observers or synchronization role. Every argument access stays
within its backing allocation. ``initial_zero=True`` additionally promises
physical positive zero before *every* invocation, including repeated launches.

These are semantic preconditions, not properties checked by scanning payloads.
Runtime checks bind the index tensor and launch geometry, reject ordinary index
mutation, and conservatively require disjoint output/input backing allocations.
Raw device writes and external concurrent accesses remain the caller's obligation.
The native pass proves destination injectivity from the actual address/mask IR;
unsupported kernels retain their atomics. No source-level atomic/store switch is
inserted. Only NVIDIA CUDA and one CTA per program are currently supported.
"""

import copy
import re

import torch
import triton

_I32_MAX = (1 << 31) - 1


def _grid_tuple(grid):
    if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3:
        raise ValueError("grid must contain one to three concrete positive dimensions")
    grid = tuple(grid) + (1, ) * (3 - len(grid))
    if any(type(x) is not int or x <= 0 or x > _I32_MAX for x in grid):
        raise ValueError("grid dimensions must be positive int32 values")
    return grid


def _signature(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), value.dtype, str(value.device),
                value.data_ptr() % 16, value.is_neg(), value.is_conj())
    if type(value) in (bool, int, float):
        return (type(value).__name__, repr(value))
    raise TypeError("native contract arguments must be tensors or concrete bool/int/float scalars")


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors have no version counter. Their immutable-input
        # promise is still a caller contract; there is no payload inspection.
        return None


def _storage_span(tensor):
    storage = tensor.untyped_storage()
    first = storage.data_ptr()
    return first, first + storage.nbytes()


def _check_storage(args, index_arg, output_arg):
    indices, output = args[index_arg], args[output_arg]
    if not isinstance(indices, torch.Tensor) or not isinstance(output, torch.Tensor):
        raise TypeError("index and output arguments must be tensors")
    if indices.dtype not in (torch.int32, torch.int64) or indices.ndim != 1:
        raise ValueError("indices must be a contiguous one-dimensional int32/int64 tensor")
    for value in args:
        if not isinstance(value, torch.Tensor):
            continue
        if value.device.type != "cuda" or torch.version.hip is not None or value.device != indices.device:
            raise ValueError("all tensor arguments must be on the same NVIDIA CUDA device")
        if value.layout != torch.strided or not value.is_contiguous() or value.is_neg() or value.is_conj():
            raise ValueError("native contract tensors must be contiguous with resolved lazy views")
        if value.numel() > _I32_MAX:
            raise ValueError("tensor size exceeds the supported int32 footprint")
    if torch.cuda.current_device() != indices.device.index:
        raise ValueError("make the tensor's CUDA device current before compiling or launching")
    out_first, out_last = _storage_span(output)
    if out_first == out_last:
        raise ValueError("the contracted output must have nonempty backing storage")
    for number, value in enumerate(args):
        if number == output_arg or not isinstance(value, torch.Tensor):
            continue
        first, last = _storage_span(value)
        if out_first < last and first < out_last:
            raise ValueError("private output backing storage overlaps an input allocation")


class NativeIndexedUpdate:
    """Compiled baseline and contract variant, with guarded optimized launches.

    ``plan(*args, grid=...)`` checks the binding before launching. The exposed
    ``baseline`` and ``optimized`` CompiledKernel objects allow controlled
    low-level experiments; direct launches bypass these runtime checks and must
    independently honor every recorded semantic precondition.
    """

    def __init__(self, baseline, optimized, args, grid, index_arg, output_arg, attrs, index_count, index_bound,
                 initial_zero):
        self.baseline = baseline
        self.optimized = optimized
        self.grid = grid
        self.index_arg = index_arg
        self.output_arg = output_arg
        self.indices = args[index_arg]
        self.index_version = _version(self.indices)
        self.signatures = tuple(_signature(value) for value in args)
        self.attrs = attrs
        self.index_count = index_count
        self.index_bound = index_bound
        self.initial_zero = initial_zero

    def __call__(self, *args, grid=None):
        actual_grid = self.grid if grid is None else _grid_tuple(grid)
        if actual_grid != self.grid:
            raise ValueError("launch grid differs from the native injectivity proof")
        if tuple(_signature(value) for value in args) != self.signatures:
            raise ValueError("argument metadata or scalar specialization changed")
        if args[self.index_arg] is not self.indices:
            raise ValueError("index tensor differs from the bound immutable sequence")
        if self.index_version is not None and _version(self.indices) != self.index_version:
            raise ValueError("index tensor was mutated after its semantic contract was bound")
        _check_storage(args, self.index_arg, self.output_arg)
        return self.optimized[self.grid](*args)

    run = __call__

    def report(self):
        variants = {}
        for name, kernel in (("baseline", self.baseline), ("optimized", self.optimized)):
            kernel._init_handles()
            ptx = kernel.asm.get("ptx", "")
            variants[name] = {
                "atomic_instruction_sites": len(re.findall(r"\b(?:atom|red)\.", ptx)),
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
                "shared_bytes": kernel.metadata.shared,
            }
        ir = self.optimized.asm.get("ttir", "") + self.optimized.asm.get("ttgir", "")
        return {
            "transformed": "tt.experimental_indexed_update_specialized" in ir,
            "grid": self.grid,
            "index_count": self.index_count,
            "index_bound": self.index_bound,
            "initial_zero": self.initial_zero,
            "index_version_guard": self.index_version is not None,
            "native_attributes": {str(key): value
                                  for key, value in self.attrs.items()},
            "proof":
            "native address/mask analysis from immutable unique index, range, launch and private-region contracts",
            "scope": "unchanged source; no host launch removal; baseline and optimized use the same kernel arguments",
            "variants": variants,
        }


def compile_indexed_update(kernel, args, *, grid, index_arg, output_arg, index_count, index_bound, initial_zero=False,
                           options=None):
    """Compile an unchanged kernel under the explicit contracts in this module.

    Argument positions may be integer positions or names. Pass every argument,
    including constexprs, positionally in ``args``. Grid, scalar values, tensor
    metadata and the particular immutable index tensor are bound to the result.
    A producer such as randperm can establish uniqueness; this function never
    infers it from example values. Initialize/reset output before each invocation
    according to the selected zero/nonzero contract.
    """
    if not isinstance(args, tuple) or len(args) != len(kernel.arg_names):
        raise ValueError("args must provide every kernel argument positionally, including constexprs")

    def position(value):
        if isinstance(value, str):
            return kernel.arg_names.index(value)
        if type(value) is not int or not 0 <= value < len(args):
            raise ValueError("invalid contracted argument position")
        return value

    index_arg, output_arg = position(index_arg), position(output_arg)
    if index_arg == output_arg:
        raise ValueError("immutable index and private output must be different arguments")
    grid = _grid_tuple(grid)
    options = dict(options or {})
    if options.get("num_ctas", 1) != 1:
        raise ValueError("native indexed-update specialization currently requires num_ctas=1")
    if (type(index_count) is not int or type(index_bound) is not int or not 0 < index_count <= index_bound <= _I32_MAX):
        raise ValueError("unique index prefix and upper bound must be positive, compatible int32 values")
    if type(initial_zero) is not bool:
        raise TypeError("initial_zero must be a boolean caller contract")
    _check_storage(args, index_arg, output_arg)
    if index_count > args[index_arg].numel():
        raise ValueError("unique index prefix exceeds the bound index tensor")
    for value in args:
        _signature(value)
    baseline_options = dict(options, enable_indexed_update_specialization=False)
    baseline = kernel.warmup(*args, grid=grid, **baseline_options)
    attrs = copy.deepcopy(baseline.src.attrs)

    def add(number, name, value):
        key = (number, )
        attrs[key] = [(old_name, old_value) for old_name, old_value in attrs.get(key, []) if old_name != name]
        attrs[key].append((name, value))

    for number, value in enumerate(args):
        if isinstance(value, torch.Tensor):
            add(number, "tt.experimental_alias_group", 1 if number == output_arg else 0)
    add(index_arg, "tt.experimental_unique_indices", index_count)
    add(index_arg, "tt.experimental_index_bound", index_bound)
    add(output_arg, "tt.experimental_private", 1)
    add(output_arg, "tt.experimental_num_ctas", 1)
    for axis, extent in zip("xyz", grid):
        add(output_arg, "tt.experimental_grid_" + axis, extent)
    if initial_zero:
        add(output_arg, "tt.experimental_initial_zero", 1)
    source = type(baseline.src)(baseline.src.fn, baseline.src.signature, constexprs=baseline.src.constants, attrs=attrs)
    optimized = triton.compile(source, target=baseline.metadata.target,
                               options=dict(options, enable_indexed_update_specialization=True))
    return NativeIndexedUpdate(baseline, optimized, args, grid, index_arg, output_arg, attrs, index_count, index_bound,
                               initial_zero)
