"""Bind explicit caller contracts to an unchanged Triton or Gluon kernel.

Unlike the FX frontend, this low-level interface does not discover a producer.
The caller establishes that the first ``index_count`` index elements are an
initialized, immutable, globally unique sequence in ``[0, index_bound)``. The
output allocation is initialized and private through kernel completion: no code
outside this invocation accesses it. Every argument access stays
within its backing allocation. ``initial_zero=True`` additionally promises
physical positive zero before *every* invocation, including repeated launches.

These are semantic preconditions, not properties checked by scanning payloads.
Runtime checks bind the index tensor and launch geometry, reject ordinary index
mutation, and conservatively require disjoint output/input backing allocations.
Raw device writes and external concurrent accesses remain the caller's obligation.
The native pass proves destination injectivity from the actual address/mask IR;
unsupported kernels retain their atomics. No source-level atomic/store switch is
inserted. Only NVIDIA CUDA and one CTA per program are currently supported.

``compile_private_atomic`` uses only private-output and disjoint-allocation
contracts. It preserves atomic updates and relaxes ordering after the compiler
proves that weakening internal ordering preserves observable results. Inputs need
not be unique or immutable between invocations, and the output need not start at
zero. Initialized output,
in-allocation accesses and absence of external observers remain caller promises.
"""

import copy
import ctypes
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
    if type(value) in (bool, int, float, str):
        return (type(value).__name__, repr(value))
    raise TypeError("native contract arguments must be tensors, concrete bool/int/float scalars, or string constexprs")


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
    _check_private_storage(args, output_arg)


def _check_private_storage(args, output_arg):
    output = args[output_arg]
    if not isinstance(output, torch.Tensor):
        raise TypeError("the private output argument must be a tensor")
    for value in args:
        if not isinstance(value, torch.Tensor):
            continue
        if value.device.type != "cuda" or torch.version.hip is not None or value.device != output.device:
            raise ValueError("all tensor arguments must be on the same NVIDIA CUDA device")
        if value.layout != torch.strided or not value.is_contiguous() or value.is_neg() or value.is_conj():
            raise ValueError("native contract tensors must be contiguous with resolved lazy views")
        if value.numel() > _I32_MAX:
            raise ValueError("tensor size exceeds the supported int32 footprint")
    if torch.cuda.current_device() != output.device.index:
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


class NativePrivateAtomic:
    """Guarded native ordering specialization, with atomicity preserved.

    Launches permit changed input contents and replacement tensors with the same
    metadata. They recheck output/input allocation disjointness. Exposed compiled
    kernels allow controlled direct launches that bypass these guards; callers
    must then enforce the documented contracts themselves.
    """

    def __init__(self, baseline, optimized, args, grid, output_arg, attrs):
        self.baseline = baseline
        self.optimized = optimized
        self.grid = grid
        self.output_arg = output_arg
        self.attrs = attrs
        self.signatures = tuple(_signature(value) for value in args)

    def __call__(self, *args, grid=None):
        actual_grid = self.grid if grid is None else _grid_tuple(grid)
        if actual_grid != self.grid:
            raise ValueError("launch grid differs from the bound private-output invocation")
        if tuple(_signature(value) for value in args) != self.signatures:
            raise ValueError("argument metadata or scalar specialization changed")
        _check_private_storage(args, self.output_arg)
        return self.optimized[self.grid](*args)

    run = __call__

    def report(self):
        variants = {}
        for name, kernel in (("baseline", self.baseline), ("optimized", self.optimized)):
            kernel._init_handles()
            ptx = kernel.asm.get("ptx", "")
            variants[name] = {
                "atomic_instruction_sites": len(re.findall(r"\b(?:atom|red)\.", ptx)),
                "atomic_instructions": sorted(set(re.findall(r"\b(?:atom|red)\.[\w.]+", ptx))),
                "bar_sync_sites": len(re.findall(r"\bbar\.sync\b", ptx)),
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
                "shared_bytes": kernel.metadata.shared,
            }
        ir = self.optimized.asm.get("ttir", "") + self.optimized.asm.get("ttgir", "")
        return {
            "transformed": "tt.experimental_private_atomic_relaxed" in ir,
            "grid": self.grid,
            "native_attributes": {str(key): value
                                  for key, value in self.attrs.items()},
            "proof": "private output plus closed memory effects and disjoint read footprints; atomicity retained",
            "scope": "unchanged source; no uniqueness or immutable-index assumptions; no host launch removal",
            "variants": variants,
        }


def compile_private_atomic(kernel, args, *, grid, output_arg, options=None):
    """Compile an unchanged kernel with a private initialized output allocation.

    The caller promises initialized output ready at invocation entry and that
    no external code accesses it until kernel completion. Observable effects of
    weakening internal ordering are ruled out by the compiler's analysis,
    not by a direct caller assertion that the atomic may be relaxed.
    Every argument access must stay within its backing allocation. The wrapper
    checks disjoint output/input backing allocations; the compiler examines the
    actual kernel effects before weakening ordering. Missing proofs leave the
    atomic unchanged. Duplicate indices are allowed and atomics are retained.

    Pass every argument, including constexprs, in ``args``. ``output_arg`` is a
    position or parameter name. All scalar values and tensor metadata are bound
    to the result, but input objects and contents may change between launches.
    This API disables indexed-update removal in both variants to isolate the
    weaker ordering transformation. Initialization is never added implicitly.
    """
    if not isinstance(args, tuple) or len(args) != len(kernel.arg_names):
        raise ValueError("args must provide every kernel argument positionally, including constexprs")
    if isinstance(output_arg, str):
        output_arg = kernel.arg_names.index(output_arg)
    if type(output_arg) is not int or not 0 <= output_arg < len(args):
        raise ValueError("invalid private output argument position")
    grid = _grid_tuple(grid)
    options = dict(options or {})
    if options.get("num_ctas", 1) != 1:
        raise ValueError("native private-atomic relaxation currently requires num_ctas=1")
    if options.get("enable_indexed_update_specialization", False):
        raise ValueError("private-atomic relaxation must be isolated from indexed-update removal")
    _check_private_storage(args, output_arg)
    for value in args:
        _signature(value)
    baseline_options = dict(options, enable_indexed_update_specialization=False, enable_private_atomic_relaxation=False)
    baseline = kernel.warmup(*args, grid=grid, **baseline_options)
    attrs = copy.deepcopy(baseline.src.attrs)
    for number, value in enumerate(args):
        if isinstance(value, torch.Tensor):
            attrs[(number, )] = [(name, attr)
                                 for name, attr in attrs.get((number, ), [])
                                 if not name.startswith("tt.experimental_")]
            attrs[(number, )].append(("tt.experimental_alias_group", 1 if number == output_arg else 0))
    attrs[(output_arg, )].extend([("tt.experimental_private", 1), ("tt.experimental_num_ctas", 1)])
    source = type(baseline.src)(baseline.src.fn, baseline.src.signature, constexprs=baseline.src.constants, attrs=attrs)
    optimized = triton.compile(source, target=baseline.metadata.target,
                               options=dict(baseline_options, enable_private_atomic_relaxation=True))
    return NativePrivateAtomic(baseline, optimized, args, grid, output_arg, attrs)


def _check_immutable_storage(args, immutable_args):
    first_tensor = args[immutable_args[0]]
    if not isinstance(first_tensor, torch.Tensor):
        raise TypeError("immutable arguments must be tensors")
    for value in args:
        if not isinstance(value, torch.Tensor):
            continue
        if value.device.type != "cuda" or torch.version.hip is not None or value.device != first_tensor.device:
            raise ValueError("all tensor arguments must be on the same NVIDIA CUDA device")
        if value.layout != torch.strided or not value.is_contiguous() or value.is_neg() or value.is_conj():
            raise ValueError("native contract tensors must be contiguous with resolved lazy views")
        if value.numel() > _I32_MAX:
            raise ValueError("tensor size exceeds the supported int32 footprint")
    if torch.cuda.current_device() != first_tensor.device.index:
        raise ValueError("make the tensor's CUDA device current before compiling or launching")
    for number in immutable_args:
        value = args[number]
        if not isinstance(value, torch.Tensor):
            raise TypeError("immutable arguments must be tensors")
        first, last = _storage_span(value)
        if first == last:
            raise ValueError("immutable arguments must have nonempty backing storage")
        for other_number, other in enumerate(args):
            if other_number == number or not isinstance(other, torch.Tensor):
                continue
            other_first, other_last = _storage_span(other)
            if first < other_last and other_first < last:
                raise ValueError("immutable input backing storage overlaps another tensor argument")


def _cuda_kernel_residency(kernel, device):
    """Query the loaded function with its actual launch threads/shared memory."""
    kernel._init_handles()
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    cuda.cuDeviceGetAttribute.restype = ctypes.c_int
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    ]
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor.restype = ctypes.c_int
    sms = ctypes.c_int()
    status = cuda.cuDeviceGetAttribute(ctypes.byref(sms), 16, device)
    if status:
        raise RuntimeError(f"cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed with CUDA status {status}")
    blocks = ctypes.c_int()
    threads = kernel.metadata.num_warps * 32
    status = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(blocks), ctypes.c_void_p(kernel.function),
                                                              threads, kernel.metadata.shared)
    if status:
        raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor failed with CUDA status {status}")
    if blocks.value <= 0 or sms.value <= 0:
        raise RuntimeError("CUDA reported no resident capacity for the compiled kernel")
    return {
        "blocks_per_sm": blocks.value, "sm_count": sms.value, "threads_per_block": threads, "registers": kernel.n_regs,
        "spills": kernel.n_spills, "shared_bytes": kernel.metadata.shared
    }


class NativeImmutableLICM:
    """Immutable-load variants with an experimental resource profitability guard.

    The guard is a heuristic, not a guarantee against regressions. It can reject
    profitable hoists. Both compiled kernels are exposed for controlled analysis;
    launching them directly bypasses this class's binding and alias checks.
    """

    def __init__(self, baseline, optimized, args, grid, immutable_args, attrs):
        self.baseline, self.optimized = baseline, optimized
        self.grid, self.immutable_args, self.attrs = grid, immutable_args, attrs
        self.signatures = tuple(_signature(value) for value in args)
        self.device = args[immutable_args[0]].device.index
        self.resources = {
            name: _cuda_kernel_residency(compiled, self.device)
            for name, compiled in (("baseline", baseline), ("optimized", optimized))
        }
        base, opt = self.resources["baseline"], self.resources["optimized"]
        self.launched_ctas = grid[0] * grid[1] * grid[2]
        self.no_residency_loss = opt["blocks_per_sm"] >= base["blocks_per_sm"]
        self.grid_limited = self.launched_ctas <= opt["sm_count"] * opt["blocks_per_sm"]
        self.accepted = opt["spills"] == 0 and (self.no_residency_loss or self.grid_limited)
        self.selected = optimized if self.accepted else baseline

    def __call__(self, *args, grid=None):
        actual_grid = self.grid if grid is None else _grid_tuple(grid)
        if actual_grid != self.grid:
            raise ValueError("launch grid differs from the immutable LICM profitability decision")
        if tuple(_signature(value) for value in args) != self.signatures:
            raise ValueError("argument metadata or scalar specialization changed")
        _check_immutable_storage(args, self.immutable_args)
        return self.selected[self.grid](*args)

    run = __call__

    def report(self):
        return {
            "selected": "optimized" if self.accepted else "baseline",
            "binary_changed": self.baseline.asm["cubin"] != self.optimized.asm["cubin"],
            "grid": self.grid,
            "launched_ctas": self.launched_ctas,
            "immutable_args": self.immutable_args,
            "native_attributes": {str(key): value
                                  for key, value in self.attrs.items()},
            "variants": copy.deepcopy(self.resources),
            "guard": {
                "no_residency_loss": self.no_residency_loss, "grid_limited": self.grid_limited, "definition":
                "no spills and (no blocks/SM loss or total launched CTAs fit optimized residency)", "limitation":
                "profitability heuristic; can miss gains and cannot guarantee no regressions"
            },
            "contract":
            "initialized immutable inputs remain unchanged through invocation completion, including aliases",
            "scope": "same source and launch; both atomic specialization passes disabled; no timing calibration",
        }


def compile_immutable_licm(kernel, args, *, grid, immutable_args, options=None):
    """Compile immutable-input LICM and select a resource-guarded variant.

    Pass every argument, including constexprs, positionally in ``args``.
    ``immutable_args`` is a nonempty tuple of parameter names or positions. Each
    named tensor must contain initialized values; neither this kernel nor any
    concurrent work may write its backing allocation until invocation completion.
    Every argument access must stay within its allocation. Full backing storage
    of each immutable argument must be disjoint from every other tensor argument.
    This intentionally conservative check rejects shared read-only allocations.

    Tensor contents and tensor objects may change between completed invocations.
    Exact tensor metadata, scalar values, current CUDA device and launch grid are
    bound to the plan and rechecked on every call. No payload or version-counter
    check establishes immutability: it is a caller semantic contract.

    The compiler proves whether loads can move. After compiling both variants,
    a CUDA occupancy query selects the optimized version only when it has no
    spills and either loses no resident blocks/SM or all launched CTAs fit its
    resident capacity. This heuristic can decline gains and cannot guarantee no
    regressions. It is separate from the FX backend's timing calibration.
    Only NVIDIA CUDA with num_ctas=1 is supported.
    """
    if not isinstance(args, tuple) or len(args) != len(kernel.arg_names):
        raise ValueError("args must provide every kernel argument positionally, including constexprs")
    if not isinstance(immutable_args, tuple) or not immutable_args:
        raise ValueError("immutable_args must be a nonempty tuple of argument names or positions")
    positions = []
    for number in immutable_args:
        if isinstance(number, str):
            number = kernel.arg_names.index(number)
        if type(number) is not int or not 0 <= number < len(args):
            raise ValueError("invalid immutable argument position")
        if number in positions:
            raise ValueError("immutable arguments must be distinct")
        positions.append(number)
    immutable_args = tuple(positions)
    grid = _grid_tuple(grid)
    options = dict(options or {})
    if options.get("num_ctas", 1) != 1:
        raise ValueError("native immutable LICM currently requires num_ctas=1")
    options.update(enable_indexed_update_specialization=False, enable_private_atomic_relaxation=False)
    _check_immutable_storage(args, immutable_args)
    for value in args:
        _signature(value)
    template = kernel.warmup(*args, grid=grid, **options)
    attrs = {
        key: [(name, value)
              for name, value in values
              if not name.startswith("tt.experimental_")]
        for key, values in template.src.attrs.items()
    }
    source = type(template.src)(template.src.fn, template.src.signature, constexprs=template.src.constants, attrs=attrs)
    baseline = triton.compile(source, target=template.metadata.target, options=options)
    attrs = copy.deepcopy(attrs)
    for number in immutable_args:
        attrs.setdefault((number, ), []).append(("tt.experimental_immutable", 1))
    source = type(template.src)(template.src.fn, template.src.signature, constexprs=template.src.constants, attrs=attrs)
    optimized = triton.compile(source, target=template.metadata.target, options=options)
    return NativeImmutableLICM(baseline, optimized, args, grid, immutable_args, attrs)
