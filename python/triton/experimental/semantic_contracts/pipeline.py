"""Opt-in, measured profitability for contract-licensed CUDA loop pipelines.

Memory contracts establish legality in the native compiler. This module chooses
among those legal compilations; timing never supplies a dependence proof.
"""

import copy
import hashlib
import math
import random
import statistics
import time

import torch
import triton
from triton.runtime.errors import OutOfResources

from .native import _grid_tuple, _signature, _storage_span, _I32_MAX


def _positions(kernel, values, count):
    result = []
    for value in values:
        if isinstance(value, str):
            if value not in kernel.arg_names:
                raise ValueError(f"unknown contracted argument: {value}")
            value = kernel.arg_names.index(value)
        if type(value) is not int or not 0 <= value < count:
            raise ValueError("invalid contracted argument position")
        if value in result:
            raise ValueError("contracted arguments must be distinct")
        result.append(value)
    return tuple(result)


def _check_storage(args, groups, immutable, mutable, device):
    tensors = {i: value for i, value in enumerate(args) if isinstance(value, torch.Tensor)}
    if set(groups) != set(tensors):
        raise ValueError("alias_groups must describe every tensor argument and only tensor arguments")
    if not set(immutable).issubset(tensors) or not set(mutable).issubset(tensors):
        raise ValueError("immutable_args and mutable_args must name tensor arguments")
    if torch.cuda.current_device() != device:
        raise ValueError("make the bound CUDA device current before compiling or launching")
    spans = {}
    for i, value in tensors.items():
        if value.device.type != "cuda" or value.device.index != device or torch.version.hip is not None:
            raise ValueError("all tensor arguments must be on the same NVIDIA CUDA device")
        if value.layout != torch.strided or value.is_neg() or value.is_conj():
            raise ValueError("pipeline contracts require strided tensors with resolved lazy views")
        if value.numel() > _I32_MAX:
            raise ValueError("tensor size exceeds the supported int32 footprint")
        spans[i] = _storage_span(value)
    for i, (lo, hi) in spans.items():
        for j, (other_lo, other_hi) in spans.items():
            if i >= j or lo >= other_hi or other_lo >= hi:
                continue
            if groups[i] != groups[j]:
                raise ValueError("different alias groups have overlapping backing storage")
            if (i in immutable and j in mutable) or (j in immutable and i in mutable):
                raise ValueError("immutable and mutable arguments have overlapping backing storage")
    if set(immutable) & set(mutable):
        raise ValueError("an argument cannot be both immutable and mutable")


def _snapshots(args, mutable):
    """Preserve entire backing allocations, including aliases and padding."""
    seen, snapshots = set(), []
    for number in mutable:
        value = args[number]
        span = _storage_span(value)
        if span in seen:
            continue
        seen.add(span)
        storage = value.untyped_storage()
        raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(), ), (1, ))
        snapshots.append((raw, raw.clone()))
    return snapshots


def _measure(functions, repetitions, rounds, seed):
    """Interleave warmed graphs; returned samples are microseconds/invocation."""
    graphs = {}
    for name, function in functions.items():
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repetitions):
                function()
        graphs[name] = graph
    for _ in range(3):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    rng = random.Random(seed)
    for _ in range(rounds):
        names = list(graphs)
        rng.shuffle(names)
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / repetitions)
    return samples


def _medians(samples):
    result = {}
    for name, values in samples.items():
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            raise RuntimeError("pipeline calibration produced invalid timing samples")
        result[name] = statistics.median(values)
    return result


class NativePipeline:
    """A cached depth/LICM decision for one exact specialization and launch.

    Calls perform binding and allocation checks, then launch exactly once. They
    do not tune, snapshot, reset, or initialize outputs. Replacement tensors and
    changed contents are allowed if the same contracts and metadata hold.
    Exposed kernels bypass guards when launched directly.
    """

    def __init__(self, candidates, selected_name, args, grid, groups, immutable, mutable, details):
        self.candidates = candidates
        self.baseline = candidates["baseline"]
        self.selected_name = selected_name
        self.selected = candidates[selected_name]
        self.grid = grid
        self.alias_groups = groups
        self.immutable_args, self.mutable_args = immutable, mutable
        self.signatures = tuple(_signature(value) for value in args)
        self.device = next(value.device.index for value in args if isinstance(value, torch.Tensor))
        self._details = details

    def __call__(self, *args, grid=None):
        if grid is not None and _grid_tuple(grid) != self.grid:
            raise ValueError("launch grid differs from the cached pipeline decision")
        if tuple(_signature(value) for value in args) != self.signatures:
            raise ValueError("argument metadata or scalar specialization changed")
        _check_storage(args, self.alias_groups, self.immutable_args, self.mutable_args, self.device)
        return self.selected[self.grid](*args)

    run = __call__

    def report(self):
        return copy.deepcopy(self._details)


def compile_pipeline(kernel, args, *, grid, alias_groups, mutable_args, immutable_args=(), options=None,
                     depths=(0, 2, 3, 4), tune_licm=False, replay_safe=False, min_gain=0.02, rounds=9, repetitions=32):
    """Compile, measure and cache a contract-licensed pipeline selection.

    Pass every argument, including constexprs, positionally. ``alias_groups``
    maps every tensor argument (name or position) to a nonnegative group number.
    Different groups promise disjoint *entire derived access footprints*, and
    are checked for full backing-allocation overlap on compilation and every
    launch. All accesses must stay within those allocations. ``immutable_args``
    additionally promises initialized contents unchanged through each invocation,
    including concurrent aliases; this is not inferred from observed values.

    ``mutable_args`` names every allocation this kernel may write, including
    state and output scratch. There must be no writes to other memory or external
    effects. Calibration requires exclusive use of these allocations, snapshots
    their full backing storage and restores it before returning. By default each
    calibration invocation also restores that state; reset copies are included
    in all candidate timings. ``replay_safe=True`` instead promises that repeated
    invocations are independent (e.g. complete overwrite of outputs, no state
    accumulation), allowing kernel-only timings. It does not disable the final
    restoration. Normal cached calls never restore state.

    This is a fixed-launch, in-process tuning plan, not a persistent global cache.
    Scalar values, tensor metadata/alignment, current device and grid are bound.
    Calibrate before CUDA graph capture; cached calls support capture. Only
    NVIDIA CUDA Triton kernels with one CTA per program are supported. Cold
    calibration synchronizes the device and can be expensive. Timing decisions
    depend on the calibration inputs/cache/clock regime and may become stale.

    Depth0 retains ordinary compilation with the *same* contracts. Explicit
    source stage/schedule annotations are honored in every candidate; only
    inferred pipelines are tuned. ``tune_licm=True`` independently adds LICM
    candidates including a depth0 LICM control. Unrelated experimental contract
    transforms are disabled. Identical binaries are timed once. A transformed
    candidate must beat baseline by ``min_gain`` in both a shuffled screening
    phase and a separate confirmation phase with the winner frozen. Resource
    exhaustion discards a candidate; other compilation/execution errors propagate.
    """
    started = time.monotonic()
    if not isinstance(args, tuple) or len(args) != len(kernel.arg_names):
        raise ValueError("args must provide every kernel argument positionally, including constexprs")
    if getattr(kernel, "is_gluon", lambda: False)():
        raise ValueError("automatic contract pipeline tuning currently supports Triton, not Gluon")
    tensors = [value for value in args if isinstance(value, torch.Tensor)]
    if not tensors or tensors[0].device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("pipeline calibration requires NVIDIA CUDA tensors")
    device = tensors[0].device.index
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compile_pipeline must calibrate before CUDA graph capture")
    grid = _grid_tuple(grid)
    if not isinstance(alias_groups, dict):
        raise TypeError("alias_groups must be a mapping of argument positions or names to groups")
    positions = _positions(kernel, alias_groups, len(args))
    groups = dict(zip(positions, alias_groups.values()))
    if any(type(group) is not int or not 0 <= group <= _I32_MAX for group in groups.values()):
        raise ValueError("alias group identifiers must be nonnegative int32 values")
    immutable = _positions(kernel, immutable_args, len(args))
    mutable = _positions(kernel, mutable_args, len(args))
    _check_storage(args, groups, immutable, mutable, device)
    for value in args:
        _signature(value)
    if type(replay_safe) is not bool or type(tune_licm) is not bool:
        raise TypeError("replay_safe and tune_licm must be boolean")
    if (not isinstance(depths, (tuple, list)) or not depths
            or any(type(depth) is not int or depth not in (0, 2, 3, 4) for depth in depths)):
        raise ValueError("depths must contain only 0, 2, 3 or 4")
    depths = sorted(set(depths) | {0})
    if type(rounds) is not int or not 3 <= rounds <= 101:
        raise ValueError("rounds must be an integer from 3 through 101")
    if type(repetitions) is not int or not 1 <= repetitions <= 1024:
        raise ValueError("repetitions must be an integer from 1 through 1024")
    if isinstance(min_gain, bool) or not isinstance(min_gain, (int, float)) or not 0 <= min_gain < 1:
        raise ValueError("min_gain must be a finite fraction from 0 up to 1")
    options = dict(options or {})
    if options.get("num_ctas", 1) != 1:
        raise ValueError("pipeline calibration currently requires num_ctas=1")
    if options.get("contract_pipeline_stages", 0) != 0 or options.get("enable_contract_licm", False):
        raise ValueError("use depths and tune_licm to select pipeline and LICM candidates")
    options.update(enable_indexed_update_specialization=False, enable_private_atomic_relaxation=False,
                   enable_private_reduction_ownership=False, enable_contract_load_reuse=False,
                   enable_contract_licm=False, contract_pipeline_stages=0)
    template = kernel.warmup(*args, grid=grid, **options)
    if template.metadata.target.backend != "cuda":
        raise ValueError("automatic pipeline tuning requires the CUDA backend")
    attrs = {
        key: [(name, value)
              for name, value in values
              if not name.startswith("tt.experimental_")]
        for key, values in template.src.attrs.items()
    }
    for number, group in groups.items():
        attrs.setdefault((number, ), []).append(("tt.experimental_alias_group", group))
    for number in immutable:
        attrs.setdefault((number, ), []).append(("tt.experimental_immutable", 1))
    candidates, variants, binaries, unique = {}, {}, {}, {}
    for licm in ((False, True) if tune_licm else (False, )):
        for depth in depths:
            name = "baseline" if depth == 0 else f"stages{depth}"
            if licm:
                name = "licm" if depth == 0 else name + "_licm"
            source = type(template.src)(template.src.fn, template.src.signature, constexprs=template.src.constants,
                                        attrs=copy.deepcopy(attrs))
            try:
                compiled = triton.compile(
                    source, target=template.metadata.target, options=dict(options, contract_pipeline_stages=depth,
                                                                          enable_contract_licm=licm))
                compiled._init_handles()
            except OutOfResources as error:
                if name == "baseline":
                    raise
                variants[name] = dict(depth=depth, licm=licm, excluded="resource_exhaustion", detail=str(error))
                continue
            candidates[name] = compiled
            digest = hashlib.sha256(compiled.asm["cubin"]).hexdigest()
            representative = binaries.setdefault(digest, name)
            if representative == name:
                unique[name] = compiled
            variants[name] = dict(depth=depth, licm=licm, equivalent_to=representative, registers=compiled.n_regs,
                                  spills=compiled.n_spills, shared_bytes=compiled.metadata.shared, cubin_sha256=digest,
                                  async_copy_sites=compiled.asm["ttgir"].count("ttg.async_copy_global_to_local"))
    compile_seconds = time.monotonic() - started
    selected = "baseline"
    calibration = dict(screen_samples_us={}, confirmation_samples_us={}, screen_medians_us={},
                       confirmation_medians_us={}, min_gain=min_gain, rounds=rounds, repetitions=repetitions,
                       reset_copies_included=bool(mutable) and not replay_safe, state_restored=True, executed=False,
                       reason="identical_binaries")
    if len(unique) > 1:
        snapshots = _snapshots(args, mutable)

        def restore():
            for raw, saved in snapshots:
                raw.copy_(saved)

        functions = {}
        for name, compiled in unique.items():
            launch = compiled[grid]

            def invoke(launch=launch):
                if not replay_safe:
                    restore()
                launch(*args)

            functions[name] = invoke
        try:
            samples = _measure(functions, repetitions, rounds, seed=1729)
            medians = _medians(samples)
            if set(medians) != set(functions):
                raise RuntimeError("pipeline calibration did not measure every candidate")
            winner = min(medians, key=medians.get)
            calibration.update(executed=True, screen_samples_us=samples, screen_medians_us=medians,
                               screen_winner=winner, reason="baseline_won_screen")
            if winner != "baseline" and medians["baseline"] / medians[winner] >= 1 + min_gain:
                confirmation = _measure({name: functions[name]
                                         for name in ("baseline", winner)}, repetitions, rounds, seed=2718)
                confirmation_medians = _medians(confirmation)
                if set(confirmation_medians) != {"baseline", winner}:
                    raise RuntimeError("pipeline confirmation did not measure both candidates")
                calibration.update(confirmation_samples_us=confirmation, confirmation_medians_us=confirmation_medians,
                                   reason="confirmation_below_margin")
                if confirmation_medians["baseline"] / confirmation_medians[winner] >= 1 + min_gain:
                    selected = winner
                    calibration["reason"] = "confirmed_gain"
            elif winner != "baseline":
                calibration["reason"] = "screen_below_margin"
        finally:
            restore()
            torch.cuda.synchronize()
    details = dict(selected=selected, grid=grid, device=device, alias_groups=groups, immutable_args=immutable,
                   mutable_args=mutable, variants=variants, compile_seconds=compile_seconds,
                   calibration_seconds=time.monotonic() - started - compile_seconds,
                   total_seconds=time.monotonic() - started, calibration=calibration,
                   scope="inferred pipeline depth at fixed launch; explicit source schedules are preserved",
                   cached=True, baseline="same contracts, no inferred pipeline or contract LICM",
                   native_attributes={str(key): values
                                      for key, values in attrs.items()})
    return NativePipeline(candidates, selected, args, grid, groups, immutable, mutable, details)
