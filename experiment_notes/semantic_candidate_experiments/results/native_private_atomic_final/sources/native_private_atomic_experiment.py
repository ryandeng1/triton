#!/usr/bin/env python3
"""Native relaxation of private reductions, including repeated destination indices.

Original FlagGems source, native ordering relaxation, and a manual relaxed control
share buffers and include scratch reset, original frequency counting and BF16
conversion. Producers, allocation and Python guards are excluded from timing.
"""

import argparse
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

import torch
import triton
from triton._C import libtriton
from triton.experimental.semantic_contracts.native import compile_private_atomic

import irregular_kernels as upstream
from coverage_experiment import digest, paired_device_times, save_json
from native_atomic_ordering_experiment import (
    _embedding_dense_backward_kernel_relaxed,
    _embedding_dense_backward_kernel_scale_by_freq_relaxed,
    ordering_source_audit,
)
from native_indexed_update_experiment import kernel_evidence, same_values


def make_indices(n, vocabulary, kind):
    if kind == "unique":
        return torch.randperm(vocabulary, device="cuda")[:n].contiguous()
    assert n % 4 == 0
    selected = torch.randperm(vocabulary, device="cuda")[:n // 4]
    indices = selected.repeat(4)
    return indices[torch.randperm(n, device="cuda")].contiguous()


def reference(grad, indices, vocabulary, frequency, padding):
    valid = indices != padding
    contributions = grad[valid].double()
    multiplicities = torch.bincount(indices, minlength=vocabulary)
    if frequency:
        contributions = contributions / multiplicities[indices[valid], None]
    expected = torch.zeros((vocabulary, grad.shape[1]), dtype=torch.float64, device="cuda")
    expected.index_add_(0, indices[valid], contributions)
    absolute_sum = torch.zeros_like(expected)
    absolute_sum.index_add_(0, indices[valid], contributions.abs())
    # Each tested index occurs either once or exactly four times, so division
    # is exact and BF16 contributions have no rounding before accumulation.
    # The conservative gamma_m bound allows every permitted atomic serialization,
    # including cancellation. Include the much smaller FP64 reference/absolute-
    # sum rounding bounds as well; this is not a selected allclose tolerance.
    m = int(multiplicities.max().item())
    assert m in (1, 4)
    u32, u64 = 2.0**-24, 2.0**-53
    gamma32, gamma64 = m * u32 / (1.0 - m * u32), m * u64 / (1.0 - m * u64)
    bound = ((gamma32 + gamma64) / (1.0 - gamma64)) * absolute_sum
    return expected, bound, m


def check_normal_output(scratch, result, expected, bound):
    error = (scratch.double() - expected).abs()
    assert torch.all(torch.isfinite(scratch))
    assert torch.all(error <= bound), (float(error.max()), float(bound.max()))
    # An allowed FP32 accumulation order can straddle a BF16 rounding boundary.
    # Validate conversion against the actual FP32 result, rather than requiring
    # every valid serialization to round to the same BF16 value.
    same_values(result, scratch.to(result.dtype))
    ratio = torch.where(bound > 0, error / bound, 0.0)
    return float(error.max()), float(ratio.max())


def normalize_ttir_body(ir):
    lines = []
    for line in ir.splitlines():
        stripped = line.strip()
        if "tt.func " in line or stripped.startswith("#loc"):
            continue
        line = re.sub(r"\s+loc\([^\n]*\)", "", line)
        lines.append(line.replace("tt.atomic_rmw fadd, acq_rel,", "tt.atomic_rmw fadd, relaxed,"))
    return "\n".join(lines)


def measurement(shape, frequency, block, seed, kind, output, settings):
    n, d, vocabulary = shape
    torch.manual_seed(seed)
    indices = make_indices(n, vocabulary, kind)
    grad = torch.randint(-4, 5, (n, d), device="cuda").to(torch.bfloat16)
    scratch = torch.empty((vocabulary, d), dtype=torch.float32, device="cuda")
    result = torch.empty((vocabulary, d), dtype=grad.dtype, device="cuda")
    counts = torch.empty(vocabulary, dtype=torch.int32, device="cuda")
    padding = 0
    grid = (n, triton.cdiv(d, block), 1)
    kernel = upstream._embedding_dense_backward_kernel_scale_by_freq if frequency else upstream._embedding_dense_backward_kernel
    manual = _embedding_dense_backward_kernel_scale_by_freq_relaxed if frequency else _embedding_dense_backward_kernel_relaxed
    arguments = ((grad, indices, counts, scratch, vocabulary, padding, block, d) if frequency else
                 (grad, indices, scratch, vocabulary, padding, block, d))
    plan = compile_private_atomic(kernel, arguments, grid=grid, output_arg="grad_weight_ptr", options={"num_warps": 4})
    compiled = {
        "original_acq_rel":
        plan.baseline,
        "native_relaxed":
        plan.optimized,
        "manual_relaxed":
        manual.warmup(*arguments, grid=grid, num_warps=4, enable_indexed_update_specialization=False,
                      enable_private_atomic_relaxation=False),
    }

    def run(name):
        scratch.zero_()
        if frequency:
            counts.zero_()
            upstream._embedding_dense_backward_count_kernel[(triton.cdiv(n, 512), )](indices, counts, n, vocabulary,
                                                                                     padding, BLOCK_N=512)
        compiled[name][grid](*arguments)
        result.copy_(scratch)
        return result

    expected, _, multiplicity = reference(grad, indices, vocabulary, frequency, padding)
    for name in compiled:
        result.fill_(float("nan"))
        run(name)
        same_values(scratch, expected.float())
        same_values(result, expected.to(result.dtype))

    # Time ordinary gradients separately from the exact integer correctness
    # domain. Their tiny FP32 accumulation-order differences use a derived bound.
    grad.copy_(torch.randn((n, d), dtype=grad.dtype, device="cuda"))
    assert torch.all((grad == 0) | (grad.abs().float() >= 2.0**-100))
    assert torch.all(grad.abs().float() < 2.0**100)
    expected, bound, _ = reference(grad, indices, vocabulary, frequency, padding)
    observed = {}
    for name in compiled:
        run(name)
        observed[name] = check_normal_output(scratch, result, expected, bound)
    functions = {name: lambda name=name: run(name) for name in compiled}
    times = paired_device_times(functions, seed=seed, **settings)

    grad.mul_(0.5)
    indices.copy_(indices.flip(0))
    expected, bound, _ = reference(grad, indices, vocabulary, frequency, padding)
    for name in compiled:
        run(name)
        check_normal_output(scratch, result, expected, bound)

    retain = kind == "unique" and seed == 0 and n == 4096 and d == 128 and block == 128
    label = f"n{n}-d{d}-v{vocabulary}-freq{int(frequency)}"
    evidence = {name: kernel_evidence(item, output, label + "-" + name, retain) for name, item in compiled.items()}
    for name, item in compiled.items():
        ptx = item.asm["ptx"]
        evidence[name]["atomic_instructions"] = sorted(set(re.findall(r"\b(?:atom|red)\.[\w.]+", ptx)))
        evidence[name]["bar_sync_sites"] = len(re.findall(r"\bbar\.sync\b", ptx))
        assert evidence[name]["atomic_sites"] > 0
    assert any("acq_rel" in op for op in evidence["original_acq_rel"]["atomic_instructions"])
    for name in ("native_relaxed", "manual_relaxed"):
        assert all("relaxed" in op and op.startswith("red.") for op in evidence[name]["atomic_instructions"])
        assert evidence[name]["bar_sync_sites"] == 0
    assert plan.baseline.src.fn is plan.optimized.src.fn
    assert normalize_ttir_body(plan.baseline.asm["ttir"]) == normalize_ttir_body(plan.optimized.asm["ttir"])
    report = plan.report()
    assert report["transformed"]
    assert not any(word in str(report["native_attributes"])
                   for word in ("unique_indices", "immutable", "index_bound", "initial_zero", "experimental_grid"))
    return {
        "shape_n_d_v": shape,
        "frequency": frequency,
        "block_d": block,
        "seed": seed,
        "index_kind": kind,
        "nonpadding_index_multiplicity": multiplicity,
        "device": times,
        "code": evidence,
        "native_report": report,
        "native_ir_change": "Only atomic ordering changes in normalized TTIR body; source function object identical",
        "validation": {
            "exact":
            "Small integer BF16 gradients; exact FP32 scratch/BF16 output, including duplicate indices and power-of-two frequency division",
            "ordinary":
            "Random ordinary BF16 gradients; gamma_m FP32 accumulation bound plus FP64 reference/sum_abs rounding bound; actual FP32-to-BF16 conversion checked exactly",
            "normal_max_error_and_bound_fraction":
            observed,
            "mutation":
            "Poisoned output and changed gradient/index contents checked; atomicity retained for multiplicity four",
        },
        "scope": "Scratch zero + optional unchanged frequency-count reset/kernel + update + BF16 conversion",
    }


def aggregate(cases):
    groups = {}
    for case in cases:
        groups.setdefault((tuple(case["shape_n_d_v"]), case["frequency"], case["index_kind"]), []).append(case)
    result = []
    for (shape, frequency, kind), rows in groups.items():
        configs = {}
        for block in sorted({row["block_d"] for row in rows}):
            selected = [row for row in rows if row["block_d"] == block]
            configs[str(block)] = {
                name: statistics.median(row["device"]["median_us"][name]
                                        for row in selected)
                for name in selected[0]["device"]["median_us"]
            }
        best = {
            name: min((values[name], int(block))
                      for block, values in configs.items())
            for name in next(iter(configs.values()))
        }
        result.append({
            "shape_n_d_v": shape,
            "frequency": frequency,
            "index_kind": kind,
            "config_median_us": configs,
            "best": {name: {"median_us": value[0], "block_d": value[1]}
                     for name, value in best.items()},
            "original_over_native": best["original_acq_rel"][0] / best["native_relaxed"][0],
            "manual_over_native": best["manual_relaxed"][0] / best["native_relaxed"][0],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "confirm"), default="confirm")
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    smoke = options.mode == "smoke"
    settings = {"rounds": 3 if smoke else 15, "repetitions": 8 if smoke else 32, "warmup_ms": 20 if smoke else 200}
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    sources = [
        Path(__file__),
        directory / "native_atomic_ordering_experiment.py",
        directory / "native_indexed_update_experiment.py",
        directory / "irregular_kernels.py",
        directory / "coverage_experiment.py",
        root / "python/triton/experimental/semantic_contracts/native.py",
        root / "lib/Dialect/Triton/Transforms/SpecializeIndexedUpdates.cpp",
        root / "include/triton/Dialect/Triton/Transforms/Passes.td",
        root / "python/src/passes.cc",
        root / "third_party/nvidia/backend/compiler.py",
        root / "python/test/unit/runtime/test_specialize.py",
        root / "test/Triton/combine.mlir",
    ]
    report = {
        "status":
        "running",
        "mode":
        options.mode,
        "gpu":
        torch.cuda.get_device_name(),
        "torch":
        torch.__version__,
        "triton":
        triton.__version__,
        "compiler_sha256":
        digest(libtriton.__file__),
        "git_revision":
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path.resolve()): digest(path)
                          for path in sources},
        "source_audit":
        ordering_source_audit(),
        "settings":
        settings,
        "cases": [],
        "scope":
        "Native private-atomic ordering relaxation of unchanged FlagGems source; no uniqueness required",
        "limits":
        "NVIDIA B300 device operation times; producers, allocation and host guards excluded. Duplicate inputs use shuffled uniform fourfold repetition, not arbitrary training distributions. Frequency count kernel remains unchanged.",
    }
    try:
        shapes = [(32, 137, 67)] if smoke else [(4096, 128, 8192), (16384, 256, 32768), (8192, 1024, 16384)]
        for shape in shapes:
            for frequency in (False, True):
                for kind in ("unique", "fourfold"):
                    for block in ((128, ) if smoke else (128, 256)):
                        for seed in range(1 if smoke else 3):
                            case = measurement(shape, frequency, block, seed, kind, options.output, settings)
                            report["cases"].append(case)
                            print(
                                json.dumps({
                                    "shape": shape, "frequency": frequency, "kind": kind, "block": block, "seed": seed,
                                    "device_us": case["device"]["median_us"]
                                }), flush=True)
                            save_json(options.output / "results.json", report)
        report["summary"] = aggregate(report["cases"])
        report["geometric_mean_ratios"] = {
            kind: {
                name:
                math.exp(statistics.mean(math.log(row[name])
                                         for row in report["summary"]
                                         if row["index_kind"] == kind))
                for name in ("original_over_native", "manual_over_native")
            }
            for kind in ("unique", "fourfold")
        }
        report["status"] = "passed"
    finally:
        save_json(options.output / "results.json", report)


if __name__ == "__main__":
    main()
