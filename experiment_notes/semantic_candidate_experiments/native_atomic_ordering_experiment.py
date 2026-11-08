#!/usr/bin/env python3
"""Isolate ordering/lowering from atomic elimination on unchanged embedding kernels.

The relaxed control changes only the atomic sem keyword. Native variants compile
original source. Every variant includes scratch zeroing, optional unchanged count
reset/kernel, update and BF16 conversion on common buffers. These are device
operation timings; allocation, producer generation and host guards are excluded.
"""

import argparse
import ast
import copy
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton._C import libtriton
from triton.experimental.semantic_contracts.native import compile_indexed_update

import irregular_kernels as upstream
from coverage_experiment import digest, paired_device_times, save_json
from native_indexed_update_experiment import kernel_evidence, same_values, source_audit


@triton.jit
def _embedding_dense_backward_kernel_relaxed(
    grad_output_ptr,
    indices_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    idx = tl.load(indices_ptr + pid_n)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0).to(tl.float32)

    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask, sem="relaxed")


@triton.jit
def _embedding_dense_backward_kernel_scale_by_freq_relaxed(
    grad_output_ptr,
    indices_ptr,
    counts_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    idx = tl.load(indices_ptr + pid_n).to(tl.int32)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    # go = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)
    go = tl.load(go_ptrs, mask=mask_d, other=0.0)

    # cnt = tl.load(counts_ptr + idx, mask=valid, other=1).to(tl.float32)
    cnt = tl.load(counts_ptr + idx, mask=valid, other=1)
    go = go / cnt

    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask, sem="relaxed")


def ordering_source_audit():
    directory = Path(__file__).parent
    original = ast.parse((directory / "irregular_kernels.py").read_text())
    changed = ast.parse(Path(__file__).read_text())
    result = {"upstream": source_audit(), "relaxed_controls": {}}
    for name in ("_embedding_dense_backward_kernel", "_embedding_dense_backward_kernel_scale_by_freq"):
        old = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
        new = copy.deepcopy(
            next(node for node in changed.body if isinstance(node, ast.FunctionDef) and node.name == name + "_relaxed"))
        new.name = name
        calls = [
            node for node in ast.walk(new) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == "atomic_add"
        ]
        assert len(calls) == 1
        sem = [item for item in calls[0].keywords if item.arg == "sem"]
        assert len(sem) == 1 and isinstance(sem[0].value, ast.Constant) and sem[0].value.value == "relaxed"
        calls[0].keywords.remove(sem[0])
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
        result["relaxed_controls"][name] = "Only sem='relaxed' added, apart from function name; AST audited"
    return result


def measurement(shape, frequency, block, seed, output, settings):
    n, d, vocabulary = shape
    torch.manual_seed(seed)
    indices = torch.randperm(vocabulary, device="cuda")[:n].contiguous()
    grad = torch.randn((n, d), dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty((vocabulary, d), dtype=torch.float32, device="cuda")
    result = torch.empty((vocabulary, d), dtype=grad.dtype, device="cuda")
    counts = torch.empty(vocabulary, dtype=torch.int32, device="cuda")
    padding = 0
    grid = (n, triton.cdiv(d, block), 1)
    kernel = upstream._embedding_dense_backward_kernel_scale_by_freq if frequency else upstream._embedding_dense_backward_kernel
    relaxed = _embedding_dense_backward_kernel_scale_by_freq_relaxed if frequency else _embedding_dense_backward_kernel_relaxed
    arguments = ((grad, indices, counts, scratch, vocabulary, padding, block, d) if frequency else
                 (grad, indices, scratch, vocabulary, padding, block, d))
    options = {"num_warps": 4}
    rmw = compile_indexed_update(kernel, arguments, grid=grid, index_arg="indices_ptr", output_arg="grad_weight_ptr",
                                 index_count=n, index_bound=vocabulary, initial_zero=False, options=options)
    zero = compile_indexed_update(kernel, arguments, grid=grid, index_arg="indices_ptr", output_arg="grad_weight_ptr",
                                  index_count=n, index_bound=vocabulary, initial_zero=True, options=options)
    compiled = {
        "original_acq_rel": rmw.baseline,
        "relaxed_atomic": relaxed.warmup(*arguments, grid=grid, enable_indexed_update_specialization=False, **options),
        "native_rmw": rmw.optimized,
        "native_zero": zero.optimized,
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

    functions = {name: lambda name=name: run(name) for name in compiled}
    expected = torch.zeros((vocabulary, d), device="cuda", dtype=torch.float32)
    valid = indices != padding
    expected[indices[valid]] = torch.where(grad[valid] == 0, 0.0, grad[valid]).float()
    for fn in functions.values():
        result.fill_(float("nan"))
        fn()
        same_values(scratch, expected)
        same_values(result, expected.to(grad.dtype))
    times = paired_device_times(functions, seed=seed, **settings)
    grad.mul_(0.5)
    expected.zero_()
    expected[indices[valid]] = torch.where(grad[valid] == 0, 0.0, grad[valid]).float()
    for fn in functions.values():
        fn()
        same_values(scratch, expected)
    retain = seed == 0 and n == 4096 and d == 128 and block == 128
    label = f"n{n}-d{d}-v{vocabulary}-freq{int(frequency)}"
    evidence = {name: kernel_evidence(item, output, label + "-" + name, retain) for name, item in compiled.items()}
    for name, item in compiled.items():
        evidence[name]["atomic_instructions"] = sorted(set(re.findall(r"\b(?:atom|red)\.[\w.]+", item.asm["ptx"])))
    assert evidence["original_acq_rel"]["atomic_sites"] > 0
    assert evidence["relaxed_atomic"]["atomic_sites"] > 0
    assert any("acq_rel" in instruction for instruction in evidence["original_acq_rel"]["atomic_instructions"])
    assert any("relaxed" in instruction for instruction in evidence["relaxed_atomic"]["atomic_instructions"])
    assert evidence["native_rmw"]["atomic_sites"] == evidence["native_zero"]["atomic_sites"] == 0
    assert rmw.baseline.src.fn is rmw.optimized.src.fn
    return {
        "shape_n_d_v": shape,
        "frequency": frequency,
        "block_d": block,
        "seed": seed,
        "device": times,
        "code": evidence,
        "validation": "Exact FP32 scratch and BF16 output; poisoned output; changed gradients",
        "scope": "Scratch zero + optional unchanged frequency-count reset/kernel + update + BF16 conversion",
    }


def aggregate(cases):
    grouped = {}
    for case in cases:
        grouped.setdefault((tuple(case["shape_n_d_v"]), case["frequency"]), []).append(case)
    result = []
    for (shape, frequency), rows in grouped.items():
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
        times = {name: value[0] for name, value in best.items()}
        result.append({
            "shape_n_d_v": shape,
            "frequency": frequency,
            "config_median_us": configs,
            "best": {name: {"median_us": value[0], "block_d": value[1]}
                     for name, value in best.items()},
            "original_over_relaxed": times["original_acq_rel"] / times["relaxed_atomic"],
            "relaxed_over_native_rmw": times["relaxed_atomic"] / times["native_rmw"],
            "relaxed_over_native_zero": times["relaxed_atomic"] / times["native_zero"],
            "original_over_native_rmw": times["original_acq_rel"] / times["native_rmw"],
            "original_over_native_zero": times["original_acq_rel"] / times["native_zero"],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "confirm"), default="confirm")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    smoke = args.mode == "smoke"
    settings = {"rounds": 3 if smoke else 15, "repetitions": 8 if smoke else 32, "warmup_ms": 20 if smoke else 200}
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    sources = [
        Path(__file__),
        directory / "native_indexed_update_experiment.py",
        directory / "irregular_kernels.py",
        directory / "coverage_experiment.py",
        root / "python/triton/experimental/semantic_contracts/native.py",
        root / "lib/Dialect/Triton/Transforms/SpecializeIndexedUpdates.cpp",
        root / "include/triton/Dialect/Triton/Transforms/Passes.td",
        root / "third_party/nvidia/backend/compiler.py",
    ]
    report = {
        "status":
        "running",
        "mode":
        args.mode,
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
        "Ordering/lowering control of original indexed-update result; no FX or Inductor",
        "limits":
        "Unique immutable randperm subsets, finite ordinary BF16 gradients; device timings exclude producers, allocations and host guards",
    }
    try:
        shapes = [(31, 137, 67)] if smoke else [(4096, 128, 8192), (16384, 256, 32768), (8192, 1024, 16384)]
        for shape in shapes:
            for frequency in (False, True):
                for block in ((128, ) if smoke else (128, 256)):
                    for seed in range(1 if smoke else 3):
                        case = measurement(shape, frequency, block, seed, args.output, settings)
                        report["cases"].append(case)
                        print(
                            json.dumps({
                                "shape": shape, "frequency": frequency, "block": block, "seed": seed, "device_us":
                                case["device"]["median_us"]
                            }), flush=True)
                        save_json(args.output / "results.json", report)
        report["summary"] = aggregate(report["cases"])
        ratio_names = [name for name in report["summary"][0] if "_over_" in name]
        report["geometric_mean_ratios"] = {
            name: math.exp(statistics.mean(math.log(row[name])
                                           for row in report["summary"]))
            for name in ratio_names
        }
        report["status"] = "passed"
    finally:
        save_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
