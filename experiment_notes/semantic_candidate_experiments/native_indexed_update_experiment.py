#!/usr/bin/env python3
"""Native contract specialization of unchanged FlagGems indexed-update kernels.

Run after rebuilding the native compiler. All variants retain output zeroing,
frequency counting when requested, and final BF16 conversion. Index generation
and allocations are excluded. Variants share physical buffers and device timing
uses randomized interleaved CUDA graphs. No FX frontend or Inductor is involved.
"""

import argparse
import ast
import hashlib
import inspect
import json
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


@triton.jit
def manual_update(Grad, Indices, Counts, Out, V, Padding, BLOCK_D: tl.constexpr, D: tl.constexpr,
                  FREQUENCY: tl.constexpr, INITIAL_ZERO: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    index = tl.load(Indices + row)
    if FREQUENCY:
        index = index.to(tl.int32)
    active = (cols < D) & (index >= 0) & (index < V) & (index != Padding)
    value = tl.load(Grad + row * D + cols, mask=cols < D, other=0)
    if FREQUENCY:
        count = tl.load(Counts + index, mask=(index >= 0) & (index < V) & (index != Padding), other=1)
        value = value / count
    value = value.to(tl.float32)
    ptr = Out + index * D + cols
    if INITIAL_ZERO:
        old = tl.full((BLOCK_D, ), 0.0, tl.float32)
    else:
        old = tl.load(ptr, mask=active, other=0.0)
    updated = tl.inline_asm_elementwise("add.rn.ftz.f32 $0, $1, $2;", constraints="=f,f,f", args=[old, value],
                                        dtype=tl.float32, is_pure=True, pack=1)
    tl.store(ptr, updated, mask=active)


def source_audit():
    directory = Path(__file__).parent
    pinned = ast.parse((directory / "vendor/irregular/flag_embedding.py").read_text())
    source = ast.parse((directory / "irregular_kernels.py").read_text())
    names = ("_embedding_dense_backward_kernel", "_embedding_dense_backward_kernel_scale_by_freq")
    result = {}
    for name in names:
        old = next(node for node in pinned.body if isinstance(node, ast.FunctionDef) and node.name == name)
        new = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == name)
        old_ast, new_ast = ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False)
        assert old_ast == new_ast, name
        result[name] = {
            "ast_identical_to_pinned_upstream": True, "ast_sha256": hashlib.sha256(old_ast.encode()).hexdigest()
        }
    return result


def same_values(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    ordinary = ~torch.isnan(expected)
    assert torch.equal(actual[ordinary].contiguous().view(torch.uint8),
                       expected[ordinary].contiguous().view(torch.uint8))


def kernel_evidence(kernel, output, label, retain):
    kernel._init_handles()
    ptx = kernel.asm["ptx"]
    result = {
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(), "atomic_sites":
        len(re.findall(r"\b(?:atom|red)\.", ptx)), "registers": kernel.n_regs, "spills": kernel.n_spills,
        "shared_bytes": kernel.metadata.shared
    }
    if retain:
        directory = output / "code"
        directory.mkdir(exist_ok=True)
        for kind in ("ttir", "ttgir", "ptx"):
            if kind in kernel.asm:
                path = directory / f"{label}.{kind}"
                path.write_text(kernel.asm[kind])
        result["retained_prefix"] = f"code/{label}"
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
    arguments = (grad, indices, counts, scratch, vocabulary, padding, block,
                 d) if frequency else (grad, indices, scratch, vocabulary, padding, block, d)
    options = {"num_warps": 4}
    rmw = compile_indexed_update(kernel, arguments, grid=grid, index_arg="indices_ptr", output_arg="grad_weight_ptr",
                                 index_count=n, index_bound=vocabulary, initial_zero=False, options=options)
    zero = compile_indexed_update(kernel, arguments, grid=grid, index_arg="indices_ptr", output_arg="grad_weight_ptr",
                                  index_count=n, index_bound=vocabulary, initial_zero=True, options=options)
    compiled = {"baseline": rmw.baseline, "native_rmw": rmw.optimized, "native_zero": zero.optimized}
    manual = {}

    def run(name):
        scratch.zero_()
        if frequency:
            counts.zero_()
            upstream._embedding_dense_backward_count_kernel[(triton.cdiv(n, 512), )](indices, counts, n, vocabulary,
                                                                                     padding, BLOCK_N=512)
        if name in compiled:
            compiled[name][grid](*arguments)
        else:
            manual[name] = manual_update[grid](grad, indices, counts, scratch, vocabulary, padding, block, d, frequency,
                                               name == "manual_zero", num_warps=4)
        result.copy_(scratch)
        return result

    functions = {
        name: lambda name=name: run(name)
        for name in ("baseline", "native_rmw", "native_zero", "manual_rmw", "manual_zero")
    }
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
    compiled.update(manual)
    retain = seed == 0 and n == 4096 and d == 128 and block == 128
    label = f"n{n}-d{d}-v{vocabulary}-freq{int(frequency)}"
    evidence = {name: kernel_evidence(item, output, label + "-" + name, retain) for name, item in compiled.items()}
    assert evidence["baseline"]["atomic_sites"] > 0
    for name in ("native_rmw", "native_zero", "manual_rmw", "manual_zero"):
        assert evidence[name]["atomic_sites"] == 0, (name, evidence[name])
    assert rmw.baseline.src.fn is rmw.optimized.src.fn
    return {
        "shape_n_d_v": shape, "frequency": frequency, "block_d": block, "seed": seed, "device": times,
        "native_rmw_report": rmw.report(), "native_zero_report": zero.report(), "code": evidence, "validation":
        "shared physical outputs; exact baseline/manual/reference outputs; poison and changed gradients", "scope":
        "zero scratch + optional original count kernel + update + BF16 conversion; allocations/producers excluded",
        "contract_source":
        "randperm producer supplies unique immutable index prefix; dedicated output storage; zero reset before each launch",
        "count_kernel_transformed": False, "input_domain":
        "producer-established unique randperm subset; ordinary duplicate-index workloads retain atomics"
    }


def aggregate(cases):
    groups = {}
    for case in cases:
        groups.setdefault((tuple(case["shape_n_d_v"]), case["frequency"]), []).append(case)
    result = []
    for (shape, frequency), rows in groups.items():
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
            "shape_n_d_v": shape, "frequency": frequency, "config_median_us": configs, "best":
            {name: {"median_us": value[0], "block_d": value[1]}
             for name, value in best.items()}, "native_rmw_speedup": best["baseline"][0] / best["native_rmw"][0],
            "native_zero_speedup": best["baseline"][0] / best["native_zero"][0]
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
    native_source = Path(inspect.getfile(compile_indexed_update))
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__),
        Path(__file__).with_name("irregular_kernels.py"),
        Path(__file__).with_name("coverage_experiment.py"), native_source,
        root / "lib/Dialect/Triton/Transforms/SpecializeIndexedUpdates.cpp",
        root / "include/triton/Dialect/Triton/Transforms/Passes.td", root / "python/src/passes.cc",
        root / "third_party/nvidia/backend/compiler.py", root / "python/test/unit/runtime/test_specialize.py"
    ]
    report = {
        "status": "running", "mode": args.mode, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "triton": triton.__version__, "compiler_sha256": digest(libtriton.__file__), "git_revision":
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "source_sha256":
        {str(path.resolve()): digest(path)
         for path in sources}, "source_audit": source_audit(), "settings": settings, "cases": [], "scope":
        "unchanged pinned FlagGems device kernels, native compiler transformation; no FX/Inductor"
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
        report["status"] = "passed"
    finally:
        save_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
