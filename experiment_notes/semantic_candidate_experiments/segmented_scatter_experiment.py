#!/usr/bin/env python3
"""Measure segmented-unique scatter updates in unchanged FlagGems source.

Indices are produced by row-wise topk/sort, so values can repeat across rows.
All consumer variants share physical buffers and include output initialization.
"""

import argparse
import ast
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton._C import libtriton

from coverage_experiment import digest, paired_device_times, save_json
from segmented_kernels import scatter_add_2d_kernel


@triton.jit(do_not_specialize=["N", "idx_ncols", "src_stride0", "out_ncols"])
def manual_scatter(src_ptr, index_ptr, out_ptr, N, idx_ncols, src_stride0, out_ncols, DIM: tl.constexpr,
                   BLOCK: tl.constexpr, LOOP: tl.constexpr, INITIAL_ZERO: tl.constexpr):
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
        if INITIAL_ZERO:
            old = tl.full((BLOCK, ), 0.0, tl.float32)
        else:
            old = tl.load(out_ptr + out_offsets, mask=mask, other=0)
        updated = tl.inline_asm_elementwise("add.rn.ftz.f32 $0, $1, $2;", constraints="=f,f,f", args=[old, src_val],
                                            dtype=tl.float32, is_pure=True, pack=1)
        tl.store(out_ptr + out_offsets, updated, mask=mask)


def source_audit():
    directory = Path(__file__).parent
    name = "scatter_add_2d_kernel"
    trees = [
        ast.parse((directory / relative).read_text())
        for relative in ("vendor/segmented/flag_scatter.py", "segmented_kernels.py")
    ]
    functions = [
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name) for tree in trees
    ]
    for fn in functions:
        fn.decorator_list = []
    signatures_and_bodies = [ast.dump(fn, include_attributes=False) for fn in functions]
    assert signatures_and_bodies[0] == signatures_and_bodies[1]
    return {
        "signature_and_body_identical":
        True,
        "ast_sha256":
        hashlib.sha256(signatures_and_bodies[0].encode()).hexdigest(),
        "dispatch_changes":
        "omit external libentry/heuristics decorators; retain upstream JIT do_not_specialize; explicit tuning",
    }


def code_evidence(kernel, output, label, retain):
    kernel._init_handles()
    ptx = kernel.asm["ptx"]
    evidence = {
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "atomic_sites": len(re.findall(r"\b(?:atom|red)\.", ptx)),
        "registers": kernel.n_regs,
        "spills": kernel.n_spills,
        "shared_bytes": kernel.metadata.shared,
    }
    if retain:
        directory = output / "code"
        directory.mkdir(exist_ok=True)
        for kind in ("ttir", "ttgir", "ptx"):
            (directory / f"{label}.{kind}").write_text(kernel.asm[kind])
        evidence["retained_prefix"] = "code/" + label
    return evidence


def same_values(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def make_inputs(shape, seed, initial_zero):
    rows, selected, width = shape
    torch.manual_seed(seed)
    scores = torch.randn((rows, width), device="cuda")
    values = torch.empty((rows, selected), device="cuda")
    indices = torch.empty((rows, selected), device="cuda", dtype=torch.int64)

    def producer():
        if selected == width:
            torch.sort(scores, dim=1, descending=True, out=(values, indices))
        else:
            torch.topk(scores, selected, dim=1, sorted=True, out=(values, indices))

    producer()
    src = torch.randn((rows, selected), device="cuda")
    initial = torch.zeros((rows, width), device="cuda") if initial_zero else torch.randn((rows, width), device="cuda")
    output = torch.empty_like(initial)
    return scores, values, indices, src, initial, output, producer


def measurement(shape, configs, seed, output, settings, initial_zero, full_chain=False):
    rows, selected, width = shape
    scores, values, indices, src, initial, destination, producer = make_inputs(shape, seed, initial_zero)
    compiled, arguments, grids = {}, {}, {}
    for name, kernel in (("baseline", scatter_add_2d_kernel), ("manual", manual_scatter)):
        block, loop = configs[name]
        grids[name] = (triton.cdiv(rows * selected, block * loop), 1, 1)
        args = (src, indices, destination, rows * selected, selected, selected, width, 1, block, loop)
        if name == "manual":
            args += (initial_zero, )
        arguments[name] = args
        compiled[name] = kernel.warmup(*args, grid=grids[name], num_warps=4, enable_indexed_update_specialization=False)

    def run(name):
        if initial_zero:
            destination.zero_()
        else:
            destination.copy_(initial)
        compiled[name][grids[name]](*arguments[name])
        return destination

    def produce_and_run(name):
        producer()
        return run(name)

    expected = initial.clone()
    expected.scatter_add_(1, indices, src)
    for name in compiled:
        destination.fill_(float("nan"))
        same_values(run(name), expected)
    functions = {name: lambda name=name: run(name) for name in compiled}
    times = paired_device_times(functions, seed=seed, **settings)
    producer_times = None
    if full_chain:
        producer_functions = {name: lambda name=name: produce_and_run(name) for name in compiled}
        producer_times = paired_device_times(producer_functions, seed=seed, **settings)
        scores.mul_(-0.5)
        producer()
        expected.copy_(initial)
        expected.scatter_add_(1, indices, src)
        for fn in producer_functions.values():
            same_values(fn(), expected)
    src.mul_(0.5)
    expected.copy_(initial)
    expected.scatter_add_(1, indices, src)
    for name in compiled:
        same_values(run(name), expected)
    block, loop = configs["baseline"]
    retain = seed == 0 and rows == 31 and block == 64 and loop in (1, 2)
    evidence = {
        name:
        code_evidence(kernel, output, f"r{rows}-k{selected}-w{width}-b{block}-l{loop}-z{int(initial_zero)}-{name}",
                      retain)
        for name, kernel in compiled.items()
    }
    assert evidence["baseline"]["atomic_sites"] > 0 and evidence["manual"]["atomic_sites"] == 0
    return {
        "shape_r_k_w": shape,
        "block": block,
        "loop": loop,
        "configs": configs,
        "seed": seed,
        "device": times,
        "initial_zero": initial_zero,
        "producer_and_consumer_device": producer_times,
        "producer": "sort" if selected == width else "topk",
        "code": evidence,
        "validation":
        "exact FP32 baseline/manual/reference; poisoned output; unwritten positions; changed source; producer-established per-row uniqueness",
        "scope": "output initialization + scatter; producer-inclusive timing reported separately; allocations excluded",
    }


def aggregate(cases):
    groups = {}
    for case in cases:
        groups.setdefault(tuple(case["shape_r_k_w"]), []).append(case)
    summary = []
    for shape, group in groups.items():
        configurations = sorted({(row["block"], row["loop"]) for row in group})
        times = {}
        for block, loop in configurations:
            rows = [row for row in group if row["block"] == block and row["loop"] == loop]
            times[f"b{block}-l{loop}"] = {
                name: statistics.median(row["device"]["median_us"][name]
                                        for row in rows)
                for name in rows[0]["device"]["median_us"]
            }
        best = {
            name: min((values[name], config)
                      for config, values in times.items())
            for name in next(iter(times.values()))
        }
        summary.append({
            "shape_r_k_w": shape,
            "config_median_us": times,
            "best": {name: {"median_us": entry[0], "configuration": entry[1]}
                     for name, entry in best.items()},
            "manual_speedup": best["baseline"][0] / best["manual"][0],
        })
    return summary


def confirm_aggregate(cases):
    groups = {}
    for case in cases:
        groups.setdefault(tuple(case["shape_r_k_w"]), []).append(case)
    summary = []
    for shape, group in groups.items():
        consumer = {
            name: statistics.median(row["device"]["median_us"][name]
                                    for row in group)
            for name in ("baseline", "manual")
        }
        chain = {
            name: statistics.median(row["producer_and_consumer_device"]["median_us"][name]
                                    for row in group)
            for name in ("baseline", "manual")
        }
        summary.append({
            "shape_r_k_w": shape, "configs": group[0]["configs"], "consumer_median_us": consumer,
            "producer_consumer_median_us": chain, "consumer_speedup": consumer["baseline"] / consumer["manual"],
            "producer_consumer_speedup": chain["baseline"] / chain["manual"], "screened":
            group[0].get("screened", True)
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--initial", choices=("nonzero", "zero"), default="nonzero")
    parser.add_argument("--screen", type=Path)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    settings = {
        "rounds": 15 if options.mode == "confirm" else 5, "repetitions": 32, "warmup_ms":
        200 if options.mode == "confirm" else 40
    }
    directory = Path(__file__).parent
    sources = [Path(__file__), directory / "segmented_kernels.py", directory / "coverage_experiment.py"]
    report = {
        "status": "running",
        "mode": options.mode,
        "initial": options.initial,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "compiler_sha256": digest(libtriton.__file__),
        "source_audit": source_audit(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path.resolve()): digest(path)
                          for path in sources},
        "settings": settings,
        "cases": [],
    }
    try:
        if options.mode == "screen":
            for shape in ((31, 17, 67), (1024, 64, 256), (4096, 128, 1024), (4096, 1024, 2048)):
                for block in (32, 64, 128, 256):
                    for loop in (1, 2, 4):
                        configs = {name: (block, loop) for name in ("baseline", "manual")}
                        case = measurement(shape, configs, 0, options.output, settings, options.initial == "zero")
                        report["cases"].append(case)
                        print(
                            json.dumps({
                                "shape": shape, "block": block, "loop": loop, "device_us": case["device"]["median_us"]
                            }), flush=True)
                        save_json(options.output / "results.json", report)
        else:
            if not options.screen:
                raise ValueError("--screen results.json is required for confirmation")
            screen = json.loads(options.screen.read_text())
            report["screen_sha256"] = digest(options.screen)
            report["screen_path"] = str(options.screen)
            for group in screen["summary"]:
                shape = group["shape_r_k_w"]
                configs = {}
                for name, choice in group["best"].items():
                    match = re.fullmatch(r"b(\d+)-l(\d+)", choice["configuration"])
                    configs[name] = tuple(map(int, match.groups()))
                for seed in range(3):
                    case = measurement(shape, configs, seed, options.output, settings, options.initial == "zero", True)
                    report["cases"].append(case)
                    print(
                        json.dumps({
                            "shape": shape, "seed": seed, "configs": configs, "device_us": case["device"]["median_us"],
                            "producer_consumer_us": case["producer_and_consumer_device"]["median_us"]
                        }), flush=True)
                    save_json(options.output / "results.json", report)
        if options.mode == "confirm":
            shape = (128, 67, 67)
            configs = {name: (128, 1) for name in ("baseline", "manual")}
            case = measurement(shape, configs, 0, options.output, settings, options.initial == "zero", True)
            case["screened"] = False
            report["cases"].append(case)
            report["summary"] = confirm_aggregate(report["cases"])
        else:
            report["summary"] = aggregate(report["cases"])
        report["status"] = "passed"
    finally:
        save_json(options.output / "results.json", report)


if __name__ == "__main__":
    main()
