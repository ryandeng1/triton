#!/usr/bin/env python3
"""Four atomic memory orders on pinned FlagGems embedding updates.

Only the atomic sem keyword changes. Both native specialization flags are off.
Device operation timing includes common reset/count/update/copy buffers. Release
versus relaxed controls the red instruction family, but changes GPU ordering as
well as CTA barriers; these measurements do not isolate pure barrier latency.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton._C import libtriton

import irregular_kernels as upstream
from coverage_experiment import digest, paired_device_times, save_json
from native_indexed_update_experiment import kernel_evidence, same_values, source_audit

MODES = ("acq_rel", "acquire", "release", "relaxed")
FLAGS = {"enable_indexed_update_specialization": False, "enable_private_atomic_relaxation": False}
RATIOS = (("acq_rel", "relaxed"), ("acq_rel", "release"), ("acq_rel", "acquire"), ("release", "relaxed"),
          ("acquire", "relaxed"), ("acquire", "release"))


def fourway_source_audit():
    directory = Path(__file__).parent
    original = ast.parse((directory / "irregular_kernels.py").read_text())
    changed = ast.parse(Path(__file__).read_text())
    result = {"upstream": source_audit(), "controls": {}}
    for name in ("_embedding_dense_backward_kernel", "_embedding_dense_backward_kernel_scale_by_freq"):
        old = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
        for mode in MODES[1:]:
            new = copy.deepcopy(
                next(node for node in changed.body
                     if isinstance(node, ast.FunctionDef) and node.name == name + "_" + mode))
            new.name = name
            calls = [
                node for node in ast.walk(new)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(
                    node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == "atomic_add"
            ]
            assert len(calls) == 1
            sem = [item for item in calls[0].keywords if item.arg == "sem"]
            assert len(sem) == 1 and isinstance(sem[0].value, ast.Constant) and sem[0].value.value == mode
            calls[0].keywords.remove(sem[0])
            assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
            result["controls"][name + "_" + mode] = "Only sem keyword added, apart from function name; AST audited"
    return result


def code_evidence(kernel, output, label, retain, mode):
    evidence = kernel_evidence(kernel, output, label, retain)
    ptx = kernel.asm["ptx"]
    evidence["atomic_instructions"] = sorted(set(re.findall(r"\b(?:atom|red)\.[\w.]+", ptx)))
    evidence["bar_sync_sites"] = len(re.findall(r"\bbar\.sync\b", ptx))
    evidence["ptx_atomic_and_barrier_lines"] = [
        line.strip() for line in ptx.splitlines() if re.search(r"\b(?:atom\.|red\.|bar\.sync\b)", line)
    ]
    expected_family = "atom." if mode in ("acq_rel", "acquire") else "red."
    assert evidence["atomic_sites"] > 0
    assert all(op.startswith(expected_family) and mode in op for op in evidence["atomic_instructions"])
    assert evidence["bar_sync_sites"] == {"acq_rel": 2, "acquire": 1, "release": 1, "relaxed": 0}[mode]
    assert kernel.metadata.enable_indexed_update_specialization is False
    assert kernel.metadata.enable_private_atomic_relaxation is False
    evidence["native_flags"] = FLAGS
    if retain:
        cuobjdump = shutil.which("cuobjdump")
        evidence["sass"] = {"tool": cuobjdump, "available": False}
        if cuobjdump:
            with tempfile.TemporaryDirectory(prefix="atomic-fourway-") as temporary:
                cubin = Path(temporary) / "kernel.cubin"
                cubin.write_bytes(kernel.asm["cubin"])
                disassembly = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], text=True, capture_output=True)
            evidence["sass"]["returncode"] = disassembly.returncode
            evidence["sass"]["stderr"] = disassembly.stderr
            if disassembly.returncode == 0:
                sass = disassembly.stdout
                path = output / "code" / f"{label}.sass"
                path.write_text(sass)
                evidence["sass"].update({
                    "available":
                    True,
                    "path":
                    str(path.relative_to(output)),
                    "sha256":
                    hashlib.sha256(sass.encode()).hexdigest(),
                    "atomic_barrier_ordering_lines": [
                        line.strip()
                        for line in sass.splitlines()
                        if re.search(r"\b(?:ATOM\w*|RED\w*|BAR|MEMBAR|FENCE)\.", line)
                    ],
                })
    return evidence


def measurement(shape, frequency, block, seed, output, settings):
    n, d, vocabulary = shape
    torch.manual_seed(seed)
    indices = torch.randperm(vocabulary, device="cuda")[:n].contiguous()
    grad = torch.randn((n, d), dtype=torch.bfloat16, device="cuda")
    assert torch.all((grad == 0) | (grad.abs().float() >= torch.finfo(torch.float32).tiny))
    scratch = torch.empty((vocabulary, d), dtype=torch.float32, device="cuda")
    result = torch.empty((vocabulary, d), dtype=grad.dtype, device="cuda")
    counts = torch.empty(vocabulary, dtype=torch.int32, device="cuda")
    padding = 0
    grid = (n, triton.cdiv(d, block), 1)
    name = "_embedding_dense_backward_kernel" + ("_scale_by_freq" if frequency else "")
    kernels = {mode: getattr(upstream, name) if mode == "acq_rel" else globals()[name + "_" + mode] for mode in MODES}
    arguments = ((grad, indices, counts, scratch, vocabulary, padding, block, d) if frequency else
                 (grad, indices, scratch, vocabulary, padding, block, d))
    compiled = {mode: kernel.warmup(*arguments, grid=grid, num_warps=4, **FLAGS) for mode, kernel in kernels.items()}

    def run(mode):
        scratch.zero_()
        if frequency:
            counts.zero_()
            upstream._embedding_dense_backward_count_kernel[(triton.cdiv(n, 512), )](indices, counts, n, vocabulary,
                                                                                     padding, BLOCK_N=512, **FLAGS)
        compiled[mode][grid](*arguments)
        result.copy_(scratch)
        return result

    valid = indices != padding
    expected = torch.zeros((vocabulary, d), device="cuda", dtype=torch.float32)

    def reference():
        expected.zero_()
        # Global FP32 atomic addition flushes subnormal inputs/results to signed
        # zero; adding any signed zero to the initial +0 produces +0 here.
        values = grad[valid].float()
        values = torch.where(values.abs() < torch.finfo(torch.float32).tiny, 0.0, values)
        expected[indices[valid]] = values

    def validate():
        reference()
        for mode in MODES:
            result.fill_(float("nan"))
            run(mode)
            same_values(scratch, expected)
            same_values(result, expected.to(grad.dtype))

    validate()
    functions = {mode: lambda mode=mode: run(mode) for mode in MODES}
    times = paired_device_times(functions, seed=seed, **settings)
    grad.mul_(0.5)
    validate()
    # Explicit FTZ/signed-zero correctness probe is outside all timed regions.
    grad.zero_()
    row = int(torch.nonzero(valid)[0].item())
    tiny = torch.finfo(torch.float32).tiny
    grad[row, :4] = torch.tensor([-0.0, tiny / 128, -tiny / 128, tiny], device="cuda", dtype=grad.dtype)
    validate()
    retain = seed == 0 and n == 4096 and d == 128 and block == 128
    label = f"n{n}-d{d}-v{vocabulary}-freq{int(frequency)}"
    evidence = {mode: code_evidence(item, output, label + "-" + mode, retain, mode) for mode, item in compiled.items()}
    return {
        "shape_n_d_v": shape,
        "frequency": frequency,
        "block_d": block,
        "seed": seed,
        "device": times,
        "code": evidence,
        "ratios":
        {left + "_over_" + right: times["median_us"][left] / times["median_us"][right]
         for left, right in RATIOS},
        "validation":
        "Exact FP32 scratch/BF16 result; poisoned output; changed gradients; explicit FTZ and signed zero",
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
                mode: statistics.median(row["device"]["median_us"][mode]
                                        for row in selected)
                for mode in MODES
            }
        best = {mode: min((values[mode], int(block)) for block, values in configs.items()) for mode in MODES}
        result.append({
            "shape_n_d_v": shape,
            "frequency": frequency,
            "config_median_us": configs,
            "config_ratios": {
                block: {left + "_over_" + right: times[left] / times[right]
                        for left, right in RATIOS}
                for block, times in configs.items()
            },
            "best": {mode: {"median_us": value[0], "block_d": value[1]}
                     for mode, value in best.items()},
            "best_ratios": {left + "_over_" + right: best[left][0] / best[right][0]
                            for left, right in RATIOS},
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
        Path(__file__), directory / "native_indexed_update_experiment.py", directory / "irregular_kernels.py",
        directory / "coverage_experiment.py", directory / "vendor/irregular/flag_embedding.py",
        root / "third_party/nvidia/backend/compiler.py", root / "lib/Analysis/Utility.cpp",
        root / "lib/Conversion/TritonGPUToLLVM/Utility.cpp",
        root / "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp"
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
        fourway_source_audit(),
        "native_flags":
        FLAGS,
        "settings":
        settings,
        "cases": [],
        "scope":
        "Four memory orders on original FlagGems indexed updates; no native transformation, FX or Inductor",
        "limits":
        "Unique randperm subsets, ordinary finite BF16 gradients; device timings exclude producers, allocations and host guards. Release versus relaxed holds red family fixed but changes GPU ordering as well as CTA barriers: no pure-barrier causal claim.",
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
        ratio_names = [left + "_over_" + right for left, right in RATIOS]
        report["geometric_mean_ratios_all_36_paired_cases"] = {
            name: math.exp(statistics.mean(math.log(row["ratios"][name])
                                           for row in report["cases"]))
            for name in ratio_names
        }
        report["geometric_mean_ratios_best_of_two_blocks"] = {
            name: math.exp(statistics.mean(math.log(row["best_ratios"][name])
                                           for row in report["summary"]))
            for name in ratio_names
        }
        report["compiler_unchanged"] = digest(libtriton.__file__) == report["compiler_sha256"]
        report["sources_unchanged"] = all(
            digest(path) == report["source_sha256"][str(path.resolve())] for path in sources)
        assert report["compiler_unchanged"] and report["sources_unchanged"]
        report["status"] = "passed"
    finally:
        save_json(args.output / "results.json", report)


@triton.jit
def _embedding_dense_backward_kernel_acquire(grad_output_ptr, indices_ptr, grad_weight_ptr, num_weights, padding_idx,
                                             BLOCK_D: tl.constexpr, EMBED_DIM: tl.constexpr):
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
    tl.atomic_add(gw_ptrs, go, mask=mask, sem='acquire')


@triton.jit
def _embedding_dense_backward_kernel_release(grad_output_ptr, indices_ptr, grad_weight_ptr, num_weights, padding_idx,
                                             BLOCK_D: tl.constexpr, EMBED_DIM: tl.constexpr):
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
    tl.atomic_add(gw_ptrs, go, mask=mask, sem='release')


@triton.jit
def _embedding_dense_backward_kernel_relaxed(grad_output_ptr, indices_ptr, grad_weight_ptr, num_weights, padding_idx,
                                             BLOCK_D: tl.constexpr, EMBED_DIM: tl.constexpr):
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
    tl.atomic_add(gw_ptrs, go, mask=mask, sem='relaxed')


@triton.jit
def _embedding_dense_backward_kernel_scale_by_freq_acquire(grad_output_ptr, indices_ptr, counts_ptr, grad_weight_ptr,
                                                           num_weights, padding_idx, BLOCK_D: tl.constexpr,
                                                           EMBED_DIM: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM
    idx = tl.load(indices_ptr + pid_n).to(tl.int32)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0)
    cnt = tl.load(counts_ptr + idx, mask=valid, other=1)
    go = go / cnt
    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask, sem='acquire')


@triton.jit
def _embedding_dense_backward_kernel_scale_by_freq_release(grad_output_ptr, indices_ptr, counts_ptr, grad_weight_ptr,
                                                           num_weights, padding_idx, BLOCK_D: tl.constexpr,
                                                           EMBED_DIM: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM
    idx = tl.load(indices_ptr + pid_n).to(tl.int32)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0)
    cnt = tl.load(counts_ptr + idx, mask=valid, other=1)
    go = go / cnt
    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask, sem='release')


@triton.jit
def _embedding_dense_backward_kernel_scale_by_freq_relaxed(grad_output_ptr, indices_ptr, counts_ptr, grad_weight_ptr,
                                                           num_weights, padding_idx, BLOCK_D: tl.constexpr,
                                                           EMBED_DIM: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM
    idx = tl.load(indices_ptr + pid_n).to(tl.int32)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0)
    cnt = tl.load(counts_ptr + idx, mask=valid, other=1)
    go = go / cnt
    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask, sem='relaxed')


if __name__ == "__main__":
    main()
