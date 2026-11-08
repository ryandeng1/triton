#!/usr/bin/env python3
"""Measure producer uniqueness/coverage, direct output writes, and real ATen lowering.

Run from the Triton checkout with its source build installed. Output must be new.
Consumer ablations share physical output/scratch buffers and include all required
zero/count/convert launches. Framework graphs include producers and use separate
functional allocations. CUDA-graph device time and ordinary-call wall time are
reported separately. This is a bounded experiment, not a model benchmark.
"""

import argparse
import hashlib
import json
import random
import re
import statistics
import subprocess
import time
from pathlib import Path

import torch
import triton
from triton._C import libtriton
from triton.experimental.semantic_contracts import FactsAnalyzer, compile
from triton.experimental.semantic_contracts import kernels


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_samples(samples):
    return {"samples": samples, "median": statistics.median(samples), "minimum": min(samples), "maximum": max(samples)}


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def independent_unique_reference(grad, indices, vocabulary, padding):
    # Performance data is finite, normal BF16, so a unique +0 contribution is
    # exactly represented by this independent assignment reference. Edge-value
    # arithmetic is covered separately by the runtime regression tests.
    expected = torch.zeros((vocabulary, grad.shape[1]), device=grad.device, dtype=grad.dtype)
    valid = (indices >= 0) & (indices < vocabulary) & (indices != padding)
    expected[indices[valid].long()] = grad[valid]
    return expected


def paired_device_times(functions, *, seed, rounds, repetitions, warmup_ms):
    for fn in functions.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graphs = {}
    for name, fn in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repetitions):
                fn()
        graphs[name] = graph
    deadline = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < deadline:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    rng = random.Random(seed)
    samples = {name: [] for name in functions}
    for _ in range(rounds):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / repetitions)
    return {
        "samples_us": samples, "median_us": {name: statistics.median(values)
                                             for name, values in samples.items()}, "rounds": rounds, "repetitions":
        repetitions, "warmup_ms": warmup_ms, "order": "randomized interleaving; clocks not locked"
    }


def paired_wall_times(functions, *, seed, rounds, repetitions):
    rng = random.Random(seed + 8192)
    samples = {name: [] for name in functions}
    for _ in range(rounds):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for _ in range(repetitions):
                functions[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter_ns() - start) / 1000 / repetitions)
    return {
        "samples_us": samples, "median_us": {name: statistics.median(values)
                                             for name, values in samples.items()}, "rounds": rounds, "repetitions":
        repetitions, "scope":
        "ordinary calls; Python interpretation/guards, allocation, dispatch, and GPU completion included"
    }


def code_metadata(compiled, output, label, *, retain=False):
    compiled._init_handles()
    ptx = compiled.asm.get("ptx", "")
    result = {
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(), "atomic_instruction_sites":
        len(re.findall(r"\b(?:atom|red)\.", ptx)), "registers": compiled.n_regs, "spills": compiled.n_spills,
        "shared_bytes": compiled.metadata.shared
    }
    if retain:
        code_dir = output / "code"
        code_dir.mkdir(exist_ok=True)
        path = code_dir / (label + ".ptx")
        path.write_text(ptx)
        result["retained_ptx"] = str(path.relative_to(output))
    return result


def consumer_case(shape, frequency, block, seed, output, settings):
    n, d, vocabulary = shape
    padding = 0
    torch.manual_seed(seed)
    indices = torch.randperm(vocabulary, device="cuda")[:n].contiguous()
    grad = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty((vocabulary, d), device="cuda", dtype=torch.float32)
    out = torch.empty((vocabulary, d), device="cuda", dtype=grad.dtype)
    counts = torch.empty(vocabulary, device="cuda", dtype=torch.int32)
    expected = independent_unique_reference(grad, indices, vocabulary, padding)
    complete = n == vocabulary
    size = vocabulary * d
    grid = (n, triton.cdiv(d, block))
    launches = {}
    compiled = {name: {} for name in ("atomic", "unique_scratch", "unique_direct")}

    def run(name):
        found = compiled[name]
        steps = []
        if name == "unique_direct":
            if not complete:
                found["zero_output"] = kernels.zero_buffer[(triton.cdiv(size, 1024), )](out, size, 1024)
                steps.append("zero_output")
            found["update"] = kernels.embedding_unique_write[grid](grad, indices, out, n, d, vocabulary, padding, block,
                                                                   complete, True, num_warps=4)
            steps.append("update")
        else:
            found["zero_scratch"] = kernels.zero_buffer[(triton.cdiv(size, 1024), )](scratch, size, 1024)
            steps.append("zero_scratch")
            if name == "atomic" and frequency:
                found["zero_counts"] = kernels.zero_buffer[(triton.cdiv(vocabulary, 1024), )](counts, vocabulary, 1024)
                found["count"] = kernels.embedding_count[(triton.cdiv(n, 512), )](indices, counts, n, vocabulary,
                                                                                  padding, 512)
                steps.extend(("zero_counts", "count"))
            found["update"] = kernels.embedding_update[grid](grad, indices, counts, scratch, n, d, vocabulary, padding,
                                                             frequency, name == "unique_scratch", True, block,
                                                             num_warps=4)
            found["convert"] = kernels.convert_output[(triton.cdiv(size, 1024), )](scratch, out, size, 1024)
            steps.extend(("update", "convert"))
        launches[name] = steps
        return out

    functions = {name: lambda name=name: run(name) for name in compiled}
    for name, fn in functions.items():
        out.fill_(float("nan"))
        scratch.fill_(float("nan"))
        fn()
        assert_bytes(out, expected)
    times = paired_device_times(functions, seed=seed, **settings["device"])
    # A second input catches accidental dependence on prior outputs/scratch.
    grad.mul_(0.5)
    changed = independent_unique_reference(grad, indices, vocabulary, padding)
    for name, fn in functions.items():
        out.fill_(float("nan"))
        fn()
        assert_bytes(out, changed)
    retain = seed == 0 and n == 4096 and d == 128 and frequency and block == 128
    code = {
        name: {
            kind: code_metadata(kernel, output, f"consumer-v{vocabulary}-{name}-{kind}", retain=retain)
            for kind, kernel in artifacts.items()
        }
        for name, artifacts in compiled.items()
    }
    assert code["atomic"]["update"]["atomic_instruction_sites"] > 0
    assert code["unique_scratch"]["update"]["atomic_instruction_sites"] == 0
    assert code["unique_direct"]["update"]["atomic_instruction_sites"] == 0
    return {
        "phase": "consumer", "shape_n_d_v": shape, "frequency": frequency, "block_d": block, "seed": seed,
        "padding_idx": padding, "complete_coverage": complete, "proof_source":
        "fixed randperm producer; full coverage iff N=V; no value-based uniqueness guard", "allocation_control":
        "all variants share physical Out; scratch/count buffers shared where needed", "timing_scope":
        "all required consumer launches; excludes index producer, allocation, and Python dispatch", "validation":
        "bitwise independent-reference checks, poisoned output, changed gradient, shared bindings", "kernel_launches":
        launches, "device": times, "code": code
    }


def producer_graph(shape, producer, frequency):
    n, _, vocabulary = shape

    def graph(gradient):
        if producer == "randperm":
            indices = torch.randperm(vocabulary, device=gradient.device)[:n]
        else:
            indices = torch.arange(vocabulary, device=gradient.device)[:n]
        result = torch.ops.aten.embedding_dense_backward.default(gradient, indices, vocabulary, 0, frequency)
        return result, indices

    return graph


def framework_case(shape, producer, frequency, seed, output, settings):
    n, d, vocabulary = shape
    torch.manual_seed(seed)
    grad = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    graph = producer_graph(shape, producer, frequency)
    start = time.perf_counter_ns()
    baseline = compile(graph, (grad, ), enabled=False, autotune=False)
    baseline_compile_us = (time.perf_counter_ns() - start) / 1000
    start = time.perf_counter_ns()
    optimized = compile(graph, (grad, ), enabled=True, autotune=False)
    optimized_compile_us = (time.perf_counter_ns() - start) / 1000
    analysis_samples = []
    for _ in range(settings["analysis_repetitions"]):
        start = time.perf_counter_ns()
        FactsAnalyzer(optimized.graph_module).analyze()
        analysis_samples.append((time.perf_counter_ns() - start) / 1000)
    functions = {
        "aten_fx": lambda: baseline(grad), "semantic_fx": lambda: optimized(grad), "pytorch": lambda: graph(grad)
    }
    first_call_us = {}
    expected = None
    for name, fn in functions.items():
        torch.manual_seed(seed + 1729)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        actual, indices = fn()
        torch.cuda.synchronize()
        first_call_us[name] = (time.perf_counter_ns() - start) / 1000
        reference = independent_unique_reference(grad, indices, vocabulary, 0)
        assert_bytes(actual, reference)
        if expected is None:
            expected = actual.clone(), indices.clone()
        else:
            assert_bytes(indices, expected[1])
            assert_bytes(actual, expected[0])
    plans = optimized.report()["nodes"]
    assert len(plans) == 1 and plans[0]["selected_optimization"] == "unique_direct_write", plans
    assert plans[0]["kernel_launches"] == (["update"] if n == vocabulary else ["zero_output", "update"])
    assert baseline.report()["nodes"][0]["selected_optimization"] == "aten_fallback"
    # Vary capture/allocation order between seeds; these functional outputs are
    # separate allocations. The consumer experiment supplies the address control.
    names = list(functions)
    random.Random(seed + 4096).shuffle(names)
    functions = {name: functions[name] for name in names}
    device = paired_device_times(functions, seed=seed, **settings["device"])
    wall = paired_wall_times(functions, seed=seed, **settings["wall"])
    grad.mul_(0.5)
    for fn in functions.values():
        actual, indices = fn()
        assert_bytes(actual, independent_unique_reference(grad, indices, vocabulary, 0))
    retain = seed == 0 and n == 4096 and d == 128 and producer == "randperm" and frequency
    code = {
        node: {
            kind: code_metadata(kernel, output, f"framework-v{vocabulary}-{node}-{kind}", retain=retain)
            for kind, kernel in variants.items()
        }
        for node, variants in optimized.kernels.items()
    }
    for variants in code.values():
        assert variants["update"]["atomic_instruction_sites"] == 0
    return {
        "phase": "framework", "shape_n_d_v": shape, "producer": producer, "seed": seed, "frequency": frequency,
        "padding_idx": 0, "plans": plans, "capture":
        "real aten.embedding_dense_backward.default; producer executes inside every graph invocation",
        "allocation_control": "separate functional allocations; capture order shuffled across seeds", "validation":
        "bitwise ATen/semantic/reference equality on normal BF16, paired RNG, changed inputs", "device": device, "wall":
        wall, "host_setup": {
            "capture_and_analysis_us": {"aten_fx": baseline_compile_us, "semantic_fx": optimized_compile_us},
            "isolated_analysis_us": summarize_samples(analysis_samples), "first_call_us": first_call_us, "note":
            "both compile modes run analysis; first calls use the current on-disk/in-process kernel cache"
        }, "code": code
    }


def aggregate(cases):
    consumer_groups = {}
    framework_groups = {}
    for case in cases:
        if case["phase"] == "consumer":
            key = (tuple(case["shape_n_d_v"]), case["frequency"])
            consumer_groups.setdefault(key, []).append(case)
        else:
            key = (tuple(case["shape_n_d_v"]), case["producer"], case["frequency"])
            framework_groups.setdefault(key, []).append(case)
    result = {
        "consumer": [], "framework": [], "aggregation":
        "median of seed medians; consumer best-to-best over blockD128/256 only; framework fixed backend blockD128"
    }
    for (shape, frequency), rows in consumer_groups.items():
        by_block = {}
        for block in sorted({row["block_d"] for row in rows}):
            selected = [row for row in rows if row["block_d"] == block]
            by_block[str(block)] = {
                name: statistics.median(row["device"]["median_us"][name]
                                        for row in selected)
                for name in ("atomic", "unique_scratch", "unique_direct")
            }
        best = {
            name: min(((int(block), values[name])
                       for block, values in by_block.items()), key=lambda pair: pair[1])
            for name in ("atomic", "unique_scratch", "unique_direct")
        }
        result["consumer"].append({
            "shape_n_d_v":
            shape, "frequency":
            frequency, "config_median_us":
            by_block, "best": {name: {"block_d": pair[0], "median_us": pair[1]}
                               for name, pair in best.items()}, "direct_over_atomic_speedup":
            best["atomic"][1] / best["unique_direct"][1], "direct_over_previous_unique_speedup":
            best["unique_scratch"][1] / best["unique_direct"][1]
        })
    for (shape, producer, frequency), rows in framework_groups.items():
        medians = {
            metric: {
                name: statistics.median(row[metric]["median_us"][name]
                                        for row in rows)
                for name in ("aten_fx", "semantic_fx", "pytorch")
            }
            for metric in ("device", "wall")
        }
        result["framework"].append({
            "shape_n_d_v": shape, "producer": producer, "frequency": frequency, "median_us": medians,
            "speedup_over_aten_fx":
            {metric: values["aten_fx"] / values["semantic_fx"]
             for metric, values in medians.items()}, "speedup_over_pytorch":
            {metric: values["pytorch"] / values["semantic_fx"]
             for metric, values in medians.items()}
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "confirm"), default="confirm")
    parser.add_argument("--phase", choices=("all", "consumer", "framework"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    smoke = args.mode == "smoke"
    settings = {
        "device": {"rounds": 3 if smoke else 15, "repetitions": 8 if smoke else 32, "warmup_ms":
                   20 if smoke else 200}, "wall": {"rounds": 2 if smoke else 5, "repetitions": 3 if smoke else 10},
        "analysis_repetitions": 3 if smoke else 10
    }
    package = Path(triton.__file__).parent / "experimental/semantic_contracts"
    source_files = [Path(__file__), *sorted(package.glob("*.py"))]
    report = {
        "status": "running", "mode": args.mode, "phase": args.phase, "torch": torch.__version__, "torch_revision":
        torch.version.git_version, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(), "cuda":
        torch.version.cuda, "compiler_sha256": digest(libtriton.__file__), "git_revision":
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "source_sha256":
        {str(path.resolve()): digest(path)
         for path in source_files}, "settings": settings, "cases": [], "scope":
        "bounded BF16 performance experiment; detailed floating-point/alias/coverage guards tested separately",
        "artifact_policy":
        "JSON samples, source/compiler hashes, and representative PTX only; no duplicate cubins/IR trees"
    }
    seeds = range(1 if smoke else 3)
    consumer_shapes = [((256, 64, 256), True), ((4096, 128, 8192), True)] if smoke else [((256, 64, 256), True),
                                                                                         ((4096, 128, 4096), False),
                                                                                         ((4096, 128, 4096), True),
                                                                                         ((4096, 128, 8192), True),
                                                                                         ((16384, 256, 16384), True)]
    framework_shapes = [(256, 64, 256), (4096, 128, 8192)] if smoke else [(256, 64, 256), (4096, 128, 4096),
                                                                          (4096, 128, 8192), (16384, 256, 16384)]
    try:
        if args.phase in ("all", "consumer"):
            for shape, frequency in consumer_shapes:
                for block in (128, 256):
                    for seed in seeds:
                        case = consumer_case(shape, frequency, block, seed, args.output, settings)
                        report["cases"].append(case)
                        print(
                            json.dumps({
                                "phase": "consumer", "shape": shape, "frequency": frequency, "block": block, "seed":
                                seed, "device_us": case["device"]["median_us"]
                            }), flush=True)
                        save_json(args.output / "results.json", report)
        if args.phase in ("all", "framework"):
            for shape in framework_shapes:
                for producer in ("arange", "randperm"):
                    for frequency in (False, True):
                        for seed in seeds:
                            case = framework_case(shape, producer, frequency, seed, args.output, settings)
                            report["cases"].append(case)
                            print(
                                json.dumps({
                                    "phase": "framework", "shape": shape, "producer": producer, "frequency": frequency,
                                    "seed": seed, "device_us": case["device"]["median_us"], "wall_us":
                                    case["wall"]["median_us"]
                                }), flush=True)
                            save_json(args.output / "results.json", report)
        report["summary"] = aggregate(report["cases"])
        report["status"] = "passed"
    finally:
        save_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
