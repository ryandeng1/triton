#!/usr/bin/env python3
"""Compare real embedding-backward producer graphs with PyTorch Inductor.

All variants include the index producer and fresh functional outputs. CUDA-graph
device timings and ordinary-call wall timings are reported separately. Inductor
default and max-autotune-no-cudagraphs use the same external capture protocol;
the latter enables max tuning without nesting Inductor's internal graph trees.
Compilation failures are recorded and excluded, never silently replaced by eager.
"""

import argparse
import hashlib
import json
import random
import re
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from unittest import mock

import torch
import triton
from triton._C import libtriton
from triton.experimental.semantic_contracts import compile as semantic_compile

from coverage_experiment import (assert_bytes, digest, independent_unique_reference, paired_device_times,
                                 paired_wall_times, producer_graph, save_json)


def code_evidence(codes, directory, label):
    """Keep hashes and diagnostic lines, not another generated-code archive."""
    directory.mkdir(exist_ok=True)
    evidence = []
    pattern = re.compile(r"(?:aten\.|extern_kernels\.)(?:embedding_dense_backward|randperm|index_put)|"
                         r"triton_.*\.run\(|async_compile\.triton|def call\(|tl\.(?:atomic_add|atomic_cas|store|load)")
    for index, code in enumerate(dict.fromkeys(codes)):
        lines = code.splitlines()
        selected = set()
        for number, line in enumerate(lines):
            if not line.lstrip().startswith("#") and pattern.search(line):
                selected.update(range(max(0, number - 1), min(len(lines), number + 2)))
        excerpt = "\n".join(f"{number + 1}: {lines[number]}" for number in sorted(selected))
        excerpt = excerpt[:24000]
        path = directory / f"{label}-{index}.txt"
        path.write_text(excerpt + "\n")
        executable = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
        evidence.append({
            "sha256":
            hashlib.sha256(code.encode()).hexdigest(), "source_bytes":
            len(code.encode()), "excerpt":
            str(path), "atomic_add_sites":
            len(re.findall(r"\btl\.atomic_add\(", code)), "external_embedding_call":
            bool(re.search(r"(?:aten\.|extern_kernels\.)embedding_dense_backward(?:\.\w+)?\(",
                           executable)), "external_randperm_call":
            bool(re.search(r"(?:aten\.|extern_kernels\.)randperm(?:\.\w+)?\(", executable)), "external_index_put_call":
            bool(re.search(r"(?:aten\.|extern_kernels\.)index_put_?(?:\.\w+)?\(", executable)),
            "triton_kernel_definitions":
            len(re.findall(r"async_compile\.triton\(", code)), "inspection_scope":
            "generated Python module source; counts are static sites, not runtime launches"
        })
    return {"captured_modules": evidence, "code_capture_succeeded": bool(codes)}


def compile_inductor(graph, grad, mode, directory, label):
    # This mirrors Torch's run_and_get_code helper, without resetting Dynamo
    # between variants. Disable the FX graph cache only during first compilation
    # so the generated module is observable even when lower-level kernels cache.
    from torch._inductor import config
    from torch._inductor.graph import GraphLowering

    codes = []
    start = time.perf_counter_ns()
    try:
        with config.patch(fx_graph_cache=False), mock.patch.object(GraphLowering, "save_output_code", codes.append):
            compiled = torch.compile(graph, backend="inductor", fullgraph=True, dynamic=False, mode=mode)
            result = compiled(grad)
            torch.cuda.synchronize()
        return compiled, result, {
            "status": "compiled", "mode": mode, "first_compile_and_call_us": (time.perf_counter_ns() - start) / 1000,
            "code": code_evidence(codes, directory, label)
        }
    except Exception as error:
        if any(message in str(error).lower() for message in ("device-side assert", "illegal memory access")):
            raise
        return None, None, {
            "status": "failed", "mode": mode, "error_type": type(error).__name__, "error": str(error)[:8000],
            "traceback": traceback.format_exc(limit=8)[-12000:], "first_compile_and_call_us":
            (time.perf_counter_ns() - start) / 1000, "code": code_evidence(codes, directory, label)
        }


def configuration(shape, producer, frequency, output, settings, seeds):
    n, d, vocabulary = shape
    label = f"n{n}-d{d}-v{vocabulary}-{producer}-freq{int(frequency)}"
    torch._dynamo.reset()
    torch.manual_seed(0)
    grad = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    reference_graph = producer_graph(shape, producer, frequency)

    def graph(gradient):
        # Returning indices would force materialization and could block producer
        # fusion. Timed variants expose only the actual gradient result.
        return reference_graph(gradient)[0]

    semantic = semantic_compile(graph, (grad, ), enabled=True, autotune=False)
    functions = {"pytorch_eager": lambda: graph(grad), "semantic_fx": lambda: semantic(grad)}
    setup = {}
    for name, mode in (("inductor_default", "default"), ("inductor_max_tune", "max-autotune-no-cudagraphs")):
        torch.manual_seed(1729)
        compiled, _, metadata = compile_inductor(graph, grad, mode, output / "code", f"{label}-{name}")
        setup[name] = metadata
        if compiled is not None:
            functions[name] = lambda compiled=compiled: compiled(grad)
    rows = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        grad.normal_()
        torch.manual_seed(seed + 1729)
        expected, indices = reference_graph(grad)
        assert_bytes(expected, independent_unique_reference(grad, indices, vocabulary, 0))
        for name, fn in functions.items():
            torch.manual_seed(seed + 1729)
            actual = fn()
            assert_bytes(actual, expected)
        plan = semantic.report()["nodes"][0]
        assert plan["selected_optimization"] == "unique_direct_write", plan
        names = list(functions)
        random.Random(seed + 4096).shuffle(names)
        ordered = {name: functions[name] for name in names}
        device = paired_device_times(ordered, seed=seed, **settings["device"])
        wall = paired_wall_times(ordered, seed=seed, **settings["wall"])
        # Verify the cached executables again with altered values after timings.
        grad.mul_(0.5)
        for fn in functions.values():
            torch.manual_seed(seed + 2718)
            actual = fn()
            torch.manual_seed(seed + 2718)
            _, indices = reference_graph(grad)
            assert_bytes(actual, independent_unique_reference(grad, indices, vocabulary, 0))
        row = {"seed": seed, "device": device, "wall": wall}
        rows.append(row)
        print(json.dumps({"case": label, "seed": seed, "device_us": device["median_us"], "wall_us": wall["median_us"]}),
              flush=True)
    medians = {
        metric: {name: statistics.median(row[metric]["median_us"][name]
                                         for row in rows)
                 for name in functions}
        for metric in ("device", "wall")
    }
    return {
        "shape_n_d_v": shape, "producer": producer, "frequency": frequency, "padding_idx": 0, "compiled_variants":
        setup, "semantic_plan": semantic.report()["nodes"][0], "seeds": rows, "median_us": medians,
        "semantic_speedup_vs": {
            metric:
            {name: latency / values["semantic_fx"]
             for name, latency in values.items()
             if name != "semantic_fx"}
            for metric, values in medians.items()
        }, "validation":
        "bitwise eager/compiler/reference equality for normal BF16; paired correctness RNG; changed gradients",
        "timing_scope":
        "output-only producer and operator graph; indices not returned; fresh separate functional allocations",
        "rng_scope": "paired correctness seeds; independent evolving RNG streams during timing"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "confirm"), default="confirm")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    smoke = args.mode == "smoke"
    settings = {
        "device": {"rounds": 3 if smoke else 15, "repetitions": 8 if smoke else 32, "warmup_ms": 20 if smoke else 200},
        "wall": {"rounds": 2 if smoke else 5, "repetitions": 3 if smoke else 10}
    }
    package = Path(triton.__file__).parent / "experimental/semantic_contracts"
    sources = [Path(__file__), Path(__file__).with_name("coverage_experiment.py"), *sorted(package.glob("*.py"))]
    torch_sources = [
        Path(torch.__file__).parent / relative
        for relative in ("_decomp/decompositions.py", "_inductor/decomposition.py", "_inductor/lowering.py")
    ]
    report = {
        "status": "running", "mode": args.mode, "torch": torch.__version__, "torch_revision": torch.version.git_version,
        "cuda": torch.version.cuda, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(),
        "compiler_sha256": digest(libtriton.__file__), "git_revision":
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "source_sha256":
        {str(path.resolve()): digest(path)
         for path in sources + torch_sources
         }, "settings": settings, "seeds": 1 if smoke else 3, "cases": [], "compiler_policy":
        "fullgraph=True,dynamic=False; default and max-autotune-no-cudagraphs; compile excluded from timing", "scope":
        "normal BF16, bounded static producer+embedding graphs; no model-level or strongest-provider claim",
        "artifact_policy": "JSON samples and bounded generated-module excerpts/hashes only"
    }
    shapes = [(256, 64, 256)] if smoke else [(4096, 128, 4096), (4096, 128, 8192), (16384, 256, 16384)]
    try:
        for shape in shapes:
            for producer in ("arange", "randperm"):
                for frequency in ((True, ) if smoke else (False, True)):
                    case = configuration(shape, producer, frequency, args.output, settings, report["seeds"])
                    report["cases"].append(case)
                    save_json(args.output / "results.json", report)
        failures = [
            variant for case in report["cases"] for variant in case["compiled_variants"].values()
            if variant["status"] != "compiled"
        ]
        report["status"] = "completed_with_compiler_failures" if failures else "passed"
        report["compiler_failure_count"] = len(failures)
    finally:
        save_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
