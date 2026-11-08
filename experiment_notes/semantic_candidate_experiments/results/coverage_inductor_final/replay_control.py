#!/usr/bin/env python3
"""Ordinary-call control for Inductor max-autotune with internal CUDA graphs."""
import io
import json
import logging
import statistics
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch
from torch._dynamo.utils import counters
from torch._inductor.cudagraph_trees import get_manager
from triton.experimental.semantic_contracts import compile as semantic_compile

EXPERIMENTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS))
from coverage_experiment import assert_bytes, digest, paired_wall_times, producer_graph, save_json  # noqa: E402


def capture_state():
    manager = get_manager(create_if_none_exists=False)
    if manager is None:
        return {"manager_exists": False, "recorded_root_graphs": 0}
    roots = [node for nodes in manager.roots.values() for node in nodes]
    return {
        "manager_exists": True, "recorded_root_graphs": sum(getattr(node, "graph", None) is not None for node in roots),
        "path_state": str(manager.path_state), "warmed_functions": len(manager.warmed_up_functions)
    }


def case(shape, producer):
    n, d, vocabulary = shape
    torch._dynamo.reset()
    torch.manual_seed(0)
    vector = torch.randn(d, device="cuda", dtype=torch.bfloat16)
    grad = vector.expand(n, d).clone()
    source = producer_graph(shape, producer, True)

    def graph(gradient):
        return source(gradient)[0]

    semantic = semantic_compile(graph, (grad, ), enabled=True, autotune=False)
    compiled = torch.compile(graph, backend="inductor", fullgraph=True, dynamic=False, mode="max-autotune")

    def replay():
        torch.compiler.cudagraph_mark_step_begin()
        return compiled(grad)

    functions = {
        "inductor_max_autotune_replay": replay, "semantic_fx": lambda: semantic(grad), "pytorch_eager":
        lambda: graph(grad)
    }
    start = time.perf_counter_ns()
    for function in functions.values():
        for _ in range(15):
            function()
        torch.cuda.synchronize()
    warmup_us = (time.perf_counter_ns() - start) / 1000
    after_warmup = capture_state()
    rows = []
    for seed in range(3):
        torch.manual_seed(seed)
        vector.normal_()
        grad.copy_(vector.expand(n, d))
        expected = grad.clone()
        expected[0].zero_()
        for function in functions.values():
            # Clone outside the compiled graph; release its output immediately
            # so later step markers cannot overwrite a retained graph output.
            actual = function().clone()
            assert_bytes(actual, expected)
            del actual
        wall = paired_wall_times(functions, seed=seed, rounds=5, repetitions=10)
        grad.mul_(0.5)
        expected.mul_(0.5)
        for function in functions.values():
            actual = function().clone()
            assert_bytes(actual, expected)
            del actual
        rows.append({"seed": seed, "wall": wall})
        print(json.dumps({"shape": shape, "producer": producer, "seed": seed, "wall_us": wall["median_us"]}),
              flush=True)
    medians = {name: statistics.median(row["wall"]["median_us"][name] for row in rows) for name in functions}
    return {
        "shape_n_d_v": shape, "producer": producer, "frequency": True, "status": "passed", "warmup_all_variants_us":
        warmup_us, "warmup_calls_per_variant": 15, "capture_after_warmup": after_warmup, "capture_after_measurement":
        capture_state(), "seeds": rows, "wall_median_us": medians, "semantic_speedup_vs_inductor_replay":
        medians["inductor_max_autotune_replay"] / medians["semantic_fx"], "validation":
        "independent full-coverage reference with nontrivial row-invariant BF16 gradients; changed gradients",
        "allocation": "functional outputs released each call; validation copies occur outside timing",
        "mark_step_policy": "cudagraph_mark_step_begin before every Inductor invocation; included in wall time"
    }


def main():
    output = Path(__file__).resolve().with_name("replay_control.json")
    if output.exists():
        raise FileExistsError(output)
    report = {
        "status": "running", "torch": torch.__version__, "torch_revision": torch.version.git_version, "cuda":
        torch.version.cuda, "gpu": torch.cuda.get_device_name(), "source_sha256": {
            str(Path(__file__).resolve()): digest(__file__),
            str(EXPERIMENTS / "coverage_experiment.py"): digest(EXPERIMENTS / "coverage_experiment.py")
        }, "mode": "torch.compile(fullgraph=True,dynamic=False,mode='max-autotune')", "scope":
        "ordinary calls with internal CUDA graph replay; no nested external capture; clocks unlocked", "limitations":
        "three bounded full-coverage frequency-scaled shapes; row-invariant gradients enable validation independently of replay RNG offsets",
        "cases": []
    }
    try:
        for shape, producer in [((4096, 128, 4096), "randperm"), ((16384, 256, 16384), "randperm"),
                                ((4096, 128, 4096), "arange")]:
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            logging.getLogger().addHandler(handler)
            before_skips = counters["inductor"]["cudagraph_skips"]
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = case(shape, producer)
                result["warnings"] = [str(w.message) for w in caught]
            except Exception as error:
                result = {
                    "shape_n_d_v": shape, "producer": producer, "status": "failed", "error": str(error), "traceback":
                    traceback.format_exc(limit=8)
                }
                if "illegal memory access" in str(error).lower() or "device-side assert" in str(error).lower():
                    raise
            finally:
                logging.getLogger().removeHandler(handler)
            result["logged_messages"] = stream.getvalue()[-16000:]
            result["cudagraph_skips_delta"] = counters["inductor"]["cudagraph_skips"] - before_skips
            report["cases"].append(result)
            save_json(output, report)
        report["status"] = "passed" if all(row["status"] == "passed"
                                           for row in report["cases"]) else "completed_with_failures"
    finally:
        save_json(output, report)


if __name__ == "__main__":
    main()
