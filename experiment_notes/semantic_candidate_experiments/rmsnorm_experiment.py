#!/usr/bin/env python3
"""AITER persistent RMSNorm: original, manual hoist, optional compiler contract.

Run --help without importing torch/Triton. Results are direct device-call
comparisons, not measurements of AITER's complete architecture-specific dispatch.
"""

import argparse
import copy
import hashlib
import json
import random
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["screen", "confirm", "validate"], default="screen")
    p.add_argument("--shapes", default="4096x1024,8192x1024,4096x4096,16384x4096")
    p.add_argument("--warps", default="4,8")
    p.add_argument("--grid-multipliers", default="1,2", help="Comma-separated SM multipliers; 'rows' means grid=M (one CTA per row)")
    p.add_argument("--seeds", default=None)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--operations", default="forward,backward")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--graph-repeats", type=int, default=32)
    p.add_argument("--warmup-ms", type=float, default=200)
    p.add_argument("--compiler-contract", action="store_true")
    p.add_argument("--select-best-from", type=Path,
                   help="Run only the union of per-variant winning configurations in a prior screen results.json")
    p.add_argument("--allow-failures", action="store_true")
    return p.parse_args()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def guarded(shape, dtype, fill=None):
    count = 1
    for d in shape:
        count *= d
    guard = 256
    storage = torch.full((count + 2 * guard,), 7.125, device="cuda", dtype=dtype)
    value = storage[guard:-guard].view(shape)
    if fill is not None:
        value.fill_(fill)
    return {"value": value, "storage": storage, "guard": guard}


def assert_guards(buffers):
    for name, item in buffers.items():
        g = item["guard"]
        assert bool((item["storage"][:g] == 7.125).all()), f"{name}: leading guard corrupted"
        assert bool((item["storage"][-g:] == 7.125).all()), f"{name}: trailing guard corrupted"


def validate_close(actual, expected, atol, rtol):
    expected = expected.to(actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    diff = (actual.double() - expected.double()).abs()
    budget = atol + rtol * expected.double().abs()
    return {"max_absolute_error": diff.max().item(), "max_error_over_tolerance": (diff / budget).max().item(), "atol": atol, "rtol": rtol}


def save_kernel(kernel, directory):
    directory.mkdir(parents=True, exist_ok=True)
    assembly = {}
    for ext in ["ttir", "ttgir", "llir", "ptx", "cubin"]:
        if ext not in kernel.asm:
            continue
        raw = kernel.asm[ext]
        if isinstance(raw, str):
            raw = raw.encode()
        (directory / ("kernel." + ext)).write_bytes(raw)
        assembly[ext] = {"sha256": sha256(raw), "size_bytes": len(raw)}
    ptx = kernel.asm.get("ptx", "")
    metrics = {
        "metadata": kernel.metadata._asdict(),
        "n_regs": getattr(kernel, "n_regs", None),
        "n_spills": getattr(kernel, "n_spills", None),
        "assembly": assembly,
        "static_ptx_global_load_sites": len(re.findall(r"\bld\.global(?:\.[\w]+)*\s", ptx)),
        "static_ptx_global_store_sites": len(re.findall(r"\bst\.global(?:\.[\w]+)*\s", ptx)),
        "static_ptx_global_async_copy_sites": len(re.findall(r"\bcp\.async[^\s]*\.global(?:\.[\w:]+)*\s", ptx)),
        "note": "Static instruction sites are not dynamic instruction counts; inspect loop placement in saved IR/PTX.",
    }
    write_json(directory / "metadata.json", metrics)
    return metrics


def contract_kernel(original, warps):
    attrs = copy.deepcopy(original.src.attrs)
    attrs[(2,)] = list(attrs.get((2,), [])) + [("tt.experimental_immutable", 1)]
    src = triton.compiler.ASTSource(original.src.fn, original.src.signature,
                                    constexprs=original.src.constants, attrs=attrs)
    return triton.compile(src, target=original.metadata.target,
                          options={"num_warps": warps})


def capture_graph(fn, repetitions):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            fn()
    return graph


def paired_timings(functions, args, seed):
    graphs = {name: capture_graph(fn, args.graph_repeats) for name, fn in functions.items()}
    deadline = time.perf_counter() + args.warmup_ms / 1000
    while time.perf_counter() < deadline:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    order_rng = random.Random(seed + 42819)
    timings = {name: [] for name in graphs}
    orders = []
    for _ in range(args.rounds):
        order = list(graphs)
        order_rng.shuffle(order)
        orders.append(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000 / args.graph_repeats)
    medians = {name: statistics.median(v) for name, v in timings.items()}
    return {"rounds_us": timings, "orders": orders, "median_us": medians,
            "speedup_over_original": {name: medians["original"] / value for name, value in medians.items()}}


def run_case(shape, warps, multiplier, seed, operation, args):
    m, n = shape
    # Upstream uses alignment promises on every row. Avoid invalid row strides.
    if n % 16:
        raise ValueError("Use n_cols divisible by 16; masked widths such as 1008 are supported.")
    torch.manual_seed(seed)
    dtype = getattr(torch, args.dtype)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    grid = m if multiplier == "rows" else max(1, int(sms * multiplier))
    block = triton.next_power_of_2(n)
    name = f"{operation}-m{m}-n{n}-w{warps}-grid{grid}-seed{seed}"
    directory = args.output / name
    directory.mkdir(parents=True, exist_ok=False)
    x = torch.randn((m, n), device="cuda", dtype=dtype)
    gamma = torch.randn((n,), device="cuda", dtype=dtype)
    dy = torch.randn_like(x)
    epsilon = 1e-6
    # Backward variants share saved FP32 statistics, independent of variant timing.
    rsigma = torch.rsqrt(x.float().square().mean(-1) + epsilon)
    snapshots = {"x": x.clone(), "gamma": gamma.clone(), "dy": dy.clone(), "rsigma": rsigma.clone()}
    variants = ["original", "hoisted"] + (["compiler"] if args.compiler_contract else [])
    outputs = {}
    compiled = {}
    final_compiled = {}
    functions = {}
    main_functions = {}
    launch_arguments = {}
    metrics = {}
    for variant in variants:
        if operation == "forward":
            buffers = {"y": guarded((m, n), dtype, float("nan")), "rsigma": guarded((m,), torch.float32, float("nan"))}
            launch_args = [x, buffers["y"]["value"], gamma, buffers["rsigma"]["value"], n, n, m, n, epsilon, block, False]
            fn = getattr(kernels, "rmsnorm_forward_" + ("original" if variant == "compiler" else variant))
        else:
            buffers = {"dx": guarded((m, n), dtype, float("nan")), "dg_tmp": guarded((grid, n), torch.float32, float("nan")), "dg": guarded((n,), dtype, float("nan"))}
            launch_args = [dy, x, gamma, rsigma, buffers["dx"]["value"], buffers["dg_tmp"]["value"], n, n, m, n, block, False, grid]
            fn = getattr(kernels, "rmsnorm_backward_" + ("original" if variant == "compiler" else variant))
        if variant == "compiler":
            kernel = contract_kernel(compiled["original"], warps)
        else:
            kernel = fn.warmup(*launch_args, grid=(grid,), num_warps=warps)
        launcher = kernel[(grid, 1, 1)]
        # CudaLauncher removes constexpr arguments internally using the signature.
        main_fn = lambda launcher=launcher, launch_args=launch_args: launcher(*launch_args)
        main_fn()
        compiled[variant] = kernel
        launch_arguments[variant] = launch_args
        main_functions[variant] = main_fn
        outputs[variant] = buffers
        if operation == "backward":
            reduce_args = [buffers["dg_tmp"]["value"], buffers["dg"]["value"], n, grid, n, 128, 64]
            final = kernels.rmsnorm_dg_reduce.warmup(*reduce_args, grid=(triton.cdiv(n, 64),), num_warps=4)
            reduce_launcher = final[(triton.cdiv(n, 64), 1, 1)]
            def full_fn(main_fn=main_fn, reduce_launcher=reduce_launcher, reduce_args=reduce_args):
                main_fn()
                reduce_launcher(*reduce_args)
            functions[variant] = full_fn
            final_compiled[variant] = final
            full_fn()
        else:
            functions[variant] = main_fn
        metrics[variant] = save_kernel(kernel, directory / variant / "main")
        if variant in final_compiled:
            save_kernel(final_compiled[variant], directory / variant / "reduction")
    torch.cuda.synchronize()
    # FP64 oracle with direct cast to the requested output dtype.
    xd, gd, dyd = x.double(), gamma.double(), dy.double()
    if operation == "forward":
        rd = torch.rsqrt(xd.square().mean(-1) + epsilon)
        reference = {"y": xd * rd[:, None] * gd[None, :], "rsigma": rd}
    else:
        rd = rsigma.double()
        grad_sum = (dyd * xd * gd[None, :]).sum(-1)
        reference = {"dx": dyd * rd[:, None] * gd[None, :] - rd[:, None].pow(3) * xd * (grad_sum / n)[:, None],
                     "dg": (dyd * xd * rd[:, None]).sum(0)}
    validation = {}
    def validate(phase):
        rows = {}
        for variant, buffers in outputs.items():
            assert_guards(buffers)
            checks = {}
            for output_name, expected in reference.items():
                actual = buffers[output_name]["value"]
                if output_name == "rsigma":
                    atol, rtol = 2e-6, 2e-6
                elif dtype == torch.float32:
                    atol, rtol = (2e-3, 2e-4) if output_name == "dg" else (2e-5, 2e-5)
                else:
                    atol, rtol = (0.125, 0.02) if output_name == "dg" else (0.02, 0.02)
                checks[output_name] = validate_close(actual, expected, atol, rtol)
            for output_name in buffers:
                actual_bytes = buffers[output_name]["value"].view(torch.uint8)
                original_bytes = outputs["original"][output_name]["value"].view(torch.uint8)
                assert torch.equal(actual_bytes, original_bytes), f"{phase}: {variant}/{output_name} differs bitwise from original"
            rows[variant] = checks
        for key, tensor in [("x", x), ("gamma", gamma), ("dy", dy), ("rsigma", rsigma)]:
            assert torch.equal(tensor, snapshots[key]), f"Input {key} changed"
        validation[phase] = {"reference": rows, "all_variants_exact_equal": True, "all_variants_bitwise_equal": True, "guards_passed": True, "inputs_unchanged": True}
    validate("before_timing")
    # Keep correctness outputs independent. All measured variants instead write
    # these exact same addresses, including backward's intermediate reduction.
    timing_buffers = {key: guarded(tuple(item["value"].shape), item["value"].dtype, float("nan"))
                      for key, item in outputs["original"].items()}
    timing_functions = {}
    timing_main_functions = {}
    for variant in variants:
        timing_args = list(launch_arguments[variant])
        if operation == "forward":
            timing_args[1] = timing_buffers["y"]["value"]
            timing_args[3] = timing_buffers["rsigma"]["value"]
        else:
            timing_args[4] = timing_buffers["dx"]["value"]
            timing_args[5] = timing_buffers["dg_tmp"]["value"]
        launcher = compiled[variant][(grid, 1, 1)]
        timing_main = lambda launcher=launcher, timing_args=timing_args: launcher(*timing_args)
        timing_main_functions[variant] = timing_main
        if operation == "backward":
            reduce_args = [timing_buffers["dg_tmp"]["value"], timing_buffers["dg"]["value"], n, grid, n, 128, 64]
            reduce_launcher = final_compiled[variant][(triton.cdiv(n, 64), 1, 1)]
            def timing_full(timing_main=timing_main, reduce_launcher=reduce_launcher, reduce_args=reduce_args):
                timing_main()
                reduce_launcher(*reduce_args)
            timing_functions[variant] = timing_full
        else:
            timing_functions[variant] = timing_main

    def validate_timing_buffers(phase):
        checked = []
        for variant in variants:
            for item in timing_buffers.values():
                item["value"].fill_(float("nan"))
            timing_functions[variant]()
            torch.cuda.synchronize()
            assert_guards(timing_buffers)
            for output_name, item in timing_buffers.items():
                expected = outputs["original"][output_name]["value"]
                assert torch.equal(item["value"].view(torch.uint8), expected.view(torch.uint8)), f"{phase}: shared {variant}/{output_name} differs bitwise from independently reference-validated output"
            checked.append(variant)
        validation[phase] = {"variants_checked_individually": checked, "all_outputs_bitwise_equal_to_reference_validated_baseline": True,
                             "guards_passed": True, "fresh_poison_before_each_variant": True}

    validate_timing_buffers("shared_outputs_before_timing")
    result = {"case": name, "operation": operation, "shape": [m, n], "warps": warps, "grid": grid,
              "grid_multiplier": multiplier, "seed": seed, "dtype": args.dtype, "kernels": metrics,
              "validation": validation,
              "timing_allocation": {"shared_outputs_across_variants": True,
                                    "shared_buffer_names": list(timing_buffers),
                                    "separate_correctness_outputs": True,
                                    "note": "Every measured variant writes identical output and intermediate addresses; independent variant buffers are used only for correctness."},
              "published_wrapper_uses_persistent_path": not (m > 8192 and n <= (2048 if operation == "forward" else 1024)),
              "wrapper_configuration_comparison": {
                  "non_gfx950_default_grid": min(m, sms),
                  "default_main_warps": 4 if operation == "forward" else 8,
                  "matches_non_gfx950_default_launch": grid == min(m, sms) and warps == (4 if operation == "forward" else 8),
                  "note": "AMD-origin kernels portably compiled on NVIDIA; direct extracted calls bypass architecture-specific provider dispatch. gfx950 forward oversubscribes up to 4x SMs and backward reduction uses 8 warps."},
              "scope": "Direct extracted persistent kernels, USE_BLOCKED=False, not full provider dispatch. Forward includes Y/rsigma; backward total includes DX and final dgamma reduction. Allocation and forward-statistics preparation excluded."}
    if args.mode != "validate":
        result["total"] = paired_timings(timing_functions, args, seed)
        if operation == "backward":
            result["main_only"] = paired_timings(timing_main_functions, args, seed + 1)
        validate_timing_buffers("shared_outputs_after_timing")
        for fn in functions.values():
            fn()
        torch.cuda.synchronize()
        validate("after_timing")
    for buffers in outputs.values():
        for item in buffers.values():
            item["value"].fill_(float("nan"))
    for fn in functions.values():
        fn()
    torch.cuda.synchronize()
    validate("fresh_poisoned_outputs")
    result["status"] = "passed"
    write_json(directory / "results.json", result)
    print(json.dumps({"case": name, "status": "passed", "total": result.get("total", {}).get("median_us")}), flush=True)
    return result


def summarize(cases):
    grouped = {}
    for case in cases:
        if case.get("status") != "passed" or "total" not in case:
            continue
        key = (case["operation"], *case["shape"], case["warps"], case["grid"])
        grouped.setdefault(key, []).append(case)
    configurations = []
    for key, runs in grouped.items():
        medians = {name: statistics.median(r["total"]["median_us"][name] for r in runs) for name in runs[0]["total"]["median_us"]}
        configurations.append({"operation": key[0], "shape": list(key[1:3]), "warps": key[3], "grid": key[4], "seed_count": len(runs), "median_of_seed_medians_us": medians,
                               "speedup_over_original": {name: medians["original"] / value for name, value in medians.items()}})
    best = {}
    for item in configurations:
        key = item["operation"] + "-" + "x".join(map(str, item["shape"]))
        current = best.setdefault(key, {})
        for name, latency in item["median_of_seed_medians_us"].items():
            if name not in current or latency < current[name]["latency_us"]:
                current[name] = {"latency_us": latency, "warps": item["warps"], "grid": item["grid"]}
    return {"configurations": configurations, "best_tested_by_variant": best}


def main():
    args = arguments()
    if args.output.exists():
        raise SystemExit("--output must be a new directory")
    args.output.mkdir(parents=True)
    global torch, triton, kernels
    import torch
    import triton
    import rmsnorm_kernels as kernels
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    args.rounds = args.rounds or (15 if args.mode == "confirm" else 7)
    seeds = [int(s) for s in (args.seeds or ("0,1,2" if args.mode == "confirm" else "0")).split(",")]
    shapes = [tuple(map(int, s.lower().split("x"))) for s in args.shapes.split(",")]
    selected = None
    selection_info = None
    if args.select_best_from:
        prior = json.loads(args.select_best_from.read_text())
        selected = set()
        for key, winners in prior["summary"]["best_tested_by_variant"].items():
            operation, dimensions = key.split("-", 1)
            m, n = map(int, dimensions.split("x"))
            selected.update((operation, m, n, item["warps"], item["grid"]) for item in winners.values())
        selection_info = {"prior_results": str(args.select_best_from), "prior_results_sha256": sha256(args.select_best_from.read_bytes()),
                          "rule": "Union of each variant's lowest median screen configuration for each operation/shape", "selected_configurations": sorted(selected)}
    report = {"arguments": vars(args), "environment": {"torch": torch.__version__, "triton": triton.__version__, "triton_path": triton.__file__, "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(), "python": sys.executable},
              "provenance": json.loads((ROOT / "vendor/aiter/provenance.json").read_text()),
              "harness_sha256": sha256(Path(__file__).read_bytes()),
              "kernel_module_sha256": sha256((ROOT / "rmsnorm_kernels.py").read_bytes()), "selection": selection_info,
              "methodology": {"timing": "Shuffled interleaved CUDA graph rounds, median per seed; clocks unlocked", "shared_timing_outputs": True,
                              "allocation": "One common set of timed output/intermediate buffers for all variants; separate correctness buffers; each rebound variant validated before capture and after timing",
                              "graph_repeats": args.graph_repeats, "warmup_ms": args.warmup_ms, "numerics": "Exact paired outputs, FP64 oracle, selected tolerances recorded for every output; not a universal application accuracy contract"}, "cases": []}
    try:
        for operation in args.operations.split(","):
            if operation not in ["forward", "backward"]:
                raise ValueError(operation)
            for shape in shapes:
                for warps in map(int, args.warps.split(",")):
                    for multiplier_text in args.grid_multipliers.split(","):
                        multiplier = "rows" if multiplier_text == "rows" else float(multiplier_text)
                        grid = shape[0] if multiplier == "rows" else max(1, int(torch.cuda.get_device_properties(0).multi_processor_count * multiplier))
                        if selected is not None and (operation, *shape, warps, grid) not in selected:
                            continue
                        for seed in seeds:
                            try:
                                case = run_case(shape, warps, multiplier, seed, operation, args)
                            except Exception:
                                case = {"status": "failed", "operation": operation, "shape": shape, "warps": warps, "grid_multiplier": multiplier, "seed": seed, "error": traceback.format_exc()}
                                print(case["error"], file=sys.stderr, flush=True)
                                report["cases"].append(case)
                                if not args.allow_failures:
                                    raise
                            else:
                                report["cases"].append(case)
                            report["summary"] = summarize(report["cases"])
                            write_json(args.output / "results.json", report)
    finally:
        report["summary"] = summarize(report["cases"])
        write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
