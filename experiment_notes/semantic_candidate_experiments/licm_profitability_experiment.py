#!/usr/bin/env python3
"""Controlled persistent RMSNorm LICM study; no compiler changes.

Examples (from the repository root, using the semantic experiment venv)::

  python experiment_notes/semantic_candidate_experiments/licm_profitability_experiment.py --output /tmp/licm-study
  ncu --section Occupancy --section SchedulerStats --section MemoryWorkloadAnalysis \
      --kernel-name regex:rmsnorm_forward_original --launch-count 1 --csv \
      python experiment_notes/semantic_candidate_experiments/licm_profitability_experiment.py \
      --output /tmp/licm-study --profile --shape 16384x4096 --grid-multiplier 32 --variant compiler

The immutable gamma contract and extracted AITER source are unchanged. All timed
variants share input/output addresses. Training and held-out cases are fixed
before measurement. A proposed static rule is an experiment, not a guarantee.
"""

import argparse
import ctypes
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--graph-repeats", type=int, default=64)
    p.add_argument("--warmup-ms", type=float, default=100)
    p.add_argument("--split", choices=["all", "training", "heldout"], default="all")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--verify-guard-wrapper", action="store_true")
    p.add_argument("--profile-repeats", type=int, default=6,
                   help="Use ncu --launch-skip 5 --launch-count 1 for a warmed launch")
    p.add_argument("--shape", default="16384x4096")
    p.add_argument("--grid-multiplier", type=int, default=32)
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--variant", choices=["original", "compiler"], default="original")
    return p.parse_args()


class CudaResidency:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self.lib.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        self.device = {name: self.attribute(number) for name, number in {
            "sm_count": 16, "warp_size": 10, "max_threads_per_sm": 39,
            "max_shared_bytes_per_sm": 81, "max_registers_per_sm": 82,
            "max_blocks_per_sm": 106}.items()}

    def attribute(self, number):
        value = ctypes.c_int()
        status = self.lib.cuDeviceGetAttribute(ctypes.byref(value), number, 0)
        if status:
            raise RuntimeError(f"cuDeviceGetAttribute({number}) returned {status}")
        return value.value

    def kernel(self, kernel):
        kernel._init_handles()
        attrs = {}
        for name, number in {"static_shared_bytes": 1, "local_bytes": 3, "registers_per_thread": 4}.items():
            value = ctypes.c_int()
            status = self.lib.cuFuncGetAttribute(ctypes.byref(value), number, ctypes.c_void_p(kernel.function))
            if status:
                raise RuntimeError(f"cuFuncGetAttribute returned {status}")
            attrs[name] = value.value
        threads = kernel.metadata.num_warps * self.device["warp_size"]
        for name, dynamic in [("active_blocks_per_sm", kernel.metadata.shared),
                              ("active_blocks_per_sm_zero_dynamic_shared", 0)]:
            value = ctypes.c_int()
            status = self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(value), ctypes.c_void_p(kernel.function), threads, dynamic)
            if status:
                raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor returned {status}")
            attrs[name] = value.value
        attrs.update(threads_per_block=threads, dynamic_shared_bytes=kernel.metadata.shared,
                     n_spills=kernel.n_spills,
                     theoretical_active_warps_per_sm=attrs["active_blocks_per_sm"] * kernel.metadata.num_warps,
                     theoretical_occupancy_fraction=attrs["active_blocks_per_sm"] * threads / self.device["max_threads_per_sm"])
        return attrs


def artifact(kernel, args, residency):
    key = hashlib.sha256(kernel.asm["cubin"]).hexdigest()
    directory = args.output / "kernels" / key[:16]
    if not directory.exists():
        metrics = old.save_kernel(kernel, directory)
        result = subprocess.run(["nvdisasm", str(directory / "kernel.cubin")], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        (directory / "kernel.sass").write_text(result.stdout)
        ptx = kernel.asm["ptx"]
        metrics.update(residency=residency.kernel(kernel),
                       sass_sha256=digest(directory / "kernel.sass"),
                       static_ptx_bar_sync_sites=len(re.findall(r"\bbar\.sync\b", ptx)),
                       static_ptx_cp_async_wait_sites=len(re.findall(r"\bcp\.async\.wait", ptx)),
                       static_sass_barrier_sites=len(re.findall(r"\bBAR\.", result.stdout)),
                       static_sass_ldgsts_sites=len(re.findall(r"\bLDGSTS\b", result.stdout)),
                       static_sass_ldg_sites=len(re.findall(r"\bLDG\.", result.stdout)))
        write(directory / "metadata.json", metrics)
    return {"artifact": str(directory.relative_to(args.output)),
            **json.loads((directory / "metadata.json").read_text())}


def cases():
    result = []
    for m in [4096, 16384]:
        for multiplier in [4, 16, 32]:
            for warps in [4, 8]:
                result.append(("training", m, 4096, warps, multiplier))
    for m in [8192, 32768]:
        for multiplier in [4, 16, 32]:
            result.append(("heldout", m, 4096, 4, multiplier))
    for m in [4096, 16384]:
        for multiplier in [4, 32]:
            result.append(("heldout", m, 2048, 4, multiplier))
    return result


def compile_pair(launch_args, warps, grid):
    baseline = kernels.rmsnorm_forward_original.warmup(*launch_args, grid=(grid,), num_warps=warps)
    return {"original": baseline, "compiler": old.contract_kernel(baseline, warps)}


def measure(config, seed, args, residency):
    split, m, n, warps, multiplier = config
    sms = residency.device["sm_count"]
    grid = multiplier * sms
    name = f"{split}-m{m}-n{n}-w{warps}-grid{grid}-seed{seed}"
    path = args.output / "cases" / (name + ".json")
    if path.exists():
        return json.loads(path.read_text())
    torch.manual_seed(seed)
    x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    input_snapshots = [x.clone(), gamma.clone()]
    buffers = {"y": old.guarded((m, n), torch.bfloat16, float("nan")),
               "rsigma": old.guarded((m,), torch.float32, float("nan"))}
    y, rsigma = buffers["y"]["value"], buffers["rsigma"]["value"]
    launch_args = [x, y, gamma, rsigma, n, n, m, n, 1e-6, triton.next_power_of_2(n), False]
    compiled = compile_pair(launch_args, warps, grid)
    functions = {}
    metrics = {}
    for variant, kernel in compiled.items():
        launcher = kernel[(grid, 1, 1)]
        functions[variant] = lambda launcher=launcher: launcher(*launch_args)
        metrics[variant] = artifact(kernel, args, residency)
    functions["original"]()
    expected_y, expected_r = y.clone(), rsigma.clone()
    # Full FP64 oracle, with output tolerances inherited from the frozen study.
    xd = x.double()
    ref_r = torch.rsqrt(xd.square().mean(-1) + 1e-6)
    reference_checks = {"y": old.validate_close(y, xd * ref_r[:, None] * gamma.double()[None, :], .02, .02),
                        "rsigma": old.validate_close(rsigma, ref_r, 2e-6, 2e-6)}
    del xd, ref_r

    def validate():
        for variant, fn in functions.items():
            y.fill_(float("nan"))
            rsigma.fill_(float("nan"))
            fn()
            old.assert_guards(buffers)
            assert torch.equal(y.view(torch.uint8), expected_y.view(torch.uint8)), variant
            assert torch.equal(rsigma.view(torch.uint8), expected_r.view(torch.uint8)), variant
        assert torch.equal(x, input_snapshots[0]) and torch.equal(gamma, input_snapshots[1])

    validate()
    timing = old.paired_timings(functions, args, seed + 100003 * m + multiplier)
    validate()
    resources = {v: a["residency"] for v, a in metrics.items()}
    active_ctas = min(m, grid)
    work = {"active_ctas": active_ctas, "launched_ctas": grid,
            "nonempty_ctas_per_sm_average": active_ctas / sms,
            "rows_per_nonempty_cta_mean": m / active_ctas,
            "loop_trip_count_min": m // grid, "loop_trip_count_max": math.ceil(m / grid),
            "logical_input_output_bytes": 4 * m * n + 4 * m,
            "logical_gamma_bytes_original": 2 * m * n,
            "logical_gamma_bytes_compiler": 2 * active_ctas * n,
            "logical_gamma_bytes_saved": 2 * (m - active_ctas) * n,
            "note": "Logical element traffic, not measured DRAM/cache transactions; invariant gamma can be cached. Zero-trip CTAs excluded from gamma hoist."}
    for v, r in resources.items():
        work[v] = {"effective_average_cta_residency_upper_bound": min(active_ctas / sms, r["active_blocks_per_sm"]),
                   "grid_waves_at_theoretical_residency": grid / (sms * r["active_blocks_per_sm"])}
    # Predeclared structural rule. Both transformed and baseline resource queries
    # are needed; this is not yet implementable by an early TTIR pass alone.
    base, moved = resources["original"], resources["compiler"]
    grid_limited = active_ctas / sms <= moved["active_blocks_per_sm"]
    no_residency_loss = moved["active_blocks_per_sm"] >= base["active_blocks_per_sm"]
    accept = moved["n_spills"] == 0 and (no_residency_loss or grid_limited)
    result = {"case": name, "split": split, "shape": [m, n], "warps": warps, "grid": grid,
              "seed": seed, "kernels": {v: a["artifact"] for v, a in metrics.items()},
              "resources": resources, "work": work, "timing": timing,
              "guard": {"accept": accept, "grid_limited_after_hoist": grid_limited,
                        "no_residency_loss": no_residency_loss},
              "validation": {"full_fp64_oracle": reference_checks, "bitwise_equal_all_variants": True,
                             "shared_outputs": True, "guards_passed": True, "inputs_unchanged": True,
                             "fresh_poison_each_variant_before_and_after_timing": True}}
    write(path, result)
    print(json.dumps({"case": name, "median_us": timing["median_us"], "guard": accept}), flush=True)
    return result


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["split"], *r["shape"], r["warps"], r["grid"])
        groups.setdefault(key, []).append(r)
    result = []
    for key, group in sorted(groups.items()):
        speeds = [r["timing"]["speedup_over_original"]["compiler"] for r in group]
        median = statistics.median(speeds)
        accept = group[0]["guard"]["accept"]
        result.append({"split": key[0], "shape": list(key[1:3]), "warps": key[3], "grid": key[4],
                       "seeds": [r["seed"] for r in group], "seed_speedups": speeds,
                       "median_speedup": median, "guard_accept": accept,
                       "guarded_speedup": median if accept else 1.,
                       "material_regression": median < .98,
                       "missed_material_improvement": not accept and median > 1.02,
                       "resources": group[0]["resources"], "work": group[0]["work"]})
    return {"configurations": result,
            "guard_definition": "accept iff no spills and (CUDA API blocks/SM does not decrease OR nonempty CTAs/SM <= transformed blocks/SM)",
            "guard_predeclared": True,
            "static_guard_limitation": "This resource/grid rule does not guarantee profitability. It requires compiling both variants and runtime launch/shape knowledge. Existing FX choose_hoist already measures a 2% timing margin.",
            "split_totals": {split: {"configurations": sum(r["split"] == split for r in result),
                                     "accepted_regressions": sum(r["split"] == split and r["guard_accept"] and r["material_regression"] for r in result),
                                     "declined_gains": sum(r["split"] == split and r["missed_material_improvement"] for r in result)}
                             for split in ["training", "heldout"]}}


def verify_guard_wrapper(args, residency):
    """Check the implemented host selector on measured and held-out workloads."""
    from triton.experimental.semantic_contracts.native import compile_immutable_licm
    import triton.experimental.semantic_contracts.native as native
    import triton._C.libtriton as libtriton
    checks = []
    for m, n, multiplier, expected in [(4096, 4096, 4, "optimized"), (16384, 4096, 32, "baseline"),
                                       (8192, 4096, 4, "optimized"), (32768, 4096, 32, "baseline"),
                                       (4096, 2048, 32, "baseline")]:
        torch.manual_seed(42)
        x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
        gamma = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
        buffers = {"y": old.guarded((m, n), torch.bfloat16, float("nan")),
                   "rsigma": old.guarded((m,), torch.float32, float("nan"))}
        y, r = buffers["y"]["value"], buffers["rsigma"]["value"]
        call_args = (x, y, gamma, r, n, n, m, n, 1e-6, triton.next_power_of_2(n), False)
        grid = (multiplier * residency.device["sm_count"],)
        plan = compile_immutable_licm(kernels.rmsnorm_forward_original, call_args, grid=grid,
                                      immutable_args=("g_ptr",), options={"num_warps": 4})
        report = plan.report()
        assert report["selected"] == expected, report
        plan.baseline[plan.grid](*call_args)
        expected_y, expected_r = y.clone(), r.clone()
        functions = {}
        for name, kernel in [("original", plan.baseline), ("compiler", plan.optimized)]:
            launcher = kernel[plan.grid]
            functions[name] = lambda launcher=launcher: launcher(*call_args)
        functions["selected"] = lambda: plan(*call_args)
        for fn in functions.values():
            y.fill_(float("nan"))
            r.fill_(float("nan"))
            fn()
            assert torch.equal(y.view(torch.uint8), expected_y.view(torch.uint8))
            assert torch.equal(r.view(torch.uint8), expected_r.view(torch.uint8))
            old.assert_guards(buffers)
        timing = old.paired_timings(functions, args, 42)
        fingerprints = {name: hashlib.sha256(kernel.asm["cubin"]).hexdigest()
                        for name, kernel in [("baseline", plan.baseline), ("optimized", plan.optimized)]}
        checks.append({"shape": [m, n], "grid": grid, "report": report, "timing": timing,
                       "cubins": fingerprints, "bitwise_validation_passed": True, "shared_outputs": True})
        print(json.dumps({"wrapper_shape": [m, n], "grid": grid, "selected": report["selected"],
                          "timing_us": timing["median_us"]}), flush=True)
    write(args.output / "wrapper_validation.json", {
        "checks": checks, "native_wrapper_sha256": digest(native.__file__),
        "harness_sha256": digest(__file__), "libtriton_sha256": digest(libtriton.__file__),
        "scope": "Five integration cases for actual compiled selector. Main study uses66 cases/3seeds; these single-seed timings only verify wrapper selection and no host launch removal."})


def main():
    global torch, triton, old, kernels
    args = arguments()
    import torch
    import triton
    import triton._C.libtriton as libtriton
    import rmsnorm_experiment as old
    import rmsnorm_kernels as kernels
    old.torch, old.triton = torch, triton
    torch.cuda.init()
    residency = CudaResidency()
    if args.verify_guard_wrapper:
        verify_guard_wrapper(args, residency)
        return
    if args.profile:
        m, n = map(int, args.shape.split("x"))
        grid = args.grid_multiplier * residency.device["sm_count"]
        x = torch.ones((m, n), device="cuda", dtype=torch.bfloat16)
        y, gamma = torch.empty_like(x), torch.ones((n,), device="cuda", dtype=torch.bfloat16)
        r = torch.empty((m,), device="cuda")
        launch_args = [x, y, gamma, r, n, n, m, n, 1e-6, triton.next_power_of_2(n), False]
        pair = compile_pair(launch_args, args.warps, grid)
        for _ in range(args.profile_repeats):
            pair[args.variant][(grid, 1, 1)](*launch_args)
        torch.cuda.synchronize()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprints = {"harness": digest(__file__), "frozen_harness": digest(old.__file__),
                    "extracted_aiter_kernels": digest(kernels.__file__), "libtriton": digest(libtriton.__file__)}
    environment = {"torch": torch.__version__, "triton": triton.__version__, "triton_path": triton.__file__,
                   "python": sys.executable, "device": torch.cuda.get_device_name(),
                   "cuda_device_attributes": residency.device, "fingerprints": fingerprints,
                   "aiter_provenance": json.loads((ROOT / "vendor/aiter/provenance.json").read_text()),
                   "scope": "Direct pinned AITER BF16 forward persistent kernel, USE_BLOCKED=False; not stock provider dispatch or global tuning.",
                   "protocol": vars(args)}
    previous = args.output / "environment.json"
    if previous.exists() and json.loads(previous.read_text())["fingerprints"] != fingerprints:
        raise RuntimeError("Output directory contains measurements from different source/compiler fingerprints; use a new output directory")
    write(args.output / "environment.json", environment)
    configs = [c for c in cases() if args.split == "all" or c[0] == args.split]
    write(args.output / "predeclared_plan.json", {"cases": configs, "guard": "no spills and (no blocks/SM residency loss or active CTAs/SM <= transformed blocks/SM)",
                                                 "seeds": args.seeds, "rounds": args.rounds,
                                                 "note": "The fixed structural guard and held-out configurations were written before GPU measurements."})
    rows = []
    for seed in map(int, args.seeds.split(",")):
        order = list(configs)
        random.Random(47291 + seed).shuffle(order)
        for config in order:
            rows.append(measure(config, seed, args, residency))
            write(args.output / "analysis.json", summarize(rows))
    if fingerprints["libtriton"] != digest(libtriton.__file__):
        raise RuntimeError("Native compiler changed during measurement")
    write(args.output / "analysis.json", summarize(rows))


if __name__ == "__main__":
    main()
