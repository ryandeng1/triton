#!/usr/bin/env python3
"""Cross-kernel controls for contract LICM and automatic loop pipelining.

All transformed variants use identical allocation contracts and original kernel
bodies. Contracts-only is the performance denominator; uncontracted source is a
separate control because source-requested pipelines may require those facts.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import random
import statistics
import time


HERE = Path(__file__).resolve().parent
VARIANTS = {
    "no_contract": (False, 0),
    "contracts": (False, 0),
    "licm": (True, 0),
    "pipeline": (False, 2),
    "both": (True, 2),
}
ADAPTERS = {
    "rms": "cross_pipeline_rms",
    "flaggems": "cross_kernel_flaggems_adapters",
    "upsample": "cross_pipeline_upsample",
}
BASE_OPTIONS = dict(enable_indexed_update_specialization=False,
                    enable_private_atomic_relaxation=False,
                    enable_private_reduction_ownership=False,
                    enable_contract_load_reuse=False,
                    enable_contract_licm=False, contract_pipeline_stages=0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def allocation_contract(args, groups):
    import torch
    tensors = {i: x for i, x in enumerate(args) if isinstance(x, torch.Tensor)}
    assert set(tensors) == set(groups), (set(tensors), set(groups))
    ranges = {}
    for i, tensor in tensors.items():
        storage = tensor.untyped_storage()
        ranges[i] = (storage.data_ptr(), storage.data_ptr() + storage.nbytes())
    for i, (start, end) in ranges.items():
        for j, (other_start, other_end) in ranges.items():
            if i < j and groups[i] != groups[j]:
                assert end <= other_start or other_end <= start, (i, j)
    return True


def resources(kernel):
    return dict(registers=kernel.n_regs, spills=kernel.n_spills,
                shared=kernel.metadata.shared,
                async_copies=kernel.asm["ttgir"].count("ttg.async_copy_global_to_local"),
                loops=kernel.asm["ttgir"].count("scf.for"),
                ptx_sha256=hashlib.sha256(kernel.asm["ptx"].encode()).hexdigest(),
                cubin_sha256=hashlib.sha256(kernel.asm["cubin"]).hexdigest())


def compile_case(adapter, case, data, config):
    import triton
    from triton.compiler import ASTSource
    spec = adapter.launch_spec(case, data, config)
    args = spec["args"]
    grid = tuple(spec["grid"]) + (1,) * (3 - len(spec["grid"]))
    groups = spec["groups"]
    allocation_contract(args, groups)
    options = dict(spec.get("options", {}), **BASE_OPTIONS)
    start = time.monotonic()
    original = spec["fn"].warmup(*args, grid=grid, **options)
    attrs = copy.deepcopy(original.src.attrs)
    assert "tt.experimental_" not in str(attrs)
    for i, group in groups.items():
        attrs.setdefault((i,), []).append(("tt.experimental_alias_group", group))
    kernels = {"no_contract": original}
    for name, (licm, stages) in VARIANTS.items():
        if name == "no_contract":
            continue
        source = ASTSource(original.src.fn, original.src.signature,
                           constexprs=original.src.constants, attrs=attrs)
        kernels[name] = triton.compile(
            source, target=original.metadata.target,
            options=dict(options, enable_contract_licm=licm,
                         contract_pipeline_stages=stages))
    tail = []
    for extra in spec.get("trailing_specs", []):
        opts = dict(extra.get("options", {}), **BASE_OPTIONS)
        extra_grid = tuple(extra["grid"]) + (1,) * (3 - len(extra["grid"]))
        kernel = extra["fn"].warmup(*extra["args"], grid=extra_grid, **opts)
        tail.append((kernel, extra_grid, extra["args"]))
    functions = {}
    for name, kernel in kernels.items():
        kernel._init_handles()
        launcher = kernel[grid]
        trailing = [(k[g], a) for k, g, a in tail]

        def invoke(launcher=launcher, args=args, trailing=trailing):
            launcher(*args)
            for final, final_args in trailing:
                final(*final_args)

        functions[name] = invoke
    return dict(kernels=kernels, functions=functions,
                resources={name: resources(k) for name, k in kernels.items()},
                trailing_resources=[resources(k) for k, _, _ in tail],
                config=config, grid=grid, options=options,
                groups=groups, compile_seconds=time.monotonic() - start,
                no_contract_matches_contracts_ptx=original.asm["ptx"] == kernels["contracts"].asm["ptx"],
                pipeline_changes_contracts_ptx=kernels["pipeline"].asm["ptx"] != kernels["contracts"].asm["ptx"],
                pipeline_changes_licm_ptx=kernels["both"].asm["ptx"] != kernels["licm"].asm["ptx"])


def validate(adapter, case, data, build):
    import torch
    records = {}
    baseline = None
    # Compare all variants to the same contracted, no-new-transform baseline.
    for name in ("contracts", "no_contract", "licm", "pipeline", "both"):
        adapter.reset(case, data)
        build["functions"][name]()
        torch.cuda.synchronize()
        checks = adapter.validate(case, data)
        outputs = adapter.output_tensors(data)
        if baseline is None:
            baseline = {key: value.clone() for key, value in outputs.items()}
        exact = {key: bool(torch.equal(value.contiguous().view(torch.uint8),
                                      baseline[key].contiguous().view(torch.uint8)))
                 for key, value in outputs.items()}
        differences = {key: float((value.double() - baseline[key].double()).abs().max())
                       if value.numel() else 0.0 for key, value in outputs.items()}
        records[name] = dict(oracle=checks, bitwise_equal_to_contracts=exact,
                             max_absolute_difference_to_contracts=differences)
    return records


def capture(function, repetitions):
    import torch
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            function()
    return graph


def measure(functions, repetitions, rounds, seed):
    import torch
    graphs = {name: capture(fn, repetitions) for name, fn in functions.items()}
    for _ in range(3):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    times = {name: [] for name in functions}
    orders = []
    rng = random.Random(seed)
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / repetitions)
    return dict(samples_us=times, orders=orders,
                median_us={k: statistics.median(v) for k, v in times.items()})


def retain(output, case, variant, kernel):
    folder = output / "code"
    folder.mkdir(exist_ok=True)
    for ext in ("ttir", "ttgir", "ptx"):
        (folder / f"{case['id']}.{variant}.{ext}.gz").write_bytes(
            gzip.compress(kernel.asm[ext].encode(), mtime=0))


def run_case(adapter, case, output, number):
    config_list = adapter.configurations(case)
    if case.get("boundary", False):
        config_list = config_list[:1]
    data = adapter.prepare(case, 19001 + number * 101)
    boundary = case.get("boundary", False)
    repetitions = case.get("repetitions", 16)
    report = dict(case=case, tuning=[], confirmations=[], failures=[])
    start = time.monotonic()
    for i, config in enumerate(config_list):
        # Failures are retained, never silently excluded from the kernel census.
        # A GPU execution/correctness failure aborts the experiment rather than
        # continuing after potentially invalid device state.
        build = compile_case(adapter, case, data, config)
        checks = validate(adapter, case, data, build)
        record = {key: build[key] for key in (
            "resources", "trailing_resources", "config", "grid", "options", "groups",
            "compile_seconds", "no_contract_matches_contracts_ptx",
            "pipeline_changes_contracts_ptx", "pipeline_changes_licm_ptx")}
        record["validation"] = checks
        if not boundary:
            record["timing"] = measure(build["functions"], repetitions, 5, 29003 + number * 101 + i)
            print("SCREEN", case["id"], config, record["timing"]["median_us"], flush=True)
        report["tuning"].append(record)
        write(output / f"{case['id']}.json", report)
        if boundary:
            for name, kernel in build["kernels"].items():
                retain(output, case, name, kernel)
            print("BOUNDARY", case["id"], "PASS", flush=True)
            return report
    best = {name: min(range(len(config_list)),
                      key=lambda i: report["tuning"][i]["timing"]["median_us"][name])
            for name in VARIANTS}
    base_i = best["contracts"]
    winner = min(("contracts", "licm", "pipeline", "both"),
                 key=lambda name: report["tuning"][best[name]]["timing"]["median_us"][name])
    screen_base = report["tuning"][base_i]["timing"]["median_us"]["contracts"]
    gain = screen_base / report["tuning"][best[winner]]["timing"]["median_us"][winner]
    if gain < 1.02:
        winner = "contracts"
    report["selection"] = dict(variant=winner, config=config_list[best[winner]],
                               baseline_config=config_list[base_i], screening_speedup=gain,
                               minimum_screen_gain=1.02,
                               calibration_seconds=time.monotonic() - start,
                               selected_configs={name: config_list[i] for name, i in best.items()},
                               comparison_baseline="same-contracts, no LICM or auto-stage inference")
    del data
    write(output / f"{case['id']}.json", report)
    for si, seed in enumerate((39019, 49031, 59051)):
        data = adapter.prepare(case, seed + number * 101)
        builds, checks = {}, {}
        for i in sorted(set(best.values())):
            builds[i] = compile_case(adapter, case, data, config_list[i])
            checks[str(i)] = validate(adapter, case, data, builds[i])
        # launch_spec may set validation-only active-config state; timing uses
        # immutable launch argument tuples captured separately in every build.
        functions = {name: builds[i]["functions"][name] for name, i in best.items()}
        timing = measure(functions, repetitions, 13, seed + 7)
        samples = timing["samples_us"]
        ratios = {name: [a / b for a, b in zip(samples["contracts"], values)]
                  for name, values in samples.items()}
        confirmation = dict(seed=seed + number * 101, validation=checks, timing=timing,
                            speedup_vs_contracts={name: statistics.median(v) for name, v in ratios.items()},
                            speedup_samples_vs_contracts=ratios,
                            contracts_vs_no_contract=statistics.median(
                                a / b for a, b in zip(samples["no_contract"], samples["contracts"])))
        report["confirmations"].append(confirmation)
        if si == 0:
            for name, i in best.items():
                retain(output, case, name, builds[i]["kernels"][name])
        write(output / f"{case['id']}.json", report)
        print("CONFIRM", case["id"], winner, confirmation["speedup_vs_contracts"], flush=True)
        del builds, functions, data
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapters", nargs="+", choices=ADAPTERS, default=list(ADAPTERS))
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Use an empty output directory to preserve earlier evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import triton
    native = Path(triton.__file__).resolve().parent / "_C/libtriton.so"
    fingerprint = sha(native)
    adapters = {name: importlib.import_module(ADAPTERS[name]) for name in args.adapters}
    plan = [(adapter, case) for adapter in adapters.values() for case in adapter.cases()
            if not args.cases or case["id"] in args.cases]
    write(args.output / "predeclared_plan.json", dict(
        cases=[case for _, case in plan],
        configurations={case["id"]: adapter.configurations(case)[:1] if case.get("boundary", False)
                        else adapter.configurations(case) for adapter, case in plan},
        variants=VARIANTS, screening_rounds=5, confirmation_rounds=13,
        confirmation_seeds=[39019, 49031, 59051], selection_minimum_gain=1.02,
        baseline="contracts only; no-contract is a separate safety-restoration control",
        no_native_or_kernel_body_edits=True))
    write(args.output / "environment.json", dict(
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        torch=torch.__version__, triton=triton.__version__, native_library_sha256=fingerprint,
        harness_sha256=sha(__file__), sources={name: adapter.provenance() for name, adapter in adapters.items()},
        scope="Complete kernel outputs including required reduction kernels; allocation, calibration, metadata generation and host dispatch excluded. Per-shape calibrated choices, not a static profitability model."))
    reports = []
    for number, (adapter, case) in enumerate(plan):
        try:
            reports.append(run_case(adapter, case, args.output, number))
        except Exception as error:
            write(args.output / "failure.json", dict(case=case, error_type=type(error).__name__,
                                                    message=str(error), completed=False))
            raise
        assert sha(native) == fingerprint, "Native compiler changed during experiment"
    summary = []
    for report in reports:
        row = dict(case=report["case"],
                   pipeline_changes_contracts_ptx=any(t["pipeline_changes_contracts_ptx"] for t in report["tuning"]),
                   contracts_change_source_ptx=any(not t["no_contract_matches_contracts_ptx"] for t in report["tuning"]))
        if "selection" in report:
            row["selection"] = report["selection"]
            row["speedups_vs_contracts"] = [c["speedup_vs_contracts"] for c in report["confirmations"]]
            row["contracts_vs_no_contract"] = [c["contracts_vs_no_contract"] for c in report["confirmations"]]
        summary.append(row)
    write(args.output / "summary.json", dict(records=summary, native_library_sha256=fingerprint,
                                             native_unchanged=True, completed=True))
    for path in [Path(__file__)] + [Path(adapter.__file__) for adapter in adapters.values()]:
        (args.output / (path.name + ".gz")).write_bytes(gzip.compress(path.read_bytes(), mtime=0))


if __name__ == "__main__":
    main()
