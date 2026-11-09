#!/usr/bin/env python3
"""Validate the integrated contract pipeline selector on pinned real kernels.

The runtime API chooses stage depth. This harness separately calibrates launch
configurations for both the baseline and the runtime-selected plans over the same
predeclared domain, then freezes both choices before fresh-input confirmation.
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import inspect
from pathlib import Path
import statistics
import time

import cross_kernel_pipeline_experiment as common


HERE = Path(__file__).resolve().parent
CASES = [
    ("cross_kernel_flaggems_adapters", "fg_logsoftmax_m256_k8192_f32"),
    ("cross_kernel_flaggems_adapters", "fg_rotary_t128_q32_k8_d128_f32"),
    ("cross_kernel_flaggems_adapters", "fg_rotary_t1024_q32_k8_d128_bf16"),
    ("cross_pipeline_upsample", "upsample_2x64_64x256_to128x512"),
    ("cross_kernel_flaggems_adapters", "fg_logsoftmax_masked"),
    ("cross_kernel_flaggems_adapters", "fg_rotary_masked"),
    ("cross_pipeline_upsample", "upsample_boundary_single_height"),
]
CALIBRATION_SEED = 111071
CONFIRMATION_SEEDS = (121073, 131101, 141107)


def mutable_positions(case):
    return {
        "log_softmax_backward": (2,),
        "rotary_embedding": (0, 1),
        "upsample_bilinear2d": (0,),
    }[case["family"]]


def storage_bytes(tensor):
    import torch
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        storage, 0, (storage.nbytes(),), (1,))


def compile_plan(adapter, case, data, config):
    import torch
    from triton.experimental.semantic_contracts.native import compile_pipeline
    spec = adapter.launch_spec(case, data, config)
    assert not spec.get("trailing_specs"), "Study only complete single-kernel operations"
    args = tuple(spec["args"])
    common.allocation_contract(args, spec["groups"])
    # Byte views include allocation guards and preserve NaN payloads. Snapshot
    # every argument allocation, including read-only ones, to audit calibration.
    allocations = {}
    for value in args:
        if isinstance(value, torch.Tensor):
            span = value.untyped_storage().data_ptr()
            allocations.setdefault(span, storage_bytes(value))
    snapshots = {address: value.clone() for address, value in allocations.items()}
    started = time.monotonic()
    plan = compile_pipeline(
        spec["fn"], args, grid=spec["grid"], alias_groups=spec["groups"],
        mutable_args=mutable_positions(case), options=spec["options"],
        replay_safe=True, tune_licm=False)
    torch.cuda.synchronize()
    unchanged = {str(address): bool(torch.equal(view, snapshots[address]))
                 for address, view in allocations.items()}
    assert all(unchanged.values()), (case["id"], config, "calibration changed caller storage")
    return plan, dict(
        wall_compile_and_calibration_seconds=time.monotonic() - started,
        all_argument_backing_storage_preserved=True,
        checked_allocation_count=len(unchanged),
        mutable_args=mutable_positions(case), report=plan.report(),
        selected_name=plan.selected_name,
        resources={name: common.resources(kernel) for name, kernel in plan.candidates.items()})


def bound_calls(adapter, case, data, config, plan):
    spec = adapter.launch_spec(case, data, config)
    args = tuple(spec["args"])
    grid = tuple(spec["grid"]) + (1,) * (3 - len(spec["grid"]))
    assert grid == plan.grid
    common.allocation_contract(args, spec["groups"])
    baseline_launcher = plan.baseline[grid]

    def baseline():
        baseline_launcher(*args)

    def selected():
        # Capture the actual guarded plan path. During CUDA graph replay only
        # its captured device work executes, as with the preceding experiments.
        plan(*args, grid=grid)

    return dict(baseline=baseline, selected=selected), args


def validate_plan(adapter, case, data, config, plan):
    import torch
    calls, args = bound_calls(adapter, case, data, config, plan)
    adapter.reset(case, data)
    calls["baseline"]()
    torch.cuda.synchronize()
    baseline_check = adapter.validate(case, data)
    baseline = {key: value.clone() for key, value in adapter.output_tensors(data).items()}
    records = {}
    selected_name, selected_kernel = plan.selected_name, plan.selected
    invocations = {name: (lambda kernel=kernel: kernel[plan.grid](*args))
                   for name, kernel in plan.candidates.items()}
    invocations["guarded_selected"] = calls["selected"]
    for name, invoke in invocations.items():
        adapter.reset(case, data)
        invoke()
        torch.cuda.synchronize()
        checks = adapter.validate(case, data)
        equal = {key: bool(torch.equal(value.contiguous().view(torch.uint8),
                                      baseline[key].contiguous().view(torch.uint8)))
                 for key, value in adapter.output_tensors(data).items()}
        assert all(equal.values()), (case["id"], config, name, equal)
        records[name] = dict(oracle=checks, bitwise_equal_to_baseline=equal)
    assert plan.selected_name == selected_name and plan.selected is selected_kernel
    return dict(baseline_oracle=baseline_check, variants=records,
                cached_selection_unchanged=True, selected_name=selected_name)


def retain_plan(output, case, index, plan):
    for name, kernel in plan.candidates.items():
        common.retain(output, {"id": f"{case['id']}.config{index}"}, name, kernel)


def run_case(adapter, case, output, number):
    configs = adapter.configurations(case)
    if case.get("boundary", False):
        configs = configs[:1]
    data = adapter.prepare(case, CALIBRATION_SEED + number)
    report = dict(case=case, configurations=configs, calibration=[], confirmations=[])
    plans, functions = [], {}
    for index, config in enumerate(configs):
        plan, compile_record = compile_plan(adapter, case, data, config)
        plans.append(plan)
        checks = validate_plan(adapter, case, data, config, plan)
        compile_record.update(config=config, validation=checks)
        report["calibration"].append(compile_record)
        calls, _ = bound_calls(adapter, case, data, config, plan)
        functions.update({f"{name}.config{index}": fn for name, fn in calls.items()})
        retain_plan(output, case, index, plan)
        common.write(output / f"{case['id']}.json", report)
        print("CALIBRATE", case["id"], config, plan.selected_name, flush=True)
    if case.get("boundary", False):
        report["boundary_passed"] = True
        common.write(output / f"{case['id']}.json", report)
        return report
    screening = common.measure(functions, repetitions=32, rounds=5, seed=151121 + number)
    base_index = min(range(len(configs)), key=lambda i: screening["median_us"][f"baseline.config{i}"])
    selected_index = min(range(len(configs)), key=lambda i: screening["median_us"][f"selected.config{i}"])
    report["configuration_screening"] = screening
    report["selection"] = dict(
        baseline_config_index=base_index, selected_config_index=selected_index,
        baseline_config=configs[base_index], selected_config=configs[selected_index],
        selected_name=plans[selected_index].selected_name,
        depth_selection="Integrated compile_pipeline calibration; no harness depth override",
        configuration_selection="Independent five-round calibration for baseline and selected plans over identical predeclared launch domain",
        confirmation_reselects=False)
    common.write(output / f"{case['id']}.json", report)
    del functions, data
    for seed in CONFIRMATION_SEEDS:
        data = adapter.prepare(case, seed + number)
        checks = {str(index): validate_plan(adapter, case, data, configs[index], plans[index])
                  for index in sorted({base_index, selected_index})}
        base_calls, _ = bound_calls(adapter, case, data, configs[base_index], plans[base_index])
        selected_calls, _ = bound_calls(adapter, case, data, configs[selected_index], plans[selected_index])
        functions = dict(baseline=base_calls["baseline"], selected=selected_calls["selected"])
        before = [(p.selected_name, p.selected) for p in plans]
        timing = common.measure(functions, repetitions=32, rounds=15, seed=seed + 7)
        assert all(p.selected_name == name and p.selected is kernel
                   for p, (name, kernel) in zip(plans, before))
        ratios = [a / b for a, b in zip(timing["samples_us"]["baseline"], timing["samples_us"]["selected"])]
        result = dict(seed=seed + number, validation=checks, timing=timing,
                      speedup_vs_independently_tuned_baseline=statistics.median(ratios),
                      speedup_samples=ratios, cached_selection_unchanged=True,
                      same_plan_new_argument_allocations=True)
        report["confirmations"].append(result)
        common.write(output / f"{case['id']}.json", report)
        print("CONFIRM", case["id"], plans[selected_index].selected_name,
              result["speedup_vs_independently_tuned_baseline"], flush=True)
        del functions, data
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    import torch
    import triton
    import triton.experimental.semantic_contracts.native as native_api
    import triton.experimental.semantic_contracts.pipeline as pipeline_api
    torch.set_num_threads(4)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Use an empty output directory to preserve earlier evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    native = Path(triton.__file__).resolve().parent / "_C/libtriton.so"
    fingerprint = common.sha(native)
    adapters = {name: importlib.import_module(name) for name, _ in CASES}
    plan = [(adapters[name], next(c for c in adapters[name].cases() if c["id"] == case_id))
            for name, case_id in CASES if not args.cases or case_id in args.cases]
    assert plan, "No matching cases"
    common.write(args.output / "plan.json", dict(
        cases=[case for _, case in plan],
        configurations={case["id"]: adapter.configurations(case)[:1] if case.get("boundary")
                        else adapter.configurations(case) for adapter, case in plan},
        calibration_seed=CALIBRATION_SEED,
        confirmation_seeds=CONFIRMATION_SEEDS, seed_offset="case index",
        runtime_policy="compile_pipeline defaults: bounded stages0/2/3/4; tune_licm=False; replay_safe=True",
        external_config_screen_rounds=5, confirmation_rounds=15, graph_repetitions=32,
        baseline="Same contracts, no inferred stages, no contract LICM; independently tuned launch configuration",
        mutable_contract="Kernels fully overwrite all designated outputs; repeated invocations do not accumulate state",
        timed_resets=False, source_bodies_unchanged=True,
        exclusions="AITER RMSNorm has explicit source stages; this policy controls only inferred staging and does not override source schedules."))
    files = [Path(__file__), Path(common.__file__), Path(native_api.__file__),
             Path(inspect.getfile(native_api.compile_pipeline)), Path(pipeline_api.__file__),
             HERE.parents[1] / "lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp"]
    files += [Path(adapter.__file__) for adapter in adapters.values()]
    files = list(dict.fromkeys(files))
    common.write(args.output / "environment.json", dict(
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        torch=torch.__version__, triton=triton.__version__, native_sha256=fingerprint,
        files_sha256={str(path): common.sha(path) for path in files},
        provenance={name: adapter.provenance() for name, adapter in adapters.items()},
        timing_scope="Cached CUDA graph device execution; compilation, calibration, allocation, metadata validation and host dispatch excluded from steady-state timing; calibration wall time reported separately."))
    reports = []
    for number, (adapter, case) in enumerate(plan):
        try:
            reports.append(run_case(adapter, case, args.output, number))
        except Exception as error:
            common.write(args.output / "failure.json", dict(case=case, completed=False,
                error_type=type(error).__name__, message=str(error)))
            raise
        assert common.sha(native) == fingerprint, "Native compiler changed during experiment"
    for path in files:
        (args.output / (path.name + ".gz")).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    validation_count = output_count = 0
    for report in reports:
        checks = [record["validation"] for record in report["calibration"]]
        checks += [v for c in report["confirmations"] for v in c["validation"].values()]
        for check in checks:
            for record in check["variants"].values():
                validation_count += 1
                output_count += len(record["bitwise_equal_to_baseline"])
                assert all(record["bitwise_equal_to_baseline"].values())
    common.write(args.output / "summary.json", dict(
        completed=True, native_unchanged=True, native_sha256=fingerprint,
        variant_validations=validation_count, bitwise_output_comparisons=output_count,
        records=[dict(case=r["case"], selection=r.get("selection"),
                      speedups=[c["speedup_vs_independently_tuned_baseline"] for c in r["confirmations"]],
                      boundary_passed=r.get("boundary_passed", False)) for r in reports]))


if __name__ == "__main__":
    main()
