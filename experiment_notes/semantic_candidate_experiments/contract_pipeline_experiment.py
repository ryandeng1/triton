#!/usr/bin/env python3
"""Select validated LICM/pipeline variants of unchanged pinned recurrent source.

All four variants carry the same host-checked alias groups. Compiler options
independently control contract LICM and automatic stage inference. Calibration
resets state before every invocation; selected configurations are frozen before
fresh-seed confirmation. This is an experiment selector, not a general runtime
autotuner or a static profitability model.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

import recurrent_licm_experiment as prior

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "baseline": (False, 0),
    "licm": (True, 0),
    "pipeline": (False, 2),
    "both": (True, 2),
}


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def compile_case(case, data, module, config):
    import torch
    import triton
    from triton.compiler import ASTSource

    bv, warps, stages = config
    args = prior.arguments(case, data, bv)
    grid = (1, triton.cdiv(case["V"], bv), data["N"] * case["HV"])
    options = dict(num_warps=warps, num_stages=stages,
                   enable_indexed_update_specialization=False,
                   enable_private_atomic_relaxation=False,
                   enable_private_reduction_ownership=False,
                   enable_contract_load_reuse=False,
                   enable_contract_licm=False, contract_pipeline_stages=0)
    fn = getattr(module, prior.NAME)
    start = time.monotonic()
    original = fn.warmup(*args, grid=grid, **options)
    attrs = copy.deepcopy(original.src.attrs)
    assert not any("tt.experimental_" in str(v) for v in attrs.values())
    for number, value in enumerate(args):
        if isinstance(value, torch.Tensor):
            attrs.setdefault((number,), []).append(
                ("tt.experimental_alias_group", 1 if number in (9, 10) else 0))
    # Same-group input views may alias. Each complete output/state backing
    # allocation must be disjoint from every input backing allocation.
    for wi in (9, 10):
        ws = args[wi].untyped_storage()
        for ri, value in enumerate(args):
            if ri in (9, 10) or not isinstance(value, torch.Tensor):
                continue
            rs = value.untyped_storage()
            assert ws.data_ptr() + ws.nbytes() <= rs.data_ptr() or rs.data_ptr() + rs.nbytes() <= ws.data_ptr()
    kernels, resources = {}, {}
    for name, (licm, pipeline_stages) in VARIANTS.items():
        variant_options = dict(options, enable_contract_licm=licm,
                               contract_pipeline_stages=pipeline_stages)
        source = ASTSource(fn, original.src.signature,
                           constexprs=original.src.constants, attrs=attrs)
        kernel = triton.compile(source, target=original.metadata.target, options=variant_options)
        kernel._init_handles()
        kernels[name] = kernel
        resources[name] = dict(
            registers=kernel.n_regs, spills=kernel.n_spills, shared=kernel.metadata.shared,
            async_copies=kernel.asm["ttgir"].count("ttg.async_copy_global_to_local"),
            ptx_sha256=hashlib.sha256(kernel.asm["ptx"].encode()).hexdigest(),
            cubin_sha256=hashlib.sha256(kernel.asm["cubin"]).hexdigest(),
            options=variant_options)
    # Contract metadata alone must not be mistaken for the selected transform.
    assert kernels["baseline"].asm["ptx"] == original.asm["ptx"]
    return dict(args=args, grid=grid, kernels=kernels, resources=resources,
                compile_seconds=time.monotonic() - start, config=list(config),
                options=options, attributes={str(k): v for k, v in attrs.items()},
                no_contract_baseline_ptx_equal=True)


def validate(case, data, build):
    import torch
    checks = prior.validate(case, data, build)
    assert all(c["same_config_bitwise_equal_to_original"] for c in checks.values())
    for name in ("A_log", "dt_bias", "q", "k", "v", "a", "b", "h0_indices", "cu_seqlens", "initial"):
        assert torch.equal(data[name].cpu(), data["cpu"][name]), name
    return checks


def repetitions(case):
    return 128 if max(case.get("lengths", [case["T"]])) <= 1 else 8


def measure(data, build, case, seed, rounds):
    reps = repetitions(case)
    graphs = {name: prior.capture(data, build, name, reps) for name in VARIANTS}
    return prior.measure(graphs, reps, rounds, seed)


def retain(output, case, label, build, variant):
    dest = output / "code"
    dest.mkdir(exist_ok=True)
    kernel = build["kernels"][variant]
    for extension in ("ttir", "ttgir", "ptx"):
        (dest / f"{case['id']}.{label}.{extension}.gz").write_bytes(
            gzip.compress(kernel.asm[extension].encode(), mtime=0))


def experiment(case, module, output, number):
    start = time.monotonic()
    data = prior.data(case, 81001 + number)
    tuning = []
    for ci, config in enumerate(prior.CONFIGS):
        build = compile_case(case, data, module, config)
        checks = validate(case, data, build)
        samples = measure(data, build, case, 82001 + number * 17 + ci, 5)
        record = dict(config=list(config), resources=build["resources"], validation=checks,
                      times_us=samples, median_us={k: statistics.median(v) for k, v in samples.items()},
                      compile_seconds=build["compile_seconds"],
                      no_contract_baseline_ptx_equal=build["no_contract_baseline_ptx_equal"])
        tuning.append(record)
        print("SCREEN", case["id"], config, record["median_us"], flush=True)
    baseline_ci = min(range(len(tuning)), key=lambda i: tuning[i]["median_us"]["baseline"])
    best_ci, best_name = min(
        ((i, name) for i in range(len(tuning)) for name in VARIANTS),
        key=lambda item: tuning[item[0]]["median_us"][item[1]])
    baseline_us = tuning[baseline_ci]["median_us"]["baseline"]
    screen_gain = baseline_us / tuning[best_ci]["median_us"][best_name]
    # Predeclared margin; retain a tuned baseline when the screen is ambiguous.
    if screen_gain < 1.02:
        best_ci, best_name = baseline_ci, "baseline"
    selected_by_variant = {
        name: min(range(len(tuning)), key=lambda i: tuning[i]["median_us"][name])
        for name in VARIANTS}
    selection_seconds = time.monotonic() - start
    report = dict(case=case, tuning=tuning,
                  selected_configurations_by_variant={name: list(prior.CONFIGS[i])
                                                      for name, i in selected_by_variant.items()},
                  selection=dict(variant=best_name, config=list(prior.CONFIGS[best_ci]),
                                 baseline_config=list(prior.CONFIGS[baseline_ci]),
                                 screened_best_speedup=screen_gain, minimum_screen_gain=1.02,
                                 rule="lowest validated median across four variants and four launch configurations; retain best baseline below 2% gain",
                                 calibration_wall_seconds=selection_seconds,
                                 excluded_from_steady_state_timings=True), confirmations=[])
    write(output / f"{case['id']}.json", report)
    for si, seed in enumerate((83003, 84007, 85021)):
        data = prior.data(case, seed + number * 101)
        indices = set(selected_by_variant.values()) | {baseline_ci, best_ci}
        builds = {i: compile_case(case, data, module, prior.CONFIGS[i]) for i in indices}
        checks = {str(i): validate(case, data, b) for i, b in builds.items()}
        reps = repetitions(case)
        graphs = {name: prior.capture(data, builds[i], name, reps)
                  for name, i in selected_by_variant.items()}
        samples = prior.measure(graphs, reps, 13, seed + 19)
        # Selected config is precisely the per-variant minimum from the screen.
        assert selected_by_variant[best_name] == best_ci
        selected_ratio = [a / b for a, b in zip(samples["baseline"], samples[best_name])]
        record = dict(seed=seed + number * 101, validation=checks, times_us=samples,
                      median_us={k: statistics.median(v) for k, v in samples.items()},
                      selected_speedup_samples=selected_ratio,
                      selected_speedup=statistics.median(selected_ratio),
                      variant_speedups={name: statistics.median([a / b for a, b in zip(samples["baseline"], values)])
                                        for name, values in samples.items()},
                      state_reset_included_every_invocation=True)
        report["confirmations"].append(record)
        if si == 0:
            for name, index in selected_by_variant.items():
                retain(output, case, name, builds[index], name)
        write(output / f"{case['id']}.json", report)
        print("CONFIRM", case["id"], best_name, record["selected_speedup"], record["variant_speedups"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", default=[c["id"] for c in prior.CASES])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Use an empty output directory to preserve previous evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import triton
    old = HERE / "results/recurrent_licm_final"
    for name in ("kernels.py", "AITer_LICENSE"):
        shutil.copy2(old / name, args.output / name)
    module = prior.load_module(args.output / "kernels.py")
    native = Path(triton.__file__).resolve().parent / "_C/libtriton.so"
    native_hash = prior.sha(native)
    write(args.output / "provenance.json", dict(
        revision=prior.SHA, upstream_path=prior.KERNEL_PATH,
        copied_source_sha256=prior.sha(args.output / "kernels.py"),
        source_unchanged=True, source_rewritten_for_pipeline=False,
        native_library_sha256=native_hash, harness_sha256=prior.sha(__file__),
        oracle_harness_sha256=prior.sha(prior.__file__),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
        selection_has_runtime_calibration=True, all_four_variants_share_same_alias_contracts=True,
        enable_contract_load_reuse=False,
        contract="Different groups have disjoint full backing storage for this invocation; all inputs remain defined and no external concurrent writes occur.",
        timing_scope="Kernel outputs and final state plus state reset; excludes allocation, host guards, compilation, calibration and validation. Not a model or strongest-provider comparison."))
    reports = [experiment(c, module, args.output, i)
               for i, c in enumerate(prior.CASES) if c["id"] in args.cases]
    boundary_records = []
    cases = copy.deepcopy(prior.BOUNDARIES)
    cases.append(dict(id="all_empty", cohort="boundary", B=1, T=0, H=1, HV=1, K=60, V=60,
                      dtype="float32", qk_dtype="float32", parameter_dtype="float32", lengths=[0, 0], scale=1.0))
    for i, case in enumerate(cases):
        data = prior.data(case, 86001 + i)
        build = compile_case(case, data, module, (16, 1, 2))
        boundary_records.append(dict(case=case, validation=validate(case, data, build)))
    write(args.output / "boundaries.json", boundary_records)
    assert prior.sha(native) == native_hash
    summary = dict(records=[dict(case=r["case"]["id"], selection=r["selection"],
                                confirmed_speedups=[c["selected_speedup"] for c in r["confirmations"]],
                                variant_speedups=[c["variant_speedups"] for c in r["confirmations"]]) for r in reports],
                   native_library_unchanged=True, native_library_sha256=native_hash,
                   unchanged_source=True, boundary_cases=len(boundary_records))
    write(args.output / "summary.json", summary)
    (args.output / "measured_harness.py.gz").write_bytes(gzip.compress(Path(__file__).read_bytes(), mtime=0))


if __name__ == "__main__":
    main()
