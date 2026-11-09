#!/usr/bin/env python3
"""Fixed-launch causal controls for the cross-kernel pipeline slowdowns.

The earlier independent tuning comparison remains the performance result. This
study changes stage depth, loop selection, or load cache annotations at a fixed
launch, preserving computations and all required synchronization.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib
import json
from pathlib import Path
import re

import cross_kernel_pipeline_experiment as common

HERE = Path(__file__).resolve().parent
EXPECTED_NATIVE = "e7e15b9337405e632e887ceae9c96d9ea85e38aa1e9c35dec937d7b148d06178"
STUDIES = {
    "logsoftmax": ("cross_kernel_flaggems_adapters", "fg_logsoftmax_m256_k8192_f32", {"block": 1024, "warps": 8}),
    "rotary_w2": ("cross_kernel_flaggems_adapters", "fg_rotary_t128_q32_k8_d128_f32", {"warps": 2}),
    "rotary_w8": ("cross_kernel_flaggems_adapters", "fg_rotary_t128_q32_k8_d128_f32", {"warps": 8}),
    "upsample": ("cross_pipeline_upsample", "upsample_2x64_64x256_to128x512", (64, 2, 4)),
    "rms_rows": ("cross_pipeline_rms", "rms_forward_4096x1024", {"warps": 4, "grid": "rows"}),
    "rms_persistent": ("cross_pipeline_rms", "rms_forward_4096x1024", {"warps": 4, "grid": "4sm"}),
}


def variants(study):
    if study.startswith("rms"):
        return {"baseline": ("source_stages1", 0),
                **{f"stages{s}": (f"source_stages{s}", 0) for s in (2, 3, 4)}}
    result = {"baseline": (None, 0), **{f"stages{s}": (None, s) for s in (2, 3, 4)}}
    if study == "logsoftmax":
        result.update(sum_only2=("logsoftmax_sum_only2", 0),
                      output_only2=("logsoftmax_output_only2", 0))
    if study.startswith("rotary"):
        result.update(q_only2=("rotary_q_only2", 0), k_only2=("rotary_k_only2", 0))
    result.update(cache_ca_baseline=("load_cache_ca", 0),
                  cache_ca_stages2=("load_cache_ca", 2),
                  cache_ca_stages3=("load_cache_ca", 3))
    return result


def setup(study, seed):
    module, case_id, config = STUDIES[study]
    adapter = importlib.import_module(module)
    case = next(c for c in adapter.cases() if c["id"] == case_id)
    data = adapter.prepare(case, seed)
    return adapter, case, data, config


def compile_variant(adapter, case, data, config, study, name, output):
    import triton
    from triton.compiler import ASTSource
    from pipeline_slowdown_controls import get_variant_fn, variant_audit
    spec = adapter.launch_spec(case, data, config)
    label, stages = variants(study)[name]
    fn = spec["fn"] if label is None else get_variant_fn(spec["fn"], label, output / "generated")
    audit = {"unchanged_source": True} if label is None else variant_audit(fn)
    assert not spec.get("trailing_specs"), "Current studies use one full-output kernel"
    common.allocation_contract(spec["args"], spec["groups"])
    grid = tuple(spec["grid"]) + (1,) * (3 - len(spec["grid"]))
    options = dict(spec["options"], **common.BASE_OPTIONS)
    options["contract_pipeline_stages"] = stages
    base = fn.warmup(*spec["args"], grid=grid, **options)
    attrs = copy.deepcopy(base.src.attrs)
    for i, group in spec["groups"].items():
        attrs.setdefault((i,), []).append(("tt.experimental_alias_group", group))
    source = ASTSource(base.src.fn, base.src.signature, constexprs=base.src.constants, attrs=attrs)
    kernel = triton.compile(source, target=base.metadata.target, options=options)
    kernel._init_handles()
    launcher = kernel[grid]
    def invoke():
        launcher(*spec["args"])
    resources = common.resources(kernel)
    ptx = kernel.asm["ptx"]
    resources.update(wait_groups=re.findall(r"cp\.async\.wait_group\s+(\d+)", ptx),
                     barriers=ptx.count("bar.sync"),
                     commits=ptx.count("cp.async.commit_group"),
                     copy_cg=ptx.count("cp.async.cg"), copy_ca=ptx.count("cp.async.ca"))
    return dict(kernel=kernel, invoke=invoke, resources=resources, audit=audit,
                options=options, grid=grid, groups=spec["groups"])


def validate(adapter, case, data, builds):
    import torch
    records, baseline = {}, None
    for name, build in builds.items():
        adapter.reset(case, data)
        build["invoke"]()
        torch.cuda.synchronize()
        oracle = adapter.validate(case, data)
        outputs = adapter.output_tensors(data)
        if baseline is None:
            baseline = {k: v.clone() for k, v in outputs.items()}
        exact = {k: bool(torch.equal(v.contiguous().view(torch.uint8),
                                    baseline[k].contiguous().view(torch.uint8))) for k, v in outputs.items()}
        assert all(exact.values()), (name, exact)
        records[name] = dict(oracle=oracle, bitwise_equal_to_baseline=exact)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--studies", nargs="+", default=list(STUDIES))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=[71011, 81013, 91019])
    args = parser.parse_args()
    import torch
    import triton
    torch.set_num_threads(4)
    native = Path(triton.__file__).resolve().parent / "_C/libtriton.so"
    assert common.sha(native) == EXPECTED_NATIVE, "Native compiler changed"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.profile:
        assert len(args.studies) == 1
        study = args.studies[0]
        adapter, case, data, config = setup(study, args.seeds[0])
        build = compile_variant(adapter, case, data, config, study, args.variant, args.output)
        torch.cuda.synchronize()
        # All input construction is outside profiling. NCU selects launch six.
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(6):
            build["invoke"]()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        adapter.validate(case, data)
        print("PROFILE", study, args.variant, build["resources"], flush=True)
        return
    assert not (args.output / "plan.json").exists(), "Preserve previous results"
    common.write(args.output / "plan.json", dict(
        studies={s: dict(adapter=STUDIES[s][0], case=STUDIES[s][1], config=STUDIES[s][2], variants=variants(s)) for s in args.studies},
        seeds=args.seeds, timing_rounds=15, graph_repetitions=32,
        comparison="Fixed launch causal ablation; not an independently tuned speedup claim",
        controls="Stage hints and load-cache annotations only; no math or synchronization removal",
        profiler_timing_is_diagnostic_only=True))
    common.write(args.output / "environment.json", dict(
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        torch=torch.__version__, triton=triton.__version__, native_sha256=EXPECTED_NATIVE,
        scripts={name: common.sha(HERE / name) for name in (
            Path(__file__).name, "pipeline_slowdown_controls.py", "cross_kernel_pipeline_experiment.py",
            "cross_kernel_flaggems_adapters.py", "cross_pipeline_rms.py", "cross_pipeline_upsample.py")}))
    for study in args.studies:
        report = dict(study=study, config=STUDIES[study][2], confirmations=[])
        for seed in args.seeds:
            adapter, case, data, config = setup(study, seed)
            builds = {name: compile_variant(adapter, case, data, config, study, name, args.output)
                      for name in variants(study)}
            validation = validate(adapter, case, data, builds)
            if seed == args.seeds[0]:
                report.update(case=case, provenance=adapter.provenance(),
                    variants={name: {k: v for k, v in b.items() if k not in ("kernel", "invoke")}
                              for name, b in builds.items()})
                for name, build in builds.items():
                    common.retain(args.output, {"id": study}, name, build["kernel"])
            timing = common.measure({name: b["invoke"] for name, b in builds.items()}, 32, 15, seed + 23)
            medians = timing["median_us"]
            speedups = {name: medians["baseline"] / value for name, value in medians.items()}
            report["confirmations"].append(dict(seed=seed, validation=validation, timing=timing,
                                                speedup_vs_fixed_baseline=speedups))
            common.write(args.output / f"{study}.json", report)
            print("TIMED", study, seed, medians, "SPEEDUP", speedups, flush=True)
            assert common.sha(native) == EXPECTED_NATIVE
            del builds, data
        print("COMPLETE", study, flush=True)
    for path in (Path(__file__), HERE / "pipeline_slowdown_controls.py"):
        (args.output / (path.name + ".gz")).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    common.write(args.output / "completion.json", dict(completed=True, native_unchanged=True, studies=args.studies))


if __name__ == "__main__":
    main()
