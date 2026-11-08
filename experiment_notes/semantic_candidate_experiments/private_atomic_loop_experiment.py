#!/usr/bin/env python3
"""Unchanged FlagGems persistent int32 frequency count with native relaxation.

The upstream file is supplied explicitly and copied into the compact evidence.
Only the manual control adds sem="relaxed". Timings include output reset and
counting, and exclude producer generation, allocation, and Python guards.
"""

import argparse
import ast
import copy
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch
import triton
from triton._C import libtriton
from triton.experimental.semantic_contracts.native import compile_private_atomic

from coverage_experiment import digest, paired_device_times, save_json
from native_indexed_update_experiment import kernel_evidence


def extract(source, output):
    original = ast.parse(source.read_text())
    name = "_embedding_dense_backward_count_kernel"
    node = copy.deepcopy(next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name))
    assert len(node.decorator_list) == 1 and ast.unparse(node.decorator_list[0]) == "triton.jit"
    manual = copy.deepcopy(node)
    manual.name += "_relaxed"
    calls = [n for n in ast.walk(manual) if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.atomic_add"]
    assert len(calls) == 1 and not any(k.arg == "sem" for k in calls[0].keywords)
    calls[0].keywords.append(ast.keyword(arg="sem", value=ast.Constant(value="relaxed")))
    directory = output / "sources"
    directory.mkdir(exist_ok=True)
    upstream_copy = directory / "flaggems_cambricon_embedding.py"
    upstream_copy.write_text(source.read_text())
    module_path = directory / "count_kernels.py"
    header = source.read_text().split("import logging")[0]
    module_path.write_text(header + "import triton\nimport triton.language as tl\n\n" + ast.unparse(node) + "\n\n" +
                           ast.unparse(manual) + "\n")
    audit = ast.parse(module_path.read_text())
    extracted = next(n for n in audit.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(node) == ast.dump(extracted)
    spec = importlib.util.spec_from_file_location("private_count_extracted", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name), getattr(module, name + "_relaxed")


def normalized_body(ir):
    return "\n".join(re.sub(r"\s+loc\([^\n]*\)", "", line).replace("add, acq_rel,", "add, relaxed,")
                     for line in ir.splitlines() if "tt.func " not in line and not line.strip().startswith("#loc"))


def grid_choices(n, block, sm_count):
    # Include a nonpersistent launch and larger persistent grids so a gain at
    # an underfilled grid is not mistaken for an advantage over tuning.
    blocks = triton.cdiv(n, block)
    return sorted({sm_count, 4 * sm_count, min(16 * sm_count, blocks), min(32 * sm_count, blocks), blocks})


def measure(kernel, manual, config, seed, output, settings, smoke=False):
    n, vocabulary, block, grid, distribution = config
    torch.manual_seed(seed)
    indices = torch.randint(1, vocabulary if distribution == "uniform" else min(33, vocabulary), (n,),
                            device="cuda", dtype=torch.int64)
    # Invalid and padding indices exercise the upstream validity masks.
    indices[::257] = -1
    indices[::263] = vocabulary
    indices[::269] = 0
    storage = torch.full((vocabulary + 32,), 123, device="cuda", dtype=torch.int32)
    counts = storage[16:-16]
    args = (indices, counts, n, vocabulary, 0, block, triton.cdiv(n, block))
    plan = compile_private_atomic(kernel, args, grid=(grid,), output_arg="counts_ptr", options={"num_warps": 4})
    compiled = {"baseline": plan.baseline, "native": plan.optimized,
                "manual": manual.warmup(*args, grid=(grid,), num_warps=4, enable_private_atomic_relaxation=False,
                                         enable_indexed_update_specialization=False)}
    report = plan.report()
    assert report["transformed"]
    assert plan.baseline.src.fn is plan.optimized.src.fn
    assert normalized_body(plan.baseline.asm["ttir"]) == normalized_body(plan.optimized.asm["ttir"])
    assert "scf.for" in plan.baseline.asm["ttir"]
    initial = 2**31 - 2 if smoke else 0

    def run(name):
        counts.fill_(initial)
        compiled[name][plan.grid](*args)

    def expected():
        valid = (indices > 0) & (indices < vocabulary)
        return (torch.bincount(indices[valid], minlength=vocabulary) + initial).to(torch.int32)

    reference = expected()
    for name in compiled:
        run(name)
        assert torch.equal(counts, reference)
        assert torch.all(storage[:16] == 123) and torch.all(storage[-16:] == 123)
    times = None if smoke else paired_device_times({name: lambda name=name: run(name) for name in compiled}, seed=seed,
                                                  **settings)
    # The contract makes no immutable-input claim; mutation between launches is legal.
    indices.fill_(7)
    reference = expected()
    for name in compiled:
        run(name)
        assert torch.equal(counts, reference)
    retain = smoke
    label = f"n{n}-v{vocabulary}-b{block}-g{grid}-{distribution}"
    code = {name: kernel_evidence(item, output, label + "-" + name, retain) for name, item in compiled.items()}
    for name in compiled:
        assert code[name]["atomic_sites"] > 0
    return {"n": n, "vocabulary": vocabulary, "block": block, "grid": grid, "distribution": distribution,
            "seed": seed, "device": times, "report": report, "code": code, "correctness":
            "exact int32 reference; invalid/padding/tail masks; guard storage; changed input; smoke includes overflow"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    kernel, manual = extract(args.source, args.output)
    fingerprints = {"harness": digest(__file__), "upstream": digest(args.source), "libtriton": digest(libtriton.__file__)}
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    settings = {"rounds": 15, "repetitions": 32, "warmup_ms": 200}
    metadata = {"device": torch.cuda.get_device_name(), "sm_count": sm_count, "torch": torch.__version__,
                "triton": triton.__version__, "fingerprints": fingerprints, "settings": settings,
                "upstream_revision": "3f1f896fd90f5db42d167d64fcb9691326071bc2",
                "upstream_path": "src/flag_gems/runtime/backend/_cambricon/ops/embedding_dense_backward.py",
                "scope": "Direct portable Cambricon-origin count body on NVIDIA; not upstream platform dispatch. "
                         "Output reset + count; producer/allocation/host guards excluded.",
                "source_audit": "Original body and decorator identical AST; manual control only adds sem=relaxed."}
    rows = []
    smoke = measure(kernel, manual, (513, 67, 256, 8, "uniform"), 0, args.output, settings, smoke=True)
    save_json(args.output / "smoke.json", smoke)
    if not args.smoke_only:
        for seed in (0, 1, 2):
            for n, vocabulary in ((4096, 8192), (65536, 32768), (1048576, 131072)):
                for distribution in ("uniform", "hot32"):
                    for block in (256, 512):
                        for grid in grid_choices(n, block, sm_count):
                            row = measure(kernel, manual, (n, vocabulary, block, grid, distribution), seed,
                                          args.output, settings)
                            rows.append(row)
                            medians = row["device"]["median_us"]
                            print(n, distribution, block, grid, seed, medians, flush=True)
                            save_json(args.output / "results.json", {"metadata": metadata, "cases": rows})
    assert fingerprints["libtriton"] == digest(libtriton.__file__)
    summary = {"metadata": metadata, "cases": len(rows), "smoke": "passed", "groups": []}
    for n in sorted({r["n"] for r in rows}):
        for distribution in ("uniform", "hot32"):
            chosen = [r for r in rows if r["n"] == n and r["distribution"] == distribution]
            variants = {name: min(statistics.median(r["device"]["median_us"][name] for r in chosen
                                                   if r["block"] == block and r["grid"] == grid)
                                  for block in (256, 512) for grid in grid_choices(n, block, sm_count))
                        for name in ("baseline", "native", "manual")}
            summary["groups"].append({"n": n, "distribution": distribution, "best_tested_us": variants,
                                      "best_tested_speedup": variants["baseline"] / variants["native"]})
    if rows:
        summary["matched_geomean_speedup"] = math.exp(statistics.mean(math.log(
            r["device"]["median_us"]["baseline"] / r["device"]["median_us"]["native"]) for r in rows))
        summary["native_vs_manual_geomean"] = math.exp(statistics.mean(math.log(
            r["device"]["median_us"]["manual"] / r["device"]["median_us"]["native"]) for r in rows))
    save_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
