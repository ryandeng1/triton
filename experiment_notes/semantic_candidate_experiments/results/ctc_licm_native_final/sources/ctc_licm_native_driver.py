#!/usr/bin/env python3
"""Pinned FlagGems CTC training-forward LICM performance gate.

The manual variant moves only the invariant previous-label load and its pure
dependencies. All barriers, recurrence storage, arithmetic, and output reduction
remain. This is a performance upper bound, not a compiler legality proof.
"""

import argparse
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REV = "3f1f896fd90f5db42d167d64fcb9691326071bc2"
SOURCE_SHA = "54ff130ab816ef2a610a083ae6239717f6d44d8e7dd3c79b3972454d092da91d"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def dump(node):
    return ast.dump(node, include_attributes=False)


def extract(args):
    source = args.cache / ("FlagGems-" + REV) / "src/flag_gems/ops/ctc_loss.py"
    raw = source.read_bytes()
    assert sha(raw) == SOURCE_SHA
    definitions = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    original = copy.deepcopy(definitions["_ctc_loss_forward_kernel"])
    helpers = [copy.deepcopy(definitions[n]) for n in ("_logaddexp", "_logaddexp3")]
    manual = copy.deepcopy(original)
    manual.name += "_manual_hoist"
    loop = next(n for n in manual.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "t")
    names = {"prev_target_index", "prev_target_value", "skip_allowed"}
    moved = [n for n in loop.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    assert len(moved) == 3
    loop.body = [n for n in loop.body if n not in moved]
    # T is constexpr. Preserve the original zero-trip load behavior at T=1.
    mask = next(k for k in moved[1].value.keywords if k.arg == "mask")
    original_mask = copy.deepcopy(mask.value)
    mask.value = ast.BinOp(left=mask.value, op=ast.BitAnd(), right=ast.Compare(left=ast.Name(id="T", ctx=ast.Load()), ops=[ast.Gt()], comparators=[ast.Constant(1)]))
    pos = manual.body.index(loop)
    manual.body[pos:pos] = moved
    # Undo precisely the proposed change; the complete original AST must return.
    undone = copy.deepcopy(manual)
    undo_loop = next(n for n in undone.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "t")
    undo_moved = [n for n in undone.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    undone.body = [n for n in undone.body if n not in undo_moved]
    next(k for k in undo_moved[1].value.keywords if k.arg == "mask").value = original_mask
    insertion = next(i for i, n in enumerate(undo_loop.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "acc")
    undo_loop.body[insertion:insertion] = undo_moved
    undone.name = original.name
    assert dump(undone) == dump(original)
    barrier_count = lambda node: sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_debug_barrier" for n in ast.walk(node))
    assert barrier_count(original) == barrier_count(manual) == 3
    for node in helpers + [original, manual]:
        node.decorator_list = [ast.Attribute(value=ast.Name(id="triton", ctx=ast.Load()), attr="jit", ctx=ast.Load())]
    module = ast.fix_missing_locations(ast.Module(body=helpers + [original, manual], type_ignores=[]))
    generated = "# Extracted from FlagGems, Apache-2.0; see ctc_loss.upstream.py.\nimport triton\nimport triton.language as tl\n_debug_barrier = tl.debug_barrier\n\n" + ast.unparse(module) + "\n"
    directory = args.output / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ctc_loss.upstream.py").write_bytes(raw)
    module_path = directory / "ctc_kernels.py"
    module_path.write_text(generated)
    spec = importlib.util.spec_from_file_location("ctc_licm_extracted", module_path)
    imported = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = imported
    spec.loader.exec_module(imported)
    audit = {"revision": REV, "url": f"https://github.com/flagos-ai/FlagGems/blob/{REV}/src/flag_gems/ops/ctc_loss.py",
             "source_sha256": SOURCE_SHA, "extracted_source_sha256": sha(generated.encode()),
             "original_body_ast_sha256": sha(dump(ast.Module(body=original.body, type_ignores=[])).encode()),
             "undo_manual_change_recovers_exact_original_ast": True, "barrier_sites_unchanged": 3,
             "manual_change": "Move prev_target_index, prev_target_value, skip_allowed immediately before the timestep loop; conjoin target load mask with T>1. No other body changes.",
             "precondition": "Targets initialized and unchanged through invocation; distinct from writable loss/alpha allocations. Original masked accesses and all barrier positions preserved.",
             "host_path": "CtcLossFunction.forward when ctx.needs_input_grad[0] is true. No-grad uses different kernels. Forward saves full log_alpha. This gate includes full training-forward GPU work, not backward or host validation/allocation latency.",
             "t0_limitation": "T=0 is not executed: unchanged upstream stores initial alpha into a zero-sized N*T*S allocation. T=1 tests a zero-trip recurrence and masked hoisted load."}
    write(args.output / "source_audit.json", audit)
    return imported


def configurations():
    return [(multiplier, warps) for multiplier in (1, 2) for warps in (1, 2, 4, 8)]


def workloads():
    # T, N, C, max target length, source dtype, target dtype, target layout,
    # reduction, zero_infinity. First four tune/training, next four held out.
    values = [
        ("training", 32, 4, 64, 16, "float32", "int32", False, "mean", True),
        ("training", 128, 32, 64, 32, "float32", "int64", False, "mean", True),
        ("training", 512, 128, 128, 64, "float32", "int32", False, "mean", True),
        ("training", 256, 256, 64, 64, "bfloat16", "int64", True, "mean", True),
        ("heldout", 64, 32, 128, 24, "float32", "int64", False, "sum", False),
        ("heldout", 256, 128, 64, 48, "float32", "int32", True, "mean", True),
        ("heldout", 1024, 64, 128, 128, "bfloat16", "int32", False, "mean", True),
        ("heldout", 128, 512, 32, 24, "float32", "int64", False, "none", False),
        ("edge", 1, 4, 8, 1, "float32", "int32", False, "none", False),
        ("edge", 1, 4, 8, 1, "bfloat16", "int64", True, "mean", True),
        ("edge", 16, 4, 8, 0, "float32", "int64", False, "mean", True),
    ]
    keys = "split T N C U dtype target_dtype target_1d reduction zero_infinity".split()
    return [dict(zip(keys, row), id=f"{row[0]}-T{row[1]}-N{row[2]}-U{row[4]}-{row[5]}-{'1d' if row[7] else '2d'}") for row in values]


def make_inputs(case, seed):
    T, N, C, U = (case[k] for k in ("T", "N", "C", "U"))
    torch.manual_seed(seed)
    source = torch.randn((T, N, C), device="cuda").log_softmax(-1).to(getattr(torch, case["dtype"]))
    lengths = torch.randint(max(0, U // 2), U + 1, (N,), dtype=torch.int64)
    lengths[-1] = U
    inputs = torch.randint(max(1, T * 3 // 4), T + 1, (N,), dtype=torch.int64)
    if T == 1:
        lengths, inputs = torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 1])
    elif case["zero_infinity"]:
        lengths[:3] = torch.tensor([0, 0, min(U, 1)])
        inputs[:3] = torch.tensor([0, 1, 2])
    target = torch.randint(1, C, (N, U), dtype=getattr(torch, case["target_dtype"]))
    for col in range(1, U, 3):
        target[:, col] = target[:, col - 1]
    if case["target_1d"]:
        target = torch.cat([row[:int(length)] for row, length in zip(target, lengths)])
    targets = target.to("cuda")
    il, tlens = inputs.to("cuda"), lengths.to("cuda")
    work = torch.empty_like(source, dtype=torch.float32) if source.dtype != torch.float32 else source
    offsets = torch.empty_like(tlens) if case["target_1d"] else tlens
    prefix = torch.empty_like(tlens) if case["target_1d"] else None
    cpu_logp = source.detach().float().cpu().double()
    ref_nll, ref_alpha = torch.ops.aten._ctc_loss.Tensor(cpu_logp, target.long(), inputs, lengths, 0, False)
    ref_output = torch.nn.functional.ctc_loss(cpu_logp, target.long(), inputs, lengths, blank=0,
                                             reduction=case["reduction"], zero_infinity=case["zero_infinity"])
    return {"source": source, "work": work, "targets": targets, "il": il, "tl": tlens,
            "offsets": offsets, "prefix": prefix, "cpu_input_lengths": inputs, "cpu_target_lengths": lengths,
            "reference_nll": ref_nll, "reference_alpha": ref_alpha, "reference_output": ref_output,
            "snapshots": [source.clone(), targets.clone(), il.clone(), tlens.clone()]}


def prepare(data):
    if data["source"] is not data["work"]:
        data["work"].copy_(data["source"])
    if data["prefix"] is not None:
        torch.cumsum(data["tl"], 0, out=data["prefix"])
        torch.sub(data["prefix"], data["tl"], out=data["offsets"])


def resources(kernel):
    kernel._init_handles()
    ptx = kernel.asm["ptx"]
    roots = {"%targets", "%arg1"}
    depth, loop_levels, target_loads = 0, [], []
    for number, line in enumerate(kernel.asm["ttgir"].splitlines(), 1):
        if re.search(r"\bscf\.(?:for|while)\b", line) and "{" in line:
            loop_levels.append(depth + 1)
        match = re.search(r"(%[A-Za-z0-9_]+)(?::\d+)?\s*=\s*([A-Za-z0-9_.]+)\s+(.*)", line)
        if match:
            result, op, rhs = match.groups()
            operands = re.findall(r"%[A-Za-z0-9_]+", rhs)
            if operands and operands[0] in roots:
                if op in ("tt.addptr", "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "ttg.convert_layout"):
                    roots.add(result)
                elif op == "tt.load":
                    target_loads.append({"ir_line": number, "loop_depth": len(loop_levels), "operation": line.strip()})
        depth += line.count("{") - line.count("}")
        loop_levels = [level for level in loop_levels if level <= depth]
    return {"registers": kernel.n_regs, "spills": kernel.n_spills, "shared_bytes": kernel.metadata.shared,
            "ptx_sha256": sha(ptx.encode()), "cubin_sha256": sha(kernel.asm["cubin"]),
            "ttgir_sha256": sha(kernel.asm["ttgir"].encode()),
            "bar_sync_static_sites": len(re.findall(r"\bbar\.sync\b", ptx)),
            "global_load_static_sites": len(re.findall(r"\bld\.global", ptx)),
            "target_loads": target_loads, "target_loads_inside_loop": sum(item["loop_depth"] > 0 for item in target_loads)}


def run_config(case, data, multiplier, warps, seed, stage, args, module):
    T, N, C, U = (case[k] for k in ("T", "N", "C", "U"))
    states = 2 * U + 1
    block = triton.next_power_of_2(states) * multiplier
    nll = old.guarded((N,), torch.float32, float("nan"))
    alpha = old.guarded((N, T, states), torch.float32, float("nan"))
    clean, denominator = torch.empty_like(nll["value"]), torch.empty_like(data["tl"])
    is_inf, normalized = torch.empty((N,), device="cuda", dtype=torch.bool), torch.empty_like(nll["value"])
    zero = torch.zeros((), device="cuda")
    unreduced = clean if case["zero_infinity"] else nll["value"]
    reduced = unreduced if case["reduction"] == "none" else torch.empty((), device="cuda")
    output = reduced if case["dtype"] == "float32" else torch.empty_like(reduced, dtype=getattr(torch, case["dtype"]))
    call_args = [data["work"], data["targets"], data["il"], data["tl"], data["offsets"], nll["value"], alpha["value"], T, N, C, U, states, 0, case["target_1d"], block]
    opts = dict(num_warps=warps, num_stages=1, enable_indexed_update_specialization=False, enable_private_atomic_relaxation=False)
    baseline = module._ctc_loss_forward_kernel.warmup(*call_args, grid=(N,), **opts)
    attrs = copy.deepcopy(baseline.src.attrs)
    attrs[(1,)] = [(name, value) for name, value in attrs.get((1,), []) if not name.startswith("tt.experimental_")] + [("tt.experimental_immutable", 1)]
    src = triton.compiler.ASTSource(baseline.src.fn, baseline.src.signature, constexprs=baseline.src.constants, attrs=attrs)
    immutable = triton.compile(src, target=baseline.metadata.target, options=opts)
    manual = module._ctc_loss_forward_kernel_manual_hoist.warmup(*call_args, grid=(N,), **opts)
    compiled = {"original": baseline, "immutable": immutable, "manual": manual}
    main_fns, full_fns = {}, {}

    def finish():
        values = nll["value"]
        if case["zero_infinity"]:
            torch.ops.aten.isinf.out(values, out=is_inf)
            torch.where(is_inf, zero, values, out=clean)
            values = clean
        if case["reduction"] == "none":
            pass
        elif case["reduction"] == "sum":
            torch.sum(values, dim=0, out=reduced)
        else:
            torch.clamp(data["tl"], min=1, out=denominator)
            torch.div(values, denominator, out=normalized)
            torch.mean(normalized, out=reduced)
        if output is not reduced:
            output.copy_(reduced)

    metrics = {}
    for name, kernel in compiled.items():
        launcher = kernel[(N, 1, 1)]
        main_fns[name] = lambda launcher=launcher: launcher(*call_args)
        def full(fn=main_fns[name]):
            prepare(data)
            fn()
            finish()
        full_fns[name] = full
        metrics[name] = resources(kernel)
    if args.native_followup:
        assert metrics["immutable"]["target_loads_inside_loop"] == 0
        assert metrics["original"]["bar_sync_static_sites"] == metrics["immutable"]["bar_sync_static_sites"]
    else:
        assert metrics["original"]["ptx_sha256"] == metrics["immutable"]["ptx_sha256"], "Native immutable result changed: inspect before interpreting experiment"
    full_fns["original"]()
    expected_nll, expected_alpha, expected_output = nll["value"].clone(), alpha["value"].clone(), output.clone()
    torch.testing.assert_close(expected_nll.cpu().double(), data["reference_nll"], atol=.004, rtol=3e-5)
    torch.testing.assert_close(expected_output.cpu().double(), data["reference_output"], atol=.03,
                               rtol=.01 if output.dtype == torch.bfloat16 else 3e-5)
    # Independent CPU FP64 recurrence oracle, including all valid states/times.
    actual_alpha = expected_alpha.cpu().double()
    max_error = 0.
    for batch in range(N):
        length, count = int(data["cpu_input_lengths"][batch]), 2 * int(data["cpu_target_lengths"][batch]) + 1
        actual, reference = actual_alpha[batch, :length, :count], data["reference_alpha"][batch, :length, :count]
        torch.testing.assert_close(actual, reference, atol=.004, rtol=3e-5)
        finite = torch.isfinite(reference)
        if finite.any():
            max_error = max(max_error, float((actual[finite] - reference[finite]).abs().max()))
    del actual_alpha

    def validate():
        for name, fn in full_fns.items():
            nll["value"].fill_(float("nan"))
            alpha["value"].fill_(float("nan"))
            output.fill_(float("nan"))
            fn()
            for actual, expected in [(nll["value"], expected_nll), (alpha["value"], expected_alpha), (output, expected_output)]:
                assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)), name
            old.assert_guards({"nll": nll, "alpha": alpha})
        for key, snapshot in zip(("source", "targets", "il", "tl"), data["snapshots"]):
            assert torch.equal(data[key], snapshot), key
    validate()
    capture_order = list(full_fns)
    random.Random(7319 + seed * 111 + block + warps).shuffle(capture_order)
    full_timing = old.paired_timings({name: full_fns[name] for name in capture_order}, args, seed + block * 17 + warps)
    kernel_timing = old.paired_timings({name: main_fns[name] for name in reversed(capture_order)}, args, seed + block * 31 + warps) if stage == "confirm" else None
    validate()
    record = {"case": case, "stage": stage, "seed": seed, "block": block, "block_multiplier": multiplier, "warps": warps,
              "graph_capture_order": capture_order,
              "resources": metrics, "full_forward": full_timing, "kernel_only": kernel_timing,
              "validation": {"independent_cpu_fp64_ctc_loss": True, "independent_cpu_fp64_log_alpha_valid_states": True,
                             "max_alpha_absolute_error": max_error, "all_variants_bitwise_equal_full_alpha_and_outputs": True,
                             "inputs_unchanged": True, "output_guards": True, "shared_timed_addresses": True}}
    if stage == "confirm" and seed == 0:
        # Only retain compact representative IR for two configurations.
        if case["id"] in ("training-T128-N32-U32-float32-2d", "heldout-T256-N128-U48-float32-1d"):
            dest = args.output / "ir" / (case["id"] + f"-b{block}-w{warps}")
            dest.mkdir(parents=True, exist_ok=True)
            for name, kernel in compiled.items():
                for ext in ("ttgir", "ptx"):
                    (dest / (name + "." + ext + ".gz")).write_bytes(gzip.compress(kernel.asm[ext].encode(), mtime=0))
    print(json.dumps({"stage": stage, "case": case["id"], "seed": seed, "block": block, "warps": warps,
                      "full_us": full_timing["median_us"]}), flush=True)
    return record


def summarize(records):
    rows = []
    for identifier in sorted({r["case"]["id"] for r in records}):
        cases = [r for r in records if r["case"]["id"] == identifier and r["stage"] == "confirm"]
        if not cases:
            continue
        best = {}
        for variant in ("original", "immutable", "manual"):
            options = []
            for config in sorted({(r["block_multiplier"], r["warps"]) for r in cases}):
                selected = [r for r in cases if (r["block_multiplier"], r["warps"]) == config]
                values = [r["full_forward"]["median_us"][variant] for r in selected]
                options.append({"multiplier": config[0], "warps": config[1], "median_us": statistics.median(values),
                                "seed_medians_us": values, "resources": selected[0]["resources"][variant]})
            best[variant] = min(options, key=lambda x: x["median_us"])
        config_rows = []
        for config in sorted({(r["block_multiplier"], r["warps"]) for r in cases}):
            selected = [r for r in cases if (r["block_multiplier"], r["warps"]) == config]
            config_rows.append({"multiplier": config[0], "warps": config[1],
                                "full_forward_seed_speedups": [r["full_forward"]["speedup_over_original"]["manual"] for r in selected],
                                "kernel_seed_speedups": [r["kernel_only"]["speedup_over_original"]["manual"] for r in selected]})
        rows.append({"case": cases[0]["case"], "best_confirmed": best,
                     "manual_speedup_best_to_best": best["original"]["median_us"] / best["manual"]["median_us"],
                     "native_speedup_best_to_best": best["original"]["median_us"] / best["immutable"]["median_us"],
                     "confirmed_configurations": config_rows})
    return {"rows": rows, "records": len(records), "all_validation_passed": True,
            "scope": "Complete training-forward GPU work including input conversion, 1D offsets, recurrence/full log_alpha, zero_infinity, loss reduction/output conversion. No backward, host length-stat/dispatch checks, allocator latency or log_softmax preparation.",
            "native_immutable": "Inspect each variant's target-load loop depth, unchanged barrier count, resources and baseline/native PTX hashes. In the original source gate native was a no-op; --native-followup verifies actual compiler hoisting."}


def main():
    global torch, triton, old
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/semantic-survey-cache"))
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--graph-repeats", type=int, default=16)
    p.add_argument("--warmup-ms", type=float, default=50)
    p.add_argument("--only", default="", help="Comma-separated T values for an initial smoke run")
    p.add_argument("--native-followup", action="store_true", help="Require actual native target-load hoisting; use a fresh output directory")
    p.add_argument("--confirm-from", type=Path, help="Reuse fixed configurations from a prior selected_configurations.json instead of screening again")
    args = p.parse_args()
    import torch
    import triton
    import triton._C.libtriton as libtriton
    import rmsnorm_experiment as old
    old.torch, old.triton = torch, triton
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "environment.json").exists():
        raise RuntimeError("Use a fresh output directory; existing experiment artifacts are immutable")
    module = extract(args)
    library_hash = sha(Path(libtriton.__file__).read_bytes())
    write(args.output / "environment.json", {"torch": torch.__version__, "triton": triton.__version__,
          "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
          "harness_sha256": sha(Path(__file__).read_bytes()), "libtriton_sha256": library_hash,
          "python": sys.executable, "args": vars(args), "configurations": configurations(),
          "screen_seed": None if args.confirm_from else 0, "confirmation_seeds": [0, 1, 2],
          "prior_selection_sha256": sha(args.confirm_from.read_bytes()) if args.confirm_from else None,
          "confirmation_selection": "Fixed shared union from prior baseline/manual/no-op-native tuning" if args.confirm_from else "Union of each variant's independently best screen configuration; all variants run on the same union."})
    cases = workloads()
    if args.only:
        cases = [c for c in cases if c["T"] in set(map(int, args.only.split(",")))]
    records = []
    selected = json.loads(args.confirm_from.read_text()) if args.confirm_from else {}
    if args.confirm_from:
        assert all(case["id"] in selected for case in cases)
        assert all(tuple(config) in configurations() for case in cases for config in selected[case["id"]])
    else:
        for case in cases:
            data = make_inputs(case, 0)
            configs = configurations()
            random.Random(2131 + case["T"] + case["N"]).shuffle(configs)
            rows = [run_config(case, data, m, w, 0, "screen", args, module) for m, w in configs]
            records.extend(rows)
            union = set()
            for variant in ("original", "immutable", "manual"):
                best = min(rows, key=lambda r: r["full_forward"]["median_us"][variant])
                union.add((best["block_multiplier"], best["warps"]))
            selected[case["id"]] = sorted(union)
            del data
            (args.output / "measurements.json.gz").write_bytes(gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0))
    write(args.output / "selected_configurations.json", selected)
    for seed in (0, 1, 2):
        order = list(cases)
        random.Random(5617 + seed).shuffle(order)
        for case in order:
            data = make_inputs(case, seed)
            for m, w in selected[case["id"]]:
                records.append(run_config(case, data, m, w, seed, "confirm", args, module))
            del data
            (args.output / "measurements.json.gz").write_bytes(gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0))
            write(args.output / "analysis.json", summarize(records))
    assert library_hash == sha(Path(libtriton.__file__).read_bytes()), "Compiler changed during study"


if __name__ == "__main__":
    main()
