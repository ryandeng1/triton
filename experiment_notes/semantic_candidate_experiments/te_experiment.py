#!/usr/bin/env python3
"""Paired TE unpermutation experiment; no provider or numerical-policy changes.

Examples (from this directory):
  python te_experiment.py --mode screen --output /tmp/te-screen
  python te_experiment.py --mode confirm --from-screen /tmp/te-screen/results.json --output /tmp/te-confirm
  python te_experiment.py --mode boundaries --output /tmp/te-boundaries

The baseline body is copied unchanged from pinned Transformer Engine. Only the
manual variant moves a route's invariant probability load before the hidden-size
loop. This assumes nonaliasing immutable inputs and fresh gradient destinations,
as supplied by this harness and the upstream PyTorch wrapper. Timings exclude
metadata construction/allocation and cover both gradient outputs of this kernel.
"""

import argparse
import ast
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import triton

import te_kernels as kernels

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "original": kernels._unpermute_bwd_with_merging_probs_kernel,
    "hoisted": kernels.unpermute_bwd_probability_hoisted,
}
PUBLISHED_CONFIGS = {(b, 4) for b in (64, 128, 256, 512, 1024, 2048, 4096)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_audit():
    original = ast.parse((ROOT / "vendor/transformer_engine_permutation.py").read_text())
    extracted = ast.parse((ROOT / "te_kernels.py").read_text())
    name = "_unpermute_bwd_with_merging_probs_kernel"
    a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
    b = next(n for n in extracted.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = lambda n: ast.dump(ast.Module(body=n.body, type_ignores=[]), include_attributes=False)
    assert body(a) == body(b), "baseline AST differs from vendored upstream"
    return {"baseline_body_ast_equal": True,
            "files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [
                ROOT / "te_kernels.py", ROOT / "te_experiment.py",
                ROOT / "vendor/transformer_engine_permutation.py",
                ROOT / "vendor/transformer_engine_permutation_wrapper.py"]},
            "upstream": json.loads((ROOT / "vendor/transformer_engine_provenance.json").read_text())}


def make_data(shape, seed, dtype, boundary=False, padding=False):
    m, h, k, e = shape
    assert m > 0 and h > 0 and 0 < k <= e
    cpu_gen = torch.Generator().manual_seed(seed)
    experts = torch.rand((m, e), generator=cpu_gen).argsort(dim=1)[:, :k]
    routing_cpu = torch.zeros((m, e), dtype=torch.int32)
    routing_cpu.scatter_(1, experts, 1)
    if boundary:
        routing_cpu[::7] = 0
        for t in range(1, m, 11):
            routing_cpu[t, experts[t, k // 2:]] = 0
    total = int(routing_cpu.sum())
    capacity = max(1, total - 3) if boundary else total
    # Reproduce the producer's expert-major row numbering and capacity drop.
    cpu_map = torch.full((m, 2 * e + 1), -1, dtype=torch.int32)
    slots = [[] for _ in range(m)]
    row = 0
    for expert in range(e):
        for token in torch.where(routing_cpu[:, expert] != 0)[0].tolist():
            if row < capacity:
                slots[token].append((row, expert))
            row += 1
    for token, entries in enumerate(slots):
        entries.sort(reverse=True)  # Upstream _argsort places valid rows before -1.
        n = len(entries)
        cpu_map[token, 2 * e] = n
        if n:
            cpu_map[token, :n] = torch.tensor([x[0] for x in entries], dtype=torch.int32)
            cpu_map[token, e:e+n] = torch.tensor([x[1] for x in entries], dtype=torch.int32)

    routing = routing_cpu.cuda()
    row_map = torch.full((m, 2 * e + 1), -987654, dtype=torch.int32, device="cuda")
    block = 1024
    workspace = torch.empty((e, triton.cdiv(m, block)), dtype=torch.int32, device="cuda")
    grid = (e, triton.cdiv(m, block))
    kernels._row_id_map_pass_1_kernel[grid](
        routing, m, *routing.stride(), *row_map.stride(), row_map, workspace, block)
    kernels._row_id_map_pass_2_kernel[grid](
        row_map, workspace, m, capacity, *row_map.stride(),
        WORKSPACE_LOAD_WIDTH=triton.next_power_of_2(workspace.numel()), BLOCK_SIZE=block)
    kernels._row_id_map_pass_3_kernel[(m,)](
        row_map, *row_map.stride(), e, triton.next_power_of_2(e))
    actual = row_map.cpu()
    for token in range(m):
        n = int(cpu_map[token, 2 * e])
        assert int(actual[token, 2 * e]) == n
        assert torch.equal(actual[token, :n], cpu_map[token, :n])
        assert torch.equal(actual[token, e:e+n], cpu_map[token, e:e+n])

    # Match fused FP8 padding: independently round each expert's kept rows to
    # a 16-row alignment, including the final expert's trailing padding.
    expert_counts = [0] * e
    for entries in slots:
        for _, expert in entries:
            expert_counts[expert] += 1
    pad_cpu = torch.zeros(e, dtype=torch.int32)
    padded_rows = capacity
    if padding:
        padding_so_far = 0
        for expert, count in enumerate(expert_counts):
            pad_cpu[expert] = padding_so_far
            padding_so_far += triton.cdiv(count, 16) * 16 - count
        padded_rows += padding_so_far
    pad = pad_cpu.cuda() if padding else None
    token_ids, expert_ids, destination_ids = [], [], []
    for token, entries in enumerate(slots):
        for dest, expert in entries:
            token_ids.append(token)
            expert_ids.append(expert)
            destination_ids.append(dest + int(pad_cpu[expert]))
    assert len(destination_ids) == len(set(destination_ids)) == capacity
    device_gen = torch.Generator(device="cuda").manual_seed(seed + 1000)
    dout = torch.randn((m, h), generator=device_gen, device="cuda", dtype=dtype)
    inp = torch.randn((padded_rows, h), generator=device_gen, device="cuda", dtype=dtype)
    probs = torch.rand((m, e), generator=device_gen, device="cuda", dtype=torch.float32)
    token_ids = torch.tensor(token_ids, device="cuda", dtype=torch.int64)
    expert_ids = torch.tensor(expert_ids, device="cuda", dtype=torch.int64)
    destination_ids = torch.tensor(destination_ids, device="cuda", dtype=torch.int64)
    # Accumulate the independent mathematical dot-product reference in FP64.
    # Chunk host-side work to keep reference memory bounded for larger cases.
    act_ref = torch.zeros_like(inp)
    prob_ref = torch.zeros_like(probs)
    for start in range(0, capacity, 128):
        ts = token_ids[start:start+128]
        es = expert_ids[start:start+128]
        ds = destination_ids[start:start+128]
        act_ref[ds] = (dout[ts].float() * probs[ts, es, None]).to(dtype)
        prob_ref[ts, es] = (dout[ts].double() * inp[ds].double()).sum(1).float()
    return {"shape": shape, "dout": dout, "inp": inp, "probs": probs,
            "row_map": row_map, "pad": pad, "padding": padding,
            "act_ref": act_ref, "prob_ref": prob_ref,
            "capacity": capacity, "rows": padded_rows, "destination_ids": destination_ids,
            "metadata_matches_upstream_producer": True}


def make_outputs(data):
    # Distinct prefix/suffix guards around both complete output allocations.
    guard = 128
    n = data["inp"].numel()
    act_storage = torch.full((n + 2 * guard,), 37.0, device="cuda", dtype=data["inp"].dtype)
    act = act_storage[guard:guard+n].view_as(data["inp"])
    pn = data["probs"].numel()
    prob_storage = torch.full((pn + 2 * guard,), 37.0, device="cuda", dtype=torch.float32)
    prob = prob_storage[guard:guard+pn].view_as(data["probs"])
    act.fill_(0.0 if data["padding"] else float("nan"))
    prob.fill_(float("nan"))
    return {"act": act, "prob": prob, "act_storage": act_storage,
            "prob_storage": prob_storage, "guard": guard}


def launch(data, out, variant, config, extra_options=None):
    m, h, k, e = data["shape"]
    block, warps = config
    bound = (
        data["dout"], data["inp"], data["probs"], data["row_map"], data["pad"],
        *data["row_map"].stride(), *data["dout"].stride(), *out["act"].stride(),
        *data["inp"].stride(), *data["probs"].stride(), *out["prob"].stride(),
        out["act"], out["prob"], e, h, triton.next_power_of_2(e), data["padding"], block)
    if variant != "contract":
        return VARIANTS[variant][(m,)](*bound, num_warps=warps, **(extra_options or {}))
    cache = data.setdefault("_contract_kernels", {})
    if config not in cache:
        from triton.compiler import ASTSource
        baseline = VARIANTS["original"].warmup(*bound, grid=(m,), num_warps=warps,
                                               **(extra_options or {}))
        source = baseline.src
        attrs = {key: list(value) for key, value in source.attrs.items()}
        attrs.setdefault((2,), []).append(("tt.experimental_immutable", 1))
        contracted = ASTSource(source.fn, dict(source.signature),
                               constexprs=dict(source.constants), attrs=attrs)
        cache[config] = triton.compile(contracted, target=baseline.metadata.target,
                                       options={"num_warps": warps, **(extra_options or {})})
    compiled = cache[config]
    # The launcher's signature includes constexpr entries and removes them itself.
    compiled[(m, 1, 1)](*bound)
    return compiled


def validate(data, out):
    for name in ["act_storage", "prob_storage"]:
        storage = out[name]
        guard = out["guard"]
        assert bool((storage[:guard] == 37).all()) and bool((storage[-guard:] == 37).all()), name
    torch.testing.assert_close(out["act"], data["act_ref"], rtol=0, atol=0)
    # FP32 dot reduction vs FP64 oracle. BF16 inputs are exactly representable;
    # finite reduction-order error is allowed, without changing tolerances by case.
    torch.testing.assert_close(out["prob"], data["prob_ref"], rtol=3e-5, atol=3e-3)
    return {"guards": True, "act_exact_reference": True,
            "prob_max_abs_error": float((out["prob"] - data["prob_ref"]).abs().max()),
            "prob_reference_rtol": 3e-5, "prob_reference_atol": 3e-3}


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def save_artifacts(kernel, path):
    path.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind in ["ttir", "ttgir", "llir", "ptx", "cubin"]:
        value = kernel.asm.get(kind)
        if value is None:
            continue
        dest = path / ("kernel." + kind)
        if isinstance(value, bytes):
            dest.write_bytes(value)
        else:
            dest.write_text(str(value))
        files[kind] = {"path": str(dest), "sha256": sha(dest), "bytes": dest.stat().st_size}
    disassembler = shutil.which("cuobjdump")
    if disassembler and "cubin" in files:
        disassembly = subprocess.run(
            [disassembler, "--dump-sass", str(path / "kernel.cubin")],
            capture_output=True, text=True, check=False)
        if disassembly.returncode == 0:
            dest = path / "kernel.sass"
            dest.write_text(disassembly.stdout)
            files["sass"] = {"path": str(dest), "sha256": sha(dest), "bytes": dest.stat().st_size}
    md = kernel.metadata._asdict() if hasattr(kernel.metadata, "_asdict") else vars(kernel.metadata)
    resource = {k: v for k, v in md.items() if isinstance(v, (str, int, float, bool, type(None)))}
    resource.update({"n_regs": getattr(kernel, "n_regs", None), "n_spills": getattr(kernel, "n_spills", None)})
    (path / "metadata.json").write_text(json.dumps(resource, indent=2))
    ptx = str(kernel.asm.get("ptx", ""))
    return {"files": files, "resources": resource,
            "static_global_loads": len(re.findall(r"\bld\.global", ptx)),
            "static_global_stores": len(re.findall(r"\bst\.global", ptx))}


def paired_timing(functions, rounds, replay, warmup_ms, seed):
    graphs = {}
    # Warm up every variant, compile before capture, then warm all replay graphs.
    for name, fn in functions.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(replay):
                fn()
        graphs[name] = graph
    until = time.monotonic() + warmup_ms / 1000.0
    while time.monotonic() < until:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    records = {name: [] for name in graphs}
    rng = random.Random(seed)
    order_log = []
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        order_log.append(order)
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            records[name].append(start.elapsed_time(end) * 1000.0 / replay)
    return {"samples_us": records,
            "medians_us": {name: statistics.median(values) for name, values in records.items()},
            "round_orders": order_log, "graph_invocations": replay, "warmup_ms": warmup_ms}


def case_key(shape):
    return "m%d-h%d-k%d-e%d" % tuple(shape)


def run_case(args, shape, config, seed, padding=False, boundary=False):
    data = make_data(shape, seed, args.dtype, boundary=boundary, padding=padding)
    folder = args.output / case_key(shape) / ("b%d-w%d-seed%d" % (*config, seed))
    result = {"shape": list(shape), "config": list(config), "seed": seed,
              "config_family": "published" if tuple(config) in PUBLISHED_CONFIGS else "expanded",
              "single_hidden_tile": config[0] >= shape[1],
              "padding": padding, "boundary": boundary,
              "metadata_matches_upstream_producer": data["metadata_matches_upstream_producer"],
              "capacity": data["capacity"], "padded_rows": data["rows"], "variants": {}}
    outputs = {}
    for variant in args.variants:
        outputs[variant] = make_outputs(data)
        fn = lambda v=variant: launch(data, outputs[v], v, config)
        kernel = fn()
        torch.cuda.synchronize()
        artifacts = save_artifacts(kernel, folder / variant)
        check = validate(data, outputs[variant])
        result["variants"][variant] = {"validation_before": check,
                                         "artifacts": artifacts}
    first = outputs[args.variants[0]]
    for out in outputs.values():
        assert byte_equal(first["act"], out["act"]), "variant act gradient bytes differ"
        assert byte_equal(first["prob"], out["prob"]), "variant probability gradient bytes differ"
    result["variants_byte_equal"] = True
    if args.mode != "boundaries" and not args.correctness_only:
        # Time every variant on identical physical inputs AND outputs. Distinct
        # output allocations can create persistent cache/address-placement bias,
        # even between kernels with byte-identical machine-code sections.
        timing_functions = {
            variant: (lambda v=variant: launch(data, first, v, config))
            for variant in args.variants}
        result["timing_binding_validation"] = {}
        for variant, fn in timing_functions.items():
            fn()
            torch.cuda.synchronize()
            result["timing_binding_validation"][variant] = validate(data, first)
            assert byte_equal(first["act"], outputs[variant]["act"])
            assert byte_equal(first["prob"], outputs[variant]["prob"])
        result["timing"] = paired_timing(timing_functions, args.rounds, args.replay, args.warmup_ms, seed)
        result["timing"]["shared_output_allocation"] = True
        med = result["timing"]["medians_us"]
        if "original" in med:
            result["speedups"] = {name: med["original"] / value for name, value in med.items()}
            result["speedup"] = result["speedups"].get("hoisted")
    for variant in args.variants:
        # Validate after replay and again with fresh poisoned gradient outputs.
        result["variants"][variant]["validation_after"] = validate(data, outputs[variant])
        fresh = make_outputs(data)
        launch(data, fresh, variant, config)
        torch.cuda.synchronize()
        result["variants"][variant]["validation_fresh"] = validate(data, fresh)
        assert byte_equal(fresh["act"], first["act"])
        assert byte_equal(fresh["prob"], first["prob"])
    result["status"] = "passed"
    return result


def summary(rows):
    out = {}
    grouped = {}
    for row in rows:
        if row.get("status") != "passed" or "timing" not in row:
            continue
        grouped.setdefault((tuple(row["shape"]), tuple(row["config"])), []).append(row)
    for (shape, config), samples in grouped.items():
        variants = samples[0]["timing"]["medians_us"]
        med = {v: statistics.median(r["timing"]["medians_us"][v] for r in samples) for v in variants}
        case = out.setdefault(case_key(shape), {"shape": list(shape), "configs": []})
        case["configs"].append({"config": list(config), "seed_count": len(samples), "medians_us": med,
                                  "config_family": "published" if config in PUBLISHED_CONFIGS else "expanded",
                                  "speedups": {v: med.get("original", math.nan) / t for v, t in med.items()},
                                  "speedup": med.get("original", math.nan) / med.get("hoisted", math.nan)})
    for case in out.values():
        case["best"] = {}
        variants = case["configs"][0]["medians_us"]
        for v in variants:
            best = min(case["configs"], key=lambda r: r["medians_us"][v])
            case["best"][v] = {"config": best["config"], "median_us": best["medians_us"][v]}
        case["best_by_config_family"] = {}
        for family in ("published", "expanded"):
            candidates = [r for r in case["configs"] if r["config_family"] == family]
            if candidates:
                family_best = case["best_by_config_family"][family] = {}
                for v in variants:
                    best = min(candidates, key=lambda r: r["medians_us"][v])
                    family_best[v] = {"config": best["config"], "median_us": best["medians_us"][v]}
        if "original" in case["best"] and "hoisted" in case["best"]:
            case["best_to_best_speedup"] = case["best"]["original"]["median_us"] / case["best"]["hoisted"]["median_us"]
        if "original" in case["best"]:
            case["best_to_best_speedups"] = {
                v: case["best"]["original"]["median_us"] / result["median_us"]
                for v, result in case["best"].items()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["screen", "confirm", "boundaries"], default="screen")
    parser.add_argument("--from-screen", type=Path)
    parser.add_argument("--case", action="append", help="M:H:K:E; repeat for multiple cases")
    parser.add_argument("--config", action="append", help="BLOCK_SIZE:WARPS; repeat")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--replay", type=int, default=32)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS) + ["contract"], default=list(VARIANTS))
    parser.add_argument("--compiler-contract", action="store_true",
                        help="Also compile unchanged source with arg2 tt.experimental_immutable=1; needs experimental local compiler")
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if args.compiler_contract and "contract" not in args.variants:
        args.variants.append("contract")
    args.output.mkdir(parents=True, exist_ok=False)
    dtype_name = args.dtype
    args.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    args.rounds = args.rounds or (15 if args.mode == "confirm" else 7)
    seeds = [int(x) for x in (args.seeds or ("0,1,2" if args.mode == "confirm" else "0")).split(",")]
    shapes = [tuple(map(int, s.split(":"))) for s in args.case] if args.case else [
        (1024, 8192, 2, 16), (512, 16384, 4, 16), (256, 32768, 8, 16)]
    configs = [tuple(map(int, c.split(":"))) for c in args.config] if args.config else [
        (64, 4), (128, 4), (256, 4), (512, 4), (1024, 4), (2048, 4), (4096, 4),
        (1024, 8), (4096, 8)]
    jobs = [(shape, config, False, False) for shape in shapes for config in configs]
    if not args.config:
        # Controls for whether ordinary larger-tile tuning removes the hidden
        # loop and its repeated scalar load. Keep the published menu identifiable.
        for shape in shapes:
            controls = {(b, w) for b in (8192, 16384, shape[1]) for w in (4, 8)}
            jobs.extend((shape, c, False, False) for c in sorted(controls - set(configs)))
    if args.from_screen:
        screen = json.loads(args.from_screen.read_text())
        jobs = []
        for case in screen["summary"].values():
            selected = {tuple(v["config"]) for v in case["best"].values()}
            for family in case.get("best_by_config_family", {}).values():
                selected.update(tuple(v["config"]) for v in family.values())
            jobs.extend((tuple(case["shape"]), c, False, False) for c in sorted(selected))
    if args.mode == "boundaries":
        jobs = [((19, 1537, 4, 8), (512, 4), True, True),
                ((7, 1024, 2, 8), (2048, 4), False, True),
                ((33, 4096, 8, 16), (1024, 4), True, True)]
    report = {"experiment": "transformer_engine_unpermute_backward_probability_hoist",
              "mode": args.mode, "dtype": dtype_name, "argv": sys.argv,
              "source_audit": source_audit(), "torch": torch.__version__,
              "compiler_contract": "arg2 contents immutable through all aliases during this invocation" if "contract" in args.variants else None,
              "triton": triton.__version__, "triton_path": triton.__file__,
              "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "timing_scope": "kernel only, both gradients; excludes row-map creation and allocation; padded cases correctness-only",
              "config_families": {"published": "Upstream BLOCK_SIZE64/128/256/512/1024/2048/4096, num_warps4",
                                  "expanded": "Additional8-warp configs and BLOCK_SIZE8192/16384/H with4/8 warps"},
              "rows": []}
    try:
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass
    destination = args.output / "results.json"
    for shape, config, padding, boundary in jobs:
        for seed in seeds:
            print(f"TE {shape=} {config=} {seed=}", flush=True)
            try:
                row = run_case(args, shape, config, seed, padding, boundary)
                print(json.dumps({"status": row["status"], "speedup": row.get("speedup"),
                                  "timing": row.get("timing", {}).get("medians_us")}), flush=True)
            except Exception as error:
                row = {"shape": list(shape), "config": list(config), "seed": seed,
                       "status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
                print(row["traceback"], flush=True)
            report["rows"].append(row)
            report["summary"] = summary(report["rows"])
            destination.write_text(json.dumps(report, indent=2) + "\n")
    report["all_passed"] = all(r["status"] == "passed" for r in report["rows"])
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
