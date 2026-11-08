#!/usr/bin/env python3
"""Reconstructed private-reduction ownership experiment on Tritonbench LayerNorm.

The historical harness was not committed. This reconstruction extracts the exact
registered tutorial provider and builds the owner from its row-computation AST.
No native compiler transformation is used by these source variants.
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton._C import libtriton

from coverage_experiment import digest, paired_device_times, save_json


FLAGS = {
    "enable_aref_release_scheduling": False,
    "enable_tmem_release_scheduling": False,
    "enable_tmem_packing": False,
    "enable_tmem_single_buffer": False,
    "enable_fp_fusion": False,
    "enable_indexed_update_specialization": False,
    "enable_private_atomic_relaxation": False,
}


@triton.jit
def ordered_reference(RowDW, RowDB, DW, DB, M: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
                      BLOCK: tl.constexpr):
    owner = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    dw = tl.load(RowDW + owner * N + cols, mask, other=0).to(DW.dtype.element_ty)
    db = tl.load(RowDB + owner * N + cols, mask, other=0).to(DB.dtype.element_ty)
    for row in range(owner + G, M, G):
        new_dw = tl.load(RowDW + row * N + cols, mask, other=0)
        new_db = tl.load(RowDB + row * N + cols, mask, other=0)
        # Operand order and scratch rounding match the original critical region.
        dw = (new_dw + dw.to(tl.float32)).to(DW.dtype.element_ty)
        db = (new_db + db.to(tl.float32)).to(DB.dtype.element_ty)
    tl.store(DW + owner * N + cols, dw, mask)
    tl.store(DB + owner * N + cols, db, mask)


def extract(source, output):
    tree = ast.parse(source.read_text())
    names = ("_layer_norm_fwd_fused", "_layer_norm_bwd_dx_fused", "_layer_norm_bwd_dwdb")
    functions = [copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
                 for name in names]
    baseline = functions[1]
    prefix = baseline.body[:next(i for i, n in enumerate(baseline.body) if isinstance(n, ast.While))]
    # The row calculation is reproduced mechanically, not re-derived from math.
    # Row/cols/mask and the immutable weight vector come from the owner caller;
    # scratch/lock address setup belongs to the reduction protocol, not the row.
    skip = {"row", "cols", "mask", "lock_id", "Count", "DW", "DB", "w"}
    row_body = []
    for node in prefix:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in skip:
                continue
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "Lock":
            continue
        row_body.append(copy.deepcopy(node))
    row_body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id="partial_dw", ctx=ast.Load()),
                                                  ast.Name(id="partial_db", ctx=ast.Load())], ctx=ast.Load())))
    helper = ast.parse("@triton.jit\ndef owner_row(DX, DY, X, Mean, Rstd, stride, N, row, cols, mask, w):\n    pass\n").body[0]
    helper.body = row_body
    ast.fix_missing_locations(helper)
    owner_source = '''
@triton.jit
def owner_same_grid(DX, DY, DW, DB, X, W, B, Mean, Rstd, Lock, stride, N, eps,
                    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    owner = tl.program_id(0)
    rows = tl.num_programs(0)
    if owner < GROUP_SIZE_M:
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        first_dw, first_db = owner_row(DX, DY, X, Mean, Rstd, stride, N, owner, cols, mask, w)
        acc_dw = first_dw.to(DW.dtype.element_ty)
        acc_db = first_db.to(DB.dtype.element_ty)
        for row in range(owner + GROUP_SIZE_M, rows, GROUP_SIZE_M):
            new_dw, new_db = owner_row(DX, DY, X, Mean, Rstd, stride, N, row, cols, mask, w)
            acc_dw = (new_dw + acc_dw.to(tl.float32)).to(DW.dtype.element_ty)
            acc_db = (new_db + acc_db.to(tl.float32)).to(DB.dtype.element_ty)
        tl.store(DW + owner * N + cols, acc_dw, mask)
        tl.store(DB + owner * N + cols, acc_db, mask)
        tl.store(Lock + GROUP_SIZE_M + owner, 1)
        tl.store(Lock + owner, 0)
'''
    directory = output / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "extracted_and_owner.py"
    path.write_text("# Extracted from Tritonbench tutorial.py; exact upstream file and license retained.\n"
                    "import torch\nimport triton\nimport triton.language as tl\n\n" +
                    "\n\n".join(ast.unparse(n) for n in functions + [helper]) + "\n" + owner_source)
    check = ast.parse(path.read_text())
    for original in functions:
        actual = next(n for n in check.body if isinstance(n, ast.FunctionDef) and n.name == original.name)
        assert ast.dump(actual) == ast.dump(original)
    (directory / "tutorial.py").write_text(source.read_text())
    spec = importlib.util.spec_from_file_location("layernorm_ownership_extracted", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audit = {"source_sha256": digest(source), "historical_source_hash_match":
             digest(source) == "58716180088be8a05e3e21fb2bab2db2bc5e7780fd3dd7e680e682a82ecdd048",
             "extracted_sha256": digest(path), "original_ast_equal": names,
             "owner_row_math": "Original prefix AST, with pointer setup and uniform row/columns/weight supplied by owner.",
             "recovery": "Historical harness/kernels absent from all committed trees; reconstructed from provider and archived protocol.",
             "signature_preserved": ast.dump(baseline.args) == ast.dump(next(n for n in check.body if isinstance(n, ast.FunctionDef) and n.name == "owner_same_grid").args)}
    assert audit["signature_preserved"]
    return module, audit


class Guarded:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.count = math.prod(shape)
        self.storage = torch.full((self.count + 32,), 17, device="cuda", dtype=dtype)
        self.tensor = self.storage[16:-16].reshape(shape)

    def check(self):
        assert torch.all(self.storage[:16] == 17) and torch.all(self.storage[-16:] == 17)


def equal_bits(actual, expected):
    return bool(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))


def direct_round(values, dtype):
    """Correct any double rounding in double->float->half/BF16 conversions.

    Choose the nearest target value between the initial conversion and its
    adjacent value, breaking exact midpoint ties by the low significand bit.
    The experiment's finite normal-range reference values need no overflow rule.
    """
    rounded = values.to(dtype)
    if dtype == torch.float32:
        return rounded
    bits = rounded.contiguous().view(torch.int16).to(torch.int32) & 65535
    candidate = rounded.double()
    up = torch.where(bits == 32768, 1, torch.where(bits < 32768, bits + 1, bits - 1))
    down = torch.where(bits == 0, 32769, torch.where(bits < 32768, bits - 1, bits + 1))
    adjacent_bits = torch.where(candidate < values, up, down)
    adjacent = adjacent_bits.to(torch.int16).view(dtype)
    first_distance = (values - candidate).abs()
    adjacent_distance = (values - adjacent.double()).abs()
    choose_adjacent = (adjacent_distance < first_distance) | (
        (adjacent_distance == first_distance) & ((adjacent_bits & 1) == 0))
    return torch.where(choose_adjacent, adjacent, rounded)


class Case:
    def __init__(self, module, m, n, dtype, scratch_dtype, seed, native_enabled=False):
        self.module, self.m, self.n, self.dtype, self.scratch_dtype = module, m, n, dtype, scratch_dtype
        torch.manual_seed(seed)
        self.x = torch.randn((m, n), device="cuda", dtype=dtype)
        self.dy = torch.randn_like(self.x)
        self.w = torch.randn(n, device="cuda", dtype=dtype)
        self.b = torch.zeros_like(self.w)
        self.mean = torch.empty(m, device="cuda", dtype=torch.float32)
        self.rstd = torch.empty_like(self.mean)
        self.y = torch.empty_like(self.x)
        self.block = triton.next_power_of_2(n)
        self.saved_inputs = [x.clone() for x in (self.x, self.dy, self.w)]
        module._layer_norm_fwd_fused[(m,)](self.x, self.y, self.w, self.b, self.mean, self.rstd, n, n, 1e-5,
                                           self.block, num_warps=4, **FLAGS)
        self.contexts = {}
        self.rows_by_warps = {}
        self.native_enabled = native_enabled
        self.modes = ("locked", "owner", "native") if native_enabled else ("locked", "owner")
        self.math_reference = self.reference()

    def reference(self):
        x, dy, w = self.x.double(), self.dy.double(), self.w.double()
        xhat = (x - self.mean.double()[:, None]) * self.rstd.double()[:, None]
        wdy = dy * w
        c1 = (xhat * wdy).sum(1, keepdim=True) / self.n
        c2 = wdy.sum(1, keepdim=True) / self.n
        dx = (wdy - (xhat * c1 + c2)) * self.rstd.double()[:, None]
        return dx, (dy * xhat).sum(0), dy.sum(0)

    def context(self, group):
        if group not in self.contexts:
            buffers = {"dx": Guarded((self.m, self.n), self.dtype),
                       "dw": Guarded((group, self.n), self.scratch_dtype),
                       "db": Guarded((group, self.n), self.scratch_dtype),
                       "lock": Guarded((2 * group,), torch.int32),
                       "final_dw": Guarded((self.n,), self.dtype),
                       "final_db": Guarded((self.n,), self.dtype)}
            args = (buffers["dx"].tensor, self.dy, buffers["dw"].tensor, buffers["db"].tensor, self.x, self.w,
                    self.b, self.mean, self.rstd, buffers["lock"].tensor, self.n, self.n, 1e-5, group, self.block)
            self.contexts[group] = {"buffers": buffers, "args": args, "compiled": {}}
        return self.contexts[group]

    def launch(self, mode, group, warps):
        ctx = self.context(group)
        buf = ctx["buffers"]
        buf["lock"].tensor.zero_()
        if mode == "native":
            from triton.experimental.semantic_contracts.native import compile_private_reduction_ownership
            if ("plan", warps) not in ctx:
                ctx[("plan", warps)] = compile_private_reduction_ownership(
                    self.module._layer_norm_bwd_dx_fused, ctx["args"], grid=(self.m,),
                    private_args=("DX", "DW", "DB", "Lock"), zero_initialized_args=("Lock",),
                    options=dict(FLAGS, num_warps=warps))
            plan = ctx[("plan", warps)]
            assert plan.report()["transformed"], "native protocol recognition declined"
            plan(*ctx["args"])
            ctx["compiled"][(mode, warps)] = plan.optimized
        else:
            kernel = self.module._layer_norm_bwd_dx_fused if mode == "locked" else self.module.owner_same_grid
            ctx["compiled"][(mode, warps)] = kernel[(self.m,)](*ctx["args"], num_warps=warps, **FLAGS)
        ctx["compiled"]["final"] = self.module._layer_norm_bwd_dwdb[(triton.cdiv(self.n, 128),)](
            buf["dw"].tensor, buf["db"].tensor, buf["final_dw"].tensor, buf["final_db"].tensor,
            min(group, self.m), self.n, 32, 128, num_warps=4, **FLAGS)
        return tuple(buf[name].tensor for name in ("dx", "final_dw", "final_db"))

    def native(self):
        return torch.ops.aten.native_layer_norm_backward.default(
            self.dy, self.x, [self.n], self.mean[:, None], self.rstd[:, None], self.w, self.b, [True, True, True])

    def record_accuracy(self, outputs):
        result = {}
        for name, output, expected in zip(("dx", "dw", "db"), outputs, self.math_reference):
            assert torch.isfinite(output).all()
            rounded = direct_round(expected, self.dtype)
            error = (output.double() - rounded.double()).abs()
            result[name] = {"max_abs_error_to_rounded_fp64": float(error.max()),
                            "historical_atol_001_pass": bool(torch.all(error <= 0.01)),
                            "max_abs_error_to_unrounded_fp64": float((output.double() - expected).abs().max())}
        return result

    def row_contributions(self, warps):
        if warps not in self.rows_by_warps:
            # G=M gives each row exclusive scratch, capturing the unchanged
            # producer's exact FP32 contributions before any inter-row addition.
            dw, db = torch.empty_like(self.x, dtype=torch.float32), torch.empty_like(self.x, dtype=torch.float32)
            dx = torch.empty_like(self.x)
            lock = torch.zeros(2 * self.m, device="cuda", dtype=torch.int32)
            args = (dx, self.dy, dw, db, self.x, self.w, self.b, self.mean, self.rstd, lock,
                    self.n, self.n, 1e-5, self.m, self.block)
            self.module._layer_norm_bwd_dx_fused[(self.m,)](*args, num_warps=warps, **FLAGS)
            self.rows_by_warps[warps] = (dx, dw, db)
        return self.rows_by_warps[warps]

    def validate(self, mode, group, warps, ordered=True):
        ctx = self.context(group)
        buf = ctx["buffers"]
        for key in ("dx", "dw", "db", "final_dw", "final_db"):
            buf[key].tensor.fill_(float("nan"))
        outputs = self.launch(mode, group, warps)
        actual = tuple(x.clone() for x in outputs)
        active = min(group, self.m)
        assert torch.all(buf["lock"].tensor[:group] == 0)
        assert torch.all(buf["lock"].tensor[group:group + active] == 1)
        assert torch.all(buf["lock"].tensor[group + active:] == 0)
        assert torch.isnan(buf["dw"].tensor[active:]).all() and torch.isnan(buf["db"].tensor[active:]).all()
        for item in buf.values():
            item.check()
        check = self.record_accuracy(actual)
        if mode in ("owner", "native") and ordered:
            row_dx, row_dw, row_db = self.row_contributions(warps)
            ref_dw, ref_db = torch.empty_like(buf["dw"].tensor), torch.empty_like(buf["db"].tensor)
            ordered_reference[(active, triton.cdiv(self.n, 128))](row_dw, row_db, ref_dw, ref_db,
                                                                 self.m, self.n, group, 128, num_warps=4, **FLAGS)
            assert equal_bits(buf["dw"].tensor[:active], ref_dw[:active]), "owner DW differs from selected serialization"
            assert equal_bits(buf["db"].tensor[:active], ref_db[:active]), "owner DB differs from selected serialization"
            assert equal_bits(actual[0], row_dx), "owner DX differs from unchanged row computation"
            # Same unchanged final reduction must therefore produce identical results.
            final_dw, final_db = torch.empty_like(actual[1]), torch.empty_like(actual[2])
            self.module._layer_norm_bwd_dwdb[(triton.cdiv(self.n, 128),)](ref_dw, ref_db, final_dw, final_db,
                                                                        active, self.n, 32, 128, num_warps=4, **FLAGS)
            assert equal_bits(actual[1], final_dw) and equal_bits(actual[2], final_db)
            check["exact_selected_serialization"] = True
        for old, current in zip(self.saved_inputs, (self.x, self.dy, self.w)):
            assert equal_bits(old, current)
        return check


def evidence(kernel, output, label):
    import re
    kernel._init_handles()
    directory = output / "code"
    directory.mkdir(exist_ok=True)
    for kind in ("ttir", "ttgir", "ptx"):
        (directory / f"{label}.{kind}").write_text(kernel.asm[kind])
    return {"registers": kernel.n_regs, "spills": kernel.n_spills, "shared": kernel.metadata.shared,
            "atomic_sites": len(re.findall(r"\b(?:atom|red)\.", kernel.asm["ptx"])),
            "barrier_sites": len(re.findall(r"\bbar\.sync\b", kernel.asm["ptx"])),
            "cubin_sha256": hashlib.sha256(kernel.asm["cubin"]).hexdigest(), "artifact_prefix": f"code/{label}"}


def one_configuration(case, group, warps, seed, settings):
    pre = {mode: case.validate(mode, group, warps) for mode in case.modes}
    functions = {mode: lambda mode=mode: case.launch(mode, group, warps) for mode in case.modes}
    functions["torch_native"] = case.native
    times = paired_device_times(functions, seed=seed, **settings)
    post = {mode: case.validate(mode, group, warps) for mode in case.modes}
    report = case.context(group)[("plan", warps)].report() if case.native_enabled else None
    return {"group": group, "warps": warps, "seed": seed, "timing": times, "pre": pre, "post": post,
            "native_report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path,
                        default=Path("/home/ryandeng/tpaa/tritonbench/tritonbench/operators/layer_norm/tutorial.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "smoke", "screen", "confirm"), default="smoke")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--scratch-dtype", choices=("float32", "input"), default="float32")
    parser.add_argument("--groups", default="64,128,256,512,1024,4096")
    parser.add_argument("--warps", default="1,2,4,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--native", action="store_true", help="Also verify the rebuilt native ownership pass")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    module, audit = extract(args.tutorial, args.output)
    save_json(args.output / "source_audit.json", audit)
    if args.mode == "prepare":
        return
    m, n = args.rows, args.cols
    dtype = getattr(torch, args.dtype)
    scratch = torch.float32 if args.scratch_dtype == "float32" else dtype
    groups, warps = [int(x) for x in args.groups.split(",")], [int(x) for x in args.warps.split(",")]
    settings = {"rounds": 15 if args.mode == "confirm" else 5, "repetitions": 32 if args.mode == "confirm" else 8,
                "warmup_ms": 200 if args.mode == "confirm" else 50}
    fingerprints = {"harness": digest(__file__), "libtriton": digest(libtriton.__file__), "tutorial": digest(args.tutorial)}
    records = []
    metadata = {"shape": [m, n], "dtype": args.dtype, "scratch_dtype": str(scratch), "flags": FLAGS,
                "settings": settings, "fingerprints": fingerprints, "source_audit": audit,
                "device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
                "scope": "Same M-program grid and signature; lock/count reset and unchanged final reduction included; allocations/forward/guards excluded.",
                "numerics": "Owner must match a permitted original increasing-row serialization bitwise. FP64 saved-statistics oracle and historical0.01 diagnostic reported separately; no tolerance enlarged.",
                "strong_provider": "torch aten.native_layer_norm_backward; Quack is unavailable in this environment."}
    for seed in map(int, args.seeds.split(",")):
        case = Case(module, m, n, dtype, scratch, seed, args.native)
        native_check = case.record_accuracy(case.native())
        for group in groups:
            for warp in warps:
                record = one_configuration(case, group, warp, seed, settings)
                records.append(record)
                print(m, n, args.dtype, group, warp, seed, record["timing"]["median_us"], flush=True)
                if seed == int(args.seeds.split(",")[0]) and group == groups[0] and warp == warps[0]:
                    ctx = case.context(group)
                    record["code"] = {mode: evidence(ctx["compiled"][(mode, warp)], args.output, mode)
                                      for mode in case.modes}
                    record["code"]["final"] = evidence(ctx["compiled"]["final"], args.output, "final")
                save_json(args.output / "results.json", {"metadata": metadata, "cases": records})
        # Compare whole backward against a stronger implementation using the same
        # saved statistics; allocation occurs during capture rather than replay.
        times = paired_device_times({"torch_native": case.native}, seed=seed, **settings)
        save_json(args.output / f"torch-native-seed{seed}.json", {"timing": times, "accuracy": native_check})
    assert fingerprints["libtriton"] == digest(libtriton.__file__), "compiler changed during experiment"
    summary = {"metadata": metadata, "cases": len(records), "best_tested": {}}
    for mode in case.modes:
        candidates = {(r["group"], r["warps"]) for r in records}
        medians = {cfg: statistics.median(r["timing"]["median_us"][mode] for r in records
                                         if (r["group"], r["warps"]) == cfg) for cfg in candidates}
        best = min(medians, key=medians.get)
        summary["best_tested"][mode] = {"group": best[0], "warps": best[1], "us": medians[best]}
    summary["best_tested_speedup"] = summary["best_tested"]["locked"]["us"] / summary["best_tested"]["owner"]["us"]
    save_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
