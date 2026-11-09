"""Unchanged AITER RMSNorm adapters for cross-kernel pipeline validation.

The persistent source loops already request two stages. These cases test whether
contracts can retain that schedule under the dependence guard, and are a control
for automatic stage inference rather than evidence of newly inferred stages.
Backward includes the original final dgamma reduction in the measured operation.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "aiter"
REVISION = "ef0a5a26b8cb38ea1308f827985e36cde964a6ab"
SOURCE_PATH = "aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py"
GUARD = 256
SENTINEL = 7.125


def provenance():
    """Audit the actual measured functions against the retained pinned source."""
    upstream = (VENDOR / "rmsnorm_upstream.py").read_text()
    measured = (HERE / "rmsnorm_kernels.py").read_text()
    parse = lambda source: {n.name: n for n in ast.parse(source).body
                            if isinstance(n, ast.FunctionDef)}
    original, extracted = parse(upstream), parse(measured)
    mappings = {
        "_rms_norm_kernel": "rmsnorm_forward_original",
        "_rmsnorm_bwd_triton": "rmsnorm_backward_original",
        "_rmsnorm_bwd_dg_reduce_triton": "rmsnorm_dg_reduce",
    }
    bodies = {}
    for old, new in mappings.items():
        body = ast.dump(ast.Module(body=original[old].body, type_ignores=[]))
        assert body == ast.dump(ast.Module(body=extracted[new].body, type_ignores=[]))
        bodies[new] = hashlib.sha256(body.encode()).hexdigest()
    return {
        "repository": "ROCm/aiter", "revision": REVISION,
        "source_path": SOURCE_PATH,
        "source_url": f"https://github.com/ROCm/aiter/blob/{REVISION}/{SOURCE_PATH}",
        "upstream_sha256": hashlib.sha256(upstream.encode()).hexdigest(),
        "measured_module_sha256": hashlib.sha256(measured.encode()).hexdigest(),
        "body_ast_sha256": bodies, "kernel_bodies_unchanged": True,
        "wrapper_changes": "Isolated renamed entry points and ordinary @triton.jit replace repr-dependent decorators; provider dispatch is not measured.",
        "source_has_explicit_loop_stages": True,
        "auto_inference_expected": False,
        "interpretation": "Existing num_stages=2 source schedule can be restored by alias contracts; this is not a new automatic-stage opportunity.",
        "scope": "Forward writes y and FP32 rsigma; backward writes dx and final dgamma, including the unchanged auxiliary reduction.",
        "source_files": [str(HERE / "rmsnorm_kernels.py"),
                         str(VENDOR / "rmsnorm_upstream.py"),
                         str(VENDOR / "rmsnorm_wrapper_upstream.py"),
                         str(VENDOR / "LICENSE")],
    }


def cases():
    result = []
    for operation in ("forward", "backward"):
        for m, n in ((4096, 1024), (16384, 4096), (31, 1008)):
            result.append({
                "id": f"rms_{operation}_{m}x{n}",
                "family": "aiter_rmsnorm", "operation": operation,
                "M": m, "N": n, "dtype": "bfloat16", "epsilon": 1e-6,
                "source_has_explicit_loop_stages": True,
                "auto_inference_expected": False,
                "boundary": m == 31,
            })
    return result


def configurations(case):
    # The row grid is a nonpersistent competitor, not only another persistent
    # setting. SM grids may contain zero-trip CTAs in the small-row boundary.
    return [dict(warps=warps, grid=grid)
            for warps in (4, 8) for grid in ("1sm", "4sm", "rows")]


def _guarded(shape, dtype):
    import math
    import torch
    storage = torch.full((math.prod(shape) + 2 * GUARD,), SENTINEL,
                         device="cuda", dtype=dtype)
    return {"storage": storage, "value": storage[GUARD:-GUARD].view(shape)}


def prepare(case, seed):
    import torch
    provenance()
    m, n = case["M"], case["N"]
    # Row alignment must satisfy the unchanged source's multiple_of(16) hints.
    assert n % 16 == 0
    dtype = getattr(torch, case["dtype"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu = {
        "x": torch.randn((m, n), generator=generator, dtype=dtype),
        "gamma": torch.randn((n,), generator=generator, dtype=dtype),
    }
    xd = cpu["x"].double()
    rd = torch.rsqrt(xd.square().mean(-1) + case["epsilon"])
    if case["operation"] == "backward":
        cpu["dy"] = torch.randn((m, n), generator=generator, dtype=dtype)
        # Backward consumes saved FP32 statistics. The oracle deliberately uses
        # those supplied values, rather than assuming an exact FP64 statistic.
        cpu["saved_rsigma"] = rd.float()
    inputs = {name: value.cuda() for name, value in cpu.items()}
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if case["operation"] == "forward":
        buffers = {"y": _guarded((m, n), dtype),
                   "rsigma": _guarded((m,), torch.float32)}
        reference = {"y": xd * rd[:, None] * cpu["gamma"].double()[None, :],
                     "rsigma": rd}
    else:
        buffers = {"dx": _guarded((m, n), dtype),
                   "dg_tmp": _guarded((max(m, 4 * sm_count), n), torch.float32),
                   "dgamma": _guarded((n,), dtype)}
        dyd, gd, saved = cpu["dy"].double(), cpu["gamma"].double(), cpu["saved_rsigma"].double()
        grad_sum = (dyd * xd * gd[None, :]).sum(-1)
        reference = {
            "dx": dyd * saved[:, None] * gd[None, :]
                  - saved[:, None].pow(3) * xd * (grad_sum / n)[:, None],
            "dgamma": (dyd * xd * saved[:, None]).sum(0),
        }
    result = dict(inputs=inputs, cpu=cpu, buffers=buffers, reference=reference,
                  sm_count=sm_count, seed=seed)
    reset(case, result)
    return result


def _check_groups(args, groups):
    # Validate complete allocations, not merely the first logical element or
    # equal/different base addresses. Same-group inputs may alias by contract.
    items = list(groups.items())
    for position, (i, group) in enumerate(items):
        first = args[i].untyped_storage()
        lo, hi = first.data_ptr(), first.data_ptr() + first.nbytes()
        for j, other_group in items[position + 1:]:
            if group == other_group:
                continue
            second = args[j].untyped_storage()
            other_lo, other_hi = second.data_ptr(), second.data_ptr() + second.nbytes()
            assert hi <= other_lo or other_hi <= lo, (i, j, "alias contract violated")


def launch_spec(case, data, config):
    import triton
    import rmsnorm_kernels as kernels
    m, n = case["M"], case["N"]
    grid = m if config["grid"] == "rows" else int(config["grid"][:-2]) * data["sm_count"]
    block = triton.next_power_of_2(n)
    inputs = data["inputs"]
    buffers = {name: entry["value"] for name, entry in data["buffers"].items()}
    if case["operation"] == "forward":
        fn = kernels.rmsnorm_forward_original
        args = [inputs["x"], buffers["y"], inputs["gamma"], buffers["rsigma"],
                n, n, m, n, case["epsilon"], block, False]
        groups = {0: 0, 2: 0, 1: 1, 3: 2}
        trailing = []
    else:
        fn = kernels.rmsnorm_backward_original
        args = [inputs["dy"], inputs["x"], inputs["gamma"], inputs["saved_rsigma"],
                buffers["dx"], buffers["dg_tmp"], n, n, m, n, block, False, grid]
        groups = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2}
        trailing = [dict(fn=kernels.rmsnorm_dg_reduce,
                         args=[buffers["dg_tmp"], buffers["dgamma"], n, grid, n, 128, 64],
                         grid=(triton.cdiv(n, 64),),
                         options=dict(num_warps=4, num_stages=1),
                         groups={0: 0, 1: 1},
                         purpose="Required final dgamma reduction, identical across variants")]
    _check_groups(args, groups)
    for spec in trailing:
        _check_groups(spec["args"], spec["groups"])
    return dict(fn=fn, args=args, grid=(grid,),
                options=dict(num_warps=config["warps"], num_stages=2),
                groups=groups, trailing_specs=trailing,
                config=config, source_has_explicit_loop_stages=True,
                auto_inference_expected=False,
                host_full_storage_disjointness_checked=True)


def reset(case, data):
    """Poison for validation only; all measured outputs are fully overwritten."""
    for entry in data["buffers"].values():
        entry["value"].fill_(float("nan"))


def output_tensors(data):
    return {name: entry["value"] for name, entry in data["buffers"].items()
            if name != "dg_tmp"}


def validate(case, data, baseline=None):
    import torch
    checks = {}
    for name, tensor in output_tensors(data).items():
        actual = tensor.cpu()
        expected = data["reference"][name]
        if name == "rsigma":
            atol, rtol = 2e-6, 2e-6
        elif name == "dgamma":
            atol, rtol = (2e-3, 2e-4) if actual.dtype == torch.float32 else (0.125, 0.02)
        else:
            atol, rtol = (2e-5, 2e-5) if actual.dtype == torch.float32 else (0.02, 0.02)
        torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=rtol)
        difference = (actual.double() - expected).abs()
        checks[name] = dict(max_absolute_error=float(difference.max()), atol=atol, rtol=rtol)
        if baseline is not None:
            assert torch.equal(actual.view(torch.uint8), baseline[name].cpu().view(torch.uint8)), name
    for name, entry in data["buffers"].items():
        storage = entry["storage"]
        assert bool((storage[:GUARD] == SENTINEL).all()), (name, "leading guard")
        assert bool((storage[-GUARD:] == SENTINEL).all()), (name, "trailing guard")
    for name, value in data["inputs"].items():
        assert torch.equal(value.cpu(), data["cpu"][name]), (name, "input modified")
    return dict(outputs=checks, independent_fp64_oracle=True,
                same_config_bitwise_equal_to_baseline=baseline is not None,
                guards_passed=True, inputs_unchanged=True,
                complete_backward_includes_final_dgamma=case["operation"] == "backward")


if __name__ == "__main__":
    # CPU-only source audit; the root experiment owns GPU execution.
    print(json.dumps(dict(provenance=provenance(), cases=cases()), indent=2))
