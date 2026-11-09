"""Pinned FlagGems bilinear resize adapter for contract-pipeline experiments.

The output grid exactly tiles OH/OW. The upstream modulo-indexed kernel is not
used with partial tiles, which could introduce overlapping CTA output writes.
The function body and interpolation arithmetic are unchanged.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor/cross_pipeline"
REVISION = "3f1f896fd90f5db42d167d64fcb9691326071bc2"
SOURCE_PATH = "src/flag_gems/ops/upsample_bilinear2d.py"
_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "cross_pipeline_upsample_kernel", VENDOR / "upsample_bilinear2d_kernel.py")
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    return _MODULE


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    upstream = ast.parse((VENDOR / "upsample_bilinear2d.upstream.py").read_text())
    copied = ast.parse((VENDOR / "upsample_bilinear2d_kernel.py").read_text())
    audited = {}
    for name in ("upsample_bilinear2d_kernel", "bilinear_reciprocal_scale"):
        a = next(n for n in upstream.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in copied.body if isinstance(n, ast.FunctionDef) and n.name == name)
        body = lambda n: ast.dump(ast.Module(body=n.body, type_ignores=[]), include_attributes=False)
        assert body(a) == body(b), name
        assert ast.dump(a.args, include_attributes=False) == ast.dump(b.args, include_attributes=False)
        audited[name] = dict(body_ast_equal=True,
                             body_ast_sha256=hashlib.sha256(body(a).encode()).hexdigest())
    return dict(
        repository="flagos-ai/FlagGems", revision=REVISION,
        source_path=SOURCE_PATH,
        source_url=f"https://github.com/flagos-ai/FlagGems/blob/{REVISION}/{SOURCE_PATH}",
        audited_functions=audited,
        files_sha256={str(p.relative_to(HERE)): _sha(p) for p in sorted(VENDOR.glob("upsample*"))},
        wrapper_changes="Remove runtime autotune wrapper/imports; preserve jit function bodies and bind tle.program_id to tl.program_id.",
        source_loop_stages=False,
        provider_config_domain=dict(BLOCK_X=[512, 256, 128, 64], BLOCK_Y=[2, 1], num_warps=[4, 8]),
        configurations="Four predeclared members of the pinned NVIDIA autotune domain, all exact tilings. The provider has no single fixed default.",
        scope="Complete NCHW bilinear output; no antialiasing, contiguous input, default scales_h/scales_w=None. Tile dimensions divide output dimensions exactly.",
        limitation="Partial modulo tiles excluded because the unchanged source may overlap output writes. No strongest-provider or full dispatch claim.",
        oracle="torch.nn.functional.interpolate on the actual input converted to float64, mode=bilinear, same size and align_corners, antialias=False.",
        oracle_tolerance=dict(atol=3e-4, rtol=1e-4),
        timed_reset_required=False)


def cases():
    return [
        dict(id="upsample_1x64_32x128_to64x256", family="upsample_bilinear2d",
             shape=[1, 64, 32, 128], output_size=[64, 256], align_corners=False,
             dtype="float32", boundary=False, cohort="calibration",
             origin="Independent 2x resize with a persistent channel loop."),
        dict(id="upsample_2x64_64x256_to128x512", family="upsample_bilinear2d",
             shape=[2, 64, 64, 256], output_size=[128, 512], align_corners=True,
             dtype="float32", boundary=False, cohort="calibration",
             origin="Independent align-corners 2x resize with batch and channel loops."),
        dict(id="upsample_32x16_128x128_to256x256", family="upsample_bilinear2d",
             shape=[32, 16, 128, 128], output_size=[256, 256], align_corners=False,
             dtype="float32", boundary=False, cohort="calibration",
             origin="Exact first upstream UPSAMPLE_SHAPES input and its supported 2x scale."),
        dict(id="upsample_boundary_single_height", family="upsample_bilinear2d",
             shape=[2, 3, 1, 33], output_size=[1, 64], align_corners=True,
             dtype="float32", boundary=True, cohort="boundary",
             origin="IH=OH=1 branch, odd input width, few batch/channel iterations; exact output tile."),
    ]


def configurations(case):
    oh, ow = case["output_size"]
    if case.get("boundary"):
        configs = [(64, 1, 4)]
    elif ow >= 512:
        configs = [(64, 2, 4), (128, 2, 4), (256, 2, 4), (512, 2, 4)]
    else:
        configs = [(64, 1, 4), (128, 1, 4), (256, 1, 4), (256, 2, 4)]
    assert all(ow % bx == 0 and oh % by == 0 for bx, by, _ in configs)
    return configs


def prepare(case, seed):
    import torch
    import torch.nn.functional as F

    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])
    output_shape = (*shape[:2], *case["output_size"])
    gen = torch.Generator().manual_seed(seed)
    input_cpu = torch.randn(shape, generator=gen, dtype=dtype)
    # This independent PyTorch implementation retains the numerical oracle in
    # float64; conversion-error tolerance is fixed before observing variants.
    reference = F.interpolate(input_cpu.double(), size=case["output_size"],
                              mode="bilinear", align_corners=case["align_corners"],
                              antialias=False)
    guard = 64
    input_storage = torch.full((input_cpu.numel() + 2 * guard,), 37.0,
                               dtype=dtype, device="cuda")
    input_tensor = input_storage[guard:-guard].view(shape)
    input_tensor.copy_(input_cpu)
    count = 1
    for dim in output_shape:
        count *= dim
    output_storage = torch.full((count + 2 * guard,), 53.0,
                                dtype=dtype, device="cuda")
    output = output_storage[guard:-guard].view(output_shape)
    output.fill_(float("nan"))
    # Full backing allocations are checked, rather than just tensor bases.
    first = input_tensor.untyped_storage()
    second = output.untyped_storage()
    assert (first.data_ptr() + first.nbytes() <= second.data_ptr()
            or second.data_ptr() + second.nbytes() <= first.data_ptr())
    return dict(input=input_tensor, input_cpu=input_cpu, reference=reference,
                output=output, input_storage=input_storage,
                output_storage=output_storage, guard=guard)


def launch_spec(case, data, config):
    import triton

    bx, by, warps = config
    n, c, ih, iw = case["shape"]
    oh, ow = case["output_size"]
    assert oh % by == 0 and ow % bx == 0
    module = _module()
    sh = module.bilinear_reciprocal_scale(ih, oh, case["align_corners"], None)
    sw = module.bilinear_reciprocal_scale(iw, ow, case["align_corners"], None)
    args = (data["output"], data["input"], n, c, oh, ow, ih, iw, sh, sw,
            case["align_corners"], bx, by)
    return dict(fn=module.upsample_bilinear2d_kernel, args=args,
                grid=(triton.cdiv(ow, bx), triton.cdiv(oh, by)),
                options=dict(num_warps=warps, num_stages=3), groups={0: 1, 1: 0},
                timed_reset_required=False)


def reset(case, data):
    data["output"].fill_(float("nan"))


def timed_reset(case, data):
    # The kernel overwrites the complete output; no state persists between calls.
    pass


def output_tensors(data):
    return {"output": data["output"]}


def validate(case, data):
    import torch

    actual = data["output"].cpu().double()
    torch.testing.assert_close(actual, data["reference"], rtol=1e-4, atol=3e-4)
    assert torch.equal(data["input"].cpu(), data["input_cpu"]), "input modified"
    guard = data["guard"]
    for name, sentinel in (("input_storage", 37.0), ("output_storage", 53.0)):
        storage = data[name]
        assert bool(torch.all(storage[:guard] == sentinel))
        assert bool(torch.all(storage[-guard:] == sentinel))
    return dict(complete_output_checked=True, input_unchanged=True,
                guards_passed=True, max_abs_error=float((actual - data["reference"]).abs().max()),
                oracle="torch_interpolate_float64", atol=3e-4, rtol=1e-4,
                exact_output_tiling=True)
