"""Pinned FlagGems adapters for the cross-kernel contract pipeline experiment.

The source kernel bodies are unchanged. All device execution belongs to the
parent experiment; this module supplies allocations, calls and independent
mathematical references only. No caller may alias outputs with any input.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor/cross_pipeline/flag_gems"
_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location("cross_pipeline_flaggems_kernels", VENDOR / "kernels.py")
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    return _MODULE


def provenance():
    data = json.loads((VENDOR / "provenance.json").read_text())
    extracted = ast.parse((VENDOR / "kernels.py").read_text())
    for record in data["files"]:
        source = VENDOR / record["local_file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record["sha256"]
        fn = next(n for n in extracted.body if isinstance(n, ast.FunctionDef) and n.name == record["function"])
        body = ast.dump(ast.Module(body=fn.body, type_ignores=[]), include_attributes=False).encode()
        assert hashlib.sha256(body).hexdigest() == record["body_ast_sha256"]
    data["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data["extracted_sha256"] = hashlib.sha256((VENDOR / "kernels.py").read_bytes()).hexdigest()
    return data


def cases():
    return [
        dict(id="fg_logsoftmax_m64_k16384_f32", family="log_softmax_backward", M=64, K=16384, dtype="float32", boundary=False),
        dict(id="fg_logsoftmax_m256_k8192_f32", family="log_softmax_backward", M=256, K=8192, dtype="float32", boundary=False),
        dict(id="fg_logsoftmax_m512_k4096_bf16", family="log_softmax_backward", M=512, K=4096, dtype="bfloat16", boundary=False),
        dict(id="fg_logsoftmax_masked", family="log_softmax_backward", M=7, K=2053, dtype="float32", boundary=True),
        dict(id="fg_rotary_t128_q32_k8_d128_f32", family="rotary_embedding", tokens=128, qheads=32, kheads=8, dim=128, dtype="float32", interleaved=False, boundary=False),
        dict(id="fg_rotary_t1024_q32_k8_d128_bf16", family="rotary_embedding", tokens=1024, qheads=32, kheads=8, dim=128, dtype="bfloat16", interleaved=False, boundary=False),
        dict(id="fg_rotary_t4096_q16_k4_d64_bf16", family="rotary_embedding", tokens=4096, qheads=16, kheads=4, dim=64, dtype="bfloat16", interleaved=True, boundary=False),
        dict(id="fg_rotary_masked", family="rotary_embedding", tokens=17, qheads=5, kheads=2, dim=96, dtype="float32", interleaved=True, boundary=True),
    ]


def configurations(case):
    if case["family"] == "log_softmax_backward":
        return [dict(block=256, warps=4), dict(block=512, warps=4),
                dict(block=1024, warps=4), dict(block=1024, warps=8)]
    return [dict(warps=w) for w in (1, 2, 4, 8)]


def _output(shape, dtype):
    import math
    import torch
    guard = 128
    size = math.prod(shape)
    storage = torch.full((size + guard * 2,), 37, dtype=dtype, device="cuda")
    return storage[guard:guard + size].view(shape), storage


def prepare(case, seed):
    import torch
    dtype = getattr(torch, case["dtype"])
    g = torch.Generator().manual_seed(seed)
    data = dict(inputs={}, input_copies={}, outputs={}, storages={}, references={})
    if case["family"] == "log_softmax_backward":
        shape = (case["M"], case["K"])
        grad = torch.randn(shape, generator=g, dtype=torch.float32).to(dtype)
        logits = torch.randn(shape, generator=g, dtype=torch.float32)
        logsm = torch.log_softmax(logits, dim=-1).to(dtype)
        data["input_copies"] = dict(grad=grad, logsm=logsm)
        # Reference uses stored input values and FP64 accumulation/exp, without
        # borrowing the candidate's chunk decomposition or reduction order.
        data["references"]["dx"] = grad.double() - logsm.double().exp() * grad.double().sum(-1, keepdim=True)
        out, storage = _output(shape, dtype)
        data["outputs"]["dx"], data["storages"]["dx"] = out, storage
    else:
        t, qh, kh, d = (case[x] for x in ("tokens", "qheads", "kheads", "dim"))
        q = torch.randn((t, qh, d), generator=g).to(dtype)
        k = torch.randn((t, kh, d), generator=g).to(dtype)
        max_pos = max(t + 7, 128)
        angles = torch.randn((max_pos, d // 2), generator=g)
        cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
        # The boundary also checks a non-unit position-id stride.
        positions = torch.randint(max_pos, (t * (2 if case["boundary"] else 1),), generator=g, dtype=torch.int64)
        positions = positions[::2] if case["boundary"] else positions
        data["input_copies"] = dict(q=q, k=k, cos=cos, sin=sin, positions=positions)
        for key, x in (("oq", q), ("ok", k)):
            a = x.double()
            c, s = cos[positions].double(), sin[positions].double()
            if case["interleaved"]:
                paired = a.reshape(t, a.shape[1], d // 2, 2)
                rotated = torch.stack((-paired[..., 1], paired[..., 0]), -1).flatten(-2)
                c, s = c.repeat_interleave(2, -1), s.repeat_interleave(2, -1)
            else:
                rotated = torch.cat((-a[..., d // 2:], a[..., :d // 2]), -1)
                c, s = torch.cat((c, c), -1), torch.cat((s, s), -1)
            data["references"][key] = a * c[:, None, :] + rotated * s[:, None, :]
            out, storage = _output(tuple(x.shape), dtype)
            data["outputs"][key], data["storages"][key] = out, storage
    for name, cpu in data["input_copies"].items():
        if name == "positions" and case["boundary"]:
            backing = torch.full((cpu.numel() * 2,), -1, dtype=cpu.dtype, device="cuda")
            value = backing[::2]
            value.copy_(cpu)
        else:
            value = cpu.cuda()
        data["inputs"][name] = value
    reset(case, data)
    return data


def launch_spec(case, data, config):
    import triton
    module = _module()
    inp, out = data["inputs"], data["outputs"]
    options = dict(num_warps=config["warps"], num_stages=1)
    if case["family"] == "log_softmax_backward":
        args = (inp["grad"], inp["logsm"], out["dx"], case["M"], case["K"], config["block"])
        return dict(fn=module._log_softmax_bwd_kernel, args=args, grid=(case["M"],), groups={0: 0, 1: 0, 2: 1}, options=options)
    q, k, c, s, p = (inp[n] for n in ("q", "k", "cos", "sin", "positions"))
    oq, ok = out["oq"], out["ok"]
    args = (oq, ok, q, k, c, s, p, *q.stride(), *k.stride(), *oq.stride(), *ok.stride(),
            p.stride(0), c.stride(0), s.stride(0), case["tokens"], case["qheads"], case["kheads"],
            case["dim"], max(triton.next_power_of_2(case["dim"]), 16), case["interleaved"], c.shape[0])
    return dict(fn=module.apply_rotary_pos_emb_kernel, args=args, grid=(case["tokens"],),
                groups={0: 1, 1: 2, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}, options=options)


def reset(case, data):
    for out in data["outputs"].values():
        out.fill_(float("nan"))


def output_tensors(data):
    return data["outputs"]


def validate(case, data):
    import torch
    result = {}
    atol, rtol = (0.032, 0.012) if case["dtype"] == "bfloat16" else (3e-6, 2e-5)
    for name, output in data["outputs"].items():
        actual, reference = output.cpu().double(), data["references"][name]
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        storage = data["storages"][name]
        assert bool((storage[:128] == 37).all()) and bool((storage[-128:] == 37).all())
        result[name] = dict(max_absolute_error=float((actual - reference).abs().max()),
                            finite=bool(torch.isfinite(actual).all()), atol=atol, rtol=rtol, guards=True)
    for name, value in data["inputs"].items():
        assert torch.equal(value.cpu(), data["input_copies"][name]), name
    return dict(outputs=result, inputs_unchanged=True, complete_outputs_checked=True,
                oracle="FP64 torch mathematics from stored input values")
