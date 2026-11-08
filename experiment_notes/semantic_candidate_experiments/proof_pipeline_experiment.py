#!/usr/bin/env python3
"""Exercise producer proof -> lowering and alias facts -> native LICM.

Run from the Triton checkout with the source build installed:
  python experiment_notes/semantic_candidate_experiments/proof_pipeline_experiment.py --output NEW_DIRECTORY

RMS timings are first-use profitability calibration on common physical output
buffers. They are not whole-model speedups. Embedding demonstrates kernel-count
and machine-code changes; no separate-allocation timing claims are made.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
import triton
from triton._C import libtriton
from triton.experimental.semantic_contracts import compile
from triton.experimental.semantic_contracts.ops import embedding_backward, rms_norm


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def save_compiled(graph, directory):
    metadata = {}
    for node, variants in graph.kernels.items():
        metadata[node] = {}
        for name, kernel in variants.items():
            target = directory / node / name
            target.mkdir(parents=True, exist_ok=True)
            for kind in ("ttir", "ttgir", "llir", "ptx", "cubin"):
                content = kernel.asm.get(kind)
                if content is not None:
                    (target / ("kernel." + kind)).write_bytes(content.encode() if isinstance(content, str) else content)
            ptx = kernel.asm.get("ptx", "")
            metadata[node][name] = {
                "atomic_instruction_sites": len(re.findall(r"\b(?:atom|red)\.", ptx)),
                "registers": getattr(kernel, "n_regs", None),
                "spills": getattr(kernel, "n_spills", None),
                "shared": kernel.metadata.shared,
            }
    save_json(directory / "code_evidence.json", metadata)
    return metadata


def rms_case(shape, grid_multiplier, root):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((shape[1], ), device="cuda", dtype=x.dtype)

    def graph(value, weights):
        return rms_norm(value, weights, num_warps=4, grid_multiplier=grid_multiplier)

    baseline = compile(graph, (x, weight), enabled=False, autotune=False)
    inferred = compile(graph, (x, weight), enabled=True, autotune=True)
    expected = baseline(x, weight)
    actual = inferred(x, weight)
    for lhs, rhs in zip(expected, actual):
        assert torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))
    inv = torch.rsqrt(x.double().square().mean(1) + 1e-6)
    torch.testing.assert_close(actual[0].float(), (x.double() * inv[:, None] * weight.double()).float(), atol=.025,
                               rtol=.025)
    torch.testing.assert_close(actual[1], inv.float(), atol=3e-6, rtol=3e-6)
    path = root / ("rms_" + "x".join(map(str, shape)))
    path.mkdir()
    save_json(path / "report.json", inferred.report())
    save_compiled(inferred, path)
    result = {
        "kind": "rms", "shape": shape, "grid_multiplier": grid_multiplier, "validation":
        "bitwise baseline equality and independent FP64 reference", "plan": inferred.report()["nodes"][0]
    }
    print(json.dumps({"case": path.name, "selected": result["plan"]["selected_optimization"]}), flush=True)
    return result


def embedding_case(scale, source, root):
    n, d, vocabulary = 4096, 128, 8192
    grad = torch.randn((n, d), dtype=torch.bfloat16, device="cuda")
    supplied = torch.arange(n, device="cuda").remainder(97)

    def graph(gradient, indices_input):
        if source == "producer":
            indices = torch.randperm(vocabulary, device=gradient.device)[:n].clone().to(torch.int32)
        elif source == "mutated":
            indices = torch.arange(n, device=gradient.device)
            indices[::2].fill_(0)
        else:
            indices = indices_input
        return embedding_backward(gradient, indices, vocabulary, padding_idx=-1, scale_by_freq=scale), indices

    baseline = compile(graph, (grad, supplied), enabled=False, autotune=False)
    optimized = compile(graph, (grad, supplied), enabled=True, autotune=False)
    torch.manual_seed(1234)
    reference, ref_indices = baseline(grad, supplied)
    torch.manual_seed(1234)
    actual, indices = optimized(grad, supplied)
    assert torch.equal(indices, ref_indices)
    if source == "producer":
        assert torch.equal(reference.view(torch.uint8), actual.view(torch.uint8))
        expected = torch.zeros((vocabulary, d), device="cuda", dtype=grad.dtype)
        expected[indices.long()] = grad
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual, reference, atol=.125, rtol=.02)
    # Reuse the executable with changed contents. Proofs concern producers, not
    # a fortunate sample of concrete values inspected during compilation.
    changed_grad = grad * .5
    torch.manual_seed(5678)
    again, again_indices = optimized(changed_grad, supplied)
    if source == "producer":
        expected.zero_()
        expected[again_indices.long()] = changed_grad
        torch.testing.assert_close(again, expected, atol=0, rtol=0)
    path = root / f"embedding_{source}_freq{int(scale)}"
    path.mkdir()
    save_json(path / "report.json", optimized.report())
    code = save_compiled(optimized, path)
    baseline_path = path / "baseline"
    baseline_path.mkdir()
    save_json(baseline_path / "report.json", baseline.report())
    save_compiled(baseline, baseline_path)
    plan = optimized.report()["nodes"][0]
    update = next(iter(code.values()))["update"]
    if source == "producer":
        assert plan["selected_optimization"] == "unique_scatter"
        assert "count" not in plan["kernel_launches"]
        assert update["atomic_instruction_sites"] == 0
    else:
        assert plan["selected_optimization"] == "atomic_scatter"
        assert update["atomic_instruction_sites"] > 0
    result = {
        "kind": "embedding", "source": source, "scale_by_freq": scale, "validation":
        "baseline/reference checks and repeat execution with changed inputs", "plan": plan, "baseline_plan":
        baseline.report()["nodes"][0]
    }
    print(
        json.dumps({"case": path.name, "selected": plan["selected_optimization"], "launches": plan["kernel_launches"]}),
        flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    package = Path(triton.__file__).parent / "experimental" / "semantic_contracts"
    sources = args.output / "sources"
    sources.mkdir()
    hashes = {}
    for path in [Path(__file__), *sorted(package.glob("*.py"))]:
        content = path.read_bytes()
        (sources / path.name).write_bytes(content)
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
    report = {
        "torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(), "argv": sys.argv,
        "compiler_sha256": hashlib.sha256(Path(libtriton.__file__).read_bytes()).hexdigest(), "source_hashes": hashes,
        "cases": []
    }
    report["git_diff"] = subprocess.check_output(["git", "diff", "--stat"], text=True)
    try:
        for shape, grid in [((31, 137), 1), ((4096, 1024), 16), ((16384, 4096), 32)]:
            report["cases"].append(rms_case(shape, grid, args.output))
            save_json(args.output / "results.json", report)
        for scale, source in [(False, "producer"), (True, "producer"), (True, "placeholder"), (True, "mutated")]:
            report["cases"].append(embedding_case(scale, source, args.output))
            save_json(args.output / "results.json", report)
        report["status"] = "passed"
    finally:
        save_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
