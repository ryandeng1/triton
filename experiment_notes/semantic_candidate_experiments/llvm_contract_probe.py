#!/usr/bin/env python3
"""Bounded, CPU-first LLVM argument-contract experiment on pinned AITER code.

No native compiler edits. The hook roundtrips LLVM-dialect MLIR on both sides,
adds only audited argument attributes, and preserves the normal LLVM optimizer.
It never changes loads, inline-asm side effects, or invariant-load metadata.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import struct
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RETAINED = ROOT / "results/recurrent_licm_final"
OUTPUT = ROOT / "results/llvm_contract_probe_final"
POINTERS = (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12)
WRITES = (9, 10)
READONLY = tuple(i for i in POINTERS if i not in WRITES)
NAME = "fused_sigmoid_gating_delta_rule_update_kernel"
ARG_NAMES = ("A_log", "a", "dt_bias", "softplus_beta", "softplus_threshold", "q", "k", "v", "b", "o", "h0_source", "h0_indices", "cu_seqlens", "scale", "T")


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_sources(output):
    output.mkdir(parents=True, exist_ok=True)
    source = RETAINED / "kernels.py"
    shutil.copyfile(source, output / "kernels.py")
    shutil.copyfile(RETAINED / "AITer_LICENSE", output / "AITer_LICENSE")
    tree = ast.parse(source.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    manual = functions["recurrent_manual_hoist"]
    # Keep the same function name and source line locations. Each module is
    # loaded separately; the only semantic edit is removing the two `other`s.
    replacements = []
    for node in manual.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id not in ("b_A_log", "b_dt_bias"):
                continue
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "load"]
            assert len(calls) == 1
            call = calls[0]
            assert {k.arg for k in call.keywords} == {"mask", "other"}
            replacements.append(call.lineno)
    lines = source.read_text().splitlines(keepends=True)
    for lineno in replacements:
        assert lines[lineno - 1].count(", other=0.0") == 1
        lines[lineno - 1] = lines[lineno - 1].replace(", other=0.0", "")
    assert len(replacements) == 2
    undefined = "".join(lines)
    (output / "kernels_undefined.py").write_text(undefined)
    restored = undefined.splitlines(keepends=True)
    for lineno in replacements:
        restored[lineno - 1] = source.read_text().splitlines(keepends=True)[lineno - 1]
    assert "".join(restored) == source.read_text()

    # Explicit audit of this pinned source, not an inferred global contract.
    for fn in functions.values():
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        assert all(isinstance(n.func, (ast.Name, ast.Attribute)) for n in calls)
        assert all(n.func.id == "range" for n in calls if isinstance(n.func, ast.Name))
        stores = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "store"]
        assert len(stores) == 2
        assert [ast.unparse(n.args[0]) for n in stores] == ["p_o", "p_h0"]
        assert [ast.unparse(n.args[1]) for n in stores] == [
            "b_o.to(p_o.dtype.element_ty)", "b_h.to(p_h0.dtype.element_ty)"]
        assert not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn))
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute):
                assert n.attr not in ("inline_asm_elementwise", "atomic_add", "store_ptr", "bitcast")
    provenance = json.loads((RETAINED / "provenance.json").read_text())
    return dict(
        source_sha256=digest(source.read_bytes()),
        copied_source_equal=(output / "kernels.py").read_bytes() == source.read_bytes(),
        upstream_revision=provenance["revision"],
        upstream_baseline_body_ast_equal=provenance["baseline_body_ast_equal"],
        manual_undefined_reverse_edit_equal=True,
        manual_undefined_changed_lines=replacements,
        pointer_argument_positions=POINTERS, readonly_argument_positions=READONLY,
        noalias_argument_positions=WRITES, captures_none_argument_positions=POINTERS,
        proof={
            "readonly": "Pinned source only stores floating-point results through o and h0_source; all other pointer roots only provide loads.",
            "captures_none": "Audited pinned source has no pointer/address stores, pointer returns, address-to-integer conversions, or calls to external helpers. Pointer-derived values are used only to address loads/stores.",
            "noalias": "Only the two modified allocation roots o and h0_source receive noalias. The launch audit requires these two complete backing storages to be mutually disjoint and disjoint from every input; no noalias is emitted on input alias-group members.",
            "invocation_scope": "Input initialization and stability last for one invocation. No invariant.load metadata or invariant intrinsic is added.",
            "compile_only": "CPU compilation establishes conditional code generation under these preconditions; launch-time storage checks are required before executing treatment code.",
        })


def signature_and_constants(case, bv):
    dtype = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
    names = ["A_log", "a", "dt_bias", "softplus_beta", "softplus_threshold", "q", "k", "v", "b", "o", "h0_source", "h0_indices", "cu_seqlens", "scale", "T"]
    types = ["*" + dtype[case["parameter_dtype"]]] * 3 + ["fp32"] * 2
    types += ["*" + dtype[case["qk_dtype"]]] * 2
    types += ["*" + dtype[case["dtype"]]] * 2
    types += ["*" + dtype[case["qk_dtype"]], "*fp32", "*i64", "*i64", "fp32", "i32"]
    constants = {k: case[k] for k in ("B", "H", "HV", "K", "V")}
    constants.update(BK=1 << (case["K"] - 1).bit_length(), BV=bv,
                     USE_INITIAL_STATE=True, USE_QK_L2NORM_IN_KERNEL=True,
                     IS_VARLEN=False)
    return dict(zip(names, types)), constants


def modify_mlir(text, enabled):
    lines = text.splitlines(keepends=True)
    indexes = [i for i, s in enumerate(lines) if "llvm.func @" + NAME + "(" in s
               or "llvm.func @recurrent_manual_hoist(" in s]
    assert len(indexes) == 1, "Expected exactly one audited kernel definition"
    line = lines[indexes[0]]
    added = {}
    for pos in POINTERS:
        pattern = rf"(%{ARG_NAMES[pos]}: !llvm\.ptr<1>)(?: \{{([^{{}}]*)\}})?"
        matches = list(re.finditer(pattern, line))
        assert len(matches) == 1, (pos, line[:400])
        attrs = ["llvm.nocapture"]
        if pos in READONLY:
            attrs.append("llvm.readonly")
        if pos in WRITES:
            attrs.append("llvm.noalias")
        added[str(pos)] = attrs if enabled else []
        if enabled:
            def replace(m):
                old = m.group(2)
                return m.group(1) + " {" + (old + ", " if old else "") + ", ".join(attrs) + "}"
            line = re.sub(pattern, replace, line)
    lines[indexes[0]] = line
    changed = "".join(lines)
    assert text.splitlines()[indexes[0] + 1:] == changed.splitlines()[indexes[0] + 1:]
    return changed, added


class LLVMHook:
    def __init__(self, directory, enabled):
        self.directory, self.enabled = directory, enabled
        self.count = 0

    def __call__(self, *args):
        if not args:
            return ("llvm-contract-probe-v1-" + str(self.directory), digest(str(self.directory)))
        backend, stages, options, language, capability = args
        original_stage = stages["llir"]
        def wrapped(src, metadata):
            from triton._C.libtriton import ir, llvm
            original_to_module, original_optimize = llvm.to_module, llvm.optimize_module
            def to_module(mod, context):
                self.count += 1
                plain = str(mod)
                annotated, self.added = modify_mlir(plain, self.enabled)
                (self.directory / "lowered.before.mlir").write_text(plain)
                path = self.directory / "lowered.input.mlir"
                path.write_text(annotated)
                parsed = ir.parse_mlir_module(str(path), src.context)
                assert parsed.verify()
                return original_to_module(parsed, context)
            def optimize(mod, *args, **kwargs):
                (self.directory / "before.llir").write_text(str(mod))
                result = original_optimize(mod, *args, **kwargs)
                (self.directory / "after.llir").write_text(str(mod))
                return result
            llvm.to_module, llvm.optimize_module = to_module, optimize
            try:
                return original_stage(src, metadata)
            finally:
                llvm.to_module, llvm.optimize_module = original_to_module, original_optimize
        stages["llir"] = wrapped


def visibility(text):
    loads = re.findall(r"^\s*%\S+ = load [^\n]+", text, re.M)
    return dict(llvm_loads=len(loads),
                llvm_global_loads=sum("ptr addrspace(1)" in line for line in loads),
                llvm_shared_loads=sum("ptr addrspace(3)" in line for line in loads),
                inline_asm_calls=len(re.findall(r"\bcall\b[^\n]*\basm\b", text)),
                sideeffect_global_load_calls=len(re.findall(r"\bcall\b[^\n]*\basm sideeffect\b[^\n]*ld\.global", text)),
                masked_load_intrinsics=len(re.findall(r"\bcall\b[^\n]*@llvm\.masked\.load", text)),
                invariant_load_metadata="!invariant.load" in text)


def machine_sections(cubin):
    """Hash ELF text bytes, including encoded scheduling/control bits."""
    assert cubin[:6] == b"\x7fELF\x02\x01", "Expected ELF64 little-endian cubin"
    shoff = struct.unpack_from("<Q", cubin, 40)[0]
    shsize, shnum, strindex = struct.unpack_from("<HHH", cubin, 58)
    assert shsize == 64
    headers = [struct.unpack_from("<IIQQQQIIQQ", cubin, shoff + shsize * i) for i in range(shnum)]
    strings = cubin[headers[strindex][4]:headers[strindex][4] + headers[strindex][5]]
    result = {}
    for header in headers:
        start = header[0]
        name = strings[start:strings.index(b"\0", start)].decode()
        if name.startswith(".text."):
            data = cubin[header[4]:header[4] + header[5]]
            result[name] = dict(bytes=len(data), sha256=digest(data))
    assert len(result) == 1
    return result


def normalized_ptx(text):
    text = re.split(r"^\s*\.section\s+\.debug", text, maxsplit=1, flags=re.M)[0]
    lines = [re.sub(r"//.*", "", line).strip() for line in text.splitlines()
             if not re.match(r"\s*\.(?:file|loc)\b", line)]
    return "\n".join(line for line in lines if line)


def host_allocation_audit(case):
    """Exercise complete-backing-storage checks on CPU, without CUDA setup.

    These are fresh CPU allocations with the retained harness's tensor shapes.
    They demonstrate the precondition checker, not runtime validation of a GPU
    invocation. The retained GPU harness uses fresh corresponding allocations.
    """
    import torch
    B, T, H, HV, K, V = (case[k] for k in ("B", "T", "H", "HV", "K", "V"))
    pdt, qdt, vdt = (getattr(torch, case[k]) for k in ("parameter_dtype", "qk_dtype", "dtype"))
    specifications = {0: ((HV,), pdt), 1: ((B, T, HV), pdt), 2: ((HV,), pdt),
                      5: ((B, T, H, K), qdt), 6: ((B, T, H, K), qdt),
                      7: ((B, T, HV, V), vdt), 8: ((B, T, HV), vdt),
                      9: ((B * T * HV * V + 128,), qdt),
                      10: ((B + 2, HV, K, V), torch.float32),
                      11: ((B,), torch.int64), 12: ((B + 1,), torch.int64)}
    tensors = {i: torch.empty(shape, dtype=dt) for i, (shape, dt) in specifications.items()}
    spans = {i: (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()) for i, t in tensors.items()}
    checked = []
    for wi in WRITES:
        wlo, wn = spans[wi]
        for other, (rlo, rn) in spans.items():
            if wi == other:
                continue
            assert wlo + wn <= rlo or rlo + rn <= wlo, (wi, other)
            checked.append([wi, other])
    assert not torch.cuda.is_initialized()
    return dict(device="cpu", gpu_initialized=False, complete_storage_pairs_checked=checked,
                storage_bytes={ARG_NAMES[i]: n for i, (_, n) in spans.items()},
                output_view_guards_included_in_footprint=True,
                caveat="Allocation/checker demonstration only; no GPU execution or input-value validation performed.")


def compile_one(case, config, source_variant, attrs_enabled, modules, output):
    import triton
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.tools.disasm import get_sass
    bv, warps, stages = config
    label = f"{case['id']}_bv{bv}_{source_variant}_{'attrs' if attrs_enabled else 'control'}"
    directory = output / label
    directory.mkdir(exist_ok=True)
    module = modules["undefined" if source_variant == "manual_undefined" else "base"]
    fn = getattr(module, "recurrent_manual_hoist" if source_variant.startswith("manual") else NAME)
    signature, constants = signature_and_constants(case, bv)
    attrs = {(i,): [("tt.divisibility", 16)] for i in POINTERS}
    if source_variant == "immutable":
        for i in (0, 2):
            attrs[(i,)].append(("tt.experimental_immutable", 1))
    source = ASTSource(fn, signature, constexprs=constants, attrs=attrs)
    hook = LLVMHook(directory, attrs_enabled)
    old_hook, old_always = knobs.runtime.add_stages_inspection_hook, knobs.compilation.always_compile
    knobs.runtime.add_stages_inspection_hook, knobs.compilation.always_compile = hook, True
    options = dict(num_warps=warps, num_stages=stages,
                   enable_indexed_update_specialization=False,
                   enable_private_atomic_relaxation=False)
    try:
        kernel = triton.compile(source, target=GPUTarget("cuda", 103, 32), options=options)
    finally:
        knobs.runtime.add_stages_inspection_hook, knobs.compilation.always_compile = old_hook, old_always
    assert hook.count == 1
    hashes = {}
    for extension in ("ttir", "ttgir", "llir", "ptx", "cubin"):
        value = kernel.asm[extension]
        data = value if isinstance(value, bytes) else value.encode()
        (directory / ("kernel." + extension)).write_bytes(data)
        hashes[extension] = digest(data)
    sass = get_sass(kernel.asm["cubin"])
    (directory / "kernel.sass").write_text(sass)
    hashes["sass"] = digest(sass)
    text_sections = machine_sections(kernel.asm["cubin"])
    hashes["machine_text"] = next(iter(text_sections.values()))["sha256"]
    hashes["normalized_ptx"] = digest(normalized_ptx(kernel.asm["ptx"]))
    hashes["llvm_body"] = digest(re.sub(r"^define ptx_kernel[^\n]+\n", "", kernel.asm["llir"], flags=re.M))
    for name in ("lowered.before.mlir", "lowered.input.mlir", "before.llir", "after.llir"):
        hashes[name] = digest((directory / name).read_bytes())
    report = dict(label=label, case=case, config=config, source_variant=source_variant,
                  llvm_attrs_enabled=attrs_enabled, emitted_mlir_arg_attrs=hook.added,
                  hashes=hashes, before_visibility=visibility((directory / "before.llir").read_text()),
                  after_visibility=visibility(kernel.asm["llir"]),
                  metadata=kernel.metadata._asdict(), cache_hash=kernel.hash,
                  machine_text_sections=text_sections, gpu_initialized=False,
                  artifacts="artifacts.tar.xz")
    # Preserve all full IR/binary evidence in a compact per-build archive.
    generated = [directory / name for name in ("lowered.before.mlir", "lowered.input.mlir", "before.llir", "after.llir")]
    generated += [directory / ("kernel." + ext) for ext in ("ttir", "ttgir", "llir", "ptx", "cubin", "sass")]
    with tarfile.open(directory / "artifacts.tar.xz", "w:xz") as archive:
        for path in generated:
            archive.add(path, arcname=path.name)
    for path in generated:
        path.unlink()
    # Remove only the superseded archive produced by this experiment's first
    # gzip implementation; never remove user-supplied inputs.
    (directory / "artifacts.tar.gz").unlink(missing_ok=True)
    save_json(directory / "report.json", report)
    print(label, report["after_visibility"], "SASS", hashes["sass"][:16], flush=True)
    return report


def cpu_probe(output):
    import triton
    from triton._C import libtriton
    audit = prepare_sources(output)
    modules = {"base": load_module(output / "kernels.py", "llvm_probe_base"),
               "undefined": load_module(output / "kernels_undefined.py", "llvm_probe_undefined")}
    reports, allocation_audits = [], {}
    plans = [("upstream500", (16, 1, 2)), ("upstream500", (32, 2, 2)),
             ("upstream2048", (8, 1, 3))]
    for case_id, config in plans:
        case = json.loads((RETAINED / (case_id + ".json")).read_text())["case"]
        if case_id not in allocation_audits:
            allocation_audits[case_id] = host_allocation_audit(case)
        for variant in ("original", "immutable", "manual"):
            for enabled in (False, True):
                reports.append(compile_one(case, config, variant, enabled, modules, output))
        if case_id == "upstream500":
            reports.append(compile_one(case, config, "manual_undefined", False, modules, output))
    pairs = []
    for control in [r for r in reports if not r["llvm_attrs_enabled"] and r["source_variant"] != "manual_undefined"]:
        treatment = next(r for r in reports if r["label"] == control["label"].replace("_control", "_attrs"))
        equality = {k: control["hashes"][k] == treatment["hashes"][k] for k in control["hashes"]}
        assert equality["ttir"] and equality["ttgir"] and equality["lowered.before.mlir"]
        assert equality["llvm_body"]
        pairs.append(dict(control=control["label"], treatment=treatment["label"], equal=equality,
                          timing_needed=not equality["machine_text"]))
    mask_pairs = []
    for undefined in [r for r in reports if r["source_variant"] == "manual_undefined"]:
        zero = next(r for r in reports if r["label"] == undefined["label"].replace("manual_undefined", "manual"))
        mask_pairs.append(dict(zero=zero["label"], undefined=undefined["label"],
                               equal={k: zero["hashes"][k] == undefined["hashes"][k] for k in zero["hashes"]}))
    native_manual_pairs = []
    for native in [r for r in reports if r["source_variant"] == "immutable" and not r["llvm_attrs_enabled"]]:
        manual = next(r for r in reports if r["label"] == native["label"].replace("immutable", "manual"))
        native_manual_pairs.append(dict(native=native["label"], manual=manual["label"],
                                        machine_text_equal=native["hashes"]["machine_text"] == manual["hashes"]["machine_text"],
                                        performance_measured=False))
    environment = dict(triton=triton.__version__, native_library_sha256=digest(Path(libtriton.__file__).read_bytes()),
                       script_sha256=digest(Path(__file__).read_bytes()), target="cuda:103", gpu_initialized=False)
    summary = dict(environment=environment, audit=audit, cpu_allocation_audits=allocation_audits,
                   attribute_pairs=pairs, mask_pairs=mask_pairs, native_manual_pairs=native_manual_pairs,
                   reports=reports, attribute_changed_sass_pairs=sum(not p["equal"]["sass"] for p in pairs))
    save_json(output / "summary.json", summary)
    print("Attribute pairs with changed SASS:", summary["attribute_changed_sass_pairs"], "/", len(pairs), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    cpu_probe(args.output)


if __name__ == "__main__":
    main()
