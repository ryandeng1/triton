#!/usr/bin/env python3
"""Pinned, source-only Triton/Gluon semantic opportunity inventory.

Run with Python 3.10+: python semantic_prevalence_survey.py --fetch
Sources are cached outside the repository. No packages are imported/executed and
no GPU/compiler is used. Counts are source patterns, never speedup estimates.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

SPECS = [
    dict(name="FlagGems", repository="flagos-ai/FlagGems", revision="3f1f896fd90f5db42d167d64fcb9691326071bc2", source_roots=["src/flag_gems/"], pin_reason="existing irregular/segmented experiment pin"),
    dict(name="Liger", repository="linkedin/Liger-Kernel", revision="e44de705ecc9b43eb33aa35ad10ee9d59a054c3b", source_roots=["src/liger_kernel/"], pin_reason="GitHub default branch HEAD queried 2026-11-08"),
    dict(name="AITER", repository="ROCm/aiter", revision="ef0a5a26b8cb38ea1308f827985e36cde964a6ab", source_roots=["aiter/ops/triton/"], pin_reason="existing RMSNorm experiment pin"),
    dict(name="TransformerEngine", repository="NVIDIA/TransformerEngine", revision="c1c2d4d2fa008386f247c2fec56a998e27134e0a", source_roots=["transformer_engine/"], pin_reason="existing permutation experiment pin"),
]
LANGUAGES = ("triton.language", "triton.experimental.gluon.language")
LOOPS = (ast.For, ast.AsyncFor, ast.While)
PURE_CALLS = {"arange", "full", "zeros", "broadcast_to", "expand_dims", "reshape", "ravel", "trans", "cast", "cast_pointer", "join", "split", "splat", "where", "minimum", "maximum", "clamp", "cdiv", "constexpr", "program_id", "num_programs", "multiple_of", "max_contiguous", "max_constancy", "assume", "abs", "abs2", "ceil", "floor", "sum", "min", "max", "to", "bitcast", "broadcast", "convert_layout", "slice", "int", "float", "bool", "len"}

def dump(value):
    return ast.dump(value, include_attributes=False)

def text(node, limit=220):
    if node is None:
        return None
    result = ast.unparse(node)
    return result if len(result) <= limit else result[:limit-3] + "..."

def qname(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        prefix = qname(node.value, aliases)
        return prefix + "." + node.attr if prefix else None
    return None

def module_aliases(tree):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or "") + "." + item.name
    # Simple aliases such as atomic = tl.atomic_add. Do not guess arbitrary wrappers.
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                value = qname(node.value, aliases)
                if value and value.startswith("triton."):
                    aliases[node.targets[0].id] = value
    return aliases

def is_language(name):
    return bool(name and any(name.startswith(prefix + ".") for prefix in LANGUAGES))

def op_kind(call, aliases, is_jit):
    name = qname(call.func, aliases)
    if is_language(name):
        return name.rsplit(".", 1)[-1], "resolved_module", name
    if isinstance(call.func, ast.Attribute) and call.func.attr == "atomic_add":
        if name and name.startswith(("cuda.", "cutlass.", "torch.", "numpy.")):
            return "atomic_add", "non_triton", name
        if is_jit:
            return "atomic_add", "inferred_tensor_member", name
        return "atomic_add", "unresolved", name
    return None, None, name

def local_nodes(function):
    stack = list(reversed(function.body))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(reversed(list(ast.iter_child_nodes(node))))

def body_hash(function, aliases):
    normalized = copy.deepcopy(function)
    normalized.name = "_function_"
    normalized.decorator_list = []
    if normalized.body and isinstance(normalized.body[0], ast.Expr) and isinstance(normalized.body[0].value, ast.Constant) and isinstance(normalized.body[0].value.value, str):
        normalized.body = normalized.body[1:]
    class Imports(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in aliases and aliases[node.id].startswith("triton."):
                return ast.copy_location(ast.Name(id=aliases[node.id], ctx=node.ctx), node)
            return node
    return hashlib.sha256(dump(Imports().visit(normalized)).encode()).hexdigest()[:16]

def names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} if node is not None else set()

def targets(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}

def assignments(nodes):
    definitions = collections.defaultdict(list)
    for n in nodes:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                for name in targets(target):
                    definitions[name].append(n.value)
        elif isinstance(n, ast.AnnAssign):
            for name in targets(n.target):
                definitions[name].append(n.value)
        elif isinstance(n, (ast.AugAssign, ast.NamedExpr)):
            for name in targets(n.target):
                definitions[name].append(None)  # recurrence or ordering-dependent binding
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            for name in targets(n.target):
                definitions[name].append(None)
    return definitions

def parameter(call, name, positional, member=False):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value, "keyword"
    index = positional - int(member)
    if len(call.args) > index:
        return call.args[index], "positional"
    if any(keyword.arg is None for keyword in call.keywords):
        return None, "forwarded_kwargs"
    return None, "omitted"

def option_value(node, origin, defaults):
    if origin == "omitted":
        return "omitted", None
    if origin == "forwarded_kwargs":
        return "forwarded_kwargs", None
    if isinstance(node, ast.Constant):
        return ("explicit_none", None) if node.value is None else ("explicit", node.value)
    if isinstance(node, ast.Name) and node.id in defaults:
        return "forwarded_parameter", text(defaults[node.id])
    return "forwarded_or_expression", text(node)

def signature_defaults(function):
    args = function.args.posonlyargs + function.args.args
    result = {a.arg: None for a in args + function.args.kwonlyargs}
    for a, default in zip(args[-len(function.args.defaults):], function.args.defaults):
        result[a.arg] = default
    for a, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
        result[a.arg] = default
    return result

def inherited_loops(node, parents, stop):
    result = []
    node = parents.get(node)
    while node is not None and node is not stop:
        if isinstance(node, LOOPS):
            result.append(node)
        node = parents.get(node)
    return result

def expression_dependencies(node, definitions, parameters, active=None):
    active = set() if active is None else active
    result = set()
    for name in names(node):
        if name in parameters:
            result.add(name)
        elif name not in active:
            for value in definitions.get(name, []):
                if value is not None:
                    result |= expression_dependencies(value, definitions, parameters, active | {name})
    return result

def loop_invariance(loop, aliases):
    definitions = assignments(ast.walk(loop))
    invariant = set()
    def pure(expr):
        for n in ast.walk(expr):
            if isinstance(n, ast.Call):
                name = qname(n.func, aliases)
                suffix = name.rsplit(".", 1)[-1] if name else ""
                if suffix not in PURE_CALLS:
                    return False
            elif isinstance(n, (ast.NamedExpr, ast.Await, ast.Yield)):
                return False
        return True
    changed = True
    while changed:
        changed = False
        for name, values in definitions.items():
            # A binding must have exactly one syntactic assignment in this loop.
            # This intentionally misses equivalent branch definitions.
            if name not in invariant and len(values) == 1 and values[0] is not None:
                value = values[0]
                if pure(value) and all(n not in definitions or n in invariant for n in names(value)):
                    invariant.add(name)
                    changed = True
    def check(expr):
        return expr is None or (pure(expr) and all(n not in definitions or n in invariant for n in names(expr)))
    return check

def region_for(path):
    if "/runtime/backend/" in path:
        return "backend:" + path.split("/runtime/backend/", 1)[1].split("/", 1)[0]
    if "/backends/" in path:
        return "backend:" + path.split("/backends/", 1)[1].split("/", 1)[0]
    if "/_gluon_kernels/" in path:
        return "gluon"
    if "/cutile/" in path:
        return "cutile"
    return "generic"

def scan(spec, cache):
    root = cache / (spec["name"] + "-" + spec["revision"])
    stats = collections.Counter()
    functions = []
    atomics = []
    licm = []
    errors = []
    excluded = []
    jit_bodies = set()
    file_hashes = set()
    wrapper_calls = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        if not any(rel.startswith(prefix) for prefix in spec["source_roots"]):
            continue
        source = path.read_text()
        stats["python_files"] += 1
        file_hashes.add(hashlib.sha256(source.encode()).hexdigest())
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            errors.append(dict(path=rel, line=error.lineno, error=error.msg))
            continue
        aliases = module_aliases(tree)
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        module_functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        # This deliberately resolves only direct same-module helper calls.
        # Primitive sites remain counted once, regardless of helper callers.
        atomic_helpers = {}
        for function in module_functions:
            primitive_sites = [n.lineno for n in local_nodes(function) if isinstance(n, ast.Call) and op_kind(n, aliases, False)[0:2] == ("atomic_add", "resolved_module")]
            if primitive_sites:
                atomic_helpers[function.name] = primitive_sites
        for function in module_functions:
            decorators = [qname(d.func if isinstance(d, ast.Call) else d, aliases) for d in function.decorator_list]
            is_jit = any(d in ("triton.jit", "triton.experimental.gluon.jit") for d in decorators)
            nodes = list(local_nodes(function))
            calls = [n for n in nodes if isinstance(n, ast.Call)]
            ops = [(n, *op_kind(n, aliases, is_jit)) for n in calls]
            if is_jit:
                for call in calls:
                    callee = qname(call.func, aliases)
                    if callee in atomic_helpers:
                        wrapper_calls.append(dict(project=spec["name"], path=rel, caller=function.name, line=call.lineno, callee=callee, primitive_lines=atomic_helpers[callee], loops=[loop.lineno for loop in inherited_loops(call, parents, function)], interpretation="Same-module JIT call to function containing atomic_add; no transitive expansion or specialization."))
            ats = [(n, origin, name) for n, op, origin, name in ops if op == "atomic_add" and origin in ("resolved_module", "inferred_tensor_member")]
            for n, op, origin, name in ops:
                if op == "atomic_add" and origin in ("unresolved", "non_triton"):
                    excluded.append(dict(path=rel, function=function.name, line=n.lineno, origin=origin, callee=name))
            if not is_jit and not ats:
                continue
            h = body_hash(function, aliases)
            key = dict(project=spec["name"], path=rel, function=function.name, body=h, region=region_for(rel))
            if is_jit:
                stats["jit_definitions_including_helpers"] += 1
                jit_bodies.add(h)
            functions.append(dict(**key, line=function.lineno, jit=is_jit, atomic_add_sites=len(ats)))
            parameters = signature_defaults(function)
            definitions = assignments(nodes)
            for ordinal, (call, origin, name) in enumerate(ats):
                member = origin == "inferred_tensor_member"
                sem_node, sem_origin = parameter(call, "sem", 3, member)
                sem, value = option_value(sem_node, sem_origin, parameters)
                scope_node, scope_origin = parameter(call, "scope", 4, member)
                scope, scope_value = option_value(scope_node, scope_origin, parameters)
                ptr = call.func.value if member else (call.args[0] if call.args else None)
                val = call.args[0 if member else 1] if len(call.args) > (0 if member else 1) else None
                loops = inherited_loops(call, parents, function)
                parent = parents.get(call)
                returned = "unused_expression" if isinstance(parent, ast.Expr) else "consumed_syntactically"
                if isinstance(parent, (ast.Assign, ast.AnnAssign)):
                    bound = set().union(*(targets(t) for t in parent.targets)) if isinstance(parent, ast.Assign) else targets(parent.target)
                    if bound and not (bound & {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}):
                        returned = "assigned_but_no_source_reads"
                value_casts = sorted({text(n.args[0]) for n in ast.walk(val) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("to", "cast") and n.args}) if val else []
                atomics.append(dict(**key, line=call.lineno, ordinal=ordinal, jit=is_jit, origin=origin, sem=sem, sem_value=value, scope=scope, scope_value=scope_value, return_use=returned, loops=[n.lineno for n in loops], loop_iterators=[text(n.iter, 70) if isinstance(n, ast.For) else "while" for n in loops], function_atomic_sites=len(ats), pointer=text(ptr, 120), argument_dependencies=sorted(expression_dependencies(ptr, definitions, parameters)), value=text(val, 100), value_casts=value_casts, destination_dtype="unknown_without_specialization_or_host_review"))
            if not is_jit:
                continue
            loads = [n for n, op, origin, _ in ops if op == "load" and origin == "resolved_module"]
            stores = [n for n, op, origin, _ in ops if op == "store" and origin == "resolved_module"]
            loops = [n for n in nodes if isinstance(n, LOOPS)]
            stats["loops_in_jit_definitions"] += len(loops)
            for loop_ordinal, loop in enumerate(loops):
                loop_stores = [s for s in stores if loop in inherited_loops(s, parents, function)]
                if not loop_stores:
                    continue
                stats["loops_containing_direct_store"] += 1
                check = loop_invariance(loop, aliases)
                for load_ordinal, call in enumerate(loads):
                    if loop not in inherited_loops(call, parents, function):
                        continue
                    stats["load_loop_pairs_with_direct_store"] += 1
                    ptr, _ = parameter(call, "pointer", 0)
                    mask, _ = parameter(call, "mask", 1)
                    other, _ = parameter(call, "other", 2)
                    if not check(ptr):
                        continue
                    stats["address_invariant_load_loop_pairs"] += 1
                    if not check(mask) or not check(other):
                        continue
                    stats["address_mask_other_invariant_load_loop_pairs"] += 1
                    deps = expression_dependencies(ptr, definitions, parameters)
                    if not deps:
                        continue
                    volatile = next((k.value for k in call.keywords if k.arg == "volatile"), None)
                    if volatile is not None and not (isinstance(volatile, ast.Constant) and volatile.value is False):
                        stats["volatile_or_unknown_volatile_rejections"] += 1
                        continue
                    if any(k.arg is None for k in call.keywords):
                        stats["forwarded_kwargs_load_rejections"] += 1
                        continue
                    conditional = []
                    ancestor = parents.get(call)
                    while ancestor is not None and ancestor is not loop:
                        if isinstance(ancestor, ast.If):
                            conditional.append(ancestor.lineno)
                        ancestor = parents.get(ancestor)
                    licm.append(dict(**key, line=call.lineno, loop=loop.lineno, loop_iterator=text(loop.iter, 70) if isinstance(loop, ast.For) else "while", loop_ordinal=loop_ordinal, load_ordinal=load_ordinal, address=text(ptr, 130), mask=text(mask, 70), other=text(other, 50), argument_dependencies=sorted(deps), store_lines=[s.lineno for s in loop_stores][:12], conditional_ancestors=conditional))
    stats["exact_unique_python_files"] = len(file_hashes)
    stats["exact_unique_jit_bodies"] = len(jit_bodies)
    stats["parse_failures"] = len(errors)
    stats["atomic_add_sites"] = len(atomics)
    stats["exact_unique_body_atomic_sites"] = len({(a["body"], a["ordinal"]) for a in atomics})
    stats["functions_with_atomic_add"] = len({(a["path"], a["function"]) for a in atomics})
    stats["exact_unique_bodies_with_atomic_add"] = len({a["body"] for a in atomics})
    stats["licm_source_candidate_pairs"] = len(licm)
    stats["licm_source_candidate_load_sites"] = len({(a["path"], a["line"]) for a in licm})
    stats["licm_source_candidate_functions"] = len({(a["path"], a["function"]) for a in licm})
    stats["exact_unique_licm_candidate_pairs"] = len({(a["body"], a["loop_ordinal"], a["load_ordinal"]) for a in licm})
    stats["exact_unique_licm_candidate_bodies"] = len({a["body"] for a in licm})
    stats["atomic_add_non_triton_or_unresolved"] = len(excluded)
    def categories(rows):
        return dict(collections.Counter((r["sem"] + (":" + str(r["sem_value"]) if r["sem"] == "explicit" else "")) for r in rows))
    unused_defaults = [a for a in atomics if a["sem"] in ("omitted", "explicit_none") and a["return_use"] in ("unused_expression", "assigned_but_no_source_reads")]
    stats["default_sem_unused_result_sites"] = len(unused_defaults)
    stats["default_sem_unused_result_inside_loop_sites"] = sum(bool(a["loops"]) for a in unused_defaults)
    stats["default_sem_unused_result_in_multiple_atomic_functions"] = sum(a["function_atomic_sites"] > 1 for a in unused_defaults)
    stats["exact_unique_default_unused_sites"] = len({(a["body"], a["ordinal"]) for a in unused_defaults})
    stats["same_module_calls_to_atomic_helpers"] = len(wrapper_calls)
    stats["same_module_calls_to_atomic_helpers_inside_loop"] = sum(bool(a["loops"]) for a in wrapper_calls)
    groups = collections.defaultdict(list)
    licm_bodies = {x["body"] for x in licm}
    for f in functions:
        if f["atomic_add_sites"] or f["body"] in licm_bodies:
            groups[f["body"]].append({k:f[k] for k in ("path", "function", "line")})
    duplicates = [dict(body=k, definitions=v) for k,v in groups.items() if len(v)>1]
    return dict(project=spec["name"], counts=dict(stats), sem_categories=categories(atomics), origin_categories=dict(collections.Counter(a["origin"] for a in atomics)), return_categories=dict(collections.Counter(a["return_use"] for a in atomics)), region_atomic_counts=dict(collections.Counter(a["region"] for a in atomics)), default_unused_atomic_region_counts=dict(collections.Counter(a["region"] for a in unused_defaults)), licm_candidate_region_counts=dict(collections.Counter(a["region"] for a in licm)), dtype_status="Destination dtype unknown globally; manually reviewed integer destinations are identified in shortlist.json.", scope_categories=dict(collections.Counter(a["scope"] + (":" + str(a["scope_value"]) if a["scope"] == "explicit" else "") for a in atomics))), atomics, licm, errors, excluded, duplicates, wrapper_calls

def fetch(spec, cache):
    root = cache / (spec["name"] + "-" + spec["revision"])
    marker = root / "READY"
    if marker.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{spec['repository']}/tar.gz/{spec['revision']}"
    count = 0
    with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            rel = member.name.split("/", 1)[-1]
            if member.isfile() and rel.endswith(".py") and any(rel.startswith(p) for p in spec["source_roots"]):
                target = root / rel
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Archive path escaped cache")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                count += 1
    marker.write_text(str(count))

def write_json(path, obj, compact=False):
    path.write_text(json.dumps(obj, indent=None if compact else 2, sort_keys=False, separators=(",", ":") if compact else None) + "\n")

def update_compiled_funnel(output):
    summary_path = output / "summary.json"
    compiled_path = output / "compiled_shortlist/summary.json"
    if not summary_path.exists() or not compiled_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    cases = json.loads(compiled_path.read_text())["cases"]
    summary["funnel"].update(
        compiled_candidates_checked=len(cases),
        compiled_kernels_with_selected_loads_remaining_in_loop=sum(case["variants"]["baseline"]["selected_loads_inside_loop"] > 0 for case in cases),
        compiled_invariant_load_sites_remaining_in_loop=sum(case["variants"]["baseline"]["selected_loads_inside_loop"] for case in cases),
        compiled_kernels_improved_under_explicit_immutable_contract=sum(case["variants"]["immutable"]["selected_loads_inside_loop"] < case["variants"]["baseline"]["selected_loads_inside_loop"] for case in cases),
        contracts_established=None,
        profitable_cases=None,
        status="Three selected LICM bodies compiled for explicit CUDA sm90 specializations. Contracts are stated preconditions, not runtime-validated allocation/immutability proofs. No followup kernel launches or timings. Atomic-loop validation belongs to the separate parent experiment.",
    )
    summary["funnel"].pop("compiled_misses", None)
    write_json(summary_path, summary)

def self_check():
    # Small scanner calibration examples: changing index/mask/other must reject;
    # an invariant local address may pass even if assigned syntactically in-loop.
    tree = ast.parse("for i in range(N):\n p = P + off\n x = tl.load(p, mask=off < K, other=0)\n y = tl.load(P + i)\n z = tl.load(P, mask=i<K)\n w = tl.load(P, other=i)\n tl.store(Q + i, x)\n")
    loop = tree.body[0]
    check = loop_invariance(loop, {"tl":"triton.language"})
    loads = [n for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="load"]
    valid = [all(check(parameter(c,n,p)[0]) for n,p in (("pointer",0),("mask",1),("other",2))) for c in loads]
    if valid != [True,False,False,False]:
        raise AssertionError(valid)
    tree = ast.parse("import triton.language as lang\nfrom triton.experimental.gluon import language as gl\nfrom triton.language import atomic_add as add\n")
    aliases = module_aliases(tree)
    for expression in ("lang.atomic_add(p,1)","gl.atomic_add(p,1)","add(p,1)"):
        if op_kind(ast.parse(expression).body[0].value,aliases,True)[1] != "resolved_module":
            raise AssertionError(expression)

def compile_shortlist(cache, output):
    """Compile unchanged selected AST bodies; never initialize or launch a GPU."""
    import importlib.util
    import re
    import sys
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    target = GPUTarget("cuda", 90, 32)
    cases = [
        dict(id="liger_scalar_cross_entropy", project="Liger", path="src/liger_kernel/ops/backends/_ascend/ops/cross_entropy.py", function="liger_cross_entropy_backward_kernel", helpers=[], immutable=["grad_output_ptr"], outputs=["dX_ptr"], constants=dict(lse_square_scale=0.0, label_smoothing=0.0, reduction="sum", BLOCK_SIZE=256, HAS_WEIGHT=False, HAS_SOFTCAPPING=False), pointer_types=dict(X_ptr="*fp32", Y_ptr="*i64", weight_ptr="*fp32", lse_ptr="*fp32", grad_output_ptr="*fp32", dX_ptr="*fp32", ce_stats_ptr="*fp32"), scalar_types=dict(softcap="fp32"), notes="Runtime positive row/column counts and strides; reduction=sum uses scalar upstream gradient. No weight, softcap, smoothing or z-loss. The kernel still has its original dynamic row and column loops. grad_output storage must be initialized and unchanged throughout invocation; disjoint from dX for alias variant."),
        dict(id="flaggems_ctc_targets", project="FlagGems", path="src/flag_gems/ops/ctc_loss.py", function="_ctc_loss_forward_kernel", helpers=["_logaddexp", "_logaddexp3"], immutable=["targets"], outputs=["log_alpha", "neg_log_likelihood"], constants=dict(T=32, N=4, C=64, MAX_TARGET=16, STATE_COUNT_MAX=33, BLANK=0, TARGET_1D=False, BLOCK_S=64), pointer_types=dict(log_probs="*fp32", targets="*i32", input_lengths="*i32", target_lengths="*i32", target_offsets="*i32", neg_log_likelihood="*fp32", log_alpha="*fp32"), scalar_types={}, notes="Padded 2D int32 targets, N=4, T=32, C=64, max target length16 and state capacity33. Original debug barriers and log_alpha recurrence retained. Valid lengths and initialized targets are caller obligations; targets are disjoint from fresh log_alpha and neg_log_likelihood outputs."),
        dict(id="aiter_recurrent_gating", project="AITER", path="aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py", function="fused_sigmoid_gating_delta_rule_update_kernel", helpers=[], immutable=["A_log", "dt_bias"], outputs=["o", "h0_source"], constants=dict(B=4,H=2,HV=2,K=32,V=32,BK=32,BV=8,USE_INITIAL_STATE=False,USE_QK_L2NORM_IN_KERNEL=False,IS_VARLEN=False), pointer_types={k:"*fp32" for k in ("A_log","a","dt_bias","q","k","v","b","o","h0_source")} | dict(h0_indices="*i32",cu_seqlens="*i32"), scalar_types=dict(softplus_beta="fp32",softplus_threshold="fp32",scale="fp32",T="i32"), notes="Original recurrent body, runtime T, B=4,H=HV=2,K=V=32,BK=32,BV=8; initial state, varlen and L2 normalization disabled. Autotune/heuristic wrappers removed to bind this specialization explicitly. A_log/dt_bias initialized immutable parameters; output o disjoint. No measurements or runtime correctness claims."),
    ]
    dest = output / "compiled_shortlist"
    dest.mkdir(parents=True, exist_ok=True)
    generated = cache / "compile_shortlist_modules"
    generated.mkdir(parents=True, exist_ok=True)
    results = []

    def pointer_loads(ir, function, signature, arguments):
        entry_args = [name for name, kind in signature.items() if kind != "constexpr"]
        selected = {entry_args.index(name):name for name in arguments}
        roots = {"%arg"+str(index):{name} for index,name in selected.items()}
        roots.update({"%"+name:{name} for name in arguments})
        depth = 0
        loop_depths = []
        found = []
        for number, line in enumerate(ir.splitlines(), 1):
            if re.search(r"\bscf\.(?:for|while)\b", line) and "{" in line:
                loop_depths.append(depth+1)
            match = re.search(r"(%[A-Za-z0-9_]+)(?::\d+)?\s*=\s*([A-Za-z0-9_.]+)\s+(.*)", line)
            if match:
                value, op, rhs = match.groups()
                operands = re.findall(r"%[A-Za-z0-9_]+", rhs)
                if op in ("tt.addptr", "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "tt.trans", "tt.bitcast", "ttg.convert_layout") and operands:
                    roots[value] = set(roots.get(operands[0], set()))
                elif op == "arith.select":
                    roots[value] = set().union(*(roots.get(x,set()) for x in operands[1:3]))
                if op == "tt.load" and operands:
                    for argument in sorted(roots.get(operands[0], set())):
                        found.append(dict(argument=argument, ir_line=number, loop_depth=len(loop_depths), operation=line.strip()))
            depth += line.count("{") - line.count("}")
            loop_depths = [level for level in loop_depths if level <= depth]
        return found

    for case in cases:
        spec = next(item for item in SPECS if item["name"] == case["project"])
        path = cache / (spec["name"] + "-" + spec["revision"]) / case["path"]
        source = path.read_text()
        tree = ast.parse(source)
        definitions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        kept = []
        for name in case["helpers"] + [case["function"]]:
            node = copy.deepcopy(definitions[name])
            node.decorator_list = [ast.Attribute(value=ast.Name(id="triton",ctx=ast.Load()),attr="jit",ctx=ast.Load())]
            kept.append(node)
        module = ast.fix_missing_locations(ast.Module(body=kept,type_ignores=[]))
        prelude = "import triton\nimport triton.language as tl\n_debug_barrier = tl.debug_barrier\n\n"
        module_path = generated / (case["id"] + ".py")
        module_path.write_text(prelude + ast.unparse(module) + "\n")
        module_ast = ast.parse(module_path.read_text())
        original_body = ast.Module(body=definitions[case["function"]].body,type_ignores=[])
        extracted_body = next(n for n in module_ast.body if isinstance(n,ast.FunctionDef) and n.name==case["function"])
        if dump(original_body) != dump(ast.Module(body=extracted_body.body,type_ignores=[])):
            raise AssertionError("Extracted body changed")
        imported_spec = importlib.util.spec_from_file_location(case["id"], module_path)
        imported = importlib.util.module_from_spec(imported_spec)
        sys.modules[case["id"]] = imported
        imported_spec.loader.exec_module(imported)
        kernel = getattr(imported,case["function"])
        signature = {name:("constexpr" if name in case["constants"] else case["pointer_types"].get(name,case["scalar_types"].get(name,"i32"))) for name in kernel.arg_names}
        entry_args = [name for name in kernel.arg_names if signature[name] != "constexpr"]
        result = dict(id=case["id"], project=case["project"], path=case["path"], function=case["function"], revision=spec["revision"], source_url=f"https://github.com/{spec['repository']}/blob/{spec['revision']}/{case['path']}#L{definitions[case['function']].lineno}", source_sha256=hashlib.sha256(source.encode()).hexdigest(), body_ast_sha256=hashlib.sha256(dump(original_body).encode()).hexdigest(), original_body_ast_equal=True, signature=signature, constants=case["constants"], immutable_arguments=case["immutable"], output_arguments=case["outputs"], assumptions=case["notes"], target=dict(backend="cuda",arch=90,warp_size=32), options=dict(num_warps=4,num_stages=1), variants={})
        for variant in ("baseline", "immutable", "alias"):
            attrs = {}
            if variant == "immutable":
                attrs = {(kernel.arg_names.index(name),):[("tt.experimental_immutable",1)] for name in case["immutable"]}
            elif variant == "alias":
                attrs = {(kernel.arg_names.index(name),):[("tt.experimental_alias_group",1 if name in case["outputs"] else 0)] for name in case["pointer_types"]}
            compiled = triton.compile(ASTSource(kernel,signature,constexprs=case["constants"],attrs=attrs),target=target,options=result["options"])
            ir = compiled.asm["ttgir"]
            irpath = dest / (case["id"] + "." + variant + ".ttgir")
            irpath.write_text(ir)
            loads = pointer_loads(ir,case["function"],signature,case["immutable"])
            result["variants"][variant] = dict(attributes={str(k):v for k,v in attrs.items()},ttgir_path=irpath.name,ttgir_sha256=hashlib.sha256(ir.encode()).hexdigest(),selected_pointer_loads=loads,selected_loads_inside_loop=sum(x["loop_depth"]>0 for x in loads),selected_loads_outside_loop=sum(x["loop_depth"]==0 for x in loads),ttgir_barrier_sites=len(re.findall(r"\b(?:gpu|ttg)\.barrier\b",ir)),shared_bytes=compiled.metadata.shared,ptx_sha256=hashlib.sha256(compiled.asm["ptx"].encode()).hexdigest())
        if case["id"] == "flaggems_ctc_targets":
            result["finding"] = "The timestep-invariant target load remains inside the loop with either contract. Original ttg.barrier appears in that loop. LoopInvariantCodeMotion.cpp::hasOnlyOrdinaryMemoryEffects rejects barriers, so the current implementation cannot use the contract here. Barrier-aware immutable-load motion needs separate legality work."
        elif case["id"] == "aiter_recurrent_gating":
            result["finding"] = "Both selected scalar loads move before the loop under immutable contracts, with a T>0 entry mask. Alias-only does not move them: function effects include loop-carried read/write pointers, which getEntryArgument deliberately does not trace."
        else:
            result["finding"] = "The scalar gradient load stays in the baseline row loop and moves before it under either contract, with an entry mask preserving zero-trip behavior."
        results.append(result)
        print(case["id"], {k:(v["selected_loads_inside_loop"],v["selected_loads_outside_loop"]) for k,v in result["variants"].items()},flush=True)
    triton_root = Path(triton.__file__).resolve().parents[2]
    native_lib = Path(triton.__file__).resolve().parent / "_C/libtriton.so"
    compiler_paths = [triton_root / "lib/Dialect/Triton/Transforms/LoopInvariantCodeMotion.cpp", triton_root / "third_party/nvidia/backend/compiler.py", native_lib]
    report = dict(status="Compile only: no GPU driver initialization, allocation, kernel launch, correctness test, or timing.", compiler_files={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in compiler_paths if path.exists()}, extraction="Bodies and helper bodies retain exact upstream ASTs; provider wrappers and unused imports are removed, explicit specialization is supplied, and _debug_barrier is bound to upstream-selected tl.debug_barrier. Liger HAS_SOFTCAPPING=False avoids its Ascend-only tanh import. Inspect TTGIR because the native LICM pass runs after TTIR.", cases=results)
    write_json(dest / "summary.json",report)
    update_compiled_funnel(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/semantic-survey-cache"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/semantic_survey_final")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--compile-shortlist-only", action="store_true", help="Requires the local experimental Triton Python environment; CPU compilation only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    self_check()
    if args.compile_shortlist_only:
        if args.fetch:
            for spec in SPECS:
                fetch(spec, args.cache)
        compile_shortlist(args.cache, args.output)
        return
    summary, atomics, licm, errors, excluded, duplicates, wrappers = [], [], [], [], [], [], []
    for spec in SPECS:
        if args.fetch:
            fetch(spec, args.cache)
        root = args.cache / (spec["name"] + "-" + spec["revision"])
        if not (root / "READY").exists():
            raise SystemExit(f"Missing pinned cache {root}; rerun --fetch")
        values = scan(spec, args.cache)
        summary.append(values[0])
        for dest, result in zip((atomics,licm,errors,excluded,duplicates,wrappers),values[1:]):
            dest.extend(result)
        print(spec["name"], json.dumps(values[0]["counts"]), flush=True)
    provenance = dict(schema=1, survey_date="2026-11-08", sources=SPECS, scanner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cache=str(args.cache), reproduction="python3 experiment_notes/semantic_candidate_experiments/semantic_prevalence_survey.py --fetch", corpus_note="Production package Python source at the listed roots, including backend forks and JIT helpers; no tests/benchmarks. Pins deliberately reuse previous experiments except Liger. This is not a synchronized current-release survey.", dedup_note="Exact AST function body+signature equality after dropping name/decorators/docstring and normalizing imported Triton aliases. Parameter/local renaming, semantic clones, generated specializations and cross-file wrapper expansion are not deduplicated. Raw counts and exact-body counts both reported.", inference_note="Source candidates only. JIT definitions include helpers. Return-use means syntactically consumed, not SSA liveness. Dtype is unknown absent manual host review. Invariant load address, mask and other are checked, but memory immutability, physical disjointness, execution safety, compiler misses and profitability are not established. Dynamic imports, rebinding, wrapper-generated kernels, descriptor loads and dynamic kwargs remain incomplete.")
    write_json(args.output / "provenance.json", provenance)
    write_json(args.output / "summary.json", dict(projects=summary, funnel=dict(source_atomic_default_unused_sites=sum(p["counts"]["default_sem_unused_result_sites"] for p in summary), source_licm_candidate_pairs=len(licm), compiled_misses=None, contracts_established=None, profitable_cases=None, status="Source survey; compiled/contract/performance stages require individual validation. Prior experiments are not extrapolated.")))
    write_json(args.output / "atomic_sites.json", atomics, True)
    write_json(args.output / "licm_candidates.json", licm, True)
    write_json(args.output / "scan_audit.json", dict(parse_errors=errors, excluded_atomic_calls=excluded, exact_duplicate_interesting_bodies=duplicates, same_module_calls_to_atomic_helpers=wrappers), True)
    update_compiled_funnel(args.output)

if __name__ == "__main__":
    main()
