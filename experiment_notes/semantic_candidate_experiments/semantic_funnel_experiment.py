#!/usr/bin/env python3
"""Map the pinned FlagGems survey to callers/tests and compile an unchanged pilot.

The inventory is source evidence, not a legality or profitability verdict. No
FlagGems package import, GPU initialization, or timing occurs without --gpu-pilot.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import urllib.request

REVISION = "3f1f896fd90f5db42d167d64fcb9691326071bc2"
REPOSITORY = "flagos-ai/FlagGems"
HERE = Path(__file__).resolve().parent
PILOT_PATH = "src/flag_gems/ops/unfold_backward.py"
PILOT_FUNCTION = "_unfold_backward_kernel"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(data, mtime=0))
    else:
        path.write_bytes(data)


def fetch(root):
    """Fetch the exact source/test/benchmark archive; do not execute it."""
    url = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{REVISION}"
    raw = urllib.request.urlopen(url, timeout=120).read()
    count = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for item in archive:
            parts = Path(item.name).parts[1:]
            if not item.isfile() or not parts or ".." in parts:
                continue
            if parts[0] not in ("src", "tests", "benchmark", "benchmarks") and parts[-1] not in ("LICENSE", "pyproject.toml", "setup.py"):
                continue
            path = root.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(item).read())
            count += 1
    write_json(root / "funnel_fetch.json", dict(url=url, revision=REVISION, archive_sha256=sha(raw), files_extracted=count))


def module_name(path):
    parts = list(Path(path).with_suffix("").parts)
    if parts[0] == "src":
        parts.pop(0)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return parent + "." + node.attr if parent else None
    if isinstance(node, ast.Subscript):
        return dotted(node.value)
    return None


def local_walk(node):
    stack = list(reversed(node.body))
    while stack:
        item = stack.pop()
        yield item
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(reversed(list(ast.iter_child_nodes(item))))


def inventory(root, survey, output):
    atomics = [x for x in json.loads((survey / "atomic_sites.json").read_text()) if x["project"] == "FlagGems" and x["region"] == "generic" and x["sem"] == "omitted" and x["return_use"] == "unused_expression"]
    licm = [x for x in json.loads((survey / "licm_candidates.json").read_text()) if x["project"] == "FlagGems" and x["region"] == "generic"]
    assert (len(atomics), len(licm)) == (91, 62), "Pinned survey denominator changed"
    groups = collections.defaultdict(lambda: {"atomic_sites": [], "licm_pairs": []})
    for kind, records in (("atomic_sites", atomics), ("licm_pairs", licm)):
        for record in records:
            item = dict(record)
            ordinal = str(item["ordinal"]) if kind == "atomic_sites" else f'{item["load_ordinal"]}:{item["loop_ordinal"]}'
            item["id"] = f'{item["path"]}:{item["function"]}:{kind}:{ordinal}'
            groups[item["path"], item["function"]][kind].append(item)

    modules, definitions, errors, autograd_classes = {}, {}, [], set()
    files = sorted(set(root.glob("src/flag_gems/**/*.py")) | set(root.glob("tests/**/*.py")) | set(root.glob("benchmark/**/*.py")))
    for path in files:
        rel = str(path.relative_to(root))
        # Vendor modules are indexed separately below for override evidence.
        if "/runtime/backend/" in rel and not rel.endswith("__init__.py"):
            continue
        try:
            source = path.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeError) as error:
            errors.append(dict(path=rel, error=str(error)))
            continue
        mod = module_name(rel)
        imports, stars = {}, []
        module_nodes = list(local_walk(tree))
        for node in module_nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    package = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
                    prefix = package.split(".")[:len(package.split(".")) - node.level + 1]
                    base = ".".join(prefix + ([base] if base else []))
                for alias in node.names:
                    if alias.name == "*":
                        stars.append(base)
                    else:
                        imports[alias.asname or alias.name] = base + "." + alias.name
        top = {node.name: node for node in module_nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        modules[mod] = dict(path=rel, tree=tree, source=source, imports=imports, stars=stars, functions=top)
        def collect(owner, scope):
            for node in local_walk(scope):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions[owner + "." + node.name] = (mod, node)
                    collect(owner + "." + node.name, node)
                elif isinstance(node, ast.ClassDef):
                    if any(dotted(base) == "torch.autograd.Function" for base in node.bases):
                        autograd_classes.add(owner + "." + node.name)
                    collect(owner + "." + node.name, node)
        collect(mod, tree)

    def resolve(name, mod, active=()):
        key = (name, mod)
        if not name or key in active or len(active) > 20:
            return None
        data = modules.get(mod)
        if not data:
            return name if name in definitions else None
        first, *tail = name.split(".")
        if first in data["imports"]:
            full = ".".join([data["imports"][first]] + tail)
            if full in definitions:
                return full
            if "." in full:
                owner, attr = full.rsplit(".", 1)
                return resolve(attr, owner, active + (key,))
        full = mod + "." + name
        if full in definitions:
            return full
        if full.endswith(".apply") and full[:-6] in autograd_classes:
            forward = full[:-6] + ".forward"
            if forward in definitions:
                return forward
        for star in reversed(data["stars"]):
            found = resolve(name, star, active + (key,))
            if found:
                return found
        return name if name in definitions else None

    def resolve_in_function(name, full, mod):
        if not name:
            return None
        prefix = full
        while prefix.startswith(mod + "."):
            candidate = prefix + "." + name
            if candidate in definitions:
                return candidate
            prefix = prefix.rsplit(".", 1)[0]
        return resolve(name, mod)

    reverse, references = collections.defaultdict(list), collections.defaultdict(list)
    for full, (mod, node) in definitions.items():
        for call in (n for n in local_walk(node) if isinstance(n, ast.Call)):
            target = resolve_in_function(dotted(call.func), full, mod)
            if target and target != full:
                reverse[target].append(dict(caller=full, line=call.lineno, expression=ast.unparse(call.func), grid_launch=isinstance(call.func, ast.Subscript), kind="direct_call_or_autograd_apply"))
            if (dotted(call.func) or "").endswith(".warp_specialize"):
                for ref in ast.walk(call):
                    if isinstance(ref, ast.Name):
                        callback = resolve_in_function(ref.id, full, mod)
                        if callback and callback != full:
                            reverse[callback].append(dict(caller=full, line=ref.lineno, expression=ref.id, grid_launch=False, kind="warp_specialize_callable_argument"))
        if not modules[mod]["path"].startswith(("tests/", "benchmark/")):
            continue
        seen = set()
        for ref in local_walk(node):
            if not isinstance(ref, (ast.Name, ast.Attribute)):
                continue
            target = resolve_in_function(dotted(ref), full, mod)
            if target and (target, ref.lineno) not in seen:
                references[target].append(dict(path=modules[mod]["path"], function=node.name, line=ref.lineno, expression=ast.unparse(ref), kind="direct_resolved_function_reference", decorators=[ast.unparse(x) for x in node.decorator_list]))
                seen.add((target, ref.lineno))

    # Dispatch registration is a separate stage: runtime vendor replacement can
    # change the globally imported symbol before _FULL_CONFIG is evaluated.
    registrations = collections.defaultdict(list)
    for node in ast.walk(modules["flag_gems"]["tree"]):
        if isinstance(node, ast.Tuple) and len(node.elts) >= 2 and isinstance(node.elts[0], ast.Constant) and isinstance(node.elts[0].value, str):
            target = resolve(dotted(node.elts[1]), "flag_gems")
            if target:
                registrations[target].append(dict(operator=node.elts[0].value, path="src/flag_gems/__init__.py", line=node.lineno))

    override_symbols = collections.defaultdict(list)
    for path in root.glob("src/flag_gems/runtime/backend/_nvidia/**/*.py"):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                override_symbols[node.name].append(dict(path=str(path.relative_to(root)), line=node.lineno, kind="same_name_definition_requires_dispatch_review"))

    def location(full):
        mod, node = definitions[full]
        return dict(symbol=full, path=modules[mod]["path"], function=node.name, line=node.lineno)

    records = []
    for (path, function), sites in sorted(groups.items()):
        full = module_name(path) + "." + function
        assert full in definitions, (path, function)
        hosts, links, visited, todo = {}, [], {full}, [full]
        while todo:
            target = todo.pop()
            for edge in reverse[target]:
                caller = edge["caller"]
                cmod, cnode = definitions[caller]
                if not modules[cmod]["path"].startswith("src/"):
                    continue
                links.append(dict(callee=target, **edge))
                jit = any("triton.jit" in ast.unparse(d) or "gluon.jit" in ast.unparse(d) for d in cnode.decorator_list)
                if not jit:
                    hosts[caller] = location(caller)
                if caller not in visited:
                    visited.add(caller)
                    todo.append(caller)
        case_refs = {json.dumps(x, sort_keys=True): x for symbol in visited for x in references[symbol]}
        regs = {json.dumps(x, sort_keys=True): x for symbol in visited for x in registrations[symbol]}
        override = [x for host in hosts.values() for x in override_symbols[host["function"]]]
        _, node = definitions[full]
        from semantic_prevalence_survey import body_hash, module_aliases
        expected_body = sites['atomic_sites'][0]['body'] if sites['atomic_sites'] else sites['licm_pairs'][0]['body']
        assert body_hash(node, module_aliases(modules[module_name(path)]['tree'])) == expected_body, (path, function, 'survey body identity mismatch')
        pilot = path == PILOT_PATH and function == PILOT_FUNCTION
        source = modules[module_name(path)]["source"]
        records.append(dict(id=path + ":" + function, path=path, function=function, line=node.lineno, source_url=f"https://github.com/{REPOSITORY}/blob/{REVISION}/{path}#L{node.lineno}", source_sha256=sha(source.encode()), body=sites["atomic_sites"][0]["body"] if sites["atomic_sites"] else sites["licm_pairs"][0]["body"], **sites,
            caller_mapping=dict(status="static_paths_found" if hosts else "no_resolved_host_path", hosts=list(hosts.values()), edges=links, caveat="Static name/import/lexical-scope resolution; includes known torch.autograd.Function.apply-to-forward and warp_specialize callback edges. No path feasibility or dynamic-wrapper proof. Backward methods are recorded as callers, without claiming an autograd execution path."),
            dispatch=dict(status="registration_found_runtime_dispatch_unverified" if regs else "no_resolved_registration", registrations=list(regs.values()), nvidia_same_name_definitions=override, runtime_replacement="src/flag_gems/__init__.py:65 SpecOpRegistrar(...).apply(); actual backend selection not executed"),
            cases=dict(status="direct_references_found" if case_refs else "no_direct_resolved_case_reference", references=list(case_refs.values()), filename_hints=[str(p.relative_to(root)) for prefix in ("tests", "benchmark") for p in sorted((root / prefix).glob("**/test_" + Path(path).stem + ".py"))], caveat="References map to concrete test/benchmark functions, not to every parameter specialization or dynamically routed operator. Filename hints are unverified."),
            contract=dict(status="reviewed_for_pilot" if pilot else "unreviewed", reason="Fresh zeros output, no alias with input, same-stream initialization, no external observer before completion; see pilot report." if pilot else None),
            compilation=dict(status="see_pilot_report" if pilot else "not_attempted"), correctness=dict(status="see_pilot_report" if pilot else "not_attempted"), timing=dict(status="see_pilot_report" if pilot else "not_attempted")))

    counts = dict(atomic_sites=len(atomics), atomic_functions=len({(x['path'], x['function']) for x in atomics}), atomic_exact_bodies=len({x['body'] for x in atomics}), atomic_sites_in_loops=sum(bool(x['loops']) for x in atomics), licm_pairs=len(licm), licm_load_sites=len({(x['path'], x['function'], x['load_ordinal']) for x in licm}), licm_functions=len({(x['path'], x['function']) for x in licm}), licm_exact_bodies=len({x['body'] for x in licm}), total_functions=len(records), static_host_path_functions=sum(bool(x['caller_mapping']['hosts']) for x in records), registration_found_functions=sum(bool(x['dispatch']['registrations']) for x in records), direct_test_reference_functions=sum(any(r['path'].startswith('tests/') for r in x['cases']['references']) for x in records), direct_benchmark_reference_functions=sum(any(r['path'].startswith('benchmark/') for r in x['cases']['references']) for x in records), contracts_reviewed_functions=1, contracts_unreviewed_functions=len(records)-1, legality_rejected_functions=0)
    write_json(output / "manifest.json.gz", dict(schema=1, revision=REVISION, records=records))
    write_json(output / "summary.json", dict(counts=counts, statuses="Unreviewed contracts, absent static links and unattempted compilations are not legality rejections. Exact body dedup is inherited from the survey, not semantic dedup. Runtime dispatch is unverified for this inventory.", parse_errors=errors, pilot_report="pilot/summary.json"))
    fetch_provenance = json.loads((root / "funnel_fetch.json").read_text())
    write_json(output / "provenance.json", dict(repository=REPOSITORY, revision=REVISION, archive=fetch_provenance, script_sha256=sha(Path(__file__).read_bytes()), inputs={p.name:sha(p.read_bytes()) for p in (survey / "atomic_sites.json", survey / "licm_candidates.json")}, mapping="Only statically resolved source calls/imported function references count. Operator-only references and filename hints do not count as case mappings. NVIDIA same-name definitions are candidates, not verified overrides.", reproduction="python3 experiment_notes/semantic_candidate_experiments/semantic_funnel_experiment.py --fetch; .venv-semantic/bin/python .../semantic_funnel_experiment.py --compile-pilot"))
    print(json.dumps(counts, indent=2), flush=True)


def compile_pilot(root, output, gpu=False, production=False):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    dest = output / "pilot" / "generic"
    dest.mkdir(parents=True, exist_ok=True)
    source = (root / PILOT_PATH).read_text()
    original = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == PILOT_FUNCTION)
    selected = copy.deepcopy(original)
    selected.decorator_list = [ast.parse("triton.jit", mode="eval").body]
    module_path = dest / "unfold_backward_extracted.py"
    module_path.write_text("# Extracted from the pinned Apache-2.0 FlagGems source; function body unchanged.\nimport triton\nimport triton.language as tl\n\n" + ast.unparse(selected) + "\n")
    original_body = ast.dump(ast.Module(body=original.body, type_ignores=[]), include_attributes=False)
    extracted = next(x for x in ast.parse(module_path.read_text()).body if isinstance(x, ast.FunctionDef))
    assert original_body == ast.dump(ast.Module(body=extracted.body, type_ignores=[]), include_attributes=False)
    spec = importlib.util.spec_from_file_location("semantic_funnel_unfold", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, PILOT_FUNCTION)
    manual_ast = copy.deepcopy(selected)
    calls = [x for x in ast.walk(manual_ast) if isinstance(x, ast.Call) and dotted(x.func) == "tl.atomic_add"]
    assert len(calls) == 1 and not any(x.arg == "sem" for x in calls[0].keywords)
    calls[0].keywords.append(ast.keyword(arg="sem", value=ast.Constant(value="relaxed")))
    manual_path = dest / "unfold_backward_manual.py"
    manual_path.write_text("# Pinned FlagGems pilot with only sem='relaxed' added.\nimport triton\nimport triton.language as tl\n\n" + ast.unparse(manual_ast) + "\n")
    audited = copy.deepcopy(manual_ast)
    audit_call = next(x for x in ast.walk(audited) if isinstance(x, ast.Call) and dotted(x.func) == "tl.atomic_add")
    audit_call.keywords = [x for x in audit_call.keywords if x.arg != "sem"]
    assert ast.dump(audited, include_attributes=False) == ast.dump(selected, include_attributes=False)
    mspec = importlib.util.spec_from_file_location("semantic_funnel_unfold_manual", manual_path)
    manual_module = importlib.util.module_from_spec(mspec)
    sys.modules[mspec.name] = manual_module
    mspec.loader.exec_module(manual_module)
    manual_kernel = getattr(manual_module, PILOT_FUNCTION)
    options = dict(num_warps=4, num_stages=1, enable_indexed_update_specialization=False, enable_private_atomic_relaxation=False, enable_contract_load_reuse=False)
    target = GPUTarget("cuda", 103, 32)
    variants, compiled = [], {}
    for dtype in ("fp16", "fp32", "bf16"):
        signature = {name: "constexpr" if name == "BLOCK" else "i32" for name in kernel.arg_names}
        signature.update(grad_in_ptr="*" + dtype, grad_out_ptr="*fp32")
        for variant in ("baseline", "private", "manual"):
            attrs = {} if variant != "private" else {(0,): [("tt.experimental_alias_group", 0)], (1,): [("tt.experimental_alias_group", 1), ("tt.experimental_private", 1), ("tt.experimental_num_ctas", 1)]}
            opts = dict(options, enable_private_atomic_relaxation=variant == "private")
            obj = triton.compile(ASTSource(manual_kernel if variant == "manual" else kernel, signature, constexprs={"BLOCK":128}, attrs=attrs), target=target, options=opts)
            compiled[dtype, variant] = obj
            assembly = {}
            for kind in ("ttir", "ttgir", "ptx"):
                data = obj.asm[kind]
                file = f"{dtype}.{variant}.{kind}.gz"
                if not production:
                    (dest / file).write_bytes(gzip.compress(data.encode(), mtime=0))
                assembly[kind] = dict(path=file, sha256=sha(data.encode()))
            variants.append(dict(dtype=dtype, variant=variant, signature=signature, constants={"BLOCK":128}, attributes={str(k):v for k,v in attrs.items()}, options=opts, assembly=assembly, transformed="tt.experimental_private_atomic_relaxed" in obj.asm["ttir"], ttir_atomics=[x.strip() for x in obj.asm['ttir'].splitlines() if 'tt.atomic_rmw' in x], ptx_atomics=[x.strip() for x in obj.asm['ptx'].splitlines() if re.search(r"\b(?:atom|red)\.", x)], ptx_cta_barriers=len(re.findall(r"\bbar\.sync\b", obj.asm['ptx'])), shared_bytes=obj.metadata.shared))
    bench_path = root / "benchmark/test_unfold.py"
    bench_tree = ast.parse(bench_path.read_text())
    shape_assignment = next(x for x in ast.walk(bench_tree) if isinstance(x, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "shapes" for t in x.targets))
    shapes = ast.literal_eval(shape_assignment.value)
    def ptx_instructions(ptx):
        # Ignore source locations/declarations/debug tables, retain executable
        # instructions and register operands for a stronger comparison than op counts.
        return [line.strip() for line in ptx.splitlines() if line.strip().endswith(";") and not line.strip().startswith((".", "//"))]
    report = dict(status="compiled_only" if not gpu else "gpu_pilot", function=PILOT_FUNCTION, source_path=PILOT_PATH, revision=REVISION, source_sha256=sha(source.encode()), original_body_ast_sha256=sha(original_body.encode()), original_body_ast_equal=True, manual_only_sem_keyword_ast_audited=True, target=dict(backend="cuda", arch=103, warp_size=32), contract=dict(status="manually_reviewed_for_wrapper_and_pilot", host_function="unfold_backward", output_argument="grad_out_ptr", evidence=[dict(path=PILOT_PATH, line=80, fact="Fresh torch.zeros fp32 output is allocated inside the wrapper before the launch."), dict(path=PILOT_PATH, line=87, fact="One launch consumes grad_in and the new output; output is only returned/cast after the launch."), dict(path=PILOT_PATH, line=55, fact="Kernel has one unused-result atomic-add and reads only grad_in. Output is never read by ordinary accesses.")], assumptions=["Same-stream initialized output is private until completion; invocation inputs remain live and within allocation bounds.", "Original wrapper valid-input domain is retained; no arbitrary pointer aliases or concurrent external observers.", "Compilation applies exactly the private-output/disjoint-input contract already implied by this allocation; no unordered-reduction assertion or changed atomic expression."]), benchmark=dict(path="benchmark/test_unfold.py", function="test_unfold_backward", sha256=sha(bench_path.read_bytes()), shape_line=shape_assignment.lineno, shapes=shapes, dtypes=["fp16", "fp32", "bf16"], input_cases=21, distinction="Seven host configurations times three input dtypes. Explicit runtime scalars yield three compiler specializations, each compiled in three variants; not 21 independent source kernels."), variants=variants, native_manual_ptx_instructions_equal={dtype:ptx_instructions(compiled[dtype,'private'].asm['ptx']) == ptx_instructions(compiled[dtype,'manual'].asm['ptx']) for dtype in ("fp16", "fp32", "bf16")}, correctness=dict(status="not_run"), timing=dict(status="not_run"), compiler=dict(version=triton.__version__, native_library_sha256=sha((Path(triton.__file__).parent / "_C/libtriton.so").read_bytes())))
    for name in (PILOT_PATH, "benchmark/test_unfold.py", "tests/test_unfold_backward.py"):
        (dest / (name.replace("/", "__") + ".gz")).write_bytes(gzip.compress((root / name).read_bytes(), mtime=0))
    if production:
        compile_production_pilot(kernel, manual_kernel, shapes, report, output)
        return
    if gpu:
        run_gpu_pilot(compiled, shapes, report)
    write_json(dest / "summary.json", report)
    print("unfold pilot:", report['status'], [(v['dtype'], v['variant'], v['transformed']) for v in variants], flush=True)


def compile_production_pilot(kernel, manual_kernel, shapes, original_report, output):
    """Use the source wrapper's actual JIT specialization and default options."""
    import triton
    from triton.compiler import ASTSource
    from triton.compiler.compiler import make_backend
    from dataclasses import asdict

    dest = output / 'pilot' / 'production'
    dest.mkdir(parents=True, exist_ok=True)
    harness = Path(__file__).read_bytes()
    (dest/'measured_harness.py.gz').write_bytes(gzip.compress(harness,mtime=0))
    (dest/'measured_harness.sha256').write_text(sha(harness)+'\n')
    report = copy.deepcopy(original_report)
    report.update(status='gpu_jit_specialized_pilot', variants=[], native_manual_ptx_instructions_equal={}, specializations={})
    report['benchmark']['distinction'] = 'All21 pinned host shape/dtype cases use original kernel.warmup with original scalar values, constexpr BLOCK128, pointer/scalar divisibility attributes, and default launch options. These specialized binaries are distinct from the explicitly runtime-i32 generic pilot. Package dispatch/allocation/final casts are not timed.'
    opts = dict(enable_indexed_update_specialization=False, enable_private_atomic_relaxation=False, enable_contract_load_reuse=False)
    artifacts = {}

    def compile_case(dtype, arguments, grid):
        args = arguments + (128,)
        baseline = kernel.warmup(*args, grid=grid, **opts)
        source = baseline.src
        actual_options = asdict(make_backend(baseline.metadata.target).parse_options(opts))
        signature_description = dict(signature=source.signature, constants={str(k):v for k,v in source.constants.items()}, baseline_attrs={str(k):v for k,v in source.attrs.items()}, options=actual_options)
        key = sha(json.dumps(signature_description, sort_keys=True, default=str).encode())[:16]
        if key not in artifacts:
            attrs = copy.deepcopy(source.attrs)
            attrs.setdefault((0,), []).append(('tt.experimental_alias_group',0))
            attrs.setdefault((1,), []).extend([('tt.experimental_alias_group',1), ('tt.experimental_private',1), ('tt.experimental_num_ctas',1)])
            private = triton.compile(ASTSource(source.fn, dict(source.signature), constexprs=copy.deepcopy(source.constants), attrs=attrs), target=baseline.metadata.target, options=dict(opts,enable_private_atomic_relaxation=True))
            manual = triton.compile(ASTSource(manual_kernel, dict(source.signature), constexprs=copy.deepcopy(source.constants), attrs=copy.deepcopy(source.attrs)), target=baseline.metadata.target, options=opts)
            pair = {'baseline':baseline,'private':private,'manual':manual}
            assert all(obj.src.signature == source.signature and obj.src.constants == source.constants for obj in pair.values())
            assert manual.src.attrs == source.attrs
            assert {k:[v for v in values if not v[0].startswith('tt.experimental_')] for k, values in private.src.attrs.items()} == source.attrs
            report['specializations'][key] = dict(dtype=dtype, **signature_description, original_signature_constants_alignment_retained=True, default_launch_options_retained=True)
            instructions = lambda obj: [line.strip() for line in obj.asm['ptx'].splitlines() if line.strip().endswith(';') and not line.strip().startswith(('.', '//'))]
            report['native_manual_ptx_instructions_equal'][key] = instructions(private) == instructions(manual)
            for variant,obj in pair.items():
                assembly = {}
                for kind in ('ttir','ttgir','ptx'):
                    data = obj.asm[kind]
                    file = f'{key}.{variant}.{kind}.gz'
                    (dest/file).write_bytes(gzip.compress(data.encode(),mtime=0))
                    assembly[kind] = dict(path=file,sha256=sha(data.encode()))
                obj._init_handles()
                report['variants'].append(dict(specialization_id=key,dtype=dtype,variant=variant,transformed='tt.experimental_private_atomic_relaxed' in obj.asm['ttir'],assembly=assembly,ptx_atomics=[x.strip() for x in obj.asm['ptx'].splitlines() if re.search(r'\b(?:atom|red)\.',x)],ptx_cta_barriers=len(re.findall(r'\bbar\.sync\b',obj.asm['ptx'])),registers=obj.n_regs,spills=obj.n_spills))
            artifacts[key] = {(dtype,v):obj for v,obj in pair.items()}
        return artifacts[key], key, args

    run_gpu_pilot({}, shapes, report, compile_case=compile_case)
    report['benchmark']['compiler_specializations'] = len(artifacts)
    report['benchmark']['compiled_variants'] = len(report['variants'])
    write_json(dest/'summary.json',report)
    print('production pilot:', len(artifacts), 'JIT specializations;', report['timing']['geometric_mean_speedup'], 'geomean',flush=True)


def run_gpu_pilot(compiled, shapes, report, compile_case=None):
    import torch
    import triton
    import triton.testing
    import statistics
    from coverage_experiment import paired_device_times
    cases = []
    torch.manual_seed(9182)
    for config in shapes:
        sizes, dim, size, step = config
        d = dim % len(sizes)
        D = sizes[d]
        L = (D - size) // step + 1
        prod_after = 1
        for value in sizes[d+1:]:
            prod_after *= value
        grad_shape = list(sizes[:d]) + [L] + list(sizes[d+1:]) + [size]
        for dtype, torch_dtype in (("fp16", torch.float16), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
            inp = torch.randn(grad_shape, dtype=torch_dtype, device="cuda")
            output = torch.zeros(sizes, dtype=torch.float32, device="cuda")
            def storage_interval(tensor):
                storage = tensor.untyped_storage()
                return storage.data_ptr(), storage.data_ptr() + storage.nbytes()
            ia, ib = storage_interval(inp)
            oa, ob = storage_interval(output)
            assert ib <= oa or ob <= ia
            arguments = (inp, output, inp.numel(), prod_after, L, size, step, D, L * prod_after * size)
            grid = (triton.cdiv(inp.numel(), 128), 1, 1)
            specialization = dtype
            active = compiled
            launch_arguments = arguments
            if compile_case:
                active, specialization, launch_arguments = compile_case(dtype, arguments, grid)
            reference = torch.ops.aten.unfold_backward(inp.double().cpu(), list(sizes), dim, size, step)
            absolute_sum = torch.ops.aten.unfold_backward(inp.double().abs().cpu(), list(sizes), dim, size, step)
            counts = torch.ops.aten.unfold_backward(torch.ones_like(inp, dtype=torch.int64, device='cpu'), list(sizes), dim, size, step)
            # Standard sequential fp32 summation error bound, per destination:
            # gamma_(n-1) * sum(abs(contribution)); atomic updates start at zero.
            n = (counts - 1).clamp(min=0).double()
            gamma = n * (2.0 ** -24) / (1 - n * (2.0 ** -24))
            bound = gamma * absolute_sum
            launch = {v: (lambda v=v: active[dtype, v][grid](*launch_arguments)) for v in ("baseline", "private", "manual")}
            error, exact = {}, {}
            for v in launch:
                output.zero_()
                launch[v]()
                difference = (output.double().cpu()-reference).abs()
                assert bool(torch.all(difference <= bound + 1e-30)), (config, dtype, v, difference.max(), bound.max())
                error[v] = float(difference.max())
            random_input = inp.clone()
            inp.copy_(torch.randint(-3, 4, grad_shape, device='cuda').to(torch_dtype))
            integer_reference = torch.ops.aten.unfold_backward(inp.double().cpu(), list(sizes), dim, size, step)
            for v in launch:
                output.zero_()
                launch[v]()
                exact[v] = bool(torch.equal(output.double().cpu(), integer_reference))
                assert exact[v], (config, dtype, v)
            inp.copy_(random_input)
            # Reset inside every timed invocation, preserving initialized output.
            functions = {}
            for v in launch:
                def trial(v=v):
                    output.zero_()
                    launch[v]()
                functions[v] = trial
            functions['reset_only'] = lambda: output.zero_()
            timings = paired_device_times(functions, seed=9182+len(cases), rounds=9, repetitions=100, warmup_ms=50)
            medians = {v: timings['median_us'][v]/1000 for v in launch}
            cases.append(dict(shape=config, dtype=dtype, specialization_id=specialization, input_elements=inp.numel(), output_elements=output.numel(), grid=grid, backing_allocations_verified_disjoint=True, destinations_with_collisions=int((counts>1).sum()), maximum_contributions=int(counts.max()), exact_integer_correctness=exact, random_maximum_error=error, random_maximum_error_bound=float(bound.max()), reset_only_ms=timings['median_us']['reset_only']/1000, reset_plus_kernel_ms=medians, device_timing=timings, speedup=medians['baseline']/medians['private'], native_to_manual=medians['private']/medians['manual']))
            print('pilot case', config, dtype, 'speedup', round(cases[-1]['speedup'], 4), flush=True)
    report['correctness'] = dict(status="passed", cases=len(cases), variants=3, exact_small_integer_cases=len(cases), colliding_cases=sum(x['destinations_with_collisions']>0 for x in cases), random_reference="CPU float64 torch.ops.aten.unfold_backward of represented input values; per-destination gamma_(n-1)*sum(abs(x)) error bound; before the wrapper's final dtype conversion")
    report['timing'] = dict(status="measured_pilot", scope="Output zero reset plus unchanged extracted kernel only. Excludes allocation, final dtype conversion, package dispatch, and full benchmark harness. Seven exact repository benchmark shapes; three dtypes. Small launch-bound cases; not a distribution over the full inventory. Reset-only is separately captured and measured, not subtracted.", protocol="Shared physical input/output buffers; CUDA graphs capture100 repetitions;9 randomized-order rounds;50ms warmup; medians across rounds. GPU exclusively assigned to this experiment. Clocks not locked.", helper=dict(path="coverage_experiment.py", sha256=sha((HERE/'coverage_experiment.py').read_bytes())), cases=cases, gpu=torch.cuda.get_device_name(), geometric_mean_speedup=float(__import__('math').exp(sum(__import__('math').log(x['speedup']) for x in cases)/len(cases))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/semantic-survey-cache"))
    parser.add_argument("--survey", type=Path, default=HERE / "results/semantic_survey_final")
    parser.add_argument("--output", type=Path, default=HERE / "results/semantic_funnel_final")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--compile-pilot", action="store_true")
    parser.add_argument("--gpu-pilot", action="store_true", help="Explicit GPU work; coordinate GPU ownership first")
    parser.add_argument("--production-pilot", action="store_true", help="GPU pilot preserving actual JIT runtime-value/alignment specialization and default launch options")
    args = parser.parse_args()
    root = args.cache / ("FlagGems-" + REVISION)
    if args.fetch:
        fetch(root)
    if not (root / "funnel_fetch.json").exists():
        raise SystemExit("Missing pinned tests/benchmarks; rerun with --fetch")
    inventory(root, args.survey, args.output)
    if args.compile_pilot or args.gpu_pilot or args.production_pilot:
        compile_pilot(root, args.output, args.gpu_pilot, args.production_pilot)
    (args.output/'LICENSE.FlagGems').write_bytes((root/'LICENSE').read_bytes())
    original_notice = (root/PILOT_PATH).read_text().split('import logging')[0]
    (args.output/'NOTICE.FlagGems').write_text('The extracted/manual unfold_backward sources and pinned snapshots derive from FlagGems commit '+REVISION+'. Their original notice follows. See LICENSE.FlagGems.\n\n'+original_notice)
    refresh_reports(args.output)


def refresh_reports(output):
    """Summarize distinct source sites, specializations, and measured cases."""
    manifest = json.loads(gzip.decompress((output/'manifest.json.gz').read_bytes()))
    summary = json.loads((output/'summary.json').read_text())
    summary.pop('pilot_report',None)
    summaries = {}
    for kind in ('generic','production'):
        path = output/'pilot'/kind/'summary.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        import importlib.util
        triton_spec=importlib.util.find_spec('triton')
        if triton_spec and triton_spec.origin:
            native=Path(triton_spec.origin).parent/'_C/libtriton.so'
            if native.exists():
                final_hash=sha(native.read_bytes())
                report['compiler']['native_library_sha256_at_finalization']=final_hash
                report['compiler']['native_library_unchanged_since_compilation']=final_hash==report['compiler']['native_library_sha256']
                assert report['compiler']['native_library_unchanged_since_compilation']
                write_json(path,report)
        times = report['timing']
        row = dict(report=str(path.relative_to(output)),status=report['status'],source_functions=1,source_atomic_sites=1,specializations=report['benchmark'].get('compiler_specializations',3),compiled_variants=len(report['variants']),correctness=report['correctness']['status'],timing=times['status'],native_transformed_specializations=sum(x['variant']=='private' and x['transformed'] for x in report['variants']),native_manual_ptx_instructions_equal=all(report['native_manual_ptx_instructions_equal'].values()))
        if times['status']=='measured_pilot':
            cases=times['cases']
            row.update(timed_shape_dtype_cases=len(cases),geometric_mean_speedup=times['geometric_mean_speedup'],speedup_range=[min(x['speedup'] for x in cases),max(x['speedup'] for x in cases)],native_to_manual_ratio_range=[min(x['native_to_manual'] for x in cases),max(x['native_to_manual'] for x in cases)],baseline_us_range=[min(x['reset_plus_kernel_ms']['baseline']*1000 for x in cases),max(x['reset_plus_kernel_ms']['baseline']*1000 for x in cases)],native_us_range=[min(x['reset_plus_kernel_ms']['private']*1000 for x in cases),max(x['reset_plus_kernel_ms']['private']*1000 for x in cases)],reset_us_range=[min(x['reset_only_ms']*1000 for x in cases),max(x['reset_only_ms']*1000 for x in cases)])
        summaries[kind]=row
    summary['pilot_reports']=summaries
    summary['pilot_report']='pilot/production/summary.json' if 'production' in summaries else ('pilot/generic/summary.json' if 'generic' in summaries else None)
    summary['funnel_limit']='Only one source function/site has contract and compiler/correctness/performance follow-through in this funnel run. The other92functions were not reviewed here; previous independent experiments are not reconciled into these counts. No population legality or profitability fraction is estimated.'
    summary['by_opportunity']={}
    for kind in ('atomic_sites','licm_pairs'):
        subset=[x for x in manifest['records'] if x[kind]]
        summary['by_opportunity'][kind]=dict(functions=len(subset),static_host_path_functions=sum(bool(x['caller_mapping']['hosts']) for x in subset),registration_found_functions=sum(bool(x['dispatch']['registrations']) for x in subset),direct_test_reference_functions=sum(any(r['path'].startswith('tests/') for r in x['cases']['references']) for x in subset),direct_benchmark_reference_functions=sum(any(r['path'].startswith('benchmark/') for r in x['cases']['references']) for x in subset))
    for record in manifest['records']:
        if record['path']==PILOT_PATH and record['function']==PILOT_FUNCTION:
            record['compilation']=dict(status='compiled' if summaries else 'not_attempted',reports=summaries,primary_report=summary['pilot_report'])
            record['correctness']=dict(status='passed' if any(x['correctness']=='passed' for x in summaries.values()) else 'not_run')
            record['timing']=dict(status='measured_pilot' if any(x['timing']=='measured_pilot' for x in summaries.values()) else 'not_run',scope='Reset plus extracted kernel only; no final cast/allocation/dispatch.')
    write_json(output/'manifest.json.gz',manifest)
    write_json(output/'summary.json',summary)
    with (output/'functions.tsv').open('w') as stream:
        stream.write('source_function\tatomic_sites\tlicm_pairs\tresolved_host_symbols\tdirect_test_references\tdirect_benchmark_references\tcontract_status\n')
        for r in manifest['records']:
            hosts=','.join(x['symbol'] for x in r['caller_mapping']['hosts']) or 'unresolved'
            tests=','.join(x['path']+':'+x['function'] for x in r['cases']['references'] if x['path'].startswith('tests/')) or 'no_direct_reference'
            benches=','.join(x['path']+':'+x['function'] for x in r['cases']['references'] if x['path'].startswith('benchmark/')) or 'no_direct_reference'
            stream.write('\t'.join(map(str,(r['id'],len(r['atomic_sites']),len(r['licm_pairs']),hosts,tests,benches,r['contract']['status'])))+'\n')
    audit={}
    for path in sorted((output/'pilot'/'generic').glob('*.py')):
        audit[str(path.relative_to(output))]=sha(path.read_bytes())
    write_json(output/'pilot/source_artifacts.json',dict(sha256=audit,notice='These extracted/manual sources are shared by both pilots. Their function bodies were AST-audited; measured production harness is retained separately. The earlier generic timing harness was extended later, so its exact full script was not archived.'))
    production=summaries.get('production',{})
    generic=summaries.get('generic',{})
    p=production.get('geometric_mean_speedup')
    g=generic.get('geometric_mean_speedup')
    c=summary['counts']
    paragraphs=[
        '# FlagGems semantic funnel: inventory and one measured pilot',
        '',
        f'Pinned FlagGems `{REVISION}`. This closes the source inventory and demonstrates one independent family; the population legality/profitability survey remains open.',
        '',
        '| Stage | Count / scope |',
        '|---|---|',
        '| Default-order unused atomic-add source sites |91sites,58functions/58exact bodies;25sites in loops |',
        '| LICM source candidates |62load-loop pairs,60loads,35functions/34exact bodies |',
        '| Combined source functions |93 |',
        f'| Static host paths |{c["static_host_path_functions"]}/93 |',
        f'| Static ATen registration references |{c["registration_found_functions"]}/93; runtime backend dispatch unverified |',
        f'| Direct repository test references |{c["direct_test_reference_functions"]}/93 |',
        f'| Direct repository benchmark references |{c["direct_benchmark_reference_functions"]}/93 |',
        '| Reviewed contract / compiled source functions |1:unfold_backward;92unreviewed |',
        '| JIT-specialized pilot |18specializations,54compiled variants,21shape/dtype cases |',
        '| Generic-signature control |3specializations,9compiled variants,21shape/dtype cases |',
        '',
        'Counts are different units: sites and load-loop pairs are not independent kernels. Exact-body dedup comes from the original survey. The [manifest](manifest.json.gz) retains every site/pair, caller edge, source hash, test/benchmark reference, registration and unresolved status; [functions.tsv](functions.tsv) is the readable function index. Missing static links, unreviewed contracts and unattempted compilation are not legality rejections.',
        '',
        'The source graph resolves lexical helpers, class methods, known autograd apply-to-forward calls and warp-specialize callbacks. It does not prove branch feasibility, backend selection or dynamic dispatch. The one function without a caller path is `ops/histogram.py::histogram_kernel`: that file’s host wrappers use torch.bucketize/scatter_add instead. This is evidence to audit reachability, not a compiler failure.',
        '',
        'The pilot uses the unchanged `ops/unfold_backward.py::_unfold_backward_kernel`. Its host wrapper creates a fresh zeroed fp32 output. The kernel reads the input and performs one unused-result atomic add. A private-output/disjoint-input contract therefore follows from that host allocation; no index uniqueness, immutable-input assertion or unordered-reduction permission was invented.',
        '',
        f'On NVIDIA B300, the JIT-specialized pilot measured **{p:.4f}×** geometric-mean baseline/native speedup over the21exact upstream benchmark shape/dtype cases.' if p else 'The JIT-specialized pilot has not been measured.',
        'This timing includes output reset plus the extracted update kernel. It excludes final dtype conversion, allocation, Python/package dispatch and the full FlagGems benchmark harness; it is not a whole-operator speedup. Original warmup scalar values, pointer/scalar divisibility, constexprs and default launch options are preserved. Other experimental transformations are disabled consistently.',
        '',
        'The native pass transformed all18JIT specializations. Baseline PTX uses `atom.global.gpu.acq_rel.add.f32` and2CTA barriers; native/manual use `red.global.gpu.relaxed.add.f32` with0CTA barriers. Native and the sem-only manually relaxed control have identical executable PTX for every specialization. This comparison combines ordering and atom/red selection; it does not isolate barrier latency.',
        '',
        'All21cases passed exact small-integer checks and random-input checks against CPU float64 accumulation with per-destination gamma_(n-1)×sum(abs(x)) bounds;12cases contain collisions. Timing uses shared physical buffers, full backing-allocation disjointness checks, reset inside every captured invocation,100graph repetitions and9randomized-order rounds. Reset-only time is recorded separately and never subtracted.',
        '',
        f'The earlier generic-signature control measured{g:.4f}×. It forces scalars to runtime i32 and omits upstream alignment/value specialization; it is retained separately under `pilot/generic/` and is not pooled with `pilot/production/`.' if g else 'The generic control is separate from the JIT-specialized pilot.',
        '',
        'The other92functions, including all35LICM functions, received no additional contract/compiler/performance review in this funnel run. Previous independent experiments are not reconciled into these counts. The selected successful pilot does not estimate the share of the91atomic sites or62LICM pairs that are legal, transformable or profitable.',
        '',
        'Reports: [funnel summary](summary.json), [JIT-specialized pilot](pilot/production/summary.json), [generic control](pilot/generic/summary.json), [provenance](provenance.json). The native compiler library hash is unchanged between compilation and artifact finalization for both pilots.',
        '',
        'Reproduce from the Triton checkout with `python3 experiment_notes/semantic_candidate_experiments/semantic_funnel_experiment.py --fetch` for the inventory, then the semantic virtual environment’s Python with `--production-pilot` for the measured JIT-specialized pilot. `--gpu-pilot` reproduces the generic control. GPU ownership must be coordinated. Pinned tests/benchmarks and archive SHA are in provenance; production measured harness, IR/PTX and source snapshots are retained. See `LICENSE.FlagGems` and `NOTICE.FlagGems`.',
        '',
    ]
    # Keep numerals separated from words in user-facing prose.
    prose='\n'.join(paragraphs)
    prose=''.join(part if part.startswith('`') else re.sub(r'([,;])(?=[0-9])',r'\1 ',re.sub(r'(?<=[0-9])(?=[A-Za-z])|(?<=[A-Za-z])(?=[0-9])',' ',part)) for part in re.split(r'(`[^`]*`)',prose))
    # Restore technical names/numeric types where whitespace is significant.
    for old,new in (('fp 32','fp32'),('i 32','i32'),('float 64','float64'),('f 32','f32'),('B 300','B300')):
        prose=prose.replace(old,new)
    (output/'README.md').write_text(prose)


if __name__ == "__main__":
    main()
