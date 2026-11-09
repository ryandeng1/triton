"""Audited source-stage controls for the cross-kernel slowdown investigation.

This helper does not compile or launch kernels.  Stage labels change ordinary loop
iterator calls: ``range(...)`` becomes ``tl.range(..., num_stages=N)``, or an
existing ``tl.range`` has its ``num_stages`` keyword replaced.  The inverse edit
must recover the entire original function AST before any variant is returned.
The separate ``load_cache_ca`` label changes only ``tl.load`` cache annotations
to ``cache_modifier=".ca"`` and preserves source loop stages; its caller can vary
the compiler's automatic pipeline stage option independently.

Use ``contract_pipeline_stages=0`` when compiling these variants, and supply the
same validated alias contracts and launch configuration as the comparison.
``source_stages1`` is the synchronous source-stage control.  It is distinct from
the compiler option ``contract_pipeline_stages=0``, which leaves pre-existing
source stages (including the AITER RMSNorm stages2 loops) enabled.

Example::

    fn = get_variant_fn(spec["fn"], "logsoftmax_sum_only2", output_dir)
    audit = variant_audit(fn)

Generated Python and JSON audit files have content-addressed stable names.
The copied original globals preserve dependencies such as ``tle = tl``; imports
of this helper and generation of a variant do not initialize a GPU context.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import types


ALL_LOOP_LABELS = tuple(f"source_stages{stage}" for stage in (1, 2, 3, 4))
SELECTIVE_LABELS = (
    "logsoftmax_sum_only2",
    "logsoftmax_output_only2",
    "rotary_q_only2",
    "rotary_k_only2",
)
LABELS = ALL_LOOP_LABELS + SELECTIVE_LABELS + ("load_cache_ca",)
_AUDIT_ATTRIBUTE = "_pipeline_slowdown_control_audit"


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _ordinary_loops(function):
    loops = []
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        iterator = node.iter
        if not isinstance(iterator, ast.Call):
            continue
        callee = iterator.func
        ordinary = isinstance(callee, ast.Name) and callee.id == "range"
        ordinary |= (
            isinstance(callee, ast.Attribute)
            and isinstance(callee.value, ast.Name)
            and callee.value.id in ("tl", "tle")
            and callee.attr == "range"
        )
        if ordinary:
            loops.append(node)
    return sorted(loops, key=lambda node: (node.lineno, node.col_offset))


def _function(module):
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or len(module.body) != 1:
        raise ValueError("Expected the source of exactly one kernel function")
    function = functions[0]
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
           for statement in function.body for node in ast.walk(statement)):
        raise ValueError("Nested functions are outside the scope of this control")
    return function


def _loads(function):
    return sorted((
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "load"
    ), key=lambda node: (node.lineno, node.col_offset))


def _set_keyword(call, name, value):
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("Cannot audit a call containing **kwargs")
    keywords = [keyword for keyword in call.keywords if keyword.arg == name]
    if len(keywords) > 1:
        raise ValueError(f"Duplicate {name} keyword")
    if keywords:
        keywords[0].value = ast.Constant(value)
    else:
        call.keywords.append(ast.keyword(arg=name, value=ast.Constant(value)))


def _stages_for_label(function, loops, label):
    if label in ALL_LOOP_LABELS:
        return [int(label[-1])] * len(loops)
    expected = {
        "logsoftmax_sum_only2": ("_log_softmax_bwd_kernel", 0),
        "logsoftmax_output_only2": ("_log_softmax_bwd_kernel", 1),
        "rotary_q_only2": ("apply_rotary_pos_emb_kernel", 0),
        "rotary_k_only2": ("apply_rotary_pos_emb_kernel", 1),
    }
    if label not in expected:
        raise ValueError(f"Unknown control {label!r}; expected one of {LABELS}")
    kernel_name, selected = expected[label]
    if function.name != kernel_name or len(loops) != 2:
        raise ValueError(f"{label} requires {kernel_name} with exactly two ordinary loops")
    # Selective controls are specific to the pinned source.  Check the original
    # bounds as well as the function name so a source drift cannot select a
    # different loop while retaining a misleading label.
    expected_arguments = (["0", "num_chunks"] if "logsoftmax" in label else
                          ["0", "NUM_Q_HEADS"])
    arguments = [ast.unparse(arg) for arg in loops[0].iter.args]
    if arguments != expected_arguments:
        raise ValueError(f"Unexpected first-loop bounds for {label}: {arguments}")
    expected_arguments = (["0", "num_chunks"] if "logsoftmax" in label else
                          ["0", "NUM_K_HEADS"])
    arguments = [ast.unparse(arg) for arg in loops[1].iter.args]
    if arguments != expected_arguments:
        raise ValueError(f"Unexpected second-loop bounds for {label}: {arguments}")
    stages = [1, 1]
    stages[selected] = 2
    return stages


def _write_identical_or_new(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Refusing to replace a different generated artifact: {path}")
    else:
        path.write_text(text)


def get_variant_fn(original, label, output_dir):
    """Return a new JITFunction and save its source and reversible-edit audit.

    ``original`` is an existing Triton JITFunction.  Its source and globals are
    never modified.  No memory addresses, arithmetic, masks, loop bounds,
    barriers, or launch settings are changed.  Stage counts are source schedule
    controls, not additional alias facts; the caller remains responsible for
    validating disjoint allocations when licensing overlapping iterations.
    """
    if label not in LABELS:
        raise ValueError(f"Unknown control {label!r}; expected one of {LABELS}")
    source = original.src
    original_module = ast.parse(source)
    original_function = _function(original_module)
    original_loops = _ordinary_loops(original_function)
    is_stage_control = label != "load_cache_ca"
    if is_stage_control and not original_loops:
        raise ValueError(f"{original_function.name} has no ordinary loops to control")
    stages = (_stages_for_label(original_function, original_loops, label)
              if is_stage_control else [])
    transformed = copy.deepcopy(original_module)
    function = _function(transformed)
    loops = _ordinary_loops(function)
    loop_audit = []
    for number, (loop, stage) in enumerate(zip(loops, stages)):
        old_iterator = copy.deepcopy(loop.iter)
        # Keep all other tl.range keyword arguments in their original order.
        if isinstance(loop.iter.func, ast.Name):
            loop.iter.func = ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()),
                                           attr="range", ctx=ast.Load())
        _set_keyword(loop.iter, "num_stages", stage)
        loop_audit.append(dict(
            ordinal=number,
            original_line=loop.lineno,
            induction_variable=ast.unparse(loop.target),
            original_iterator=ast.unparse(old_iterator),
            variant_iterator=ast.unparse(loop.iter),
            source_num_stages=stage,
        ))
    load_audit = []
    if not is_stage_control:
        loads = _loads(function)
        if not loads:
            raise ValueError(f"{original_function.name} has no tl.load calls to control")
        for number, load in enumerate(loads):
            # This checkout's tl.load accepts pointer, mask, and other before
            # its keyword-only cache options.  Refuse any different call form.
            if len(load.args) > 3:
                raise ValueError("Unexpected positional tl.load arguments")
            old_cache = next((ast.unparse(keyword.value) for keyword in load.keywords
                              if keyword.arg == "cache_modifier"), None)
            _set_keyword(load, "cache_modifier", ".ca")
            load_audit.append(dict(ordinal=number, original_line=load.lineno,
                                   original_cache_modifier=old_cache,
                                   variant_cache_modifier=".ca"))
    function.name = f"{original_function.name}__{label}"
    ast.fix_missing_locations(transformed)

    # Reparse the actual generated file text, then invert only our edits.  This
    # proves the complete executed function AST agrees with the original after
    # restoring its name and ordinary loop iterators (including signatures and
    # every nested statement), rather than merely comparing selected op counts.
    generated_source = (
        "# Generated by pipeline_slowdown_controls.py; audited compiler annotations only.\n"
        f"# Original source SHA256: {_sha(source)}\n"
        f"# Variant: {label}; automatic pipeline option: "
        + ("must be 0.\n" if is_stage_control else "controlled by caller.\n")
        + ast.unparse(transformed) + "\n"
    )
    reparsed = ast.parse(generated_source)
    restored = copy.deepcopy(reparsed)
    restored_function = _function(restored)
    restored_function.name = original_function.name
    restored_loops = _ordinary_loops(restored_function)
    if len(restored_loops) != len(original_loops):
        raise AssertionError("The generated function changed the ordinary loop count")
    if is_stage_control:
        for loop, original_loop in zip(restored_loops, original_loops):
            loop.iter = copy.deepcopy(original_loop.iter)
    else:
        restored_loads = _loads(restored_function)
        original_loads = _loads(original_function)
        if len(restored_loads) != len(original_loads):
            raise AssertionError("The generated function changed the tl.load count")
        for load, original_load in zip(restored_loads, original_loads):
            # Restore cache_modifier alone while retaining all other variant
            # keyword nodes, so the inverse check also catches unintended edits
            # to masks, volatility, eviction policy, and all load arguments.
            load.keywords = [keyword for keyword in load.keywords
                             if keyword.arg != "cache_modifier"]
            for index, keyword in enumerate(original_load.keywords):
                if keyword.arg == "cache_modifier":
                    load.keywords.insert(index, copy.deepcopy(keyword))
    if _dump(restored) != _dump(original_module):
        raise AssertionError("Stage-control inverse edit did not recover the original AST")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = _sha(generated_source)
    stem = f"{original_function.name}__{label}_{digest[:16]}"
    path = output_dir / f"{stem}.py"
    audit_path = output_dir / f"{stem}.audit.json"
    audit = dict(
        label=label,
        original_function=original_function.name,
        original_source_file=str(Path(original.file_name).resolve()),
        original_source_sha256=_sha(source),
        original_function_ast_sha256=_sha(_dump(original_module)),
        variant_function=function.name,
        variant_source_file=str(path),
        variant_source_sha256=digest,
        variant_function_ast_sha256=_sha(_dump(reparsed)),
        audit_file=str(audit_path),
        inverse_edit_recovers_original_function_ast=True,
        signature_unchanged=True,
        computations_and_memory_operations_unchanged=True,
        source_loop_controls=loop_audit,
        load_cache_controls=load_audit,
        permitted_edit=("ordinary loop range callee and num_stages annotation"
                        if is_stage_control else "tl.load cache_modifier annotation"),
        required_contract_pipeline_stages=0 if is_stage_control else None,
        alias_contracts="Caller supplies the same independently validated alias facts as the baseline.",
    )
    _write_identical_or_new(path, generated_source)
    _write_identical_or_new(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")

    # A real source file lets Triton's inspect-based JIT constructor work, and a
    # unique module keeps its function registry separate from the original.
    module_name = "_pipeline_control_" + re.sub(r"[^a-zA-Z0-9_]", "_", stem)
    module = types.ModuleType(module_name)
    module.__dict__.update(original.get_capture_scope())
    module.__name__ = module_name
    module.__file__ = str(path)
    module.__package__ = ""
    import triton
    import triton.language as tl
    module.__dict__["triton"] = triton
    module.__dict__["tl"] = tl
    exec(compile(generated_source, str(path), "exec"), module.__dict__)
    generated_function = module.__dict__[function.name]
    variant = triton.jit(
        generated_function,
        version=original.version,
        do_not_specialize=original.do_not_specialize,
        do_not_specialize_on_alignment=original.do_not_specialize_on_alignment,
        debug=original.debug,
        noinline=original.noinline,
        repr=original._repr,
        launch_metadata=original.launch_metadata,
    )
    module.__dict__[function.name] = variant
    sys.modules[module_name] = module
    setattr(variant, _AUDIT_ATTRIBUTE, audit)
    return variant


def variant_audit(variant):
    """Return a defensive copy of the audit attached by ``get_variant_fn``."""
    try:
        return copy.deepcopy(getattr(variant, _AUDIT_ATTRIBUTE))
    except AttributeError as error:
        raise ValueError("The function was not generated by get_variant_fn") from error
