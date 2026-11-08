import numpy
import pytest
import torch
import triton
import triton.language as tl
from collections import namedtuple
from triton._C.libtriton import native_specialize_impl
from triton.runtime.jit import MockTensor, JITCallable
from triton._utils import canonicalize_dtype
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.amd.compiler import HIPBackend
from triton.language import constexpr
from triton.language import target_info
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor
from triton.experimental.gluon.language._layouts import NVMMASharedLayout
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def mock_tensor_from_tensor(tensor):
    return MockTensor(tensor.dtype, tensor.shape)


class MockJITCallable(JITCallable):

    def __init__(self):
        pass

    def cache_key(self):
        return "mock_jit_callable"


class MockFloat(float):

    def __new__(cls, value):
        return super().__new__(cls, value)


class MockInt(int):

    def __new__(cls, value):
        return super().__new__(cls, value)


def reference_specialize_impl(backend, arg, is_const, specialize_value, align, *, nested=False):
    if arg is None:
        return ("constexpr", None)
    elif isinstance(arg, bool):
        return ("u1", None)
    elif isinstance(arg, int):
        key = backend.get_int_specialization(arg, align=align) if specialize_value else None
        if arg == 1 and specialize_value:
            return ("constexpr", 1)
        elif -(2**31) <= arg and arg <= 2**31 - 1:
            return ("i32", key)
        elif 2**63 <= arg and arg <= 2**64 - 1:
            return ("u64", key)
        else:
            return ("i64", key)
    elif isinstance(arg, float):
        return ("fp32", None)
    elif hasattr(arg, "data_ptr"):
        dsk = (arg.dtype, is_const)
        res = ("*k" if dsk[1] else "*") + canonicalize_dtype(dsk[0])
        key = backend.get_tensor_specialization(arg, align=align) if specialize_value else None
        return (res, key)
    elif isinstance(arg, JITCallable):
        return ("constexpr", arg.cache_key)
    elif isinstance(arg, constexpr):
        return ("constexpr", arg)
    elif isinstance(arg, tuple):
        fields = getattr(arg, "_fields", ())
        skip = getattr(type(arg), "__triton_do_not_specialize__", ())
        spec = []
        for i, x in enumerate(arg):
            specialize = (not nested or specialize_value) and not (fields and fields[i] in skip)
            spec.append(reference_specialize_impl(backend, x, False, specialize, True, nested=True))
        make_tuple = lambda vals: type(arg)(*vals) if hasattr(arg, "_fields") else tuple(vals)
        tys = make_tuple([x[0] for x in spec])
        keys = make_tuple([x[1] for x in spec])
        return (tys, keys)
    elif isinstance(arg, TensorDescriptor):
        assert hasattr(arg.base, "data_ptr")
        inner = canonicalize_dtype(arg.base.dtype)
        return (f"tensordesc<{inner}{list(arg.block_shape)}>", None)
    elif isinstance(arg, GluonTensorDescriptor):
        assert hasattr(arg.base, "data_ptr")
        inner = canonicalize_dtype(arg.base.dtype)
        is_im2col = arg.__class__.__name__ == "TensorDescriptorIm2Col"
        type_name = "tensordesc_im2col" if is_im2col else "tensordesc"
        # For im2col mode, include the original tensor rank in the signature
        rank_suffix = f",input_rank={len(arg.shape)}" if is_im2col else ""
        return (f"{type_name}<{inner}{list(arg.block_shape)}{rank_suffix},{arg.layout!r}>", None)
    else:
        raise TypeError("Unsupported type: %s" % type(arg))


def native_inputs_to_specialize():
    return [
        1.0,
        None,
        False,
        True,
        1,
        0,
        -1,
        16,
        17,
        2**31 - 1,
        2**31,
        -2 * 31 - 1,
        2**63 - 1,
        2**63,
        2**63 + 1,
        2**64 - 1,
    ]


def derived_inputs_to_specialize():
    return [
        constexpr(1),
        constexpr(False),
        constexpr(1.0),
        numpy.float64(1.0),
        MockFloat(1.0),
        MockInt(1),
        MockJITCallable(),
    ]


def tuples_to_specialize():
    return [
        (1, 1),
        (False, True),
        namedtuple('strides', ['x', 'y'])(1, 1),
        namedtuple('flags', ['x', 'y'])(False, True),
    ]


def tensors_to_specialize():
    return [
        torch.empty(shape, dtype=dtype, device="cpu")
        for shape in [(1, ), (1, 1), (16, ), (16, 16), (128, ), (128, 128)]
        for dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
    ]


def tensordescriptors_to_specialize():
    return [
        TensorDescriptor.from_tensor(tensor, block_shape=tensor.shape)
        for tensor in tensors_to_specialize()
        if tensor.shape[-1] % 16 == 0
    ]


def gluon_tensordescriptors_to_specialize():
    return [
        GluonTensorDescriptor.from_tensor(
            tensor,
            block_shape=tensor.shape,
            layout=NVMMASharedLayout(0, tensor.dtype.itemsize * 8, len(tensor.shape)),
        ) for tensor in tensors_to_specialize() if tensor.shape[-1] % 16 == 0 and tensor.dtype.itemsize <= 4
    ]


def test_specialize_gluon_fp4_descriptor_without_active_target(monkeypatch):
    monkeypatch.setattr(target_info, "current_target", lambda: None)
    tensor = torch.empty((128, 128), dtype=torch.uint8)
    layout = NVMMASharedLayout(128, 8, fp4_padded=True)

    descriptor = GluonTensorDescriptor.from_tensor(tensor, [128, 64], layout)

    assert native_specialize_impl(CUDABackend, descriptor, False, True,
                                  True) == reference_specialize_impl(CUDABackend, descriptor, False, True, True)


def mock_tensors_to_specialize():
    return [mock_tensor_from_tensor(tensor) for tensor in tensors_to_specialize()]


@pytest.mark.parametrize("handle", [1, 7, 16])
def test_namedtuple_do_not_specialize(handle):

    class Arguments(namedtuple("Arguments", "handle dynamic default")):
        __triton_do_not_specialize__ = ("handle", "dynamic")

    pointer = torch.empty(1, device="cpu", dtype=torch.float32)
    constant = constexpr(3)
    arg = Arguments(handle, (pointer, constant), (pointer, 1))
    (types, ), (keys, ) = native_specialize_impl(CUDABackend, (arg, ), False, True, True)
    assert types == Arguments("i32", ("*fp32", "constexpr"), ("*fp32", "constexpr"))
    assert keys == Arguments(None, (None, constant), ("D", 1))
    assert ((types, ), (keys, )) == reference_specialize_impl(CUDABackend, (arg, ), False, True, True)


@pytest.mark.parametrize("names,exc,message", [
    ("handle", TypeError, "must be a tuple of field names"),
    (("unknown", ), ValueError, "unknown field 'unknown'"),
])
def test_namedtuple_do_not_specialize_invalid_fields(names, exc, message):

    class Arguments(namedtuple("Arguments", "handle")):
        __triton_do_not_specialize__ = names

    with pytest.raises(exc, match=message):
        native_specialize_impl(CUDABackend, Arguments(1), False, True, True)


@pytest.mark.parametrize("input_generator", [
    native_inputs_to_specialize,
    tuples_to_specialize,
    tensors_to_specialize,
    tensordescriptors_to_specialize,
    gluon_tensordescriptors_to_specialize,
    mock_tensors_to_specialize,
])
@pytest.mark.parametrize("backend", [CUDABackend, HIPBackend])
@pytest.mark.parametrize("is_const", [True, False])
@pytest.mark.parametrize("specialize_value", [True, False])
@pytest.mark.parametrize("align", [True, False])
def test_specialize_impl(input_generator, backend, is_const, specialize_value, align):
    for arg in input_generator():
        result = native_specialize_impl(backend, arg, is_const, specialize_value, align)
        expected = reference_specialize_impl(backend, arg, is_const, specialize_value, align)
        assert result == expected


@pytest.mark.parametrize("torch_name,dtype,canonical", [
    ("float4_e2m1fn_x2", tl.float4e2m1x2, "fp4e2m1x2"),
    ("float8_e8m0fnu", tl.float8e8m0fnu, "fp8e8m0fnu"),
])
def test_native_dtype_torch_contract(torch_name, dtype, canonical):
    torch_dtype = getattr(torch, torch_name, None)
    if torch_dtype is None:
        pytest.skip(f"Torch does not expose {torch_name}")
    storage = torch.arange(32, dtype=torch.uint8)
    native = storage.view(torch_dtype)
    assert canonicalize_dtype(torch_dtype) == canonicalize_dtype(dtype) == canonical
    assert dtype.itemsize == torch_dtype.itemsize == 1
    for backend in (CUDABackend, HIPBackend):
        for value, name in ((storage, "u8"), (native, canonical), (storage, "u8"), (native, canonical)):
            signature, _ = native_specialize_impl(backend, value, False, True, True)
            assert signature == "*" + name


def _semantic_fact_outputs(fn, *args):
    from torch.fx.experimental.proxy_tensor import make_fx
    from triton.experimental.semantic_contracts.analysis import FactsAnalyzer

    graph = make_fx(fn, tracing_mode="fake")(*args)
    analysis = FactsAnalyzer(graph).analyze()
    output = next(node for node in graph.graph.nodes if node.op == "output")
    values = output.args[0]
    if not isinstance(values, (list, tuple)):
        values = (values, )
    return graph, analysis, output, [analysis.at(output, value) for value in values]


def test_semantic_facts_permutation_preserved_by_subsequence_and_reordering():

    def graph(x):
        indices = torch.randperm(32, dtype=torch.int64, device=x.device)
        return indices[1::2].clone().reshape(4, 4).transpose(0, 1).reshape(-1)

    _, _, _, facts = _semantic_fact_outputs(graph, torch.empty(1))
    assert facts[0].unique_global
    assert facts[0].integer_range.lower == 0
    assert facts[0].integer_range.upper == 31


def test_semantic_facts_range_proves_lossless_cast():

    def graph(x):
        small = torch.arange(16, device=x.device, dtype=torch.int64).to(torch.int32)
        # Narrowing loses distinct values after wraparound: these endpoints are
        # separated by 2**32, so both become zero in int32.
        large = torch.arange(0, 2**32 + 1, 2**32, device=x.device, dtype=torch.int64).to(torch.int32)
        return small, large

    _, _, _, facts = _semantic_fact_outputs(graph, torch.empty(1))
    assert facts[0].unique_global
    assert not facts[1].unique_global


@pytest.mark.parametrize("operation", ["remainder", "clamp", "cat", "gather", "repeat", "expand"])
def test_semantic_facts_noninjective_operation_drops_uniqueness(operation):

    def graph(x):
        indices = torch.arange(8, device=x.device, dtype=torch.int64)
        if operation == "remainder":
            return indices.remainder(3)
        if operation == "clamp":
            return indices.clamp(max=3)
        if operation == "cat":
            return torch.cat((indices, indices))
        if operation == "gather":
            return indices.gather(0, torch.zeros(8, dtype=torch.int64, device=x.device))
        if operation == "repeat":
            return indices.repeat(2)
        return indices[:1].expand(8)

    _, _, _, facts = _semantic_fact_outputs(graph, torch.empty(1))
    assert facts[0] is None or not facts[0].unique_global


def test_semantic_facts_mutation_invalidates_alias_but_preserves_clone():

    def graph(x):
        indices = torch.randperm(16, device=x.device)
        saved = indices.clone()
        indices[::2].fill_(0)
        return indices, saved

    _, _, _, facts = _semantic_fact_outputs(graph, torch.empty(1))
    assert not facts[0].unique_global
    assert facts[1].unique_global


def test_semantic_facts_unsafe_view_mutation_invalidates_base():

    def graph(x):
        indices = torch.arange(16, device=x.device)
        view = torch.ops.aten._unsafe_view.default(indices, [4, 4])
        view[0].fill_(0)
        return indices

    _, _, _, facts = _semantic_fact_outputs(graph, torch.empty(1))
    assert not facts[0].unique_global


def _semantic_unknown_effect(value):
    return value


def test_semantic_facts_unknown_call_invalidates_proof():
    from torch.fx.experimental.proxy_tensor import make_fx
    from triton.experimental.semantic_contracts.analysis import FactsAnalyzer

    graph = make_fx(lambda x: torch.arange(8, device=x.device), tracing_mode="fake")(torch.empty(1))
    output = next(node for node in graph.graph.nodes if node.op == "output")
    value = output.args[0]
    with graph.graph.inserting_before(output):
        graph.graph.call_function(_semantic_unknown_effect, (value, ))
    analysis = FactsAnalyzer(graph).analyze()
    assert not analysis.at(output, value).unique_global


def test_semantic_facts_unmodeled_custom_schema_invalidates_proof():
    from torch.fx.experimental.proxy_tensor import make_fx
    from triton.experimental.semantic_contracts.analysis import FactsAnalyzer

    # A custom schema saying that its argument is not mutated is insufficient
    # authority for the analysis: unmodeled effects can access captured aliases.
    library = torch.library.Library("triton_semantics_test_effects", "FRAGMENT")
    library.define("unknown(Tensor value) -> Tensor")
    target = torch.ops.triton_semantics_test_effects.unknown.default
    graph = make_fx(lambda x: torch.arange(8, device=x.device), tracing_mode="fake")(torch.empty(1))
    output = next(node for node in graph.graph.nodes if node.op == "output")
    value = output.args[0]
    with graph.graph.inserting_before(output):
        unknown = graph.graph.call_function(target, (value, ))
        unknown.meta["val"] = torch.empty(8, dtype=torch.int64)
    analysis = FactsAnalyzer(graph).analyze()
    assert not analysis.at(output, value).unique_global


@pytest.mark.parametrize("operation", ["resize", "set_storage"])
def test_semantic_facts_storage_mutation_erases_freshness(operation):
    from torch.fx.experimental.proxy_tensor import make_fx
    from triton.experimental.semantic_contracts.analysis import FactsAnalyzer

    graph = make_fx(lambda x: torch.arange(8, device=x.device), tracing_mode="fake")(torch.empty(16, dtype=torch.int64))
    output = next(node for node in graph.graph.nodes if node.op == "output")
    value = output.args[0]
    external = next(node for node in graph.graph.nodes if node.op == "placeholder")
    with graph.graph.inserting_before(output):
        if operation == "resize":
            mutation = graph.graph.call_function(torch.ops.aten.resize_.default, (value, [16]))
        else:
            mutation = graph.graph.call_function(torch.ops.aten.set_.source_Tensor, (value, external))
        mutation.meta["val"] = torch.empty(16, dtype=torch.int64)
    facts = FactsAnalyzer(graph).analyze().at(output, value)
    assert not facts.unique_global
    assert not facts.fresh


def test_semantic_facts_placeholder_metadata_cannot_assert_uniqueness():
    from torch.fx.experimental.proxy_tensor import make_fx
    from triton.experimental.semantic_contracts.analysis import FactsAnalyzer

    graph = make_fx(lambda x: x, tracing_mode="fake")(torch.arange(8))
    value = next(node for node in graph.graph.nodes if node.op == "placeholder")
    # Unsupported user metadata is never a proof authority.
    value.meta["unique_global"] = True
    value.meta["semantic_contracts"] = {"unique_global": True}
    output = next(node for node in graph.graph.nodes if node.op == "output")
    assert not FactsAnalyzer(graph).analyze().at(output, value).unique_global


def _semantic_require_cuda(device):
    if torch.version.cuda is None or not str(device).startswith("cuda"):
        pytest.skip("Semantic-contract CUDA lowerings require NVIDIA PTX")


def _semantic_report_node(compiled, operation):
    return next(node for node in compiled.report()["nodes"] if node["op"] == operation)


def _semantic_update_has_atomic(compiled):
    import re
    node = _semantic_report_node(compiled, "embedding_backward")
    ptx = compiled.kernels[node["name"]]["update"].asm["ptx"]
    return bool(re.search(r"\b(?:atom|red)\.", ptx))


@pytest.mark.parametrize("baseline,inferred,expected", [
    (10.0, 8.0, "inferred"),
    (10.0, 11.0, "baseline"),
    (10.0, 9.9, "baseline"),
    (10.0, 10.0, "baseline"),
    (10.0, 9.8, "baseline"),
    (10.0, float("nan"), "baseline"),
])
def test_semantic_profitability_keeps_baseline_without_clear_win(baseline, inferred, expected):
    from triton.experimental.semantic_contracts.backend import choose_hoist
    assert choose_hoist({"median_us": {"baseline": baseline, "inferred": inferred}}) == expected


def _semantic_embedding_reference(grad, indices, vocabulary, padding=-1, scale=False):
    # Compute an independent host FP64 oracle, rather than using the same GPU
    # scatter implementation as the candidate.
    g = grad.detach().cpu().double()
    i = indices.detach().cpu().long()
    valid = (i >= 0) & (i < vocabulary) & (i != padding)
    result = torch.zeros((vocabulary, grad.shape[1]), dtype=torch.float64)
    counts = torch.bincount(i[valid], minlength=vocabulary)
    for row in torch.where(valid)[0].tolist():
        update = g[row] / counts[i[row]] if scale else g[row]
        result[i[row]] += update
    return result.to(device=grad.device, dtype=grad.dtype)


@pytest.mark.parametrize("producer", ["randperm", "arange"])
@pytest.mark.parametrize("scale", [False, True])
def test_semantic_graph_unique_producer_removes_atomics_and_count(device, producer, scale):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.ops import embedding_backward

    vocabulary, count, dim = 67, 31, 137
    grad = torch.randn((count, dim), device=device, dtype=torch.float32)

    def graph(gradient):
        if producer == "randperm":
            indices = torch.randperm(vocabulary, device=gradient.device)[:count]
        else:
            indices = torch.arange(vocabulary, device=gradient.device)[1:count + 1]
        indices = indices.clone()
        return embedding_backward(gradient, indices, vocabulary, padding_idx=0, scale_by_freq=scale), indices

    compiled = compile(graph, (grad, ), enabled=True, autotune=False)
    result, indices = compiled(grad)
    expected = _semantic_embedding_reference(grad, indices, vocabulary, padding=0, scale=scale)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    node = _semantic_report_node(compiled, "embedding_backward")
    assert node["selected_optimization"] == "unique_scatter"
    assert "count" not in node["kernel_launches"]
    assert "zero_counts" not in node["kernel_launches"]
    assert node["kernel_launches"] == ["zero_output", "update"]
    assert node["direct_output"] and not node["complete_coverage"]
    assert not _semantic_update_has_atomic(compiled)


@pytest.mark.parametrize("source", ["placeholder", "mutated", "remainder"])
def test_semantic_graph_unproven_or_invalidated_indices_use_atomics(device, source):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.ops import embedding_backward

    count, dim, vocabulary = 19, 137, 23
    grad = torch.randn((count, dim), device=device, dtype=torch.float32)
    supplied_indices = torch.arange(count, device=device).remainder(5)

    def graph(gradient, supplied):
        if source == "placeholder":
            indices = supplied
        else:
            indices = torch.arange(count, device=gradient.device)
            if source == "mutated":
                indices[::2].fill_(0)
            else:
                indices = indices.remainder(5)
        return embedding_backward(gradient, indices, vocabulary, padding_idx=0, scale_by_freq=True), indices

    compiled = compile(graph, (grad, supplied_indices), enabled=True, autotune=False)
    result, indices = compiled(grad, supplied_indices)
    expected = _semantic_embedding_reference(grad, indices, vocabulary, padding=0, scale=True)
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
    node = _semantic_report_node(compiled, "embedding_backward")
    assert node["selected_optimization"] == "atomic_scatter"
    assert "count" in node["kernel_launches"]
    assert _semantic_update_has_atomic(compiled)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [False, True])
def test_semantic_graph_unique_preserves_atomic_special_values(device, dtype, scale):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.ops import embedding_backward

    if dtype == torch.float32:
        bits = [
            0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f800000, 0xff800000,
            0x7fc12345, 0x3f800000
        ]
        storage_dtype = torch.uint32
    elif dtype == torch.float16:
        bits = [0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x7c00, 0xfc00, 0x7e01, 0x3c00]
        storage_dtype = torch.uint16
    else:
        bits = [0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x7f80, 0xff80, 0x7fc1, 0x3f80]
        storage_dtype = torch.uint16
    grad = torch.tensor(bits, dtype=storage_dtype, device=device).view(dtype).reshape(4, 3)

    def graph(gradient):
        indices = torch.arange(gradient.shape[0], device=gradient.device)
        return embedding_backward(gradient, indices, 9, scale_by_freq=scale)

    baseline = compile(graph, (grad, ), enabled=False, autotune=False)
    optimized = compile(graph, (grad, ), enabled=True, autotune=False)
    expected, actual = baseline(grad), optimized(grad)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    ordinary = ~torch.isnan(expected)
    assert torch.equal(actual[ordinary].view(torch.uint8), expected[ordinary].view(torch.uint8))
    assert _semantic_report_node(optimized, "embedding_backward")["selected_optimization"] == "unique_scatter"


def test_semantic_graph_unique_masks_invalid_and_padding_indices(device):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.ops import embedding_backward

    grad = torch.randn((15, 137), device=device, dtype=torch.float32)

    def graph(gradient):
        indices = torch.arange(-2, 13, device=gradient.device)
        return embedding_backward(gradient, indices, 9, padding_idx=3, scale_by_freq=True), indices

    compiled = compile(graph, (grad, ), enabled=True, autotune=False)
    actual, indices = compiled(grad)
    expected = _semantic_embedding_reference(grad, indices, 9, padding=3, scale=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert _semantic_report_node(compiled, "embedding_backward")["selected_optimization"] == "unique_scatter"


def test_semantic_kernel_unique_update_without_zero_fact(device):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.kernels import embedding_update

    old_bits = [0x80000000, 1, 0x80000001, 0x3f800000, 0x7f800000, 0xff800000, 0x7fc12345]
    old = torch.tensor(old_bits, dtype=torch.uint32, device=device).view(torch.float32).reshape(-1, 1)
    grad = torch.tensor([0.0, -0.0, 1.0, 2.0, -float("inf"), float("inf"), 3.0], device=device).reshape(-1, 1)
    indices = torch.arange(old.shape[0], device=device)
    baseline, optimized = old.clone(), old.clone()
    for destination, unique in [(baseline, False), (optimized, True)]:
        embedding_update[(old.shape[0], 1)](
            grad,
            indices,
            None,
            destination,
            old.shape[0],
            1,
            old.shape[0],
            PADDING_IDX=-1,
            SCALE_BY_FREQ=False,
            UNIQUE_INDICES=unique,
            KNOWN_ZERO=False,
            BLOCK_D=32,
        )
    assert torch.equal(torch.isnan(baseline), torch.isnan(optimized))
    ordinary = ~torch.isnan(baseline)
    assert torch.equal(baseline[ordinary].view(torch.uint8), optimized[ordinary].view(torch.uint8))


@pytest.mark.parametrize("input_alias", [False, True])
def test_semantic_graph_rms_infers_immutable_regions_and_preserves_empty_ctas(device, input_alias):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.ops import rms_norm

    x = torch.randn((31, 137), device=device, dtype=torch.float32)
    weights = x[0] if input_alias else torch.randn((137, ), device=device)

    def graph(value, weight):
        return rms_norm(value, weight, num_warps=4, grid_multiplier=1)

    baseline = compile(graph, (x, weights), enabled=False, autotune=False)
    optimized = compile(graph, (x, weights), enabled=True, autotune=False)
    expected = baseline(x, weights)
    actual = optimized(x, weights)
    for got, ref in zip(actual, expected):
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
    inverse_rms = torch.rsqrt(x.double().square().mean(1) + 1e-6)
    torch.testing.assert_close(actual[0], (x.double() * inverse_rms[:, None] * weights.double()).float(), atol=3e-6,
                               rtol=3e-6)
    torch.testing.assert_close(actual[1], inverse_rms.float(), atol=2e-6, rtol=2e-6)
    node = _semantic_report_node(optimized, "rms_norm")
    assert node["selected_optimization"] == "immutable_hoist"
    attr_names = {name for attrs in node["native_attrs"].values() for name, value in attrs}
    assert "tt.experimental_alias_group" in attr_names
    assert "tt.experimental_immutable" not in attr_names
    variants = optimized.kernels[node["name"]]
    original_ir = variants["baseline"].asm["ttgir"]
    inferred_ir = variants["inferred"].asm["ttgir"]
    # The baseline pipelines both X and W. Native no-clobber inference moves
    # the single ordinary W load before the loop, leaving only X in the async
    # pipeline. This checks actual compiler consumption of the alias proof.
    assert inferred_ir.count("tt.load") == 1
    assert inferred_ir.index("tt.load") < inferred_ir.index("scf.for")
    assert original_ir.count("async_copy_global_to_local") > inferred_ir.count("async_copy_global_to_local")


@pytest.mark.parametrize("change", ["shape", "dtype", "stride", "alignment"])
def test_semantic_graph_rejects_changed_input_metadata(device, change):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.ops import rms_norm

    x = torch.randn((16, 64), device=device)
    w = torch.randn((64, ), device=device)
    compiled = compile(lambda value, weight: rms_norm(value, weight), (x, w), enabled=True, autotune=False)
    if change == "shape":
        changed = torch.randn((17, 64), device=device)
    elif change == "dtype":
        changed = x.to(torch.float16)
    elif change == "stride":
        changed = torch.randn((64, 16), device=device).t()
    else:
        changed = torch.randn((16 * 64 + 1, ), device=device)[1:].reshape(16, 64)
        assert changed.stride() == x.stride()
        assert changed.data_ptr() % 16 != x.data_ptr() % 16
    with pytest.raises((ValueError, RuntimeError)):
        compiled(changed, w)


def test_semantic_graph_rejects_unresolved_negative_view(device):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.ops import rms_norm

    x = torch.randn((16, 64), device=device)
    w = torch.randn((64, ), device=device)
    compiled = compile(lambda value, weight: rms_norm(value, weight), (x, w), enabled=True, autotune=False)
    negative = torch._neg_view(x)
    assert negative.is_neg()
    with pytest.raises((ValueError, RuntimeError)):
        compiled(negative, w)


def test_semantic_graph_guards_captured_tensor_metadata(device):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.ops import rms_norm

    x = torch.randn((16, 64), device=device)
    weight = torch.randn((64, ), device=device)

    def graph(value):
        return rms_norm(value, weight)

    compiled = compile(graph, (x, ), enabled=True, autotune=False)
    assert any(node.op == "get_attr" for node in compiled.graph_module.graph.nodes)
    weight.resize_(32)
    with pytest.raises((ValueError, RuntimeError)):
        compiled(x)


def test_semantic_graph_rms_rejects_induction_overflow(device):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.ops import rms_norm

    x = torch.randn((1, 16), device=device)
    weight = torch.randn((16, ), device=device)

    def graph(value, scale):
        return rms_norm(value, scale, grid_multiplier=2**31)

    with pytest.raises(ValueError, match="induction.*int32"):
        compiled = compile(graph, (x, weight), enabled=True, autotune=False)
        compiled(x, weight)


@pytest.mark.parametrize("shape,dtype,bounds,unique,lower,upper,expected", [
    ((7, ), torch.int64, (0, 6), True, 0, 7, True),
    ((2, 3), torch.int32, (-3, 2), True, -3, 3, True),
    ((6, ), torch.int64, (0, 6), True, 0, 7, False),
    ((7, ), torch.int64, (-1, 6), True, 0, 7, False),
    ((7, ), torch.int64, (0, 7), True, 0, 7, False),
    ((7, ), torch.int64, (0, 6), False, 0, 7, False),
    ((7, ), torch.float32, (0, 6), True, 0, 7, False),
    ((7, ), torch.int64, None, True, 0, 7, False),
    (None, torch.int64, (0, 6), True, 0, 7, False),
    ((0, ), torch.int64, None, False, 3, 3, True),
    ((0, ), torch.int64, None, False, 3, 2, False),
])
def test_semantic_coverage_needs_global_injectivity_bounds_and_cardinality(shape, dtype, bounds, unique, lower, upper,
                                                                           expected):
    from triton.experimental.semantic_contracts import IntegerRange, TensorFacts
    facts = TensorFacts(shape, dtype, frozenset(), (), unique_global=unique, unique_dims=frozenset({0}),
                        integer_range=None if bounds is None else IntegerRange(*bounds))
    assert facts.covers_integer_interval(lower, upper) is expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("complete", [False, True])
def test_semantic_unique_direct_write_covers_poisoned_output_and_padding(device, dtype, complete):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    from triton.experimental.semantic_contracts.kernels import embedding_unique_write
    from triton.experimental.semantic_contracts.ops import embedding_backward

    count, dim = 19, 137
    vocabulary = count if complete else count + 7
    grad = torch.randn((count, dim), device=device, dtype=dtype)
    grad[0].fill_(float("nan"))

    def graph(gradient):
        indices = torch.arange(count, device=gradient.device)
        return embedding_backward(gradient, indices, vocabulary, padding_idx=0, scale_by_freq=True)

    baseline = compile(graph, (grad, ), enabled=False, autotune=False)
    optimized = compile(graph, (grad, ), enabled=True, autotune=False)
    expected, actual = baseline(grad), optimized(grad)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    node = _semantic_report_node(optimized, "embedding_backward")
    assert node["complete_coverage"] is complete
    assert node["direct_output"]
    assert node["kernel_launches"] == (["update"] if complete else ["zero_output", "update"])
    assert not _semantic_update_has_atomic(optimized)
    assert torch.equal(actual[0].view(torch.uint8), torch.zeros_like(actual[0]).view(torch.uint8))
    if complete:
        # A complete overwrite must work on arbitrary physical contents, including
        # NaNs in the padding destination. Surround the destination with canaries.
        storage = torch.full((vocabulary * dim + 32, ), 11, device=device, dtype=dtype)
        destination = storage[16:-16].view(vocabulary, dim)
        indices = torch.arange(count, device=device)
        for poison in (float("nan"), -9.0):
            destination.fill_(poison)
            embedding_unique_write[(count, 2)](grad, indices, destination, count, dim, vocabulary, 0, 128, True, True,
                                               num_warps=4)
            assert torch.equal(destination.view(torch.uint8), expected.view(torch.uint8))
            assert torch.all(storage[:16] == 11) and torch.all(storage[-16:] == 11)


def _semantic_aten_edge_gradient(dtype, count, dim, device):
    if dtype == torch.float32:
        values = [
            0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f800000, 0xff800000,
            0x7fc12345, 0x3f800000
        ]
        storage_dtype = torch.uint32
    elif dtype == torch.float16:
        values = [0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x7c00, 0xfc00, 0x7e01, 0x3c00]
        storage_dtype = torch.uint16
    else:
        values = [0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x7f80, 0xff80, 0x7fc1, 0x3f80]
        storage_dtype = torch.uint16
    pattern = torch.tensor(values, device=device, dtype=storage_dtype).view(dtype)
    grad = pattern.repeat((count * dim + len(values) - 1) // len(values))[:count * dim].reshape(count, dim)
    grad[0].fill_(float("nan"))
    return grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("count", [3072, 3073])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_semantic_aten_embedding_preserves_dispatch_boundary_edge_values(device, dtype, count, scale, complete):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import _ATEN_EMBEDDING_REVISION, compile
    if torch.version.git_version != _ATEN_EMBEDDING_REVISION:
        pytest.skip("This exact CUDA embedding implementation has not been audited")
    dim = 13
    vocabulary = count if complete else count + 7
    grad = _semantic_aten_edge_gradient(dtype, count, dim, device)

    def graph(gradient):
        indices = torch.arange(count, device=gradient.device).clone().to(torch.int32)
        return torch.ops.aten.embedding_dense_backward.default(gradient, indices, vocabulary, 0, scale)

    compiled = compile(graph, (grad, ), enabled=True, autotune=False)
    actual = compiled(grad)
    expected = graph(grad)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    ordinary = ~torch.isnan(expected)
    assert torch.equal(actual[ordinary].view(torch.uint8), expected[ordinary].view(torch.uint8))
    assert torch.equal(actual[0].view(torch.uint8), torch.zeros_like(actual[0]).view(torch.uint8))
    node = _semantic_report_node(compiled, "aten.embedding_dense_backward")
    assert node["selected_optimization"] == "unique_direct_write"
    assert node["complete_coverage"] is complete
    assert node["kernel_launches"] == (["update"] if complete else ["zero_output", "update"])
    ptx = compiled.kernels[node["name"]]["update"].asm["ptx"]
    assert "add.rn.f32" in ptx and "add.rn.ftz.f32" not in ptx


@pytest.mark.parametrize("source", ["placeholder", "mutated", "wide_bounds", "noncontiguous", "float64", "revision"])
def test_semantic_aten_embedding_unsupported_cases_keep_framework_fallback(device, source, monkeypatch):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import compile
    count, dim, vocabulary = 19, 13, 23
    if source == "float64":
        grad = torch.randn((count, dim), device=device, dtype=torch.float64)
    elif source == "noncontiguous":
        grad = torch.randn((dim, count), device=device).t()
    else:
        grad = torch.randn((count, dim), device=device)
    supplied = torch.arange(count, device=device).remainder(5)
    if source == "revision":
        monkeypatch.setattr(torch.version, "git_version", "unreviewed-embedding-implementation")

    def graph(gradient, external):
        if source == "placeholder":
            indices = external
        elif source == "wide_bounds":
            # Actual values are valid and unique; retained range is conservatively
            # [-1, count-1], so the adapter cannot prove validity at this consumer.
            indices = torch.arange(-1, count, device=gradient.device)[1:]
        else:
            indices = torch.arange(count, device=gradient.device)
            if source == "mutated":
                indices[::2].fill_(0)
        return torch.ops.aten.embedding_dense_backward.default(gradient, indices, vocabulary, 0, True)

    compiled = compile(graph, (grad, supplied), enabled=True, autotune=False)
    actual, expected = compiled(grad, supplied), graph(grad, supplied)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    node = _semantic_report_node(compiled, "aten.embedding_dense_backward")
    assert node["selected_optimization"] == "aten_fallback"
    assert node["kernel_launches"] == ["original_aten_operator"]
    assert compiled.kernels[node["name"]] == {}


def test_semantic_aten_embedding_accepts_schema_keyword_arguments(device):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import _ATEN_EMBEDDING_REVISION, compile
    if torch.version.git_version != _ATEN_EMBEDDING_REVISION:
        pytest.skip("This exact CUDA embedding implementation has not been audited")
    grad = torch.randn((17, 13), device=device)

    def graph(gradient):
        return torch.ops.aten.embedding_dense_backward.default(grad_output=gradient,
                                                               indices=torch.arange(17, device=gradient.device),
                                                               num_weights=17, padding_idx=-1, scale_grad_by_freq=True)

    compiled = compile(graph, (grad, ), enabled=True, autotune=False)
    torch.testing.assert_close(compiled(grad), graph(grad), atol=0, rtol=0)
    assert _semantic_report_node(compiled,
                                 "aten.embedding_dense_backward")["selected_optimization"] == "unique_direct_write"


def test_semantic_aten_embedding_consumes_public_autograd_graph(device):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.backend import _ATEN_EMBEDDING_REVISION, compile
    if torch.version.git_version != _ATEN_EMBEDDING_REVISION:
        pytest.skip("This exact CUDA embedding implementation has not been audited")
    grad = torch.randn((31, 13), device=device, dtype=torch.bfloat16)

    def graph(gradient):
        indices = torch.randperm(67, device=gradient.device)[:gradient.shape[0]]
        weights = torch.zeros((67, gradient.shape[1]), device=gradient.device, dtype=gradient.dtype)
        return torch.func.grad(lambda w: (torch.nn.functional.embedding(indices, w, padding_idx=0, scale_grad_by_freq=
                                                                        True) * gradient).sum())(weights)

    compiled = compile(graph, (grad, ), enabled=True, autotune=False)
    for seed in (42, 43):
        torch.manual_seed(seed)
        actual = compiled(grad)
        torch.manual_seed(seed)
        expected = graph(grad)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert _semantic_report_node(compiled,
                                 "aten.embedding_dense_backward")["selected_optimization"] == "unique_direct_write"


@triton.jit
def _native_indexed_tl(Grad, Indices, Out, V, Padding, BLOCK_D: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    index = tl.load(Indices + row)
    value = tl.load(Grad + row * D + cols, mask=cols < D, other=0).to(tl.float32)
    mask = (cols < D) & (index >= 0) & (index < V) & (index != Padding)
    tl.atomic_add(Out + index * D + cols, value, mask=mask)


@gluon.jit
def _native_indexed_gluon(Grad, Indices, Out, V, Padding, BLOCK_D: gl.constexpr, D: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.program_id(1) * BLOCK_D + gl.arange(0, BLOCK_D, layout=gl.BlockedLayout([1], [32], [4], [0]))
    index = gl.load(Indices + row)
    value = gl.load(Grad + row * D + cols, mask=cols < D, other=0).to(gl.float32)
    mask = (cols < D) & (index >= 0) & (index < V) & (index != Padding)
    gl.atomic_add(Out + index * D + cols, value, mask=mask)


def _native_indexed_has_atomic(compiled):
    import re
    return re.search(r"\b(?:atom|red)\.", compiled.asm["ptx"]) is not None


def _native_indexed_same_values(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    ordinary = ~torch.isnan(expected)
    assert torch.equal(actual[ordinary].contiguous().view(torch.uint8),
                       expected[ordinary].contiguous().view(torch.uint8))


@pytest.mark.parametrize("frontend", ["triton", "gluon"])
@pytest.mark.parametrize("initial_zero", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_native_indexed_update_preserves_atomic_values_and_masks(device, frontend, initial_zero, index_dtype):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_indexed_update

    count, dim, vocabulary, block, guard = 19, 137, 37, 128, 16
    indices = torch.cat((torch.zeros(1, device=device,
                                     dtype=index_dtype), torch.arange(count - 1, 0, -1, device=device,
                                                                      dtype=index_dtype)))
    bits = torch.tensor([
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f800000, 0xff800000,
        0x7fc12345, 0x3f800000
    ], device=device, dtype=torch.uint32)
    grad = bits.repeat(triton.cdiv(count * dim, bits.numel()))[:count * dim].view(torch.float32).view(count, dim)
    grad[0].fill_(float("nan"))
    storage = torch.full((vocabulary * dim + guard * 2, ), 17.0, device=device)
    destination = storage[guard:-guard].view(vocabulary, dim)
    if initial_zero:
        initial = torch.zeros_like(destination)
    else:
        initial = bits.view(torch.float32).roll(3).repeat(triton.cdiv(vocabulary * dim,
                                                                      bits.numel()))[:vocabulary * dim].reshape(
                                                                          vocabulary, dim)
    kernel = _native_indexed_tl if frontend == "triton" else _native_indexed_gluon
    arguments = (grad, indices, destination, vocabulary, 0, block, dim)
    grid = (count, triton.cdiv(dim, block), 1)
    plan = compile_indexed_update(kernel, arguments, grid=grid, index_arg="Indices", output_arg="Out",
                                  index_count=count, index_bound=vocabulary, initial_zero=initial_zero,
                                  options={"num_warps": 4})
    destination.copy_(initial)
    plan.baseline[plan.grid](*arguments)
    expected = destination.clone()
    destination.copy_(initial)
    plan(*arguments)
    _native_indexed_same_values(destination, expected)
    assert torch.all(storage[:guard] == 17) and torch.all(storage[-guard:] == 17)
    _native_indexed_same_values(destination[0], initial[0])
    _native_indexed_same_values(destination[count:], initial[count:])
    assert _native_indexed_has_atomic(plan.baseline)
    assert not _native_indexed_has_atomic(plan.optimized)
    assert "add.rn.ftz.f32" in plan.optimized.asm["ptx"]
    assert plan.baseline.src.fn is plan.optimized.src.fn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_indexed_update_accepts_low_precision_inputs(device, dtype):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_indexed_update

    count, dim, vocabulary = 31, 65, 67
    grad = torch.randn((count, dim), device=device, dtype=dtype)
    indices = torch.randperm(vocabulary, device=device)[:count]
    out = torch.zeros((vocabulary, dim), device=device)
    args = (grad, indices, out, vocabulary, -1, 128, dim)
    plan = compile_indexed_update(_native_indexed_tl, args, grid=(count, 1), index_arg=1, output_arg=2,
                                  index_count=count, index_bound=vocabulary, initial_zero=True)
    plan.baseline[plan.grid](*args)
    expected = out.clone()
    out.zero_()
    plan(*args)
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    assert not _native_indexed_has_atomic(plan.optimized)


def test_native_indexed_update_without_contract_retains_duplicate_atomics(device):
    _semantic_require_cuda(device)
    count, dim, vocabulary = 31, 65, 7
    indices = torch.arange(count, device=device) % vocabulary
    grad = torch.ones((count, dim), device=device)
    out = torch.zeros((vocabulary, dim), device=device)
    compiled = _native_indexed_tl[(count, 1)](grad, indices, out, vocabulary, -1, 128, dim,
                                              enable_indexed_update_specialization=True)
    expected = torch.bincount(indices, minlength=vocabulary).float()[:, None].expand_as(out)
    assert torch.equal(out, expected)
    assert _native_indexed_has_atomic(compiled)


@pytest.mark.parametrize("change", ["grid", "scalar", "index_alias_mutation", "new_index_object", "output_alias"])
def test_native_indexed_update_launch_guard_rejects_stale_contract(device, change):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_indexed_update

    count, dim = 19, 65
    grad = torch.randn((count, dim), device=device)
    indices = torch.arange(count, device=device)
    out = torch.zeros_like(grad)
    args = (grad, indices, out, count, -1, 128, dim)
    plan = compile_indexed_update(_native_indexed_tl, args, grid=(count, 1), index_arg=1, output_arg=2,
                                  index_count=count, index_bound=count, initial_zero=False)
    changed = list(args)
    kwargs = {}
    if change == "grid":
        kwargs["grid"] = (count, 2)
    elif change == "scalar":
        changed[3] = count + 1
    elif change == "index_alias_mutation":
        indices.view(-1)[1].fill_(0)
    elif change == "new_index_object":
        changed[1] = indices.clone()
    else:
        changed[2] = grad
    with pytest.raises((ValueError, RuntimeError)):
        plan(*changed, **kwargs)


def test_native_indexed_update_compile_rejects_overlapping_output(device):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_indexed_update

    grad = torch.randn((19, 65), device=device)
    indices = torch.arange(19, device=device)
    args = (grad, indices, grad, 19, -1, 128, 65)
    with pytest.raises((ValueError, RuntimeError)):
        compile_indexed_update(_native_indexed_tl, args, grid=(19, 1), index_arg=1, output_arg=2, index_count=19,
                               index_bound=19, initial_zero=False)


@triton.jit
def _native_private_atomic_tl(Grad, Indices, Out, V, Padding, BLOCK_D: tl.constexpr, D: tl.constexpr,
                              SEM: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    index = tl.load(Indices + row)
    value = tl.load(Grad + row * D + cols, mask=cols < D, other=0).to(tl.float32)
    mask = (cols < D) & (index >= 0) & (index < V) & (index != Padding)
    tl.atomic_add(Out + index * D + cols, value, mask=mask, sem=SEM)


@gluon.jit
def _native_private_atomic_gluon(Grad, Indices, Out, V, Padding, BLOCK_D: gl.constexpr, D: gl.constexpr,
                                 SEM: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.program_id(1) * BLOCK_D + gl.arange(0, BLOCK_D, layout=gl.BlockedLayout([1], [32], [4], [0]))
    index = gl.load(Indices + row)
    value = gl.load(Grad + row * D + cols, mask=cols < D, other=0).to(gl.float32)
    mask = (cols < D) & (index >= 0) & (index < V) & (index != Padding)
    gl.atomic_add(Out + index * D + cols, value, mask=mask, sem=SEM)


@pytest.mark.parametrize("frontend", ["triton", "gluon"])
@pytest.mark.parametrize("sem", ["acq_rel", "acquire", "release"])
def test_native_private_atomic_preserves_duplicate_reductions(device, frontend, sem):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_atomic

    count, dim, vocabulary = 257, 137, 13
    indices = torch.arange(count, device=device) % vocabulary
    grad = torch.ones((count, dim), device=device)
    out = torch.full((vocabulary, dim), 5.0, device=device)
    kernel = _native_private_atomic_tl if frontend == "triton" else _native_private_atomic_gluon
    args = (grad, indices, out, vocabulary, 0, 128, dim, sem)
    plan = compile_private_atomic(kernel, args, grid=(count, triton.cdiv(dim, 128)), output_arg="Out")
    expected = torch.bincount(indices, minlength=vocabulary).float()[:, None].expand_as(out) + 5
    expected[0].fill_(5)
    plan.baseline[plan.grid](*args)
    assert torch.equal(out, expected)
    out.fill_(5)
    plan(*args)
    assert torch.equal(out, expected)
    report = plan.report()
    assert report["transformed"]
    assert report["variants"]["baseline"]["bar_sync_sites"] > 0
    assert report["variants"]["optimized"]["bar_sync_sites"] == 0
    assert _native_indexed_has_atomic(plan.optimized)
    assert "red.global.gpu.relaxed" in plan.optimized.asm["ptx"]
    names = {name for entries in report["native_attributes"].values() for name, value in entries}
    assert "tt.experimental_unique_indices" not in names
    assert "tt.experimental_index_bound" not in names
    assert "tt.experimental_initial_zero" not in names
    assert plan.baseline.src.fn is plan.optimized.src.fn


@pytest.mark.parametrize("frontend", ["triton", "gluon"])
@pytest.mark.parametrize("initial_zero", [False, True])
def test_native_private_atomic_preserves_edge_values_and_masks(device, frontend, initial_zero):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_atomic

    count, dim, vocabulary, guard = 19, 137, 37, 16
    indices = torch.arange(count - 1, -1, -1, device=device)
    bits = torch.tensor([
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f800000, 0xff800000,
        0x7fc12345, 0x3f800000
    ], device=device, dtype=torch.uint32).view(torch.float32)
    grad = bits.repeat(triton.cdiv(count * dim, bits.numel()))[:count * dim].reshape(count, dim)
    grad[-1].fill_(float("nan"))
    storage = torch.full((vocabulary * dim + guard * 2, ), 17.0, device=device)
    out = storage[guard:-guard].view(vocabulary, dim)
    initial = torch.zeros_like(out) if initial_zero else bits.roll(3).repeat(triton.cdiv(
        vocabulary * dim, bits.numel()))[:vocabulary * dim].reshape(vocabulary, dim)
    kernel = _native_private_atomic_tl if frontend == "triton" else _native_private_atomic_gluon
    args = (grad, indices, out, vocabulary, 0, 128, dim, "acq_rel")
    plan = compile_private_atomic(kernel, args, grid=(count, triton.cdiv(dim, 128)), output_arg=2)
    out.copy_(initial)
    plan.baseline[plan.grid](*args)
    expected = out.clone()
    out.copy_(initial)
    plan(*args)
    _native_indexed_same_values(out, expected)
    _native_indexed_same_values(out[0], initial[0])
    _native_indexed_same_values(out[count:], initial[count:])
    assert torch.all(storage[:guard] == 17) and torch.all(storage[-guard:] == 17)
    assert _native_indexed_has_atomic(plan.optimized)
    assert plan.report()["transformed"]


@pytest.mark.parametrize("frontend", ["triton", "gluon"])
def test_native_private_atomic_already_relaxed_is_unchanged(device, frontend):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_atomic

    count, dim, vocabulary = 31, 65, 7
    indices = torch.arange(count, device=device) % vocabulary
    grad = torch.ones((count, dim), device=device)
    out = torch.zeros((vocabulary, dim), device=device)
    kernel = _native_private_atomic_tl if frontend == "triton" else _native_private_atomic_gluon
    args = (grad, indices, out, vocabulary, -1, 128, dim, "relaxed")
    plan = compile_private_atomic(kernel, args, grid=(count, 1), output_arg=2)
    plan(*args)
    expected = torch.bincount(indices, minlength=vocabulary).float()[:, None].expand_as(out)
    assert torch.equal(out, expected)
    assert not plan.report()["transformed"]
    assert "red.global.gpu.relaxed" in plan.optimized.asm["ptx"]


@pytest.mark.parametrize("frontend", ["triton", "gluon"])
def test_native_private_atomic_allows_input_mutation_and_replacement(device, frontend):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_atomic

    count, dim, vocabulary = 31, 65, 11
    indices = torch.arange(count, device=device) % vocabulary
    grad = torch.ones((count, dim), device=device)
    out = torch.zeros((vocabulary, dim), device=device)
    kernel = _native_private_atomic_tl if frontend == "triton" else _native_private_atomic_gluon
    args = (grad, indices, out, vocabulary, -1, 128, dim, "acq_rel")
    plan = compile_private_atomic(kernel, args, grid=(count, 1), output_arg=2)
    indices.fill_(3)
    grad.fill_(2)
    out.fill_(7)
    plan(*args)
    expected = torch.full_like(out, 7)
    expected[3] += 2 * count
    assert torch.equal(out, expected)
    replacement_indices = torch.arange(count, device=device) % 5
    replacement_grad = torch.full_like(grad, 3)
    replacement_out = torch.full_like(out, 9)
    plan(replacement_grad, replacement_indices, replacement_out, *args[3:])
    expected = 9 + 3 * torch.bincount(replacement_indices, minlength=vocabulary).float()[:, None].expand_as(out)
    assert torch.equal(replacement_out, expected)


@pytest.mark.parametrize("change", ["grid", "scalar", "output_alias", "metadata"])
def test_native_private_atomic_launch_guard_rejects_invalid_binding(device, change):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_atomic

    count, dim = 31, 65
    indices = torch.arange(count, device=device) % 7
    grad = torch.ones((count, dim), device=device)
    out = torch.zeros_like(grad)
    args = (grad, indices, out, count, -1, 128, dim, "acq_rel")
    plan = compile_private_atomic(_native_private_atomic_tl, args, grid=(count, 1), output_arg=2)
    changed, kwargs = list(args), {}
    if change == "grid":
        kwargs["grid"] = (count, 2)
    elif change == "scalar":
        changed[3] = count + 1
    elif change == "output_alias":
        changed[2] = grad
    else:
        changed[1] = indices.to(torch.int32)
    with pytest.raises((ValueError, RuntimeError)):
        plan(*changed, **kwargs)


@pytest.mark.parametrize("invalid", ["output_alias", "multiple_ctas", "indexed_removal"])
def test_native_private_atomic_compile_rejects_invalid_contract(device, invalid):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_atomic

    count, dim = 31, 65
    indices = torch.arange(count, device=device) % 7
    grad = torch.ones((count, dim), device=device)
    out = grad if invalid == "output_alias" else torch.zeros_like(grad)
    args = (grad, indices, out, count, -1, 128, dim, "acq_rel")
    options = {"num_ctas": 2} if invalid == "multiple_ctas" else {
        "enable_indexed_update_specialization": invalid == "indexed_removal"
    }
    with pytest.raises((ValueError, RuntimeError)):
        compile_private_atomic(_native_private_atomic_tl, args, grid=(count, 1), output_arg=2, options=options)


def test_native_private_atomic_without_contract_keeps_ordering(device):
    _semantic_require_cuda(device)
    count, dim, vocabulary = 31, 65, 7
    indices = torch.arange(count, device=device) % vocabulary
    grad = torch.ones((count, dim), device=device)
    out = torch.zeros((vocabulary, dim), device=device)
    compiled = _native_private_atomic_tl[(count, 1)](grad, indices, out, vocabulary, -1, 128, dim, "acq_rel",
                                                     enable_private_atomic_relaxation=True)
    expected = torch.bincount(indices, minlength=vocabulary).float()[:, None].expand_as(out)
    assert torch.equal(out, expected)
    assert "atom.global.gpu.acq_rel" in compiled.asm["ptx"]
    assert "bar.sync" in compiled.asm["ptx"]


@triton.jit
def _native_private_integer_loop_tl(Values, Indices, Bounds, Out, N: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr,
                                    NESTED: tl.constexpr, SEM0: tl.constexpr, SEM1: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    index = tl.load(Indices + offsets, offsets < N, other=-1)
    value = tl.load(Values + offsets, offsets < N, other=0)
    valid = (offsets < N) & (index >= 0) & (index < V)
    stop = tl.load(Bounds)
    step = tl.load(Bounds + 1)
    for i in range(0, stop, step):
        inner_stop = i + 1 if NESTED else 1
        for j in range(inner_stop):
            tl.atomic_add(Out + index, value + i + j, mask=valid, sem=SEM0)
            tl.atomic_add(Out + (index + 1) % V, value * 3 - i, mask=valid, sem=SEM1)


@gluon.jit
def _native_private_integer_loop_gluon(Values, Indices, Bounds, Out, N: gl.constexpr, V: gl.constexpr,
                                       BLOCK: gl.constexpr, NESTED: gl.constexpr, SEM0: gl.constexpr,
                                       SEM1: gl.constexpr):
    offsets = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0]))
    index = gl.load(Indices + offsets, offsets < N, other=-1)
    value = gl.load(Values + offsets, offsets < N, other=0)
    valid = (offsets < N) & (index >= 0) & (index < V)
    stop = gl.load(Bounds)
    step = gl.load(Bounds + 1)
    for i in range(0, stop, step):
        inner_stop = i + 1 if NESTED else 1
        for j in range(inner_stop):
            gl.atomic_add(Out + index, value + i + j, mask=valid, sem=SEM0)
            gl.atomic_add(Out + (index + 1) % V, value * 3 - i, mask=valid, sem=SEM1)


@pytest.mark.parametrize("frontend", ["triton", "gluon"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("orders", [("acq_rel", "release"), ("acquire", "relaxed"), ("relaxed", "relaxed")])
def test_native_private_integer_loops_preserve_modular_sums(device, frontend, nested, orders):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_atomic

    count, vocabulary, block, guard = 137, 7, 128, 16
    values_cpu = (torch.arange(count, dtype=torch.int64) * 100000003 + 2147483600).to(torch.int32)
    indices_cpu = torch.arange(count) % vocabulary
    indices_cpu[:2] = torch.tensor([-1, vocabulary])
    values, indices = values_cpu.to(device), indices_cpu.to(device)
    bounds = torch.tensor([4, 1], device=device, dtype=torch.int32)
    storage = torch.full((vocabulary + 2 * guard, ), 17, device=device, dtype=torch.int32)
    out = storage[guard:-guard]
    initial_cpu = torch.arange(vocabulary, dtype=torch.int64) + 2147483640
    initial = initial_cpu.to(device=device, dtype=torch.int32)
    kernel = _native_private_integer_loop_tl if frontend == "triton" else _native_private_integer_loop_gluon
    args = (values, indices, bounds, out, count, vocabulary, block, nested, *orders)
    plan = compile_private_atomic(kernel, args, grid=(triton.cdiv(count, block), ), output_arg="Out")
    report = plan.report()
    assert report["transformed"] == (orders != ("relaxed", "relaxed"))
    assert "scf.for" in plan.optimized.asm["ttgir" if frontend == "gluon" else "ttir"]
    assert _native_indexed_has_atomic(plan.baseline) and _native_indexed_has_atomic(plan.optimized)
    assert report["variants"]["optimized"]["bar_sync_sites"] == 0
    assert all("relaxed" in op for op in report["variants"]["optimized"]["atomic_instructions"])
    assert plan.baseline.src.fn is plan.optimized.src.fn

    for stop, step in ((0, 1), (1, 1), (4, 1), (5, 2)):
        # Bounds are loaded at runtime, so these launches also cover zero trips
        # and dynamic steps without compiling a different specialization.
        bounds.copy_(torch.tensor([stop, step], device=device, dtype=torch.int32))
        expected64 = initial_cpu.clone()
        valid = (indices_cpu >= 0) & (indices_cpu < vocabulary)
        index = indices_cpu[valid]
        value = values_cpu[valid].to(torch.int64)
        for i in range(0, stop, step):
            for j in range(i + 1 if nested else 1):
                expected64.index_add_(0, index, value + i + j)
                expected64.index_add_(0, (index + 1) % vocabulary, value * 3 - i)
        if stop:
            assert torch.any((expected64 < -(2**31)) | (expected64 >= 2**31))
        expected = (((expected64 + 2**31) % 2**32) - 2**31).to(device=device, dtype=torch.int32)
        for variant in (plan.baseline, plan.optimized):
            out.copy_(initial)
            variant[plan.grid](*args)
            assert torch.equal(out, expected)
            assert torch.all(storage[:guard] == 17) and torch.all(storage[-guard:] == 17)


@triton.jit
def _native_immutable_loop_tl(X, Gamma, Y, M, N: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        x = tl.load(X + row * N + col, col < N, other=0)
        gamma = tl.load(Gamma + col, col < N, other=0)
        tl.store(Y + row * N + col, x * gamma, col < N)


@gluon.jit
def _native_immutable_loop_gluon(X, Gamma, Y, M, N: gl.constexpr, BLOCK: gl.constexpr):
    col = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for row in range(gl.program_id(0), M, gl.num_programs(0)):
        x = gl.load(X + row * N + col, col < N, other=0)
        gamma = gl.load(Gamma + col, col < N, other=0)
        gl.store(Y + row * N + col, x * gamma, col < N)


@pytest.mark.parametrize("frontend", ["triton", "gluon"])
@pytest.mark.parametrize("grid", [1, 17])
def test_native_immutable_licm_preserves_outputs_and_allows_new_inputs(device, frontend, grid):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_immutable_licm

    m, n = 11, 65
    x = torch.arange(m * n, device=device, dtype=torch.float32).reshape(m, n)
    gamma = torch.full((n, ), 2, device=device)
    y = torch.full_like(x, float("nan"))
    kernel = _native_immutable_loop_tl if frontend == "triton" else _native_immutable_loop_gluon
    args = (x, gamma, y, m, n, 128)
    plan = compile_immutable_licm(kernel, args, grid=(grid, ), immutable_args=("Gamma", ))
    assert plan.baseline.src.fn is plan.optimized.src.fn
    assert not any(
        name.startswith("tt.experimental_") for attrs in plan.baseline.src.attrs.values() for name, _ in attrs)
    assert ("tt.experimental_immutable", 1) in plan.optimized.src.attrs[(1, )]
    for variant in (plan.baseline, plan.optimized, plan.selected):
        y.fill_(float("nan"))
        variant[plan.grid](*args)
        assert torch.equal(y, x * gamma)
    # The contract lasts through one invocation, not for the lifetime of a plan.
    gamma.fill_(3)
    plan(*args)
    assert torch.equal(y, x * gamma)
    replacement = (torch.full_like(x, 7), torch.full_like(gamma, 5), torch.empty_like(y), *args[3:])
    plan(*replacement)
    assert torch.equal(replacement[2], torch.full_like(y, 35))
    report = plan.report()
    assert report["variants"]["optimized"]["blocks_per_sm"] > 0
    assert report["launched_ctas"] == grid
    assert not plan.baseline.metadata.enable_private_atomic_relaxation
    assert not plan.optimized.metadata.enable_indexed_update_specialization


@pytest.mark.parametrize("change", ["grid", "scalar", "metadata", "direct_alias", "disjoint_views_one_storage"])
def test_native_immutable_licm_launch_rechecks_binding_and_allocations(device, change):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_immutable_licm

    m, n = 11, 64
    x = torch.ones((m, n), device=device)
    gamma = torch.ones(n, device=device)
    y = torch.empty_like(x)
    args = (x, gamma, y, m, n, 64)
    plan = compile_immutable_licm(_native_immutable_loop_tl, args, grid=(2, ), immutable_args=(1, ))
    changed, kwargs = list(args), {}
    if change == "grid":
        kwargs["grid"] = (3, )
    elif change == "scalar":
        changed[3] = m + 1
    elif change == "metadata":
        changed[1] = gamma.to(torch.float64)
    elif change == "direct_alias":
        changed[1] = x[0]
    else:
        backing = torch.empty(m * n + n, device=device)
        changed[1], changed[2] = backing[:n], backing[n:].reshape_as(y)
    with pytest.raises(ValueError):
        plan(*changed, **kwargs)


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "scalar", "alias", "multiple_ctas"])
def test_native_immutable_licm_compile_rejects_invalid_contract(device, invalid):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_immutable_licm

    m, n = 11, 64
    x = torch.ones((m, n), device=device)
    gamma = x[0] if invalid == "alias" else torch.ones(n, device=device)
    args = (x, gamma, torch.empty_like(x), m, n, 64)
    immutable = {"empty": (), "duplicate": (1, "Gamma"), "scalar": (3, )}.get(invalid, (1, ))
    options = {"num_ctas": 2} if invalid == "multiple_ctas" else {}
    with pytest.raises((ValueError, TypeError)):
        compile_immutable_licm(_native_immutable_loop_tl, args, grid=(2, ), immutable_args=immutable, options=options)


@triton.jit
def _native_private_reduction_protocol(X, Y, Sum, Square, Lock, N, G: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    values = tl.load(X + row * N + cols, mask, other=0)
    tl.store(Y + row * N + cols, values * 2, mask)
    first, second = values, values * values
    group = row % G
    mutex = Lock + group
    count = mutex + G
    sum_ptr = Sum + group * N + cols
    square_ptr = Square + group * N + cols
    while tl.atomic_cas(mutex, 0, 1) == 1:
        pass
    if tl.load(count) == 0:
        tl.atomic_xchg(count, 1)
    else:
        first += tl.load(sum_ptr, mask)
        second += tl.load(square_ptr, mask)
    tl.store(sum_ptr, first, mask)
    tl.store(square_ptr, second, mask)
    tl.debug_barrier()
    tl.atomic_xchg(mutex, 0)


def _private_reduction_args(device, rows, columns, groups):
    x = torch.randint(-4, 5, (rows, columns), device=device).float()
    y = torch.full_like(x, 17)
    sums = torch.full((groups, columns), 17.0, device=device)
    squares = torch.full_like(sums, 19)
    locks = torch.zeros(2 * groups, device=device, dtype=torch.int32)
    return x, y, sums, squares, locks, columns, groups, triton.next_power_of_2(columns)


@pytest.mark.parametrize("shape", [(17, 13, 4), (5, 65, 2), (3, 7, 8)])
def test_native_private_reduction_ownership_preserves_complete_state(device, shape):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_reduction_ownership

    rows, columns, groups = shape
    args = _private_reduction_args(device, rows, columns, groups)
    plan = compile_private_reduction_ownership(_native_private_reduction_protocol, args, grid=(rows, ),
                                               private_args=("Y", "Sum", "Square", "Lock"),
                                               zero_initialized_args=("Lock", ))
    assert plan.report()["transformed"]
    assert plan.baseline.src.fn is plan.optimized.src.fn
    assert plan.baseline.src.signature == plan.optimized.src.signature
    assert plan.baseline.src.constants == plan.optimized.src.constants
    assert _native_indexed_has_atomic(plan.baseline)
    assert not _native_indexed_has_atomic(plan.optimized)
    for shift in (0, 1):
        args[0].add_(shift)
        expected_sum = torch.full_like(args[2], 17)
        expected_square = torch.full_like(args[3], 19)
        for group in range(min(rows, groups)):
            values = args[0][group::groups]
            expected_sum[group] = values.sum(0)
            expected_square[group] = (values * values).sum(0)
        expected_count = torch.zeros_like(args[4])
        expected_count[groups:groups + min(rows, groups)] = 1
        for run in (lambda: plan.baseline[plan.grid](*args), lambda: plan(*args)):
            args[1].fill_(17)
            args[2].fill_(17)
            args[3].fill_(19)
            args[4].zero_()
            run()
            assert torch.equal(args[1], args[0] * 2)
            assert torch.equal(args[2], expected_sum)
            assert torch.equal(args[3], expected_square)
            assert torch.equal(args[4], expected_count)


def test_native_private_reduction_ownership_preserves_first_assignment_bits(device):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_reduction_ownership

    args = _private_reduction_args(device, 4, 13, 4)
    bits = torch.tensor([
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f800000, 0xff800000,
        0x3f800000, 0xbf800000, 0x7fc12345
    ], device=device, dtype=torch.uint32).view(torch.float32)
    args[0].copy_(bits.expand_as(args[0]))
    plan = compile_private_reduction_ownership(_native_private_reduction_protocol, args, grid=(4, ),
                                               private_args=(1, 2, 3, 4), zero_initialized_args=(4, ))
    assert plan.report()["transformed"]
    plan(*args)
    assert torch.equal(args[2].view(torch.uint32), args[0].view(torch.uint32))
    expected = [value.clone() for value in args[1:5]]
    args[4].zero_()
    plan.baseline[plan.grid](*args)
    for actual, wanted in zip(args[1:5], expected):
        _native_indexed_same_values(actual, wanted)


@pytest.mark.parametrize("contract", ["missing_zero", "missing_private", "folded_group"])
def test_native_private_reduction_ownership_unproved_protocol_keeps_locks(device, contract):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_reduction_ownership

    # G=1 canonicalizes away the modulo operation required by the bounded proof.
    args = _private_reduction_args(device, 17, 13, 1 if contract == "folded_group" else 4)
    private = (2, 3, 4) if contract == "missing_private" else (1, 2, 3, 4)
    zero = () if contract == "missing_zero" else (4, )
    plan = compile_private_reduction_ownership(_native_private_reduction_protocol, args, grid=(17, ),
                                               private_args=private, zero_initialized_args=zero)
    assert not plan.report()["transformed"]
    assert _native_indexed_has_atomic(plan.optimized)


@pytest.mark.parametrize("change", ["grid", "scalar", "metadata", "alias"])
def test_native_private_reduction_ownership_checks_launch_binding(device, change):
    _semantic_require_cuda(device)
    from triton.experimental.semantic_contracts.native import compile_private_reduction_ownership

    args = _private_reduction_args(device, 17, 13, 4)
    plan = compile_private_reduction_ownership(_native_private_reduction_protocol, args, grid=(17, ),
                                               private_args=(1, 2, 3, 4), zero_initialized_args=(4, ))
    changed, kwargs = list(args), {}
    if change == "grid":
        kwargs["grid"] = (18, )
    elif change == "scalar":
        changed[5] = 12
    elif change == "metadata":
        changed[0] = args[0].half()
    else:
        changed[1] = args[0]
    with pytest.raises(ValueError):
        plan(*changed, **kwargs)


@triton.jit
def _native_immutable_barrier_recurrence_tl(X, Gamma, Scratch, Y, STEPS, N: tl.constexpr, BLOCK: tl.constexpr,
                                            VOLATILE: tl.constexpr):
    batch = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    initial = tl.load(X + batch * N + col, mask, other=0)
    tl.store(Scratch + batch * 2 * N + col, initial, mask)
    tl.debug_barrier()
    for step in range(STEPS):
        neighbor = tl.where(col > 0, col - 1, N - 1)
        previous = tl.load(Scratch + batch * 2 * N + (step % 2) * N + neighbor, mask, other=0)
        gamma = tl.load(Gamma + col, mask, other=0, volatile=VOLATILE)
        tl.store(Scratch + batch * 2 * N + ((step + 1) % 2) * N + col, previous + gamma, mask)
        tl.debug_barrier()
    final = tl.load(Scratch + batch * 2 * N + (STEPS % 2) * N + col, mask, other=0)
    tl.store(Y + batch * N + col, final, mask)


@gluon.jit
def _native_immutable_barrier_recurrence_gluon(X, Gamma, Scratch, Y, STEPS, N: gl.constexpr, BLOCK: gl.constexpr,
                                               VOLATILE: gl.constexpr):
    batch = gl.program_id(0)
    col = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0]))
    mask = col < N
    initial = gl.load(X + batch * N + col, mask, other=0)
    gl.store(Scratch + batch * 2 * N + col, initial, mask)
    gl.barrier()
    for step in range(STEPS):
        neighbor = gl.where(col > 0, col - 1, N - 1)
        previous = gl.load(Scratch + batch * 2 * N + (step % 2) * N + neighbor, mask, other=0)
        gamma = gl.load(Gamma + col, mask, other=0, volatile=VOLATILE)
        gl.store(Scratch + batch * 2 * N + ((step + 1) % 2) * N + col, previous + gamma, mask)
        gl.barrier()
    final = gl.load(Scratch + batch * 2 * N + (STEPS % 2) * N + col, mask, other=0)
    gl.store(Y + batch * N + col, final, mask)


@pytest.mark.parametrize("frontend", ["triton", "gluon"])
@pytest.mark.parametrize("steps", [0, 3])
@pytest.mark.parametrize("volatile", [False, True])
def test_native_immutable_cta_barrier_preserves_recurrence_and_zero_trip(device, frontend, steps, volatile):
    _semantic_require_cuda(device)
    batch, width, guard = 4, 65, 64
    x = torch.arange(batch * width, device=device, dtype=torch.float32).reshape(batch, width)
    # For zero trips a null allocation must remain unread. This low-level
    # compiler test binds the vacuous immutable-input contract directly instead
    # of the conservative host wrapper, which rejects empty backing storage.
    gamma = torch.arange(width, device=device, dtype=torch.float32) % 7 if steps else torch.empty(0, device=device)
    scratch_storage = torch.full((batch * 2 * width + 2 * guard, ), 123., device=device)
    output_storage = torch.full((batch * width + 2 * guard, ), 123., device=device)
    scratch = scratch_storage[guard:-guard].reshape(batch, 2, width)
    output = output_storage[guard:-guard].reshape(batch, width)
    kernel = _native_immutable_barrier_recurrence_tl if frontend == "triton" else _native_immutable_barrier_recurrence_gluon
    args = (x, gamma, scratch, output, steps, width, 128, volatile)
    options = dict(num_warps=4, num_stages=1, enable_indexed_update_specialization=False,
                   enable_private_atomic_relaxation=False)
    baseline = kernel.warmup(*args, grid=(batch, ), **options)

    def compile_with(attributes):
        attrs = {key: list(value) for key, value in baseline.src.attrs.items()}
        for key, value in attributes.items():
            attrs[key] = attrs.get(key, []) + value
        source = type(baseline.src)(baseline.src.fn, baseline.src.signature, constexprs=baseline.src.constants,
                                    attrs=attrs)
        return triton.compile(source, target=baseline.metadata.target, options=options)

    immutable = compile_with({(1, ): [("tt.experimental_immutable", 1)]})
    alias = compile_with({(i, ): [("tt.experimental_alias_group", 0 if i == 1 else 1)] for i in range(4)})
    if steps:
        before = baseline.asm["ttgir"].split("scf.for", 1)[0].count("tt.load")
        moved = immutable.asm["ttgir"].split("scf.for", 1)[0].count("tt.load")
        # Production Gluon lowering currently does not invoke Triton LICM.
        # Keep it as an unchanged-pipeline correctness control; adding a general
        # LICM stage to user-scheduled Gluon kernels is a separate decision.
        assert moved == before + (not volatile and frontend == "triton")
        assert alias.asm["ttgir"].split("scf.for", 1)[0].count("tt.load") == before
    # The communication protocol is unchanged even when the global input load
    # moves. Every state update depends on another logical element's prior row.
    assert immutable.asm["ttgir"].count("ttg.barrier") == baseline.asm["ttgir"].count("ttg.barrier")
    expected = x.clone()
    for _ in range(steps):
        expected = torch.roll(expected, 1, -1) + gamma
    for compiled in (baseline, immutable, alias):
        scratch.fill_(float("nan"))
        output.fill_(float("nan"))
        compiled[(batch, 1, 1)](*args)
        assert torch.equal(output, expected)
        for storage in (scratch_storage, output_storage):
            assert torch.all(storage[:guard] == 123.) and torch.all(storage[-guard:] == 123.)
