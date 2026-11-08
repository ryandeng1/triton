import numpy
import pytest
import torch
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
    assert "zero_scratch" in node["kernel_launches"]
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
