"""A bounded FX-to-Triton path that consumes derived semantic facts.

``compile(fn, example_args)`` captures an ATen FX graph using FakeTensors. It
supports tensor-only positional inputs with fixed shape/dtype/device/strides.
Pointer-alignment and lazy-view metadata are guarded as well. Captured tensor
metadata and the captured graph are fixed; tensor contents may change between
invocations, since no value proof is inferred from external/captured payloads.
Ordinary ATen nodes execute through PyTorch; the two functional integration ops
in ``ops`` and a bounded, version-checked ``aten.embedding_dense_backward``
specialization lower to Triton. This is not a torch.compile backend or a
torch-mlir integration. Inputs must not be concurrently mutated by another execution.

Uniqueness comes exclusively from the graph analysis at each consumer. It is
never inferred from sample index values, tensor addresses, or user metadata.
RMS alias groups come from fresh outputs and validated complete tensor footprints;
the native pass determines whether those groups make loads invariant.
"""

import copy
import math
import random
import statistics

import torch
import triton
from torch.fx.experimental.proxy_tensor import make_fx

from . import kernels
from . import ops  # noqa: F401 - register the owned functional operator schemas

_FLOATS = (torch.float16, torch.bfloat16, torch.float32)
_MAX_ELEMENTS = (1 << 31) - 1
# The floating-point path below is audited against this exact CUDA implementation.
# Other Torch revisions keep the original ATen operator until separately audited.
_ATEN_EMBEDDING_REVISION = "5c4886908584029761b579af026dcfb627c84070"


def _tensor_signature(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("semantic compile accepts tensor-only positional inputs")
    if tensor.layout != torch.strided:
        raise ValueError("only strided tensors are supported")
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, str(tensor.device), tensor.requires_grad,
            tensor.is_neg(), tensor.is_conj(), tensor.data_ptr() % 16)


def _validate_tensor(tensor, *, dtypes, ndim, device=None):
    if tensor.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("semantic kernel lowering currently requires NVIDIA CUDA")
    if tensor.dtype not in dtypes or tensor.ndim != ndim or not tensor.is_contiguous():
        raise ValueError(f"expected contiguous rank-{ndim} tensor with dtype in {dtypes}")
    if tensor.requires_grad:
        raise ValueError("semantic integration operators do not register autograd formulas")
    if tensor.is_neg() or tensor.is_conj():
        raise ValueError("resolve lazy negative/conjugate views before raw-pointer kernel lowering")
    if device is not None and tensor.device != device:
        raise ValueError("all operator tensors must be on the same CUDA device")
    if tensor.numel() > _MAX_ELEMENTS:
        raise ValueError("tensor footprint exceeds this backend's checked index range")


def _span(tensor):
    # Only used after checking contiguity. Registered kernels never access
    # outside these logical spans: every tail/invalid row is masked.
    first = tensor.data_ptr()
    return first, first + tensor.numel() * tensor.element_size()


def _overlap(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _fresh_output_groups(inputs, outputs):
    """All inputs may alias; each newly allocated output must be disjoint."""
    input_spans = [_span(t) for t in inputs]
    output_spans = []
    for output in outputs:
        span = _span(output)
        if any(_overlap(span, other) for other in input_spans + output_spans):
            raise RuntimeError("fresh-output allocation violated the registered storage contract")
        output_spans.append(span)
    return {
        "inputs_may_alias": True, "output_spans_disjoint": True, "footprints":
        "complete contiguous logical tensor spans; masked tails", "source":
        "functional operator output allocation plus runtime span validation"
    }


def _launch(compiled, grid, arguments):
    return compiled[tuple(grid) + (1, ) * (3 - len(grid))](*arguments)


def _node_argument(node, position, name):
    return node.args[position] if position < len(node.args) else node.kwargs[name]


def _resources(kernel):
    # warmup compiles but does not initialize the driver's module handle.
    kernel._init_handles()
    return {"registers": kernel.n_regs, "spills": kernel.n_spills, "shared_bytes": kernel.metadata.shared}


def _measure_shared(functions):
    """First-use calibration; identical arguments/output storage for variants."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm up semantic plans before CUDA graph capture")
    graphs = {}
    for name, function in functions.items():
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(32):
                function()
        graphs[name] = graph
    for _ in range(5):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    rng = random.Random(1729)
    for _ in range(9):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 32)
    return {
        "samples_us": samples, "median_us": {name: statistics.median(v)
                                             for name, v in samples.items()}, "shared_output_buffers": True,
        "selection_margin": 0.02
    }


def choose_hoist(calibration):
    """Select the inferred variant only with a measured margin over baseline."""
    medians = calibration["median_us"]
    return "inferred" if medians["inferred"] < medians["baseline"] * 0.98 else "baseline"


class RMSPlan:

    def __init__(self, enabled=True, autotune=True):
        self.enabled, self.autotune = enabled, autotune
        self.kernels = {}
        self.selected = None
        self.key = None
        self.details = {
            "op": "rms_norm", "selected_optimization": "pending", "kernel_launches": ["rms_norm"], "native_attrs": {},
            "proofs": []
        }

    def run(self, x, weight, eps, num_warps, grid_multiplier):
        _validate_tensor(x, dtypes=_FLOATS, ndim=2)
        _validate_tensor(weight, dtypes=_FLOATS, ndim=1, device=x.device)
        if not self.kernels and self.enabled and self.autotune and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up semantic plans before CUDA graph capture")
        m, n = x.shape
        if n == 0 or weight.numel() != n or not math.isfinite(eps) or eps <= 0:
            raise ValueError("invalid RMSNorm width, weight shape, or epsilon")
        if num_warps not in (4, 8) or grid_multiplier < 1:
            raise ValueError("supported RMS configurations use 4/8 warps and a positive grid multiplier")
        y = torch.empty_like(x)
        rsigma = torch.empty((m, ), dtype=torch.float32, device=x.device)
        storage_proof = _fresh_output_groups([x, weight], [y, rsigma])
        grid = (torch.cuda.get_device_properties(x.device).multi_processor_count * grid_multiplier, )
        if m + grid[0] > _MAX_ELEMENTS:
            raise ValueError("RMS loop induction could overflow the supported int32 range")
        arguments = [x, y, weight, rsigma, m, n, eps, triton.next_power_of_2(n)]
        key = (_tensor_signature(x), _tensor_signature(weight), eps, num_warps, grid)
        if self.key is not None and key != self.key:
            raise ValueError("RMS plan specialization metadata changed")
        if not self.kernels:
            self.key = key
            baseline = kernels.rmsnorm_persistent.warmup(*arguments, grid=grid, num_warps=num_warps)
            self.kernels["baseline"] = baseline
            if self.enabled:
                attrs = copy.deepcopy(baseline.src.attrs)
                # Group0 may alias internally; fresh output groups cannot alias it
                # or one another. No immutable or initialized assertion is added.
                groups = {0: 0, 1: 1, 2: 0, 3: 2}
                for index, group in groups.items():
                    attrs[(index, )] = list(attrs.get((index, ), [])) + [("tt.experimental_alias_group", group)]
                source = triton.compiler.ASTSource(baseline.src.fn, baseline.src.signature,
                                                   constexprs=baseline.src.constants, attrs=attrs)
                inferred = triton.compile(source, target=baseline.metadata.target, options={"num_warps": num_warps})
                self.kernels["inferred"] = inferred
                self.details["native_attrs"] = {str(i): [["tt.experimental_alias_group", g]] for i, g in groups.items()}
                self.details["proofs"] = [
                    storage_proof, {
                        "source": "native complete read/write effect analysis", "conclusion":
                        "input loads cannot be clobbered by disjoint output stores"
                    }
                ]
            if self.enabled and self.autotune:
                functions = {
                    name: lambda kernel=kernel: _launch(kernel, grid, arguments)
                    for name, kernel in self.kernels.items()
                }
                functions["baseline"]()
                expected_y, expected_r = y.clone(), rsigma.clone()
                functions["inferred"]()
                if not torch.equal(y.view(torch.uint8), expected_y.view(torch.uint8)) or not torch.equal(
                        rsigma.view(torch.uint8), expected_r.view(torch.uint8)):
                    raise RuntimeError("inferred RMS variant differs from the baseline")
                calibration = _measure_shared(functions)
                self.selected = choose_hoist(calibration)
                self.details["profitability"] = calibration
            else:
                self.selected = "inferred" if self.enabled else "baseline"
                self.details["profitability"] = {"policy": "forced legal variant" if self.enabled else "disabled"}
            self.details["selected_optimization"] = "immutable_hoist" if self.selected == "inferred" else "baseline"
            self.details["resources"] = {name: _resources(k) for name, k in self.kernels.items()}
        _launch(self.kernels[self.selected], grid, arguments)
        return y, rsigma


class EmbeddingPlan:

    def __init__(self, unique=False, proof=None, complete_coverage=False, atomic_ftz=True):
        self.unique = unique
        self.complete_coverage = bool(unique and complete_coverage)
        self.atomic_ftz = atomic_ftz
        self.kernels = {}
        self.details = {
            "op": "embedding_backward", "selected_optimization": "unique_scatter" if unique else "atomic_scatter",
            "proofs": [] if proof is None else [proof], "native_attrs": {}, "kernel_launches": []
        }

    def _run_unique(self, grad, indices, num_weights, padding_idx, block_d):
        n, d = grad.shape
        out = torch.empty((num_weights, d), dtype=grad.dtype, device=grad.device)
        storage_proof = _fresh_output_groups([grad, indices], [out])
        launches = []
        size = num_weights * d
        if size and not self.complete_coverage:
            self.kernels["zero_output"] = kernels.zero_buffer[(triton.cdiv(size, 1024), )](out, size, 1024)
            launches.append("zero_output")
        if n and num_weights:
            self.kernels["update"] = kernels.embedding_unique_write[(n, triton.cdiv(
                d, block_d))](grad, indices, out, n, d, num_weights, padding_idx, block_d, self.complete_coverage,
                              self.atomic_ftz, num_warps=4)
            launches.append("update")
        self.details["kernel_launches"] = launches
        self.details["complete_coverage"] = self.complete_coverage
        self.details["direct_output"] = True
        self.details["destination_mapping"] = {
            "formula":
            "index[token] * D + feature",
            "domain":
            "valid indices and 0 <= feature < D across the complete grid",
            "injectivity":
            True,
            "storage":
            storage_proof,
            "initialization":
            "complete overwrite, including explicit +0 for padding"
            if self.complete_coverage else "output zeroed before scatter; invalid/padding indices do not update it",
            "initial_value":
            "logical +0; unique updates do not read destination memory",
            "arithmetic":
            "FP32 add.rn.ftz(+0, contribution), then output cast"
            if self.atomic_ftz else "FP32 add.rn(+0, contribution), then output cast; subnormals preserved",
        }
        return out

    def run(self, grad, indices, num_weights, padding_idx, scale_by_freq, block_d):
        _validate_tensor(grad, dtypes=_FLOATS, ndim=2)
        _validate_tensor(indices, dtypes=(torch.int32, torch.int64), ndim=1, device=grad.device)
        n, d = grad.shape
        if indices.numel() != n or d <= 0 or num_weights < 0 or num_weights * d > _MAX_ELEMENTS:
            raise ValueError("invalid embedding shapes or unchecked address range")
        if n > _MAX_ELEMENTS or block_d not in (64, 128, 256, 512, 1024):
            raise ValueError("embedding count or tile exceeds supported bounds")
        if self.unique:
            return self._run_unique(grad, indices, num_weights, padding_idx, block_d)
        scratch = torch.empty((num_weights, d), dtype=torch.float32, device=grad.device)
        out = scratch if grad.dtype == torch.float32 else torch.empty_like(scratch, dtype=grad.dtype)
        _fresh_output_groups([grad, indices], [scratch] + ([] if out is scratch else [out]))
        launches = []
        size = num_weights * d
        if size:
            self.kernels["zero_scratch"] = kernels.zero_buffer[(triton.cdiv(size, 1024), )](scratch, size, 1024)
            launches.append("zero_scratch")
        counts = None
        if scale_by_freq and not self.unique and num_weights:
            counts = torch.empty((num_weights, ), dtype=torch.int32, device=grad.device)
            self.kernels["zero_counts"] = kernels.zero_buffer[(triton.cdiv(num_weights, 1024), )](counts, num_weights,
                                                                                                  1024)
            launches.append("zero_counts")
            if n:
                self.kernels["count"] = kernels.embedding_count[(triton.cdiv(n, 512), )](indices, counts, n,
                                                                                         num_weights, padding_idx, 512)
                launches.append("count")
        if n and num_weights:
            self.kernels["update"] = kernels.embedding_update[(n, triton.cdiv(d,
                                                                              block_d))](grad, indices, counts, scratch,
                                                                                         n, d, num_weights, padding_idx,
                                                                                         scale_by_freq, self.unique,
                                                                                         True, block_d, num_warps=4)
            launches.append("update")
        if out is not scratch and size:
            self.kernels["convert"] = kernels.convert_output[(triton.cdiv(size, 1024), )](scratch, out, size, 1024)
            launches.append("convert")
        self.details["kernel_launches"] = launches
        self.details["destination_mapping"] = {
            "formula": "index[token] * D + feature", "domain":
            "valid unpadded rows and 0 <= feature < D across the complete grid", "injectivity": self.unique, "storage":
            "fresh contiguous [V,D], initialized +0, no intermediate observers", "arithmetic":
            "int64 address offsets within checked bounds; FP32 add.rn.ftz matches global atomic addition"
        }
        return out


class ATenEmbeddingPlan:
    """A conservative consumer of producer proofs for a real framework operator.

    In the pinned Torch CUDA implementation, unique valid indices imply N <= V.
    Its large-input path has max_segments=N and max_partial=N+floor(N/10),
    whereas sum_scatter_grid>=N, so max_partial>4*sum_scatter_grid is false.
    The atomic fused branch is unreachable. Both the remaining segmented path
    and the small-input feature path reduce one contribution with ordinary +0
    addition. We preserve subnormals here, unlike the owned atomic operator.
    NaN payload bits are not part of either operator's contract.

    Source: pytorch/pytorch at _ATEN_EMBEDDING_REVISION,
    aten/src/ATen/native/cuda/{Embedding.cu,EmbeddingBackwardKernel.cu}.
    """

    def __init__(self, facts, enabled):
        self.facts = facts
        self.enabled = enabled
        self.specialized_plan = None
        self.specialized_vocabulary = None
        self.kernels = {}
        self.details = {
            "op": "aten.embedding_dense_backward", "selected_optimization": "pending", "proofs": [], "kernel_launches":
            [], "native_attrs": {}
        }

    def _unsupported(self, grad, indices, num_weights, padding_idx, scale_grad_by_freq):
        if not self.enabled:
            return "semantic specialization disabled"
        if torch.version.git_version != _ATEN_EMBEDDING_REVISION:
            return "Torch CUDA implementation revision has not been audited"
        if not isinstance(num_weights, int) or not isinstance(padding_idx, int) or not isinstance(
                scale_grad_by_freq, bool):
            return "operator scalar arguments are outside the static contract"
        facts = self.facts
        if facts is None or not facts.unique_global or facts.integer_range is None:
            return "no valid producer proof of global uniqueness and bounds"
        if facts.shape != tuple(indices.shape):
            return "index metadata differs from the proved domain"
        if not (0 <= facts.integer_range.lower <= facts.integer_range.upper < num_weights):
            return "index range is not proved within the output vocabulary"
        try:
            _validate_tensor(grad, dtypes=_FLOATS, ndim=2)
            _validate_tensor(indices, dtypes=(torch.int32, torch.int64), ndim=1, device=grad.device)
        except (ValueError, TypeError) as error:
            return str(error)
        if torch.cuda.current_device() != grad.device.index:
            return "Triton specialization requires the tensor's CUDA device to be current"
        n, d = grad.shape
        if n == 0 or d == 0 or num_weights <= 0 or indices.numel() != n or n > num_weights:
            return "shape is outside the nonempty, unique row-scatter contract"
        if num_weights * d > _MAX_ELEMENTS:
            return "output exceeds the checked address range"
        # The audited native selector multiplies its int32 grid size by four.
        # With narrow rows grid=N; keep that calculation representable. For wide
        # rows the checked N*D footprint also bounds the larger grid safely.
        if n > _MAX_ELEMENTS // 4:
            return "native dispatch arithmetic exceeds the audited int32 range"
        if not (padding_idx == -1 or 0 <= padding_idx < num_weights):
            return "padding index is outside the audited contract"
        return None

    def run(self, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
        reason = self._unsupported(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)
        if reason is not None:
            self.kernels = {}
            self.details = {
                "op": "aten.embedding_dense_backward", "selected_optimization": "aten_fallback", "fallback_reason":
                reason, "kernel_launches": ["original_aten_operator"], "native_attrs": {}, "proofs": []
            }
            return torch.ops.aten.embedding_dense_backward.default(grad_output, indices, num_weights, padding_idx,
                                                                   scale_grad_by_freq)
        if self.specialized_plan is None or self.specialized_vocabulary != num_weights:
            complete = self.facts.covers_integer_interval(0, num_weights)
            proof = {
                "source": "producer analysis at consumer", "fact": self.facts.to_dict(), "complete_coverage": complete,
                "coverage_rule": "global uniqueness + in-range bounds + cardinality equals vocabulary",
                "torch_revision": _ATEN_EMBEDDING_REVISION
            }
            self.specialized_plan = EmbeddingPlan(unique=True, proof=proof, complete_coverage=complete,
                                                  atomic_ftz=False)
            self.specialized_vocabulary = num_weights
        plan = self.specialized_plan
        result = plan.run(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, 128)
        self.kernels = plan.kernels
        self.details = dict(plan.details, op="aten.embedding_dense_backward",
                            selected_optimization="unique_direct_write", fallback_reason=None,
                            replaces="original ATen feature/grouping or sort/segment/count machinery")
        return result


class _Interpreter(torch.fx.Interpreter):

    def __init__(self, owner):
        super().__init__(owner.graph_module)
        self.owner = owner
        self.current = None

    def run_node(self, node):
        self.current = node
        return super().run_node(node)

    def call_function(self, target, args, kwargs):
        plan = self.owner.plans.get(self.current.name)
        if plan is not None:
            return plan.run(*args, **kwargs)
        return super().call_function(target, args, kwargs)


class CompiledGraph:

    def __init__(self, graph_module, example_args, *, enabled, autotune):
        from .analysis import FactsAnalyzer
        self.graph_module = graph_module
        self.signatures = tuple(_tensor_signature(t) for t in example_args)
        self.captured_signatures = {}
        for node in graph_module.graph.nodes:
            if node.op == "get_attr":
                value = self._captured(node.target)
                if not isinstance(value, torch.Tensor):
                    raise TypeError("only tensor get_attr captures are supported")
                self.captured_signatures[node.target] = _tensor_signature(value)
        self.analysis = FactsAnalyzer(graph_module).analyze()
        self.plans = {}
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.triton_semantics.rms_norm.default:
                self.plans[node.name] = RMSPlan(enabled=enabled, autotune=autotune)
            elif node.target == torch.ops.triton_semantics.embedding_backward.default:
                index_node = _node_argument(node, 1, "indices")
                facts = self.analysis.at(node, index_node)
                unique = bool(enabled and facts is not None and facts.unique_global)
                proof = {
                    "source": "producer analysis at consumer", "indices_node": index_node.name, "global_unique": unique,
                    "fact": None if facts is None else facts.to_dict()
                }
                vocabulary = _node_argument(node, 2, "num_weights")
                complete = bool(unique and isinstance(vocabulary, int) and facts.covers_integer_interval(0, vocabulary))
                proof["complete_coverage"] = complete
                proof["coverage_rule"] = "global uniqueness + in-range bounds + cardinality equals vocabulary"
                self.plans[node.name] = EmbeddingPlan(unique=unique, proof=proof, complete_coverage=complete)
            elif node.target == torch.ops.aten.embedding_dense_backward.default:
                facts = self.analysis.at(node, _node_argument(node, 1, "indices"))
                self.plans[node.name] = ATenEmbeddingPlan(facts, enabled)
        self.enabled = enabled

    def _captured(self, name):
        value = self.graph_module
        for part in name.split("."):
            value = getattr(value, part)
        return value

    def __call__(self, *args):
        signatures = tuple(_tensor_signature(t) for t in args)
        if signatures != self.signatures:
            raise ValueError("compiled graph input metadata or pointer alignment changed")
        for name, signature in self.captured_signatures.items():
            if _tensor_signature(self._captured(name)) != signature:
                raise ValueError("compiled graph captured tensor metadata or pointer alignment changed")
        return _Interpreter(self).run(*args)

    @property
    def kernels(self):
        return {name: plan.kernels for name, plan in self.plans.items()}

    def report(self):
        return {
            "enabled": self.enabled, "nodes":
            [dict(name=name, **copy.deepcopy(plan.details))
             for name, plan in self.plans.items()], "analysis": self.analysis.to_dict(), "scope":
            "static-shape ATen FX graph with explicit integration ops and a bounded ATen embedding specialization"
        }


def compile(fn, example_args, *, enabled=True, autotune=True):
    """Capture and specialize a tensor-only function without inspecting index data.

    Facts are derived from the captured program for each consumer and invocation.
    The resulting executable guards input metadata. Producer-created indices are
    regenerated on every execution; external index tensors never acquire a
    uniqueness proof from their example values. Use autotune=False for deterministic
    legality tests; the default calibrates RMS variants once and retains baseline
    when inferred hoisting does not beat it by at least two percent.
    """
    if not isinstance(example_args, tuple):
        raise TypeError("example_args must be a tuple of tensors")
    for tensor in example_args:
        _tensor_signature(tensor)
        if tensor.requires_grad:
            raise ValueError("semantic graph integration currently requires inputs without autograd")
    graph = make_fx(fn, tracing_mode="fake", _allow_non_fake_inputs=True)(*example_args)
    return CompiledGraph(graph, example_args, enabled=enabled, autotune=autotune)
