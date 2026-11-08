"""Conservative value and storage facts for a bounded set of ATen FX graphs.

This module reads operator semantics and tensor *metadata*, never tensor data or
user-supplied proof annotations. Facts are scoped to a program point. Consumers
must use ``result.at(consumer, value)``: a fact recorded at a producer can have
been invalidated by a later write through an alias.

The analysis proves uniqueness of logical integer elements, not arbitrary
physical scatter addresses. A lowering must separately validate its destination
layout, bounds, update domain, and observation/synchronization semantics.
"""

from dataclasses import dataclass, replace
from math import prod
from typing import Any, FrozenSet, Mapping, Optional, Tuple
import operator

import torch
from torch.fx import GraphModule, Node

# Absence of alias annotations alone is not enough for arbitrary internal ATen
# operations: _unsafe_view deliberately omits them despite sharing storage.
# This bounded registry names operations whose fresh-result semantics we use.
_FRESH_RESULTS = frozenset({
    "aten.randperm.default",
    "aten.arange.default",
    "aten.arange.start",
    "aten.arange.start_step",
    "aten.zeros.default",
    "aten.zeros_like.default",
    "aten.new_zeros.default",
    "aten.full.default",
    "aten.full_like.default",
    "aten.new_full.default",
    "aten.empty.memory_format",
    "aten.empty_like.default",
    "aten.new_empty.default",
    "aten.randn.default",
    "aten.randn_like.default",
    "aten.rand.default",
    "aten.rand_like.default",
    "aten.clone.default",
    "aten._to_copy.default",
    "aten.add.Tensor",
    "aten.add.Scalar",
    "aten.sub.Tensor",
    "aten.sub.Scalar",
    "aten.mul.Tensor",
    "aten.mul.Scalar",
    # Functional custom operators implemented by this experimental backend.
    "triton_semantics.rms_norm.default",
    "triton_semantics.embedding_backward.default",
})
_INITIALIZED_CUSTOM_RESULTS = frozenset({
    "triton_semantics.rms_norm.default",
    "triton_semantics.embedding_backward.default",
})
_CONTENT_WRITES = frozenset({
    "aten.zero_.default",
    "aten.fill_.Scalar",
    "aten.fill_.Tensor",
    "aten.copy_.default",
    "aten.add_.Tensor",
    "aten.add_.Scalar",
    "aten.sub_.Tensor",
    "aten.sub_.Scalar",
    "aten.mul_.Tensor",
    "aten.mul_.Scalar",
})


@dataclass(frozen=True)
class IntegerRange:
    """An inclusive conservative range; empty tensors may use an overestimate."""

    lower: int
    upper: int

    def fits(self, dtype: torch.dtype) -> bool:
        bounds = _integer_bounds(dtype)
        return bounds is not None and bounds[0] <= self.lower <= self.upper <= bounds[1]

    def to_dict(self):
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class TensorFacts:
    shape: Optional[Tuple[int, ...]]
    dtype: Optional[torch.dtype]
    storage_ids: FrozenSet[str]
    storage_versions: Tuple[Tuple[str, int], ...]
    unique_global: bool = False
    # For an axis d, fixing all other coordinates gives distinct elements along d.
    # Axis-wise uniqueness does not in general imply global uniqueness.
    unique_dims: FrozenSet[int] = frozenset()
    integer_range: Optional[IntegerRange] = None
    known_zero: bool = False
    initialized: bool = False
    proof: Tuple[str, ...] = ()
    scope: str = ""

    @property
    def fresh(self) -> bool:
        return bool(self.storage_ids) and "external" not in self.storage_ids

    def to_dict(self):
        return {
            "shape": self.shape,
            "dtype": str(self.dtype) if self.dtype is not None else None,
            "storage_ids": sorted(self.storage_ids),
            "storage_versions": dict(self.storage_versions),
            "fresh": self.fresh,
            "unique_global": self.unique_global,
            "unique_dims": sorted(self.unique_dims),
            "integer_range": self.integer_range.to_dict() if self.integer_range else None,
            "known_zero": self.known_zero,
            "initialized": self.initialized,
            "proof": list(self.proof),
            "scope": self.scope,
        }


@dataclass
class AnalysisResult:
    """Snapshots before operations; ``produced`` is diagnostic, not a use fact."""

    before: Mapping[Node, Mapping[Node, Any]]
    produced: Mapping[Node, Any]
    events: Tuple[dict, ...]

    def at(self, consumer: Node, value: Node) -> Optional[TensorFacts]:
        fact = self.before.get(consumer, {}).get(value)
        return fact if isinstance(fact, TensorFacts) else None

    def to_dict(self):

        def encode(value):
            if isinstance(value, TensorFacts):
                return value.to_dict()
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            return None

        return {
            "scope": "Facts valid at the selected FX program point; no payload inspection",
            "produced": {node.name: encode(value)
                         for node, value in self.produced.items()},
            "before": {
                node.name: {value.name: encode(fact)
                            for value, fact in snapshot.items()}
                for node, snapshot in self.before.items()
            },
            "events": list(self.events),
        }


def _integer_bounds(dtype):
    if dtype is None or dtype == torch.bool:
        return None
    try:
        info = torch.iinfo(dtype)
    except TypeError:
        return None
    return info.min, info.max


def _tensor_meta(value):
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None, None
    shape = tuple(value.shape)
    # Do not force symbolic integers to concrete values or introduce hidden guards.
    if any(type(dim) is not int or dim < 0 for dim in shape):
        shape = None
    return shape, value.dtype


def _nodes(value):
    if isinstance(value, Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for element in value:
            yield from _nodes(element)
    elif isinstance(value, dict):
        for element in value.values():
            yield from _nodes(element)


def _facts(value):
    if isinstance(value, TensorFacts):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _facts(item)


def _constant_int(value):
    return type(value) is int


def _literal_zero(value):
    # Do not invoke Tensor/SymInt/custom equality or read a tensor payload.
    return type(value) in (int, float, bool, complex) and value == 0


class FactsAnalyzer:
    """Analyze supported ATen producers, views, copies, and mutation effects.

    All external tensors may alias each other. Fresh allocations have separate
    provenance. Schemas supply positive alias/mutation effects, but omitted alias
    annotations only prove freshness for the bounded operator registry above.
    Schema-less calls invalidate tracked facts and conservatively alias all
    extant storage. Unknown operations never supply value proofs.
    """

    def __init__(self, graph_module: GraphModule):
        self.graph_module = graph_module
        self._current = {}
        self._versions = {"external": 0}
        self._events = []

    def analyze(self) -> AnalysisResult:
        self._current = {}
        self._versions = {"external": 0}
        self._events = []
        before, produced = {}, {}
        for node in self.graph_module.graph.nodes:
            before[node] = dict(self._current)
            if node.op == "output":
                continue
            if node.op in ("placeholder", "get_attr"):
                value = self._unknown_output(node, frozenset({"external"}), "external tensor; values unknown")
            elif node.op == "call_function" and node.target is operator.getitem:
                value = self._getitem(node)
            else:
                schema = getattr(node.target, "_schema", None) if node.op == "call_function" else None
                if schema is None:
                    # An arbitrary Python/module call can capture tensors as well as
                    # mutate arguments. Do not retain proofs across such a call.
                    storage = frozenset(self._versions)
                    self._invalidate(node, storage, "call has no modeled effect schema")
                    value = self._unknown_output(node, storage, "unknown call and alias behavior")
                else:
                    arguments = self._schema_arguments(node, schema)
                    if not str(node.target).startswith("aten.") and str(node.target) not in _INITIALIZED_CUSTOM_RESULTS:
                        # A custom schema describes argument effects but does not
                        # by itself exclude hidden captured/global effects. Only
                        # this backend's owned functional integration ops are
                        # trusted beyond the ATen operator set.
                        self._invalidate(node, frozenset(self._versions),
                                         "unmodeled custom operator effects: " + str(node.target))
                    writes = [v for a, v in arguments if a.alias_info is not None and a.alias_info.is_write]
                    if writes:
                        # Besides contents, set_/resize-like operations can change
                        # alias relations. Join all tensor argument storage and
                        # invalidate aliases of either side, conservatively.
                        storage = self._input_storage(node)
                        if str(node.target) not in _CONTENT_WRITES:
                            # A storage-changing operator may receive a Storage
                            # object, not a tracked Tensor argument. Its new
                            # storage must not keep a fresh-allocation claim.
                            storage = storage | frozenset({"external"})
                        self._invalidate(node, storage, "mutable operator " + str(node.target))
                    value = self._schema_output(node, schema, arguments)
                    if not writes and isinstance(value, TensorFacts):
                        value = self._derive(node, value)
                    elif writes and isinstance(value, TensorFacts) and str(
                            node.target) in ("aten.zero_.default", "aten.fill_.Scalar"):
                        scalar = 0 if str(node.target) == "aten.zero_.default" else self._arg(node, 1, "value", None)
                        if _literal_zero(scalar):
                            value = self._zero_fact(node, value, "entire logical tensor initialized to zero")
                            if isinstance(node.args[0], Node):
                                self._current[node.args[0]] = value
            if value is not None:
                self._current[node] = value
                produced[node] = value
        return AnalysisResult(before, produced, tuple(self._events))

    def _arg(self, node, index, name, default=None):
        return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)

    def _source(self, node):
        source = node.args[0] if node.args else None
        value = self._current.get(source) if isinstance(source, Node) else None
        return value if isinstance(value, TensorFacts) else None

    def _versions_for(self, storage):
        return tuple((name, self._versions.setdefault(name, 0)) for name in sorted(storage))

    def _base(self, node, meta, storage, reason):
        shape, dtype = _tensor_meta(meta)
        if dtype is None:
            return None
        return TensorFacts(shape, dtype, storage, self._versions_for(storage), proof=(reason, ), scope=node.name)

    def _meta(self, node):
        return node.meta.get("val", node.meta.get("tensor_meta"))

    def _unknown_output(self, node, storage, reason):
        meta = self._meta(node)
        if isinstance(meta, (tuple, list)) and not hasattr(meta, "dtype"):
            return tuple(self._base(node, m, storage, reason) for m in meta)
        return self._base(node, meta, storage, reason)

    def _schema_arguments(self, node, schema):
        return [(arg, self._arg(node, i, arg.name, arg.default_value)) for i, arg in enumerate(schema.arguments)]

    def _input_storage(self, node):
        storage = set()
        for argument in _nodes((node.args, node.kwargs)):
            for fact in _facts(self._current.get(argument)):
                storage.update(fact.storage_ids)
        return frozenset(storage or {"external"})

    def _schema_output(self, node, schema, arguments):
        meta = self._meta(node)
        multi = isinstance(meta, (tuple, list)) and not hasattr(meta, "dtype")
        metas = list(meta) if multi else [meta]
        values = []
        for index, item in enumerate(metas):
            result = schema.returns[min(index, len(schema.returns) - 1)] if schema.returns else None
            if result is None:
                storage = self._input_storage(node)
            elif str(node.target) == "aten._unsafe_view.default":
                # Its schema omits alias metadata for autograd purposes.
                storage = self._input_storage(node)
            elif result.alias_info is None and str(node.target) in _FRESH_RESULTS:
                storage = frozenset({node.name + ":" + str(index)})
            elif result.alias_info is None:
                storage = frozenset(self._versions)
            else:
                result_sets = result.alias_info.before_set | result.alias_info.after_set
                storage_set = set()
                for argument, supplied in arguments:
                    alias = argument.alias_info
                    if alias is None:
                        continue
                    argument_sets = alias.before_set | alias.after_set
                    if "*" in result_sets or "*" in argument_sets or result_sets & argument_sets:
                        for source in _nodes(supplied):
                            for fact in _facts(self._current.get(source)):
                                storage_set.update(fact.storage_ids)
                storage = frozenset(storage_set or self._input_storage(node))
            fact = self._base(node, item, storage,
                              str(node.target) + ": storage provenance only; value properties unknown")
            if fact is not None and str(node.target) in _INITIALIZED_CUSTOM_RESULTS:
                fact = replace(fact, initialized=True,
                               proof=fact.proof + ("owned functional operator defines every logical output element", ))
            values.append(fact)
        return tuple(values) if multi else values[0]

    def _invalidate(self, node, storage, reason):
        for name in storage:
            self._versions[name] = self._versions.get(name, 0) + 1

        def invalidate(value):
            if isinstance(value, TensorFacts) and value.storage_ids & storage:
                joined = value.storage_ids | storage
                return TensorFacts(None, None, joined, self._versions_for(joined),
                                   proof=value.proof + ("invalidated at " + node.name + ": " + reason, ),
                                   scope=node.name)
            if isinstance(value, (tuple, list)):
                return tuple(invalidate(v) for v in value)
            return value

        self._current = {key: invalidate(value) for key, value in self._current.items()}
        self._events.append({"node": node.name, "invalidated_storage": sorted(storage), "reason": reason})

    def _getitem(self, node):
        source, index = node.args[:2]
        value = self._current.get(source) if isinstance(source, Node) else None
        if isinstance(value, (tuple, list)) and _constant_int(index) and -len(value) <= index < len(value):
            selected = value[index]
            if isinstance(selected, TensorFacts):
                return replace(selected, proof=selected.proof + ("tuple result selection", ), scope=node.name)
            return None
        # Tensor indexing in a non-ATen FX graph is deliberately unsupported.
        # Python __getitem__ can execute arbitrary effects for other objects.
        storage = frozenset(self._versions)
        self._invalidate(node, storage, "getitem operand is not a modeled tuple")
        return self._unknown_output(node, storage, "unsupported getitem")

    def _zero_fact(self, node, result, reason):
        unique = result.shape is not None and prod(result.shape) <= 1
        integer_range = IntegerRange(0, 0) if _integer_bounds(result.dtype) else None
        return replace(result, known_zero=True, initialized=True, unique_global=unique,
                       unique_dims=frozenset(range(len(result.shape))) if unique else frozenset(),
                       integer_range=integer_range, proof=result.proof + (reason, ), scope=node.name)

    def _inherit(self, node, result, source, reason, *, unique=None, dims=None, integer_range=None):
        unique = source.unique_global if unique is None else unique
        dims = source.unique_dims if dims is None else dims
        if unique and result.shape is not None:
            dims = frozenset(range(len(result.shape)))
        return replace(result, unique_global=unique, unique_dims=frozenset(dims),
                       integer_range=source.integer_range if integer_range is None else integer_range,
                       known_zero=source.known_zero, initialized=source.initialized, proof=source.proof + (reason, ),
                       scope=node.name)

    def _derive(self, node, result):
        name = str(node.target)
        if name == "aten.randperm.default":
            size = self._arg(node, 0, "n")
            if _constant_int(size) and size >= 0 and result.shape == (size, ):
                bounds = IntegerRange(0, max(size - 1, 0))
                if bounds.fits(result.dtype):
                    return replace(result, unique_global=True, unique_dims=frozenset({0}), integer_range=bounds,
                                   initialized=True,
                                   proof=(f"{node.name}: integer randperm({size}) is a permutation", ))
        elif name in ("aten.arange.default", "aten.arange.start", "aten.arange.start_step"):
            if name == "aten.arange.default":
                start, end, step = 0, self._arg(node, 0, "end"), 1
            else:
                start, end = self._arg(node, 0, "start"), self._arg(node, 1, "end")
                step = self._arg(node, 2, "step", 1)
            if all(_constant_int(v) for v in (start, end, step)) and step > 0 and end >= start:
                count = (end - start + step - 1) // step
                bounds = IntegerRange(start, start + max(count - 1, 0) * step)
                if result.shape == (count, ) and bounds.fits(result.dtype):
                    return replace(result, unique_global=True, unique_dims=frozenset({0}), integer_range=bounds,
                                   initialized=True, known_zero=count <= 1 and start == 0,
                                   proof=(f"{node.name}: nonoverflowing positive-step integer arange", ))
        elif name in ("aten.zeros.default", "aten.zeros_like.default", "aten.new_zeros.default"):
            return self._zero_fact(node, result, "zero allocation")
        elif name in ("aten.full.default", "aten.full_like.default", "aten.new_full.default"):
            position = 2 if name == "aten.new_full.default" else 1
            fill = self._arg(node, position, "fill_value")
            if _literal_zero(fill):
                return self._zero_fact(node, result, "allocation filled with zero")

        source = self._source(node)
        if source is None or source.shape is None or result.shape is None:
            return result
        if name in ("aten.clone.default", "aten.contiguous.default", "aten.detach.default", "aten.alias.default"):
            if result.shape == source.shape and result.dtype == source.dtype:
                return self._inherit(node, result, source, name + " preserves logical elements")
        elif name == "aten._to_copy.default":
            if result.shape == source.shape:
                if result.dtype == source.dtype:
                    return self._inherit(node, result, source, "copy preserves dtype and logical elements")
                source_bounds = source.integer_range
                if source_bounds is None and _integer_bounds(source.dtype):
                    source_bounds = IntegerRange(*_integer_bounds(source.dtype))
                if source_bounds is not None and source_bounds.fits(result.dtype):
                    return self._inherit(node, result, source, "integer conversion proven lossless by range",
                                         integer_range=source_bounds)
                if source.known_zero:
                    return self._zero_fact(node, result, "conversion of known zero")
                return replace(result, proof=source.proof + ("conversion not proven injective; uniqueness dropped", ))
        elif name in ("aten.view.default", "aten.reshape.default", "aten._unsafe_view.default"):
            if result.dtype == source.dtype and prod(source.shape) == prod(result.shape):
                dims = source.unique_dims if source.shape == result.shape else frozenset()
                return self._inherit(node, result, source, "reshape preserves the logical element multiset", dims=dims)
        elif name in ("aten.permute.default", "aten.transpose.int", "aten.t.default"):
            rank = len(source.shape)
            if name == "aten.permute.default":
                dims = self._arg(node, 1, "dims")
                if not isinstance(dims, (tuple, list)) or not all(_constant_int(d) for d in dims):
                    return result
                permutation = [d + rank if d < 0 else d for d in dims]
            elif name == "aten.t.default":
                permutation = list(reversed(range(rank))) if rank <= 2 else []
            else:
                a, b = self._arg(node, 1, "dim0"), self._arg(node, 2, "dim1")
                if not all(_constant_int(d) and -rank <= d < rank for d in (a, b)):
                    return result
                permutation = list(range(rank))
                a, b = a % rank, b % rank
                permutation[a], permutation[b] = permutation[b], permutation[a]
            if sorted(permutation) == list(range(rank)) and result.shape == tuple(source.shape[d] for d in permutation):
                dims = {i for i, old in enumerate(permutation) if old in source.unique_dims}
                return self._inherit(node, result, source, "dimension permutation is bijective", dims=dims)
        elif name in ("aten.slice.Tensor", "aten.select.int"):
            rank = len(source.shape)
            dim = self._arg(node, 1, "dim", 0)
            if not _constant_int(dim) or not -rank <= dim < rank:
                return result
            dim %= rank
            expected = list(source.shape)
            if name == "aten.slice.Tensor":
                start, end = self._arg(node, 2, "start", None), self._arg(node, 3, "end", None)
                step = self._arg(node, 4, "step", 1)
                if not all(v is None or _constant_int(v) for v in (start, end)) or not _constant_int(step) or step <= 0:
                    return result
                expected[dim] = len(range(*slice(start, end, step).indices(source.shape[dim])))
                dims = source.unique_dims
            else:
                index = self._arg(node, 2, "index")
                if not _constant_int(index) or not -source.shape[dim] <= index < source.shape[dim]:
                    return result
                expected.pop(dim)
                dims = {d if d < dim else d - 1 for d in source.unique_dims if d != dim}
            if result.shape == tuple(expected):
                varying = {d for d, size in enumerate(result.shape) if size > 1}
                unique = source.unique_global or (len(varying) == 1 and varying <= set(dims))
                return self._inherit(node, result, source, "selection is a subsequence without repeated positions",
                                     unique=unique, dims=dims)
        elif name == "aten.expand.default":
            # Preserve axis-local uniqueness where broadcasting does not duplicate
            # elements along that axis. Do not confuse this with global uniqueness.
            extra = len(result.shape) - len(source.shape)
            if extra >= 0:
                old = (1, ) * extra + source.shape
                if all(a == b or a == 1 for a, b in zip(old, result.shape)):
                    dims = {
                        d + extra
                        for d in source.unique_dims
                        if source.shape[d] > 1 and source.shape[d] == result.shape[d + extra]
                    }
                    repeated = any(a != b for a, b in zip(old, result.shape))
                    return self._inherit(node, result, source, "broadcast: only unchanged axes retain uniqueness",
                                         unique=source.unique_global and not repeated, dims=dims)
        elif name in ("aten.add.Tensor", "aten.add.Scalar", "aten.sub.Tensor", "aten.sub.Scalar", "aten.mul.Tensor",
                      "aten.mul.Scalar"):
            scalar = self._arg(node, 1, "other")
            if source.integer_range is None or not _constant_int(scalar) or result.shape != source.shape:
                return result
            lo, hi = source.integer_range.lower, source.integer_range.upper
            if name.startswith("aten.mul"):
                new = sorted((lo * scalar, hi * scalar))
                injective = scalar != 0
            else:
                alpha = self._arg(node, 2, "alpha", 1)
                if not _constant_int(alpha):
                    return result
                offset = scalar * alpha * (-1 if name.startswith("aten.sub") else 1)
                new, injective = (lo + offset, hi + offset), True
            bounds = IntegerRange(*new)
            if bounds.fits(result.dtype):
                inherited = self._inherit(node, result, source, "integer affine map proven nonoverflowing",
                                          unique=source.unique_global and injective,
                                          dims=source.unique_dims if injective else frozenset(), integer_range=bounds)
                return replace(inherited, known_zero=bounds.lower == bounds.upper == 0)
        return replace(result,
                       proof=source.proof + (name + ": no supported value-preservation proof; properties dropped", ),
                       scope=node.name)
