"""Experimental producer facts and graph-to-Triton specialization.

This is an opt-in, bounded prototype. It does not infer facts from tensor payloads
or accept user assertions of uniqueness as proof.
"""

from .analysis import AnalysisResult, FactsAnalyzer, IntegerRange, TensorFacts


def compile(*args, **kwargs):
    """Compile a supported graph; see the experimental backend for its contract."""
    from .backend import compile as compile_graph

    return compile_graph(*args, **kwargs)


__all__ = ["AnalysisResult", "FactsAnalyzer", "IntegerRange", "TensorFacts", "compile"]
