# Semantic optimization experiments — start here

Last consolidated: 2026-11-07.

Read **[findings_overview.md](findings_overview.md)** for the complete narrative:
what we tried, measured gains and negative results, how ownership and other
semantic properties enter the proofs, numerical caveats, and the proposed next
compiler step. This README and that overview are the two entry files; the
existing reports below retain the detailed evidence and reproduction commands.

The strongest new source-prototype result is LayerNorm backward: approximately
1.9–2.0× and 1.78× gains on two BF16 shapes against a tuned, numerically eligible
baseline from Tritonbench's registered tutorial-derived provider, with FP32
partial buffers on both sides. An automatic compiler pass for this rewrite is
not implemented. The earlier native scheduling/packing experiments did not
establish a gain over the best tested unchanged real-kernel configurations.

| Find this | Read this file |
| --- | --- |
| All findings, limitations, and recommended next work | [Consolidated overview](findings_overview.md) |
| LayerNorm ownership explanation, actual Tritonbench registration, BF16 results and reference corrections | [LayerNorm findings](layernorm_ownership_findings_11_07.md) |
| LayerNorm legality conditions, serialization, aliasing, launch and scratch requirements | [LayerNorm protocol](layernorm_ownership_protocol.md) |
| FlashAttention-4/Gluon comparison, identity rescaling, typed storage reuse and accuracy limits | [Attention findings](semantic_attention_findings_11_07.md) |
| Attention ownership, synchronization, and asynchronous lifetime reasoning | [Attention protocol](semantic_attention_protocol.md) |
| Native compiler experiments on GEMM/attention, including negative results | [November 7 compiler experiments](writeup_11_07.md) |
| Earlier controlled scheduling/packing work, before the real-kernel experiments | [November 4 historical writeup](writeup_11_04.md) |

Machine-readable summaries: [LayerNorm](layernorm_ownership_summary_11_07.json),
[attention](semantic_attention_summary_11_07.json),
[November 7 compiler experiments](summary_11_07.json), and
[November 4 historical results](summary_11_04.json).

Raw benchmark reports, compiler artifacts, audits, and logs are under
[`layernorm_ownership_results/`](layernorm_ownership_results/),
[`real_kernel_results/`](real_kernel_results/), and
[`semantic_attention_results/`](semantic_attention_results/).
Use the selected report links in the findings files: some preserved runs failed,
were diagnostic only, or were superseded by corrected validation. Do not treat
every JSON file as a successful performance result.

The November 4 summary contains absolute paths to raw artifacts that are absent
from this workspace. Its writeup and JSON remain historical records; those
figures were not freshly verified from the missing raw files during this
consolidation. The November 7 findings link to available evidence.
