# LLVM argument-contract feasibility probe

Adding audited LLVM argument attributes changed **none of nine executable kernels**. Every on/off pair has identical PTX, complete cubin bytes, SASS including scheduling annotations, and raw ELF `.text` bytes. Optimized LLVM instruction bodies are identical; only argument attributes differ. Consequently no GPU timing was performed.

| Pinned AITER case | Fixed `(BV, warps, stages)` | Source variants | Attribute pairs with changed code |
|---|---|---|---:|
| B2, T500, H4/HV4, D60 | `(16, 1, 2)` | Original, native immutable LICM, manual hoist | 0/3 |
| Same T500 case | `(32, 2, 2)` | Same three | 0/3 |
| B4, T2048, H4/HV8, D64 | `(8, 1, 3)` | Same three | 0/3 |

T500 includes the retained manual-selected and native-selected configurations. T2048 is a retained negative case. This is a new compile experiment using the current compiler, targeting `cuda:103`; it does not reuse the old binaries or claim a new performance measurement. The complete source, compiler-library and script hashes are in [summary.json](summary.json). Current `enable_contract_load_reuse=True` is uniform across all builds; all backend options are recorded.

## Where the attributes enter

The Python stage hook intercepts LLVM-dialect MLIR immediately before `llvm.to_module`. Both control and treatment roundtrip through the same parser. Treatment changes only the audited function's argument attributes. Triton's earlier passes and the normal NVIDIA LLVM O3 pipeline are unchanged. Paired TTIR, TTGIR and pre-hook LLVM-dialect MLIR hashes match exactly. The emitted LLVM `readonly`, `noalias` and `captures(none)` attributes survive optimization.

Every global load in these kernels is a side-effecting inline-PTX call: T500 has 17 such static calls and T2048 has 13. There are **zero ordinary LLVM global loads and zero masked-load intrinsics**. The 8–16 ordinary LLVM loads are in shared address space 3. These input facts therefore do not expose the global memory operations to LLVM load LICM/GVN. The result bounds this attribute-only translation; it does not rule out changing the LLVM representation of global memory operations in a separate experiment.

## Contract and escape audit

This experiment applies a deliberately bounded, source-audited contract:

- `readonly` goes on the nine input pointer arguments. The pinned source has only two stores, through `o` and `h0_source`, with floating-point result values.
- `captures(none)` goes on the eleven explicit pointer arguments. The source has no pointer/address stores, pointer returns, address-to-integer escape, external helper calls, or user inline assembly. Lowering passes their derived addresses to inline PTX loads/stores; the target instructions use those values as addresses and do not retain them. This is a source/target audit, not an inference from immutability or an assertion that LLVM understands the assembly.
- `noalias` goes **only on the two modified allocation roots**, `o` and `h0_source`. Its precondition is that their complete backing storages are mutually disjoint and disjoint from every input for the invocation, with no concurrent conflicting access. Equal-group input arguments do not receive `noalias`.
- No `invariant.load` metadata or invariant-region intrinsic is emitted. Invocation-scoped immutable inputs may change between calls.

The retained GPU harness allocates the corresponding tensors separately. This new probe additionally exercises complete-storage interval checks using fresh **CPU** tensors with the same shapes, including output guard storage; [summary.json](summary.json) records every checked pair and storage size. These CPU checks demonstrate the allocation precondition and checker. They are not GPU runtime correctness tests. The attributes are sound only for launches satisfying the stated preconditions.

## Separate inactive-value control

The manual T500 source was copied twice with identical function names and line numbers. The only edit removes `other=0.0` from the two parameter loads guarded by `T>0`; the reverse edit reproduces the copied source exactly. Their results are unused when `T<=0`, so the edit does not change observable computation.

At both BV16 and BV32, the zero and undefined variants have **identical raw machine `.text` bytes and SASS**. PTX does change, including initialization and virtual-register details; `ptxas` produces the same machine instructions. The complete cubins differ because source/debug metadata differs. Native and manual machine code still differs at both configurations. This isolated manual-side control does not explain or resolve the historical native/manual T500 performance gap. It does not establish that every possible native-side fill-value change is irrelevant.

## Evidence and reproduction

Each build directory contains a `report.json` and `artifacts.tar.xz`. The archive retains original and annotated LLVM-dialect MLIR, LLVM before/after O3, TTIR, TTGIR, PTX, cubin and SASS. Per-file hashes and raw machine-code hashes are recorded. The source copy retains the AITER license and pinned revision.

```sh
/home/ryandeng/tpaa/.venv-semantic/bin/python experiment_notes/semantic_candidate_experiments/llvm_contract_probe.py
```

The script is CPU-only, requires the installed NVIDIA assembler/disassembler, and explicitly targets `cuda:103`. It makes no native compiler changes and initializes no CUDA context. It verifies parser validity, paired pre-LLVM identity, attribute-only LLVM differences, source-edit bounds, and final machine-code equality.
