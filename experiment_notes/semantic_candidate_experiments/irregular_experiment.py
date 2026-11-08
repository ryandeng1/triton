#!/usr/bin/env python3
"""Pinned source-prototype screens; no native compiler optimization is claimed.

All timings are serialized GPU work. Run only when other experiment workers are
idle. Embedding includes reset and dtype conversion; segment reports device and
host-discovery costs separately; Mamba scan explicitly changes FP evaluation.
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import struct
import sys
import time
from pathlib import Path

import torch
import triton

import irregular_kernels as k


def cubin_text_hashes(data):
    """Compare executable ELF sections independently of debug/metadata bytes."""
    if data[:6] != b'\x7fELF\x02\x01':
        return None
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, names_index = struct.unpack_from('<HHH', data, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size)
                for i in range(count)]
    names_section = sections[names_index]
    names = data[names_section[4]:names_section[4]+names_section[5]]
    result = {}
    for section in sections:
        name = names[section[0]:].split(b'\0', 1)[0].decode()
        if name.startswith('.text'):
            result[name] = hashlib.sha256(data[section[4]:section[4]+section[5]]).hexdigest()
    return result


def save_kernel(kernel, path):
    path.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, value in kernel.asm.items():
        if not isinstance(value, (str, bytes)):
            continue
        data = value.encode() if isinstance(value, str) else value
        (path / ('kernel.' + name)).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    meta = (dict(kernel.metadata._asdict()) if hasattr(kernel.metadata, '_asdict')
            else dict(getattr(kernel.metadata, '__dict__', {})))
    for name in ('n_regs', 'n_spills'):
        meta[name] = getattr(kernel, name, None)
    result = {'metadata': meta, 'hashes': hashes,
              'cubin_text_hashes': cubin_text_hashes(kernel.asm['cubin']) if 'cubin' in kernel.asm else None}
    (path / 'metadata.json').write_text(json.dumps(result, default=str, indent=2))
    return result


def _paired_graph_times(functions, rounds, seed, repetitions=32):
    graphs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for fn in functions.values():
            for _ in range(3):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for name, fn in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repetitions):
                fn()
        graphs[name] = graph
    deadline = time.perf_counter() + .2
    while time.perf_counter() < deadline:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    rng = random.Random(seed)
    samples = {name: [] for name in functions}
    for _ in range(rounds):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / repetitions)
    return {'samples_us': samples,
            'median_us': {name: statistics.median(vals) for name, vals in samples.items()},
            'graph_repetitions': repetitions}


def paired_graph_times(functions, rounds, seed, repetitions=32, *, destinations, validate):
    """Use identical physical destination allocations for every timed variant.

    The caller's distinct correctness outputs are restored after graph timing.
    This controls address-dependent cache and partition effects, including when
    two variants compile to exactly the same machine code.
    """
    snapshots = [(mapping, dict(mapping)) for mapping in destinations]
    pointers = []
    for mapping, _ in snapshots:
        shared = next(iter(mapping.values()))
        pointers.append(shared.data_ptr())
        for name in mapping:
            mapping[name] = shared
    try:
        for name, fn in functions.items():
            fn()
            validate(name)
        result = _paired_graph_times(functions, rounds, seed, repetitions)
    finally:
        for mapping, original in snapshots:
            mapping.update(original)
    result['allocation_control'] = 'all variants use the same physical destination buffers'
    result['shared_destination_addresses'] = pointers
    result['common_buffer_validation'] = True
    return result


def check(actual, expected, *, atol=0, rtol=0):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {'max_abs': (actual.float()-expected.float()).abs().max().item(),
            'exact': torch.equal(actual, expected)}


def embedding_case(n, d, v, block, freq, seed, output, rounds):
    torch.manual_seed(seed)
    idx = torch.randperm(v, device='cuda')[:n].contiguous()
    # randperm establishes global uniqueness; this fact is supplied at the
    # producer boundary, never guessed from a race-free atomic implementation.
    assert torch.unique(idx).numel() == n
    grad = torch.randn((n, d), dtype=torch.bfloat16, device='cuda')
    magnitude = grad.float().abs()
    assert bool(torch.isfinite(magnitude).all())
    assert bool(((magnitude == 0) | (magnitude >= torch.finfo(torch.float32).tiny)).all())
    scratch = {name: torch.empty((v, d), dtype=torch.float32, device='cuda')
               for name in ('atomic', 'unique')}
    result = {name: torch.empty((v, d), dtype=grad.dtype, device='cuda') for name in scratch}
    counts = torch.empty(v, dtype=torch.int32, device='cuda')
    artifacts = {}
    def run(name):
        dst = scratch[name]
        dst.zero_()  # Untouched rows still require zeros in both variants.
        if name == 'atomic' and freq:
            counts.zero_()
            count_kernel = k._embedding_dense_backward_count_kernel[(triton.cdiv(n, 512),)](
                idx, counts, n, v, 0, BLOCK_N=512)
            kernel = k._embedding_dense_backward_kernel_scale_by_freq[(n, triton.cdiv(d, block))](
                grad, idx, counts, dst, v, 0, BLOCK_D=block, EMBED_DIM=d)
            artifacts['count'] = count_kernel
        elif name == 'atomic':
            kernel = k._embedding_dense_backward_kernel[(n, triton.cdiv(d, block))](
                grad, idx, dst, v, 0, BLOCK_D=block, EMBED_DIM=d)
        else:
            kernel = k.embedding_unique_kernel[(n, triton.cdiv(d, block))](
                grad, idx, dst, v, 0, BLOCK_D=block, EMBED_DIM=d)
        result[name].copy_(dst)
        artifacts[name] = kernel
    valid = idx != 0
    reference_fp32 = torch.zeros((v, d), device='cuda', dtype=torch.float32)
    reference_fp32.index_add_(0, idx[valid], grad[valid].float())
    reference = reference_fp32.to(grad.dtype)
    validation = {}
    for name in scratch:
        run(name)
        validation[name] = check(result[name], reference)
        validation[name]['fp32_scratch'] = check(scratch[name], reference_fp32)
        assert torch.equal(scratch[name].view(torch.int32), reference_fp32.view(torch.int32))
        validation[name]['fp32_bitwise'] = True
    validation['paired'] = check(result['atomic'], result['unique'])
    def validate_common(name):
        check(result[name], reference)
        check(scratch[name], reference_fp32)
        assert torch.equal(scratch[name].view(torch.int32), reference_fp32.view(torch.int32))
    times = paired_graph_times({name: lambda name=name: run(name) for name in scratch},
                               rounds, seed, destinations=(scratch, result), validate=validate_common)
    for name in scratch:
        result[name].fill_(float('nan'))
        run(name)
        check(result[name], reference)
    saved = {name: save_kernel(kernel, output / name) for name, kernel in artifacts.items()}
    return {'shape': [n, d, v], 'block_d': block, 'scale_by_freq': freq,
            'seed': seed, 'contract': 'globally unique randperm subset; finite ordinary BF16 gradients; fresh FP32 zero scratch; padding row 0',
            'scope': 'reset + original/specialized update + final BF16 copy; allocations excluded; source prototype',
            'validation': validation, 'timing': times, 'artifacts': saved,
            'speedup': times['median_us']['atomic']/times['median_us']['unique']}


def embedding_duplicate_control(seed):
    torch.manual_seed(seed)
    n, d, v = 257, 137, 97
    idx = torch.randint(0, v, (n,), device='cuda')
    idx[:2] = 3
    grad = torch.randn((n, d), device='cuda', dtype=torch.bfloat16)
    dst = torch.zeros((v, d), device='cuda', dtype=torch.float32)
    unique_proven = False  # The producer supplied no uniqueness contract.
    selected = k.embedding_unique_kernel if unique_proven else k._embedding_dense_backward_kernel
    selected[(n, triton.cdiv(d, 128))](grad, idx, dst, v, 0, BLOCK_D=128, EMBED_DIM=d)
    valid = idx != 0
    ref = torch.zeros_like(dst).index_add_(0, idx[valid], grad[valid].float())
    validation = check(dst, ref, atol=1e-5, rtol=1e-5)
    return {'shape': [n, d, v], 'duplicate_indices_present': True,
            'selected': 'original_atomic', 'validation': validation}


def embedding_boundary_control():
    """Padding/invalid destinations and ordinary FP32 boundary values."""
    v, d = 12, 13
    idx = torch.tensor([0, -1, v, 3, 6, 9, 10], device='cuda')
    limit = torch.finfo(torch.bfloat16)
    values = [0., -0., limit.tiny, -limit.tiny, limit.max, -limit.max,
              1., -1., .5, -.5, 2., -2., 3.]
    grad = torch.tensor(values, device='cuda', dtype=torch.bfloat16).expand(len(idx), -1).contiguous()
    valid = (idx > 0) & (idx < v)
    ref = torch.zeros((v, d), device='cuda', dtype=torch.float32)
    ref.index_add_(0, idx[valid], grad[valid].float())
    checks = {}
    for name, kernel in [('atomic', k._embedding_dense_backward_kernel),
                         ('unique', k.embedding_unique_kernel)]:
        dst = torch.zeros_like(ref)
        kernel[(len(idx), 1)](grad, idx, dst, v, 0, BLOCK_D=128, EMBED_DIM=d)
        checks[name] = check(dst, ref)
        assert torch.equal(dst.view(torch.int32), ref.view(torch.int32))
        checks[name]['fp32_bitwise'] = True
    return {'kind': 'embedding_boundary', 'shape': [len(idx), d, v],
            'valid_destinations_unique': True, 'invalid_and_padding_untouched': True,
            'tested': 'signed zero, min normal BF16, max finite BF16, ragged dimension',
            'validation': checks}


def segment_case(segments, length, seed, output, rounds):
    torch.manual_seed(seed)
    data = torch.randn(segments * length, device='cuda', dtype=torch.float32)
    lengths = torch.full((segments,), length, device='cuda', dtype=torch.int64)
    offsets = torch.arange(segments + 1, device='cuda', dtype=torch.int64) * length
    outs = {name: torch.empty(segments, device='cuda') for name in ('generic', 'uniform')}
    artifacts = {}
    def launch(name):
        if name == 'uniform':
            kernel = k._segment_reduce_uniform_inner1_forward_kernel[(triton.cdiv(segments, 32),)](
                data, outs[name], segments, segments, length, data.numel(),
                True, False, False, False, False, BLOCK_M=32,
                BLOCK_N=triton.next_power_of_2(length))
        else:
            kernel = k._segment_reduce_forward_kernel[(segments,)](
                data, offsets, outs[name], segments, 1, data.numel(),
                True, False, False, False, False, False, 0.0, BLOCK_SIZE=1024)
        artifacts[name] = kernel
    reference = data.double().reshape(segments, length).sum(1).float()
    validation = {}
    for name in outs:
        launch(name)
        validation[name] = check(outs[name], reference, atol=1e-4, rtol=1e-5)
    times = paired_graph_times({name: lambda name=name: launch(name) for name in outs},
                               rounds, seed, destinations=(outs,),
                               validate=lambda name: check(outs[name], reference, atol=1e-4, rtol=1e-5))
    # Match the actual upstream uniform early-return control flow. The generic
    # _validate_lengths path is NOT called on this path, even with unsafe=False.
    # Output is preallocated in all paths to isolate discovery/dispatch cost.
    def discovered(cold):
        if cold:
            k._UNIFORM_LENGTHS_CACHE.clear()
        k._check_reduce_and_dtype(data, 'sum')
        axis = k._wrap_axis(0, data.dim())
        k._check_index_tensor(data, lengths, 'lengths', axis)
        assert not k._is_unit_lengths(data, lengths, axis)
        found = k._get_uniform_segment_length(data.contiguous(), lengths, axis)
        assert found == length
        launch('uniform')
    def known():
        # torch.full producer establishes uniformity, positivity, total length.
        launch('uniform')
    host_samples = {name: [] for name in ('cold_discovery', 'cached_discovery', 'producer_known')}
    funcs = {'cold_discovery': lambda: discovered(True),
             'cached_discovery': lambda: discovered(False), 'producer_known': known}
    # The first randomized cached trial must also start with a populated cache.
    assert k._get_uniform_segment_length(data, lengths, 0) == length
    rng = random.Random(seed)
    for _ in range(rounds):
        names = list(funcs)
        rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            funcs[name]()
            torch.cuda.synchronize()
            host_samples[name].append((time.perf_counter_ns()-start)/1000)
    # Verify nonuniform lengths do not receive the specialized contract.
    bad = lengths.clone()
    bad[0] += 1
    bad[1] -= 1
    assert k._get_uniform_segment_length(data, bad, 0) is None
    for name in outs:
        outs[name].fill_(float('nan'))
        launch(name)
        check(outs[name], reference, atol=1e-4, rtol=1e-5)
    saved = {name: save_kernel(kernel, output / name) for name, kernel in artifacts.items()}
    return {'shape': [segments, length], 'seed': seed, 'validation': validation,
            'nonuniform_rejected': True, 'cached_discovery_preprimed': True,
            'device_timing': times,
            'host_samples_us': host_samples,
            'host_median_us': {name: statistics.median(vals) for name, vals in host_samples.items()},
            'artifacts': saved,
            'scope': 'Existing upstream algorithms; producer-known facts skip host discovery. Cold discovery includes _all_lengths_equal CPU copy; cached discovery is already free of that transfer. Matches actual uniform early-return flow (no _validate_lengths). Output allocation excluded equally. Generic device baseline excludes already-available offset construction; uniform already selected by upstream on eligible data. No new compiler pass or new uniform algorithm.'}


def mamba_case(batch, chunks, heads, dim, seed, output, rounds, has_seq=False, full_configs=True):
    torch.manual_seed(seed)
    states = torch.randn((batch, chunks, heads, dim), device='cuda') * .1
    da = -torch.rand((batch, chunks, heads), device='cuda') * .2
    initial = torch.randn((batch, heads, dim), device='cuda') * .1
    seq = (torch.arange(chunks, device='cuda', dtype=torch.int32)[None, :] // max(1, chunks // 4)).expand(batch, -1).contiguous()
    configs = {f'sequential_{block}': ('sequential', block)
               for block in ([64, 128, 256, 512, 1024, 2048] if full_configs else [128])}
    configs.update({f'scan_{block}': ('scan', block)
                    for block in ([16, 32, 64] if full_configs else [16])})
    outs = {name: torch.empty_like(states) for name in configs}
    finals = {name: torch.empty_like(initial) for name in outs}
    artifacts = {}
    def launch(name):
        kind, block = configs[name]
        if kind == 'sequential':
            kernel = k._state_passing_fwd_kernel[(triton.cdiv(dim, block), batch, heads)](
                states, outs[name], finals[name], da, initial, seq,
                dim, chunks, chunks, 1, *states.stride(), *outs[name].stride(),
                *finals[name].stride(), da.stride(0), da.stride(1), da.stride(2),
                *initial.stride(), *seq.stride(), True, has_seq, BLOCK_SIZE=block)
        else:
            kernel = k.state_passing_scan[(triton.cdiv(dim, block), batch * heads)](
                states, outs[name], finals[name], da, initial, seq,
                dim, chunks, heads, block, triton.next_power_of_2(chunks), True, has_seq,
                num_warps=4)
        artifacts[name] = kernel
    ref = torch.empty_like(states, dtype=torch.float64)
    acc = initial.double()
    scale = da.double().exp()
    for c in range(chunks):
        ref[:, c] = acc
        a = scale[:, c]
        if has_seq:
            prev = seq[:, c-1] if c else torch.zeros(batch, dtype=seq.dtype, device='cuda')
            a = torch.where((seq[:, c] == prev)[:, None], a, 0.0)
        acc = a[:, :, None] * acc + states[:, c].double()
    validation = {}
    for name in outs:
        launch(name)
        validation[name] = {'states': check(outs[name], ref.float(), atol=2e-5, rtol=2e-4),
                            'final': check(finals[name], acc.float(), atol=2e-5, rtol=2e-4)}
    def validate_common(name):
        check(outs[name], ref.float(), atol=2e-5, rtol=2e-4)
        check(finals[name], acc.float(), atol=2e-5, rtol=2e-4)
    times = paired_graph_times({name: lambda name=name: launch(name) for name in outs},
                               rounds, seed, destinations=(outs, finals), validate=validate_common)
    best = {kind: min((name for name in configs if configs[name][0] == kind),
                     key=lambda name: times['median_us'][name]) for kind in ('sequential', 'scan')}
    # Timing deliberately reused one allocation. Refresh the separately owned
    # correctness outputs before comparing the selected variants.
    launch(best['sequential'])
    launch(best['scan'])
    validation['paired'] = {'states': check(outs[best['scan']], outs[best['sequential']], atol=2e-5, rtol=2e-4),
                            'final': check(finals[best['scan']], finals[best['sequential']], atol=2e-5, rtol=2e-4)}
    for name in outs:
        outs[name].fill_(float('nan'))
        finals[name].fill_(float('nan'))
        launch(name)
        check(outs[name], ref.float(), atol=2e-5, rtol=2e-4)
        check(finals[name], acc.float(), atol=2e-5, rtol=2e-4)
    saved = {name: save_kernel(kernel, output / name) for name, kernel in artifacts.items()}
    return {'shape': [batch, chunks, heads, dim], 'has_seq_resets': has_seq,
            'seed': seed, 'validation': validation, 'timing': times, 'artifacts': saved,
            'best_configs': best, 'configs': configs,
            'speedup': times['median_us'][best['sequential']]/times['median_us'][best['scan']],
            'scope': 'Source algorithm prototype: real-arithmetic affine associativity; changed FP order. Domain dA in [-0.2,0], random states scaled .1, FP32 outputs. Selected tolerances are experimental, not a universal Mamba contract. Screen/confirm sweeps all six upstream original BLOCK_SIZE configs and scan widths16/32/64 in shuffled paired timing; smoke uses128 versus16. All intermediate and final states retained.'}


def tbe_case(runs, run_length, dim, seed, output, rounds, compiler_contract=False,
             schedule='upstream'):
    """Actual TorchRec static persistent short-run kernel, one FP32 table.

    Inputs are the outputs of sorting/run classification. That preprocessing is
    common to both variants and excluded. Reset copies are included in timing.
    No CLC, stochastic rounding, variable batch embedding, or split rows.
    """
    torch.manual_seed(seed)
    batch, rows, guard = 2048, runs * 2, 32
    device = 'cuda'
    dout = torch.randn((batch, dim), device=device) * .1
    infos = torch.randint(batch, (runs * run_length,), device=device, dtype=torch.int64)
    row_ids = torch.arange(0, rows, 2, device=device, dtype=torch.int64)
    cumulative = torch.arange(runs + 1, device=device, dtype=torch.int64) * run_length
    short_ids = torch.randperm(runs, device=device)
    def tensor(values):
        return torch.tensor(values, device=device, dtype=torch.int64)
    table_offsets, dims, embedding_offsets = tensor([0]), tensor([dim]), tensor([0, dim])
    feature_map, hash_cumsum, rows_cumsum = tensor([0]), tensor([0, rows]), tensor([0, rows])
    num_runs, dummy = tensor([runs]), tensor([0])
    weight_initial = torch.randn((rows, dim), device=device) * .1
    momentum_initial = torch.rand(rows, device=device) * .1
    names = ['original', 'contract'] if compiler_contract else ['original']
    storages, weights, momenta = {}, {}, {}
    for name in names:
        ws = torch.full((rows * dim + 2 * guard,), 37., device=device)
        ms = torch.full((rows + 2 * guard,), 37., device=device)
        storages[name] = (ws, ms)
        weights[name] = ws[guard:-guard].view(rows, dim)
        momenta[name] = ms[guard:-guard]
    # Exact pinned Blackwell config: num_warps=1, short_run_programs=32*24576,
    # wrapper caps at the input entry count when dimension bucketing is unused.
    # Also expose a separately labelled static persistent schedule diagnostic.
    grid = ((min(32 * 24576, runs * run_length) if schedule == 'upstream'
             else min(runs, torch.cuda.get_device_properties(0).multi_processor_count)),)
    learning_rate, eps, artifacts, compiled_contract = .1, 1e-8, {}, None
    def launch(name):
        nonlocal compiled_contract
        weights[name].copy_(weight_initial)
        momenta[name].copy_(momentum_initial)
        bound = (dout, (weights[name],), (0,), (), infos, row_ids, cumulative,
                 short_ids, table_offsets, dims, embedding_offsets, feature_map,
                 hash_cumsum, momenta[name], rows_cumsum, num_runs, dummy, dummy,
                 dim, batch, learning_rate, eps, 1, triton.next_power_of_2(dim),
                 11, 2047, False, True, False, 0, False, 2)
        if name == 'original':
            kernel = k.triton_tbe_backward_short_run_unweighted[grid](*bound, num_warps=1)
        else:
            if compiled_contract is None:
                from triton.compiler import ASTSource
                baseline = k.triton_tbe_backward_short_run_unweighted.warmup(
                    *bound, grid=grid, num_warps=1)
                source = baseline.src
                attrs = {key: list(value) for key, value in source.attrs.items()}
                # Immutable contents through every alias, established by the
                # host producer and separate weight/momentum allocations.
                for arg in list(range(4, 13)) + [14, 15]:
                    attrs.setdefault((arg,), []).append(('tt.experimental_immutable', 1))
                annotated = ASTSource(source.fn, dict(source.signature),
                    constexprs=dict(source.constants), attrs=attrs)
                compiled_contract = triton.compile(annotated,
                    target=baseline.metadata.target, options={'num_warps': 1})
            kernel = compiled_contract
            kernel[(grid[0], 1, 1)](*bound)
        artifacts[name] = kernel
    # Independent FP64 oracle, preserving untouched rows and optimizer state.
    gradients = dout[infos].double().reshape(runs, run_length, dim).sum(1)
    m_ref = momentum_initial.double().clone()
    m_ref[row_ids] += gradients.square().mean(1)
    w_ref = weight_initial.double().clone()
    w_ref[row_ids] -= learning_rate * gradients / (m_ref[row_ids].sqrt()[:, None] + eps)
    validation = {}
    def validate(name):
        for pair in storages.values():
            for storage in pair:
                assert bool((storage[:guard] == 37.).all())
                assert bool((storage[-guard:] == 37.).all())
        return {'weights': check(weights[name], w_ref.float(), atol=2e-6, rtol=2e-5),
                'momentum': check(momenta[name], m_ref.float(), atol=2e-6, rtol=2e-5),
                'guards': True,
                'untouched_rows_exact': bool(torch.equal(weights[name][1::2], weight_initial[1::2]))}
    for name in names:
        launch(name)
        validation[name] = validate(name)
    if compiler_contract:
        validation['paired'] = {'weights': check(weights['original'], weights['contract']),
                                'momentum': check(momenta['original'], momenta['contract'])}
    times = paired_graph_times({name: lambda name=name: launch(name) for name in names},
                               rounds, seed, destinations=(weights, momenta), validate=validate)
    for name in names:
        weights[name].fill_(float('nan'))
        momenta[name].fill_(float('nan'))
        launch(name)
        validate(name)
    saved = {name: save_kernel(kernel, output / name) for name, kernel in artifacts.items()}
    case = {'shape': [runs, run_length, dim], 'seed': seed,
            'grid': list(grid), 'num_warps': 1, 'schedule': schedule,
            'optimizer': 'rowwise_adagrad',
            'validation': validation, 'timing': times, 'artifacts': saved,
            'contract': 'run metadata and table descriptors immutable through all aliases during invocation',
            'scope': 'Pinned TorchRec original short-run unweighted kernel, USE_CLC=False, one FP32 table, rowwise Adagrad, precomputed classified runs. Upstream schedule follows pinned Blackwell grid/warp config; persistent schedule explicitly overrides grid to SM count as a diagnostic. Reset weights and momentum included; preprocessing excluded. No source algorithm or protocol rewrite.'}
    if compiler_contract:
        case['speedup'] = times['median_us']['original'] / times['median_us']['contract']
        case['ptx_identical'] = saved['original']['hashes']['ptx'] == saved['contract']['hashes']['ptx']
        case['cubin_text_identical'] = (saved['original']['cubin_text_hashes'] is not None and
            saved['original']['cubin_text_hashes'] == saved['contract']['cubin_text_hashes'])
    return case


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=['embedding', 'segment', 'mamba', 'tbe', 'all'], default='all')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['screen', 'confirm', 'smoke'], default='screen')
    p.add_argument('--seeds', type=int, default=None)
    p.add_argument('--compiler-contract', action='store_true',
                   help='TBE original plus tt.experimental_immutable metadata contract; requires local experimental compiler')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = args.seeds if args.seeds is not None else (3 if args.mode == 'confirm' else 1)
    rounds = 15 if args.mode == 'confirm' else (3 if args.mode == 'smoke' else 7)
    here = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
                     for name in ['irregular_experiment.py', 'irregular_kernels.py',
                                  'vendor/irregular/provenance.json']}
    for name in source_hashes:
        snapshot = args.output/'run_sources'/name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((here/name).read_bytes())
    from triton._C import libtriton
    with Path(libtriton.__file__).open('rb') as stream:
        compiler_binary_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'torch': torch.__version__, 'triton': triton.__version__,
              'triton_path': triton.__file__, 'gpu': torch.cuda.get_device_name(),
              'cuda': torch.version.cuda, 'source_sha256': source_hashes,
              'command': [sys.executable, *sys.argv], 'cwd': os.getcwd(),
              'pythonpath': os.environ.get('PYTHONPATH'),
              'compiler_binary': libtriton.__file__, 'compiler_binary_sha256': compiler_binary_hash,
              'mode': args.mode, 'rounds': rounds, 'cases': [], 'controls': []}
    out = args.output / 'results.json'
    def record(case):
        report['cases'].append(case)
        out.write_text(json.dumps(report, default=str, indent=2))
        print(json.dumps({key: case[key] for key in ('suite', 'shape', 'seed', 'speedup') if key in case}), flush=True)
    for seed in range(seeds):
        if args.suite in ('embedding', 'all'):
            report['controls'].append(embedding_duplicate_control(seed))
            if seed == 0:
                report['controls'].append(embedding_boundary_control())
            shapes = [(4096, 128, 8192), (16384, 256, 32768), (8192, 1024, 16384)]
            if args.mode == 'smoke': shapes = [(257, 137, 512)]
            for n, d, v in shapes:
                for block in ([128] if args.mode == 'smoke' else [128, 256]):
                    for freq in (False, True):
                        case = embedding_case(n, d, v, block, freq, seed,
                            args.output/f'embedding_{n}_{d}_{block}_{freq}_{seed}', rounds)
                        record(dict(suite='embedding', **case))
        if args.suite in ('segment', 'all'):
            shapes = [(65536, 16), (8192, 128)]
            if args.mode == 'smoke': shapes = shapes[:1]
            for segments, length in shapes:
                case = segment_case(segments, length, seed,
                    args.output/f'segment_{segments}_{length}_{seed}', rounds)
                record(dict(suite='segment', **case))
        if args.suite in ('mamba', 'all'):
            if seed == 0:
                for resets in (False, True):
                    boundary = mamba_case(1, 65, 3, 137, seed,
                        args.output/f'mamba_boundary_{resets}', 3, resets, full_configs=False)
                    report['controls'].append(dict(kind='mamba_boundary',
                        timing_use='diagnostic_only', **boundary))
            # Original dim flattens headdim*dstate. The 128-dimension cases
            # probe low parallelism; 4096/8192 represent full SSM state rows.
            shapes = [(1, 64, 4, 128), (1, 256, 4, 128), (4, 256, 16, 128),
                      (1, 64, 32, 4096), (1, 256, 32, 8192)]
            if args.mode == 'smoke': shapes = shapes[:1]
            for b, c, h, d in shapes:
                for resets in (False, True):
                    case = mamba_case(b, c, h, d, seed,
                        args.output/f'mamba_{b}_{c}_{h}_{d}_{resets}_{seed}', rounds, resets,
                        full_configs=args.mode != 'smoke')
                    record(dict(suite='mamba', **case))
        if args.suite in ('tbe', 'all'):
            shapes = [(512, 1, 64), (4096, 8, 128), (8192, 32, 256)]
            if args.mode == 'smoke': shapes = [(257, 5, 137)]
            for runs, length, dim in shapes:
                for schedule in ('upstream', 'persistent'):
                    case = tbe_case(runs, length, dim, seed,
                        args.output/f'tbe_{runs}_{length}_{dim}_{schedule}_{seed}', rounds,
                        args.compiler_contract, schedule)
                    record(dict(suite='tbe', **case))
    report['status'] = 'completed'
    out.write_text(json.dumps(report, default=str, indent=2))


if __name__ == '__main__':
    main()
