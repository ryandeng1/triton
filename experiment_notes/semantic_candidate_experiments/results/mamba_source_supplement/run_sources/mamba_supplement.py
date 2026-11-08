#!/usr/bin/env python3
"""Broader scan tiling screen to check the previous search-space boundary.

The kernel source, numeric domain, references, and complete-output checks are
unchanged. Copied case driver extends only the configuration search and records
per-config compile resource failures. It imports timing helpers from the exact
recorded irregular_experiment.py snapshot.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
import triton
from triton.runtime.errors import OutOfResources, PTXASError
import irregular_kernels as k
from irregular_experiment import check, paired_graph_times, save_kernel

def mamba_supplement_case(batch, chunks, heads, dim, seed, output, rounds, has_seq=False):
    torch.manual_seed(seed)
    states = torch.randn((batch, chunks, heads, dim), device='cuda') * .1
    da = -torch.rand((batch, chunks, heads), device='cuda') * .2
    initial = torch.randn((batch, heads, dim), device='cuda') * .1
    seq = (torch.arange(chunks, device='cuda', dtype=torch.int32)[None, :] // max(1, chunks // 4)).expand(batch, -1).contiguous()
    configs = {f'sequential_{block}': ('sequential', block, 4)
               for block in [64, 128, 256, 512, 1024, 2048]}
    configs.update({f'scan_{block}_w{warps}': ('scan', block, warps)
                    for block, warps in [(16, 4), (32, 4), (64, 4), (128, 4), (256, 4),
                                         (64, 8), (128, 8), (256, 8)]})
    requested_configs = dict(configs)
    resource_failures = {}
    outs = {name: torch.empty_like(states) for name in configs}
    finals = {name: torch.empty_like(initial) for name in outs}
    artifacts = {}
    def launch(name):
        kind, block, warps = configs[name]
        if kind == 'sequential':
            kernel = k._state_passing_fwd_kernel[(triton.cdiv(dim, block), batch, heads)](
                states, outs[name], finals[name], da, initial, seq,
                dim, chunks, chunks, 1, *states.stride(), *outs[name].stride(),
                *finals[name].stride(), da.stride(0), da.stride(1), da.stride(2),
                *initial.stride(), *seq.stride(), True, has_seq, BLOCK_SIZE=block, num_warps=warps)
        else:
            kernel = k.state_passing_scan[(triton.cdiv(dim, block), batch * heads)](
                states, outs[name], finals[name], da, initial, seq,
                dim, chunks, heads, block, triton.next_power_of_2(chunks), True, has_seq,
                num_warps=warps)
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
    for name in list(outs):
        try:
            launch(name)
        except (OutOfResources, PTXASError) as error:
            resource_failures[name] = {'type': type(error).__name__, 'error': str(error)}
            print(json.dumps({'resource_failure': name, **resource_failures[name]}), flush=True)
            del outs[name], finals[name], configs[name]
            continue
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
            'best_configs': best, 'configs': configs, 'requested_configs': requested_configs,
            'resource_failures': resource_failures,
            'speedup': times['median_us'][best['sequential']]/times['median_us'][best['scan']],
            'scope': 'Source algorithm prototype: real-arithmetic affine associativity; changed FP order. Domain dA in [-0.2,0], random states scaled .1, FP32 outputs. Selected tolerances are experimental, not a universal Mamba contract. Supplemental bounded sweep: all six upstream original BLOCK_SIZE configs, scan widths16/32/64/128/256 at4 warps and64/128/256 at8 warps, same physical timing outputs. Configuration resource failures recorded explicitly. All intermediate and final states retained.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    hashes = {}
    for name in ['mamba_supplement.py', 'irregular_experiment.py', 'irregular_kernels.py',
                 'vendor/irregular/provenance.json']:
        data = (here/name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        snapshot = args.output/'run_sources'/name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(data)
    from triton._C import libtriton
    with Path(libtriton.__file__).open('rb') as stream:
        compiler_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'torch': torch.__version__, 'triton': triton.__version__,
        'triton_path': triton.__file__, 'gpu': torch.cuda.get_device_name(),
        'cuda': torch.version.cuda, 'source_sha256': hashes,
        'command': [sys.executable, *sys.argv], 'cwd': os.getcwd(),
        'pythonpath': os.environ.get('PYTHONPATH'), 'compiler_binary': libtriton.__file__,
        'compiler_binary_sha256': compiler_hash, 'rounds': args.rounds,
        'scope': 'Supplemental bounded Mamba tiling sweep, actual kernel bodies unchanged',
        'cases': []}
    for seed in range(args.seeds):
        for b, c, h, d in [(1, 64, 32, 4096), (1, 256, 32, 8192)]:
            for resets in [False, True]:
                case = mamba_supplement_case(b, c, h, d, seed,
                    args.output/f'mamba_{b}_{c}_{h}_{d}_{resets}_{seed}', args.rounds, resets)
                report['cases'].append(dict(suite='mamba', **case))
                (args.output/'results.json').write_text(json.dumps(report, indent=2, default=str))
                print(json.dumps({key:case[key] for key in ['shape','seed','has_seq_resets','best_configs','speedup','resource_failures']}),flush=True)
    report['status'] = 'completed'
    (args.output/'results.json').write_text(json.dumps(report, indent=2, default=str))

if __name__ == '__main__':
    main()
