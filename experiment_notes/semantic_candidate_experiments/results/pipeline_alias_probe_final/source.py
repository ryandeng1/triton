import argparse
import gzip
import hashlib
import json
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def shift_rows(X, T, BLOCK: tl.constexpr, STAGES: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    for t in tl.range(0, T, num_stages=STAGES):
        value = tl.load(X + t * BLOCK + cols)
        tl.store(X + (t + 1) * BLOCK + cols, value + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', default='/tmp/pipeline_alias_probe')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    kernels = {}
    for stages in (1, 2, 3):
        source = ASTSource(shift_rows, signature={'X': '*fp32', 'T': 'i32'},
                           constexprs={'BLOCK': 128, 'STAGES': stages},
                           attrs={(0,): [('tt.divisibility', 16)]})
        k = triton.compile(source, target=GPUTarget('cuda', 103, 32),
                           options={'num_warps': 4, 'num_stages': 1})
        kernels[stages] = k
        record = dict(stages=stages, shared=k.metadata.shared,
                      async_copies=k.asm['ttgir'].count('ttg.async_copy_global_to_local'),
                      ptx_sha256=hashlib.sha256(k.asm['ptx'].encode()).hexdigest())
        for extension in ('ttir', 'ttgir', 'ptx'):
            with gzip.open(out / f'stages{stages}.{extension}.gz', 'wt') as f:
                f.write(k.asm[extension])
        records.append(record)
    if args.run:
        import torch
        # One CTA, one vector per iteration, same tensor coordinates carry each
        # dependence. The entire allocation is initialized and in bounds.
        # No atomics, hidden hardware IDs, concurrent CTAs, or external observers.
        base = torch.arange(128, dtype=torch.float32, device='cuda')
        for record in records:
            checks = []
            for trips in (0, 1, 2, 4, 17):
                initial = torch.full((trips + 1, 128), -1000., device='cuda')
                initial[0].copy_(base)
                expected = base[None, :] + torch.arange(trips + 1, device='cuda')[:, None]
                x = initial.clone()
                kernels[record['stages']][(1, 1, 1)](x, trips)
                torch.cuda.synchronize()
                checks.append(dict(trips=trips, exact=torch.equal(x, expected),
                                   max_error=(x - expected).abs().max().item(),
                                   first_column=x[:, 0].tolist()))
            record['checks'] = checks
    (out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
