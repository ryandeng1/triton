#!/usr/bin/env python3
"""Source-only recurrent loop-pipeline oracle against a frozen native library.

Reuses the retained independent FP64 oracle, full-state validation and reset-inclusive
CUDA graph timing. The only kernel-body edit is range(T) -> tl.range(T,num_stages=2).
This experiment does not claim to implement contract-licensed automatic pipelining.
"""
from __future__ import annotations
import argparse
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

import recurrent_licm_experiment as prior

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/recurrent_pipeline_final"
OLD = ROOT / "results/recurrent_licm_final"
VARIANTS = ("original", "alias", "pipeline", "pipeline_alias")
FIXED = {"upstream500": (32, 2, 2), "upstream1024": (8, 1, 3), "decode16": (16, 1, 2)}
CASES = [next(c for c in prior.CASES if c['id'] == name) for name in FIXED]


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    frozen = ast.parse((OLD / "kernels.py").read_text())
    original = next(n for n in frozen.body if isinstance(n, ast.FunctionDef) and n.name == prior.NAME)
    source_path = Path('/tmp/semantic-survey-cache') / ('AITER-' + prior.SHA) / prior.KERNEL_PATH
    upstream = next(n for n in ast.parse(source_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == prior.NAME)
    assert prior.body(original) == prior.body(upstream), 'Original body differs from pinned upstream'
    pipeline = copy.deepcopy(original)
    pipeline.name = "recurrent_loop_pipeline"
    loops = [n for n in ast.walk(pipeline) if isinstance(n, ast.For)]
    assert len(loops) == 1 and ast.unparse(loops[0].iter) == 'range(T)'
    old_iter = copy.deepcopy(loops[0].iter)
    loops[0].iter = ast.parse('tl.range(T, num_stages=2)', mode='eval').body
    tree = ast.fix_missing_locations(ast.Module(body=[original, pipeline], type_ignores=[]))
    path = output / 'kernels.py'
    path.write_text('# SPDX-License-Identifier: MIT\n# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.\n# See AITer_LICENSE. Pinned AITER: ' + prior.SHA + '\nimport triton\nimport triton.language as tl\n\n' + ast.unparse(tree) + '\n')
    restored = copy.deepcopy(pipeline)
    next(n for n in ast.walk(restored) if isinstance(n, ast.For)).iter = old_iter
    assert prior.body(restored) == prior.body(original), 'Unexpected source edit'
    (output / 'AITer_LICENSE').write_bytes((OLD / 'AITer_LICENSE').read_bytes())
    audit = dict(repository=prior.REPO, revision=prior.SHA, upstream_source=prior.KERNEL_PATH,
                 upstream_source_sha256=prior.sha(source_path), retained_source_sha256=prior.sha(OLD / 'kernels.py'),
                 source_sha256=prior.sha(path), baseline_body_ast_equal_to_upstream=True,
                 pipeline_inverse_edit_body_ast_equal=True, changes=['range(T) -> tl.range(T, num_stages=2)'],
                 launch_num_stages_distinct_from_loop_attribute=True, fixed_launch_configurations=FIXED,
                 enable_contract_load_reuse=False,
                 isolation_reason='Current alias contracts also enable redundant-load reuse; disable uniformly to isolate contract LICM',
                 retune_configurations=prior.CONFIGS,
                 reused_oracle_and_timing_harness_sha256=prior.sha(prior.__file__),
                 scope='Every timestep output, indexed final-state writes, identical state reset before each invocation; no complete-model claim')
    prior.write(output / 'provenance.json', audit)
    return path


def metrics(kernel):
    kernel._init_handles()
    ptx, ir = kernel.asm['ptx'], kernel.asm['ttgir']
    return dict(registers=kernel.n_regs, spills=kernel.n_spills, shared=kernel.metadata.shared,
                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest(),
                ttgir_async_copy_count=ir.count('ttg.async_copy_global_to_local'),
                ttgir_async_wait_count=ir.count('ttg.async_wait'),
                ptx_cp_async_count=sum('cp.async.' in line for line in ptx.splitlines()),
                ptx_local_load_store_count=sum(('ld.local.' in line or 'st.local.' in line) for line in ptx.splitlines()))


def compile_case(case, d, module, config, load_reuse=False):
    import torch
    import triton
    from triton.compiler import ASTSource
    BV, warps, stages = config
    args = prior.arguments(case, d, BV)
    grid = (1, triton.cdiv(case['V'], BV), d['N'] * case['HV'])
    options = dict(num_warps=warps, num_stages=stages, enable_indexed_update_specialization=False,
                   enable_private_atomic_relaxation=False, enable_contract_load_reuse=load_reuse)
    # Input allocations may alias each other; neither writable allocation may
    # overlap any input. Full backing storage is checked, not just base pointers.
    for wi in (9, 10):
        ws = args[wi].untyped_storage()
        for ri, value in enumerate(args):
            if ri in (9, 10) or not isinstance(value, torch.Tensor):
                continue
            rs = value.untyped_storage()
            assert ws.data_ptr() + ws.nbytes() <= rs.data_ptr() or rs.data_ptr() + rs.nbytes() <= ws.data_ptr()
    kernels, equivalent, immutable_kernels = {}, {}, {}
    for prefix, fn in (('', getattr(module, prior.NAME)), ('pipeline', module.recurrent_loop_pipeline)):
        original = fn.warmup(*args, grid=grid, **options)
        attrs = copy.deepcopy(original.src.attrs)
        assert not any('tt.experimental_' in str(x) for x in attrs.values())
        alias_attrs = copy.deepcopy(attrs)
        for i, value in enumerate(args):
            if isinstance(value, torch.Tensor):
                alias_attrs.setdefault((i,), []).append(('tt.experimental_alias_group', 1 if i in (9, 10) else 0))
        def contracted(contract_attrs):
            return triton.compile(ASTSource(fn, original.src.signature, constexprs=original.src.constants, attrs=contract_attrs),
                                  target=original.metadata.target, options=options)
        alias = contracted(alias_attrs)
        immutable_attrs = copy.deepcopy(attrs)
        for i in (0, 2):
            immutable_attrs.setdefault((i,), []).append(('tt.experimental_immutable', 1))
        immutable = contracted(immutable_attrs)
        immutable_kernels[prefix or 'ordinary'] = immutable
        equivalent[prefix or 'ordinary'] = alias.asm['ptx'] == immutable.asm['ptx']
        if not load_reuse:
            assert equivalent[prefix or 'ordinary'], 'Alias-only and explicit immutable code differ'
        kernels[prefix or 'original'] = original
        kernels[(prefix + '_alias') if prefix else 'alias'] = alias
    return dict(kernels=kernels, args=args, grid=grid, resources={k:metrics(v) for k,v in kernels.items()},
                alias_and_immutable_ptx_equal=equivalent, alias_allocation_disjointness_checked=True,
                options=options, config=list(config), immutable_kernels=immutable_kernels)


def readonly_unchanged(d):
    import torch
    for name in ('A_log','dt_bias','q','k','v','a','b','h0_indices','cu_seqlens','initial'):
        assert torch.equal(d[name].cpu(), d['cpu'][name]), name + ' mutated'


def timed(case, d, build, seed, rounds):
    repetitions = 128 if max(case.get('lengths', [case['T']])) <= 1 else 8
    graphs = {name:prior.capture(d, build, name, repetitions) for name in VARIANTS}
    times = prior.measure(graphs, repetitions, rounds, seed)
    pairs = {label:[x/y for x,y in zip(times[a], times[b])] for label,a,b in (
        ('pipeline_vs_original','original','pipeline'), ('pipeline_alias_vs_alias','alias','pipeline_alias'),
        ('alias_vs_original','original','alias'), ('pipeline_alias_vs_original','original','pipeline_alias'))}
    readonly_unchanged(d)
    return dict(times_us=times, median_us={k:statistics.median(v) for k,v in times.items()},
                paired_speedup_medians={k:statistics.median(v) for k,v in pairs.items()},
                paired_speedup_samples=pairs, repetitions=repetitions,
                state_reset_included_every_invocation=True)


def retain_ir(output, case, build):
    dest = output / 'representative_ir'
    dest.mkdir(exist_ok=True)
    for name, kernel in build['kernels'].items():
        for extension in ('ttgir', 'ptx'):
            with gzip.open(dest / (case['id'] + '.' + name + '.' + extension + '.gz'), 'wt') as f:
                f.write(kernel.asm[extension])
    kernel = build['kernels']['original']
    def entries(mapping):
        return [[list(k) if isinstance(k, tuple) else k,v] for k,v in mapping.items()]
    prior.write(output / (case['id'] + '.compile_manifest.json'),
                dict(function=kernel.src.fn.__name__, signature=entries(kernel.src.signature),
                     constants=entries(kernel.src.constants), attributes=entries(kernel.src.attrs),
                     options=build['options'], grid=build['grid'], target=vars(kernel.metadata.target)))


def boundaries(module, output):
    cases = copy.deepcopy(prior.BOUNDARIES)
    cases.append(dict(id='all_empty', cohort='boundary', B=1,T=0,H=1,HV=1,K=60,V=60,
                      dtype='float32',qk_dtype='float32',parameter_dtype='float32',lengths=[0,0],scale=1.0))
    records = []
    for i, case in enumerate(cases):
        d = prior.data(case, 61901 + i)
        build = compile_case(case, d, module, (16,1,2))
        checks = prior.validate(case, d, build)
        readonly_unchanged(d)
        records.append(dict(case=case, validation=checks, resources=build['resources']))
        print('BOUNDARY', case['id'], 'passed', flush=True)
    prior.write(output / 'boundary_validation.json', records)
    return records


def precompile(module, output):
    """Populate Triton's cache without CUDA initialization or tensor allocation."""
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    records = []
    def unpack(items):
        return {tuple(k) if isinstance(k,list) else k:v for k,v in items}
    for case in CASES:
        manifest = json.loads((output / (case['id'] + '.compile_manifest.json')).read_text())
        target = GPUTarget(**manifest['target'])
        for config in prior.CONFIGS:
            constants = unpack(manifest['constants']); constants[(21,)] = config[0]
            options = dict(manifest['options'],num_warps=config[1],num_stages=config[2])
            for name,fn in (('original',getattr(module,prior.NAME)),('pipeline',module.recurrent_loop_pipeline)):
                attrs = unpack(manifest['attributes'])
                for contract in ('absent','alias','immutable'):
                    attributes = copy.deepcopy(attrs)
                    if contract == 'alias':
                        for i in (0,1,2,5,6,7,8,9,10,11,12):
                            attributes.setdefault((i,),[]).append(('tt.experimental_alias_group',1 if i in (9,10) else 0))
                    elif contract == 'immutable':
                        for i in (0,2):attributes.setdefault((i,),[]).append(('tt.experimental_immutable',1))
                    kernel = triton.compile(ASTSource(fn,unpack(manifest['signature']),constexprs=constants,attrs=attributes),target=target,options=options)
                    records.append(dict(case=case['id'],config=config,source=name,contract=contract,
                                        shared=kernel.metadata.shared,ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest()))
            print('CPU PRECOMPILE',case['id'],config,flush=True)
    prior.write(output / 'cpu_precompile.json',dict(records=records,harness_sha256=prior.sha(__file__)))


def screen(module, output, native):
    """Bounded retune after the original fixed-configuration negative screen."""
    old_environment = json.loads((output / 'environment.json').read_text())
    expected = old_environment['native_library_sha256']
    assert prior.sha(native) == expected
    reports = []
    for i,case in enumerate(CASES):
        d = prior.data(case,68023+i)
        tuning = []
        for j,config in enumerate(prior.CONFIGS):
            build = compile_case(case,d,module,config)
            checks = prior.validate(case,d,build)
            timing = timed(case,d,build,69001+i*11+j,5)
            tuning.append(dict(config=list(config),validation=checks,resources=build['resources'],**timing))
            print('BOUNDED SCREEN',case['id'],config,timing['median_us'],flush=True)
        selected = {name:min(range(len(tuning)),key=lambda j:tuning[j]['median_us'][name]) for name in VARIANTS}
        best = {name:tuning[j]['median_us'][name] for name,j in selected.items()}
        report = dict(case=case['id'],seed=68023+i,tuning=tuning,selected_config_indices=selected,
                      selected_configurations={name:list(prior.CONFIGS[j]) for name,j in selected.items()},
                      independently_selected_screen_median_us=best,
                      screen_speedups=dict(pipeline_vs_original=best['original']/best['pipeline'],
                                           pipeline_alias_vs_alias=best['alias']/best['pipeline_alias']))
        reports.append(report)
        prior.write(output / (case['id'] + '.bounded_screen.json'),report)
        print('BEST SCREEN',case['id'],report['screen_speedups'],flush=True)
    positive = any(max(r['screen_speedups'].values()) > 1.02 for r in reports)
    summary = dict(records=[{k:v for k,v in r.items() if k!='tuning'} for r in reports],
                   positive_screen=positive,confirmation_needed=positive,
                   native_library_sha256=expected,harness_sha256=prior.sha(__file__))
    assert prior.sha(native) == expected
    prior.write(output / 'bounded_screen_summary.json',summary)


def confirm_screen(module, output, native):
    expected = json.loads((output / 'environment.json').read_text())['native_library_sha256']
    assert prior.sha(native) == expected
    reports = []
    for i,case in enumerate(CASES):
        chosen = json.loads((output / (case['id'] + '.bounded_screen.json')).read_text())
        selected = chosen['selected_config_indices']
        records = []
        for seed in (71009,72019,73009):
            d = prior.data(case,seed+i*101)
            builds = {idx:compile_case(case,d,module,prior.CONFIGS[idx]) for idx in set(selected.values())}
            checks = {str(idx):prior.validate(case,d,build) for idx,build in builds.items()}
            repetitions = 128 if max(case.get('lengths',[case['T']])) <= 1 else 8
            graphs = {name:prior.capture(d,builds[idx],name,repetitions) for name,idx in selected.items()}
            times = prior.measure(graphs,repetitions,13,seed+11)
            pairs = {label:[x/y for x,y in zip(times[a],times[b])] for label,a,b in (
                ('pipeline_vs_original','original','pipeline'), ('pipeline_alias_vs_alias','alias','pipeline_alias'),
                ('alias_vs_original','original','alias'), ('pipeline_alias_vs_original','original','pipeline_alias'))}
            readonly_unchanged(d)
            record = dict(seed=seed+i*101,validation=checks,times_us=times,
                          median_us={k:statistics.median(v) for k,v in times.items()},
                          paired_speedup_medians={k:statistics.median(v) for k,v in pairs.items()},
                          paired_speedup_samples=pairs,state_reset_included_every_invocation=True,
                          selected_resources={name:builds[idx]['resources'][name] for name,idx in selected.items()})
            records.append(record)
            print('FRESH CONFIRM',case['id'],seed,record['paired_speedup_medians'],flush=True)
            if seed == 71009:
                dest = output / 'selected_ir'; dest.mkdir(exist_ok=True)
                for name,idx in selected.items():
                    for ext in ('ttgir','ptx'):
                        with gzip.open(dest / (case['id']+'.'+name+'.'+ext+'.gz'),'wt') as f:
                            f.write(builds[idx]['kernels'][name].asm[ext])
        report = dict(case=case['id'],selected_configurations=chosen['selected_configurations'],
                      no_retuning=True,confirmations=records)
        reports.append(report)
        prior.write(output / (case['id'] + '.confirmation.json'),report)
    assert prior.sha(native) == expected
    prior.write(output / 'confirmation_summary.json',dict(native_library_sha256=expected,harness_sha256=prior.sha(__file__),
                records=[dict(case=r['case'],selected=r['selected_configurations'],
                         speedups=[x['paired_speedup_medians'] for x in r['confirmations']]) for r in reports]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--fixed-only', action='store_true')
    parser.add_argument('--precompile', action='store_true')
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--confirm-screen', action='store_true')
    args = parser.parse_args()
    source = args.output / 'kernels.py' if args.precompile or args.screen or args.confirm_screen else prepare(args.output)
    if not args.run and not args.precompile and not args.screen and not args.confirm_screen:
        return
    import torch
    import triton
    module = prior.load_module(source)
    native = Path(triton.__file__).resolve().parent / '_C/libtriton.so'
    if args.precompile:
        precompile(module,args.output)
        return
    if args.screen:
        screen(module,args.output,native)
        return
    if args.confirm_screen:
        confirm_screen(module,args.output,native)
        return
    native_sha = prior.sha(native)
    environment = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,triton=triton.__version__,
                       native_library_sha256=native_sha, harness_sha256=prior.sha(__file__), start_time=time.time(),
                       baseline_native_library_frozen=True)
    prior.write(args.output / 'environment.json', environment)
    reports = []
    for i, case in enumerate(CASES):
        d = prior.data(case, 61001 + i)
        if i == 0:
            diagnostic = compile_case(case,d,module,FIXED[case['id']],load_reuse=True)
            evidence = dict(case=case['id'],config=FIXED[case['id']],
                            alias_and_immutable_ptx_equal=diagnostic['alias_and_immutable_ptx_equal'],
                            explanation='Alias contracts enable redundant-load reuse in addition to LICM. Main factorial experiment uniformly disables that pass.')
            for name,kernel in (('alias',diagnostic['kernels']['alias']),('immutable',diagnostic['immutable_kernels']['ordinary'])):
                evidence[name] = metrics(kernel)
                evidence[name]['ttgir_load_count'] = kernel.asm['ttgir'].count('tt.load')
                with gzip.open(args.output / ('load_reuse_diagnostic.' + name + '.ttgir.gz'),'wt') as f:
                    f.write(kernel.asm['ttgir'])
            prior.write(args.output / 'load_reuse_diagnostic.json',evidence)
        build = compile_case(case, d, module, FIXED[case['id']])
        checks = prior.validate(case, d, build)
        report = dict(case=case, fixed_configuration=list(FIXED[case['id']]), fixed_validation=checks,
                      fixed_resources=build['resources'], alias_and_immutable_ptx_equal=build['alias_and_immutable_ptx_equal'])
        retain_ir(args.output, case, build)
        print('CODEGEN', case['id'], json.dumps(build['resources']), 'validation passed', flush=True)
        report['fixed_timing'] = timed(case, d, build, 62001+i, 13)
        print('FIXED', case['id'], report['fixed_timing']['paired_speedup_medians'], flush=True)
        prior.write(args.output / (case['id'] + '.json'), report)
        reports.append(report)
    boundaries(module, args.output)
    # Retune only if a fixed-config pipeline comparison exceeds a prespecified 2%.
    promising = any(max(r['fixed_timing']['paired_speedup_medians'][k] for k in
                        ('pipeline_vs_original','pipeline_alias_vs_alias')) > 1.02 for r in reports)
    if promising and not args.fixed_only:
        for i, (case, report) in enumerate(zip(CASES, reports)):
            d = prior.data(case, 63001+i)
            tuning = []
            for j, config in enumerate(prior.CONFIGS):
                build = compile_case(case, d, module, config)
                checks = prior.validate(case, d, build)
                result = timed(case, d, build, 64001+i*11+j, 5)
                tuning.append(dict(config=list(config), validation=checks, resources=build['resources'], **result))
                print('TUNE', case['id'], config, result['median_us'], flush=True)
            selected = {name:min(range(len(tuning)),key=lambda j:tuning[j]['median_us'][name]) for name in VARIANTS}
            report.update(tuning=tuning, selected_config_indices=selected,
                          selected_configurations={name:list(prior.CONFIGS[j]) for name,j in selected.items()}, confirmations=[])
            for j, seed in enumerate((65003,66007,67021)):
                d = prior.data(case, seed+i*101)
                builds = {idx:compile_case(case,d,module,prior.CONFIGS[idx]) for idx in set(selected.values())}
                checks = {str(idx):prior.validate(case,d,build) for idx,build in builds.items()}
                # All variants have identical argument values except constexpr BV;
                # capture separately so each selected launch uses its own binding.
                repetitions = 128 if max(case.get('lengths',[case['T']])) <= 1 else 8
                graphs = {name:prior.capture(d,builds[idx],name,repetitions) for name,idx in selected.items()}
                times = prior.measure(graphs,repetitions,13,seed+11)
                pairs = {label:[x/y for x,y in zip(times[a],times[b])] for label,a,b in (
                    ('pipeline_vs_original','original','pipeline'), ('pipeline_alias_vs_alias','alias','pipeline_alias'),
                    ('alias_vs_original','original','alias'), ('pipeline_alias_vs_original','original','pipeline_alias'))}
                readonly_unchanged(d)
                confirmation = dict(seed=seed+i*101,validation=checks,times_us=times,
                                    median_us={k:statistics.median(v) for k,v in times.items()},
                                    paired_speedup_medians={k:statistics.median(v) for k,v in pairs.items()},
                                    paired_speedup_samples=pairs, state_reset_included_every_invocation=True)
                report['confirmations'].append(confirmation)
                print('CONFIRM',case['id'],seed,confirmation['paired_speedup_medians'],flush=True)
            prior.write(args.output / (case['id'] + '.json'), report)
    assert prior.sha(native) == native_sha, 'Native library changed during experiment'
    summary = dict(native_library_frozen=True, native_library_sha256=native_sha, fixed_config_pipeline_promising=promising,
                   retuned=promising and not args.fixed_only, no_automatic_pipeline_implementation=True,
                   end_time=time.time(), cases=[dict(case=r['case']['id'],fixed_speedups=r['fixed_timing']['paired_speedup_medians'],
                   selected=r.get('selected_configurations'), confirmations=[c['paired_speedup_medians'] for c in r.get('confirmations',[])]) for r in reports])
    prior.write(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
