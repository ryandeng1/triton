#!/usr/bin/env python3
"""Pinned AITER recurrent LICM performance gate; no native changes.

--prepare downloads/audits pinned source and writes baseline/manual source only.
--run requires exclusive GPU ownership. Every measured invocation resets state;
the reset cost is INCLUDED. Timings cover every output and final-state write.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHA = "ef0a5a26b8cb38ea1308f827985e36cde964a6ab"
REPO = "ROCm/aiter"
KERNEL_PATH = "aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py"
TEST_PATH = "op_tests/test_gated_delta_rule.py"
BENCH_PATH = "op_tests/op_benchmarks/triton/bench_fused_kda_decode.py"
NAME = "fused_sigmoid_gating_delta_rule_update_kernel"
CONFIGS = [(8, 1, 3), (16, 1, 2), (32, 2, 2), (64, 4, 1)]
VARIANTS = ("original", "immutable", "alias", "manual")
CASES = [
    dict(id="decode16", cohort="decode_screen", B=1, T=16, H=8, HV=8, K=128, V=128, dtype="bfloat16", qk_dtype="bfloat16", parameter_dtype="bfloat16", lengths=[1]*16, scale=128**-0.5),
    dict(id="upstream1024", cohort="upstream_test_screen", B=2,T=1024,H=4,HV=8,K=128,V=128,dtype="bfloat16",qk_dtype="float32",parameter_dtype="float32",scale=1.0),
    dict(id="decode64", cohort="decode_holdout", B=1,T=64,H=12,HV=12,K=128,V=128,dtype="bfloat16",qk_dtype="bfloat16",parameter_dtype="bfloat16",lengths=[1]*64,scale=128**-0.5),
    dict(id="upstream63", cohort="upstream_test_holdout", B=1,T=63,H=1,HV=1,K=64,V=64,dtype="float32",qk_dtype="float32",parameter_dtype="float32",scale=1.0),
    dict(id="upstream500", cohort="upstream_test_holdout", B=2,T=500,H=4,HV=4,K=60,V=60,dtype="float32",qk_dtype="float32",parameter_dtype="float32",scale=1.0),
    dict(id="upstream2048", cohort="upstream_test_holdout", B=4,T=2048,H=4,HV=8,K=64,V=64,dtype="bfloat16",qk_dtype="float32",parameter_dtype="float32",scale=0.1),
]
BOUNDARIES = [
    dict(id="mixed_varlen",cohort="diagnostic_correctness_only",B=1,T=13,H=4,HV=8,K=64,V=64,dtype="bfloat16",qk_dtype="float32",parameter_dtype="float32",lengths=[0,1,4,8],negative_index=True,scale=64**-0.5),
    dict(id="no_initial_state",cohort="diagnostic_correctness_only",B=2,T=4,H=4,HV=8,K=128,V=128,dtype="float16",qk_dtype="float32",parameter_dtype="float32",initial=False,scale=128**-0.5),
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2)+"\n")

def body(node):
    return ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False)

def fetch(cache, rel):
    path=cache/rel
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(urllib.request.urlopen(f"https://raw.githubusercontent.com/{REPO}/{SHA}/{rel}").read())
    return path

def prepare(cache,output):
    output.mkdir(parents=True,exist_ok=True)
    paths={rel:fetch(cache,rel) for rel in (KERNEL_PATH,TEST_PATH,BENCH_PATH,"LICENSE")}
    (output/"AITer_LICENSE").write_text(paths["LICENSE"].read_text())
    source=paths[KERNEL_PATH].read_text()
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==NAME)
    base=copy.deepcopy(original)
    base.decorator_list=ast.parse("@triton.jit(do_not_specialize=['T'])\ndef f(): pass").body[0].decorator_list
    manual=copy.deepcopy(base);manual.name="recurrent_manual_hoist"
    loop=next(n for n in manual.body if isinstance(n,ast.For))
    moved=[]
    for statement in list(loop.body):
        if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id in ("b_A_log","b_dt_bias"):
            moved.append(statement);loop.body.remove(statement)
    if len(moved)!=2:raise AssertionError("Expected exactly two invariant loads")
    # Preserve zero-trip behavior for an empty variable-length sequence.
    for statement in moved:
        load=next(n for n in ast.walk(statement.value) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="load")
        load.keywords += [ast.keyword(arg="mask",value=ast.Compare(left=ast.Name(id="T",ctx=ast.Load()),ops=[ast.Gt()],comparators=[ast.Constant(0)])),ast.keyword(arg="other",value=ast.Constant(0.0))]
    exp=ast.parse("negative_exp_A = -tl.exp(b_A_log)").body[0]
    class Exp(ast.NodeTransformer):
        def visit_UnaryOp(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(exp.value,include_attributes=False):
                return ast.copy_location(ast.Name(id="negative_exp_A",ctx=ast.Load()),node)
            return self.generic_visit(node)
    Exp().visit(loop)
    where=manual.body.index(loop);manual.body[where:where]=moved+[exp]
    tree=ast.fix_missing_locations(ast.Module(body=[base,manual],type_ignores=[]))
    module=output/"kernels.py"
    module.write_text("# SPDX-License-Identifier: MIT\n# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.\n# See AITer_LICENSE in this directory.\n# Derived from ROCm/aiter at "+SHA+".\n# Original body unchanged; manual clone moves two loads and dependent neg-exp.\nimport triton\nimport triton.language as tl\n\n"+ast.unparse(tree)+"\n")
    new=ast.parse(module.read_text());parsed={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
    if body(parsed[NAME])!=body(original):raise AssertionError("Baseline AST differs")
    # Reverse only the prescribed manual edit, then compare full bodies.
    restored=copy.deepcopy(parsed[manual.name]);rloop=next(n for n in restored.body if isinstance(n,ast.For))
    original_loop=next(n for n in original.body if isinstance(n,ast.For))
    restored.body=[n for n in restored.body if not (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("b_A_log","b_dt_bias","negative_exp_A"))]
    class Undo(ast.NodeTransformer):
        def visit_Name(self,node):
            return copy.deepcopy(exp.value) if node.id=="negative_exp_A" and isinstance(node.ctx,ast.Load) else node
    Undo().visit(rloop)
    for position,statement in enumerate(original_loop.body):
        if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id in ("b_A_log","b_dt_bias"):
            rloop.body.insert(position,copy.deepcopy(statement))
    if body(restored)!=body(original):raise AssertionError("Unexpected manual AST changes")
    audit=dict(repository=REPO,revision=SHA,source_files={rel:dict(sha256=sha(path),url=f"https://github.com/{REPO}/blob/{SHA}/{rel}") for rel,path in paths.items()},baseline_body_ast_equal=True,manual_reverse_edit_ast_equal=True,baseline_body_sha256=hashlib.sha256(body(original).encode()).hexdigest(),manual_changes=["Move b_A_log and b_dt_bias loads immediately before timestep loop, masked by T>0","Move their dependent negative exp(A_log) outside loop","No arithmetic reassociation or output/state scope change"],wrapper_changes="Remove provider autotune/heuristic decorators; preserve do_not_specialize=['T']; bind flags explicitly. Each variant independently tunes identical configurations.",case_origin="Exact kernel has upstream correctness tests at T=63/500/1000/1024/2048 and D=60/64/128, fp32/fp16/bf16, grouped heads, normalized QK and initial state. These are supported test inputs, not production-frequency evidence. Decode benchmark per-sequence length is1; our decode dimensions follow it but correct scalar-gating argument shapes.",benchmark_caveat="Pinned bench_fused_kda_decode's legacy3-kernel branch supplies per-channel gate/dt_bias to scalar-gating API and ignores returned recurrence output before RMSNorm. It is not used as a correctness or complete-block performance baseline. Speculative KDA benchmarks call another fused kernel, so short multi-token varlen cases here are diagnostics only.",work_scope="One complete scalar-gating recurrent operator: every timestep output plus final in-place state writes. Every timed invocation includes reset of state from identical snapshot. Does not include convolution or gated RMSNorm/model block.",liger_correction="Plain fast backward path already hoists scalar gradient; prior general-path plain specialization must not be treated as a production LICM opportunity.",configurations=[dict(BV=b,num_warps=w,num_stages=s) for b,w,s in CONFIGS],cases=CASES,boundaries=BOUNDARIES)
    write(output/"provenance.json",audit)
    return module

def load_module(path):
    spec=importlib.util.spec_from_file_location("recurrent_licm_kernels",path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def data(case,seed):
    import torch
    torch.set_num_threads(4)
    g=torch.Generator().manual_seed(seed)
    B,T,H,HV,K,V=(case[k] for k in ("B","T","H","HV","K","V"))
    dtype=getattr(torch,case["dtype"]);qk=getattr(torch,case["qk_dtype"]);param=getattr(torch,case["parameter_dtype"])
    rand=lambda shape,dt:torch.randn(shape,generator=g,dtype=torch.float32).to(dt)
    cpu=dict(q=rand((B,T,H,K),qk),k=rand((B,T,H,K),qk),v=rand((B,T,HV,V),dtype),b=rand((B,T,HV),dtype),a=rand((B,T,HV),param),A_log=rand((HV,),param),dt_bias=rand((HV,),param))
    lengths=case.get("lengths",[T]*B);N=len(lengths)
    cpu['initial']=rand((N+2,HV,K,V),torch.float32)*0.1
    cpu['h0_indices']=torch.arange(1,N+1,dtype=torch.int64)
    if case.get('negative_index'):cpu['h0_indices'][-1]=-1
    cpu['cu_seqlens']=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),dtype=torch.int64)
    out=torch.empty((B,T,HV,V),dtype=qk)
    ref_state=cpu['initial'].double().clone();ref=torch.empty_like(out,dtype=torch.float64)
    flat={name:t.reshape(-1,*t.shape[2:]).double() for name,t in cpu.items() if name in ('q','k','v','a','b')}
    pos=0
    for sequence,length in enumerate(lengths):
        idx=int(cpu['h0_indices'][sequence]);h=ref_state[idx].clone() if case.get('initial',True) and idx>=0 else torch.zeros((HV,K,V),dtype=torch.float64)
        for step in range(length):
            j=pos+step;qt=flat['q'][j].repeat_interleave(HV//H,dim=0);kt=flat['k'][j].repeat_interleave(HV//H,dim=0)
            qt=qt/torch.sqrt((qt*qt).sum(-1,keepdim=True)+1e-6)*case['scale'];kt=kt/torch.sqrt((kt*kt).sum(-1,keepdim=True)+1e-6)
            x=flat['a'][j]+cpu['dt_bias'].double();sp=torch.where(x<=20,torch.log1p(torch.exp(x)),x)
            decay=torch.exp(-torch.exp(cpu['A_log'].double())*sp)
            h=h*decay[:,None,None]
            adjusted=(flat['v'][j]-(h*kt[:,:,None]).sum(1))*torch.sigmoid(flat['b'][j])[:,None]
            h=h+kt[:,:,None]*adjusted[:,None,:]
            ref.reshape(-1,HV,V)[j]=(h*qt[:,:,None]).sum(1)
        if case.get('initial',True) and idx>=0:ref_state[idx]=h
        pos+=length
    result={name:t.cuda() for name,t in cpu.items()}
    result['h0_source']=result['initial'].clone()
    # Prefix/suffix guards exercise full output coverage without changing strides.
    storage=torch.full((out.numel()+128,),17.0,device='cuda',dtype=qk)
    result['o']=storage[64:-64].view_as(out);result['output_storage']=storage
    result['cpu']=cpu;result['reference']=ref;result['reference_state']=ref_state;result['N']=N
    return result

def arguments(case,d,BV):
    return (d['A_log'],d['a'],d['dt_bias'],1.0,20.0,d['q'],d['k'],d['v'],d['b'],d['o'],d['h0_source'],d['h0_indices'],d['cu_seqlens'],case['scale'],case['T'],case['B'],case['H'],case['HV'],case['K'],case['V'],1<<(case['K']-1).bit_length(),BV,case.get('initial',True),True,'lengths' in case)

def compiled(case,d,module,config):
    import torch,triton
    from triton.compiler import ASTSource
    from triton.experimental.semantic_contracts.native import compile_immutable_licm
    BV,warps,stages=config;args=arguments(case,d,BV);grid=(1,triton.cdiv(case['V'],BV),d['N']*case['HV']);options=dict(num_warps=warps,num_stages=stages)
    plan=compile_immutable_licm(getattr(module,NAME),args,grid=grid,immutable_args=('A_log','dt_bias'),options=options)
    original=plan.baseline
    attrs=copy.deepcopy(original.src.attrs)
    for number,value in enumerate(args):
        if isinstance(value,torch.Tensor):attrs.setdefault((number,),[]).append(('tt.experimental_alias_group',1 if number in (9,10) else 0))
    # Alias group0 contains all read inputs; group1 contains output/state.
    # Verify complete backing-storage disjointness, not merely base pointers.
    for wi in (9,10):
        ws=args[wi].untyped_storage();wlo=ws.data_ptr();whi=wlo+ws.nbytes()
        for ri,value in enumerate(args):
            if ri in (9,10) or not isinstance(value,torch.Tensor):continue
            rs=value.untyped_storage();rlo=rs.data_ptr();rhi=rlo+rs.nbytes()
            assert whi<=rlo or rhi<=wlo,('alias group overlap',wi,ri)
    alias=triton.compile(ASTSource(original.src.fn,original.src.signature,constexprs=original.src.constants,attrs=attrs),target=original.metadata.target,options=dict(options,enable_indexed_update_specialization=False,enable_private_atomic_relaxation=False))
    manual=module.recurrent_manual_hoist.warmup(*args,grid=grid,**options)
    kernels=dict(original=original,immutable=plan.optimized,alias=alias,manual=manual)
    resources={}
    for name,k in kernels.items():
        k._init_handles();resources[name]=dict(registers=k.n_regs,spills=k.n_spills,shared=k.metadata.shared,ptx_sha256=hashlib.sha256(k.asm['ptx'].encode()).hexdigest(),cubin_sha256=hashlib.sha256(k.asm['cubin']).hexdigest())
    return dict(kernels=kernels,args=args,grid=grid,guard=plan.report(),resources=resources,alias_allocation_disjointness_checked=True)

def validate(case,d,build):
    import torch
    records={};baseline=None;statebase=None
    for name,kernel in build['kernels'].items():
        d['h0_source'].copy_(d['initial']);d['o'].fill_(float('nan'))
        kernel[build['grid']](*build['args']);torch.cuda.synchronize()
        actual=d['o'].cpu();state=d['h0_source'].cpu()
        out_atol,out_rtol=(0.008,0.025) if actual.dtype==torch.bfloat16 else ((0.002,0.01) if actual.dtype==torch.float16 else (0.0004,0.002))
        torch.testing.assert_close(actual.double(),d['reference'],atol=out_atol,rtol=out_rtol)
        torch.testing.assert_close(state.double(),d['reference_state'],atol=0.0005,rtol=0.003)
        assert torch.all(d['output_storage'][:64]==17) and torch.all(d['output_storage'][-64:]==17)
        if baseline is None:baseline=actual.clone();statebase=state.clone()
        exact=bool(torch.equal(actual,baseline) and torch.equal(state,statebase))
        if not exact:
            torch.testing.assert_close(actual,baseline,atol=out_atol,rtol=out_rtol)
            torch.testing.assert_close(state,statebase,atol=0.0005,rtol=0.003)
        records[name]=dict(max_output_error=float((actual.double()-d['reference']).abs().max()) if actual.numel() else 0.0,max_state_error=float((state.double()-d['reference_state']).abs().max()),same_config_bitwise_equal_to_original=exact,guards_passed=True)
    return records

def capture(d,build,name,repetitions):
    import torch
    launcher=build['kernels'][name][build['grid']];args=build['args']
    reset_dst=d['h0_source'][1:d['N']+1];reset_src=d['initial'][1:d['N']+1]
    for _ in range(2):reset_dst.copy_(reset_src);launcher(*args)
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            reset_dst.copy_(reset_src)
            launcher(*args)
    return graph

def measure(graphs,repetitions,rounds,seed):
    import torch
    values={key:[] for key in graphs};rng=random.Random(seed)
    for _ in range(rounds):
        order=list(graphs);rng.shuffle(order)
        for key in order:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();graphs[key].replay();end.record();end.synchronize()
            values[key].append(start.elapsed_time(end)*1000/repetitions)
    return values

def case_run(case,module,output,seed,tune_rounds=3,confirm_rounds=11):
    import torch
    started=time.time();d=data(case,seed)
    reps=128 if max(case.get('lengths',[case['T']]))<=1 else (8 if case['T']>=500 else 32)
    builds={};tuning=[]
    for number,config in enumerate(CONFIGS):
        build=compiled(case,d,module,config);builds[number]=build
        checks=validate(case,d,build)
        graphs={name:capture(d,build,name,reps) for name in VARIANTS}
        times=measure(graphs,reps,tune_rounds,seed+number)
        tuning.append(dict(config=list(config),times_us=times,median_us={k:statistics.median(v) for k,v in times.items()},validation=checks,resources=build['resources'],profitability_guard=build['guard']))
        print(case['id'],'config',config,{k:round(statistics.median(v),3) for k,v in times.items()},flush=True)
        del graphs
    selected={name:min(range(len(CONFIGS)),key=lambda i:tuning[i]['median_us'][name]) for name in VARIANTS}
    # New payload seed, independent graphs, randomized order: no timing reuse.
    confirm_data=data(case,seed+10000);confirm_builds={};graphs={};validation={}
    for idx in set(selected.values()):
        confirm_builds[idx]=compiled(case,confirm_data,module,CONFIGS[idx])
        validation[str(idx)]=validate(case,confirm_data,confirm_builds[idx])
    for name,idx in selected.items():graphs[name]=capture(confirm_data,confirm_builds[idx],name,reps)
    held=measure(graphs,reps,confirm_rounds,seed+20000)
    ratios={name:[b/v for b,v in zip(held['original'],held[name])] for name in VARIANTS if name!='original'}
    result=dict(case=case,seed=seed,confirmation_seed=seed+10000,repetitions=reps,state_reset_included_every_invocation=True,state_reset_bytes=d['N']*case['HV']*case['K']*case['V']*4,scope="all timestep outputs and final-state writes plus identical state reset; allocations, oracle, compile and tuning excluded",tuning=tuning,selected_config_indices=selected,selected_configurations={k:list(CONFIGS[i]) for k,i in selected.items()},confirmation_us=held,confirmation_median_us={k:statistics.median(v) for k,v in held.items()},paired_speedup_vs_independently_tuned_original={k:dict(median=statistics.median(v),min=min(v),max=max(v),ratios=v) for k,v in ratios.items()},confirmation_validation=validation,elapsed_seconds=time.time()-started)
    for name in ('A_log','dt_bias','q','k','v','a','b','h0_indices','cu_seqlens','initial'):
        assert torch.equal(confirm_data[name].cpu(),confirm_data['cpu'][name]),name+' mutated'
    result['readonly_inputs_unchanged']=True
    write(output/(case['id']+'.json'),result)
    if case['id']=='upstream1024':
        ir=output/'representative_ir';ir.mkdir(exist_ok=True)
        for name,idx in selected.items():
            k=confirm_builds[idx]['kernels'][name]
            (ir/(name+'.ttgir')).write_text(k.asm['ttgir'])
    print('CONFIRM',case['id'],result['confirmation_median_us'],{k:v['median'] for k,v in result['paired_speedup_vs_independently_tuned_original'].items()},flush=True)
    return result


def confirm_fixed(case,module,output):
    import torch,triton
    prior=json.loads((output/(case['id']+'.json')).read_text())
    selected=prior['selected_config_indices'];records=[]
    native=Path(triton.__file__).resolve().parent/'_C/libtriton.so'
    expected=json.loads((output/'environment.json').read_text())['native_library_sha256']
    assert sha(native)==expected,'Native library changed before fixed confirmation'
    for seed in (33017,44027):
        d=data(case,seed);builds={};checks={};graphs={}
        reps=prior['repetitions']
        for idx in set(selected.values()):
            builds[idx]=compiled(case,d,module,CONFIGS[idx]);checks[str(idx)]=validate(case,d,builds[idx])
        for name,idx in selected.items():graphs[name]=capture(d,builds[idx],name,reps)
        times=measure(graphs,reps,13,seed+7)
        ratios={name:[x/y for x,y in zip(times['original'],times[name])] for name in VARIANTS if name!='original'}
        record=dict(seed=seed,selected_config_indices=selected,confirmation_us=times,median_us={k:statistics.median(v) for k,v in times.items()},paired_speedup_medians={k:statistics.median(v) for k,v in ratios.items()},paired_speedup_samples=ratios,validation=checks)
        records.append(record)
        print('FIXED',case['id'],seed,record['paired_speedup_medians'],flush=True)
        if case['id']=='upstream500' and seed==33017:
            dest=output/'representative_ir';dest.mkdir(exist_ok=True)
            for idx in set(selected.values()):
                for name in ('original','immutable','manual'):
                    (dest/('upstream500_cfg'+str(idx)+'_'+name+'.ttgir')).write_text(builds[idx]['kernels'][name].asm['ttgir'])
        for name in ('A_log','dt_bias','q','k','v','a','b','h0_indices','cu_seqlens','initial'):
            assert torch.equal(d[name].cpu(),d['cpu'][name])
    assert sha(native)==expected,'Native library changed during fixed confirmation'
    report=dict(case=case['id'],no_retuning=True,state_reset_included_every_invocation=True,records=records,harness_sha256=sha(__file__),native_library_sha256=expected)
    write(output/(case['id']+'.fixed_confirmation.json'),report)
    return report


def postbuild_alias(case,module,output):
    import torch,triton
    prior=json.loads((output/(case['id']+'.json')).read_text())
    picked=prior['selected_config_indices'];d=data(case,55031)
    builds={idx:compiled(case,d,module,CONFIGS[idx]) for idx in {picked['original'],picked['immutable']}}
    checks={str(idx):validate(case,d,build) for idx,build in builds.items()}
    choices=dict(original=picked['original'],immutable=picked['immutable'],alias=picked['immutable'])
    graphs={name:capture(d,builds[idx],name,prior['repetitions']) for name,idx in choices.items()}
    times=measure(graphs,prior['repetitions'],13,55037)
    optimized=builds[picked['immutable']]['kernels']
    alias=optimized['alias'];immutable=optimized['immutable']
    assert 'tt.experimental_immutable' not in str(alias.src.attrs)
    ir=alias.asm['ttgir'];lines=ir.splitlines();first_loop=next(i for i,line in enumerate(lines) if 'scf.for' in line)
    roots={'%A_log':'A_log','%dt_bias':'dt_bias','%arg0':'A_log','%arg2':'dt_bias'};selected=[]
    for i,line in enumerate(lines):
        match=re.search(r'(%[A-Za-z0-9_]+)\s*=\s*([A-Za-z0-9_.]+)\s+(.*)',line)
        if not match:continue
        value,op,rhs=match.groups();operands=re.findall(r'%[A-Za-z0-9_]+',rhs)
        if operands and op in ('tt.addptr','tt.splat','tt.broadcast','tt.expand_dims','tt.reshape','ttg.convert_layout') and operands[0] in roots:roots[value]=roots[operands[0]]
        if operands and op=='tt.load' and operands[0] in roots:selected.append((i,line.strip()))
    assert len(selected)==2,(case['id'],selected)
    assert all(i<first_loop for i,line in selected),(case['id'],selected,first_loop)
    native=Path(triton.__file__).resolve().parent/'_C/libtriton.so'
    result=dict(case=case['id'],seed=55031,selected_configurations={name:list(CONFIGS[idx]) for name,idx in choices.items()},state_reset_included_every_invocation=True,alias_has_no_immutable_attribute=True,alias_allocation_disjointness_checked=True,selected_alias_parameter_loads=[dict(ir_line=i+1,operation=line) for i,line in selected],alias_timestep_loop_line=first_loop+1,alias_and_immutable_ptx_exact_equal=alias.asm['ptx']==immutable.asm['ptx'],alias_and_immutable_resources_equal=all(builds[picked['immutable']]['resources']['alias'][field]==builds[picked['immutable']]['resources']['immutable'][field] for field in ('registers','spills','shared')),resources=builds[picked['immutable']]['resources'],validation=checks,times_us=times,median_us={k:statistics.median(v) for k,v in times.items()},paired_alias_speedup_vs_original=statistics.median([b/a for b,a in zip(times['original'],times['alias'])]),paired_alias_over_immutable_time=statistics.median([a/i for a,i in zip(times['alias'],times['immutable'])]),native_library_sha256=sha(native),harness_sha256=sha(__file__))
    if case['id']=='upstream1024':
        (output/'representative_ir'/'postbuild_alias.ttgir').write_text(ir)
    print('POSTBUILD',case['id'],result['paired_alias_speedup_vs_original'],result['paired_alias_over_immutable_time'],result['alias_and_immutable_ptx_exact_equal'],flush=True)
    write(output/(case['id']+'.postbuild_alias.json'),result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('/tmp/semantic-survey-cache')/('AITER-'+SHA))
    parser.add_argument('--output',type=Path,default=ROOT/'results/recurrent_licm_final')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--confirm-fixed',action='store_true')
    parser.add_argument('--postbuild-alias',action='store_true')
    parser.add_argument('--cases',nargs='*')
    args=parser.parse_args();module_path=prepare(args.cache,args.output)
    if not args.run and not args.confirm_fixed and not args.postbuild_alias:return
    import torch,triton
    module=load_module(module_path)
    if args.postbuild_alias:
        reports=[postbuild_alias(case,module,args.output) for case in CASES if not args.cases or case['id'] in args.cases]
        write(args.output/'postbuild_alias_validation.json',dict(reports=reports))
        return
    if args.confirm_fixed:
        reports=[confirm_fixed(case,module,args.output) for case in CASES if not args.cases or case['id'] in args.cases]
        write(args.output/'fixed_confirmation_summary.json',dict(reports=reports))
        return
    props=torch.cuda.get_device_properties(0)
    native=Path(triton.__file__).resolve().parent/'_C/libtriton.so'
    metadata=dict(gpu=props.name,compute_capability=list(torch.cuda.get_device_capability()),torch=torch.__version__,triton=triton.__version__,native_library_sha256=sha(native),harness_sha256=sha(__file__),kernel_source_sha256=sha(module_path),start_time=time.time())
    write(args.output/'environment.json',metadata)
    records=[]
    for index,case in enumerate(CASES):
        if args.cases and case['id'] not in args.cases:continue
        records.append(case_run(case,module,args.output,1701+index*31))
    boundary=[]
    if not args.cases:
        for index,case in enumerate(BOUNDARIES):
            d=data(case,9981+index);build=compiled(case,d,module,CONFIGS[0]);boundary.append(dict(case=case,validation=validate(case,d,build)))
        write(args.output/'boundary_validation.json',boundary)
    assert sha(native)==metadata['native_library_sha256'],'Native library changed during study'
    summary=dict(state_reset_included=True,measurements=[dict(case=r['case']['id'],cohort=r['case']['cohort'],medians_us=r['confirmation_median_us'],speedups={k:v['median'] for k,v in r['paired_speedup_vs_independently_tuned_original'].items()},selected=r['selected_configurations']) for r in records],boundary_cases=len(boundary),end_time=time.time(),native_library_frozen=True,limitations=['Pinned upstream long-T correctness tests do not prove production prevalence.','Decode benchmark call site is not validated equivalent complete-block work; its malformed legacy baseline is not measured.','Measurements cover scalar-gating recurrence + final state + identical reset, not convolution/RMSNorm or model end-to-end.','Different launch configurations are independently tuned over four bounded choices, not global autotuning.'])
    write(args.output/'summary.json',summary)

if __name__=='__main__':main()
