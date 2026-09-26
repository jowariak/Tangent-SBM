"""Fresh SNS small-action selection benchmark using frozen models, no training.

Primary policy predicts a terminal mean-field CHANGE as delta * E[J].
Secondary policy uses common-noise model finite changes. Both see the requested
change, x0 and a0; neither receives simulator outcomes for candidate actions.
"""
import argparse,json,math,statistics
from pathlib import Path
import torch
import spdebench_sns_conditional_dsbm_v2 as base
import spdebench_sns_response_dataset as gen
import run_sns_local_budget as helper

SEEDS=(32,42,52)
ACTIONS=(0.,-.125,.125,-.25,.25)  
METHODS=('conditional','tangent','tsbm')
GEN_SEED=2026092116
def emit(path,obj):helper.save_json(path,obj)
def spec(args):
    meta=helper.metadata(args)
    return dict(name='SNS local mean-field-change action selection',contexts=8,a0_values=[-.75,-.25,.25,.75]*2,
        actions=list(ACTIONS),goal_actions=[-.25,.25],goal_mc=32,outcome_mc=64,model_mc=16,
        generation_seed=GEN_SEED,physical_batch=args.sim_batch,metadata_sha256=helper.sha(args.data_dir/'metadata.json'),
        initial_conditions_sha256=helper.sha(args.data_dir/'initial_conditions.pt'),sim_config=meta['sim_config'],
        sources={n:helper.sha(Path(__file__).parent/n) for n in ['run_sns_intervention_test.py','run_sns_local_budget.py','spdebench_sns_response_dataset.py','spdebench_sns_conditional_dsbm_v2.py']},
        task='Track a requested change in conditional terminal mean field; primary local linear response policy, secondary finite-change policy',
        exclusion='Frozen original checkpoints only; no endpoint-study candidates or FINAL2 data',
        precision={'matmul_tf32':False,'cudnn_tf32':False})
def checkpoint_path(args,method,seed):
    if method=='conditional':return args.conditional_root/'sigma_1p4'/f'seed_{seed}'/'imf_5.pt'
    if method=='tangent':return args.tangent_root/'lambda_1000'/f'seed_{seed}'/'imf_5.pt'
    return args.tsbm_root/f'seed_{seed}'/'final.pt'
def audit_tangent(ck):
    expected=dict(fork_imf=3,lambda_sens=1000,sens_batch_size=8,sens_pairs=1,sens_every=10)
    actual={k:ck.get(k) for k in expected};origin='explicit checkpoint sens_pairs'
    
    
    
    if 'sens_pairs' not in ck and ck.get('response_objective')=='conditional_mean_two_independent_rollout_cross':
        actual['sens_pairs']=1;origin='legacy two-independent-rollout objective (one cross pair)'
    for key,value in expected.items():
        if actual[key]!=value:
            raise ValueError(f'Frozen tangent check failed: {key}={ck.get(key)!r}, expected {value}; '
                             f'response_objective={ck.get("response_objective")!r}')
    return dict(settings=actual,sens_pairs_evidence=origin,response_objective=ck.get('response_objective'))
def model(args,method,seed,device):
    ck=base.safe_load(checkpoint_path(args,method,seed));cfg=base.Config(**ck['config'])
    mean,std=base.normalization_from_metadata(helper.metadata(args))
    if cfg.seed!=seed or cfg.reference_sigma!=1.4:raise ValueError('Wrong model seed/sigma')
    m=base.ConditionalFieldDSBM(cfg,device)
    if method=='tsbm':
        if ck['completed_passes']!=10 or ck['normalization']!=helper.metadata(args)['normalization']:raise ValueError('Wrong TSBM protocol')
        if ck['train_data_sha256']!=helper.sha(args.data_dir/'endpoint_train.pt'):raise ValueError('Wrong TSBM dataset')
        m.net_f.load_state_dict(ck['weights']['fwd']);m.net_b.load_state_dict(ck['weights']['bwd'])
    else:
        if ck['imf']!=5:raise ValueError('Wrong IMF')
        if not math.isclose(ck['state_mean'],mean,abs_tol=1e-8) or not math.isclose(ck['state_std'],std,abs_tol=1e-8):raise ValueError('Normalization mismatch')
        if method=='tangent':
            audit=audit_tangent(ck)
            print(f'Tangent seed {seed}: {audit["sens_pairs_evidence"]}',flush=True)
        m.load_state_dict(ck['model'])
    m.net_f.eval();m.net_b.eval();return m,mean,std
def verify_inputs(args):
    base.resolve_device(args.device)
    for method in METHODS:
        for seed in SEEDS:
            m,_,_=model(args,method,seed,torch.device('cpu'));del m
def physical_changes(sim,x,a0,deltas,mc,batch,seed):
    
    sums={d:torch.zeros_like(x.cpu()) for d in deltas}
    for start in range(0,mc,batch):
        n=min(batch,mc-start);gen.set_seed(seed+start)
        xx=x.to(sim.device).expand(n,-1,-1).contiguous();aa=torch.full((n,),float(a0),device=sim.device)
        state=gen.capture_rng_state(sim.device);nominal=sim.endpoint(xx,aa)
        for d in deltas:
            if d==0:continue
            gen.restore_rng_state(state,sim.device)
            changed=sim.endpoint(xx,aa+d)-nominal;helper.finite(changed)
            sums[d]+=changed.sum(0,keepdim=True).cpu()/mc
    return torch.stack([sums[d] for d in deltas])  
def prepare(args):
    verify_inputs(args);pr=spec(args);sim,source=helper.simulator(args);pr['simulator']=source
    args.run_root.mkdir(parents=True,exist_ok=True);protocol=args.run_root/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=pr:raise ValueError('Existing task protocol differs')
    else:emit(protocol,pr)
    manifest=args.run_root/'cases_manifest.json'
    if manifest.exists():
        cases(args);print('Reusing completed fresh cases',flush=True);return
    gen.set_seed(GEN_SEED)
    w=base.safe_load(args.data_dir/'initial_conditions.pt')['w_star'].to(sim.device)
    x0=(w+sim.grf.sample(8)).cpu();hashes={}
    for i,a0 in enumerate(pr['a0_values']):
        path=args.run_root/f'case_{i:02d}.pt'
        if path.exists():
            ck=base.safe_load(path)
            if ck['protocol']!=pr or ck['a0']!=a0:raise ValueError('Stale case')
            torch.testing.assert_close(ck['x0'],x0[i:i+1],rtol=0,atol=0)
        else:
            goals=physical_changes(sim,x0[i:i+1],a0,[-.25,.25],32,args.sim_batch,GEN_SEED+10000+i*1000)
            outcomes=physical_changes(sim,x0[i:i+1],a0,ACTIONS,64,args.sim_batch,GEN_SEED+100000+i*1000)
            ck=dict(protocol=pr,x0=x0[i:i+1],a0=a0,goals=goals,outcomes=outcomes)
            helper.atomic_save(ck,path)
        hashes[path.name]=helper.sha(path);print(f'Fresh intervention cases {i+1}/8',flush=True)
    emit(manifest,dict(protocol=pr,cases=hashes,physical_trajectories=8*(3*32+5*64)))
def cases(args):
    manifest=json.loads((args.run_root/'cases_manifest.json').read_text());pr=spec(args)
    if any(manifest['protocol'].get(k)!=v for k,v in pr.items()):raise ValueError('Case protocol mismatch')
    out=[]
    for n,h in manifest['cases'].items():
        if helper.sha(args.run_root/n)!=h:raise ValueError('Case hash mismatch')
        out.append(base.safe_load(args.run_root/n))
    return out,manifest
def forecast(m,x0,a0,std,mc,seed):
    
    derivative=torch.zeros_like(x0.cpu());finite=torch.zeros(len(ACTIONS),*x0.shape[1:])
    for k in range(mc):
        base.set_seed(seed+k);nb=m._noise_bank(x0)
        derivative+=m.tangent_direction_rollout(x0,a0,torch.ones_like(a0),nb).cpu()*std/mc
        nominal=m.sample_sde(x0,a0,noise_bank=nb)
        for j,d in enumerate(ACTIONS):
            if d:finite[j]+=(m.sample_sde(x0,a0+d,noise_bank=nb)-nominal)[0].cpu()*std/mc
    linear=torch.cat([d*derivative for d in ACTIONS])
    helper.finite(linear);helper.finite(finite)
    return {'linear_response':linear,'finite_change':finite}
def choose(predicted,goal):return int((predicted-goal).square().flatten(1).mean(1).argmin())
def score(outcomes,goal,selected):
    errors=(outcomes-goal).square().flatten(1).mean(1);oracle=int(errors.argmin());zero=float(errors[0])
    norm=max(float(goal.square().mean()),1e-20)
    return dict(action=ACTIONS[selected],oracle_action=ACTIONS[oracle],mean_change_rmse=math.sqrt(float(errors[selected])),
        relative_change_error=math.sqrt(float(errors[selected])/norm),normalized_regret=(float(errors[selected])-float(errors[oracle]))/norm,
        improves_over_no_change=float(errors[selected])<zero,oracle_action_match=selected==oracle)
def evaluate(args):
    rows,manifest=cases(args);verify_inputs(args);output=args.run_root/'intervention_results.json'
    if output.exists():raise FileExistsError(output)
    dev=base.resolve_device(args.device);per={}
    for method in METHODS:
        per[method]={}
        for seed in SEEDS:
            m,mean,std=model(args,method,seed,dev);records=[]
            provenance=dict(cases_manifest_sha256=helper.sha(args.run_root/'cases_manifest.json'),checkpoint_sha256=helper.sha(checkpoint_path(args,method,seed)),sources=spec(args)['sources'])
            if method=='tangent':provenance['tangent_training_audit']=audit_tangent(base.safe_load(checkpoint_path(args,method,seed)))
            cache=args.run_root/f'results_{method}_{seed}.json'
            if cache.exists():
                saved=json.loads(cache.read_text())
                if saved['provenance']!=provenance:raise ValueError('Stale result cache')
                records=saved['records']
            else:
                for i,row in enumerate(rows):
                    x=base.normalize_state(row['x0'].unsqueeze(1),mean,std).to(dev);a=torch.tensor([[row['a0']]],device=dev)
                    predictions=forecast(m,x,a,std,16,880000+seed*100000+i*1000)
                    for policy,pred in predictions.items():
                        for g,goal in enumerate(row['goals']):
                            selected=choose(pred,goal)
                            records.append(dict(case=i,goal_index=g,policy=policy,**score(row['outcomes'],goal,selected)))
                    print(f'{method} seed {seed}: decisions {i+1}/8 contexts',flush=True)
                emit(cache,dict(provenance=provenance,records=records))
            per[method][str(seed)]=records;del m
    keys=['relative_change_error','normalized_regret','improves_over_no_change','oracle_action_match']
    aggregate={}
    for method,seedrows in per.items():
        aggregate[method]={}
        for policy in ['linear_response','finite_change']:
            aggregate[method][policy]={}
            for key in keys:
                values=[statistics.mean(float(r[key]) for r in rr if r['policy']==policy) for rr in seedrows.values()]
                aggregate[method][policy][key]=dict(mean=statistics.mean(values),std=statistics.stdev(values))
    controls={}
    for name in ['no_change','oracle']:
        control=[]
        for row in rows:
            for goal in row['goals']:
                j=0 if name=='no_change' else choose(row['outcomes'],goal)
                control.append(score(row['outcomes'],goal,j))
        controls[name]={key:statistics.mean(float(r[key]) for r in control) for key in keys}
    emit(output,dict(protocol=manifest['protocol'],physical_trajectories=manifest['physical_trajectories'],seeds=SEEDS,
        per_seed=per,aggregate=aggregate,controls=controls,
        caveat='Eight independent IC contexts, two goals each. Across-seed SD measures model-seed variation, not uncertainty over new tasks. Oracle uses finite-MC simulator outcomes.'))
    print(json.dumps(dict(aggregate=aggregate,controls=controls),indent=2));print('Saved',output,flush=True)
def smoke(args):
    verify_inputs(args);dev=base.resolve_device(args.device);sim,_=helper.simulator(args)
    gen.set_seed(GEN_SEED-1);w=base.safe_load(args.data_dir/'initial_conditions.pt')['w_star'].to(dev)
    x=(w+sim.grf.sample(1)).cpu();out=physical_changes(sim,x,0.,ACTIONS,2,2,GEN_SEED-2)
    assert out.shape==(5,1,64,64);torch.testing.assert_close(out[0],torch.zeros_like(out[0]),rtol=0,atol=0)
    for method in METHODS:
        m,mean,std=model(args,method,32,dev)
        p=forecast(m,base.normalize_state(x.unsqueeze(1),mean,std).to(dev),torch.zeros(1,1,device=dev),std,1,99)
        for v in p.values():assert v.shape==out.shape
        del m
    goal=out[-1];j=choose(out,goal);assert j==len(ACTIONS)-1
    assert score(out,goal,j)['normalized_regret']==0
    print('PASS: frozen checkpoint loading, physical CRN changes, JVP/finite forecasts, decision/scoring. Smoke simulation overhead excluded from experimental count.',flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['smoke','prepare','evaluate'])
    p.add_argument('--data-dir',type=Path,default=Path('runs/spdebench_sns_response'))
    p.add_argument('--conditional-root',type=Path,default=Path('runs/spdebench_sns_conditional_final'))
    p.add_argument('--tangent-root',type=Path,default=Path('runs/spdebench_sns_tangent_stability_b8'))
    p.add_argument('--tsbm-root',type=Path,default=Path('runs/sns_tsbm_official_v1'))
    p.add_argument('--run-root',type=Path,default=Path('runs/sns_intervention_selection_v1'))
    p.add_argument('--spdebench-root',type=Path);p.add_argument('--sim-batch',type=int,default=8);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.sim_batch<=0:p.error('sim-batch must be positive')
    globals()[args.mode](args)
if __name__=='__main__':main()
