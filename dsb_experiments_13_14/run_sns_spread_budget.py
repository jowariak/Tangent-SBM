"""SNS #14: trajectory-budget matched broad endpoint coverage, without J* training.

Fresh ICs follow the training GRF law with the saved w_star; a ~ Uniform[-1,1].
One new stochastic endpoint per condition. Original normalization stays frozen.
IMF 3 -> 5, original bridge update count, shared extra dataset across seeds.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import torch
import spdebench_sns_conditional_dsbm_v2 as base
import spdebench_sns_eval_final2 as final2
import spdebench_sns_response_dataset as generator

torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

SEEDS=(32,42,52)
GEN_SEED=2026092114

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)

def atomic_save(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.incomplete.pt');torch.save(obj,tmp);tmp.replace(path)

def finite(x):
    if not torch.isfinite(x).all():raise ValueError('Nonfinite simulation/training tensor')

def metadata(args):return json.loads((args.data_dir/'metadata.json').read_text())

def budget_from_counts(counts):
    values=[counts[k] for k in ['anchor_response_conditions','collocation_response_conditions','sensitivity_mc']]
    if any(type(v) is not int or v<=0 for v in values):raise ValueError('Invalid response training counts')
    return 2*(values[0]+values[1])*values[2]

def data_protocol(args):
    meta=metadata(args);budget=budget_from_counts(meta['counts'])
    return dict(extra_trajectories=budget,generation_seed=GEN_SEED,batch_size=args.sim_batch,
        allocation='fresh training-law ICs around saved w_star; a uniform [-1,1]; one endpoint each',
        matching='two simulator trajectories per centered-FD MC pair; all training anchor+collocation labels',
        metadata_sha256=sha(args.data_dir/'metadata.json'),
        initial_conditions_sha256=sha(args.data_dir/'initial_conditions.pt'),
        original_endpoints_sha256=sha(args.data_dir/'endpoint_train.pt'),
        generator_sha256=sha(generator.__file__),adapter_sha256=sha(__file__),sim_config=meta['sim_config'])

def fork(args,seed):
    path=args.conditional_root/'sigma_1p4'/f'seed_{seed}'/'imf_3.pt'
    ck=base.safe_load(path);cfg=base.Config(**ck['config'])
    if ck['imf']!=3 or cfg.reference_sigma!=1.4 or cfg.seed!=seed:raise ValueError(f'Wrong frozen fork: {path}')
    mean,std=base.normalization_from_metadata(metadata(args))
    if not math.isclose(ck['state_mean'],mean,abs_tol=1e-8) or not math.isclose(ck['state_std'],std,abs_tol=1e-8):
        raise ValueError('Fork normalization differs from original dataset')
    if ck.get('rng_state') is None:raise ValueError('Fork is missing its RNG state')
    cfg.total_imf=5;cfg.fork_imf=3
    return path,ck,cfg,mean,std

def simulator(args):
    meta=metadata(args)
    candidates=[args.spdebench_root] if args.spdebench_root else [Path(meta.get('spdebench_root','')),Path('SPDE_hackathon'),Path('SPDEBench')]
    root=next((p for p in candidates if (p/'data_gen/src/generator_sns.py').is_file()),None)
    if root is None:raise FileNotFoundError('Pass --spdebench-root pointing to the original SPDE_hackathon checkout')
    # Match simulator code, including local modifications, when reusing chunks.
    files={str(p.relative_to(root)):sha(p) for p in sorted((root/'data_gen/src').rglob('*.py'))}
    if not files:raise FileNotFoundError(f'No simulator sources under {root}/data_gen/src')
    commit=generator.git_commit(root);expected=meta.get('spdebench_git_commit')
    if expected and expected.lower() not in ['unknown','unavailable'] and commit!=expected:
        raise ValueError(f'Simulator revision differs: {commit} versus dataset {expected}')
    solver,grf=generator.load_spdebench(root)
    sim=generator.SNSimulator(generator.SimConfig(**meta['sim_config']),solver,grf,base.resolve_device(args.device))
    return sim,dict(commit=commit,source_sha256=files)

def prepare(args):
    protocol=data_protocol(args)
    for seed in SEEDS:fork(args,seed)
    sim,source=simulator(args);protocol['simulator']=source
    directory=args.run_root/'extra_data';directory.mkdir(parents=True,exist_ok=True)
    config=directory/'protocol.json'
    if config.exists():
        if json.loads(config.read_text())!=protocol:raise ValueError('Existing extra-data protocol differs')
    else:save_json(config,protocol)
    final=directory/'endpoint_extra.pt';manifest=directory/'complete.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['sha256']!=sha(final):raise ValueError('Extra dataset hash mismatch')
        print('Reusing completed extra endpoints:',final,flush=True);return
    w=base.safe_load(args.data_dir/'initial_conditions.pt')['w_star'].to(sim.device)
    if tuple(w.shape)!=(1,sim.cfg.s,sim.cfg.s):raise ValueError('Unexpected saved w_star shape')
    total=protocol['extra_trajectories'];chunks=[]
    print(f'SpreadBudget: {total} additional simulator trajectories, shared across all three seeds',flush=True)
    for start in range(0,total,args.sim_batch):
        count=min(args.sim_batch,total-start);path=directory/f'chunk_{start:07d}.pt'
        if path.exists():row=base.safe_load(path)
        else:
            generator.set_seed(GEN_SEED+start)
            x=w+sim.grf.sample(count);a=2*torch.rand(count,device=sim.device)-1
            y=sim.endpoint(x,a);finite(x);finite(y)
            row=dict(x0=x.cpu().unsqueeze(1),xT=y.cpu().unsqueeze(1),a=a.cpu().reshape(-1,1))
            atomic_save(row,path)
        if row['a'].shape!=(count,1) or row['x0'].shape!=(count,1,sim.cfg.s,sim.cfg.s) or row['xT'].shape!=row['x0'].shape:
            raise ValueError(f'Invalid chunk {path}')
        for value in row.values():finite(value)
        chunks.append(row);print(f'Extra endpoints {start+count}/{total}',flush=True)
    atomic_save({k:torch.cat([r[k] for r in chunks]) for k in ['x0','xT','a']},final)
    save_json(manifest,dict(sha256=sha(final),count=total,protocol=protocol))
    print('Saved',final,flush=True)

def checked_extra(args):
    directory=args.run_root/'extra_data';record=json.loads((directory/'complete.json').read_text())
    expected=data_protocol(args)
    if any(record['protocol'].get(k)!=v for k,v in expected.items()):raise ValueError('Extra-data provenance differs')
    path=directory/'endpoint_extra.pt'
    if sha(path)!=record['sha256']:raise ValueError('Extra data changed after generation')
    return base.safe_load(path),record

def train(args):
    dev=base.resolve_device(args.device);path,ck,cfg,mean,std=fork(args,args.seed)
    extra,record=checked_extra(args);raw=base.safe_load(args.data_dir/'endpoint_train.pt')
    data={key:torch.cat([base.normalize_state(r[source].float(),mean,std) if key!='a' else r[source].float()
                        for r in [raw,extra]]) for key,source in [('x0','x0'),('x1','xT'),('a','a')]}
    for value in data.values():finite(value)
    directory=args.run_root/f'seed_{args.seed}'
    if directory.exists() and any(directory.iterdir()):raise FileExistsError(f'Refusing to overwrite {directory}')
    directory.mkdir(parents=True);log=base.make_logger(directory)
    model=base.ConditionalFieldDSBM(cfg,dev);model.load_state_dict(ck['model']);base.restore_rng_state(ck['rng_state'])
    protocol=dict(method='SNS SpreadBudget',seed=args.seed,config=asdict(cfg),fork_sha256=sha(path),
        extra_data=record,reference_sha256=sha(base.__file__),adapter_sha256=sha(__file__),
        original_endpoints_sha256=sha(args.data_dir/'endpoint_train.pt'),
        no_response_supervision=True,normalization=metadata(args)['normalization'],
        precision={'cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':False})
    save_json(directory/'protocol.json',protocol)
    log('Original/new/combined endpoint counts:',len(raw['a']),len(extra['a']),len(data['a']))
    log('Frozen continuation config:',json.dumps(asdict(cfg)))
    history=[]
    for imf in [4,5]:
        b=model.train_pass(data,'b',log);f=model.train_pass(data,'f',log)
        conv=base.convergence_metric(model,data,cfg,dev)
        if not all(math.isfinite(v) for v in [b['bridge_loss_last100'],f['bridge_loss_last100'],conv]):
            raise ValueError('Nonfinite training diagnostics')
        history.append(dict(imf=imf,backward=b,forward=f,train_endpoint_rmse_norm=conv))
        save=dict(model=model.state_dict(),config=asdict(cfg),imf=imf,seed=args.seed,
            state_mean=mean,state_std=std,rng_state=base.capture_rng_state(),protocol=protocol,history=history)
        atomic_save(save,directory/f'imf_{imf}.pt');log('Saved IMF',imf)
    save_json(directory/'training.json',history)

def evaluate(args):
    output=args.run_root/'FINAL2_spread_budget_mc16.json'
    if output.exists():raise FileExistsError(output)
    _,record=checked_extra(args);meta=metadata(args);mean,std=base.normalization_from_metadata(meta)
    dev=base.resolve_device(args.device);checkpoints=[]
    for seed in SEEDS:
        ckpath=args.run_root/f'seed_{seed}/imf_5.pt';ck=base.safe_load(ckpath);pr=ck['protocol']
        fp,_,cfg,_,_=fork(args,seed)
        if ck['seed']!=seed or ck['imf']!=5 or ck['config']!=asdict(cfg):raise ValueError('Wrong/incomplete checkpoint')
        if pr['extra_data']!=record or pr['fork_sha256']!=sha(fp) or pr['normalization']!=meta['normalization']:
            raise ValueError('Checkpoint data/fork provenance differs')
        if pr['adapter_sha256']!=sha(__file__) or pr['reference_sha256']!=sha(base.__file__):raise ValueError('Training source changed')
        checkpoints.append((seed,ckpath,ck))
    manifest,data=final2.load_final2_data(args.data_dir,mean,std);per={}
    for seed,ckpath,ck in checkpoints:
        model=base.ConditionalFieldDSBM(base.Config(**ck['config']),dev);model.load_state_dict(ck['model'])
        model.net_f.eval();model.net_b.eval();per[str(seed)]={'standard':{},'response_diag':{}}
        for i,split in enumerate(final2.FINAL2_FILES):
            cache=args.run_root/f'eval_seed_{seed}_{split}.json'
            provenance=dict(checkpoint_sha256=sha(ckpath),split_sha256=sha(args.data_dir/final2.FINAL2_FILES[split]),
                manifest_sha256=sha(args.data_dir/'final2_eval_manifest.json'),evaluator_sha256=sha(final2.__file__),
                adapter_sha256=sha(__file__),response_mc=16,eval_seed_base=880000)
            if cache.exists():
                result=json.loads(cache.read_text())
                if result['provenance']!=provenance:raise ValueError('Stale evaluation cache')
            else:
                print(f'Evaluating seed {seed}: {split}',flush=True);seedbase=880000+seed*100000
                base.set_seed(seedbase+i*1000)
                standard=base.evaluate_split(split,model,data[split],model.cfg,dev,mean,std,lambda *x:print(*x,flush=True))
                pred=final2.predict_mean_j(model,data[split],model.cfg,dev,std,16,seedbase+50000+i*1000)
                result=dict(provenance=provenance,standard=standard,response_diag=final2.response_diag(pred,data[split]['Jv_star_raw']))
                save_json(cache,result)
            for section in ['standard','response_diag']:per[str(seed)][section][split]=result[section]
            print(json.dumps(result),flush=True)
        del model
    agg={section:{split:{key:final2.mean_std([per[str(s)][section][split][key] for s in SEEDS])
         for key in per['32'][section][split]} for split in final2.FINAL2_FILES} for section in ['standard','response_diag']}
    save_json(output,dict(method='SNS SpreadBudget',seeds=SEEDS,per_seed=per,aggregate=agg,
        final2_manifest=manifest,extra_data=record,response_mc=16,eval_seed_base=880000))
    print(json.dumps(agg,indent=2));print('Saved',output,flush=True)

def smoke(args):
    print('Extra simulator trajectory budget:',data_protocol(args)['extra_trajectories'],flush=True)
    counts=metadata(args)['counts']
    for filename,key in [('anchor_response.pt','anchor_response_conditions'),('response_collocation.pt','collocation_response_conditions')]:
        labels=base.safe_load(args.data_dir/filename)
        if labels['Jv_samples'].shape[:2]!=(counts[key],counts['sensitivity_mc']):
            raise ValueError(f'Response sample counts disagree with metadata: {filename}')
        del labels
    for seed in SEEDS:fork(args,seed)
    torch.set_num_threads(1);base.set_seed(14);dev=base.resolve_device(args.device)
    cfg=base.Config(base_channels=4,num_steps=3,batch_size=2,inner_steps=2,reference_sigma=1.4)
    model=base.ConditionalFieldDSBM(cfg,dev)
    data=dict(x0=torch.randn(4,1,16,16),x1=torch.randn(4,1,16,16),a=torch.zeros(4,1))
    for direction in ['b','f']:model.train_pass(data,direction,lambda *x:print(*x,flush=True))
    clone=base.ConditionalFieldDSBM(cfg,dev);clone.load_state_dict(model.state_dict());clone.net_f.eval()
    x=data['x0'][:2].to(dev);a=data['a'][:2].to(dev);noise=clone._noise_bank(x)
    j=clone.tangent_direction_rollout(x,a,torch.ones_like(a),noise)
    fd=(clone.sample_sde(x,a+.001,noise_bank=noise)-clone.sample_sde(x,a-.001,noise_bank=noise))/.002
    torch.testing.assert_close(j,fd,rtol=.05,atol=.003)
    sim,_=simulator(args);generator.set_seed(GEN_SEED-1)
    w=base.safe_load(args.data_dir/'initial_conditions.pt')['w_star'].to(dev)
    y=sim.endpoint(w+sim.grf.sample(1),torch.zeros(1,device=dev));finite(y)
    if tuple(y.shape)!=(1,sim.cfg.s,sim.cfg.s):raise ValueError('Simulator output shape mismatch')
    print('PASS: forks/normalization, bridge updates, checkpoint transfer, JVP/FD and one simulator trajectory. Smoke trajectory excluded from experimental budget.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['smoke','prepare','train','evaluate'])
    p.add_argument('--data-dir',type=Path,default=Path('runs/spdebench_sns_response'))
    p.add_argument('--conditional-root',type=Path,default=Path('runs/spdebench_sns_conditional_final'))
    p.add_argument('--spdebench-root',type=Path)
    p.add_argument('--run-root',type=Path,default=Path('runs/sns_spread_budget_v1'))
    p.add_argument('--seed',type=int,choices=SEEDS,default=32)
    p.add_argument('--sim-batch',type=int,default=8)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.sim_batch<=0:p.error('--sim-batch must be positive')
    globals()[args.mode](args)

if __name__=='__main__':main()
