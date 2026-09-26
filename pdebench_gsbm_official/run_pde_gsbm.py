"""PDEBench conditional GSBM using unmodified authors' computational modules.

The single-GPU driver replaces Lightning orchestration, not GSBM's path solver,
sampling, drift targets, matching loss, or EMA. No response labels in training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import torch
import pde_reference as base
from fetch_official import ROOT, REVISION, sha


def source_record():
    record=json.loads((ROOT/'manifest.json').read_text())
    if record['revision']!=REVISION:raise RuntimeError('Wrong official revision')
    for name,h in record['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError(f'Official file modified: {name}')
    return record


source_record()
sys.path.insert(0,str(ROOT))
from gsbm import gaussian_path as gp
from gsbm.match_loss import bm_loss
from gsbm.sde import sdeint, ZeroBaseDrift
from gsbm.ema import EMA
for module in [gp,sys.modules['gsbm.match_loss'],sys.modules['gsbm.sde'],sys.modules['gsbm.ema']]:
    if not Path(module.__file__).resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError('An unrelated installed gsbm package shadowed the pinned code')


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,allow_nan=False)
    with path.open('x',encoding='utf-8') as f:f.write(text)


def finite(tensor,where):
    if not bool(torch.isfinite(tensor).all()):raise RuntimeError(f'Nonfinite values: {where}')


class ConditionalField(torch.nn.Module):
    def __init__(self,width,shape):
        super().__init__();self.shape=tuple(shape);self.net=base.ConditionalUNetDrift(width)

    def forward(self,x,t,a):
        return self.net(x.reshape(-1,*self.shape),a,t.reshape(-1,1)).flatten(1)


class SpatialCost:
    
    def __init__(self,beta,shape):self.beta=beta;self.shape=tuple(shape)
    def __call__(self,x,t,gpath):
        v=x.reshape(*x.shape[:-1],*self.shape)
        return .5*self.beta*((v[...,1:,:]-v[...,:-1,:]).square().sum((-3,-2,-1))+
                             (v[...,:,1:]-v[...,:,:-1]).square().sum((-3,-2,-1)))


def fit_config(args):
    return SimpleNamespace(name='pdebench',T_mean=8,T_gamma=8,N=args.path_mc,S=args.path_times,
        nitr=args.fit_steps,optim='adam',lr_mean=.03,lr_gamma=.03,scale_by_sigma=True,IW=False)


def random_conditional_pairs(x0,x1,a):
    
    out=x1.clone()
    for anchor in torch.unique(a,dim=0):
        idx=torch.where((a==anchor).all(1))[0]
        out[idx]=x1[idx[torch.randperm(len(idx))]]
    return x0,out,a


class GSBM:
    def __init__(self,cfg,shape,dev):
        self.cfg=cfg;self.shape=tuple(shape);self.device=dev;self.sigma=cfg.reference_sigma
        self.nets={d:EMA(ConditionalField(cfg.base_channels,shape),decay=.999).to(dev) for d in ['fwd','bwd']}
        self.basedrift=ZeroBaseDrift()

    def fit_population(self,data,previous,args,pass_id):
        ccfg=fit_config(args);cost=SpatialCost(args.beta,self.shape)
        x0,x1,a=random_conditional_pairs(data['x0'],data['x1'],data['a'])
        means=[];gammas=[];records=[]
        for start in range(0,len(a),args.fit_batch):
            stop=min(start+args.fit_batch,len(a));ab=a[start:stop].to(self.device)
            z0=x0[start:stop].to(self.device).flatten(1);z1=x1[start:stop].to(self.device).flatten(1)
            if previous is None:
                t=torch.linspace(0,1,ccfg.T_mean,device=self.device)
                xt=(1-t[None,:,None])*z0[:,None]+t[None,:,None]*z1[:,None]
            else:
                net=self.nets[previous];net.eval()
                with torch.no_grad():
                    result=sdeint(z0 if previous=='fwd' else z1,
                        lambda x,t:net(x,t,ab),lambda x,t:self.sigma,
                        previous,nfe=self.cfg.num_steps,log_steps=ccfg.T_mean)
                t,xt=result['t'],result['xs']
            finite(xt,'regenerated coupling')
            st=torch.linspace(0,1,ccfg.T_gamma,device=self.device)
            raw_gamma=torch.zeros(len(ab),ccfg.T_gamma,1,device=self.device)
            path=gp.EndPointGaussianPath(t,xt,st,raw_gamma,self.sigma,self.basedrift)
            loss=gp.build_loss_fn(path,self.sigma,cost,ccfg)
            print(f'pass {pass_id}: official CondSOC pairs {start+1}-{stop}/{len(a)}',flush=True)
            
            
            with torch.enable_grad():
                fit=gp.fit(ccfg,path,previous or 'fwd',loss,verbose=False)
            finite(path.mean.xt,'fitted mean');finite(path.gamma.xt,'fitted variance')
            if not all(math.isfinite(float(v)) for v in fit['losses']):raise RuntimeError('Nonfinite CondSOC objective')
            means.append(path.mean.xt.detach().cpu());gammas.append(path.gamma.xt.detach().cpu())
            records.append(dict(first=float(fit['losses'][0]),last=float(fit['losses'][-1]),
                                trace=fit['losses'].tolist()))
            del fit,path
        return dict(mean_t=t.detach().cpu(),gamma_s=st.detach().cpu(),mean_xt=torch.cat(means),
                    gamma_xs=torch.cat(gammas),a=a),records

    def match(self,population,direction,optimizer,args,pass_id):
        net=self.nets[direction];net.train(True);trace=[];n=len(population['a'])
        for step in range(1,args.match_steps+1):
            idx=torch.randint(n,(self.cfg.batch_size,));a=population['a'][idx].to(self.device)
            path=gp.EndPointGaussianPath(population['mean_t'].to(self.device),
                population['mean_xt'][idx].to(self.device),population['gamma_s'].to(self.device),
                population['gamma_xs'][idx].to(self.device),self.sigma,self.basedrift)
            
            t=torch.rand(len(idx),device=self.device)*(1-2e-4)+1e-4
            with torch.no_grad():
                xt=path.sample_xt(t,N=1)
            vt=path.ut(t,xt,direction).detach()
            diagonal=torch.arange(len(idx),device=self.device)
            xt=xt[diagonal,0,diagonal].detach();vt=vt[diagonal,0,diagonal]
            optimizer.zero_grad(set_to_none=True)
            loss=bm_loss(lambda x,t:net(x,t,a),xt,t,vt)
            finite(loss,'bridge matching loss');loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),self.cfg.grad_clip)
            finite(grad,'matching gradient');optimizer.step();net.update_ema()
            if step==1 or step%100==0 or step==args.match_steps:
                row=dict(step=step,loss=float(loss.detach().cpu()));trace.append(row)
                print(f'pass {pass_id} {direction}: matching {step}/{args.match_steps}, loss={row["loss"]:.6g}',flush=True)
        return trace


def options(args):
    return {k:getattr(args,k) for k in ['cycles','match_steps','fit_steps','fit_batch','path_mc','path_times','beta']}


def train(args):
    dev=torch.device(args.device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; training will not silently fall back to CPU')
    directory=args.run_root/f'seed_{args.seed}'
    if directory.exists() and any(directory.iterdir()):raise FileExistsError(f'Existing seed directory: {directory}')
    data_path=args.data_dir/'train.pt';raw=base.safe_load(data_path)
    data={k:raw[r].float() for k,r in [('x0','x0_norm'),('x1','xT_norm'),('a','a')]}
    if tuple(data['x0'].shape[1:])!=(2,128,128) or data['x0'].shape!=data['x1'].shape or data['a'].shape!=(len(data['x0']),3):
        raise ValueError('Unexpected PDEBench training shape')
    for k,v in data.items():finite(v,k)
    metadata=json.loads((args.data_dir/'metadata.json').read_text())
    cfg=base.Config(seed=args.seed,total_imf=args.cycles,fork_imf=0,inner_steps=args.match_steps)
    base.set_seed(args.seed);model=GSBM(cfg,(2,128,128),dev)
    
    optimizer=torch.optim.AdamW([p for net in model.nets.values() for p in net.parameters() if p.requires_grad],
                               lr=cfg.lr,weight_decay=1e-5,eps=1e-8)
    directory.mkdir(parents=True,exist_ok=True);history=[];previous=None
    if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev);torch.cuda.synchronize(dev)
    started=time.perf_counter()
    print(f'Official-source GSBM: seed={args.seed}, device={dev}, from scratch, {2*args.cycles} passes',flush=True)
    for k in range(2*args.cycles):
        direction='fwd' if k%2==0 else 'bwd'
        population,reciprocal=model.fit_population(data,previous,args,k+1)
        matching=model.match(population,direction,optimizer,args,k+1)
        history.append(dict(pass_number=k+1,direction=direction,cond_soc_direction=previous or 'fwd',
                            reciprocal=reciprocal,matching=matching))
        previous=direction;del population
        
        
        weights={}
        for d,net in model.nets.items():
            net.eval();weights[d]={name:p.detach().cpu().clone() for name,p in net.model.net.state_dict().items()}
        checkpoint=dict(seed=args.seed,config=asdict(cfg),options=options(args),completed_passes=k+1,
            weights=weights,history=history,official_source=source_record(),normalization=metadata['normalization'],
            train_data_sha256=sha(data_path),adapter_sha256=sha(__file__),reference_sha256=sha(Path(base.__file__)),
            initialization='random weights; independent empirical endpoint coupling within each intervention anchor',
            no_response_supervision=True)
        temp=directory/'latest.incomplete.pt';torch.save(checkpoint,temp);temp.replace(directory/'latest.pt')
    if dev.type=='cuda':torch.cuda.synchronize(dev)
    checkpoint['train_seconds']=time.perf_counter()-started
    checkpoint['peak_allocated_bytes']=torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else None
    torch.save(checkpoint,directory/'final.pt')
    save_json(directory/'training.json',{k:v for k,v in checkpoint.items() if k!='weights'})
    print('Saved',directory/'final.pt',flush=True)


class EvalModel(base.ConditionalFieldDSBM):
    
    def __init__(self,ck,dev):
        cfg=base.Config(**ck['config']);super().__init__(cfg,(2,128,128),3,dev)
        self.net_f.load_state_dict(ck['weights']['fwd']);self.net_b.load_state_dict(ck['weights']['bwd'])
        self.net_f.eval();self.net_b.eval()


def evaluate(args):
    output=args.run_root/'aggregate.json'
    if output.exists():raise FileExistsError(output)
    checkpoints=[base.safe_load(args.run_root/f'seed_{s}/final.pt') for s in [32,42,52]]
    first=checkpoints[0]
    for s,ck in zip([32,42,52],checkpoints):
        if ck['seed']!=s or ck['completed_passes']!=2*ck['options']['cycles']:raise ValueError('Incomplete/wrong seed')
        for key in ['options','official_source','normalization','train_data_sha256','adapter_sha256','reference_sha256']:
            if ck[key]!=first[key]:raise ValueError(f'Seed protocols differ: {key}')
        if ck['official_source']!=source_record() or ck['adapter_sha256']!=sha(__file__) or ck['reference_sha256']!=sha(base.__file__):
            raise ValueError('Source changed since training')
    endpoint,response,metadata,_,_=base.load_dataset(args.data_dir)
    if metadata['normalization']!=first['normalization'] or sha(args.data_dir/'train.pt')!=first['train_data_sha256']:
        raise ValueError('Dataset provenance mismatch')
    dev=torch.device(args.device);per={}
    for ck in checkpoints:
        seed=ck['seed'];model=EvalModel(ck,dev);per[str(seed)]={}
        for i,split in enumerate(base.RESP_FILES):
            print(f'Evaluating seed {seed}: {split}',flush=True)
            base.set_seed(880000+seed*100000+i*1000)
            per[str(seed)][split]=base.evaluate_split(split,model,endpoint[split],response[split],model.cfg,dev,metadata)
            print(json.dumps(per[str(seed)][split]),flush=True)
        del model
    agg={split:{k:dict(mean=statistics.mean(per[s][split][k] for s in per),
                      std=statistics.stdev(per[s][split][k] for s in per)) for k in per['32'][split]} for split in base.RESP_FILES}
    save_json(output,dict(method='GSBM official computational modules with conditional PDEBench adapter',
        seeds=[32,42,52],per_seed=per,aggregate=agg,options=first['options'],official_source=first['official_source'],
        evaluation_note='Existing PDEBench Euler/CRN/JVP evaluator; endpoint all conditions, response first eval_batch_size; fixed new RNG, not paired to historical results.'))
    print(json.dumps(agg,indent=2));print('Saved',output,flush=True)


def smoke(args):
    torch.set_num_threads(1);base.set_seed(11)
    cfg=base.Config(base_channels=4,num_steps=8,batch_size=2,inner_steps=2)
    dev=torch.device(args.device);model=GSBM(cfg,(2,16,16),dev)
    data=dict(x0=torch.randn(4,2,16,16),x1=torch.randn(4,2,16,16),a=torch.zeros(4,3))
    tiny=SimpleNamespace(beta=1.,fit_steps=2,fit_batch=2,path_mc=2,path_times=4,match_steps=2)
    opt=torch.optim.AdamW([p for net in model.nets.values() for p in net.parameters() if p.requires_grad],lr=cfg.lr)
    previous=None
    for k,d in enumerate(['fwd','bwd','fwd']):
        pop,_=model.fit_population(data,previous,tiny,k+1)
        before=[p.detach().clone() for p in model.nets[d].model.parameters()]
        model.match(pop,d,opt,tiny,k+1)
        if not any(not torch.equal(x,y) for x,y in zip(before,model.nets[d].model.parameters())):raise RuntimeError('No network update')
        previous=d
    
    weights={}
    for d,net in model.nets.items():
        net.eval();weights[d]=net.model.net.state_dict()
    adapter=base.ConditionalFieldDSBM(cfg,(2,16,16),3,dev)
    adapter.net_f.load_state_dict(weights['fwd']);adapter.net_b.load_state_dict(weights['bwd'])
    adapter.net_f.eval();adapter.net_b.eval()
    x=data['x0'][:2].to(dev);a=data['a'][:2].to(dev);d=torch.ones_like(a);nb=adapter._noise_bank(x)
    j=adapter.tangent_direction_rollout(x,a,d,nb);eps=.001
    fd=(adapter.sample_sde(x,a+eps*d,noise_bank=nb)-adapter.sample_sde(x,a-eps*d,noise_bank=nb))/(2*eps)
    finite(j,'evaluation JVP');finite(fd,'evaluation FD')
    torch.testing.assert_close(j,fd,rtol=.05,atol=.003)
    
    
    model.nets['fwd'].eval();base.set_seed(998)
    official=sdeint(x.flatten(1),lambda xx,t:model.nets['fwd'](xx,t,a),lambda x,t:cfg.reference_sigma,
                    'fwd',nfe=cfg.num_steps,log_steps=2)['xs'][:,-1].reshape_as(x)
    base.set_seed(998);ours=adapter.sample_sde(x,a)
    torch.testing.assert_close(official,ours,rtol=2e-4,atol=2e-4)
    print('PASS: official fitting/matching, both regenerated directions, EMA export, evaluator sampling and JVP.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['smoke','train','evaluate'])
    p.add_argument('--data-dir',type=Path,default=Path('runs/pdebench_reaction_diffusion_official'))
    p.add_argument('--run-root',type=Path,default=Path('runs/pdebench_gsbm_official_v1'))
    p.add_argument('--seed',type=int,choices=[32,42,52],default=32)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cycles',type=int,default=5)
    p.add_argument('--match-steps',type=int,default=800)
    p.add_argument('--fit-steps',type=int,default=150)
    p.add_argument('--fit-batch',type=int,default=2)
    p.add_argument('--path-mc',type=int,default=4)
    p.add_argument('--path-times',type=int,default=16)
    p.add_argument('--beta',type=float,default=1.)
    a=p.parse_args()
    if min(a.cycles,a.match_steps,a.fit_steps,a.fit_batch,a.path_mc,a.path_times)<1 or not math.isfinite(a.beta) or a.beta<0:
        p.error('Positive counts and finite nonnegative beta required')
    {'smoke':smoke,'train':train,'evaluate':evaluate}[a.mode](a)


if __name__=='__main__':main()
