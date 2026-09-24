"""Gaussian GSBM/TSBM with a quadratic cost and zero-cost control; no J* training."""
import argparse
import copy
import hashlib
import importlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import gaussian_reference as base
from backward_metrics import evaluate_backward_response

HERE=Path(__file__).resolve().parent
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(value,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(p)
def emit(value,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)

class Evaluator(base.ConditionalDSBM):
    @property
    def sens_steps(self):return self.cfg.num_steps
    def tangent_training_rollout_direction(self,xstart,u,noise_bank,fb):
        x=xstart.clone();r=torch.zeros_like(x);dt=1/self.cfg.num_steps
        for k,z in enumerate(noise_bank):
            tv=k*dt if fb=='f' else 1-k*dt
            t=torch.full((len(x),1),tv,device=x.device,dtype=x.dtype)
            def drift(xx,uu):return self.nets[fb](xx,uu,t)
            b,j=torch.autograd.functional.jvp(drift,(x,u),(r,torch.ones_like(u)),create_graph=False)
            x=(x+dt*b+self.cfg.reference_sigma*math.sqrt(dt)*z).detach()
            r=(r+dt*j).detach()
        return r.unsqueeze(-1)

def export(engine,cfg,device):
    m=Evaluator(cfg,2,1,device)
    for long,short in [('fwd','f'),('bwd','b')]:
        net=engine.nets[long];net.eval()
        m.nets[short].load_state_dict(net.model.net.state_dict())
    m.net_f.eval();m.net_b.eval()
    return m

def setup(args):
    engine=importlib.import_module(args.method+'_engine')
    cfg=base.Config(seed=args.seed,total_imf=7)
    cfg.base_channels=cfg.hidden  # adapter alias, Gaussian MLP is width128/depth3
    dev=torch.device(args.device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    return engine,cfg,dev

def cost_record(args):
    raw=base.safe_torch_load(args.data_dir/'train.pt')
    # A common, zero-centred state scale from observed training endpoints only.
    scale=float(.5*(raw['x0'].double().square().sum(-1).mean()+raw['xT'].double().square().sum(-1).mean()))
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid endpoint scale')
    return dict(kind=args.cost,dimensionless_strength=args.cost_strength,
        training_endpoint_second_moment=scale,
        beta=args.cost_strength/scale if args.cost=='quadratic' else 0.,
        formula='V_GSBM(x)=beta*||x||^2/2; V_TSBM(x)=sigma^2*V_GSBM(x)',
        selection='Strength fixed in advance at 0.1 by default, not tuned; scale from train.pt only')

def cost_checks(module,dev):
    x=torch.tensor([[.4,-.7],[1.2,.3]],device=dev,requires_grad=True)
    for beta in [0.,.3]:
        v=module.SpatialCost(beta,(2,))(x,None,None)
        grad=torch.autograd.grad(v.sum(),x)[0]
        torch.testing.assert_close(grad,beta*x)
    if hasattr(module,'TwistedCost'):
        B=2; sigma=.5; beta=.3
        z0=torch.randn(B,2,device=dev);z1=torch.randn_like(z0);zt=torch.randn_like(z0)
        t=torch.tensor([[.25],[.65]],device=dev);knots=torch.linspace(0,1,8,device=dev)
        xt=(1-knots[None,:,None])*z0[:,None]+knots[None,:,None]*z1[:,None]
        path=module.tgp.EndPointGaussianPath(knots,xt,knots,torch.zeros(B,8,1,device=dev),sigma,1.,module.BrownianBridgeDrift(),device=dev)
        path.eval();zs=torch.randn(B,1,2,device=dev)
        for fb in ['f','b']:
            s=(t[:,None]+1)/2 if fb=='f' else t[:,None]/2
            for b in [0.,beta]:
                cost=module.TwistedCost(b,(2,),sigma)
                torch.testing.assert_close(cost(x,None),sigma**2*module.SpatialCost(b,(2,))(x,None,None))
                sampler=module.Twisted_BM_GeneralCost(cost,'tsbm',1.,sigma,(2,),'velocity',dev)
                actual=sampler.get_train_target_from_cond_sampler(z0,z1,zt,t,fb,path,z_s=zs,s=s,s_times=s,mask_s=torch.ones_like(s,dtype=torch.bool))
                expected=(z1-zt)/(1-t) if fb=='f' else (z0-zt)/t
                weight=1-s[:,0] if fb=='f' else s[:,0]
                expected=expected-weight*b*sigma**2*zs[:,0]
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
    print('PASS: quadratic gradient, zero limit, and (TSBM) scaled analytic twisted targets.',flush=True)

def smoke(args):
    module,cfg,dev=setup(args);base.set_seed(1919)
    cost_checks(module,dev)
    cfg.num_steps=8;cfg.batch_size=4;cfg.hidden=8;cfg.base_channels=8
    opts=SimpleNamespace(**vars(args));opts.fit_steps=2;opts.fit_batch=2
    opts.match_steps=2;opts.path_mc=2;opts.path_times=4;opts.future_mc=2
    m=getattr(module,args.method.upper())(cfg,(2,),dev)
    data=dict(x0=torch.randn(4,2),x1=torch.randn(4,2),a=torch.tensor([[-1.],[-1.],[1.],[1.]]))
    optim={d:torch.optim.AdamW(n.parameters(),lr=cfg.lr,weight_decay=1e-5) for d,n in m.nets.items()}
    previous=None
    for step,d in enumerate(['fwd','bwd','fwd'],1):
        pop,_=m.fit_population(data,previous,opts,step)
        m.match(pop,d,optim[d],opts,step);previous=d
    e=export(m,cfg,dev)
    x=data['x0'][:2].to(dev);u=data['a'][:2].to(dev);nb=e._noise_bank(x)
    for fb in ['f','b']:
        j=e.tangent_training_rollout_direction(x,u,nb,fb).squeeze(-1)
        h=.01
        fd=(e.sample_sde(x,u+h,fb=fb,noise_bank=nb)-e.sample_sde(x,u-h,fb=fb,noise_bank=nb))/(2*h)
        torch.testing.assert_close(j,fd,rtol=.03,atol=2e-4)
    j=e.tangent_rollout(x,u,nb)
    torch.testing.assert_close(j,e.tangent_training_rollout_direction(x,u,nb,'f'),rtol=1e-5,atol=1e-6)
    clone=Evaluator(cfg,2,1,dev);clone.load_state_dict(e.state_dict());clone.net_f.eval();clone.net_b.eval()
    torch.testing.assert_close(e.sample_sde(x,u,noise_bank=nb),clone.sample_sde(x,u,noise_bank=nb),rtol=1e-5,atol=1e-6)
    print('PASS: fitting, both regenerated directions, matching, EMA export, checkpoint transfer, forward/backward JVP.',flush=True)

def train(args):
    module,cfg,dev=setup(args)
    cost=cost_record(args);args.beta=cost['beta']
    # Only endpoint observations are exposed to the training engine.
    raw=base.safe_torch_load(args.data_dir/'train.pt')
    data=dict(x0=raw['x0'].float(),x1=raw['xT'].float(),a=raw['u'].float());del raw
    if data['x0'].shape[1:]!=(2,) or data['a'].shape[1:]!=(1,):raise ValueError('Expected 2D state and scalar intervention')
    if set(data['a'].flatten().tolist())!={-1.,0.,1.}:raise ValueError('Expected Gaussian training anchors -1,0,1')
    directory=args.run_root/f'seed_{args.seed}';directory.mkdir(parents=True,exist_ok=True)
    protocol=dict(method=args.method,seed=args.seed,config=asdict(cfg),
        options={k:getattr(args,k) for k in ['fit_steps','fit_batch','path_mc','path_times','future_mc','gamma_grid','match_steps','beta']},
        train_sha256=sha(args.data_dir/'train.pt'),metadata_sha256=sha(args.data_dir/'metadata.json'),
        source_record=module.source_record(),adapter_sources={p.name:sha(p) for p in HERE.glob('*.py')},
        cycles=7,no_response_supervision=True,running_cost=cost,
        initialization='Independent empirical coupling within intervention anchors; learned alternating couplings thereafter',
        inference='EMA 0.999; unchanged Gaussian reference evaluator',tf32=False)
    pp=directory/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Existing training protocol differs')
    emit(protocol,pp);base.set_seed(args.seed)
    m=getattr(module,args.method.upper())(cfg,(2,),dev)
    optim={d:torch.optim.AdamW(n.parameters(),lr=cfg.lr,weight_decay=1e-5) for d,n in m.nets.items()}
    previous=None;history=[];start=0;resume=directory/'resume.pt'
    if resume.exists():
        ck=base.safe_torch_load(resume)
        if ck['protocol']!=protocol:raise ValueError('Resume provenance mismatch')
        for d in m.nets:m.nets[d].load_state_dict(ck['nets'][d]);optim[d].load_state_dict(ck['optim'][d])
        if hasattr(m,'cached_gamma'):m.cached_gamma=ck['cached_gamma']
        previous=ck['previous'];history=ck['history'];start=ck['completed_passes']
        r=ck['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
        if 'cuda' in r and torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])
    for k in range(start,14):
        direction='fwd' if k%2==0 else 'bwd'
        pop,fit=m.fit_population(data,previous,args,k+1)
        trace=m.match(pop,direction,optim[direction],args,k+1)
        previous=direction;history.append(dict(pass_id=k+1,direction=direction,fit=fit,matching=trace))
        for net in m.nets.values():net.train(True)
        save(dict(protocol=protocol,nets={d:n.state_dict() for d,n in m.nets.items()},
            optim={d:o.state_dict() for d,o in optim.items()},previous=previous,
            cached_gamma=getattr(m,'cached_gamma',{}),history=history,completed_passes=k+1,rng=base.capture_rng_state()),resume)
    evaluator=export(m,cfg,dev)
    save(dict(protocol=protocol,config=asdict(cfg),model=evaluator.state_dict(),history=history),directory/'final.pt')
    print('Saved:',directory/'final.pt',flush=True)

def evaluate(args):
    _,cfg,dev=setup(args)
    cost=cost_record(args)
    data,meta,sd,ud,cov=base.load_dataset(args.data_dir)
    if (sd,ud)!=(2,1):raise ValueError('Wrong dataset dimensions')
    per={}
    for seed in [32,42,52]:
        cp=args.run_root/f'seed_{seed}/final.pt';ck=base.safe_torch_load(cp)
        if ck['protocol']['seed']!=seed or ck['protocol']['method']!=args.method:raise ValueError('Wrong checkpoint')
        if ck['protocol']['running_cost']!=cost:raise ValueError('Wrong cost protocol')
        if ck['protocol']['train_sha256']!=sha(args.data_dir/'train.pt') or ck['protocol']['metadata_sha256']!=sha(args.data_dir/'metadata.json'):raise ValueError('Dataset mismatch')
        cfg=base.Config(**ck['config']);m=Evaluator(cfg,sd,ud,dev);m.load_state_dict(ck['model'])
        m.net_f.eval();m.net_b.eval();per[str(seed)]={}
        for i,split in enumerate(base.SPLITS[1:]):
            path=args.run_root/f'eval_{seed}_{split}.json'
            provenance=dict(checkpoint=sha(cp),data=sha(args.data_dir/f'{split}.pt'),
                evaluator=sha(HERE/'gaussian_reference.py'),backward=sha(HERE/'backward_metrics.py'),driver=sha(__file__),seed_base=99190000)
            if path.exists():
                row=json.loads(path.read_text())
                if row['provenance']!=provenance:raise ValueError('Stale evaluation')
            else:
                base.set_seed(99190000+seed*1000+i*10)
                metrics=base.evaluate_split(split,m,data[split],cov,meta,cfg,dev)
                metrics.update(evaluate_backward_response(m,data[split],meta,cfg,dev))
                row=dict(provenance=provenance,metrics=metrics);emit(row,path)
            per[str(seed)][split]=row['metrics'];print(seed,split,json.dumps(row['metrics']),flush=True)
    aggregate={}
    for split in base.SPLITS[1:]:
        aggregate[split]={}
        for key,val in per['32'][split].items():
            if isinstance(val,(int,float)):
                values=[per[str(s)][split][key] for s in [32,42,52]]
                aggregate[split][key]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
    emit(dict(method=args.method,running_cost=cost,no_response_supervision=True,per_seed=per,aggregate=aggregate),args.run_root/'results.json')
    print(json.dumps(aggregate,indent=2));print('Saved:',args.run_root/'results.json',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['smoke','train','evaluate'])
    p.add_argument('--method',required=True,choices=['gsbm','tsbm'])
    p.add_argument('--data-dir',type=Path,default=Path('runs/gaussian_nonlinear_data'))
    p.add_argument('--run-root',type=Path)
    p.add_argument('--seed',type=int,choices=[32,42,52],default=32)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cost',choices=['quadratic','zero'],default='quadratic')
    p.add_argument('--cost-strength',type=float,default=.1)
    for name,value in [('fit-steps',50),('fit-batch',128),('path-mc',4),('path-times',16),('future-mc',4),('gamma-grid',100),('match-steps',1200)]:
        p.add_argument('--'+name,type=int,default=value)
    args=p.parse_args()
    if not math.isfinite(args.cost_strength) or args.cost_strength<=0:p.error('cost-strength must be positive and finite')
    args.beta=args.cost_strength if args.cost=='quadratic' else 0.  # synthetic smoke scale=1
    if args.run_root is None:
        tag=f'quadratic_{args.cost_strength:g}'.replace('.','p') if args.cost=='quadratic' else 'zero'
        args.run_root=Path('runs')/f'gaussian_{args.method}_cost_v2'/tag
    globals()[args.mode](args)

if __name__=='__main__':main()
