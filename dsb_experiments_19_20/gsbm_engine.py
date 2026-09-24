"""Gaussian conditional GSBM using unmodified authors' computational modules.

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
import gaussian_reference as base
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
        super().__init__()
        self.net=base.ConditionalDriftNet(shape[0],1,width,3)
    def forward(self,x,t,a):
        return self.net(x,a,t.reshape(-1,1))



class SpatialCost:
    """Quadratic state cost; beta is normalized using training endpoints only."""
    def __init__(self,beta,shape):
        if not math.isfinite(beta) or beta < 0: raise ValueError('Invalid cost coefficient')
        self.beta=beta
    def __call__(self,x,t,gpath):
        return .5*self.beta*x.square().sum(-1)



def fit_config(args):
    return SimpleNamespace(name='gaussian',T_mean=8,T_gamma=8,N=args.path_mc,S=args.path_times,
        nitr=args.fit_steps,optim='adam',lr_mean=.03,lr_gamma=.03,scale_by_sigma=True,IW=False)


def random_conditional_pairs(x0,x1,a):
    """Independent empirical coupling within each exact intervention anchor."""
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
            # Match upstream validation_step: optimize the coupling's direction,
            # then use the fitted path to match the next direction.
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
            # Same sampling and diagonal pairing as upstream sample_gpath (BM).
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

