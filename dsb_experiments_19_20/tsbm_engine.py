
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace, ModuleType
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
PRECISION = {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False}
import gaussian_reference as base
from fetch_official import ROOT, REVISION, sha
from fetch_tsbm import source_record as tsbm_record, ROOT as TSBM_ROOT
tsbm_record()


_tsbm_package = ModuleType('_dsb_tsbm_vendor')
_tsbm_package.__path__ = [str(TSBM_ROOT / 'bridge')]
sys.modules['_dsb_tsbm_vendor'] = _tsbm_package
from _dsb_tsbm_vendor.spline import gaussian_path as tgp
from _dsb_tsbm_vendor.spline.sde import BrownianBridgeDrift
from _dsb_tsbm_vendor.sde.diffusion_bridge import Twisted_BM_GeneralCost, get_sde_timesteps
for module in [tgp,sys.modules['_dsb_tsbm_vendor.spline.sde'],sys.modules['_dsb_tsbm_vendor.sde.diffusion_bridge']]:
    if not Path(module.__file__).resolve().is_relative_to(TSBM_ROOT.resolve()):
        raise RuntimeError('An unrelated installed bridge package shadowed official TSBM')


def source_record():
    record=json.loads((ROOT/'manifest.json').read_text())
    if record['revision']!=REVISION:raise RuntimeError('Wrong official revision')
    for name,h in record['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError(f'Official file modified: {name}')
    return {'gsbm_support':record,'tsbm':tsbm_record()}


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
    
    def __init__(self,beta,shape):
        if not math.isfinite(beta) or beta < 0: raise ValueError('Invalid cost coefficient')
        self.beta=beta
    def __call__(self,x,t,gpath):
        return .5*self.beta*x.square().sum(-1)



def fit_config(args):
    return SimpleNamespace(name='gaussian',T_mean=8,T_gamma=8,N=args.path_mc,S=args.path_times,
        nitr=args.fit_steps,optim='adam',lr_mean=.03,lr_gamma=.03,scale_by_sigma=True,IW=False)


def random_conditional_pairs(x0,x1,a):
    
    out=x1.clone()
    for anchor in torch.unique(a,dim=0):
        idx=torch.where((a==anchor).all(1))[0]
        out[idx]=x1[idx[torch.randperm(len(idx))]]
    return x0,out,a


class TwistedCost:
    
    min_t_cost=0.
    max_t_cost=1.
    def __init__(self,beta,shape,sigma):self.spatial=SpatialCost(beta*sigma**2,shape)
    def __call__(self,xt,t,cond_sampler=None,imf_train=False):
        return self.spatial(xt,t,cond_sampler)


class TSBM:
    def __init__(self,cfg,shape,dev):
        self.cfg=cfg;self.shape=tuple(shape);self.device=dev;self.sigma=cfg.reference_sigma
        self.nets={d:EMA(ConditionalField(cfg.base_channels,shape),decay=.999).to(dev) for d in ['fwd','bwd']}
        self.basedrift=BrownianBridgeDrift();self.cached_gamma={}

    def path(self,t,xt,st,ys,args):
        return tgp.EndPointGaussianPath(t,xt,st,ys,self.sigma,1.,self.basedrift,
            device=self.device,grid_ys=torch.linspace(0,1,args.gamma_grid,device=self.device))

    def fit_population(self,data,previous,args,pass_id):
        
        
        fb='f' if previous in [None,'bwd'] else 'b'
        ccfg=SimpleNamespace(N=args.path_mc,T=args.path_times,nitr=args.fit_steps,
            nitr_first_it=args.fit_steps,optim='adam',lr_mean=.03,lr_gamma=.03)
        cost=TwistedCost(args.beta,self.shape,self.sigma)
        sampler=Twisted_BM_GeneralCost(cost,'tsbm',1.,self.sigma,(math.prod(self.shape),),'velocity',self.device)
        
        
        grids,_=get_sde_timesteps(1.,self.cfg.num_steps,0.,self.device)
        ids=torch.linspace(0,self.cfg.num_steps,8,device=self.device).long()
        t=grids['f'][ids];st=torch.linspace(0,1,8,device=self.device)
        x0,x1,a=random_conditional_pairs(data['x0'],data['x1'],data['a'])
        means=[];gammas=[];records=[]
        for start in range(0,len(a),args.fit_batch):
            stop=min(start+args.fit_batch,len(a));ab=a[start:stop].to(self.device)
            z0=x0[start:stop].to(self.device).flatten(1);z1=x1[start:stop].to(self.device).flatten(1)
            if previous is None:
                xt=(1-t[None,:,None])*z0[:,None]+t[None,:,None]*z1[:,None]
            else:
                direction='f' if previous=='fwd' else 'b';net=self.nets[previous];net.eval()
                with torch.no_grad():
                    xt,_,_=sampler.sample_sde(z0 if direction=='f' else z1,
                        lambda x,t:net(x,t,ab),direction,grids,ids,verbose=False)
                if direction=='b':xt=xt.flip(1)
            finite(xt,'TSBM regenerated coupling')
            ys=torch.zeros(len(ab),8,1,device=self.device)
            if fb in self.cached_gamma:ys=self.cached_gamma[fb][start:stop].to(self.device)
            path=self.path(t,xt,st,ys,args)
            print(f'pass {pass_id}: official TSBM VI pairs {start+1}-{stop}/{len(a)}',flush=True)
            loss_fn=tgp.build_loss_fn(path,cost)
            with torch.enable_grad():
                fit=tgp.fit(ccfg,path,fb,loss_fn,cost_name='spatial_quadratic',
                    first_it=previous is None,eps=self.cfg.bridge_eps,verbose=False)
            finite(path.mean.xt,'TSBM fitted mean');finite(path.gamma.xt,'TSBM fitted variance')
            if not all(math.isfinite(float(v)) for v in fit['losses']):raise RuntimeError('Nonfinite TSBM VI loss')
            means.append(path.mean.xt.detach().cpu());gammas.append(path.gamma.xt.detach().cpu())
            records.append(dict(first=float(fit['losses'][0]),last=float(fit['losses'][-1]),trace=fit['losses'].tolist()))
            del path,fit
        self.cached_gamma[fb]=torch.cat(gammas)
        return dict(mean_t=t.cpu(),gamma_s=st.cpu(),mean_xt=torch.cat(means),gamma_xs=torch.cat(gammas),a=a),records

    def match(self,population,direction,optimizer,args,pass_id):
        net=self.nets[direction];net.train(True);trace=[];n=len(population['a'])
        fb='f' if direction=='fwd' else 'b';eps=self.cfg.bridge_eps
        cost=TwistedCost(args.beta,self.shape,self.sigma)
        sampler=Twisted_BM_GeneralCost(cost,'tsbm',1.,self.sigma,(math.prod(self.shape),),'velocity',self.device)
        for step in range(1,args.match_steps+1):
            idx=torch.randint(n,(self.cfg.batch_size,));a=population['a'][idx].to(self.device);B=len(idx);S=args.future_mc
            path=self.path(population['mean_t'].to(self.device),population['mean_xt'][idx].to(self.device),
                population['gamma_s'].to(self.device),population['gamma_xs'][idx].to(self.device),args)
            path.eval();path.gamma.build_grid()
            
            offset=torch.arange(B,device=self.device).view(B,1)/B
            t=2*eps+(1-4*eps)*torch.remainder(torch.rand(1,1,device=self.device)+offset,1)
            r=torch.remainder(torch.rand(1,S,1,device=self.device)+offset[:,None],1)
            s=t[:,None]+eps+(1-t[:,None]-2*eps)*r if fb=='f' else eps+(t[:,None]-2*eps)*r
            with torch.no_grad():
                diagonal=torch.arange(B,device=self.device)
                zt=path.sample_xt(t[:,0],1)[diagonal,0,diagonal].detach()
                zs=path.sample_s_given_t(t,zt[:,None,:],s[:,:,0][:,None,:],fb).squeeze(1).detach()
            target=sampler.get_train_target_from_cond_sampler(path.mean.xt[:,0].detach(),path.mean.xt[:,-1].detach(),
                zt,t,fb,path,z_s=zs,s=s,s_times=s,mask_s=torch.ones_like(s,dtype=torch.bool),cv_net=None).detach()
            finite(target,'twisted drift target')
            pred=net(zt,t,a)[:,None,:].expand(-1,S,-1).reshape_as(target)
            loss=(pred-target).square().mean()
            optimizer.zero_grad(set_to_none=True);finite(loss,'TSBM matching');loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),self.cfg.grad_clip)
            finite(grad,'TSBM gradient');optimizer.step();net.update_ema()
            if step==1 or step%100==0 or step==args.match_steps:
                row=dict(step=step,loss=float(loss.detach().cpu()));trace.append(row)
                print(f'pass {pass_id} {direction}: matching {step}/{args.match_steps}, loss={row["loss"]:.6g}',flush=True)
        return trace




