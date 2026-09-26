"""SNS conditional TSBM using unmodified authors' computational modules.

The single-GPU adapter calls the authors' variational solver, conditional
sampler and twisted target. Alternating, velocity, no-CV variant; EMA is an
adapter choice. No response labels in training. See README for all differences.
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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
PRECISION = {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False}
import spdebench_sns_conditional_dsbm_v2 as base
import spdebench_sns_eval_final2 as final2
from fetch_official import ROOT, REVISION, sha
from fetch_tsbm import source_record as tsbm_record, ROOT as TSBM_ROOT
if tsbm_record()['revision']!='c9dfec715266a2fd45e1ab195eb3b388fa6e8ee8':
    raise RuntimeError('Expected the same TSBM revision as experiment 11')
from types import ModuleType
_vendor = ModuleType("_dsb_tsbm_vendor")
_vendor.__path__ = [str(TSBM_ROOT / "bridge")]
sys.modules["_dsb_tsbm_vendor"] = _vendor
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
        super().__init__();self.shape=tuple(shape);self.net=base.ConditionalUNetDrift(width)

    def forward(self,x,t,a):
        return self.net(x.reshape(-1,*self.shape),a,t.reshape(-1,1)).flatten(1)


class SpatialCost:
    
    def __init__(self,beta,shape):self.beta=beta;self.shape=tuple(shape)
    def __call__(self,x,t,gpath):
        v=x.reshape(*x.shape[:-1],*self.shape)
        return .5*self.beta*((v-torch.roll(v,1,-2)).square().sum((-3,-2,-1))+
                             (v-torch.roll(v,1,-1)).square().sum((-3,-2,-1)))



def fit_config(args):
    return SimpleNamespace(name='sns',T_mean=8,T_gamma=8,N=args.path_mc,S=args.path_times,
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
            if fb in self.cached_gamma:
                previous_gamma=self.cached_gamma[fb].to(self.device)
                if len(previous_gamma)>=len(ab):ys=previous_gamma[:len(ab)].clone()
                else:ys=previous_gamma.repeat(math.ceil(len(ab)/len(previous_gamma)),1,1)[:len(ab)].clone()
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
            self.cached_gamma[fb]=path.gamma.xt.detach().cpu().clone()
            del path,fit
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


def twisted_checks(dev):
    
    shape=(1,4,4);B=2;D=math.prod(shape);sigma=1.4
    t=torch.tensor([[.25],[.65]],device=dev);s=torch.tensor([[[.6]],[[.8]]],device=dev)
    z0=torch.randn(B,D,device=dev);z1=torch.randn_like(z0);zt=torch.randn_like(z0);zs=torch.randn(B,1,D,device=dev)
    knots=torch.linspace(0,1,8,device=dev)
    xt=(1-knots[None,:,None])*z0[:,None]+knots[None,:,None]*z1[:,None]
    path=tgp.EndPointGaussianPath(knots,xt,knots,torch.zeros(B,8,1,device=dev),sigma,1.,BrownianBridgeDrift(),device=dev)
    path.eval()
    for fb in ['f','b']:
        ss=s if fb=='f' else t[:,None]*.5
        expected=(z1-zt)/(1-t) if fb=='f' else (z0-zt)/t
        for beta in [0.,1.]:
            cost=TwistedCost(beta,shape,sigma)
            sampler=Twisted_BM_GeneralCost(cost,'tsbm',1.,sigma,(D,),'velocity',dev)
            actual=sampler.get_train_target_from_cond_sampler(z0,z1,zt,t,fb,path,z_s=zs,s=ss,s_times=ss,mask_s=torch.ones_like(ss,dtype=torch.bool))
            x=zs[:,0].reshape(B,*shape)
            
            grad=4*x-torch.roll(x,1,-1)-torch.roll(x,-1,-1)-torch.roll(x,1,-2)-torch.roll(x,-1,-2)
            weight=1-ss[:,0] if fb=='f' else ss[:,0]
            truth=expected-weight*beta*sigma**2*grad.flatten(1)
            torch.testing.assert_close(actual,truth,rtol=1e-5,atol=1e-5)
    print('PASS: forward/backward zero-cost limit and analytic spatial-potential correction.',flush=True)


def options(args):
    return {k:getattr(args,k) for k in ['cycles','match_steps','fit_steps','fit_batch','path_mc','path_times','beta','future_mc','gamma_grid']}


def train(args):
    dev=torch.device(args.device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; training will not silently fall back to CPU')
    directory=args.run_root/f'seed_{args.seed}'
    if directory.exists() and any(directory.iterdir()):raise FileExistsError(f'Existing seed directory: {directory}')
    data_path=args.data_dir/'endpoint_train.pt';raw=base.safe_load(data_path)
    metadata=json.loads((args.data_dir/'metadata.json').read_text())
    mean,std=base.normalization_from_metadata(metadata)
    data={'x0':base.normalize_state(raw['x0'].float(),mean,std),
          'x1':base.normalize_state(raw['xT'].float(),mean,std),'a':raw['a'].float()}
    if tuple(data['x0'].shape[1:])!=(1,64,64) or data['x0'].shape!=data['x1'].shape or data['a'].shape!=(len(data['x0']),1):
        raise ValueError('Unexpected SNS training shape')
    for k,v in data.items():finite(v,k)
    cfg=base.Config(seed=args.seed,reference_sigma=1.4,total_imf=args.cycles,fork_imf=0,inner_steps=args.match_steps)
    base.set_seed(args.seed);model=TSBM(cfg,(1,64,64),dev)
    
    optimizer=torch.optim.AdamW([p for net in model.nets.values() for p in net.parameters() if p.requires_grad],
                               lr=cfg.lr,weight_decay=1e-5,eps=1e-8)
    directory.mkdir(parents=True,exist_ok=True);history=[];previous=None
    if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev);torch.cuda.synchronize(dev)
    started=time.perf_counter()
    print(f'Official-source TSBM: seed={args.seed}, device={dev}, from scratch, {2*args.cycles} passes',flush=True)
    for k in range(2*args.cycles):
        direction='fwd' if k%2==0 else 'bwd'
        population,reciprocal=model.fit_population(data,previous,args,k+1)
        matching=model.match(population,direction,optimizer,args,k+1)
        history.append(dict(pass_number=k+1,direction=direction,reciprocal_direction=direction,
                            reciprocal=reciprocal,matching=matching))
        previous=direction;del population
        
        
        weights={}
        for d,net in model.nets.items():
            net.eval();weights[d]={name:p.detach().cpu().clone() for name,p in net.model.net.state_dict().items()}
        checkpoint=dict(seed=args.seed,config=asdict(cfg),options=options(args),completed_passes=k+1,
            weights=weights,history=history,official_source=source_record(),normalization=metadata['normalization'],
            train_data_sha256=sha(data_path),adapter_sha256=sha(__file__),reference_sha256=sha(Path(base.__file__)),
            initialization='random weights; independent empirical endpoint coupling within each intervention anchor',
            no_response_supervision=True,precision=PRECISION,
            runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()})
        torch.save(checkpoint,directory/f'pass_{k+1:02d}.pt')
        torch.save(dict(networks={d:n.state_dict() for d,n in model.nets.items()},
                        optimizer=optimizer.state_dict(),cached_gamma=model.cached_gamma,
                        rng_state=base.capture_rng_state(),completed_passes=k+1),directory/'training_state.pt')
        temp=directory/'latest.incomplete.pt';torch.save(checkpoint,temp);temp.replace(directory/'latest.pt')
    if dev.type=='cuda':torch.cuda.synchronize(dev)
    checkpoint['train_seconds']=time.perf_counter()-started
    checkpoint['peak_allocated_bytes']=torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else None
    torch.save(checkpoint,directory/'final.pt')
    save_json(directory/'training.json',{k:v for k,v in checkpoint.items() if k!='weights'})
    print('Saved',directory/'final.pt',flush=True)


class EvalModel(base.ConditionalFieldDSBM):
    
    def __init__(self,ck,dev):
        cfg=base.Config(**ck['config']);super().__init__(cfg,dev)
        self.net_f.load_state_dict(ck['weights']['fwd']);self.net_b.load_state_dict(ck['weights']['bwd'])
        self.net_f.eval();self.net_b.eval()


def evaluate(args):
    output=args.run_root/'FINAL2_tsbm_mc16.json'
    if output.exists():raise FileExistsError(output)
    checkpoints=[base.safe_load(args.run_root/f'seed_{s}/final.pt') for s in [32,42,52]]
    first=checkpoints[0]
    for s,ck in zip([32,42,52],checkpoints):
        if ck.get('precision')!=PRECISION:raise ValueError('Checkpoint precision protocol differs from this runner')
        if ck['seed']!=s or ck['completed_passes']!=2*ck['options']['cycles']:raise ValueError('Incomplete/wrong seed')
        for key in ['options','official_source','normalization','train_data_sha256','adapter_sha256','reference_sha256']:
            if ck[key]!=first[key]:raise ValueError(f'Seed protocols differ: {key}')
        if ck['official_source']!=source_record() or ck['adapter_sha256']!=sha(__file__) or ck['reference_sha256']!=sha(base.__file__):
            raise ValueError('Source changed since training')
    metadata=json.loads((args.data_dir/'metadata.json').read_text())
    if metadata['normalization']!=first['normalization'] or sha(args.data_dir/'endpoint_train.pt')!=first['train_data_sha256']:
        raise ValueError('Dataset provenance mismatch')
    mean,std=base.normalization_from_metadata(metadata)
    manifest,data=final2.load_final2_data(args.data_dir,mean,std)
    dev=torch.device(args.device);per={}
    for ck in checkpoints:
        seed=ck['seed'];model=EvalModel(ck,dev);per[str(seed)]={'standard':{},'response_diag':{}}
        checkpoint_hash=sha(args.run_root/f'seed_{seed}/final.pt')
        for i,split in enumerate(final2.FINAL2_FILES):
            cache=args.run_root/f'eval_seed_{seed}_{split}.json'
            provenance=dict(checkpoint_sha256=checkpoint_hash,split_sha256=sha(args.data_dir/final2.FINAL2_FILES[split]),
                manifest_sha256=sha(args.data_dir/'final2_eval_manifest.json'),evaluator_sha256=sha(final2.__file__),
                adapter_sha256=sha(__file__),response_mc=16,eval_seed_base=880000)
            if cache.exists():
                saved=json.loads(cache.read_text())
                if saved['provenance']!=provenance:raise RuntimeError(f'Stale evaluation cache: {cache}')
                for section in ['standard','response_diag']:per[str(seed)][section][split]=saved[section]
                print(f'Reusing completed seed {seed} split {split}',flush=True)
                continue
            print(f'Evaluating seed {seed}: {split} endpoint/spread/finite response',flush=True)
            seed_base=880000+seed*100000
            base.set_seed(seed_base+i*1000)
            row=base.evaluate_split(split,model,data[split],model.cfg,dev,mean,std,lambda *x:print(*x,flush=True))
            per[str(seed)]['standard'][split]=row
            print(f'Evaluating seed {seed}: {split} response diagnostics, MC=16',flush=True)
            pred=final2.predict_mean_j(model,data[split],model.cfg,dev,std,16,seed_base+50000+i*1000)
            diag=final2.response_diag(pred,data[split]['Jv_star_raw'])
            per[str(seed)]['response_diag'][split]=diag
            print(json.dumps({'standard':row,'response_diag':diag}),flush=True)
            save_json(cache,{'provenance':provenance,'standard':row,'response_diag':diag})
        del model
    agg={section:{split:{k:final2.mean_std([per[s][section][split][k] for s in per])
        for k in per['32'][section][split]} for split in final2.FINAL2_FILES}
        for section in ['standard','response_diag']}
    save_json(output,dict(method='TSBM official computational modules with conditional SNS adapter',
        seeds=[32,42,52],per_seed=per,aggregate=agg,options=first['options'],official_source=first['official_source'],
        final2_manifest=manifest,evaluation_files_sha256={f:sha(args.data_dir/f) for f in final2.FINAL2_FILES.values()},
        response_mc=16,eval_seed_base=880000))
    print(json.dumps(agg,indent=2));print('Saved',output,flush=True)


def smoke(args):
    torch.set_num_threads(1);base.set_seed(11)
    cfg=base.Config(base_channels=4,num_steps=8,batch_size=2,inner_steps=2,reference_sigma=1.4)
    dev=torch.device(args.device);model=TSBM(cfg,(1,16,16),dev)
    data=dict(x0=torch.randn(4,1,16,16),x1=torch.randn(4,1,16,16),a=torch.zeros(4,1))
    tiny=SimpleNamespace(future_mc=2,gamma_grid=1001,beta=1.,fit_steps=2,fit_batch=2,path_mc=2,path_times=4,match_steps=2)
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
    adapter=base.ConditionalFieldDSBM(cfg,dev)
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
    grids,_=get_sde_timesteps(1.,cfg.num_steps,0.,dev)
    sampler=Twisted_BM_GeneralCost(TwistedCost(1.,model.shape,cfg.reference_sigma),'tsbm',1.,
        cfg.reference_sigma,(math.prod(model.shape),),'velocity',dev)
    for direction,fb in [('fwd','f'),('bwd','b')]:
        model.nets[direction].eval();base.set_seed(996)
        path,_,_=sampler.sample_sde(x.flatten(1),lambda xx,t:model.nets[direction](xx,t,a),
            fb,grids,torch.tensor([0,cfg.num_steps],device=dev))
        base.set_seed(996);ours=adapter.sample_sde(x,a,fb)
        torch.testing.assert_close(path[:,-1].reshape_as(x),ours,rtol=2e-4,atol=2e-4)
    twisted_checks(dev)
    print('PASS: official fitting/matching, both regenerated directions, EMA export, evaluator sampling and JVP.',flush=True)


def main():
    print('Numerical policy: TF32 disabled for matmul and convolutions.',flush=True)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['smoke','train','evaluate'])
    p.add_argument('--data-dir',type=Path,default=Path('runs/spdebench_sns_response'))
    p.add_argument('--run-root',type=Path,default=Path('runs/sns_tsbm_official_v1'))
    p.add_argument('--seed',type=int,choices=[32,42,52],default=32)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cycles',type=int,default=5)
    p.add_argument('--match-steps',type=int,default=800)
    p.add_argument('--fit-steps',type=int,default=150)
    p.add_argument('--fit-batch',type=int,default=2)
    p.add_argument('--path-mc',type=int,default=4)
    p.add_argument('--path-times',type=int,default=16)
    p.add_argument('--future-mc',type=int,default=4)
    p.add_argument('--gamma-grid',type=int,default=10001)
    p.add_argument('--beta',type=float,default=1.)
    a=p.parse_args()
    if min(a.cycles,a.match_steps,a.fit_steps,a.fit_batch,a.path_mc,a.path_times,a.future_mc)<1 or a.gamma_grid<101 or not math.isfinite(a.beta) or a.beta<0:
        p.error('Positive counts and finite nonnegative beta required')
    {'smoke':smoke,'train':train,'evaluate':evaluate}[a.mode](a)


if __name__=='__main__':main()
