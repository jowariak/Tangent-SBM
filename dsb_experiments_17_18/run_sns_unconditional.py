"""Pooled, intervention-blind DSBM; no response training or model selection."""
from dataclasses import asdict
from common import *

class UnconditionalDrift(base.ConditionalUNetDrift):
    def __init__(self,width):
        super().__init__(width)
        self.e1=base.Block(2,width)  
    def forward(self,x,a,t):
        B,_,H,W=x.shape
        time=t.reshape(B,1,1,1).expand(-1,-1,H,W)
        e1=self.e1(torch.cat([x,time],1))
        e2=self.e2(self.d1(e1));e3=self.e3(self.d2(e2));m=self.mid(self.d3(e3))
        y=self.c3(torch.cat([self.u3(m),e3],1))
        y=self.c2(torch.cat([self.u2(y),e2],1))
        return self.out(self.c1(torch.cat([self.u1(y),e1],1)))

class UnconditionalDSBM(base.ConditionalFieldDSBM):
    def __init__(self,cfg,device):
        self.cfg,self.device=cfg,device
        self.net_f=UnconditionalDrift(cfg.base_channels).to(device)
        self.net_b=UnconditionalDrift(cfg.base_channels).to(device)
        self.nets={'f':self.net_f,'b':self.net_b};self.prev_fb=None
    def regenerate_coupling(self,data):
        if self.prev_fb is not None:return super().regenerate_coupling(data)
        x0=data['x0'].to(self.device);x1=data['x1'].to(self.device)
        
        order=torch.randperm(len(x1),device=self.device)
        return x0,x1[order],torch.zeros(len(x0),1,device=self.device)
    def tangent_direction_rollout(self,x0,a,direction,noise_bank=None):
        
        return torch.zeros_like(x0)

def setup(args,seed):
    cp=checkpoint_path(args,'conditional',seed);ck=base.safe_load(cp)
    cfg=base.Config(**ck['config']);mean,std=base.normalization_from_metadata(metadata(args))
    if cfg.seed!=seed or cfg.reference_sigma!=1.4 or ck['imf']!=5:raise ValueError('Wrong reference checkpoint')
    if not math.isclose(mean,ck['state_mean'],abs_tol=1e-8) or not math.isclose(std,ck['state_std'],abs_tol=1e-8):raise ValueError('Wrong normalization')
    cfg.total_imf=5
    protocol=dict(method='SNS pooled unconditional DSBM',config=asdict(cfg),seed=seed,
        reference_config_sha256=sha(cp),training_sha256=sha(args.data_dir/'endpoint_train.pt'),
        metadata_sha256=sha(args.data_dir/'metadata.json'),adapter_sha256=sha(__file__),
        base_sha256=sha(base.__file__),common_sha256=sha(Path(__file__).with_name('common.py')),
        state_mean=mean,state_std=std,initialization='random from seed; no conditional weights copied',
        coupling='independent empirical marginal coupling, then alternating learned couplings',
        no_intervention_input=True,no_response_supervision=True,tf32=False)
    return cfg,mean,std,protocol

def train(args):
    cfg,mean,std,protocol=setup(args,args.seed);dev=base.resolve_device(args.device)
    directory=args.run_root/f'seed_{args.seed}';directory.mkdir(parents=True,exist_ok=True)
    pp=directory/'protocol.json'
    if pp.exists():
        if json.loads(pp.read_text())!=protocol:raise ValueError('Existing run provenance mismatch')
    else:emit(pp,protocol)
    raw=base.safe_load(args.data_dir/'endpoint_train.pt')
    data={'x0':base.normalize_state(raw['x0'].float(),mean,std),'x1':base.normalize_state(raw['xT'].float(),mean,std),'a':torch.zeros_like(raw['a'].float())}
    base.set_seed(args.seed);m=UnconditionalDSBM(cfg,dev);history=[];start=1
    completed=[i for i in range(1,6) if (directory/f'imf_{i}.pt').exists()]
    if completed:
        ck=base.safe_load(directory/f'imf_{max(completed)}.pt')
        if ck['protocol']!=protocol:raise ValueError('Checkpoint provenance mismatch')
        m.load_state_dict(ck['model']);base.restore_rng_state(ck['rng_state']);history=ck['history'];start=ck['imf']+1
    log=base.make_logger(directory)
    for imf in range(start,6):
        print(f'Seed {args.seed}, IMF {imf}/5',flush=True)
        b=m.train_pass(data,'b',log);f=m.train_pass(data,'f',log)
        if not all(math.isfinite(x['bridge_loss_last100']) for x in [b,f]):raise ValueError('Nonfinite loss')
        history.append(dict(imf=imf,backward=b,forward=f))
        save(dict(model=m.state_dict(),config=asdict(cfg),imf=imf,protocol=protocol,history=history,rng_state=base.capture_rng_state()),directory/f'imf_{imf}.pt')
    print('Training complete:',directory,flush=True)

def evaluate(args):
    dev=base.resolve_device(args.device);per={};manifest=None
    for seed in SEEDS:
        cfg,mean,std,protocol=setup(args,seed)
        cp=args.run_root/f'seed_{seed}/imf_5.pt';ck=base.safe_load(cp)
        if ck['protocol']!=protocol or ck['imf']!=5:raise ValueError('Wrong training provenance')
        m=UnconditionalDSBM(cfg,dev);m.load_state_dict(ck['model']);m.net_f.eval();m.net_b.eval()
        manifest,data=final2.load_final2_data(args.data_dir,mean,std)
        per[str(seed)]={'standard':{},'response_diag':{}}
        for i,split in enumerate(final2.FINAL2_FILES):
            cache=args.run_root/f'eval_{seed}_{split}.json'
            provenance=dict(checkpoint=sha(cp),data=sha(args.data_dir/final2.FINAL2_FILES[split]),manifest=sha(args.data_dir/'final2_eval_manifest.json'),evaluator=sha(final2.__file__),protocol=protocol,eval_seed_base=880000,response_mc=16)
            if cache.exists():
                r=json.loads(cache.read_text())
                if r['provenance']!=provenance:raise ValueError('Stale evaluation cache')
            else:
                print(f'Evaluating seed {seed}: {split}',flush=True);base.set_seed(880000+seed*100000+i*1000)
                standard=base.evaluate_split(split,m,data[split],cfg,dev,mean,std,lambda *x:print(*x,flush=True))
                diag=final2.response_diag(torch.zeros_like(data[split]['Jv_star_raw']),data[split]['Jv_star_raw'])
                r=dict(provenance=provenance,standard=standard,response_diag=diag);emit(cache,r)
            for section in per[str(seed)]:per[str(seed)][section][split]=r[section]
            print(json.dumps(r),flush=True)
        del m
    aggregate={section:{split:{key:final2.mean_std([per[str(s)][section][split][key] for s in SEEDS]) for key in per['32'][section][split]} for split in final2.FINAL2_FILES} for section in ['standard','response_diag']}
    out=args.run_root/'FINAL2_unconditional_dsbm.json'
    result=dict(method='SNS pooled unconditional DSBM',per_seed=per,aggregate=aggregate,manifest=manifest,seeds=SEEDS,response_note='Exactly zero by architecture; zero-vector cosine reported by shared evaluator is a convention, not defined alignment.')
    if not out.exists():emit(out,result)
    elif json.loads(out.read_text())!=json.loads(json.dumps(result)):raise ValueError('Existing aggregate differs')
    print(json.dumps(aggregate,indent=2));print('Saved:',out,flush=True)

def smoke(args):
    for seed in SEEDS:setup(args,seed)
    dev=base.resolve_device(args.device);base.set_seed(18)
    cfg=base.Config(base_channels=4,num_steps=3,inner_steps=2,batch_size=2,reference_sigma=1.4)
    m=UnconditionalDSBM(cfg,dev)
    data=dict(x0=torch.randn(4,1,16,16),x1=torch.randn(4,1,16,16),a=torch.zeros(4,1))
    for fb in ['b','f']:m.train_pass(data,fb,lambda *x:print(*x,flush=True))
    m.net_f.eval();x=data['x0'][:2].to(dev);a=torch.zeros(2,1,device=dev);nb=m._noise_bank(x)
    y=m.sample_sde(x,a,noise_bank=nb);other=m.sample_sde(x,a+1,noise_bank=nb)
    
    
    torch.testing.assert_close(y,other,rtol=1e-5,atol=1e-6)
    j=base.ConditionalFieldDSBM.tangent_direction_rollout(m,x,a,torch.ones_like(a),nb)
    torch.testing.assert_close(j,torch.zeros_like(j),rtol=0,atol=0)
    clone=UnconditionalDSBM(cfg,dev);clone.load_state_dict(m.state_dict());clone.net_f.eval()
    torch.testing.assert_close(y,clone.sample_sde(x,a,noise_bank=nb),rtol=1e-5,atol=1e-6)
    assert torch.isfinite(y).all()
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        cpu=UnconditionalDSBM(cfg,torch.device('cpu'))
        cpu.load_state_dict(m.state_dict());cpu.net_f.eval()
        cx=x.cpu();ca=a.cpu();cn=[n.cpu() for n in nb]
        cy=cpu.sample_sde(cx,ca,noise_bank=cn)
        torch.testing.assert_close(cy,cpu.sample_sde(cx,ca+1,noise_bank=cn),rtol=0,atol=0)
        cj=base.ConditionalFieldDSBM.tangent_direction_rollout(cpu,cx,ca,torch.ones_like(ca),cn)
        torch.testing.assert_close(cj,torch.zeros_like(cj),rtol=0,atol=0)
    finally:
        torch.set_num_threads(threads)
    print('PASS: both bridge updates, device checkpoint transfer, exact CPU intervention invariance and exact actual JVP=0 on CPU/device.',flush=True)

if __name__=='__main__':
    args=arguments(__doc__,['smoke','train','evaluate'],'runs/sns_unconditional_dsbm_v1');globals()[args.mode](args)
