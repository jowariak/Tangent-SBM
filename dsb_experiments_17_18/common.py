import argparse,hashlib,json,math
from pathlib import Path
import torch
import spdebench_sns_conditional_dsbm_v2 as base
import spdebench_sns_eval_final2 as final2
SEEDS=(32,42,52)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def metadata(args):return json.loads((args.data_dir/'metadata.json').read_text())

def emit(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)

def save(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.incomplete.pt');torch.save(obj,temp);temp.replace(path)

def arguments(description,modes,root):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('mode',choices=modes)
    p.add_argument('--data-dir',type=Path,default=Path('runs/spdebench_sns_response'))
    p.add_argument('--conditional-root',type=Path,default=Path('runs/spdebench_sns_conditional_final'))
    p.add_argument('--tangent-root',type=Path,default=Path('runs/spdebench_sns_tangent_stability_b8'))
    p.add_argument('--tsbm-root',type=Path,default=Path('runs/sns_tsbm_official_v1'))
    p.add_argument('--run-root',type=Path,default=Path(root))
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,choices=SEEDS,default=32)
    return p.parse_args()

def checkpoint_path(args,method,seed):
    if method=='conditional':return args.conditional_root/'sigma_1p4'/f'seed_{seed}'/'imf_5.pt'
    if method=='tangent':return args.tangent_root/'lambda_1000'/f'seed_{seed}'/'imf_5.pt'
    return args.tsbm_root/f'seed_{seed}'/'final.pt'
def audit_tangent(ck):
    expected=dict(fork_imf=3,lambda_sens=1000,sens_batch_size=8,sens_pairs=1,sens_every=10)
    actual={k:ck.get(k) for k in expected};origin='explicit checkpoint sens_pairs'
    # The original single-pair trainer predates the sens_pairs field. Its saved
    # objective explicitly states two independent rollouts, i.e. one cross pair.
    # Do not apply this inference to multipair objectives or explicit mismatches.
    if 'sens_pairs' not in ck and ck.get('response_objective')=='conditional_mean_two_independent_rollout_cross':
        actual['sens_pairs']=1;origin='legacy two-independent-rollout objective (one cross pair)'
    for key,value in expected.items():
        if actual[key]!=value:
            raise ValueError(f'Frozen tangent check failed: {key}={ck.get(key)!r}, expected {value}; '
                             f'response_objective={ck.get("response_objective")!r}')
    return dict(settings=actual,sens_pairs_evidence=origin,response_objective=ck.get('response_objective'))
def model(args,method,seed,device):
    ck=base.safe_load(checkpoint_path(args,method,seed));cfg=base.Config(**ck['config'])
    mean,std=base.normalization_from_metadata(metadata(args))
    if cfg.seed!=seed or cfg.reference_sigma!=1.4:raise ValueError('Wrong model seed/sigma')
    m=base.ConditionalFieldDSBM(cfg,device)
    if method=='tsbm':
        if ck['completed_passes']!=10 or ck['normalization']!=metadata(args)['normalization']:raise ValueError('Wrong TSBM protocol')
        if ck['train_data_sha256']!=sha(args.data_dir/'endpoint_train.pt'):raise ValueError('Wrong TSBM dataset')
        m.net_f.load_state_dict(ck['weights']['fwd']);m.net_b.load_state_dict(ck['weights']['bwd'])
    else:
        if ck['imf']!=5:raise ValueError('Wrong IMF')
        if not math.isclose(ck['state_mean'],mean,abs_tol=1e-8) or not math.isclose(ck['state_std'],std,abs_tol=1e-8):raise ValueError('Normalization mismatch')
        if method=='tangent':
            audit=audit_tangent(ck)
            print(f'Tangent seed {seed}: {audit["sens_pairs_evidence"]}',flush=True)
        m.load_state_dict(ck['model'])
    m.net_f.eval();m.net_b.eval();return m,mean,std
