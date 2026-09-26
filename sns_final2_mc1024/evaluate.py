"""Fixed-budget FINAL2 response reevaluation; frozen checkpoints, MC1024."""
import argparse
import json
from pathlib import Path
import torch
import model_loader as loader
import spdebench_sns_eval_final2 as final2
base=loader.base
METHODS=['conditional','tangent','gsbm','tsbm']
SEEDS=[32,42,52]
MC=1024

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('runs/spdebench_sns_response'))
    for name,root in [('conditional','spdebench_sns_conditional_final'),('tangent','spdebench_sns_tangent_stability_b8'),('gsbm','sns_gsbm_official_v1'),('tsbm','sns_tsbm_official_v1')]:
        p.add_argument('--'+name+'-root',type=Path,default=Path('runs')/root)
    p.add_argument('--development-results',type=Path,default=Path('runs/sns_mc_diagnostic_1024_v1/results.json'))
    p.add_argument('--output-dir',type=Path,default=Path('runs/sns_final2_response_mc1024_v1'))
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=base.resolve_device(args.device)
    previous=json.loads(args.development_results.read_text())['protocol']
    hashes={f'{m}/{s}':loader.sha(loader.checkpoint(args,m,s)) for m in METHODS for s in SEEDS}
    if hashes!=previous['checkpoints']:raise ValueError('Checkpoints differ from development diagnostic')
    if loader.sha(args.data_dir/'metadata.json')!=previous['metadata_sha256']:raise ValueError('Normalization metadata changed')
    if loader.sha(args.data_dir/'validation.pt')!=previous['validation_sha256']:raise ValueError('Development data changed')
    for name in ['frozen_common.py','spdebench_sns_conditional_dsbm_v2.py','spdebench_sns_eval_final2.py']:
        if loader.sha(Path(__file__).parent/name)!=previous['sources'][name]:raise ValueError('Model/evaluator source changed')
    meta=json.loads((args.data_dir/'metadata.json').read_text());mean,std=base.normalization_from_metadata(meta)
    manifest,data=final2.load_final2_data(args.data_dir,mean,std)
    protocol=dict(methods=METHODS,seeds=SEEDS,response_mc=MC,batch_size=2,device=str(device),
        checkpoints=hashes,metadata_sha256=loader.sha(args.data_dir/'metadata.json'),
        final2_manifest_sha256=loader.sha(args.data_dir/'final2_eval_manifest.json'),
        data_hashes={s:loader.sha(args.data_dir/f) for s,f in final2.FINAL2_FILES.items()},
        sources={f.name:loader.sha(f) for f in Path(__file__).parent.glob('*.py')},
        development_results_sha256=loader.sha(args.development_results),sampling_seed_base=880000,
        note='Revised-budget response evaluation after development MC diagnostic. No retraining, calibration or selection. Original MC16 results retained. Endpoint and finite-intervention metrics not recomputed.')
    args.output_dir.mkdir(parents=True,exist_ok=True);pp=args.output_dir/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Existing output protocol differs')
    loader.write(pp,protocol)
    results={m:{} for m in METHODS}
    for method in METHODS:
        for seed in SEEDS:
            model,_,scale=loader.model(args,method,seed,meta,device);results[method][str(seed)]={}
            for split_id,(split,d) in enumerate(data.items()):
                chunks=[];n=len(d['x0'])
                for start in range(0,n,2):
                    cache=args.output_dir/f'{method}_{seed}_{split}_batch{start}.pt'
                    if cache.exists():
                        pred=base.safe_load(cache)
                    else:
                        
                        base.set_seed(880000+seed*100000+50000+split_id*1000+start)
                        x=d['x0'][start:start+2].to(device);a=d['a'][start:start+2].to(device);v=d['direction'][start:start+2].to(device)
                        total=torch.zeros_like(x,device='cpu',dtype=torch.float64)
                        for draw in range(1,MC+1):
                            with torch.enable_grad():j=model.tangent_direction_rollout(x,a,v).detach().cpu().double()*scale
                            if not torch.isfinite(j).all():raise ValueError('Nonfinite response')
                            total+=j
                            if draw%64==0:print(f'{method} seed={seed} {split} cases {start+1}-{min(start+2,n)}/{n}: MC {draw}/{MC}',flush=True)
                        pred=(total/MC).float();tmp=cache.with_suffix('.tmp');torch.save(pred,tmp);tmp.replace(cache)
                    chunks.append(pred)
                pred=torch.cat(chunks);truth=d['Jv_star_raw']
                diag=final2.response_diag(pred,truth)
                results[method][str(seed)][split]=diag
                loader.write(args.output_dir/f'{method}_{seed}_{split}.json',diag)
                print(method,seed,split,json.dumps(diag),flush=True)
            del model
    aggregate={m:{s:{k:final2.mean_std([results[m][str(seed)][s][k] for seed in SEEDS]) for k in results[m]['32'][s]} for s in data} for m in METHODS}
    zero={s:final2.response_diag(torch.zeros_like(d['Jv_star_raw']),d['Jv_star_raw']) for s,d in data.items()}
    loader.write(args.output_dir/'results.json',dict(protocol=protocol,manifest=manifest,per_seed=results,aggregate=aggregate,zero_response=zero))
    print(json.dumps(aggregate,indent=2));print('Saved:',args.output_dir/'results.json',flush=True)

if __name__=='__main__':main()
