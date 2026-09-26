#!/usr/bin/env python3
"""Frozen-checkpoint endpoint, sensitivity, and finite-change MC evaluation."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import gaussian_reference as gaussian
import double_well_reference as double_well

DEFAULTS = {
    'gaussian': {
        'data': 'runs/gaussian_nonlinear_data',
        'models': {
            'conditional': 'runs/gaussian_conditional_nonlinear_imf7/seed_{seed}/final_model.pt',
            'gsbm': 'runs/gaussian_gsbm_cost_v2/quadratic_0p1/seed_{seed}/final.pt',
            'tsbm': 'runs/gaussian_tsbm_cost_v2/quadratic_0p1/seed_{seed}/final.pt',
            'tangent': 'runs/gaussian_ablation_lambda_0p50/seed_{seed}/final_model.pt',
        },
    },
    'double_well': {
        'data': 'runs/double_well_data',
        'models': {
            'conditional': 'runs/double_well_conditional/seed_{seed}/final_model.pt',
            'gsbm': 'runs/double_well_gsbm/beta_1/seed_{seed}/final_model.pt',
            'tsbm': 'runs/double_well_tsbm/beta_1/seed_{seed}/final_model.pt',
            'sobolev': 'runs/double_well_tangent_naive_single_mse/seed_{seed}/final_model.pt',
            'tangent': 'runs/double_well_tangent_lam0p25/seed_{seed}/final_model.pt',
        },
    },
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def noise_for_draw(dataset, split, seed, draw, shape):
    
    key = f'lowdim-mc-v1/{dataset}/{split}/{seed}/{draw}'
    s = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63 - 1)
    g = torch.Generator(device='cpu').manual_seed(s)
    return torch.randn(shape, generator=g)


def metrics(total, squares, count, target):
    mean = total / count
    error = mean - target
    denom = target.norm().clamp_min(1e-8)
    variance = ((squares - total.square() / count) / (count - 1)).clamp_min(0)
    return {
        'mc': count,
        'E_J': (error.norm() / denom).item(),
        'jacobian_rmse': error.square().mean().sqrt().item(),
        'estimated_mc_noise_relative': ((variance.sum() / count).sqrt() / denom).item(),
        'zero_response_E_J': (target.norm() / denom).item(),
    }


def output_metrics(total, squares, count, target, kind):
    if kind == 'sensitivity':
        return metrics(total, squares, count, target)
    error = total / count - target
    variance = ((squares - total.square() / count) / (count - 1)).clamp_min(0)
    name = 'mean_rmse' if kind == 'endpoint' else 'finite_response_rmse'
    return dict(mc=count, **{name: error.square().mean().sqrt().item()},
                estimated_mc_noise_rmse=(variance.mean()/count).sqrt().item())


def sample_quantity(model, x, u, noise, kind, delta):
    if kind == 'sensitivity':
        with torch.enable_grad():
            return model.tangent_rollout(x, u, noise_bank=noise)
    with torch.no_grad():
        y0 = model.sample_sde(x, u, fb='f', noise_bank=noise)
        if kind == 'endpoint':
            return y0
        y1 = model.sample_sde(x, u + delta, fb='f', noise_bank=noise)
        return y1 - y0


def evaluate(args, module, cfg, model, raw, metadata, provenance, path):
    kind = provenance['quantity']
    n = len(raw['x0']) if kind == 'endpoint' else min(cfg.eval_batch_size, len(raw['x0']))
    x = raw['x0'][:n].float().to(args.device)
    u = raw['u'][:n].float().to(args.device)
    delta = cfg.finite_delta * torch.ones(u.shape[1], device=x.device) / (u.shape[1]**0.5)
    if kind == 'sensitivity':
        key = 'J_star' if args.dataset == 'gaussian' else 'J_star_mean'
        target = raw[key][:n].double().cpu()
        expected_shape = (n, x.shape[1], u.shape[1])
    elif kind == 'endpoint':
        target = raw['true_conditional_mean'].double().cpu()
        expected_shape = (n, x.shape[1])
    else:
        if args.dataset == 'gaussian':
            target = (module.oracle_mean(x, u+delta, metadata)-module.oracle_mean(x,u,metadata)).double().cpu()
        else:
            target = raw['true_finite_response'][:n].double().cpu()
        expected_shape = (n, x.shape[1])
    assert target.shape == expected_shape, (target.shape, expected_shape)
    cache = path.with_suffix('.pt')
    count, records = 0, []
    total, squares = torch.zeros_like(target), torch.zeros_like(target)
    if cache.exists():
        saved = module.safe_torch_load(cache)
        if saved['provenance'] != provenance:
            raise RuntimeError(f'Changed inputs/protocol: use a different --out directory: {cache}')
        count, records = saved['count'], saved['records']
        total, squares = saved['total'], saved['squares']
    start, initial = time.monotonic(), count
    for limit in args.mc:
        while count < limit:
            k = min(args.draw_batch, limit - count)
            
            noises = torch.stack([
                noise_for_draw(args.dataset, provenance['split']+'/'+kind, provenance['seed'], d,
                               (cfg.num_steps, n, x.shape[1]))
                for d in range(count, count + k)
            ], dim=1)
            chunks = []
            for begin in range(0,n,args.condition_batch):
                end = min(begin+args.condition_batch,n)
                noise = noises[:,:,begin:end].reshape(cfg.num_steps,k*(end-begin),x.shape[1]).to(args.device)
                value = sample_quantity(model,x[begin:end].repeat(k,1),u[begin:end].repeat(k,1),noise,kind,delta)
                chunks.append(value.detach().reshape(k,end-begin,*target.shape[1:]).double().cpu())
            j = torch.cat(chunks,dim=1)
            if not torch.isfinite(j).all():
                raise RuntimeError('Nonfinite response samples; evaluation stopped.')
            total += j.sum(0)
            squares += j.square().sum(0)
            count += k
            if count == limit:
                records.append(output_metrics(total, squares, count, target, kind))
            if count % 64 == 0 or count == limit:
                tmp = cache.with_suffix('.tmp')
                torch.save(dict(provenance=provenance, count=count, records=records,
                                total=total, squares=squares), tmp)
                tmp.replace(cache)
                print(f"{provenance['method']} seed={provenance['seed']} {provenance['split']} {kind} "
                      f"MC {count}/{args.mc[-1]} elapsed={time.monotonic()-start:.1f}s", flush=True)
    result = dict(provenance=provenance, conditions=n, results=records,
                  seconds_this_invocation=time.monotonic()-start, draws_this_invocation=count-initial)
    write_json(path, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=list(DEFAULTS), required=True)
    p.add_argument('--manifest', type=Path, help='Optional JSON with the same structure as paths.json')
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--out', type=Path, default=Path('runs/lowdim_all_metrics_mc1024'))
    p.add_argument('--methods', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[32, 42, 52])
    p.add_argument('--splits', nargs='+', choices=gaussian.SPLITS[1:], default=['test_id'])
    p.add_argument('--mc', nargs='+', type=int, default=[8,16,32,64,128,256,512,1024])
    p.add_argument('--draw-batch', type=int, default=8)
    p.add_argument('--condition-batch', type=int, default=256)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--check-only', action='store_true', help='Validate data and load every selected checkpoint, then exit')
    args = p.parse_args()
    args.mc = sorted(set(args.mc))
    if min(args.mc) < 2 or args.draw_batch < 1 or args.condition_batch < 1:
        p.error('MC must be >=2 and draw-batch >=1')
    spec = (json.loads(args.manifest.read_text()) if args.manifest else DEFAULTS)[args.dataset]
    methods = args.methods or list(spec['models'])
    if set(methods) - set(spec['models']):
        p.error('Unknown method for this dataset')
    module = gaussian if args.dataset == 'gaussian' else double_well
    jobs = [(method, seed, args.root / spec['models'][method].format(seed=seed))
            for method in methods for seed in args.seeds]
    meta_path = args.root/spec['data']/'metadata.json'
    files = [meta_path] + [cp for _, _, cp in jobs] + [args.root/spec['data']/f'{s}.pt' for s in args.splits]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        p.error('Missing files (edit paths.json and pass --manifest if paths differ):\n'+'\n'.join(missing))
    raw = {s: module.safe_torch_load(args.root/spec['data']/f'{s}.pt') for s in args.splits}
    metadata = json.loads(meta_path.read_text())
    hashes = {s: sha(args.root/spec['data']/f'{s}.pt') for s in args.splits}
    first = raw[args.splits[0]]
    sd, ud = first['x0'].shape[1], first['u'].shape[1]
    fields = {f.name for f in dataclasses.fields(module.Config)}
    prepared = []
    for method, seed, cp in jobs:
        ck = module.safe_torch_load(cp)
        if set(ck['config']) - fields:
            raise ValueError(f'Unrecognized checkpoint config fields: {cp}')
        cfg = module.Config(**ck['config'])
        if args.dataset == 'double_well' and abs(float(metadata['finite_delta'])-cfg.finite_delta)>1e-12:
            raise ValueError('Saved finite-change targets use a different displacement from the checkpoint.')
        if cfg.seed != seed:
            raise ValueError(f'Checkpoint seed mismatch: {cp}')
        model = module.ConditionalDSBM(cfg, sd, ud, torch.device('cpu'))
        model.load_state_dict(ck['model'])
        prepared.append((method, seed, cp, cfg))
    if len({(c.eval_batch_size,c.num_steps,c.reference_sigma,c.finite_delta) for _,_,_,c in prepared}) != 1:
        raise ValueError('Methods have different condition counts/rollout steps/diffusion; check protocol.')
    print(f'Validated {len(jobs)} checkpoints and {len(raw)} splits.', flush=True)
    if args.check_only:
        return
    results = []
    for method, seed, cp, cfg in prepared:
        model = module.ConditionalDSBM(cfg, sd, ud, torch.device(args.device))
        model.load_state_dict(module.safe_torch_load(cp)['model'])
        model.net_f.eval()
        model.net_b.eval()
        cp_hash = sha(cp)
        for split in args.splits:
            provenance = dict(dataset=args.dataset, method=method, seed=seed, split=split,
                checkpoint=str(cp.resolve()), checkpoint_sha256=cp_hash, data_sha256=hashes[split],
                evaluator_sha256=sha(Path(__file__)), base_sha256=sha(Path(module.__file__)),
                config=dataclasses.asdict(cfg), mc=args.mc, draw_batch=args.draw_batch,
                condition_batch=args.condition_batch, metadata_sha256=sha(meta_path),
                device=args.device, torch_version=str(torch.__version__))
            for kind in ['endpoint','sensitivity','finite_change']:
                prov = dict(provenance,quantity=kind)
                path = args.out/args.dataset/method/f'seed_{seed}'/f'{split}_{kind}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                results.append(evaluate(args,module,cfg,model,raw[split],metadata,prov,path))
    aggregate = []
    for method in methods:
        for split in args.splits:
            for mc in args.mc:
                rows = [r for item in results if item['provenance']['method']==method
                        and item['provenance']['split']==split for r in item['results'] if r['mc']==mc]
                row = dict(method=method, split=split, mc=mc, seeds=args.seeds)
                for metric in ['E_J','jacobian_rmse','estimated_mc_noise_relative','mean_rmse','finite_response_rmse']:
                    vals = [r[metric] for r in rows if metric in r]
                    row[metric] = dict(mean=statistics.mean(vals), sample_sd=statistics.stdev(vals) if len(vals)>1 else None)
                aggregate.append(row)
                print(f"{method:12} {split:14} MC={mc:4}: endpoint={row['mean_rmse']['mean']:.6f} "
                      f"E_J={row['E_J']['mean']:.6f} finite={row['finite_response_rmse']['mean']:.6f}")
    
    tag = hashlib.sha256(json.dumps([methods,args.seeds,args.splits,args.mc]).encode()).hexdigest()[:10]
    output = args.out/args.dataset/f'summary_{tag}.json'
    write_json(output, dict(scope='Endpoint mean RMSE, pooled sensitivity E_J, finite-change RMSE',
                            per_seed=results, aggregate=aggregate))
    print('Saved', output)


if __name__ == '__main__':
    main()
