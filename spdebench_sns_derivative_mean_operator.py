"""SNS #7: deterministic mean operator with response supervision. Train and evaluate separately.

Imports the existing conditional baseline and (only for evaluation) FINAL2 helper.
Training loads endpoint pairs and response-only labels; never FINAL2.
Standalone driver: no dependency on the experiment #6 script.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import spdebench_sns_conditional_dsbm_v2 as base


def predict(net, x, a):
    
    return net(x, a, x.new_zeros((x.shape[0], 1)))


def load_responses(path, mean, std):
    data = base.safe_load(path)
    forbidden = {'xT', 'x1', 'xT_samples', 'endpoint', 'endpoint_labels',
                 'finite_response_star', 'finite_response_samples'}
    if forbidden.intersection(data) or data.get('endpoint_labels_included', False):
        raise ValueError(f'Response-only labels expected: {path}')
    x = data['x0_norm'].float() if 'x0_norm' in data else base.normalize_state(data['x0'].float(), mean, std)
    j = data['Jv_star_norm'].float() if 'Jv_star_norm' in data else data['Jv_star'].float() / std
    a = data['a'].float().reshape(-1, 1)
    d = data.get('direction', torch.ones_like(a)).float().reshape(-1, 1)
    if x.shape != j.shape or tuple(x.shape[1:]) != (1, 64, 64) or len(x) != len(a) or len(x) != len(d) or not len(x):
        raise ValueError(f'Invalid response shapes: {path}')
    return dict(x0=x, a=a, direction=d, Jv_star=j)


def response_batch(anchor, colloc, generator, device):
    
    pieces = []
    for src, size in [(anchor, 3), (colloc, 5)]:
        idx = torch.randint(len(src['x0']), (size,), generator=generator)
        pieces.append({k: v[idx] for k, v in src.items()})
    return {k: torch.cat([p[k] for p in pieces]).to(device) for k in pieces[0]}


def response_loss(net, batch):
    _, j = torch.autograd.functional.jvp(
        lambda a: predict(net, batch['x0'], a),
        (batch['a'],), (batch['direction'],), create_graph=True)
    return F.mse_loss(j, batch['Jv_star'])


def response_due(step, updates, inner_steps):
    
    local = step - (updates - 2 * inner_steps)
    return local > 0 and local % 10 == 0


def train(args):
    cp = args.baseline_root / 'sigma_1p4' / f'seed_{args.seed}' / 'imf_3.pt'
    reference = base.safe_load(cp)
    if reference.get('imf') != 3 or reference['config']['reference_sigma'] != 1.4:
        raise ValueError('Expected the frozen sigma=1.4 IMF-3 reference checkpoint')
    cfg = base.Config(**reference['config'])
    cfg.seed = args.seed
    
    
    updates = 2 * 5 * cfg.inner_steps
    directory = args.run_root / f'seed_{args.seed}'
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Existing run: {directory}')
    device = base.resolve_device(args.device)
    metadata = json.loads((args.data_dir / 'metadata.json').read_text())
    mean, std = base.normalization_from_metadata(metadata)
    raw = base.safe_load(args.data_dir / 'endpoint_train.pt')
    x = base.normalize_state(raw['x0'].float(), mean, std)
    y = base.normalize_state(raw['xT'].float(), mean, std)
    a = raw['a'].float().reshape(-1, 1)
    if x.shape != y.shape or tuple(x.shape[1:]) != (1, 64, 64) or len(a) != len(x):
        raise ValueError('Unexpected endpoint training shapes')
    if not len(x):
        raise ValueError('Empty endpoint training dataset')
    anchor = load_responses(args.data_dir / 'anchor_response.pt', mean, std)
    colloc = load_responses(args.data_dir / 'response_collocation.pt', mean, std)
    sens_rng = torch.Generator(device='cpu').manual_seed(910000 + args.seed)
    response_updates = 0
    if args.quick:
        x, y, a = x[:32], y[:32], a[:32]
        updates = min(updates, 10)
    base.set_seed(args.seed)
    net = base.ConditionalUNetDrift(cfg.base_channels).to(device)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-5)
    start = time.time()
    losses = []
    directory.mkdir(parents=True)
    print(f'Derivative-informed mean operator: seed={args.seed}, updates={updates}', flush=True)
    for step in range(1, updates + 1):
        idx = torch.randint(len(x), (min(cfg.batch_size, len(x)),))
        output = predict(net, x[idx].to(device), a[idx].to(device))
        loss = F.mse_loss(output, y[idx].to(device))
        endpoint_loss = loss
        if response_due(step, updates, cfg.inner_steps):
            loss = loss + 1000.0 * response_loss(net, response_batch(anchor, colloc, sens_rng, device))
            response_updates += 1
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        if not torch.isfinite(norm):
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        losses.append(float(endpoint_loss.detach().cpu()))
        losses = losses[-100:]
        if step == 1 or step % 100 == 0 or step == updates:
            print(f'step {step}/{updates} endpoint_mse={sum(losses)/len(losses):.8g}', flush=True)
    summary = dict(method='deterministic_derivative_mean_operator', seed=args.seed,
                   config=asdict(cfg), state_mean=mean, state_std=std, quick=args.quick,
                   reference_checkpoint=str(cp), pretrained_weights_used=False,
                   supervision='endpoint_train_plus_anchor_and_collocation_mean_responses', optimizer_updates=updates,
                   lambda_sens=1000.0, sens_every=10, sens_batch_size=8,
                   anchor_count_per_update=3, collocation_count_per_update=5,
                   response_updates=response_updates, sensitivity_window=2*cfg.inner_steps,
                   response_objective='deterministic_JVP_MSE_in_normalized_state_units',
                   update_budget='2 directions x 5 IMF x reference inner_steps; not compute matched',
                   parameters=sum(p.numel() for p in net.parameters()),
                   train_seconds=time.time()-start, endpoint_mse_last100=sum(losses)/len(losses))
    torch.save({**summary, 'model': net.state_dict()}, directory / 'final_model.pt')
    (directory / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Saved {directory / "final_model.pt"}; no test evaluation performed.', flush=True)


def evaluate_split(net, data, cfg, device, mean, std, diagnostics):
    predictions, responses, finite = [], [], []
    for start in range(0, len(data['x0']), cfg.eval_batch_size):
        sl = slice(start, start + cfg.eval_batch_size)
        x, a, d = [data[k][sl].to(device) for k in ['x0', 'a', 'direction']]
        with torch.no_grad():
            y = predict(net, x, a)
            change = predict(net, x, a + cfg.finite_delta * d) - y
        with torch.enable_grad():
            _, j = torch.autograd.functional.jvp(lambda ai: predict(net, x, ai),
                                                 (a,), (d,), create_graph=False)
        predictions.append(base.unnormalize_state(y, mean, std).cpu())
        responses.append(j.detach().cpu() * std)
        finite.append(change.cpu() * std)
    pred = torch.cat(predictions)
    response = torch.cat(responses)
    change = torch.cat(finite)
    true = data['xT_samples_raw'].mean(1)
    delta = data['finite_response_raw']
    return dict(field_mean_rel_l2=base.mean_relative_l2(pred, true),
                field_mean_rmse_raw=float((pred-true).square().mean().sqrt()),
                finite_response_rel_l2=base.mean_relative_l2(change, delta),
                finite_response_rmse_raw=float((change-delta).square().mean().sqrt()),
                response_diag=diagnostics(response, data['Jv_star_raw']),
                spread_rel_l2=None, energy_distance_per_sqrt_pixel=None,
                sliced_wasserstein_raw=None, enstrophy_w1=None)


def evaluate(args):
    import spdebench_sns_eval_final2 as final2
    output = args.output or args.run_root / 'FINAL2_derivative_mean.json'
    if output.exists():
        raise FileExistsError(f'Result exists: {output}')
    device = base.resolve_device(args.device)
    seeds = [32, 42, 52]
    
    checkpoints = []
    for seed in seeds:
        path = args.run_root / f'seed_{seed}' / 'final_model.pt'
        ck = base.safe_load(path)
        if ck.get('quick') or ck.get('seed') != seed or ck.get('method') != 'deterministic_derivative_mean_operator':
            raise ValueError(f'Wrong or quick checkpoint: {path}')
        if ck.get('lambda_sens') != 1000.0 or ck.get('response_updates') != (2 * ck['config']['inner_steps']) // 10:
            raise ValueError(f'Wrong sensitivity protocol: {path}')
        checkpoints.append((path, ck))
    metadata = json.loads((args.data_dir / 'metadata.json').read_text())
    mean, std = base.normalization_from_metadata(metadata)
    for path, ck in checkpoints:
        if ck['state_mean'] != mean or ck['state_std'] != std:
            raise ValueError(f'Normalization mismatch: {path}')
    manifest, data = final2.load_final2_data(args.data_dir, mean, std)
    if manifest.get('generation_seed') != 2026091702:
        raise ValueError('Expected the existing locked FINAL2 generation seed 2026091702')
    per_seed = {}
    for seed, (path, ck) in zip(seeds, checkpoints):
        cfg = base.Config(**ck['config'])
        net = base.ConditionalUNetDrift(cfg.base_channels).to(device)
        net.load_state_dict(ck['model'])
        net.eval()
        metrics = {split: evaluate_split(net, item, cfg, device, mean, std, final2.response_diag)
                   for split, item in data.items()}
        per_seed[str(seed)] = metrics
        print(f'SEED {seed}\n{json.dumps(metrics, indent=2)}', flush=True)
    def aggregate(items):
        result = {}
        for key, value in items[0].items():
            if isinstance(value, dict):
                result[key] = aggregate([i[key] for i in items])
            elif value is None:
                result[key] = None
            else:
                result[key] = final2.mean_std([i[key] for i in items])
        return result
    summary = {split: aggregate([per_seed[str(seed)][split] for seed in seeds]) for split in data}
    payload = dict(method='deterministic_derivative_mean_operator', manifest=manifest,
                   protocol={'FINAL2': True, 'seeds': seeds, 'model_mc': 1,
                             'deterministic': True, 'distribution_metrics': 'not applicable',
                             'no_model_selection': True},
                   checkpoints=[str(p) for p, _ in checkpoints], per_seed=per_seed, aggregate=summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print('FINAL2 AGGREGATE mean ± sample std\n' + json.dumps(summary, indent=2), flush=True)
    print(f'Saved {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate'])
    p.add_argument('--data-dir', type=Path, default=Path('runs/spdebench_sns_response'))
    p.add_argument('--baseline-root', type=Path, default=Path('runs/spdebench_sns_conditional_final'))
    p.add_argument('--run-root', type=Path, default=Path('runs/sns_derivative_mean/production'))
    p.add_argument('--seed', type=int, choices=[32,42,52], default=32)
    p.add_argument('--device', default='auto')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.mode == 'evaluate' and args.quick:
        p.error('--quick is training-only')
    (train if args.mode == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
