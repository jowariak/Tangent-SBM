"""SNS #6: deterministic endpoint-only mean operator. Train and evaluate separately.

Imports the existing conditional baseline and (only for evaluation) FINAL2 helper.
No response-training labels or test data are loaded during training.
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
    # Reuse the drift UNet as a terminal regressor with constant time channel.
    return net(x, a, x.new_zeros((x.shape[0], 1)))


def train(args):
    cp = args.baseline_root / 'sigma_1p4' / f'seed_{args.seed}' / 'imf_3.pt'
    reference = base.safe_load(cp)
    if reference.get('imf') != 3 or reference['config']['reference_sigma'] != 1.4:
        raise ValueError('Expected the frozen sigma=1.4 IMF-3 reference checkpoint')
    cfg = base.Config(**reference['config'])
    cfg.seed = args.seed
    # Match all forward+backward optimizer updates across five IMF iterations.
    # Network weights are initialized from scratch; no fork weights are loaded.
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
    print(f'Endpoint-only direct mean operator: seed={args.seed}, updates={updates}', flush=True)
    for step in range(1, updates + 1):
        idx = torch.randint(len(x), (min(cfg.batch_size, len(x)),))
        output = predict(net, x[idx].to(device), a[idx].to(device))
        loss = F.mse_loss(output, y[idx].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        if not torch.isfinite(norm):
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        losses = losses[-100:]
        if step == 1 or step % 100 == 0 or step == updates:
            print(f'step {step}/{updates} endpoint_mse={sum(losses)/len(losses):.8g}', flush=True)
    summary = dict(method='deterministic_direct_mean_operator', seed=args.seed,
                   config=asdict(cfg), state_mean=mean, state_std=std, quick=args.quick,
                   reference_checkpoint=str(cp), pretrained_weights_used=False,
                   supervision='endpoint_train_only', optimizer_updates=updates,
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
    output = args.output or args.run_root / 'FINAL2_direct_mean.json'
    if output.exists():
        raise FileExistsError(f'Result exists: {output}')
    device = base.resolve_device(args.device)
    seeds = [32, 42, 52]
    # Validate all checkpoints before opening FINAL2.
    checkpoints = []
    for seed in seeds:
        path = args.run_root / f'seed_{seed}' / 'final_model.pt'
        ck = base.safe_load(path)
        if ck.get('quick') or ck.get('seed') != seed or ck.get('method') != 'deterministic_direct_mean_operator':
            raise ValueError(f'Wrong or quick checkpoint: {path}')
        checkpoints.append((path, ck))
    metadata = json.loads((args.data_dir / 'metadata.json').read_text())
    mean, std = base.normalization_from_metadata(metadata)
    for path, ck in checkpoints:
        if ck['state_mean'] != mean or ck['state_std'] != std:
            raise ValueError(f'Normalization mismatch: {path}')
    manifest, data = final2.load_final2_data(args.data_dir, mean, std)
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
    payload = dict(method='deterministic_direct_mean_operator', manifest=manifest,
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
    p.add_argument('--run-root', type=Path, default=Path('runs/sns_direct_mean/production'))
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
