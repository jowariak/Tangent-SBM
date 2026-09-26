"""SNS #8: endpoint-only conditional flow matching, Gaussian source, midpoint ODE.

No IMF training, bridge coupling regeneration, or response supervision.
Imports the existing SNS baseline utilities and FINAL2 evaluation utilities.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import spdebench_sns_conditional_dsbm_v2 as base

METHOD = 'conditional_flow_matching_endpoint_only'
SOLVER_STEPS = 32


class ConditionalFlowNet(base.ConditionalUNetDrift):
    def __init__(self, width):
        super().__init__(width)
        
        self.e1 = base.Block(4, width)

    def forward(self, z, x0, a, t):
        return super().forward(torch.cat([z, x0], dim=1), a, t)


def flow_loss(net, x0, a, target, noise, t):
    tau = t.reshape(-1, 1, 1, 1)
    interpolant = (1 - tau) * noise + tau * target
    return F.mse_loss(net(interpolant, x0, a, t), target - noise)


class FlowSampler:
    
    def __init__(self, net, steps=SOLVER_STEPS):
        if steps < 1:
            raise ValueError('Positive solver steps required')
        self.net, self.steps = net, steps
        
        
        self.net_f = self.net
        self.net_b = self.net

    def _noise_bank(self, x):
        
        return torch.randn_like(x)

    @torch.no_grad()
    def sample_sde(self, x0, a, fb='f', noise_bank=None):
        
        if fb != 'f':
            raise ValueError('Conditional flow has no backward bridge network')
        z = self._noise_bank(x0) if noise_bank is None else noise_bank.clone()
        h = 1.0 / self.steps
        for i in range(self.steps):
            t = a.new_full((len(a), 1), i * h)
            v = self.net(z, x0, a, t)
            zmid = z + (0.5 * h) * v
            z = z + h * self.net(zmid, x0, a, t + 0.5 * h)
        return z

    def tangent_direction_rollout(self, x0, a, direction, noise_bank=None):
        
        z = self._noise_bank(x0) if noise_bank is None else noise_bank.clone()
        r = torch.zeros_like(z)
        h = 1.0 / self.steps
        with torch.enable_grad():
            for i in range(self.steps):
                t = a.new_full((len(a), 1), i * h)
                v, dv = torch.autograd.functional.jvp(
                    lambda zi, ai: self.net(zi, x0, ai, t),
                    (z, a), (r, direction), create_graph=False)
                zmid, rmid = z + 0.5*h*v, r + 0.5*h*dv
                vmid, dvmid = torch.autograd.functional.jvp(
                    lambda zi, ai: self.net(zi, x0, ai, t + 0.5*h),
                    (zmid, a), (rmid, direction), create_graph=False)
                z, r = (z + h*vmid).detach(), (r + h*dvmid).detach()
        return r


def train(args):
    cp = args.baseline_root / 'sigma_1p4' / f'seed_{args.seed}' / 'imf_3.pt'
    reference = base.safe_load(cp)
    if reference.get('imf') != 3 or reference['config']['reference_sigma'] != 1.4:
        raise ValueError('Expected the frozen sigma1.4 IMF3 reference configuration')
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
    if x.shape != y.shape or tuple(x.shape[1:]) != (1,64,64) or len(a) != len(x) or not len(x):
        raise ValueError('Invalid endpoint-training shapes')
    if args.quick:
        x, y, a = x[:32], y[:32], a[:32]
        updates = min(updates, 10)
    base.set_seed(args.seed)
    net = ConditionalFlowNet(cfg.base_channels).to(device).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-5)
    directory.mkdir(parents=True, exist_ok=True)
    start, losses = time.time(), []
    print(f'Conditional flow matching seed={args.seed}, updates={updates}', flush=True)
    for step in range(1, updates+1):
        idx = torch.randint(len(x), (min(cfg.batch_size,len(x)),))
        xb, yb, ab = x[idx].to(device), y[idx].to(device), a[idx].to(device)
        noise, t = torch.randn_like(yb), torch.rand(len(idx),1,device=device)
        loss = flow_loss(net, xb, ab, yb, noise, t)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(),cfg.grad_clip)
        if not torch.isfinite(norm):
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        losses.append(float(loss.detach().cpu())); losses=losses[-100:]
        if step == 1 or step % 100 == 0 or step == updates:
            print(f'step {step}/{updates} flow_mse={sum(losses)/len(losses):.8g}',flush=True)
    summary = dict(method=METHOD,seed=args.seed,config=asdict(cfg),quick=args.quick,
        state_mean=mean,state_std=std,reference_checkpoint=str(cp),pretrained_weights_used=False,
        supervision='endpoint_train_only',response_training_labels_used=False,
        optimizer_updates=updates,update_budget='same total updates as direct operator; not compute matched',
        source_distribution='independent standard Gaussian in normalized endpoint coordinates',
        conditioning='fixed initial field x0 and intervention a',
        flow_path='z_t=(1-t)*z+t*xT; target_velocity=xT-z',
        solver='explicit_midpoint',solver_steps=SOLVER_STEPS,
        parameters=sum(p.numel() for p in net.parameters()),train_seconds=time.time()-start,
        flow_mse_last100=sum(losses)/len(losses))
    torch.save({**summary,'model':net.state_dict()},directory/'final_model.pt')
    (directory/'training_summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Saved {directory / "final_model.pt"}; no evaluation performed.',flush=True)


def aggregate(items):
    import spdebench_sns_eval_final2 as final2
    return {key:aggregate([i[key] for i in items]) if isinstance(value,dict)
            else final2.mean_std([i[key] for i in items])
            for key,value in items[0].items()}


def evaluate(args):
    import spdebench_sns_eval_final2 as final2
    output=args.output or args.run_root/'FINAL2_conditional_flow_mc16.json'
    if output.exists():
        raise FileExistsError(f'Result exists: {output}')
    device=base.resolve_device(args.device)
    seeds=[32,42,52]
    metadata=json.loads((args.data_dir/'metadata.json').read_text())
    mean,std=base.normalization_from_metadata(metadata)
    checkpoints=[]
    for seed in seeds:
        path=args.run_root/f'seed_{seed}'/'final_model.pt'
        ck=base.safe_load(path)
        if ck.get('method') != METHOD or ck.get('quick') or ck.get('seed') != seed:
            raise ValueError(f'Wrong/quick checkpoint: {path}')
        if ck['solver_steps'] != SOLVER_STEPS or ck['solver'] != 'explicit_midpoint':
            raise ValueError(f'Unexpected solver: {path}')
        if ck['state_mean'] != mean or ck['state_std'] != std:
            raise ValueError(f'Normalization mismatch: {path}')
        if ck['optimizer_updates'] != 10*ck['config']['inner_steps']:
            raise ValueError(f'Wrong training budget: {path}')
        checkpoints.append((path,ck))
    manifest,data=final2.load_final2_data(args.data_dir,mean,std)
    if manifest.get('generation_seed') != 2026091702:
        raise ValueError('Expected existing FINAL2 generation seed 2026091702')
    per_seed={}
    for seed,(path,ck) in zip(seeds,checkpoints):
        cfg=base.Config(**ck['config'])
        if (cfg.eval_mc,cfg.eval_sens_mc,cfg.eval_finite_mc,cfg.finite_delta) != (32,16,16,0.25):
            raise ValueError(f'Unexpected frozen evaluation configuration: {path}')
        net=ConditionalFlowNet(cfg.base_channels).to(device)
        net.load_state_dict(ck['model']); net.eval()
        model=FlowSampler(net,ck['solver_steps'])
        seed_base=880000+seed*100000
        standard=final2.evaluate_standard(model,cfg,data,device,mean,std,seed_base)
        diagnostic={}
        for j,split in enumerate(final2.FINAL2_FILES):
            pred=final2.predict_mean_j(model,data[split],cfg,device,std,16,seed_base+50000+j*1000)
            diagnostic[split]=final2.response_diag(pred,data[split]['Jv_star_raw'])
        per_seed[str(seed)]={'standard':standard,'response_diag':diagnostic}
        print(f'SEED {seed}\n'+json.dumps(per_seed[str(seed)],indent=2),flush=True)
    summary=aggregate(list(per_seed.values()))
    payload=dict(method=METHOD,manifest=manifest,
        protocol={'FINAL2':True,'no_model_selection':True,'seeds':seeds,
                  'endpoint_mc':32,'response_diag_mc':16,'finite_mc':16,
                  'eval_seed_base':880000,'solver':'explicit_midpoint','solver_steps':SOLVER_STEPS,
                  'common_latent_for_finite_differences':True,'response_training_labels_used':False},
        checkpoints=[str(p) for p,_ in checkpoints],per_seed=per_seed,aggregate=summary)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2))
    print('FINAL2 AGGREGATE mean ± sample std\n'+json.dumps(summary,indent=2),flush=True)
    print(f'Saved {output}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('--data-dir',type=Path,default=Path('runs/spdebench_sns_response'))
    p.add_argument('--baseline-root',type=Path,default=Path('runs/spdebench_sns_conditional_final'))
    p.add_argument('--run-root',type=Path,default=Path('runs/sns_conditional_flow/production'))
    p.add_argument('--seed',type=int,choices=[32,42,52],default=32)
    p.add_argument('--device',default='auto')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.mode=='evaluate' and args.quick: p.error('--quick is training-only')
    (train if args.mode=='train' else evaluate)(args)


if __name__=='__main__':
    main()
