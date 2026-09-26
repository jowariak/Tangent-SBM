#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng_state(state) -> None:
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def make_logger(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    def log(*args):
        logger.info(" ".join(str(x) for x in args))

    return log


@dataclass
class Config:
    seed: int = 32
    num_steps: int = 20
    reference_sigma: float = 0.15
    bridge_eps: float = 1e-3
    base_channels: int = 32
    total_imf: int = 4
    fork_imf: int = 2
    inner_steps: int = 800
    batch_size: int = 8
    rollout_batch_size: int = 8
    lr: float = 2e-4
    grad_clip: float = 5.0
    eval_mc: int = 32
    eval_sens_mc: int = 16
    eval_finite_mc: int = 16
    eval_batch_size: int = 8
    swd_projections: int = 32
    finite_delta: float = 0.25
    convergence_conditions: int = 8
    convergence_mc: int = 2


EVAL_FILES = {
    "test_seen": "eval_seen.pt",
    "test_id": "eval_id.pt",
    "test_ood_near": "eval_ood_near.pt",
    "test_ood_far": "eval_ood_far.pt",
}


def normalization_from_metadata(metadata):
    d = metadata["normalization"]["state_scalar"]
    mean, std = float(d["mean"]), float(d["std"])
    if not math.isfinite(std) or std <= 0:
        raise RuntimeError(f"Invalid state std {std}")
    return mean, std


def normalize_state(x, mean, std):
    return (x - mean) / std


def unnormalize_state(x, mean, std):
    return x * std + mean


def load_dataset(data_dir: Path):
    data_dir = Path(data_dir)
    metadata = json.load(open(data_dir / "metadata.json"))
    mean, std = normalization_from_metadata(metadata)

    tr = safe_load(data_dir / "endpoint_train.pt")
    for k in ["x0", "a", "xT"]:
        if k not in tr:
            raise RuntimeError(f"endpoint_train.pt missing {k}")
    if tuple(tr["x0"].shape[1:]) != (1, 64, 64):
        raise RuntimeError(f"Expected [N,1,64,64], got {tuple(tr['x0'].shape)}")
    if tr["xT"].shape != tr["x0"].shape:
        raise RuntimeError("x0/xT mismatch")
    if tr["a"].ndim != 2 or tr["a"].shape[1] != 1:
        raise RuntimeError(f"Expected a=[N,1], got {tuple(tr['a'].shape)}")

    train = {
        "x0": normalize_state(tr["x0"].float(), mean, std),
        "x1": normalize_state(tr["xT"].float(), mean, std),
        "x0_raw": tr["x0"].float(),
        "x1_raw": tr["xT"].float(),
        "a": tr["a"].float(),
    }

    eval_data = {}
    for split, fn in EVAL_FILES.items():
        o = safe_load(data_dir / fn)
        for k in [
            "x0", "a", "xT_samples", "direction", "Jv_star",
            "finite_response_star", "enstrophy_samples",
        ]:
            if k not in o:
                raise RuntimeError(f"{fn} missing {k}")
        eval_data[split] = {
            "x0": normalize_state(o["x0"].float(), mean, std),
            "x0_raw": o["x0"].float(),
            "a": o["a"].float(),
            "xT_samples_raw": o["xT_samples"].float(),
            "direction": o["direction"].float(),
            "Jv_star_raw": o["Jv_star"].float(),
            "finite_response_raw": o["finite_response_star"].float(),
            "enstrophy_samples": o["enstrophy_samples"].float(),
        }
    return train, eval_data, metadata, mean, std


def empirical_training_spread(train, metadata):
    counts = metadata.get("counts", {})
    reps = int(counts.get("endpoint_replicates_per_condition", counts.get("endpoint_reps", 1)))
    n = train["x1"].shape[0]
    if reps <= 1 or n % reps != 0:
        return {"available": False, "endpoint_reps": reps}
    c = n // reps
    y_raw = train["x1_raw"].reshape(c, reps, 1, 64, 64)
    y_norm = train["x1"].reshape(c, reps, 1, 64, 64)
    vr = y_raw.var(1, unbiased=False)
    vn = y_norm.var(1, unbiased=False)
    return {
        "available": True,
        "endpoint_reps": reps,
        "num_conditions": c,
        "rms_within_condition_spread_raw": float(torch.sqrt(vr.mean())),
        "rms_within_condition_spread_norm": float(torch.sqrt(vn.mean())),
        "mean_pointwise_std_raw": float(torch.sqrt(vr).mean()),
        "mean_pointwise_std_norm": float(torch.sqrt(vn).mean()),
    }


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        g = min(8, co)
        while co % g:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(g, co), nn.SiLU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(g, co), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalUNetDrift(nn.Module):
    
    def __init__(self, base=32):
        super().__init__()
        b = base
        self.e1 = Block(3, b); self.d1 = nn.Conv2d(b, 2*b, 4, 2, 1)
        self.e2 = Block(2*b, 2*b); self.d2 = nn.Conv2d(2*b, 4*b, 4, 2, 1)
        self.e3 = Block(4*b, 4*b); self.d3 = nn.Conv2d(4*b, 8*b, 4, 2, 1)
        self.mid = Block(8*b, 8*b)
        self.u3 = nn.ConvTranspose2d(8*b, 4*b, 4, 2, 1); self.c3 = Block(8*b, 4*b)
        self.u2 = nn.ConvTranspose2d(4*b, 2*b, 4, 2, 1); self.c2 = Block(4*b, 2*b)
        self.u1 = nn.ConvTranspose2d(2*b, b, 4, 2, 1); self.c1 = Block(2*b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x, a, t):
        B, _, H, W = x.shape
        if t.ndim == 1:
            t = t[:, None]
        cond = torch.cat([a.reshape(B, -1)[:, :1], t.reshape(B, -1)[:, :1]], 1)
        cond = cond[:, :, None, None].expand(-1, -1, H, W)
        e1 = self.e1(torch.cat([x, cond], 1))
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        m = self.mid(self.d3(e3))
        y = self.c3(torch.cat([self.u3(m), e3], 1))
        y = self.c2(torch.cat([self.u2(y), e2], 1))
        y = self.c1(torch.cat([self.u1(y), e1], 1))
        return self.out(y)


class ConditionalFieldDSBM:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.net_f = ConditionalUNetDrift(cfg.base_channels).to(device)
        self.net_b = ConditionalUNetDrift(cfg.base_channels).to(device)
        self.nets = {"f": self.net_f, "b": self.net_b}
        self.prev_fb: Optional[str] = None

    def state_dict(self):
        return {"net_f": self.net_f.state_dict(), "net_b": self.net_b.state_dict(), "prev_fb": self.prev_fb}

    def load_state_dict(self, s):
        self.net_f.load_state_dict(s["net_f"])
        self.net_b.load_state_dict(s["net_b"])
        self.prev_fb = s.get("prev_fb")

    def get_train_tuple(self, z0, z1, a, fb):
        B, eps = z0.shape[0], self.cfg.bridge_eps
        t = torch.rand(B, 1, 1, 1, device=z0.device) * (1 - 2*eps) + eps
        noise = torch.randn_like(z0)
        zt = (1-t)*z0 + t*z1 + self.cfg.reference_sigma*torch.sqrt(t*(1-t))*noise
        if fb == "f":
            target = (z1-z0) - self.cfg.reference_sigma*torch.sqrt(t/(1-t))*noise
        elif fb == "b":
            target = -(z1-z0) - self.cfg.reference_sigma*torch.sqrt((1-t)/t)*noise
        else:
            raise ValueError(fb)
        return zt, a, t.reshape(B, 1), target

    def _noise_bank(self, x, nsteps=None):
        n = self.cfg.num_steps if nsteps is None else int(nsteps)
        return [torch.randn_like(x) for _ in range(n)]

    @torch.no_grad()
    def sample_sde(self, xstart, a, fb="f", noise_bank=None, nsteps=None):
        n = self.cfg.num_steps if nsteps is None else int(nsteps)
        dt, x, net = 1.0/n, xstart.clone(), self.nets[fb]
        if noise_bank is None:
            noise_bank = self._noise_bank(x, n)
        for k in range(n):
            tv = k/n if fb == "f" else 1-k/n
            t = torch.full((x.shape[0], 1), tv, device=x.device, dtype=x.dtype)
            x = x + dt*net(x, a, t) + self.cfg.reference_sigma*math.sqrt(dt)*noise_bank[k]
        return x.detach()

    @torch.no_grad()
    def sample_sde_chunked(self, x, a, fb):
        outs = []
        for s in range(0, x.shape[0], self.cfg.rollout_batch_size):
            e = min(x.shape[0], s+self.cfg.rollout_batch_size)
            outs.append(self.sample_sde(x[s:e], a[s:e], fb).detach())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def regenerate_coupling(self, data):
        x0, x1, a = data["x0"].to(self.device), data["x1"].to(self.device), data["a"].to(self.device)
        if self.prev_fb is None:
            return x0, x1, a
        if self.prev_fb == "f":
            z0, z1 = x0, self.sample_sde_chunked(x0, a, "f")
        else:
            z0, z1 = self.sample_sde_chunked(x1, a, "b"), x1
        return z0.detach(), z1.detach(), a

    def train_pass(self, data, fb, log):
        z0, z1, a = self.regenerate_coupling(data)
        net = self.nets[fb]
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        n, recent = z0.shape[0], []
        for step in range(1, self.cfg.inner_steps+1):
            b = min(self.cfg.batch_size, n)
            idx = torch.randint(0, n, (b,), device=self.device)
            zt, ba, t, target = self.get_train_tuple(z0[idx], z1[idx], a[idx], fb)
            pred = net(zt, ba, t)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
            opt.step()
            recent.append(float(loss.detach().cpu())); recent = recent[-100:]
            rep = max(50, self.cfg.inner_steps//5)
            if step == 1 or step % rep == 0 or step == self.cfg.inner_steps:
                log(f"{fb} step {step:5d}/{self.cfg.inner_steps}", f"bridge={float(loss.detach().cpu()):.6f}")
        self.prev_fb = fb
        return {"bridge_loss_last100": float(np.mean(recent))}

    def tangent_direction_rollout(self, x0, a, direction, noise_bank=None):
        dt, x, R = 1.0/self.cfg.num_steps, x0.clone(), torch.zeros_like(x0)
        if noise_bank is None:
            noise_bank = self._noise_bank(x)
        for k in range(self.cfg.num_steps):
            t = torch.full((x.shape[0], 1), k/self.cfg.num_steps, device=x.device, dtype=x.dtype)
            def drift_fn(xi, ai):
                return self.net_f(xi, ai, t)
            drift, td = torch.autograd.functional.jvp(
                drift_fn, (x, a), (R, direction), create_graph=False, strict=False
            )
            x = (x + dt*drift + self.cfg.reference_sigma*math.sqrt(dt)*noise_bank[k]).detach()
            R = (R + dt*td).detach()
        return R


def mean_relative_l2(pred, true):
    n = pred.shape[0]
    num = torch.linalg.vector_norm((pred-true).reshape(n, -1), dim=1)
    den = torch.linalg.vector_norm(true.reshape(n, -1), dim=1).clamp_min(1e-8)
    return float((num/den).mean())


def energy_distance_condition(pred_samples, true_samples):
    x = pred_samples.reshape(pred_samples.shape[0], -1).float()
    y = true_samples.reshape(true_samples.shape[0], -1).float()
    ed = 2*torch.cdist(x, y).mean() - torch.cdist(x, x).mean() - torch.cdist(y, y).mean()
    return float(torch.clamp(ed, min=0) / math.sqrt(x.shape[1]))


def sliced_wasserstein_condition(pred_samples, true_samples, nproj, seed):
    x = pred_samples.reshape(pred_samples.shape[0], -1).float()
    y = true_samples.reshape(true_samples.shape[0], -1).float()
    m = min(x.shape[0], y.shape[0]); x, y = x[:m], y[:m]
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    p = torch.randn(nproj, x.shape[1], generator=g)
    p = p / torch.linalg.vector_norm(p, dim=1, keepdim=True).clamp_min(1e-12)
    xp, yp = torch.sort(x @ p.T, dim=0).values, torch.sort(y @ p.T, dim=0).values
    return float(torch.abs(xp-yp).mean())


def w1_1d(x, y):
    x, y = x.flatten().float(), y.flatten().float()
    m = min(x.numel(), y.numel())
    return float(torch.abs(torch.sort(x[:m]).values - torch.sort(y[:m]).values).mean())


@torch.no_grad()
def sample_endpoint_distribution(model, data, cfg, device):
    C, chunks = data["x0"].shape[0], []
    for s in range(0, C, cfg.eval_batch_size):
        e = min(C, s+cfg.eval_batch_size)
        xb, ab = data["x0"][s:e].to(device), data["a"][s:e].to(device)
        draws = [model.sample_sde(xb, ab, "f").cpu() for _ in range(cfg.eval_mc)]
        chunks.append(torch.stack(draws, 1))
    return torch.cat(chunks, 0)


@torch.no_grad()
def endpoint_distribution_metrics(model, data, cfg, device, mean, std):
    pred_norm = sample_endpoint_distribution(model, data, cfg, device)
    pred = unnormalize_state(pred_norm, mean, std)
    true = data["xT_samples_raw"]
    K = min(pred.shape[1], true.shape[1]); pk, tk = pred[:, :K], true[:, :K]
    pm, tm = pred.mean(1), true.mean(1)
    ps, ts = pred.std(1, unbiased=False), true.std(1, unbiased=False)
    ed, swd, ew1, emean = [], [], [], []
    for i in range(pk.shape[0]):
        ed.append(energy_distance_condition(pk[i], tk[i]))
        swd.append(sliced_wasserstein_condition(pk[i], tk[i], cfg.swd_projections, cfg.seed*100000+i))
        pe = 0.5*torch.mean(pk[i]**2, dim=(-3,-2,-1))
        te = data["enstrophy_samples"][i][:K]
        ew1.append(w1_1d(pe, te)); emean.append(float(torch.abs(pe.mean()-te.mean())))
    return {
        "field_mean_rel_l2": mean_relative_l2(pm, tm),
        "field_mean_rmse_raw": float(torch.sqrt(((pm-tm)**2).mean())),
        "spread_rel_l2": mean_relative_l2(ps, ts),
        "spread_rmse_raw": float(torch.sqrt(((ps-ts)**2).mean())),
        "pred_mean_pointwise_std_raw": float(ps.mean()),
        "true_mean_pointwise_std_raw": float(ts.mean()),
        "energy_distance_per_sqrt_pixel": float(np.mean(ed)),
        "sliced_wasserstein_raw": float(np.mean(swd)),
        "enstrophy_w1": float(np.mean(ew1)),
        "enstrophy_mean_abs_error": float(np.mean(emean)),
        "eval_pred_samples": int(pred.shape[1]),
        "eval_true_samples": int(true.shape[1]),
    }


def response_rel(pred, true):
    return mean_relative_l2(pred, true)


def sensitivity_metrics(model, data, cfg, device, state_std):
    C, chunks = data["x0"].shape[0], []
    for s in range(0, C, cfg.eval_batch_size):
        e = min(C, s+cfg.eval_batch_size)
        x0, a, d = data["x0"][s:e].to(device), data["a"][s:e].to(device), data["direction"][s:e].to(device)
        samples = []
        with torch.enable_grad():
            for _ in range(cfg.eval_sens_mc):
                samples.append(model.tangent_direction_rollout(x0, a, d).detach().cpu())
        chunks.append(torch.stack(samples).mean(0))
    pred = torch.cat(chunks, 0) * state_std
    true = data["Jv_star_raw"]
    return {
        "directional_j_rel_error": response_rel(pred, true),
        "directional_j_rmse_raw": float(torch.sqrt(((pred-true)**2).mean())),
    }


@torch.no_grad()
def finite_response_metrics(model, data, cfg, device, state_std):
    C, chunks = data["x0"].shape[0], []
    for s in range(0, C, cfg.eval_batch_size):
        e = min(C, s+cfg.eval_batch_size)
        x0, a, d = data["x0"][s:e].to(device), data["a"][s:e].to(device), data["direction"][s:e].to(device)
        changes = []
        for _ in range(cfg.eval_finite_mc):
            nb = model._noise_bank(x0)
            y0 = model.sample_sde(x0, a, "f", noise_bank=nb)
            y1 = model.sample_sde(x0, a+cfg.finite_delta*d, "f", noise_bank=nb)
            changes.append((y1-y0).cpu())
        chunks.append(torch.stack(changes).mean(0))
    pred = torch.cat(chunks, 0) * state_std
    true = data["finite_response_raw"]
    return {
        "finite_response_rel_l2": response_rel(pred, true),
        "finite_response_rmse_raw": float(torch.sqrt(((pred-true)**2).mean())),
    }


def evaluate_split(split, model, data, cfg, device, mean, std, log):
    model.net_f.eval(); model.net_b.eval()
    out = endpoint_distribution_metrics(model, data, cfg, device, mean, std)
    out.update(sensitivity_metrics(model, data, cfg, device, std))
    out.update(finite_response_metrics(model, data, cfg, device, std))
    log(split, json.dumps(out, indent=2))
    return out


@torch.no_grad()
def convergence_metric(model, train, cfg, device):
    n = min(cfg.convergence_conditions, train["x0"].shape[0])
    x0, a, x1 = train["x0"][:n].to(device), train["a"][:n].to(device), train["x1"][:n].to(device)
    acc = torch.zeros_like(x1)
    for _ in range(max(1, cfg.convergence_mc)):
        acc += model.sample_sde(x0, a, "f")
    pred = acc / max(1, cfg.convergence_mc)
    return float(torch.sqrt(((pred-x1)**2).mean()).cpu())


def save_checkpoint(path, model, cfg, imf, mean, std):
    torch.save({
        "model": model.state_dict(), "config": asdict(cfg), "imf": int(imf),
        "state_shape": [1,64,64], "intervention_dim": 1,
        "state_mean": float(mean), "state_std": float(std), "rng_state": capture_rng_state(),
    }, path)


def load_checkpoint(path, model, restore_rng=False):
    o = safe_load(path); model.load_state_dict(o["model"])
    if restore_rng: restore_rng_state(o.get("rng_state"))
    return o


def write_history(run_dir, history):
    (run_dir/"convergence.json").write_text(json.dumps(history, indent=2))
    cols = ["imf","backward_bridge_loss_last100","forward_bridge_loss_last100","train_endpoint_rmse_norm","seconds"]
    with (run_dir/"convergence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(history)


def write_metrics(run_dir, results, cfg, spread):
    payload = {"method":"conditional_dsbm","benchmark":"spdebench_sns_response","seed":cfg.seed,
               "config":asdict(cfg),"training_spread_diagnostic":spread,"metrics":results}
    (run_dir/"metrics.json").write_text(json.dumps(payload, indent=2))
    rows = []
    for split,m in results.items():
        r = {"split":split}; r.update(m); rows.append(r)
    if rows:
        with (run_dir/"metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("runs/spdebench_sns_response"))
    p.add_argument("--run-root", type=Path, default=Path("runs/spdebench_sns_conditional"))
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--total-imf", type=int, default=4)
    p.add_argument("--fork-imf", type=int, default=2)
    p.add_argument("--inner-steps", type=int, default=800)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--reference-sigma", type=float, default=0.15)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rollout-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--eval-mc", type=int, default=32)
    p.add_argument("--eval-sens-mc", type=int, default=16)
    p.add_argument("--eval-finite-mc", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--swd-projections", type=int, default=32)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--eval-only", type=Path, default=None)
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument(
        "--skip-final-test-eval",
        action="store_true",
        help=(
            "Train/save IMF checkpoints but do not load/evaluate the Seen/ID/Near/Far "
            "test sets. Use this for hyperparameter/checkpoint tuning with the separate "
            "held-out validation selector."
        ),
    )
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args(); device = resolve_device(args.device)
    cfg = Config(seed=args.seed,num_steps=args.num_steps,reference_sigma=args.reference_sigma,
                 base_channels=args.base_channels,total_imf=args.total_imf,fork_imf=args.fork_imf,
                 inner_steps=args.inner_steps,batch_size=args.batch_size,rollout_batch_size=args.rollout_batch_size,
                 lr=args.lr,grad_clip=args.grad_clip,eval_mc=args.eval_mc,eval_sens_mc=args.eval_sens_mc,
                 eval_finite_mc=args.eval_finite_mc,eval_batch_size=args.eval_batch_size,swd_projections=args.swd_projections)
    if cfg.fork_imf > cfg.total_imf: raise ValueError("fork-imf must be <= total-imf")
    if args.quick:
        cfg.total_imf=1; cfg.fork_imf=1; cfg.inner_steps=min(cfg.inner_steps,10); cfg.num_steps=min(cfg.num_steps,3)
        cfg.base_channels=min(cfg.base_channels,16); cfg.batch_size=min(cfg.batch_size,2); cfg.rollout_batch_size=min(cfg.rollout_batch_size,2)
        cfg.eval_mc=2; cfg.eval_sens_mc=1; cfg.eval_finite_mc=1; cfg.eval_batch_size=2; cfg.swd_projections=8
        cfg.convergence_conditions=2; cfg.convergence_mc=1

    run_dir = args.run_root / f"sigma_{str(cfg.reference_sigma).replace('.', 'p')}" / f"seed_{cfg.seed}"
    log = make_logger(run_dir)
    log("="*80); log("SPDEBench stochastic Navier-Stokes Conditional DSBM"); log("="*80)
    log("device =",device); log("config =",json.dumps(asdict(cfg),indent=2))

    set_seed(cfg.seed)
    train, eval_data, metadata, state_mean, state_std = load_dataset(args.data_dir)
    spread = empirical_training_spread(train, metadata)
    log("state_mean_raw =",state_mean); log("state_std_raw =",state_std)
    log("training stochastic spread diagnostic =",json.dumps(spread,indent=2))
    log("reference_sigma is in normalized learned-SDE units; it is NOT the physical SPDE sigma")
    if args.inspect_only:
        return

    if args.quick:
        train = {k:(v[:32] if torch.is_tensor(v) else v) for k,v in train.items()}
        for split in eval_data:
            eval_data[split] = {k:(v[:2] if torch.is_tensor(v) else v) for k,v in eval_data[split].items()}

    model = ConditionalFieldDSBM(cfg, device)
    log("forward_parameters =",sum(p.numel() for p in model.net_f.parameters()))
    log("backward_parameters =",sum(p.numel() for p in model.net_b.parameters()))

    if args.eval_only is not None:
        ckpt = load_checkpoint(args.eval_only, model, False); log("Loaded",args.eval_only,"IMF",ckpt.get("imf"))
        results = {s:evaluate_split(s,model,eval_data[s],cfg,device,state_mean,state_std,log) for s in EVAL_FILES}
        write_metrics(run_dir,results,cfg,spread); return

    start_imf, history = 1, []
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, model, True); start_imf = int(ckpt["imf"])+1
        log("Resumed from",args.resume,"starting IMF",start_imf)

    for imf in range(start_imf,cfg.total_imf+1):
        t0=time.time(); log(""); log("="*80); log(f"IMF {imf}/{cfg.total_imf} BACKWARD"); log("="*80)
        bs=model.train_pass(train,"b",log)
        log(""); log("="*80); log(f"IMF {imf}/{cfg.total_imf} FORWARD"); log("="*80)
        fs=model.train_pass(train,"f",log)
        conv=convergence_metric(model,train,cfg,device)
        h={"imf":imf,"backward_bridge_loss_last100":bs["bridge_loss_last100"],
           "forward_bridge_loss_last100":fs["bridge_loss_last100"],"train_endpoint_rmse_norm":conv,"seconds":time.time()-t0}
        history.append(h); write_history(run_dir,history)
        cp=run_dir/f"imf_{imf}.pt"; save_checkpoint(cp,model,cfg,imf,state_mean,state_std); log("Saved",cp); log("IMF summary =",json.dumps(h,indent=2))
        if imf==cfg.fork_imf: log("Fork checkpoint reached:",cp,"-- preserve this for Tangent-SBM")

    if args.skip_final_test_eval:
        log("")
        log("="*80)
        log("TRAINING COMPLETE -- FINAL TEST EVALUATION SKIPPED")
        log("="*80)
        log("Use spdebench_sns_select_conditional.py on the held-out validation set.")
        return

    log(""); log("="*80); log("FINAL EVALUATION"); log("="*80)
    results={s:evaluate_split(s,model,eval_data[s],cfg,device,state_mean,state_std,log) for s in EVAL_FILES}
    write_metrics(run_dir,results,cfg,spread)
    log(""); log("="*80); log("FINAL SUMMARY"); log("="*80)
    for s,m in results.items():
        log(s,"| mean_rel",f"{m['field_mean_rel_l2']:.6f}","| spread_rel",f"{m['spread_rel_l2']:.6f}",
            "| energy",f"{m['energy_distance_per_sqrt_pixel']:.6f}","| SWD",f"{m['sliced_wasserstein_raw']:.6f}",
            "| Jv_rel",f"{m['directional_j_rel_error']:.6f}","| finite_rel",f"{m['finite_response_rel_l2']:.6f}")
    log("Saved:",run_dir)


if __name__ == "__main__":
    main()
