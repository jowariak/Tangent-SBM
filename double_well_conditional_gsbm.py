#!/usr/bin/env python3


import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import double_well_conditional_dsbm as base






def setup_logging(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def log(*args):
    logging.info(" ".join(str(x) for x in args))


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])






def double_well_potential(x, u):
    
    xx = x[..., 0]
    return 0.25 * xx.pow(4) - 0.50 * xx.pow(2) - 0.30 * xx * torch.tanh(u)






def _interp_grid(ctrl, t):
    
    B, K, D = ctrl.shape
    s = t.clamp(0.0, 1.0) * (K - 1)
    idx = torch.floor(s).long().clamp(0, K - 2)
    a = (s - idx.to(s.dtype)).view(1, -1, 1)

    left = ctrl[:, idx, :]
    right = ctrl[:, idx + 1, :]
    val = (1.0 - a) * left + a * right
    dval = (K - 1) * (right - left)
    return val, dval


def _interp_pairwise(ctrl, t):
    
    if t.ndim == 2:
        t = t[:, 0]
    B, K, D = ctrl.shape
    if t.shape != (B,):
        raise ValueError(f"Expected t shape {(B,)}, got {tuple(t.shape)}")

    s = t.clamp(0.0, 1.0) * (K - 1)
    idx = torch.floor(s).long().clamp(0, K - 2)
    a = (s - idx.to(s.dtype)).unsqueeze(-1)
    b = torch.arange(B, device=ctrl.device)

    left = ctrl[b, idx]
    right = ctrl[b, idx + 1]
    val = (1.0 - a) * left + a * right
    dval = (K - 1) * (right - left)
    return val, dval


def _gamma_grid(raw_ctrl, t, sigma, eps=1e-8):
    raw, draw = _interp_grid(raw_ctrl, t)  
    tt = t.view(1, -1, 1).clamp(eps, 1.0 - eps)

    sp = F.softplus(raw)
    dsp = torch.sigmoid(raw) * draw

    root = torch.sqrt(tt * (1.0 - tt))
    base_std = sigma * root
    dbase = sigma * (1.0 - 2.0 * tt) / (2.0 * root.clamp_min(eps))

    gamma = base_std * sp
    dgamma = dbase * sp + base_std * dsp
    return gamma.clamp_min(1e-6), dgamma


def _gamma_pairwise(raw_ctrl, t, sigma, eps=1e-8):
    raw, draw = _interp_pairwise(raw_ctrl, t)  
    if t.ndim == 2:
        tt = t
    else:
        tt = t[:, None]
    tt = tt.clamp(eps, 1.0 - eps)

    sp = F.softplus(raw)
    dsp = torch.sigmoid(raw) * draw

    root = torch.sqrt(tt * (1.0 - tt))
    base_std = sigma * root
    dbase = sigma * (1.0 - 2.0 * tt) / (2.0 * root.clamp_min(eps))

    gamma = base_std * sp
    dgamma = dbase * sp + base_std * dsp
    return gamma.clamp_min(1e-6), dgamma






class TrainableGaussianPath(nn.Module):
    def __init__(self, mean_init, gamma_knots, sigma):
        
        super().__init__()
        B, Km, D = mean_init.shape
        if Km < 3 or gamma_knots < 3:
            raise ValueError("Need at least 3 spline control points.")

        self.B = B
        self.D = D
        self.Km = Km
        self.Kg = int(gamma_knots)
        self.sigma = float(sigma)

        self.register_buffer("x0", mean_init[:, :1].detach().clone())
        self.register_buffer("x1", mean_init[:, -1:].detach().clone())
        self.mean_knots = nn.Parameter(mean_init[:, 1:-1].detach().clone())
        self.gamma_knots = nn.Parameter(torch.zeros(B, self.Kg - 2, 1, device=mean_init.device, dtype=mean_init.dtype))

    @property
    def mean_ctrl(self):
        return torch.cat([self.x0, self.mean_knots, self.x1], dim=1)

    @property
    def gamma_ctrl(self):
        z0 = torch.zeros(self.B, 1, 1, device=self.gamma_knots.device, dtype=self.gamma_knots.dtype)
        z1 = torch.zeros_like(z0)
        return torch.cat([z0, self.gamma_knots, z1], dim=1)

    def grid_stats(self, t):
        mean, dmean = _interp_grid(self.mean_ctrl, t)
        gamma, dgamma = _gamma_grid(self.gamma_ctrl, t, self.sigma)
        return mean, dmean, gamma, dgamma

    def reciprocal_loss(self, t, u, beta, n_mc, direction):
        
        mean, dmean, gamma, dgamma = self.grid_stats(t)
        B, S, D = mean.shape

        noise = torch.randn(B, int(n_mc), S, D, device=mean.device, dtype=mean.dtype)
        x = mean[:, None] + gamma[:, None] * noise

        centered = x - mean[:, None]
        if direction == "f":
            a = (dgamma - self.sigma ** 2 / (2.0 * gamma)) / gamma
            drift = dmean[:, None] + a[:, None] * centered
        elif direction == "b":
            a = (-dgamma - self.sigma ** 2 / (2.0 * gamma)) / gamma
            drift = -dmean[:, None] + a[:, None] * centered
        else:
            raise ValueError(direction)

        
        uu = u[:, None, None, 0]
        state_cost = float(beta) * double_well_potential(x, uu)
        control_cost = 0.5 / (self.sigma ** 2) * drift.pow(2).sum(dim=-1)
        return (state_cost + control_cost).mean()


class FrozenGaussianPath:
    

    def __init__(self, mean_ctrl, gamma_ctrl, sigma):
        self.mean_ctrl = mean_ctrl
        self.gamma_ctrl = gamma_ctrl
        self.sigma = float(sigma)

    def sample_pairwise(self, t):
        mean, _ = _interp_pairwise(self.mean_ctrl, t)
        gamma, _ = _gamma_pairwise(self.gamma_ctrl, t, self.sigma)
        return mean + gamma * torch.randn_like(mean)

    def drift_pairwise(self, t, x, direction):
        mean, dmean = _interp_pairwise(self.mean_ctrl, t)
        gamma, dgamma = _gamma_pairwise(self.gamma_ctrl, t, self.sigma)
        centered = x - mean

        if direction == "f":
            a = (dgamma - self.sigma ** 2 / (2.0 * gamma)) / gamma
            return dmean + a * centered
        elif direction == "b":
            a = (-dgamma - self.sigma ** 2 / (2.0 * gamma)) / gamma
            return -dmean + a * centered
        raise ValueError(direction)






class ConditionalGSBM(base.ConditionalDSBM):
    def __init__(
        self,
        cfg,
        state_dim,
        intervention_dim,
        device,
        *,
        beta,
        spline_mean_knots,
        spline_gamma_knots,
        spline_fit_iters,
        spline_fit_times,
        spline_mc,
        spline_chunk,
        spline_lr_mean,
        spline_lr_gamma,
    ):
        super().__init__(cfg, state_dim, intervention_dim, device)
        if state_dim != 1 or intervention_dim != 1:
            raise NotImplementedError("This first GSBM adapter is for scalar double-well x and u.")

        self.beta = float(beta)
        self.spline_mean_knots = int(spline_mean_knots)
        self.spline_gamma_knots = int(spline_gamma_knots)
        self.spline_fit_iters = int(spline_fit_iters)
        self.spline_fit_times = int(spline_fit_times)
        self.spline_mc = int(spline_mc)
        self.spline_chunk = int(spline_chunk)
        self.spline_lr_mean = float(spline_lr_mean)
        self.spline_lr_gamma = float(spline_lr_gamma)

    @torch.no_grad()
    def _sample_sde_path(self, xstart, u, fb, n_ctrl):
        
        nsteps = int(self.cfg.num_steps)
        dt = 1.0 / nsteps
        ids = torch.linspace(0, nsteps, n_ctrl, device=self.device).round().long().tolist()
        ids_set = set(ids)

        x = xstart.clone()
        saved = {0: x.clone()}
        net = self.nets[fb]

        for k in range(nsteps):
            tv = k / nsteps if fb == "f" else 1.0 - k / nsteps
            t = torch.full((x.shape[0], 1), tv, device=x.device, dtype=x.dtype)
            drift = net(x, u, t)
            x = x + dt * drift + self.cfg.reference_sigma * math.sqrt(dt) * torch.randn_like(x)
            if (k + 1) in ids_set:
                saved[k + 1] = x.clone()

        path = torch.stack([saved[i] for i in ids], dim=1)
        if fb == "b":
            path = torch.flip(path, dims=[1])
        return path

    @torch.no_grad()
    def _coupling_and_path_init(self, data):
        x0 = data["x0"].to(self.device)
        x1 = data["x1"].to(self.device)
        u = data["u"].to(self.device)

        if self.prev_fb is None:
            t = torch.linspace(0.0, 1.0, self.spline_mean_knots, device=self.device, dtype=x0.dtype)
            mean_init = (1.0 - t[None, :, None]) * x0[:, None] + t[None, :, None] * x1[:, None]
            return x0, x1, u, mean_init

        if self.prev_fb == "f":
            mean_init = self._sample_sde_path(x0, u, "f", self.spline_mean_knots)
            return x0, mean_init[:, -1].detach(), u, mean_init.detach()

        mean_init = self._sample_sde_path(x1, u, "b", self.spline_mean_knots)
        return mean_init[:, 0].detach(), x1, u, mean_init.detach()

    def _fit_chunk(self, mean_init, u, direction):
        path = TrainableGaussianPath(
            mean_init=mean_init,
            gamma_knots=self.spline_gamma_knots,
            sigma=self.cfg.reference_sigma,
        ).to(self.device)

        opt = torch.optim.Adam(
            [
                {"params": [path.mean_knots], "lr": self.spline_lr_mean},
                {"params": [path.gamma_knots], "lr": self.spline_lr_gamma},
            ]
        )

        t = torch.linspace(
            self.cfg.bridge_eps,
            1.0 - self.cfg.bridge_eps,
            self.spline_fit_times,
            device=self.device,
            dtype=mean_init.dtype,
        )

        recent = []
        for _ in range(self.spline_fit_iters):
            loss = path.reciprocal_loss(
                t=t,
                u=u,
                beta=self.beta,
                n_mc=self.spline_mc,
                direction=direction,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite GSBM reciprocal-projection loss.")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(path.parameters(), 20.0)
            opt.step()
            recent.append(float(loss.detach().cpu()))

        return (
            path.mean_ctrl.detach(),
            path.gamma_ctrl.detach(),
            float(np.mean(recent[-10:])),
        )

    def _fit_all_paths(self, mean_init, u, direction):
        n = mean_init.shape[0]
        means = []
        gammas = []
        losses = []

        for start in range(0, n, self.spline_chunk):
            end = min(start + self.spline_chunk, n)
            m, g, loss = self._fit_chunk(mean_init[start:end], u[start:end], direction)
            means.append(m.cpu())
            gammas.append(g.cpu())
            losses.append(loss)

        return (
            torch.cat(means, dim=0),
            torch.cat(gammas, dim=0),
            float(np.mean(losses)),
        )

    def train_pass_gsbm(self, data, fb):
        z0, z1, u, mean_init = self._coupling_and_path_init(data)

        log(
            f"[{fb}] GSBM reciprocal projection:",
            f"N={z0.shape[0]}",
            f"beta={self.beta}",
            f"fit_iters={self.spline_fit_iters}",
        )

        mean_ctrl_cpu, gamma_ctrl_cpu, reciprocal_loss = self._fit_all_paths(
            mean_init=mean_init,
            u=u,
            direction=fb,
        )

        net = self.nets[fb]
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        n = z0.shape[0]
        recent = []

        for step in range(1, self.cfg.inner_steps + 1):
            bsz = min(self.cfg.batch_size, n)
            idx_cpu = torch.randint(0, n, (bsz,), device="cpu")
            idx = idx_cpu.to(self.device)

            bu = u[idx]
            mctrl = mean_ctrl_cpu[idx_cpu].to(self.device)
            gctrl = gamma_ctrl_cpu[idx_cpu].to(self.device)

            eps = self.cfg.bridge_eps
            t = torch.rand(bsz, 1, device=self.device) * (1.0 - 2.0 * eps) + eps

            gpath = FrozenGaussianPath(mctrl, gctrl, self.cfg.reference_sigma)
            xt = gpath.sample_pairwise(t)
            with torch.no_grad():
                target = gpath.drift_pairwise(t, xt, fb)

            pred = net(xt, bu, t)
            loss = (pred - target).pow(2).sum(dim=1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
            opt.step()

            recent.append(float(loss.detach().cpu()))
            recent = recent[-100:]

            report_every = max(100, self.cfg.inner_steps // 5)
            if step == 1 or step % report_every == 0 or step == self.cfg.inner_steps:
                log(
                    f"{fb} step {step:5d}/{self.cfg.inner_steps}",
                    f"gsbm_match={float(loss.detach().cpu()):.6f}",
                    f"reciprocal={reciprocal_loss:.6f}",
                )

        self.prev_fb = fb
        return {
            "bridge_loss_last100": float(np.mean(recent)),
            "reciprocal_projection_loss": reciprocal_loss,
        }






def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="runs/double_well_data")
    p.add_argument("--baseline-run-root", default="runs/double_well_conditional")
    p.add_argument("--run-root", default="runs/double_well_gsbm")
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--fork-imf", type=int, default=3)
    p.add_argument("--total-imf", type=int, default=7)
    p.add_argument("--beta", type=float, default=1.0)

    p.add_argument("--spline-mean-knots", type=int, default=8)
    p.add_argument("--spline-gamma-knots", type=int, default=8)
    p.add_argument("--spline-fit-iters", type=int, default=75)
    p.add_argument("--spline-fit-times", type=int, default=16)
    p.add_argument("--spline-mc", type=int, default=4)
    p.add_argument("--spline-chunk", type=int, default=256)
    p.add_argument("--spline-lr-mean", type=float, default=3e-2)
    p.add_argument("--spline-lr-gamma", type=float, default=3e-2)

    p.add_argument("--inner-steps", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-rng-replay", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    baseline_seed_dir = Path(args.baseline_run_root) / f"seed_{args.seed}"
    fork = baseline_seed_dir / f"imf_{args.fork_imf}.pt"
    if not fork.exists():
        raise FileNotFoundError(f"Missing conditional DSBM fork checkpoint: {fork}")

    ckpt = base.safe_torch_load(fork, map_location="cpu")
    cfg = base.Config(**ckpt["config"])
    cfg.fork_imf = args.fork_imf
    cfg.total_imf = args.total_imf
    if args.inner_steps is not None:
        cfg.inner_steps = int(args.inner_steps)

    if args.quick:
        cfg.inner_steps = min(cfg.inner_steps, 80)
        cfg.eval_mc = min(cfg.eval_mc, 4)
        cfg.eval_sens_mc = min(cfg.eval_sens_mc, 2)
        cfg.eval_batch_size = min(cfg.eval_batch_size, 128)
        args.spline_fit_iters = min(args.spline_fit_iters, 10)
        args.spline_fit_times = min(args.spline_fit_times, 8)
        args.spline_mc = min(args.spline_mc, 2)
        args.spline_chunk = min(args.spline_chunk, 128)

    data_dir = Path(args.data_dir)
    datasets, metadata, state_dim, intervention_dim = base.load_dataset(data_dir)
    cfg.finite_delta = float(metadata.get("finite_delta", cfg.finite_delta))

    run_dir = Path(args.run_root) / f"beta_{args.beta:g}" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "conditional_gsbm.log")

    log("=" * 80)
    log("DOUBLE-WELL CONDITIONAL GSBM ADAPTATION")
    log("=" * 80)
    log("Fork:", fork)
    log("Device:", device)
    log("beta:", args.beta)

    model = ConditionalGSBM(
        cfg=cfg,
        state_dim=state_dim,
        intervention_dim=intervention_dim,
        device=device,
        beta=args.beta,
        spline_mean_knots=args.spline_mean_knots,
        spline_gamma_knots=args.spline_gamma_knots,
        spline_fit_iters=args.spline_fit_iters,
        spline_fit_times=args.spline_fit_times,
        spline_mc=args.spline_mc,
        spline_chunk=args.spline_chunk,
        spline_lr_mean=args.spline_lr_mean,
        spline_lr_gamma=args.spline_lr_gamma,
    )
    model.load_state_dict(ckpt["model"])

    if not args.no_rng_replay and "rng_state" in ckpt:
        restore_rng_state(ckpt["rng_state"])
    else:
        base.set_seed(args.seed + 300000)

    train_data = datasets["train"]
    history = []
    start = time.time()

    for imf in range(args.fork_imf + 1, args.total_imf + 1):
        log("")
        log("=" * 80)
        log(f"IMF {imf}/{args.total_imf} - BACKWARD GSBM")
        log("=" * 80)
        b = model.train_pass_gsbm(train_data, "b")

        log("")
        log("=" * 80)
        log(f"IMF {imf}/{args.total_imf} - FORWARD GSBM")
        log("=" * 80)
        f = model.train_pass_gsbm(train_data, "f")

        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(cfg),
                "imf": imf,
                "state_dim": state_dim,
                "intervention_dim": intervention_dim,
                "gsbm_config": {
                    "beta": args.beta,
                    "potential": "x^4/4 - x^2/2 - 0.30*x*tanh(u)",
                    "spline_mean_knots": args.spline_mean_knots,
                    "spline_gamma_knots": args.spline_gamma_knots,
                    "spline_fit_iters": args.spline_fit_iters,
                    "spline_fit_times": args.spline_fit_times,
                    "spline_mc": args.spline_mc,
                    "spline_lr_mean": args.spline_lr_mean,
                    "spline_lr_gamma": args.spline_lr_gamma,
                    "uses_J_star": False,
                },
                "rng_state": base.capture_rng_state(),
            },
            run_dir / f"imf_{imf}.pt",
        )

        conv = base.convergence_metrics(model, train_data, cfg, device)
        history.append(
            {
                "imf": imf,
                "backward_match_last100": b["bridge_loss_last100"],
                "backward_reciprocal": b["reciprocal_projection_loss"],
                "forward_match_last100": f["bridge_loss_last100"],
                "forward_reciprocal": f["reciprocal_projection_loss"],
                "train_empirical_endpoint_rmse": conv["train_empirical_endpoint_rmse"],
            }
        )
        with open(run_dir / "convergence.json", "w") as fp:
            json.dump(history, fp, indent=2)

    train_seconds = time.time() - start

    model.net_f.eval()
    model.net_b.eval()
    results = {
        split: base.evaluate_split(split, model, datasets[split], cfg, device)
        for split in ["test_seen", "test_id", "test_ood_near", "test_ood_far"]
    }

    final_model = run_dir / "final_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "metadata": metadata,
            "gsbm_config": {
                "beta": args.beta,
                "potential": "x^4/4 - x^2/2 - 0.30*x*tanh(u)",
                "uses_J_star": False,
            },
            "fork_checkpoint": str(fork),
        },
        final_model,
    )

    payload = {
        "method": "conditional_gsbm",
        "benchmark": "double_well",
        "seed": args.seed,
        "beta": args.beta,
        "fork_checkpoint": str(fork),
        "train_seconds_after_fork": train_seconds,
        "metrics": results,
        "final_checkpoint": str(final_model),
    }
    with open(run_dir / "metrics.json", "w") as fp:
        json.dump(payload, fp, indent=2)

    rows = []
    for split, m in results.items():
        rows.append(
            {
                "method": "conditional_gsbm",
                "seed": args.seed,
                "beta": args.beta,
                "split": split,
                "mean_rmse": m["mean_rmse"],
                "right_well_prob_rmse": m["right_well_prob_rmse"],
                "jacobian_rel_error": m["jacobian_rel_error"],
                "jacobian_rmse": m["jacobian_rmse"],
                "finite_response_rmse": m["finite_response_rmse"],
                "train_seconds_after_fork": train_seconds,
            }
        )

    with open(run_dir / "metrics.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log("")
    log("=" * 80)
    log("DOUBLE-WELL CONDITIONAL GSBM FINAL SUMMARY")
    log("=" * 80)
    for split, m in results.items():
        log(
            split,
            "| mean RMSE =", f"{m['mean_rmse']:.6f}",
            "| right-well prob RMSE =", f"{m['right_well_prob_rmse']:.6f}",
            "| J rel err =", f"{m['jacobian_rel_error']:.6f}",
            "| finite response RMSE =", f"{m['finite_response_rmse']:.6f}",
        )
    log("Saved:", run_dir)


if __name__ == "__main__":
    main()
