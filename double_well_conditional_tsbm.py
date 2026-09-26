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

import double_well_conditional_dsbm as base
import double_well_conditional_gsbm as gsbm






def setup_logging(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, mode="a"), logging.StreamHandler(sys.stdout)],
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


def grad_double_well_potential(x, u):
    
    return x.pow(3) - x - 0.30 * torch.tanh(u)






class ConditionalFrozenGaussianPath(gsbm.FrozenGaussianPath):
    

    def _integral_inv_gamma_sq(self, t, s, n_quad=24):
        
        if t.ndim == 1:
            t = t[:, None]
        if s.ndim == 1:
            s = s[:, None]
        if t.shape != s.shape:
            raise ValueError(f"t/s mismatch: {t.shape} vs {s.shape}")

        lo = torch.minimum(t, s)
        hi = torch.maximum(t, s)
        alphas = torch.linspace(0.0, 1.0, int(n_quad), device=t.device, dtype=t.dtype)

        vals = []
        for a in alphas:
            r = lo + a * (hi - lo)
            gamma, _ = gsbm._gamma_pairwise(self.gamma_ctrl, r, self.sigma)
            vals.append(1.0 / gamma.clamp_min(1e-5).pow(2))
        vals = torch.stack(vals, dim=1)  

        
        h = 1.0 / float(len(alphas) - 1)
        integral_alpha = h * (
            0.5 * vals[:, 0]
            + vals[:, 1:-1].sum(dim=1)
            + 0.5 * vals[:, -1]
        )
        return (hi - lo) * integral_alpha

    def sample_s_given_t(self, t, z_t, s, n_quad=24):
        
        mean_t, _ = gsbm._interp_pairwise(self.mean_ctrl, t)
        mean_s, _ = gsbm._interp_pairwise(self.mean_ctrl, s)
        gamma_t, _ = gsbm._gamma_pairwise(self.gamma_ctrl, t, self.sigma)
        gamma_s, _ = gsbm._gamma_pairwise(self.gamma_ctrl, s, self.sigma)

        integ = self._integral_inv_gamma_sq(t, s, n_quad=n_quad)
        J = -0.5 * (self.sigma ** 2) * integ

        mean_cond = mean_s + gamma_s * ((z_t - mean_t) / gamma_t.clamp_min(1e-5)) * torch.exp(J)
        coeff = (-torch.expm1(2.0 * J)).clamp_min(0.0)
        std_cond = gamma_s * torch.sqrt(coeff)
        return mean_cond + std_cond * torch.randn_like(mean_cond)






class ConditionalTSBM(gsbm.ConditionalGSBM):
    def __init__(self, *args, num_s_per_t=2, conditional_quad=24, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_s_per_t = int(num_s_per_t)
        self.conditional_quad = int(conditional_quad)
        if self.num_s_per_t < 1:
            raise ValueError("num_s_per_t must be >=1")
        if self.conditional_quad < 4:
            raise ValueError("conditional_quad must be >=4")

    def _sample_s(self, t, fb):
        
        B = t.shape[0]
        eps = float(self.cfg.bridge_eps)
        T = 1.0
        outs = []
        for _ in range(self.num_s_per_t):
            r = torch.rand(B, 1, device=t.device, dtype=t.dtype)
            if fb == "f":
                
                low = t + eps
                high = torch.full_like(t, T - eps)
                s = low + (high - low).clamp_min(eps) * r
            elif fb == "b":
                
                low = torch.full_like(t, eps)
                high = t - eps
                s = low + (high - low).clamp_min(eps) * r
            else:
                raise ValueError(fb)
            outs.append(s)
        return outs

    def train_pass_tsbm(self, data, fb):
        z0, z1, u, mean_init = self._coupling_and_path_init(data)

        log(
            f"[{fb}] TSBM reciprocal projection:",
            f"N={z0.shape[0]}",
            f"beta={self.beta}",
            f"fit_iters={self.spline_fit_iters}",
            f"S={self.num_s_per_t}",
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
        recent_target_var = []

        for step in range(1, self.cfg.inner_steps + 1):
            bsz = min(self.cfg.batch_size, n)
            idx_cpu = torch.randint(0, n, (bsz,), device="cpu")
            idx = idx_cpu.to(self.device)

            bx0 = z0[idx]
            bx1 = z1[idx]
            bu = u[idx]
            mctrl = mean_ctrl_cpu[idx_cpu].to(self.device)
            gctrl = gamma_ctrl_cpu[idx_cpu].to(self.device)

            
            eps = float(self.cfg.bridge_eps)
            t = torch.rand(bsz, 1, device=self.device) * (1.0 - 4.0 * eps) + 2.0 * eps

            gpath = ConditionalFrozenGaussianPath(mctrl, gctrl, self.cfg.reference_sigma)
            zt = gpath.sample_pairwise(t)

            targets = []
            s_list = self._sample_s(t, fb)
            for s in s_list:
                zs = gpath.sample_s_given_t(t, zt, s, n_quad=self.conditional_quad)
                gradV = grad_double_well_potential(zs, bu)

                if fb == "f":
                    target = (bx1 - zt) / (1.0 - t).clamp_min(eps)
                    target = target - self.beta * (1.0 - s) * gradV
                else:
                    target = (bx0 - zt) / t.clamp_min(eps)
                    target = target - self.beta * s * gradV

                targets.append(target)

            target_stack = torch.stack(targets, dim=1)  
            pred = net(zt, bu, t).unsqueeze(1).expand_as(target_stack)
            sq = (pred - target_stack).pow(2)
            loss = sq.mean()

            
            tvar = target_stack.var(dim=1, unbiased=False).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
            opt.step()

            recent.append(float(loss.detach().cpu()))
            recent = recent[-100:]
            recent_target_var.append(float(tvar.detach().cpu()))
            recent_target_var = recent_target_var[-100:]

            report_every = max(100, self.cfg.inner_steps // 5)
            if step == 1 or step % report_every == 0 or step == self.cfg.inner_steps:
                log(
                    f"{fb} step {step:5d}/{self.cfg.inner_steps}",
                    f"tsbm_match={float(loss.detach().cpu()):.6f}",
                    f"target_s_var={float(tvar.detach().cpu()):.6f}",
                    f"reciprocal={reciprocal_loss:.6f}",
                )

        self.prev_fb = fb
        return {
            "bridge_loss_last100": float(np.mean(recent)),
            "target_s_var_last100": float(np.mean(recent_target_var)),
            "reciprocal_projection_loss": reciprocal_loss,
        }






def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="runs/double_well_data")
    p.add_argument("--baseline-run-root", default="runs/double_well_conditional")
    p.add_argument("--run-root", default="runs/double_well_tsbm")
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

    p.add_argument("--num-s-per-t", type=int, default=2)
    p.add_argument("--conditional-quad", type=int, default=24)
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
        args.conditional_quad = min(args.conditional_quad, 8)

    data_dir = Path(args.data_dir)
    datasets, metadata, state_dim, intervention_dim = base.load_dataset(data_dir)
    cfg.finite_delta = float(metadata.get("finite_delta", cfg.finite_delta))

    run_dir = Path(args.run_root) / f"beta_{args.beta:g}" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "conditional_tsbm.log")

    log("=" * 80)
    log("DOUBLE-WELL CONDITIONAL TSBM ADAPTATION")
    log("=" * 80)
    log("Fork:", fork)
    log("Device:", device)
    log("beta:", args.beta)
    log("num_s_per_t:", args.num_s_per_t)
    log("conditional_quad:", args.conditional_quad)

    model = ConditionalTSBM(
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
        num_s_per_t=args.num_s_per_t,
        conditional_quad=args.conditional_quad,
    )
    model.load_state_dict(ckpt["model"])

    if not args.no_rng_replay and "rng_state" in ckpt:
        restore_rng_state(ckpt["rng_state"])
    else:
        base.set_seed(args.seed + 400000)

    train_data = datasets["train"]
    history = []
    start = time.time()

    for imf in range(args.fork_imf + 1, args.total_imf + 1):
        log("")
        log("=" * 80)
        log(f"IMF {imf}/{args.total_imf} - BACKWARD TSBM")
        log("=" * 80)
        b = model.train_pass_tsbm(train_data, "b")

        log("")
        log("=" * 80)
        log(f"IMF {imf}/{args.total_imf} - FORWARD TSBM")
        log("=" * 80)
        f = model.train_pass_tsbm(train_data, "f")

        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(cfg),
                "imf": imf,
                "state_dim": state_dim,
                "intervention_dim": intervention_dim,
                "tsbm_config": {
                    "beta": args.beta,
                    "potential": "x^4/4 - x^2/2 - 0.30*x*tanh(u)",
                    "spline_mean_knots": args.spline_mean_knots,
                    "spline_gamma_knots": args.spline_gamma_knots,
                    "spline_fit_iters": args.spline_fit_iters,
                    "spline_fit_times": args.spline_fit_times,
                    "spline_mc": args.spline_mc,
                    "num_s_per_t": args.num_s_per_t,
                    "conditional_quad": args.conditional_quad,
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
                "backward_target_s_var_last100": b["target_s_var_last100"],
                "backward_reciprocal": b["reciprocal_projection_loss"],
                "forward_match_last100": f["bridge_loss_last100"],
                "forward_target_s_var_last100": f["target_s_var_last100"],
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
            "tsbm_config": {
                "beta": args.beta,
                "potential": "x^4/4 - x^2/2 - 0.30*x*tanh(u)",
                "num_s_per_t": args.num_s_per_t,
                "conditional_quad": args.conditional_quad,
                "uses_J_star": False,
            },
            "fork_checkpoint": str(fork),
        },
        final_model,
    )

    payload = {
        "method": "conditional_tsbm",
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
                "method": "conditional_tsbm",
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
    log("DOUBLE-WELL CONDITIONAL TSBM FINAL SUMMARY")
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
