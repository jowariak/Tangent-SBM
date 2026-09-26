#!/usr/bin/env python3


import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import gaussian_conditional_dsbm_nonlinear as cond
except ImportError as exc:
    raise ImportError(
        "Place gaussian_dsbm_unconditional.py in the same directory as "
        "gaussian_conditional_dsbm_nonlinear.py"
    ) from exc






def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_list(x):
    return x.detach().cpu().numpy().tolist()


def setup_logging(path: Path):
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






@dataclass
class Config:
    seed: int = 32

    num_steps: int = 30
    reference_sigma: float = 0.50
    bridge_eps: float = 1e-3

    hidden: int = 128
    depth: int = 3

    total_imf: int = 7
    inner_steps: int = 1200
    batch_size: int = 512
    lr: float = 1e-4
    grad_clip: float = 5.0

    eval_mc: int = 32
    eval_batch_size: int = 512

    finite_delta: float = 0.25






class UnconditionalDriftNet(nn.Module):
    

    def __init__(
        self,
        state_dim,
        hidden,
        depth,
    ):
        super().__init__()

        d = state_dim + 1
        layers = []

        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(d, hidden),
                    nn.SiLU(),
                ]
            )
            d = hidden

        layers.append(
            nn.Linear(d, state_dim)
        )

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]

        return self.net(
            torch.cat(
                [x, t],
                dim=1,
            )
        )






class UnconditionalDSBM:

    def __init__(
        self,
        cfg,
        state_dim,
        device,
    ):
        self.cfg = cfg
        self.state_dim = state_dim
        self.device = device

        self.net_f = UnconditionalDriftNet(
            state_dim,
            cfg.hidden,
            cfg.depth,
        ).to(device)

        self.net_b = UnconditionalDriftNet(
            state_dim,
            cfg.hidden,
            cfg.depth,
        ).to(device)

        self.nets = {
            "f": self.net_f,
            "b": self.net_b,
        }

        self.prev_fb: Optional[str] = None

    def state_dict(self):
        return {
            "net_f": self.net_f.state_dict(),
            "net_b": self.net_b.state_dict(),
            "prev_fb": self.prev_fb,
        }

    
    
    

    def get_train_tuple(
        self,
        z0,
        z1,
        fb,
    ):
        bsz = z0.shape[0]
        eps = self.cfg.bridge_eps

        t = (
            torch.rand(
                bsz,
                1,
                device=z0.device,
            )
            * (1.0 - 2.0 * eps)
            + eps
        )

        noise = torch.randn_like(z0)

        zt = (
            (1.0 - t) * z0
            + t * z1
            + self.cfg.reference_sigma
            * torch.sqrt(
                t * (1.0 - t)
            )
            * noise
        )

        if fb == "f":
            target = (
                (z1 - z0)
                - self.cfg.reference_sigma
                * torch.sqrt(
                    t / (1.0 - t)
                )
                * noise
            )

        elif fb == "b":
            target = (
                -(z1 - z0)
                - self.cfg.reference_sigma
                * torch.sqrt(
                    (1.0 - t) / t
                )
                * noise
            )
        else:
            raise ValueError(fb)

        return zt, t, target

    
    
    

    def _noise_bank(
        self,
        x,
        num_steps=None,
    ):
        nsteps = (
            self.cfg.num_steps
            if num_steps is None
            else int(num_steps)
        )

        return [
            torch.randn_like(x)
            for _ in range(nsteps)
        ]

    def sample_sde(
        self,
        xstart,
        fb="f",
        noise_bank=None,
        num_steps=None,
    ):
        nsteps = (
            self.cfg.num_steps
            if num_steps is None
            else int(num_steps)
        )

        dt = 1.0 / nsteps
        x = xstart.clone()

        if noise_bank is None:
            noise_bank = self._noise_bank(
                x,
                nsteps,
            )

        if fb == "f":
            times = [
                k / nsteps
                for k in range(nsteps)
            ]
        elif fb == "b":
            times = [
                1.0 - k / nsteps
                for k in range(nsteps)
            ]
        else:
            raise ValueError(fb)

        net = self.nets[fb]

        for k, tv in enumerate(times):
            t = torch.full(
                (
                    x.shape[0],
                    1,
                ),
                tv,
                device=x.device,
                dtype=x.dtype,
            )

            drift = net(
                x,
                t,
            )

            x = (
                x
                + dt * drift
                + self.cfg.reference_sigma
                * math.sqrt(dt)
                * noise_bank[k]
            )

            x = x.detach()

        return x

    
    
    

    @torch.no_grad()
    def regenerate_coupling(
        self,
        data,
    ):
        x0 = data["x0"].to(
            self.device
        )

        x1 = data["x1"].to(
            self.device
        )

        if self.prev_fb is None:
            return x0, x1

        if self.prev_fb == "f":
            z0 = x0
            z1 = self.sample_sde(
                x0,
                fb="f",
            )
        else:
            z0 = self.sample_sde(
                x1,
                fb="b",
            )
            z1 = x1

        return (
            z0.detach(),
            z1.detach(),
        )

    
    
    

    def train_pass(
        self,
        data,
        fb,
    ):
        cfg = self.cfg

        z0, z1 = self.regenerate_coupling(
            data
        )

        net = self.nets[fb]
        net.train()

        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=cfg.lr,
            weight_decay=1e-5,
        )

        n = z0.shape[0]
        recent = []

        for step in range(
            1,
            cfg.inner_steps + 1,
        ):
            bsz = min(
                cfg.batch_size,
                n,
            )

            idx = torch.randint(
                0,
                n,
                (bsz,),
                device=self.device,
            )

            bz0 = z0[idx]
            bz1 = z1[idx]

            zt, t, target = (
                self.get_train_tuple(
                    bz0,
                    bz1,
                    fb,
                )
            )

            pred = net(
                zt,
                t,
            )

            loss = (
                ((pred - target) ** 2)
                .sum(dim=1)
                .mean()
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                cfg.grad_clip,
            )

            optimizer.step()

            recent.append(
                float(
                    loss.detach()
                    .cpu()
                )
            )

            if len(recent) > 100:
                recent.pop(0)

            report_every = max(
                100,
                cfg.inner_steps // 5,
            )

            if (
                step == 1
                or step % report_every == 0
                or step == cfg.inner_steps
            ):
                log(
                    f"{fb} step {step:5d}/{cfg.inner_steps}",
                    f"bridge={float(loss.detach().cpu()):.6f}",
                )

        self.prev_fb = fb

        return {
            "bridge_loss_last100":
                float(
                    np.mean(recent)
                )
        }






@torch.no_grad()
def mc_mean_prediction(
    model,
    x0,
    mc,
):
    ys = []

    for _ in range(mc):
        ys.append(
            model.sample_sde(
                x0,
                fb="f",
            )
        )

    return torch.stack(
        ys,
        dim=0,
    ).mean(dim=0)


@torch.no_grad()
def endpoint_metrics(
    model,
    data,
    sigma_eps_true,
    cfg,
    device,
):
    x0 = data["x0"].to(
        device
    )

    true_mean = data["true_mean"].to(
        device
    )

    pred_mean = mc_mean_prediction(
        model,
        x0,
        cfg.eval_mc,
    )

    mean_rmse = torch.sqrt(
        (
            (pred_mean - true_mean)
            ** 2
        ).mean()
    )

    residuals = []

    for _ in range(
        cfg.eval_mc
    ):
        y = model.sample_sde(
            x0,
            fb="f",
        )

        residuals.append(
            y - pred_mean
        )

    residuals = torch.cat(
        residuals,
        dim=0,
    )

    residuals = (
        residuals
        - residuals.mean(
            dim=0,
            keepdim=True,
        )
    )

    cov = (
        residuals.T
        @ residuals
        / max(
            1,
            residuals.shape[0] - 1,
        )
    )

    sigma_eps_true = (
        sigma_eps_true.to(
            device
        )
    )

    cov_rel_error = (
        torch.linalg.matrix_norm(
            cov - sigma_eps_true
        )
        / torch.linalg.matrix_norm(
            sigma_eps_true
        ).clamp_min(1e-8)
    )

    return {
        "mean_rmse":
            float(
                mean_rmse.detach()
                .cpu()
            ),
        "cov_rel_error":
            float(
                cov_rel_error.detach()
                .cpu()
            ),
        "estimated_covariance":
            tensor_to_list(cov),
    }


@torch.no_grad()
def zero_response_metrics(
    data,
    metadata,
    cfg,
    device,
):
    
    n = min(
        cfg.eval_batch_size,
        data["x0"].shape[0],
    )

    x0 = data["x0"][:n].to(
        device
    )

    u = data["u"][:n].to(
        device
    )

    J_true = data["J_star"][:n].to(
        device
    )

    J_pred = torch.zeros_like(
        J_true
    )

    diff = (
        J_pred
        - J_true
    )

    j_rel_error = (
        torch.linalg.vector_norm(
            diff.reshape(-1)
        )
        / torch.linalg.vector_norm(
            J_true.reshape(-1)
        ).clamp_min(1e-8)
    )

    j_rmse = torch.sqrt(
        (diff ** 2)
        .mean()
    )

    d = u.shape[1]

    v = torch.ones(
        d,
        device=device,
        dtype=u.dtype,
    )

    v = (
        v
        / torch.linalg.vector_norm(
            v
        ).clamp_min(1e-8)
    )

    delta_u = (
        cfg.finite_delta
        * v
    )

    
    pred_change = torch.zeros_like(
        x0
    )

    true_change = (
        cond.oracle_mean(
            x0,
            u + delta_u[None, :],
            metadata,
        )
        - cond.oracle_mean(
            x0,
            u,
            metadata,
        )
    )

    finite_rmse = torch.sqrt(
        (
            (pred_change - true_change)
            ** 2
        ).mean()
    )

    return {
        "jacobian_rel_error":
            float(
                j_rel_error.detach()
                .cpu()
            ),
        "jacobian_rmse":
            float(
                j_rmse.detach()
                .cpu()
            ),
        "finite_response_rmse":
            float(
                finite_rmse.detach()
                .cpu()
            ),
        "mean_predicted_jacobian":
            tensor_to_list(
                J_pred.mean(dim=0)
            ),
        "mean_true_jacobian":
            tensor_to_list(
                J_true.mean(dim=0)
            ),
        "response_note":
            "Plain DSBM has no u input; J_theta and finite predicted response are identically zero.",
    }


def evaluate_split(
    name,
    model,
    data,
    sigma_eps_true,
    metadata,
    cfg,
    device,
):
    log("")
    log("=" * 60)
    log("EVALUATING", name)
    log("=" * 60)

    out = endpoint_metrics(
        model,
        data,
        sigma_eps_true,
        cfg,
        device,
    )

    out.update(
        zero_response_metrics(
            data,
            metadata,
            cfg,
            device,
        )
    )

    log(
        json.dumps(
            out,
            indent=2,
        )
    )

    return out






def save_checkpoint(
    path,
    model,
    cfg,
    imf,
    state_dim,
):
    torch.save(
        {
            "model":
                model.state_dict(),
            "config":
                asdict(cfg),
            "imf":
                imf,
            "state_dim":
                state_dim,
        },
        path,
    )


def write_convergence(
    run_dir,
    history,
):
    with open(
        run_dir / "convergence.json",
        "w",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    rows = []

    for row in history:
        m = row["metrics"]

        rows.append(
            {
                "imf": row["imf"],
                "backward_bridge_loss_last100":
                    row[
                        "backward_bridge_loss_last100"
                    ],
                "forward_bridge_loss_last100":
                    row[
                        "forward_bridge_loss_last100"
                    ],
                "train_mean_rmse":
                    m["mean_rmse"],
                "train_cov_rel_error":
                    m["cov_rel_error"],
                "train_jacobian_rel_error":
                    m["jacobian_rel_error"],
                "train_finite_response_rmse":
                    m["finite_response_rmse"],
            }
        )

    with open(
        run_dir / "convergence.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def convergence_metrics(
    model,
    train_data,
    sigma_eps_true,
    metadata,
    cfg,
    device,
):
    n = min(
        1024,
        train_data["x0"].shape[0],
    )

    subset = {
        "x0":
            train_data["x0"][:n],
        "u":
            train_data["u"][:n],
        "x1":
            train_data["x1"][:n],
        "true_mean":
            train_data["true_mean"][:n],
        "J_star":
            train_data["J_star"][:n],
    }

    old_mc = cfg.eval_mc
    old_batch = cfg.eval_batch_size

    cfg.eval_mc = min(
        cfg.eval_mc,
        8,
    )
    cfg.eval_batch_size = min(
        cfg.eval_batch_size,
        n,
    )

    try:
        return evaluate_split(
            "train_convergence",
            model,
            subset,
            sigma_eps_true,
            metadata,
            cfg,
            device,
        )
    finally:
        cfg.eval_mc = old_mc
        cfg.eval_batch_size = old_batch






def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        type=str,
        default="runs/gaussian_nonlinear_data",
    )

    p.add_argument(
        "--run-root",
        type=str,
        default="runs/gaussian_dsbm_only_imf7",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=32,
    )

    p.add_argument(
        "--total-imf",
        type=int,
        default=7,
    )

    p.add_argument(
        "--inner-steps",
        type=int,
        default=1200,
    )

    p.add_argument(
        "--num-steps",
        type=int,
        default=30,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--eval-mc",
        type=int,
        default=32,
    )

    p.add_argument(
        "--device",
        type=str,
        default=None,
    )

    p.add_argument(
        "--quick",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        seed=args.seed,
        total_imf=args.total_imf,
        inner_steps=args.inner_steps,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_mc=args.eval_mc,
    )

    if args.quick:
        cfg.inner_steps = min(
            cfg.inner_steps,
            50,
        )
        cfg.num_steps = min(
            cfg.num_steps,
            10,
        )
        cfg.batch_size = min(
            cfg.batch_size,
            128,
        )
        cfg.eval_mc = min(
            cfg.eval_mc,
            4,
        )
        cfg.eval_batch_size = min(
            cfg.eval_batch_size,
            128,
        )

    device = (
        args.device
        if args.device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    data_dir = Path(
        args.data_dir
    )

    run_dir = (
        Path(args.run_root)
        / f"seed_{cfg.seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    setup_logging(
        run_dir
        / "dsbm_only.log"
    )

    set_seed(
        cfg.seed
    )

    (
        datasets,
        metadata,
        state_dim,
        intervention_dim,
        sigma_eps_true,
    ) = cond.load_dataset(
        data_dir
    )

    log("=" * 76)
    log("PLAIN DSBM BASELINE — NO INTERVENTION CONDITION")
    log("=" * 76)
    log("Device:", device)
    log("Data dir:", data_dir.resolve())
    log("Run dir:", run_dir.resolve())
    log("state_dim:", state_dim)
    log(
        "intervention_dim in dataset:",
        intervention_dim,
        "(NOT provided to DSBM)",
    )
    log("Config:")
    log(
        json.dumps(
            asdict(cfg),
            indent=2,
        )
    )

    model = UnconditionalDSBM(
        cfg=cfg,
        state_dim=state_dim,
        device=device,
    )

    n_params = sum(
        p.numel()
        for p in list(
            model.net_f.parameters()
        )
        + list(
            model.net_b.parameters()
        )
        if p.requires_grad
    )

    log(
        "Trainable bridge parameters:",
        n_params,
    )

    train_data = (
        datasets["train"]
    )

    history = []
    start = time.time()

    for imf in range(
        1,
        cfg.total_imf + 1,
    ):
        log("")
        log("=" * 76)
        log(
            f"IMF {imf}/{cfg.total_imf} - BACKWARD"
        )
        log("=" * 76)

        b_stats = model.train_pass(
            train_data,
            fb="b",
        )

        log("")
        log("=" * 76)
        log(
            f"IMF {imf}/{cfg.total_imf} - FORWARD"
        )
        log("=" * 76)

        f_stats = model.train_pass(
            train_data,
            fb="f",
        )

        ckpt = (
            run_dir
            / f"imf_{imf}.pt"
        )

        save_checkpoint(
            ckpt,
            model,
            cfg,
            imf,
            state_dim,
        )

        log(
            "Saved checkpoint:",
            ckpt,
        )

        model.net_f.eval()
        model.net_b.eval()

        conv = convergence_metrics(
            model,
            train_data,
            sigma_eps_true,
            metadata,
            cfg,
            device,
        )

        history.append(
            {
                "imf": imf,
                "backward_bridge_loss_last100":
                    b_stats[
                        "bridge_loss_last100"
                    ],
                "forward_bridge_loss_last100":
                    f_stats[
                        "bridge_loss_last100"
                    ],
                "metrics":
                    conv,
            }
        )

        write_convergence(
            run_dir,
            history,
        )

        log(
            f"IMF {imf} convergence",
            "| train mean RMSE =",
            f"{conv['mean_rmse']:.6f}",
            "| J rel err =",
            f"{conv['jacobian_rel_error']:.6f}",
        )

    train_seconds = (
        time.time()
        - start
    )

    model.net_f.eval()
    model.net_b.eval()

    results = {}

    for split in [
        "test_seen",
        "test_id",
        "test_ood_near",
        "test_ood_far",
    ]:
        results[split] = evaluate_split(
            split,
            model,
            datasets[split],
            sigma_eps_true,
            metadata,
            cfg,
            device,
        )

    final_model = (
        run_dir
        / "final_model.pt"
    )

    torch.save(
        {
            "model":
                model.state_dict(),
            "config":
                asdict(cfg),
            "state_dim":
                state_dim,
            "intervention_conditioned":
                False,
            "metadata":
                metadata,
        },
        final_model,
    )

    payload = {
        "method":
            "dsbm_only",
        "seed":
            cfg.seed,
        "config":
            asdict(cfg),
        "intervention_conditioned":
            False,
        "train_seconds":
            train_seconds,
        "metrics":
            results,
        "final_checkpoint":
            str(
                final_model
            ),
    }

    with open(
        run_dir
        / "metrics.json",
        "w",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    rows = []

    for split, m in results.items():
        rows.append(
            {
                "method":
                    "dsbm_only",
                "seed":
                    cfg.seed,
                "split":
                    split,
                "mean_rmse":
                    m["mean_rmse"],
                "cov_rel_error":
                    m["cov_rel_error"],
                "jacobian_rel_error":
                    m["jacobian_rel_error"],
                "jacobian_rmse":
                    m["jacobian_rmse"],
                "finite_response_rmse":
                    m[
                        "finite_response_rmse"
                    ],
                "train_seconds":
                    train_seconds,
            }
        )

    with open(
        run_dir
        / "metrics.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    log("")
    log("=" * 76)
    log("PLAIN DSBM FINAL SUMMARY")
    log("=" * 76)

    for split, m in results.items():
        log(
            split,
            "| mean RMSE =",
            f"{m['mean_rmse']:.6f}",
            "| cov rel err =",
            f"{m['cov_rel_error']:.6f}",
            "| J rel err =",
            f"{m['jacobian_rel_error']:.6f}",
            "| finite response RMSE =",
            f"{m['finite_response_rmse']:.6f}",
        )

    log("")
    log(
        "NOTE: J rel error is evaluated with J_theta=0 because "
        "plain DSBM has no u input."
    )
    log(
        "Saved final model:",
        final_model,
    )
    log(
        "Metrics:",
        run_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
