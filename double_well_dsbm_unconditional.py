#!/usr/bin/env python3
"""
double_well_dsbm_unconditional.py

Plain DSBM baseline for the stochastic double-well benchmark.

Unlike Conditional DSBM / Tangent-SBM, this model does NOT receive u:

    b_theta(x,t)

Therefore its intervention Jacobian is identically zero:
    dX_T/du = 0.

We still evaluate it against the same simulator truth:
  - conditional mean RMSE
  - right-well probability RMSE
  - Jacobian relative error (exactly 1 when J* != 0)
  - finite-response RMSE (model predicts zero response)

The implementation mirrors double_well_conditional_dsbm.py as closely
as possible for a fair architecture/training-budget comparison.
"""

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

import numpy as np
import torch
import torch.nn as nn

import double_well_conditional_dsbm as cond


def setup_logging(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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


class UnconditionalDriftNet(nn.Module):
    def __init__(self, state_dim, hidden, depth):
        super().__init__()

        d = state_dim + 1
        layers = []

        for _ in range(depth):
            layers += [
                nn.Linear(d, hidden),
                nn.SiLU(),
            ]
            d = hidden

        layers.append(
            nn.Linear(d, state_dim)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]

        return self.net(
            torch.cat([x, t], dim=1)
        )


class UnconditionalDSBM:
    def __init__(self, cfg, state_dim, device):
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

        self.prev_fb = None

    def state_dict(self):
        return {
            "net_f": self.net_f.state_dict(),
            "net_b": self.net_b.state_dict(),
            "prev_fb": self.prev_fb,
        }

    def _noise_bank(self, x, num_steps=None):
        nsteps = (
            self.cfg.num_steps
            if num_steps is None
            else int(num_steps)
        )

        return [
            torch.randn_like(x)
            for _ in range(nsteps)
        ]

    def get_train_tuple(self, z0, z1, fb):
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
            * torch.sqrt(t * (1.0 - t))
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
                (x.shape[0], 1),
                tv,
                device=x.device,
                dtype=x.dtype,
            )

            drift = net(x, t)

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
    def regenerate_coupling(self, data):
        x0 = data["x0"].to(self.device)
        x1 = data["x1"].to(self.device)

        if self.prev_fb is None:
            return x0, x1

        if self.prev_fb == "f":
            return (
                x0,
                self.sample_sde(
                    x0,
                    fb="f",
                ).detach(),
            )

        return (
            self.sample_sde(
                x1,
                fb="b",
            ).detach(),
            x1,
        )

    def train_pass(self, data, fb):
        z0, z1 = self.regenerate_coupling(
            data
        )

        net = self.nets[fb]
        net.train()

        opt = torch.optim.AdamW(
            net.parameters(),
            lr=self.cfg.lr,
            weight_decay=1e-5,
        )

        n = z0.shape[0]
        recent = []

        for step in range(
            1,
            self.cfg.inner_steps + 1,
        ):
            bsz = min(
                self.cfg.batch_size,
                n,
            )

            idx = torch.randint(
                0,
                n,
                (bsz,),
                device=self.device,
            )

            zt, t, target = self.get_train_tuple(
                z0[idx],
                z1[idx],
                fb,
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

            opt.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                self.cfg.grad_clip,
            )

            opt.step()

            recent.append(
                float(
                    loss.detach().cpu()
                )
            )
            recent = recent[-100:]

            report_every = max(
                100,
                self.cfg.inner_steps // 5,
            )

            if (
                step == 1
                or step % report_every == 0
                or step == self.cfg.inner_steps
            ):
                log(
                    f"{fb} step {step:5d}/{self.cfg.inner_steps}",
                    f"bridge={float(loss.detach().cpu()):.6f}",
                )

        self.prev_fb = fb

        return {
            "bridge_loss_last100":
                float(np.mean(recent))
        }


@torch.no_grad()
def endpoint_metrics(
    model,
    data,
    cfg,
    device,
):
    x0 = data["x0"].to(device)
    true_mean = data["true_mean"].to(device)
    true_right = data["true_right_prob"].to(device)

    sum_y = torch.zeros_like(
        true_mean
    )
    sum_right = torch.zeros_like(
        true_right
    )

    for _ in range(cfg.eval_mc):
        y = model.sample_sde(
            x0,
            fb="f",
        )

        sum_y += y
        sum_right += (
            y > 0.0
        ).float()

    pred_mean = (
        sum_y / float(cfg.eval_mc)
    )
    pred_right = (
        sum_right / float(cfg.eval_mc)
    )

    mean_rmse = torch.sqrt(
        ((pred_mean - true_mean) ** 2)
        .mean()
    )

    right_rmse = torch.sqrt(
        ((pred_right - true_right) ** 2)
        .mean()
    )

    return {
        "mean_rmse":
            float(
                mean_rmse.cpu()
            ),
        "right_well_prob_rmse":
            float(
                right_rmse.cpu()
            ),
    }


@torch.no_grad()
def response_metrics(
    data,
    cfg,
    device,
):
    n = min(
        cfg.eval_batch_size,
        data["x0"].shape[0],
    )

    J_true = (
        data["J_star"][:n]
        .to(device)
    )

    # Plain DSBM is independent of u.
    J_pred = torch.zeros_like(
        J_true
    )

    diff = J_pred - J_true

    j_rel = (
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

    true_finite = (
        data[
            "true_finite_response"
        ][:n]
        .to(device)
    )

    finite_rmse = torch.sqrt(
        (true_finite ** 2)
        .mean()
    )

    return {
        "jacobian_rel_error":
            float(j_rel.cpu()),
        "jacobian_rmse":
            float(j_rmse.cpu()),
        "finite_response_rmse":
            float(finite_rmse.cpu()),
    }


def evaluate_split(
    split,
    model,
    data,
    cfg,
    device,
):
    out = endpoint_metrics(
        model,
        data,
        cfg,
        device,
    )

    out.update(
        response_metrics(
            data,
            cfg,
            device,
        )
    )

    log(
        split,
        "| mean RMSE =",
        f"{out['mean_rmse']:.6f}",
        "| right-well prob RMSE =",
        f"{out['right_well_prob_rmse']:.6f}",
        "| J rel err =",
        f"{out['jacobian_rel_error']:.6f}",
        "| finite response RMSE =",
        f"{out['finite_response_rmse']:.6f}",
    )

    return out


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        default="runs/double_well_data",
    )

    p.add_argument(
        "--run-root",
        default="runs/double_well_dsbm",
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
        "--reference-sigma",
        type=float,
        default=0.50,
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
        "--eval-batch-size",
        type=int,
        default=512,
    )

    p.add_argument(
        "--device",
        default=None,
    )

    return p.parse_args()


def main():
    args = parse_args()

    cfg = cond.Config(
        seed=args.seed,
        num_steps=args.num_steps,
        reference_sigma=args.reference_sigma,
        total_imf=args.total_imf,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_mc=args.eval_mc,
        eval_batch_size=args.eval_batch_size,
    )

    device = torch.device(
        args.device
        if args.device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    cond.set_seed(
        args.seed
    )

    (
        datasets,
        metadata,
        state_dim,
        _,
    ) = cond.load_dataset(
        Path(args.data_dir)
    )

    cfg.finite_delta = float(
        metadata.get(
            "finite_delta",
            cfg.finite_delta,
        )
    )

    run_dir = (
        Path(args.run_root)
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    setup_logging(
        run_dir / "dsbm.log"
    )

    log("=" * 80)
    log("DOUBLE-WELL PLAIN DSBM")
    log("=" * 80)
    log("Seed:", args.seed)
    log("Device:", device)

    model = UnconditionalDSBM(
        cfg,
        state_dim,
        device,
    )

    train_data = datasets["train"]

    start = time.time()

    history = []

    for imf in range(
        1,
        cfg.total_imf + 1,
    ):
        log("")
        log(
            f"IMF {imf}/{cfg.total_imf} - BACKWARD"
        )
        b = model.train_pass(
            train_data,
            "b",
        )

        log(
            f"IMF {imf}/{cfg.total_imf} - FORWARD"
        )
        f = model.train_pass(
            train_data,
            "f",
        )

        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(cfg),
                "imf": imf,
                "state_dim": state_dim,
            },
            run_dir / f"imf_{imf}.pt",
        )

        history.append(
            {
                "imf": imf,
                "backward_bridge_loss_last100":
                    b[
                        "bridge_loss_last100"
                    ],
                "forward_bridge_loss_last100":
                    f[
                        "bridge_loss_last100"
                    ],
            }
        )

        with open(
            run_dir / "convergence.json",
            "w",
        ) as fp:
            json.dump(
                history,
                fp,
                indent=2,
            )

    train_seconds = (
        time.time() - start
    )

    log("")
    log("=" * 80)
    log("DOUBLE-WELL PLAIN DSBM FINAL SUMMARY")
    log("=" * 80)

    results = {
        split:
            evaluate_split(
                split,
                model,
                datasets[split],
                cfg,
                device,
            )
        for split in [
            "test_seen",
            "test_id",
            "test_ood_near",
            "test_ood_far",
        ]
    }

    payload = {
        "method": "dsbm",
        "benchmark": "double_well",
        "seed": args.seed,
        "config": asdict(cfg),
        "train_seconds": train_seconds,
        "metrics": results,
    }

    with open(
        run_dir / "metrics.json",
        "w",
    ) as fp:
        json.dump(
            payload,
            fp,
            indent=2,
        )

    rows = []

    for split, m in results.items():
        rows.append(
            {
                "method": "dsbm",
                "seed": args.seed,
                "split": split,
                "mean_rmse":
                    m["mean_rmse"],
                "right_well_prob_rmse":
                    m[
                        "right_well_prob_rmse"
                    ],
                "jacobian_rel_error":
                    m[
                        "jacobian_rel_error"
                    ],
                "jacobian_rmse":
                    m[
                        "jacobian_rmse"
                    ],
                "finite_response_rmse":
                    m[
                        "finite_response_rmse"
                    ],
                "train_seconds":
                    train_seconds,
            }
        )

    with open(
        run_dir / "metrics.csv",
        "w",
        newline="",
    ) as fp:
        w = csv.DictWriter(
            fp,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
