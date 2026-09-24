#!/usr/bin/env python3
"""
gaussian_conditional_dsbm_nonlinear.py

Conditional DSBM / IMF baseline for the nonlinear conditional-Gaussian
Tangent-SBM benchmark produced by gaussian_nonlinear_data.py.

The bridge receives u persistently through b_theta(x,u,t), but receives
NO J* supervision during training.

This script:
  - trains ordinary conditional DSBM for N IMF iterations,
  - saves every IMF checkpoint,
  - saves IMF-3 as the future Tangent-SBM fork,
  - writes per-IMF TRAIN convergence diagnostics,
  - evaluates seen-anchor, in-range interpolation, near-OOD, far-OOD splits,
  - evaluates the learned samplewise Jacobian against exact J*(u),
  - evaluates finite intervention response using the exact analytic oracle.

Run:
    python gaussian_conditional_dsbm_nonlinear.py \
        --data-dir runs/gaussian_nonlinear_data \
        --run-root runs/gaussian_conditional_nonlinear \
        --seed 32 \
        --total-imf 5
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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def tensor_to_list(x: torch.Tensor):
    return (
        x.detach()
        .cpu()
        .numpy()
        .tolist()
    )


# ============================================================
# Logging
# ============================================================

def setup_logging(path: Path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        handlers=[
            logging.FileHandler(
                path,
                mode="a",
            ),
            logging.StreamHandler(
                sys.stdout
            ),
        ],
        force=True,
    )

    logging.captureWarnings(True)


def log(*args):
    logging.info(
        " ".join(
            str(x)
            for x in args
        )
    )


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    seed: int = 32

    num_steps: int = 30
    reference_sigma: float = 0.50
    bridge_eps: float = 1e-3

    hidden: int = 128
    depth: int = 3

    total_imf: int = 5
    fork_imf: int = 3

    inner_steps: int = 1200
    batch_size: int = 512
    lr: float = 1e-4
    grad_clip: float = 5.0

    eval_mc: int = 32
    eval_sens_mc: int = 8
    eval_batch_size: int = 512

    finite_delta: float = 0.25


# ============================================================
# Dataset / oracle
# ============================================================

SPLITS = [
    "train",
    "test_seen",
    "test_id",
    "test_ood_near",
    "test_ood_far",
]


def load_dataset(
    data_dir: Path,
):
    objects = {}

    for split in SPLITS:
        path = (
            data_dir
            / f"{split}.pt"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. "
                "Run gaussian_nonlinear_data.py first."
            )

        obj = safe_torch_load(
            path
        )

        objects[split] = {
            "x0": obj["x0"].float(),
            "u": obj["u"].float(),
            "x1": obj["xT"].float(),
            "true_mean":
                obj["true_conditional_mean"].float(),
            # [N, state_dim, intervention_dim]
            "J_star":
                obj["J_star"].float(),
        }

    with open(
        data_dir / "metadata.json",
        "r",
    ) as f:
        metadata = json.load(f)

    state_dim = (
        objects["train"]["x0"]
        .shape[1]
    )

    intervention_dim = (
        objects["train"]["u"]
        .shape[1]
    )

    for split in SPLITS:
        d = objects[split]
        n = d["x0"].shape[0]

        if d["x1"].shape != d["x0"].shape:
            raise RuntimeError(
                f"{split}: x0/x1 shape mismatch."
            )

        if d["u"].shape[0] != n:
            raise RuntimeError(
                f"{split}: u sample count mismatch."
            )

        expected_j_shape = (
            n,
            state_dim,
            intervention_dim,
        )

        if tuple(
            d["J_star"].shape
        ) != expected_j_shape:
            raise RuntimeError(
                f"{split}: expected J_star shape "
                f"{expected_j_shape}, got "
                f"{tuple(d['J_star'].shape)}"
            )

    sigma_eps = torch.tensor(
        metadata["Sigma_eps"],
        dtype=torch.float32,
    )

    return (
        objects,
        metadata,
        state_dim,
        intervention_dim,
        sigma_eps,
    )


def oracle_components(
    metadata,
    device,
    dtype,
):
    A = torch.tensor(
        metadata["A"],
        device=device,
        dtype=dtype,
    )

    B1 = torch.tensor(
        metadata["B1"],
        device=device,
        dtype=dtype,
    )

    B2 = torch.tensor(
        metadata["B2"],
        device=device,
        dtype=dtype,
    )

    B3 = torch.tensor(
        metadata["B3"],
        device=device,
        dtype=dtype,
    )

    return A, B1, B2, B3


def oracle_mean(
    x0,
    u,
    metadata,
):
    A, B1, B2, B3 = (
        oracle_components(
            metadata,
            x0.device,
            x0.dtype,
        )
    )

    return (
        x0 @ A.T
        + u @ B1.T
        + (u ** 2) @ B2.T
        + torch.sin(
            math.pi * u
        ) @ B3.T
    )


def oracle_jacobian(
    u,
    metadata,
):
    _, B1, B2, B3 = (
        oracle_components(
            metadata,
            u.device,
            u.dtype,
        )
    )

    deriv = (
        B1.T
        + 2.0 * u * B2.T
        + math.pi
        * torch.cos(
            math.pi * u
        )
        * B3.T
    )

    return deriv.unsqueeze(-1)


# ============================================================
# Conditional drift
# ============================================================

class ConditionalDriftNet(nn.Module):

    def __init__(
        self,
        state_dim,
        intervention_dim,
        hidden,
        depth,
    ):
        super().__init__()

        d = (
            state_dim
            + intervention_dim
            + 1
        )

        layers = []

        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(
                        d,
                        hidden,
                    ),
                    nn.SiLU(),
                ]
            )
            d = hidden

        layers.append(
            nn.Linear(
                d,
                state_dim,
            )
        )

        self.net = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x,
        u,
        t,
    ):
        if t.ndim == 1:
            t = t[:, None]

        return self.net(
            torch.cat(
                [
                    x,
                    u,
                    t,
                ],
                dim=1,
            )
        )


# ============================================================
# Conditional DSBM
# ============================================================

class ConditionalDSBM:

    def __init__(
        self,
        cfg,
        state_dim,
        intervention_dim,
        device,
    ):
        self.cfg = cfg
        self.state_dim = state_dim
        self.intervention_dim = (
            intervention_dim
        )
        self.device = device

        self.net_f = ConditionalDriftNet(
            state_dim,
            intervention_dim,
            cfg.hidden,
            cfg.depth,
        ).to(device)

        self.net_b = ConditionalDriftNet(
            state_dim,
            intervention_dim,
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
            "net_f":
                self.net_f.state_dict(),
            "net_b":
                self.net_b.state_dict(),
            "prev_fb":
                self.prev_fb,
        }

    def load_state_dict(
        self,
        state,
    ):
        self.net_f.load_state_dict(
            state["net_f"]
        )

        self.net_b.load_state_dict(
            state["net_b"]
        )

        self.prev_fb = state.get(
            "prev_fb",
            None,
        )

    # --------------------------------------------------------
    # Reciprocal bridge matching sample
    # --------------------------------------------------------

    def get_train_tuple(
        self,
        z0,
        z1,
        u,
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

        noise = torch.randn_like(
            z0
        )

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

        return (
            zt,
            u,
            t,
            target,
        )

    # --------------------------------------------------------
    # Learned SDE rollout
    # --------------------------------------------------------

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
        u,
        fb="f",
        noise_bank=None,
        num_steps=None,
    ):
        nsteps = (
            self.cfg.num_steps
            if num_steps is None
            else int(num_steps)
        )

        dt = (
            1.0
            / nsteps
        )

        x = xstart.clone()

        if noise_bank is None:
            noise_bank = (
                self._noise_bank(
                    x,
                    nsteps,
                )
            )

        if fb == "f":
            times = [
                k / nsteps
                for k in range(
                    nsteps
                )
            ]
        elif fb == "b":
            times = [
                1.0 - k / nsteps
                for k in range(
                    nsteps
                )
            ]
        else:
            raise ValueError(fb)

        net = self.nets[fb]

        for k, tv in enumerate(
            times
        ):
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
                u,
                t,
            )

            x = (
                x
                + dt * drift
                + self.cfg.reference_sigma
                * math.sqrt(dt)
                * noise_bank[k]
            )

            # Baseline rollout/eval does not need path graph.
            x = x.detach()

        return x

    # --------------------------------------------------------
    # IMF coupling regeneration
    # --------------------------------------------------------

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

        u = data["u"].to(
            self.device
        )

        if self.prev_fb is None:
            return (
                x0,
                x1,
                u,
            )

        if self.prev_fb == "f":
            z0 = x0

            z1 = self.sample_sde(
                x0,
                u,
                fb="f",
            )

        else:
            z0 = self.sample_sde(
                x1,
                u,
                fb="b",
            )

            z1 = x1

        return (
            z0.detach(),
            z1.detach(),
            u,
        )

    # --------------------------------------------------------
    # One Markov projection pass
    # --------------------------------------------------------

    def train_pass(
        self,
        data,
        fb,
    ):
        cfg = self.cfg

        z0, z1, u = (
            self.regenerate_coupling(
                data
            )
        )

        net = self.nets[fb]
        net.train()

        optimizer = (
            torch.optim.AdamW(
                net.parameters(),
                lr=cfg.lr,
                weight_decay=1e-5,
            )
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
            bu = u[idx]

            (
                zt,
                bu,
                t,
                target,
            ) = self.get_train_tuple(
                bz0,
                bz1,
                bu,
                fb,
            )

            pred = net(
                zt,
                bu,
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
                or step
                % report_every
                == 0
                or step
                == cfg.inner_steps
            ):
                log(
                    f"{fb} step "
                    f"{step:5d}/"
                    f"{cfg.inner_steps}",
                    f"bridge="
                    f"{float(loss.detach().cpu()):.6f}",
                )

        self.prev_fb = fb

        return {
            "bridge_loss_last100":
                float(
                    np.mean(recent)
                )
        }

    # --------------------------------------------------------
    # Learned tangent evaluation
    # --------------------------------------------------------

    def tangent_rollout(
        self,
        x0,
        u,
        noise_bank=None,
    ):
        """
        Evaluate samplewise dX_T/du of the learned forward SDE.

        IMPORTANT:
        This is diagnostic only in the conditional baseline.
        J* is never used to train this script.
        """
        cfg = self.cfg

        dt = (
            1.0
            / cfg.num_steps
        )

        x = x0.clone()
        u_var = u.clone()

        tangents = [
            torch.zeros_like(x)
            for _ in range(
                self.intervention_dim
            )
        ]

        if noise_bank is None:
            noise_bank = (
                self._noise_bank(
                    x,
                    cfg.num_steps,
                )
            )

        for k in range(
            cfg.num_steps
        ):
            tv = (
                k
                / cfg.num_steps
            )

            t = torch.full(
                (
                    x.shape[0],
                    1,
                ),
                tv,
                device=x.device,
                dtype=x.dtype,
            )

            def drift_fn(
                x_in,
                u_in,
            ):
                return self.net_f(
                    x_in,
                    u_in,
                    t,
                )

            next_tangents = []
            drift_for_state = None

            for j in range(
                self.intervention_dim
            ):
                v = torch.zeros_like(
                    u_var
                )

                v[:, j] = 1.0

                drift, tangent_drift = (
                    torch.autograd.functional.jvp(
                        drift_fn,
                        (
                            x,
                            u_var,
                        ),
                        (
                            tangents[j],
                            v,
                        ),
                        create_graph=False,
                        strict=False,
                    )
                )

                if drift_for_state is None:
                    drift_for_state = drift

                next_tangents.append(
                    tangents[j]
                    + dt
                    * tangent_drift
                )

            x = (
                x
                + dt
                * drift_for_state
                + cfg.reference_sigma
                * math.sqrt(dt)
                * noise_bank[k]
            )

            x = x.detach()

            tangents = [
                r.detach()
                for r
                in next_tangents
            ]

        return torch.stack(
            tangents,
            dim=-1,
        )


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def mc_mean_prediction(
    model,
    x0,
    u,
    mc,
):
    ys = []

    for _ in range(mc):
        ys.append(
            model.sample_sde(
                x0,
                u,
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

    u = data["u"].to(
        device
    )

    true_mean = (
        data["true_mean"]
        .to(device)
    )

    pred_mean = (
        mc_mean_prediction(
            model,
            x0,
            u,
            cfg.eval_mc,
        )
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
            u,
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
        sigma_eps_true
        .to(device)
    )

    cov_rel_error = (
        torch.linalg.matrix_norm(
            cov
            - sigma_eps_true
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


def sensitivity_metrics(
    model,
    data,
    cfg,
    device,
):
    """
    Samplewise response metric:
      compare E_omega[J_theta(u,omega)] against exact J*(u)
      for each evaluated sample, then aggregate over samples.
    """
    n = min(
        cfg.eval_batch_size,
        data["x0"].shape[0],
    )

    x0 = (
        data["x0"][:n]
        .to(device)
    )

    u = (
        data["u"][:n]
        .to(device)
    )

    J_true = (
        data["J_star"][:n]
        .to(device)
    )

    Js = []

    with torch.enable_grad():
        for _ in range(
            cfg.eval_sens_mc
        ):
            Js.append(
                model.tangent_rollout(
                    x0,
                    u,
                ).detach()
            )

    # [N,state_dim,d]
    J_pred = torch.stack(
        Js,
        dim=0,
    ).mean(dim=0)

    diff = (
        J_pred
        - J_true
    )

    rel_error = (
        torch.linalg.vector_norm(
            diff.reshape(-1)
        )
        / torch.linalg.vector_norm(
            J_true.reshape(-1)
        ).clamp_min(1e-8)
    )

    rmse = torch.sqrt(
        (diff ** 2)
        .mean()
    )

    return {
        "jacobian_rel_error":
            float(
                rel_error.detach()
                .cpu()
            ),
        "jacobian_rmse":
            float(
                rmse.detach()
                .cpu()
            ),

        # Useful summaries for logs, while preserving the full
        # samplewise metric above.
        "mean_predicted_jacobian":
            tensor_to_list(
                J_pred.mean(dim=0)
            ),
        "mean_true_jacobian":
            tensor_to_list(
                J_true.mean(dim=0)
            ),
    }


@torch.no_grad()
def finite_intervention_metric(
    model,
    data,
    metadata,
    cfg,
    device,
):
    n = min(
        cfg.eval_batch_size,
        data["x0"].shape[0],
    )

    x0 = (
        data["x0"][:n]
        .to(device)
    )

    u = (
        data["u"][:n]
        .to(device)
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

    predicted_changes = []

    for _ in range(
        cfg.eval_mc
    ):
        noise_bank = (
            model._noise_bank(
                x0,
                cfg.num_steps,
            )
        )

        y0 = model.sample_sde(
            x0,
            u,
            fb="f",
            noise_bank=noise_bank,
        )

        y1 = model.sample_sde(
            x0,
            u
            + delta_u[None, :],
            fb="f",
            noise_bank=noise_bank,
        )

        predicted_changes.append(
            y1 - y0
        )

    pred_change = torch.stack(
        predicted_changes,
        dim=0,
    ).mean(dim=0)

    true_change = (
        oracle_mean(
            x0,
            u
            + delta_u[None, :],
            metadata,
        )
        - oracle_mean(
            x0,
            u,
            metadata,
        )
    )

    rmse = torch.sqrt(
        (
            (pred_change - true_change)
            ** 2
        ).mean()
    )

    return {
        "finite_response_rmse":
            float(
                rmse.detach()
                .cpu()
            ),
        "delta_u":
            tensor_to_list(
                delta_u
            ),
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
        sensitivity_metrics(
            model,
            data,
            cfg,
            device,
        )
    )

    out.update(
        finite_intervention_metric(
            model,
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


# ============================================================
# Convergence diagnostics
# ============================================================

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

    orig_eval_mc = cfg.eval_mc
    orig_sens_mc = cfg.eval_sens_mc
    orig_batch = cfg.eval_batch_size

    cfg.eval_mc = min(
        cfg.eval_mc,
        8,
    )

    cfg.eval_sens_mc = min(
        cfg.eval_sens_mc,
        2,
    )

    cfg.eval_batch_size = min(
        cfg.eval_batch_size,
        n,
    )

    try:
        out = endpoint_metrics(
            model,
            subset,
            sigma_eps_true,
            cfg,
            device,
        )

        out.update(
            sensitivity_metrics(
                model,
                subset,
                cfg,
                device,
            )
        )

        out.update(
            finite_intervention_metric(
                model,
                subset,
                metadata,
                cfg,
                device,
            )
        )

    finally:
        cfg.eval_mc = (
            orig_eval_mc
        )
        cfg.eval_sens_mc = (
            orig_sens_mc
        )
        cfg.eval_batch_size = (
            orig_batch
        )

    return out


def write_convergence(
    run_dir,
    history,
):
    with open(
        run_dir
        / "convergence.json",
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
                "imf":
                    row["imf"],
                "backward_bridge_loss_last100":
                    row["backward_bridge_loss_last100"],
                "forward_bridge_loss_last100":
                    row["forward_bridge_loss_last100"],
                "train_mean_rmse":
                    m["mean_rmse"],
                "train_cov_rel_error":
                    m["cov_rel_error"],
                "train_jacobian_rel_error":
                    m["jacobian_rel_error"],
                "train_jacobian_rmse":
                    m["jacobian_rmse"],
                "train_finite_response_rmse":
                    m["finite_response_rmse"],
            }
        )

    with open(
        run_dir
        / "convergence.csv",
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


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    cfg,
    imf,
    state_dim,
    intervention_dim,
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
            "intervention_dim":
                intervention_dim,
            "rng_state":
                capture_rng_state(),
        },
        path,
    )


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        type=str,
        default=(
            "runs/"
            "gaussian_nonlinear_data"
        ),
    )

    p.add_argument(
        "--run-root",
        type=str,
        default=(
            "runs/"
            "gaussian_conditional_nonlinear"
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=32,
    )

    p.add_argument(
        "--total-imf",
        type=int,
        default=5,
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
        "--eval-sens-mc",
        type=int,
        default=8,
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
        eval_sens_mc=args.eval_sens_mc,
    )

    if args.quick:
        cfg.inner_steps = min(
            cfg.inner_steps,
            50,
        )
        cfg.batch_size = min(
            cfg.batch_size,
            128,
        )
        cfg.num_steps = min(
            cfg.num_steps,
            10,
        )
        cfg.eval_mc = min(
            cfg.eval_mc,
            4,
        )
        cfg.eval_sens_mc = min(
            cfg.eval_sens_mc,
            2,
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
        / "conditional_dsbm.log"
    )

    log("=" * 70)
    log(
        "NONLINEAR CONDITIONAL-GAUSSIAN "
        "DSBM BASELINE"
    )
    log("=" * 70)

    log(
        "Device:",
        device,
    )

    log(
        "Data dir:",
        data_dir.resolve(),
    )

    log(
        "Run dir:",
        run_dir.resolve(),
    )

    log(
        "Config:"
    )

    log(
        json.dumps(
            asdict(cfg),
            indent=2,
        )
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
    ) = load_dataset(
        data_dir
    )

    log("")
    log("DATA SUMMARY")
    log("------------")

    for split in SPLITS:
        log(
            split,
            "N=",
            datasets[split]["x0"]
            .shape[0],
        )

    model = ConditionalDSBM(
        cfg=cfg,
        state_dim=state_dim,
        intervention_dim=
            intervention_dim,
        device=device,
    )

    n_params = sum(
        p.numel()
        for p
        in list(
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

    # ========================================================
    # IMF
    # ========================================================

    for imf in range(
        1,
        cfg.total_imf + 1,
    ):
        log("")
        log("=" * 70)
        log(
            f"IMF {imf}/"
            f"{cfg.total_imf} "
            "- BACKWARD"
        )
        log("=" * 70)

        b_stats = model.train_pass(
            train_data,
            fb="b",
        )

        log("")
        log("=" * 70)
        log(
            f"IMF {imf}/"
            f"{cfg.total_imf} "
            "- FORWARD"
        )
        log("=" * 70)

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
            intervention_dim,
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
                "imf":
                    imf,
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
            "| train J rel err =",
            f"{conv['jacobian_rel_error']:.6f}",
            "| finite response RMSE =",
            f"{conv['finite_response_rmse']:.6f}",
        )

        if imf == cfg.fork_imf:
            log(
                "IMPORTANT: IMF-3 checkpoint "
                "saved as future Tangent-SBM fork."
            )

    train_seconds = (
        time.time()
        - start
    )

    # ========================================================
    # Final evaluation
    # ========================================================

    model.net_f.eval()
    model.net_b.eval()

    results = {}

    for split in [
        "test_seen",
        "test_id",
        "test_ood_near",
        "test_ood_far",
    ]:
        results[split] = (
            evaluate_split(
                split,
                model,
                datasets[split],
                sigma_eps_true,
                metadata,
                cfg,
                device,
            )
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
            "intervention_dim":
                intervention_dim,
            "metadata":
                metadata,
        },
        final_model,
    )

    payload = {
        "method":
            "conditional_dsbm",
        "seed":
            cfg.seed,
        "config":
            asdict(cfg),
        "train_seconds":
            train_seconds,
        "metrics":
            results,
        "fork_checkpoint":
            str(
                run_dir
                / f"imf_{cfg.fork_imf}.pt"
            ),
        "final_checkpoint":
            str(final_model),
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
                    "conditional_dsbm",
                "seed":
                    cfg.seed,
                "split":
                    split,
                "mean_rmse":
                    m["mean_rmse"],
                "cov_rel_error":
                    m["cov_rel_error"],
                "jacobian_rel_error":
                    m[
                        "jacobian_rel_error"
                    ],
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
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)

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
        "Saved final model:",
        final_model,
    )

    log(
        "Shared Tangent-SBM fork:",
        run_dir
        / f"imf_{cfg.fork_imf}.pt"
    )

    log(
        "Convergence:",
        run_dir
        / "convergence.csv"
    )


if __name__ == "__main__":
    main()
