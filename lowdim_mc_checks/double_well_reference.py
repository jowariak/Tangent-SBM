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






def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(
    path,
    map_location="cpu",
):
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
        "python":
            random.getstate(),
        "numpy":
            np.random.get_state(),
        "torch":
            torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def tensor_to_list(x):
    return (
        x.detach()
        .cpu()
        .numpy()
        .tolist()
    )






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






@dataclass
class Config:
    seed: int = 32

    
    num_steps: int = 30
    reference_sigma: float = 0.50
    bridge_eps: float = 1e-3

    
    hidden: int = 128
    depth: int = 3

    
    total_imf: int = 7
    fork_imf: int = 3

    
    inner_steps: int = 1200
    batch_size: int = 512
    lr: float = 1e-4
    grad_clip: float = 5.0

    
    eval_mc: int = 32
    eval_sens_mc: int = 16
    eval_batch_size: int = 512

    finite_delta: float = 0.25






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
                "Run double_well_data.py first."
            )

        obj = safe_torch_load(
            path
        )

        d = {
            "x0":
                obj["x0"].float(),
            "u":
                obj["u"].float(),
            "x1":
                obj["xT"].float(),
        }

        
        if split != "train":
            required_truth = [
                "true_conditional_mean",
                "true_right_well_prob",
                "J_star_mean",
                "true_finite_response",
            ]

            for key in required_truth:
                if key not in obj:
                    raise RuntimeError(
                        f"{split}: missing simulator truth '{key}'. "
                        "Regenerate with double_well_data.py."
                    )

            d.update(
                {
                    "true_mean":
                        obj[
                            "true_conditional_mean"
                        ].float(),
                    "true_right_prob":
                        obj[
                            "true_right_well_prob"
                        ].float(),
                    "J_star":
                        obj[
                            "J_star_mean"
                        ].float(),
                    "true_finite_response":
                        obj[
                            "true_finite_response"
                        ].float(),
                }
            )

        objects[
            split
        ] = d

    with open(
        data_dir
        / "metadata.json",
        "r",
    ) as f:
        metadata = json.load(
            f
        )

    state_dim = int(
        objects["train"][
            "x0"
        ].shape[1]
    )

    intervention_dim = int(
        objects["train"][
            "u"
        ].shape[1]
    )

    for split in SPLITS:
        d = objects[
            split
        ]

        n = d[
            "x0"
        ].shape[0]

        if d["x1"].shape != d["x0"].shape:
            raise RuntimeError(
                f"{split}: x0/x1 shape mismatch."
            )

        if d["u"].shape[0] != n:
            raise RuntimeError(
                f"{split}: u sample count mismatch."
            )

        if split != "train":
            expected_j = (
                n,
                state_dim,
                intervention_dim,
            )

            if tuple(
                d["J_star"].shape
            ) != expected_j:
                raise RuntimeError(
                    f"{split}: expected J_star shape "
                    f"{expected_j}, got "
                    f"{tuple(d['J_star'].shape)}"
                )

    return (
        objects,
        metadata,
        state_dim,
        intervention_dim,
    )






class ConditionalDriftNet(
    nn.Module
):

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

        for _ in range(
            depth
        ):
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

        self.net = (
            nn.Sequential(
                *layers
            )
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






class ConditionalDSBM:

    def __init__(
        self,
        cfg,
        state_dim,
        intervention_dim,
        device,
    ):
        self.cfg = cfg
        self.state_dim = int(
            state_dim
        )
        self.intervention_dim = int(
            intervention_dim
        )
        self.device = device

        self.net_f = (
            ConditionalDriftNet(
                state_dim,
                intervention_dim,
                cfg.hidden,
                cfg.depth,
            )
            .to(device)
        )

        self.net_b = (
            ConditionalDriftNet(
                state_dim,
                intervention_dim,
                cfg.hidden,
                cfg.depth,
            )
            .to(device)
        )

        self.nets = {
            "f":
                self.net_f,
            "b":
                self.net_b,
        }

        self.prev_fb: Optional[str] = (
            None
        )

    def state_dict(
        self,
    ):
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
            state[
                "net_f"
            ]
        )

        self.net_b.load_state_dict(
            state[
                "net_b"
            ]
        )

        self.prev_fb = state.get(
            "prev_fb",
            None,
        )

    
    
    

    def get_train_tuple(
        self,
        z0,
        z1,
        u,
        fb,
    ):
        bsz = (
            z0.shape[0]
        )

        eps = (
            self.cfg.bridge_eps
        )

        t = (
            torch.rand(
                bsz,
                1,
                device=
                    z0.device,
            )
            * (
                1.0
                - 2.0
                * eps
            )
            + eps
        )

        noise = (
            torch.randn_like(
                z0
            )
        )

        zt = (
            (
                1.0
                - t
            )
            * z0
            + t
            * z1
            + self.cfg.reference_sigma
            * torch.sqrt(
                t
                * (
                    1.0
                    - t
                )
            )
            * noise
        )

        if fb == "f":
            target = (
                (
                    z1
                    - z0
                )
                - self.cfg.reference_sigma
                * torch.sqrt(
                    t
                    / (
                        1.0
                        - t
                    )
                )
                * noise
            )

        elif fb == "b":
            target = (
                -(
                    z1
                    - z0
                )
                - self.cfg.reference_sigma
                * torch.sqrt(
                    (
                        1.0
                        - t
                    )
                    / t
                )
                * noise
            )

        else:
            raise ValueError(
                fb
            )

        return (
            zt,
            u,
            t,
            target,
        )

    
    
    

    def _noise_bank(
        self,
        x,
        num_steps=None,
    ):
        nsteps = (
            self.cfg.num_steps
            if num_steps
            is None
            else int(
                num_steps
            )
        )

        return [
            torch.randn_like(
                x
            )
            for _ in range(
                nsteps
            )
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
            if num_steps
            is None
            else int(
                num_steps
            )
        )

        dt = (
            1.0
            / nsteps
        )

        x = (
            xstart.clone()
        )

        if noise_bank is None:
            noise_bank = (
                self._noise_bank(
                    x,
                    nsteps,
                )
            )

        if fb == "f":
            times = [
                k
                / nsteps
                for k
                in range(
                    nsteps
                )
            ]

        elif fb == "b":
            times = [
                1.0
                - k
                / nsteps
                for k
                in range(
                    nsteps
                )
            ]

        else:
            raise ValueError(
                fb
            )

        net = (
            self.nets[
                fb
            ]
        )

        for k, tv in enumerate(
            times
        ):
            t = (
                torch.full(
                    (
                        x.shape[0],
                        1,
                    ),
                    tv,
                    device=
                        x.device,
                    dtype=
                        x.dtype,
                )
            )

            drift = net(
                x,
                u,
                t,
            )

            x = (
                x
                + dt
                * drift
                + self.cfg.reference_sigma
                * math.sqrt(
                    dt
                )
                * noise_bank[
                    k
                ]
            )

            x = (
                x.detach()
            )

        return x

    
    
    

    @torch.no_grad()
    def regenerate_coupling(
        self,
        data,
    ):
        x0 = (
            data["x0"]
            .to(
                self.device
            )
        )

        x1 = (
            data["x1"]
            .to(
                self.device
            )
        )

        u = (
            data["u"]
            .to(
                self.device
            )
        )

        if self.prev_fb is None:
            return (
                x0,
                x1,
                u,
            )

        if self.prev_fb == "f":
            z0 = x0

            z1 = (
                self.sample_sde(
                    x0,
                    u,
                    fb="f",
                )
            )

        else:
            z0 = (
                self.sample_sde(
                    x1,
                    u,
                    fb="b",
                )
            )

            z1 = x1

        return (
            z0.detach(),
            z1.detach(),
            u,
        )

    
    
    

    def train_pass(
        self,
        data,
        fb,
    ):
        cfg = (
            self.cfg
        )

        (
            z0,
            z1,
            u,
        ) = (
            self.regenerate_coupling(
                data
            )
        )

        net = (
            self.nets[
                fb
            ]
        )

        net.train()

        optimizer = (
            torch.optim.AdamW(
                net.parameters(),
                lr=
                    cfg.lr,
                weight_decay=
                    1e-5,
            )
        )

        n = (
            z0.shape[0]
        )

        recent = []

        for step in range(
            1,
            cfg.inner_steps
            + 1,
        ):
            bsz = min(
                cfg.batch_size,
                n,
            )

            idx = (
                torch.randint(
                    0,
                    n,
                    (
                        bsz,
                    ),
                    device=
                        self.device,
                )
            )

            bz0 = z0[
                idx
            ]

            bz1 = z1[
                idx
            ]

            bu = u[
                idx
            ]

            (
                zt,
                bu,
                t,
                target,
            ) = (
                self.get_train_tuple(
                    bz0,
                    bz1,
                    bu,
                    fb,
                )
            )

            pred = net(
                zt,
                bu,
                t,
            )

            loss = (
                (
                    (
                        pred
                        - target
                    )
                    ** 2
                )
                .sum(
                    dim=1
                )
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

            if len(
                recent
            ) > 100:
                recent.pop(
                    0
                )

            report_every = max(
                100,
                cfg.inner_steps
                // 5,
            )

            if (
                step
                == 1
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

        self.prev_fb = (
            fb
        )

        return {
            "bridge_loss_last100":
                float(
                    np.mean(
                        recent
                    )
                )
        }

    
    
    

    def tangent_rollout(
        self,
        x0,
        u,
        noise_bank=None,
    ):
        
        cfg = (
            self.cfg
        )

        dt = (
            1.0
            / cfg.num_steps
        )

        x = (
            x0.clone()
        )

        u_var = (
            u.clone()
        )

        tangents = [
            torch.zeros_like(
                x
            )
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

            t = (
                torch.full(
                    (
                        x.shape[0],
                        1,
                    ),
                    tv,
                    device=
                        x.device,
                    dtype=
                        x.dtype,
                )
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

            drift_for_state = (
                None
            )

            for j in range(
                self.intervention_dim
            ):
                v = (
                    torch.zeros_like(
                        u_var
                    )
                )

                v[
                    :,
                    j,
                ] = 1.0

                (
                    drift,
                    tangent_drift,
                ) = (
                    torch.autograd.functional.jvp(
                        drift_fn,
                        (
                            x,
                            u_var,
                        ),
                        (
                            tangents[
                                j
                            ],
                            v,
                        ),
                        create_graph=
                            False,
                        strict=
                            False,
                    )
                )

                if drift_for_state is None:
                    drift_for_state = (
                        drift
                    )

                next_tangents.append(
                    tangents[
                        j
                    ]
                    + dt
                    * tangent_drift
                )

            x = (
                x
                + dt
                * drift_for_state
                + cfg.reference_sigma
                * math.sqrt(
                    dt
                )
                * noise_bank[
                    k
                ]
            )

            x = (
                x.detach()
            )

            tangents = [
                r.detach()
                for r
                in next_tangents
            ]

        return (
            torch.stack(
                tangents,
                dim=-1,
            )
        )






@torch.no_grad()
def endpoint_distribution_metrics(
    model,
    data,
    cfg,
    device,
):
    x0 = (
        data["x0"]
        .to(device)
    )

    u = (
        data["u"]
        .to(device)
    )

    true_mean = (
        data["true_mean"]
        .to(device)
    )

    true_right_prob = (
        data["true_right_prob"]
        .to(device)
    )

    sum_y = (
        torch.zeros_like(
            true_mean
        )
    )

    sum_right = (
        torch.zeros_like(
            true_right_prob
        )
    )

    for _ in range(
        cfg.eval_mc
    ):
        y = (
            model.sample_sde(
                x0,
                u,
                fb="f",
            )
        )

        sum_y = (
            sum_y
            + y
        )

        sum_right = (
            sum_right
            + (
                y
                > 0.0
            )
            .float()
        )

    pred_mean = (
        sum_y
        / float(
            cfg.eval_mc
        )
    )

    pred_right_prob = (
        sum_right
        / float(
            cfg.eval_mc
        )
    )

    mean_rmse = (
        torch.sqrt(
            (
                (
                    pred_mean
                    - true_mean
                )
                ** 2
            )
            .mean()
        )
    )

    right_prob_rmse = (
        torch.sqrt(
            (
                (
                    pred_right_prob
                    - true_right_prob
                )
                ** 2
            )
            .mean()
        )
    )

    return {
        "mean_rmse":
            float(
                mean_rmse
                .detach()
                .cpu()
            ),

        "right_well_prob_rmse":
            float(
                right_prob_rmse
                .detach()
                .cpu()
            ),

        "mean_predicted_endpoint":
            float(
                pred_mean
                .mean()
                .detach()
                .cpu()
            ),

        "mean_true_endpoint":
            float(
                true_mean
                .mean()
                .detach()
                .cpu()
            ),

        "mean_predicted_right_well_prob":
            float(
                pred_right_prob
                .mean()
                .detach()
                .cpu()
            ),

        "mean_true_right_well_prob":
            float(
                true_right_prob
                .mean()
                .detach()
                .cpu()
            ),
    }


def sensitivity_metrics(
    model,
    data,
    cfg,
    device,
):
    
    n = min(
        cfg.eval_batch_size,
        data["x0"]
        .shape[0],
    )

    x0 = (
        data["x0"][
            :n
        ]
        .to(
            device
        )
    )

    u = (
        data["u"][
            :n
        ]
        .to(
            device
        )
    )

    J_true = (
        data["J_star"][
            :n
        ]
        .to(
            device
        )
    )

    Js = []

    with torch.enable_grad():
        for _ in range(
            cfg.eval_sens_mc
        ):
            J = (
                model.tangent_rollout(
                    x0,
                    u,
                )
            )

            Js.append(
                J.detach()
            )

    J_pred = (
        torch.stack(
            Js,
            dim=0,
        )
        .mean(
            dim=0
        )
    )

    diff = (
        J_pred
        - J_true
    )

    rel = (
        torch.linalg.vector_norm(
            diff.reshape(
                -1
            )
        )
        / torch.linalg.vector_norm(
            J_true.reshape(
                -1
            )
        )
        .clamp_min(
            1e-8
        )
    )

    rmse = (
        torch.sqrt(
            (
                diff
                ** 2
            )
            .mean()
        )
    )

    return {
        "jacobian_rel_error":
            float(
                rel.detach()
                .cpu()
            ),

        "jacobian_rmse":
            float(
                rmse.detach()
                .cpu()
            ),

        "mean_predicted_jacobian":
            float(
                J_pred
                .mean()
                .detach()
                .cpu()
            ),

        "mean_true_jacobian":
            float(
                J_true
                .mean()
                .detach()
                .cpu()
            ),
    }


@torch.no_grad()
def finite_response_metric(
    model,
    data,
    cfg,
    device,
):
    n = min(
        cfg.eval_batch_size,
        data["x0"]
        .shape[0],
    )

    x0 = (
        data["x0"][
            :n
        ]
        .to(
            device
        )
    )

    u = (
        data["u"][
            :n
        ]
        .to(
            device
        )
    )

    true_change = (
        data[
            "true_finite_response"
        ][
            :n
        ]
        .to(
            device
        )
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

        y0 = (
            model.sample_sde(
                x0,
                u,
                fb="f",
                noise_bank=
                    noise_bank,
            )
        )

        y1 = (
            model.sample_sde(
                x0,
                u
                + cfg.finite_delta,
                fb="f",
                noise_bank=
                    noise_bank,
            )
        )

        predicted_changes.append(
            y1
            - y0
        )

    pred_change = (
        torch.stack(
            predicted_changes,
            dim=0,
        )
        .mean(
            dim=0
        )
    )

    rmse = (
        torch.sqrt(
            (
                (
                    pred_change
                    - true_change
                )
                ** 2
            )
            .mean()
        )
    )

    return {
        "finite_response_rmse":
            float(
                rmse.detach()
                .cpu()
            ),

        "mean_predicted_finite_response":
            float(
                pred_change
                .mean()
                .detach()
                .cpu()
            ),

        "mean_true_finite_response":
            float(
                true_change
                .mean()
                .detach()
                .cpu()
            ),
    }


def evaluate_split(
    name,
    model,
    data,
    cfg,
    device,
):
    log("")
    log(
        "=" * 68
    )
    log(
        "EVALUATING",
        name,
    )
    log(
        "=" * 68
    )

    out = (
        endpoint_distribution_metrics(
            model,
            data,
            cfg,
            device,
        )
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
        finite_response_metric(
            model,
            data,
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






def convergence_metrics(
    model,
    train_data,
    cfg,
    device,
):
    
    n = min(
        1024,
        train_data["x0"]
        .shape[0],
    )

    x0 = (
        train_data[
            "x0"
        ][
            :n
        ]
        .to(
            device
        )
    )

    u = (
        train_data[
            "u"
        ][
            :n
        ]
        .to(
            device
        )
    )

    x1 = (
        train_data[
            "x1"
        ][
            :n
        ]
        .to(
            device
        )
    )

    with torch.no_grad():
        pred_sum = (
            torch.zeros_like(
                x1
            )
        )

        mc = min(
            cfg.eval_mc,
            8,
        )

        for _ in range(
            mc
        ):
            pred_sum += (
                model.sample_sde(
                    x0,
                    u,
                    fb="f",
                )
            )

        pred_mean = (
            pred_sum
            / float(
                mc
            )
        )

        empirical_endpoint_rmse = (
            torch.sqrt(
                (
                    (
                        pred_mean
                        - x1
                    )
                    ** 2
                )
                .mean()
            )
        )

    return {
        "train_empirical_endpoint_rmse":
            float(
                empirical_endpoint_rmse
                .detach()
                .cpu()
            )
    }






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
                asdict(
                    cfg
                ),
            "imf":
                int(
                    imf
                ),
            "state_dim":
                int(
                    state_dim
                ),
            "intervention_dim":
                int(
                    intervention_dim
                ),
            "rng_state":
                capture_rng_state(),
        },
        path,
    )


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

    for h in history:
        rows.append(
            {
                "imf":
                    h[
                        "imf"
                    ],
                "backward_bridge_loss_last100":
                    h[
                        "backward_bridge_loss_last100"
                    ],
                "forward_bridge_loss_last100":
                    h[
                        "forward_bridge_loss_last100"
                    ],
                "train_empirical_endpoint_rmse":
                    h[
                        "metrics"
                    ][
                        "train_empirical_endpoint_rmse"
                    ],
            }
        )

    with open(
        run_dir
        / "convergence.csv",
        "w",
        newline="",
    ) as f:
        writer = (
            csv.DictWriter(
                f,
                fieldnames=
                    list(
                        rows[
                            0
                        ]
                        .keys()
                    ),
            )
        )

        writer.writeheader()
        writer.writerows(
            rows
        )






def parse_args():
    p = (
        argparse.ArgumentParser()
    )

    p.add_argument(
        "--data-dir",
        type=str,
        default=
            "runs/double_well_data",
    )

    p.add_argument(
        "--run-root",
        type=str,
        default=
            "runs/double_well_conditional",
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
        "--fork-imf",
        type=int,
        default=3,
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
        "--eval-sens-mc",
        type=int,
        default=16,
    )

    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=512,
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

    return (
        p.parse_args()
    )






def main():
    args = (
        parse_args()
    )

    cfg = Config(
        seed=
            args.seed,
        num_steps=
            args.num_steps,
        reference_sigma=
            args.reference_sigma,
        total_imf=
            args.total_imf,
        fork_imf=
            args.fork_imf,
        inner_steps=
            args.inner_steps,
        batch_size=
            args.batch_size,
        lr=
            args.lr,
        eval_mc=
            args.eval_mc,
        eval_sens_mc=
            args.eval_sens_mc,
        eval_batch_size=
            args.eval_batch_size,
    )

    if cfg.fork_imf >= cfg.total_imf:
        
        if cfg.fork_imf != cfg.total_imf:
            raise ValueError(
                "fork-imf must be <= total-imf."
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
        if args.device
        is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    device = (
        torch.device(
            device
        )
    )

    data_dir = (
        Path(
            args.data_dir
        )
    )

    run_dir = (
        Path(
            args.run_root
        )
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

    set_seed(
        cfg.seed
    )

    (
        datasets,
        metadata,
        state_dim,
        intervention_dim,
    ) = load_dataset(
        data_dir
    )

    
    cfg.finite_delta = float(
        metadata.get(
            "finite_delta",
            cfg.finite_delta,
        )
    )

    log(
        "=" * 80
    )

    log(
        "DOUBLE-WELL CONDITIONAL DSBM"
    )

    log(
        "=" * 80
    )

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
            asdict(
                cfg
            ),
            indent=2,
        )
    )

    model = (
        ConditionalDSBM(
            cfg=
                cfg,
            state_dim=
                state_dim,
            intervention_dim=
                intervention_dim,
            device=
                device,
        )
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
        datasets[
            "train"
        ]
    )

    history = []

    start_time = (
        time.time()
    )

    for imf in range(
        1,
        cfg.total_imf
        + 1,
    ):
        log("")
        log(
            "=" * 80
        )
        log(
            f"IMF {imf}/"
            f"{cfg.total_imf} "
            "- BACKWARD"
        )
        log(
            "=" * 80
        )

        b_stats = (
            model.train_pass(
                train_data,
                fb="b",
            )
        )

        log("")
        log(
            "=" * 80
        )
        log(
            f"IMF {imf}/"
            f"{cfg.total_imf} "
            "- FORWARD"
        )
        log(
            "=" * 80
        )

        f_stats = (
            model.train_pass(
                train_data,
                fb="f",
            )
        )

        checkpoint_path = (
            run_dir
            / f"imf_{imf}.pt"
        )

        save_checkpoint(
            checkpoint_path,
            model,
            cfg,
            imf,
            state_dim,
            intervention_dim,
        )

        log(
            "Saved checkpoint:",
            checkpoint_path,
        )

        model.net_f.eval()
        model.net_b.eval()

        conv = (
            convergence_metrics(
                model,
                train_data,
                cfg,
                device,
            )
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
            "| empirical train endpoint RMSE =",
            f"{conv['train_empirical_endpoint_rmse']:.6f}",
        )

    train_seconds = (
        time.time()
        - start_time
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
        results[
            split
        ] = (
            evaluate_split(
                split,
                model,
                datasets[
                    split
                ],
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
                asdict(
                    cfg
                ),
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
        "benchmark":
            "double_well",
        "seed":
            cfg.seed,
        "config":
            asdict(
                cfg
            ),
        "train_seconds":
            train_seconds,
        "shared_tangent_fork":
            str(
                run_dir
                / f"imf_{cfg.fork_imf}.pt"
            ),
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

    for split, m in (
        results.items()
    ):
        rows.append(
            {
                "method":
                    "conditional_dsbm",
                "seed":
                    cfg.seed,
                "split":
                    split,
                "mean_rmse":
                    m[
                        "mean_rmse"
                    ],
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
        run_dir
        / "metrics.csv",
        "w",
        newline="",
    ) as f:
        writer = (
            csv.DictWriter(
                f,
                fieldnames=
                    list(
                        rows[
                            0
                        ]
                        .keys()
                    ),
            )
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    log("")
    log(
        "=" * 80
    )

    log(
        "DOUBLE-WELL CONDITIONAL DSBM FINAL SUMMARY"
    )

    log(
        "=" * 80
    )

    for split, m in (
        results.items()
    ):
        log(
            split,
            "| mean RMSE =",
            f"{m['mean_rmse']:.6f}",
            "| right-well prob RMSE =",
            f"{m['right_well_prob_rmse']:.6f}",
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
        / f"imf_{cfg.fork_imf}.pt",
    )

    log(
        "Metrics:",
        run_dir
        / "metrics.json",
    )


if __name__ == "__main__":
    main()
