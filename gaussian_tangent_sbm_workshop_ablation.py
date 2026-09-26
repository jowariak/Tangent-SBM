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

try:
    import gaussian_conditional_dsbm_nonlinear as base
except ImportError as exc:
    raise ImportError(
        "Could not import gaussian_conditional_dsbm_nonlinear.py.\n"
        "Place gaussian_tangent_sbm_nonlinear.py in the SAME repo "
        "directory as gaussian_conditional_dsbm_nonlinear.py."
    ) from exc






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






def restore_rng_state(state):
    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch.set_rng_state(
        state["torch"]
    )

    if (
        torch.cuda.is_available()
        and "cuda" in state
    ):
        torch.cuda.set_rng_state_all(
            state["cuda"]
        )







def load_response_collocation(
    path,
    state_dim,
    intervention_dim,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing response-only collocation file:\n{path}\n\n"
            "Run gaussian_response_collocation.py first."
        )

    obj = base.safe_torch_load(
        path,
        map_location="cpu",
    )

    required = [
        "x0",
        "u",
        "J_star",
    ]

    for key in required:
        if key not in obj:
            raise RuntimeError(
                f"Response collocation file missing key: {key}"
            )

    if not bool(
        obj.get(
            "response_only",
            False,
        )
    ):
        raise RuntimeError(
            "Expected a response-only collocation file."
        )

    if "xT" in obj:
        raise RuntimeError(
            "Response collocation file unexpectedly contains xT. "
            "This experiment is intended to add sensitivity-only supervision."
        )

    x0 = obj["x0"].float()
    u = obj["u"].float()
    J_star = obj["J_star"].float()

    if x0.ndim != 2 or x0.shape[1] != state_dim:
        raise RuntimeError(
            f"Collocation x0 must be [N,{state_dim}], got {tuple(x0.shape)}"
        )

    if (
        u.ndim != 2
        or u.shape[1]
        != intervention_dim
    ):
        raise RuntimeError(
            "Collocation u shape mismatch."
        )

    expected_j = (
        x0.shape[0],
        state_dim,
        intervention_dim,
    )

    if tuple(
        J_star.shape
    ) != expected_j:
        raise RuntimeError(
            f"Collocation J_star must have shape {expected_j}, "
            f"got {tuple(J_star.shape)}"
        )

    return {
        "x0": x0,
        "u": u,
        "J_star": J_star,
        "metadata": obj.get(
            "metadata",
            {},
        ),
    }






class TangentDSBM(
    base.ConditionalDSBM
):

    def __init__(
        self,
        cfg,
        state_dim,
        intervention_dim,
        device,
        lambda_sens,
        sens_every,
        sens_batch_size,
        sens_steps,
        sens_seed,
        anchor_fraction,
    ):
        super().__init__(
            cfg=cfg,
            state_dim=state_dim,
            intervention_dim=
                intervention_dim,
            device=device,
        )

        self.lambda_sens = (
            float(lambda_sens)
        )

        self.sens_every = int(
            sens_every
        )

        self.sens_batch_size = int(
            sens_batch_size
        )

        self.sens_steps = int(
            sens_steps
        )

        self.anchor_fraction = float(
            anchor_fraction
        )

        if not (
            0.0
            <= self.anchor_fraction
            <= 1.0
        ):
            raise ValueError(
                "anchor_fraction must lie in [0,1]."
            )

        
        
        self.sens_generator = (
            torch.Generator(
                device="cpu"
            )
            .manual_seed(
                int(sens_seed)
            )
        )

    
    
    

    def _sens_rand_indices(
        self,
        n,
        bsz,
    ):
        return torch.randint(
            low=0,
            high=n,
            size=(bsz,),
            generator=self.sens_generator,
            device="cpu",
        )

    def _sens_noise_bank(
        self,
        batch_size,
        state_dim,
        dtype,
    ):
        
        bank = []

        for _ in range(
            self.sens_steps
        ):
            eps_cpu = torch.randn(
                batch_size,
                state_dim,
                generator=self.sens_generator,
                dtype=dtype,
                device="cpu",
            )

            bank.append(
                eps_cpu.to(
                    self.device,
                    non_blocking=False,
                )
            )

        return bank

    
    
    

    def tangent_training_rollout(
        self,
        x0,
        u,
        noise_bank,
    ):
        
        if self.intervention_dim != 1:
            raise NotImplementedError(
                "The current nonlinear Gaussian benchmark uses scalar u."
            )

        dt = (
            1.0
            / self.sens_steps
        )

        x = x0
        u_var = u

        
        R = torch.zeros_like(
            x
        )

        
        v = torch.ones_like(
            u_var
        )

        for k in range(
            self.sens_steps
        ):
            tv = (
                k
                / self.sens_steps
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
                        R,
                        v,
                    ),
                    create_graph=True,
                    strict=False,
                )
            )

            
            
            
            x = (
                x
                + dt * drift
                + self.cfg.reference_sigma
                * math.sqrt(dt)
                * noise_bank[k]
            )

            R = (
                R
                + dt * tangent_drift
            )

        
        
        return R.unsqueeze(-1)

    def _sample_response_batch(
        self,
        source_data,
        batch_size,
    ):
        
        if batch_size <= 0:
            return None

        n = int(
            source_data["x0"]
            .shape[0]
        )

        idx_cpu = (
            self._sens_rand_indices(
                n,
                batch_size,
            )
        )

        return {
            "x0":
                source_data["x0"][
                    idx_cpu
                ]
                .to(
                    self.device
                ),
            "u":
                source_data["u"][
                    idx_cpu
                ]
                .to(
                    self.device
                ),
            "J_star":
                source_data["J_star"][
                    idx_cpu
                ]
                .to(
                    self.device
                ),
        }

    def mixed_sensitivity_loss(
        self,
        anchor_data,
        collocation_data,
    ):
        
        total_bsz = min(
            self.sens_batch_size,
            int(
                anchor_data["x0"]
                .shape[0]
            )
            + int(
                collocation_data["x0"]
                .shape[0]
            ),
        )

        n_anchor = int(
            round(
                total_bsz
                * self.anchor_fraction
            )
        )

        n_anchor = max(
            0,
            min(
                n_anchor,
                total_bsz,
            ),
        )

        n_colloc = (
            total_bsz
            - n_anchor
        )

        pieces = []

        anchor_batch = (
            self._sample_response_batch(
                anchor_data,
                n_anchor,
            )
        )

        if anchor_batch is not None:
            pieces.append(
                (
                    "anchor",
                    anchor_batch,
                )
            )

        colloc_batch = (
            self._sample_response_batch(
                collocation_data,
                n_colloc,
            )
        )

        if colloc_batch is not None:
            pieces.append(
                (
                    "collocation",
                    colloc_batch,
                )
            )

        if not pieces:
            raise RuntimeError(
                "Sensitivity minibatch is empty."
            )

        x0 = torch.cat(
            [
                batch["x0"]
                for _,
                batch
                in pieces
            ],
            dim=0,
        )

        u = torch.cat(
            [
                batch["u"]
                for _,
                batch
                in pieces
            ],
            dim=0,
        )

        J_star = torch.cat(
            [
                batch["J_star"]
                for _,
                batch
                in pieces
            ],
            dim=0,
        )

        
        
        perm_cpu = torch.randperm(
            x0.shape[0],
            generator=
                self.sens_generator,
            device="cpu",
        )

        perm = perm_cpu.to(
            self.device
        )

        x0 = x0[
            perm
        ]

        u = u[
            perm
        ]

        J_star = J_star[
            perm
        ]

        noise_bank = (
            self._sens_noise_bank(
                batch_size=
                    x0.shape[0],
                state_dim=
                    self.state_dim,
                dtype=x0.dtype,
            )
        )

        J_theta = (
            self.tangent_training_rollout(
                x0=x0,
                u=u,
                noise_bank=noise_bank,
            )
        )

        per_sample_sq = (
            (
                J_theta
                - J_star
            )
            ** 2
        ).mean(
            dim=(1, 2)
        )

        loss = (
            per_sample_sq.mean()
        )

        
        
        
        
        diagnostics = {
            "n_anchor":
                int(
                    n_anchor
                ),
            "n_collocation":
                int(
                    n_colloc
                ),
            "anchor_fraction":
                float(
                    self.anchor_fraction
                ),
        }

        return (
            loss,
            J_theta.detach(),
            J_star.detach(),
            diagnostics,
        )

    
    
    

    def train_pass_tangent(
        self,
        data,
        fb,
        anchor_sensitivity_data=None,
        collocation_sensitivity_data=None,
    ):
        
        cfg = self.cfg

        (
            z0,
            z1,
            u,
        ) = self.regenerate_coupling(
            data
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

        recent_bridge = []
        recent_sens = []
        recent_total = []

        n_sens_updates = 0

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

            bridge_loss = (
                (
                    (pred - target)
                    ** 2
                )
                .sum(dim=1)
                .mean()
            )

            sens_loss = None

            if (
                fb == "f"
                and self.lambda_sens > 0.0
                and step
                % self.sens_every
                == 0
            ):
                if (
                    anchor_sensitivity_data
                    is None
                    or collocation_sensitivity_data
                    is None
                ):
                    raise RuntimeError(
                        "Forward Tangent-SBM update requires both "
                        "anchor and collocation sensitivity data."
                    )

                (
                    sens_loss,
                    _,
                    _,
                    mix_diag,
                ) = self.mixed_sensitivity_loss(
                    anchor_data=
                        anchor_sensitivity_data,
                    collocation_data=
                        collocation_sensitivity_data,
                )

                total_loss = (
                    bridge_loss
                    + self.lambda_sens
                    * sens_loss
                )

                n_sens_updates += 1

            else:
                total_loss = (
                    bridge_loss
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                cfg.grad_clip,
            )

            optimizer.step()

            bridge_val = float(
                bridge_loss
                .detach()
                .cpu()
            )

            total_val = float(
                total_loss
                .detach()
                .cpu()
            )

            recent_bridge.append(
                bridge_val
            )

            recent_total.append(
                total_val
            )

            if sens_loss is not None:
                recent_sens.append(
                    float(
                        sens_loss
                        .detach()
                        .cpu()
                    )
                )

            if len(
                recent_bridge
            ) > 100:
                recent_bridge.pop(0)

            if len(
                recent_total
            ) > 100:
                recent_total.pop(0)

            if len(
                recent_sens
            ) > 100:
                recent_sens.pop(0)

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
                if recent_sens:
                    sens_text = (
                        f"{np.mean(recent_sens):.6f}"
                    )
                else:
                    sens_text = "n/a"

                log(
                    f"{fb} step "
                    f"{step:5d}/"
                    f"{cfg.inner_steps}",
                    f"bridge="
                    f"{bridge_val:.6f}",
                    f"sens_recent="
                    f"{sens_text}",
                    f"total="
                    f"{total_val:.6f}",
                )

        self.prev_fb = fb

        return {
            "bridge_loss_last100":
                float(
                    np.mean(
                        recent_bridge
                    )
                ),
            "sensitivity_loss_last100":
                (
                    float(
                        np.mean(
                            recent_sens
                        )
                    )
                    if recent_sens
                    else None
                ),
            "total_loss_last100":
                float(
                    np.mean(
                        recent_total
                    )
                ),
            "num_sensitivity_updates":
                int(
                    n_sens_updates
                ),
        }







def gradient_flow_sanity_check(
    model,
    train_data,
    max_batch_size=32,
):
    
    original_batch_size = (
        model.sens_batch_size
    )

    
    sens_rng_state = (
        model.sens_generator
        .get_state()
    )

    model.sens_batch_size = min(
        int(max_batch_size),
        int(original_batch_size),
        int(
            train_data["x0"]
            .shape[0]
        ),
    )

    model.net_f.train()
    model.net_b.train()

    model.net_f.zero_grad(
        set_to_none=True
    )
    model.net_b.zero_grad(
        set_to_none=True
    )

    try:
        (
            sens_loss,
            _,
            _,
        ) = model.sensitivity_loss(
            train_data
        )

        if not torch.isfinite(
            sens_loss
        ):
            raise RuntimeError(
                "Gradient sanity check failed: "
                "sensitivity loss is non-finite."
            )

        sens_loss.backward()

        forward_sq_sum = 0.0
        forward_num_grads = 0
        all_forward_finite = True

        for param in (
            model.net_f.parameters()
        ):
            if param.grad is None:
                continue

            grad = (
                param.grad.detach()
            )

            forward_num_grads += 1

            if not torch.isfinite(
                grad
            ).all():
                all_forward_finite = False

            forward_sq_sum += float(
                (grad ** 2)
                .sum()
                .cpu()
            )

        forward_grad_norm = (
            forward_sq_sum
            ** 0.5
        )

        backward_sq_sum = 0.0
        backward_num_grads = 0

        for param in (
            model.net_b.parameters()
        ):
            if param.grad is None:
                continue

            backward_num_grads += 1

            backward_sq_sum += float(
                (
                    param.grad.detach()
                    ** 2
                )
                .sum()
                .cpu()
            )

        backward_grad_norm = (
            backward_sq_sum
            ** 0.5
        )

        if not all_forward_finite:
            raise RuntimeError(
                "Gradient sanity check failed: "
                "non-finite forward-drift gradient."
            )

        if (
            forward_num_grads == 0
            or forward_grad_norm
            <= 0.0
        ):
            raise RuntimeError(
                "Gradient sanity check failed: "
                "tangent sensitivity loss did not "
                "produce a nonzero gradient on net_f."
            )

        
        if backward_grad_norm > 0.0:
            raise RuntimeError(
                "Gradient sanity check failed: "
                "forward sensitivity loss unexpectedly "
                "produced gradient on net_b."
            )

        return {
            "sensitivity_loss":
                float(
                    sens_loss
                    .detach()
                    .cpu()
                ),
            "forward_grad_norm":
                float(
                    forward_grad_norm
                ),
            "forward_num_grad_tensors":
                int(
                    forward_num_grads
                ),
            "backward_grad_norm":
                float(
                    backward_grad_norm
                ),
            "backward_num_grad_tensors":
                int(
                    backward_num_grads
                ),
            "batch_size":
                int(
                    model.sens_batch_size
                ),
        }

    finally:
        
        model.net_f.zero_grad(
            set_to_none=True
        )
        model.net_b.zero_grad(
            set_to_none=True
        )

        model.sens_batch_size = (
            original_batch_size
        )

        model.sens_generator.set_state(
            sens_rng_state
        )



def mixed_gradient_flow_sanity_check(
    model,
    anchor_data,
    collocation_data,
    max_batch_size=32,
):
    
    original_batch_size = (
        model.sens_batch_size
    )

    sens_rng_state = (
        model.sens_generator
        .get_state()
    )

    model.sens_batch_size = min(
        int(max_batch_size),
        int(original_batch_size),
    )

    model.net_f.train()
    model.net_b.train()

    model.net_f.zero_grad(
        set_to_none=True
    )
    model.net_b.zero_grad(
        set_to_none=True
    )

    try:
        (
            sens_loss,
            _,
            _,
            mix_diag,
        ) = model.mixed_sensitivity_loss(
            anchor_data=
                anchor_data,
            collocation_data=
                collocation_data,
        )

        if not torch.isfinite(
            sens_loss
        ):
            raise RuntimeError(
                "Mixed gradient sanity check failed: "
                "non-finite sensitivity loss."
            )

        sens_loss.backward()

        forward_sq_sum = 0.0
        forward_num_grads = 0

        for param in (
            model.net_f.parameters()
        ):
            if param.grad is None:
                continue

            grad = (
                param.grad.detach()
            )

            if not torch.isfinite(
                grad
            ).all():
                raise RuntimeError(
                    "Mixed gradient sanity check failed: "
                    "non-finite net_f gradient."
                )

            forward_num_grads += 1

            forward_sq_sum += float(
                (
                    grad
                    ** 2
                )
                .sum()
                .cpu()
            )

        forward_grad_norm = (
            forward_sq_sum
            ** 0.5
        )

        backward_sq_sum = 0.0

        for param in (
            model.net_b.parameters()
        ):
            if param.grad is None:
                continue

            backward_sq_sum += float(
                (
                    param.grad.detach()
                    ** 2
                )
                .sum()
                .cpu()
            )

        backward_grad_norm = (
            backward_sq_sum
            ** 0.5
        )

        if (
            forward_num_grads == 0
            or forward_grad_norm
            <= 0.0
        ):
            raise RuntimeError(
                "Mixed gradient sanity check failed: "
                "no nonzero gradient on net_f."
            )

        if backward_grad_norm > 0.0:
            raise RuntimeError(
                "Mixed gradient sanity check failed: "
                "unexpected gradient on net_b."
            )

        return {
            "sensitivity_loss":
                float(
                    sens_loss
                    .detach()
                    .cpu()
                ),
            "forward_grad_norm":
                float(
                    forward_grad_norm
                ),
            "backward_grad_norm":
                float(
                    backward_grad_norm
                ),
            "n_anchor":
                int(
                    mix_diag[
                        "n_anchor"
                    ]
                ),
            "n_collocation":
                int(
                    mix_diag[
                        "n_collocation"
                    ]
                ),
        }

    finally:
        model.net_f.zero_grad(
            set_to_none=True
        )
        model.net_b.zero_grad(
            set_to_none=True
        )

        model.sens_batch_size = (
            original_batch_size
        )

        model.sens_generator.set_state(
            sens_rng_state
        )






def save_tangent_checkpoint(
    path,
    model,
    cfg,
    imf,
    state_dim,
    intervention_dim,
    tangent_config,
):
    torch.save(
        {
            "model":
                model.state_dict(),
            "config":
                asdict(cfg),
            "tangent_config":
                tangent_config,
            "imf":
                imf,
            "state_dim":
                state_dim,
            "intervention_dim":
                intervention_dim,
            "rng_state":
                base.capture_rng_state(),
        },
        path,
    )


def maybe_write_comparison(
    baseline_metrics_path,
    tangent_results,
    run_dir,
    seed,
):
    if not baseline_metrics_path.exists():
        log(
            "Baseline metrics not found; "
            "skipping automatic comparison:",
            baseline_metrics_path,
        )
        return

    with open(
        baseline_metrics_path,
        "r",
    ) as f:
        baseline_payload = json.load(
            f
        )

    baseline_results = (
        baseline_payload[
            "metrics"
        ]
    )

    rows = []

    for split, tm in (
        tangent_results.items()
    ):
        if split not in baseline_results:
            continue

        bm = baseline_results[
            split
        ]

        rows.append(
            {
                "seed":
                    seed,
                "split":
                    split,

                "conditional_mean_rmse":
                    bm["mean_rmse"],
                "tangent_mean_rmse":
                    tm["mean_rmse"],
                "delta_mean_rmse":
                    tm["mean_rmse"]
                    - bm["mean_rmse"],

                "conditional_j_rel_error":
                    bm[
                        "jacobian_rel_error"
                    ],
                "tangent_j_rel_error":
                    tm[
                        "jacobian_rel_error"
                    ],
                "delta_j_rel_error":
                    tm[
                        "jacobian_rel_error"
                    ]
                    - bm[
                        "jacobian_rel_error"
                    ],

                "conditional_finite_response_rmse":
                    bm[
                        "finite_response_rmse"
                    ],
                "tangent_finite_response_rmse":
                    tm[
                        "finite_response_rmse"
                    ],
                "delta_finite_response_rmse":
                    tm[
                        "finite_response_rmse"
                    ]
                    - bm[
                        "finite_response_rmse"
                    ],
            }
        )

    if not rows:
        return

    out = (
        run_dir
        / "comparison_vs_conditional.csv"
    )

    with open(
        out,
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
        writer.writerows(
            rows
        )

    log(
        "Saved automatic baseline comparison:",
        out,
    )







def _clone_response_view(data):
    
    return {
        "x0": data["x0"].clone(),
        "u": data["u"].clone(),
        "J_star": data["J_star"].clone(),
    }


def _deterministic_response_subset(
    data,
    fraction,
    seed,
):
    
    fraction = float(fraction)

    if not (
        0.0 < fraction <= 1.0
    ):
        raise ValueError(
            "response_fraction must lie in (0,1]."
        )

    n = int(
        data["x0"].shape[0]
    )

    keep = max(
        1,
        int(
            round(
                n * fraction
            )
        ),
    )

    g = torch.Generator(
        device="cpu"
    ).manual_seed(
        int(seed)
    )

    perm = torch.randperm(
        n,
        generator=g,
        device="cpu",
    )

    idx = perm[:keep]

    return {
        "x0":
            data["x0"][
                idx
            ].clone(),
        "u":
            data["u"][
                idx
            ].clone(),
        "J_star":
            data["J_star"][
                idx
            ].clone(),
    }


def _apply_target_mode(
    data,
    mode,
    seed,
):
    
    out = _clone_response_view(
        data
    )

    mode = str(
        mode
    ).lower()

    if mode == "correct":
        return out

    if mode == "shuffle":
        n = int(
            out["J_star"]
            .shape[0]
        )

        g = torch.Generator(
            device="cpu"
        ).manual_seed(
            int(seed)
        )

        perm = torch.randperm(
            n,
            generator=g,
            device="cpu",
        )

        out["J_star"] = (
            out["J_star"][
                perm
            ].clone()
        )

        return out

    if mode == "signflip":
        out["J_star"] = (
            -out["J_star"]
        )

        return out

    raise ValueError(
        "target_mode must be one of: "
        "correct, shuffle, signflip"
    )






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
        "--baseline-run-root",
        type=str,
        default=(
            "runs/"
            "gaussian_conditional_nonlinear"
        ),
        help=(
            "Root containing the shared IMF-3 checkpoint used for the fork."
        ),
    )

    p.add_argument(
        "--comparison-baseline-root",
        type=str,
        default=None,
        help=(
            "Optional root containing the matched final conditional baseline "
            "metrics for automatic comparison. If omitted, uses "
            "--baseline-run-root."
        ),
    )

    p.add_argument(
        "--run-root",
        type=str,
        default=(
            "runs/"
            "gaussian_tangent_nonlinear"
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=32,
    )

    p.add_argument(
        "--fork-imf",
        type=int,
        default=3,
    )

    p.add_argument(
        "--total-imf",
        type=int,
        default=5,
    )

    p.add_argument(
        "--lambda-sens",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--response-collocation",
        type=str,
        default=(
            "runs/"
            "gaussian_nonlinear_data/"
            "response_collocation.pt"
        ),
        help=(
            "Response-only supervision file containing x0, u, J_star "
            "and no endpoint xT."
        ),
    )

    p.add_argument(
        "--sens-every",
        type=int,
        default=5,
    )

    p.add_argument(
        "--sens-batch-size",
        type=int,
        default=256,
    )

    p.add_argument(
        "--anchor-fraction",
        type=float,
        default=0.50,
        help=(
            "Fraction of each sensitivity minibatch drawn from the "
            "original endpoint-anchor samples; the remainder comes "
            "from response-only collocation samples."
        ),
    )

    p.add_argument(
        "--response-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of response-collocation operating points retained. "
            "Use 0.25, 0.50, 1.0 for the workshop coverage ablation."
        ),
    )

    p.add_argument(
        "--target-mode",
        type=str,
        default="correct",
        choices=[
            "correct",
            "shuffle",
            "signflip",
        ],
        help=(
            "Response-target control. 'shuffle' preserves the marginal "
            "J* distribution but destroys alignment with u."
        ),
    )

    p.add_argument(
        "--sens-steps",
        type=int,
        default=None,
        help=(
            "Tangent rollout steps. "
            "Default: same as bridge num_steps."
        ),
    )

    
    p.add_argument(
        "--inner-steps",
        type=int,
        default=None,
        help=(
            "Override checkpoint inner_steps. "
            "Use only for smoke testing unless "
            "you rerun all baselines fairly."
        ),
    )

    p.add_argument(
        "--eval-mc",
        type=int,
        default=None,
    )

    p.add_argument(
        "--eval-sens-mc",
        type=int,
        default=None,
    )

    p.add_argument(
        "--device",
        type=str,
        default=None,
    )

    p.add_argument(
        "--no-rng-replay",
        action="store_true",
        help=(
            "Skip replaying the baseline's IMF-fork "
            "convergence diagnostic. Normally leave this OFF."
        ),
    )

    p.add_argument(
        "--skip-gradient-sanity-check",
        action="store_true",
        help=(
            "Skip the one-time pre-training check that the "
            "tangent loss produces nonzero gradients on net_f. "
            "Normally leave this OFF."
        ),
    )

    return p.parse_args()






def main():
    args = parse_args()

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

    baseline_seed_dir = (
        Path(
            args.baseline_run_root
        )
        / f"seed_{args.seed}"
    )

    fork_checkpoint = (
        baseline_seed_dir
        / f"imf_{args.fork_imf}.pt"
    )

    if not fork_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing baseline fork checkpoint:\n"
            f"{fork_checkpoint}\n\n"
            "Run gaussian_conditional_dsbm_nonlinear.py first."
        )

    checkpoint = (
        base.safe_torch_load(
            fork_checkpoint,
            map_location="cpu",
        )
    )

    if int(
        checkpoint["imf"]
    ) != int(
        args.fork_imf
    ):
        raise RuntimeError(
            "Checkpoint IMF does not match requested fork."
        )

    
    cfg = base.Config(
        **checkpoint["config"]
    )

    cfg.fork_imf = int(
        args.fork_imf
    )

    cfg.total_imf = int(
        args.total_imf
    )

    if (
        cfg.total_imf
        <= cfg.fork_imf
    ):
        raise ValueError(
            "total_imf must be greater than fork_imf."
        )

    
    if args.inner_steps is not None:
        cfg.inner_steps = int(
            args.inner_steps
        )

    if args.eval_mc is not None:
        cfg.eval_mc = int(
            args.eval_mc
        )

    if (
        args.eval_sens_mc
        is not None
    ):
        cfg.eval_sens_mc = int(
            args.eval_sens_mc
        )

    sens_steps = (
        int(args.sens_steps)
        if args.sens_steps
        is not None
        else int(
            cfg.num_steps
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
        run_dir
        / "tangent_sbm.log"
    )

    log("=" * 76)
    log(
        "NONLINEAR CONDITIONAL-GAUSSIAN "
        "TANGENT-SBM"
    )
    log("=" * 76)

    log(
        "Device:",
        device,
    )

    log(
        "Data dir:",
        data_dir.resolve(),
    )

    log(
        "Fork checkpoint:",
        fork_checkpoint,
    )

    log(
        "Run dir:",
        run_dir.resolve(),
    )

    tangent_config = {
        "lambda_sens":
            float(
                args.lambda_sens
            ),
        "sens_every":
            int(
                args.sens_every
            ),
        "sens_batch_size":
            int(
                args.sens_batch_size
            ),
        "sens_steps":
            int(
                sens_steps
            ),
        "sens_seed":
            int(
                args.seed
                + 900001
            ),
        "anchor_fraction":
            float(
                args.anchor_fraction
            ),
        "response_fraction":
            float(
                args.response_fraction
            ),
        "target_mode":
            str(
                args.target_mode
            ),
        "response_collocation":
            str(
                Path(
                    args.response_collocation
                )
            ),
    }

    log(
        "Baseline config from checkpoint:"
    )

    log(
        json.dumps(
            asdict(cfg),
            indent=2,
        )
    )

    log(
        "Tangent config:"
    )

    log(
        json.dumps(
            tangent_config,
            indent=2,
        )
    )

    (
        datasets,
        metadata,
        state_dim,
        intervention_dim,
        sigma_eps_true,
    ) = base.load_dataset(
        data_dir
    )

    response_collocation_path = Path(
        args.response_collocation
    )

    response_data = (
        load_response_collocation(
            path=response_collocation_path,
            state_dim=state_dim,
            intervention_dim=
                intervention_dim,
        )
    )

    log("")
    log(
        "Loaded response-only collocation supervision:",
        response_collocation_path.resolve(),
    )
    log(
        "Response-only N:",
        response_data["x0"].shape[0],
        "| u range:",
        (
            float(
                response_data["u"].min()
            ),
            float(
                response_data["u"].max()
            ),
        ),
    )
    log(
        "Endpoint labels in response supervision:",
        False,
    )

    
    
    
    response_data = (
        _deterministic_response_subset(
            data=response_data,
            fraction=
                args.response_fraction,
            seed=
                args.seed
                + 610001,
        )
    )

    response_data = (
        _apply_target_mode(
            data=response_data,
            mode=
                args.target_mode,
            seed=
                args.seed
                + 620001,
        )
    )

    log(
        "Ablation response fraction:",
        args.response_fraction,
        "| retained N:",
        response_data["x0"].shape[0],
    )

    log(
        "Ablation target mode:",
        args.target_mode,
    )

    if int(
        checkpoint["state_dim"]
    ) != state_dim:
        raise RuntimeError(
            "State dimension mismatch between checkpoint and data."
        )

    if int(
        checkpoint[
            "intervention_dim"
        ]
    ) != intervention_dim:
        raise RuntimeError(
            "Intervention dimension mismatch."
        )

    model = TangentDSBM(
        cfg=cfg,
        state_dim=state_dim,
        intervention_dim=
            intervention_dim,
        device=device,
        lambda_sens=
            args.lambda_sens,
        sens_every=
            args.sens_every,
        sens_batch_size=
            args.sens_batch_size,
        sens_steps=
            sens_steps,
        sens_seed=
            tangent_config[
                "sens_seed"
            ],
        anchor_fraction=
            args.anchor_fraction,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    log("")
    log(
        "Loaded conditional model state from IMF",
        args.fork_imf,
    )

    
    
    

    restore_rng_state(
        checkpoint[
            "rng_state"
        ]
    )

    train_data = (
        datasets["train"]
    )

    
    
    
    anchor_response_data = (
        _apply_target_mode(
            data={
                "x0":
                    train_data["x0"],
                "u":
                    train_data["u"],
                "J_star":
                    train_data["J_star"],
            },
            mode=
                args.target_mode,
            seed=
                args.seed
                + 630001,
        )
    )

    
    
    
    
    if not args.no_rng_replay:
        log("")
        log(
            "Replaying baseline IMF-fork convergence "
            "diagnostic to align global RNG..."
        )

        model.net_f.eval()
        model.net_b.eval()

        replay = (
            base.convergence_metrics(
                model=model,
                train_data=train_data,
                sigma_eps_true=
                    sigma_eps_true,
                metadata=metadata,
                cfg=cfg,
                device=device,
            )
        )

        log(
            "Fork replay complete | "
            f"train mean RMSE="
            f"{replay['mean_rmse']:.6f} | "
            f"train J rel err="
            f"{replay['jacobian_rel_error']:.6f}"
        )

    
    
    
    if not args.skip_gradient_sanity_check:
        log("")
        log(
            "Running tangent gradient-flow sanity check..."
        )

        grad_check = (
            mixed_gradient_flow_sanity_check(
                model=model,
                anchor_data=
                    anchor_response_data,
                collocation_data=
                    response_data,
                max_batch_size=32,
            )
        )

        log(
            "Gradient sanity PASSED | "
            f"sens_loss={grad_check['sensitivity_loss']:.6f} | "
            f"net_f grad norm={grad_check['forward_grad_norm']:.6e} | "
            f"net_b grad norm={grad_check['backward_grad_norm']:.6e} | "
            f"anchors={grad_check['n_anchor']} | "
            f"collocation={grad_check['n_collocation']}"
        )

    
    
    

    history = []
    start = time.time()

    for imf in range(
        cfg.fork_imf + 1,
        cfg.total_imf + 1,
    ):
        log("")
        log("=" * 76)
        log(
            f"IMF {imf}/"
            f"{cfg.total_imf} "
            "- BACKWARD "
            "(ordinary conditional DSBM)"
        )
        log("=" * 76)

        b_stats = (
            model.train_pass_tangent(
                train_data,
                fb="b",
                anchor_sensitivity_data=None,
                collocation_sensitivity_data=None,
            )
        )

        log(
            "Backward pass summary:",
            b_stats,
        )

        log("")
        log("=" * 76)
        log(
            f"IMF {imf}/"
            f"{cfg.total_imf} "
            "- FORWARD "
            "(bridge + tangent response)"
        )
        log("=" * 76)

        f_stats = (
            model.train_pass_tangent(
                train_data,
                fb="f",
                anchor_sensitivity_data=
                    anchor_response_data,
                collocation_sensitivity_data=
                    response_data,
            )
        )

        log(
            "Forward pass summary:",
            f_stats,
        )

        checkpoint_path = (
            run_dir
            / f"imf_{imf}.pt"
        )

        save_tangent_checkpoint(
            path=checkpoint_path,
            model=model,
            cfg=cfg,
            imf=imf,
            state_dim=state_dim,
            intervention_dim=
                intervention_dim,
            tangent_config=
                tangent_config,
        )

        log(
            "Saved Tangent-SBM checkpoint:",
            checkpoint_path,
        )

        
        model.net_f.eval()
        model.net_b.eval()

        conv = (
            base.convergence_metrics(
                model=model,
                train_data=train_data,
                sigma_eps_true=
                    sigma_eps_true,
                metadata=metadata,
                cfg=cfg,
                device=device,
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
                "forward_sensitivity_loss_last100":
                    f_stats[
                        "sensitivity_loss_last100"
                    ],
                "forward_total_loss_last100":
                    f_stats[
                        "total_loss_last100"
                    ],
                "num_sensitivity_updates":
                    f_stats[
                        "num_sensitivity_updates"
                    ],
                "metrics":
                    conv,
            }
        )

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
            m = row[
                "metrics"
            ]

            rows.append(
                {
                    "imf":
                        row["imf"],
                    "backward_bridge_loss_last100":
                        row[
                            "backward_bridge_loss_last100"
                        ],
                    "forward_bridge_loss_last100":
                        row[
                            "forward_bridge_loss_last100"
                        ],
                    "forward_sensitivity_loss_last100":
                        row[
                            "forward_sensitivity_loss_last100"
                        ],
                    "forward_total_loss_last100":
                        row[
                            "forward_total_loss_last100"
                        ],
                    "num_sensitivity_updates":
                        row[
                            "num_sensitivity_updates"
                        ],
                    "train_mean_rmse":
                        m["mean_rmse"],
                    "train_cov_rel_error":
                        m["cov_rel_error"],
                    "train_jacobian_rel_error":
                        m[
                            "jacobian_rel_error"
                        ],
                    "train_jacobian_rmse":
                        m[
                            "jacobian_rmse"
                        ],
                    "train_finite_response_rmse":
                        m[
                            "finite_response_rmse"
                        ],
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
            writer.writerows(
                rows
            )

        log(
            f"IMF {imf} convergence",
            "| train mean RMSE =",
            f"{conv['mean_rmse']:.6f}",
            "| train J rel err =",
            f"{conv['jacobian_rel_error']:.6f}",
            "| finite-response RMSE =",
            f"{conv['finite_response_rmse']:.6f}",
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
        results[split] = (
            base.evaluate_split(
                name=split,
                model=model,
                data=
                    datasets[split],
                sigma_eps_true=
                    sigma_eps_true,
                metadata=metadata,
                cfg=cfg,
                device=device,
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
            "tangent_config":
                tangent_config,
            "state_dim":
                state_dim,
            "intervention_dim":
                intervention_dim,
            "metadata":
                metadata,
            "fork_checkpoint":
                str(
                    fork_checkpoint
                ),
        },
        final_model,
    )

    payload = {
        "method":
            "tangent_sbm",
        "seed":
            int(
                args.seed
            ),
        "config":
            asdict(cfg),
        "tangent_config":
            tangent_config,
        "train_seconds_after_fork":
            train_seconds,
        "fork_checkpoint":
            str(
                fork_checkpoint
            ),
        "response_only_supervision":
            str(
                response_collocation_path
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

    metric_rows = []

    for split, m in (
        results.items()
    ):
        metric_rows.append(
            {
                "method":
                    "tangent_sbm",
                "seed":
                    args.seed,
                "split":
                    split,
                "mean_rmse":
                    m[
                        "mean_rmse"
                    ],
                "cov_rel_error":
                    m[
                        "cov_rel_error"
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
                "train_seconds_after_fork":
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
                metric_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            metric_rows
        )

    comparison_baseline_root = (
        Path(
            args.comparison_baseline_root
        )
        if args.comparison_baseline_root
        is not None
        else Path(
            args.baseline_run_root
        )
    )

    baseline_metrics_path = (
        comparison_baseline_root
        / f"seed_{args.seed}"
        / "metrics.json"
    )

    maybe_write_comparison(
        baseline_metrics_path=
            baseline_metrics_path,
        tangent_results=
            results,
        run_dir=run_dir,
        seed=args.seed,
    )

    log("")
    log("=" * 76)
    log(
        "TANGENT-SBM FINAL SUMMARY"
    )
    log("=" * 76)

    for split, m in (
        results.items()
    ):
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
        "Metrics:",
        run_dir
        / "metrics.json",
    )

    log(
        "Matched comparison baseline:",
        baseline_metrics_path,
    )

    log(
        "Comparison vs conditional:",
        run_dir
        / "comparison_vs_conditional.csv",
    )


if __name__ == "__main__":
    main()
