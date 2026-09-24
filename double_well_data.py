#!/usr/bin/env python3
"""
double_well_data.py

Controlled nonlinear stochastic double-well benchmark for Tangent-SBM.

Dynamics
--------
We simulate the 1D overdamped stochastic double-well system

    dX_t = [X_t - X_t^3 + eta(u)] dt + sigma dW_t

with a saturating intervention-dependent tilt

    eta(u) = tilt_scale * tanh(u).

Equivalently, the potential is

    V(x;u) = 1/4 x^4 - 1/2 x^2 - eta(u) x.

For the default tilt_scale=0.30, |eta(u)| < 0.30, which remains below
the critical tilt 2/(3*sqrt(3)) ~= 0.3849; therefore the benchmark
retains a genuine double-well structure throughout the evaluation domain.

Directional tangent / response
------------------------------
For scalar u and additive u-independent diffusion,

    R_t = dX_t/du

obeys

    dR_t/dt
      = (1 - 3 X_t^2) R_t
        + tilt_scale * sech^2(u),

with R_0 = 0 because X_0 is sampled independently of u.

Unlike the Gaussian benchmark, R_T depends on the stochastic trajectory.
Therefore the scientifically natural target used later by Tangent-SBM is
the EXPECTED path response

    J*(x0,u) = E_W[dX_T/du | x0,u].

This script estimates that target with Monte Carlo simulator rollouts.

Splits
------
Endpoint training:
    train:
        u in {-1, 0, +1}

Evaluation:
    test_seen:
        fresh endpoints at {-1,0,+1}

    test_id:
        u ~ Uniform[-1,1]

    test_ood_near:
        |u| ~ Uniform[1.10,1.30]

    test_ood_far:
        |u| ~ Uniform[1.40,1.70]

Response-only supervision:
    anchor_response.pt:
        expected J* on a subset of endpoint-anchor operating points

    response_collocation.pt:
        expected J* at continuous u ~ Uniform[-1.30,1.30]
        with NO xT endpoint labels.

Thus:
    [-1,1]       endpoint-training range
    [-1.3,1.3]   response-supervised range
    [1.4,1.7]    fully OOD beyond both.

Stored evaluation truth
-----------------------
For each test operating point (x0,u), simulator Monte Carlo estimates:
    true_conditional_mean
    true_right_well_prob = P(X_T > 0 | x0,u)
    J_star_mean          = E[dX_T/du | x0,u]
    true_finite_response = E[X_T(u+delta)-X_T(u) | x0,u]
                           under common random numbers.

The right-well probability explicitly evaluates the multimodal/bistable
distribution rather than only its mean.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Simulator
# ============================================================

def tilt(u, tilt_scale):
    return tilt_scale * torch.tanh(u)


def tilt_derivative(u, tilt_scale):
    # sech^2(u) = 1 - tanh^2(u)
    th = torch.tanh(u)
    return tilt_scale * (1.0 - th * th)


def simulate_paths(
    x0,
    u,
    noise,
    *,
    total_time,
    sigma,
    tilt_scale,
    return_tangent=True,
):
    """
    Euler-Maruyama simulator.

    Shapes
    ------
    x0:    [..., 1]
    u:     [..., 1]
    noise: [steps, ..., 1]

    Returns
    -------
    xT
    RT if return_tangent
    """
    steps = noise.shape[0]
    dt = float(total_time) / float(steps)
    sqrt_dt = math.sqrt(dt)

    x = x0.clone()

    if return_tangent:
        r = torch.zeros_like(x)

    eta = tilt(
        u,
        tilt_scale,
    )

    deta = tilt_derivative(
        u,
        tilt_scale,
    )

    for k in range(steps):
        # IMPORTANT: update tangent from the same pre-step x used by Euler.
        if return_tangent:
            r = (
                r
                + dt
                * (
                    (1.0 - 3.0 * x * x)
                    * r
                    + deta
                )
            )

        drift = (
            x
            - x ** 3
            + eta
        )

        x = (
            x
            + dt * drift
            + float(sigma)
            * sqrt_dt
            * noise[k]
        )

    if return_tangent:
        return x, r

    return x


def sample_initial(
    n,
    x0_std,
    generator,
    device,
):
    return (
        float(x0_std)
        * torch.randn(
            n,
            1,
            generator=generator,
            device=device,
        )
    )


def sample_anchor_u(
    n,
    generator,
    device,
):
    anchors = torch.tensor(
        [-1.0, 0.0, 1.0],
        dtype=torch.float32,
        device=device,
    )

    idx = torch.randint(
        0,
        3,
        (n,),
        generator=generator,
        device=device,
    )

    return anchors[idx].unsqueeze(1)


def sample_u(
    n,
    split,
    generator,
    device,
):
    if split == "test_id":
        return (
            2.0
            * torch.rand(
                n,
                1,
                generator=generator,
                device=device,
            )
            - 1.0
        )

    if split == "response_collocation":
        return (
            2.6
            * torch.rand(
                n,
                1,
                generator=generator,
                device=device,
            )
            - 1.3
        )

    if split == "test_ood_near":
        lo, hi = 1.10, 1.30
    elif split == "test_ood_far":
        lo, hi = 1.40, 1.70
    else:
        raise ValueError(split)

    magnitude = (
        lo
        + (hi - lo)
        * torch.rand(
            n,
            1,
            generator=generator,
            device=device,
        )
    )

    signs = torch.where(
        torch.rand(
            n,
            1,
            generator=generator,
            device=device,
        )
        < 0.5,
        -torch.ones(
            n,
            1,
            device=device,
        ),
        torch.ones(
            n,
            1,
            device=device,
        ),
    )

    return signs * magnitude


# ============================================================
# Endpoint data
# ============================================================

def generate_endpoint_split(
    *,
    n,
    split,
    seed,
    device,
    steps,
    total_time,
    sigma,
    tilt_scale,
    x0_std,
):
    g = torch.Generator(
        device=device
    ).manual_seed(
        int(seed)
    )

    x0 = sample_initial(
        n,
        x0_std,
        g,
        device,
    )

    if split in {
        "train",
        "test_seen",
    }:
        u = sample_anchor_u(
            n,
            g,
            device,
        )
    else:
        u = sample_u(
            n,
            split,
            g,
            device,
        )

    noise = torch.randn(
        steps,
        n,
        1,
        generator=g,
        device=device,
    )

    xT, RT = simulate_paths(
        x0,
        u,
        noise,
        total_time=
            total_time,
        sigma=sigma,
        tilt_scale=
            tilt_scale,
        return_tangent=True,
    )

    return {
        "x0":
            x0.detach()
            .cpu()
            .float(),
        "u":
            u.detach()
            .cpu()
            .float(),
        "xT":
            xT.detach()
            .cpu()
            .float(),

        # One pathwise tangent for diagnostics only.
        # Main Tangent-SBM supervision later uses MC expected response.
        "pathwise_J":
            RT.detach()
            .cpu()
            .float()
            .unsqueeze(-1),

        "split":
            split,
    }


# ============================================================
# MC oracle statistics
# ============================================================

@torch.no_grad()
def mc_oracle_batch(
    x0,
    u,
    *,
    mc,
    steps,
    total_time,
    sigma,
    tilt_scale,
    finite_delta,
    generator,
):
    """
    Estimate conditional statistics at fixed (x0,u).

    Uses common random numbers for the base and u+delta finite-response
    simulations to reduce Monte Carlo variance.
    """
    b = x0.shape[0]

    # [B, MC, 1]
    x0_rep = (
        x0[:, None, :]
        .expand(
            b,
            mc,
            1,
        )
        .contiguous()
    )

    u_rep = (
        u[:, None, :]
        .expand(
            b,
            mc,
            1,
        )
        .contiguous()
    )

    noise = torch.randn(
        steps,
        b,
        mc,
        1,
        generator=generator,
        device=x0.device,
    )

    x_base, r_base = (
        simulate_paths(
            x0_rep,
            u_rep,
            noise,
            total_time=
                total_time,
            sigma=sigma,
            tilt_scale=
                tilt_scale,
            return_tangent=True,
        )
    )

    u_plus = (
        u_rep
        + float(
            finite_delta
        )
    )

    x_plus = simulate_paths(
        x0_rep,
        u_plus,
        noise,
        total_time=
            total_time,
        sigma=sigma,
        tilt_scale=
            tilt_scale,
        return_tangent=False,
    )

    true_mean = (
        x_base.mean(
            dim=1
        )
    )

    true_right_prob = (
        (x_base > 0.0)
        .float()
        .mean(
            dim=1
        )
    )

    j_mean = (
        r_base.mean(
            dim=1
        )
        .unsqueeze(-1)
    )

    finite_response = (
        (
            x_plus
            - x_base
        )
        .mean(
            dim=1
        )
    )

    return {
        "true_conditional_mean":
            true_mean,
        "true_right_well_prob":
            true_right_prob,
        "J_star_mean":
            j_mean,
        "true_finite_response":
            finite_response,
    }


def attach_mc_truth(
    endpoint_obj,
    *,
    device,
    mc,
    batch_size,
    seed,
    steps,
    total_time,
    sigma,
    tilt_scale,
    finite_delta,
):
    x0_all = (
        endpoint_obj["x0"]
        .to(device)
    )

    u_all = (
        endpoint_obj["u"]
        .to(device)
    )

    n = x0_all.shape[0]

    g = torch.Generator(
        device=device
    ).manual_seed(
        int(seed)
    )

    chunks = {
        "true_conditional_mean":
            [],
        "true_right_well_prob":
            [],
        "J_star_mean":
            [],
        "true_finite_response":
            [],
    }

    for start in range(
        0,
        n,
        batch_size,
    ):
        end = min(
            n,
            start
            + batch_size,
        )

        out = mc_oracle_batch(
            x0_all[start:end],
            u_all[start:end],
            mc=mc,
            steps=steps,
            total_time=
                total_time,
            sigma=sigma,
            tilt_scale=
                tilt_scale,
            finite_delta=
                finite_delta,
            generator=g,
        )

        for key in chunks:
            chunks[key].append(
                out[key]
                .detach()
                .cpu()
                .float()
            )

    for key, vals in chunks.items():
        endpoint_obj[key] = (
            torch.cat(
                vals,
                dim=0,
            )
        )

    return endpoint_obj


# ============================================================
# Response-only data
# ============================================================

def make_anchor_response(
    train_obj,
    *,
    n_response,
    device,
    mc,
    batch_size,
    seed,
    steps,
    total_time,
    sigma,
    tilt_scale,
    finite_delta,
):
    n_train = (
        train_obj["x0"]
        .shape[0]
    )

    g_cpu = (
        torch.Generator(
            device="cpu"
        )
        .manual_seed(
            int(seed)
        )
    )

    perm = torch.randperm(
        n_train,
        generator=g_cpu,
    )

    idx = perm[
        :min(
            n_response,
            n_train,
        )
    ]

    base = {
        "x0":
            train_obj["x0"][
                idx
            ].clone(),
        "u":
            train_obj["u"][
                idx
            ].clone(),
    }

    tmp = {
        "x0":
            base["x0"],
        "u":
            base["u"],
    }

    # Reuse attach helper by supplying only needed tensors.
    x0_all = tmp["x0"].to(
        device
    )

    u_all = tmp["u"].to(
        device
    )

    g = torch.Generator(
        device=device
    ).manual_seed(
        int(seed)
        + 1
    )

    j_chunks = []

    for start in range(
        0,
        x0_all.shape[0],
        batch_size,
    ):
        end = min(
            x0_all.shape[0],
            start
            + batch_size,
        )

        out = mc_oracle_batch(
            x0_all[start:end],
            u_all[start:end],
            mc=mc,
            steps=steps,
            total_time=
                total_time,
            sigma=sigma,
            tilt_scale=
                tilt_scale,
            finite_delta=
                finite_delta,
            generator=g,
        )

        j_chunks.append(
            out["J_star_mean"]
            .cpu()
        )

    return {
        "x0":
            base["x0"]
            .float(),
        "u":
            base["u"]
            .float(),
        "J_star_mean":
            torch.cat(
                j_chunks,
                dim=0,
            )
            .float(),

        "response_only":
            True,
        "endpoint_labels_included":
            False,
        "source":
            "endpoint_anchor_operating_points",
    }


def make_response_collocation(
    *,
    n,
    device,
    mc,
    batch_size,
    seed,
    steps,
    total_time,
    sigma,
    tilt_scale,
    x0_std,
    finite_delta,
):
    g = torch.Generator(
        device=device
    ).manual_seed(
        int(seed)
    )

    x0 = sample_initial(
        n,
        x0_std,
        g,
        device,
    )

    u = sample_u(
        n,
        "response_collocation",
        g,
        device,
    )

    j_chunks = []

    for start in range(
        0,
        n,
        batch_size,
    ):
        end = min(
            n,
            start
            + batch_size,
        )

        out = mc_oracle_batch(
            x0[start:end],
            u[start:end],
            mc=mc,
            steps=steps,
            total_time=
                total_time,
            sigma=sigma,
            tilt_scale=
                tilt_scale,
            finite_delta=
                finite_delta,
            generator=g,
        )

        j_chunks.append(
            out["J_star_mean"]
            .detach()
            .cpu()
            .float()
        )

    return {
        "x0":
            x0.detach()
            .cpu()
            .float(),
        "u":
            u.detach()
            .cpu()
            .float(),
        "J_star_mean":
            torch.cat(
                j_chunks,
                dim=0,
            ),

        "response_only":
            True,
        "endpoint_labels_included":
            False,
        "source":
            "continuous_response_collocation",
    }


# ============================================================
# Finite-difference tangent sanity check
# ============================================================

@torch.no_grad()
def finite_difference_check(
    *,
    device,
    seed,
    n,
    steps,
    total_time,
    sigma,
    tilt_scale,
    x0_std,
    delta,
):
    g = torch.Generator(
        device=device
    ).manual_seed(
        int(seed)
    )

    x0 = sample_initial(
        n,
        x0_std,
        g,
        device,
    )

    u = (
        2.0
        * torch.rand(
            n,
            1,
            generator=g,
            device=device,
        )
        - 1.0
    )

    noise = torch.randn(
        steps,
        n,
        1,
        generator=g,
        device=device,
    )

    _, r = simulate_paths(
        x0,
        u,
        noise,
        total_time=
            total_time,
        sigma=sigma,
        tilt_scale=
            tilt_scale,
        return_tangent=True,
    )

    x_plus = simulate_paths(
        x0,
        u + delta,
        noise,
        total_time=
            total_time,
        sigma=sigma,
        tilt_scale=
            tilt_scale,
        return_tangent=False,
    )

    x_minus = simulate_paths(
        x0,
        u - delta,
        noise,
        total_time=
            total_time,
        sigma=sigma,
        tilt_scale=
            tilt_scale,
        return_tangent=False,
    )

    fd = (
        x_plus
        - x_minus
    ) / (
        2.0
        * delta
    )

    return {
        "max_abs_error":
            float(
                (fd - r)
                .abs()
                .max()
                .cpu()
            ),
        "mean_abs_error":
            float(
                (fd - r)
                .abs()
                .mean()
                .cpu()
            ),
    }


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--out-dir",
        type=str,
        default="runs/double_well_data",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=32,
    )

    p.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    p.add_argument(
        "--n-train",
        type=int,
        default=20000,
    )

    p.add_argument(
        "--n-test",
        type=int,
        default=3000,
    )

    p.add_argument(
        "--n-anchor-response",
        type=int,
        default=6000,
    )

    p.add_argument(
        "--n-response-collocation",
        type=int,
        default=12000,
    )

    p.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    p.add_argument(
        "--total-time",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--sigma",
        type=float,
        default=0.45,
    )

    p.add_argument(
        "--tilt-scale",
        type=float,
        default=0.30,
    )

    p.add_argument(
        "--x0-std",
        type=float,
        default=0.35,
    )

    p.add_argument(
        "--oracle-mc-eval",
        type=int,
        default=32,
    )

    p.add_argument(
        "--oracle-mc-response",
        type=int,
        default=16,
    )

    p.add_argument(
        "--oracle-batch-size",
        type=int,
        default=256,
    )

    p.add_argument(
        "--finite-delta",
        type=float,
        default=0.25,
    )

    args = p.parse_args()

    set_seed(
        args.seed
    )

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = (
            args.device
        )

    device = torch.device(
        device
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Device:",
        device,
    )

    print(
        "Generating endpoint datasets..."
    )

    specs = {
        "train":
            (
                args.n_train,
                args.seed
                + 101,
            ),
        "test_seen":
            (
                args.n_test,
                args.seed
                + 202,
            ),
        "test_id":
            (
                args.n_test,
                args.seed
                + 303,
            ),
        "test_ood_near":
            (
                args.n_test,
                args.seed
                + 404,
            ),
        "test_ood_far":
            (
                args.n_test,
                args.seed
                + 505,
            ),
    }

    objects = {}

    for split, (
        n,
        split_seed,
    ) in specs.items():

        obj = generate_endpoint_split(
            n=n,
            split=split,
            seed=split_seed,
            device=device,
            steps=args.steps,
            total_time=
                args.total_time,
            sigma=args.sigma,
            tilt_scale=
                args.tilt_scale,
            x0_std=
                args.x0_std,
        )

        objects[split] = obj

        print(
            f"{split:16s}",
            "| N =",
            n,
            "| x0",
            tuple(
                obj["x0"]
                .shape
            ),
            "| xT",
            tuple(
                obj["xT"]
                .shape
            ),
        )

    print(
        "\nEstimating evaluation oracle statistics..."
    )

    for split in [
        "test_seen",
        "test_id",
        "test_ood_near",
        "test_ood_far",
    ]:
        print(
            "  oracle:",
            split,
        )

        objects[split] = (
            attach_mc_truth(
                objects[split],
                device=device,
                mc=
                    args.oracle_mc_eval,
                batch_size=
                    args.oracle_batch_size,
                seed=
                    args.seed
                    + {
                        "test_seen":
                            1202,
                        "test_id":
                            1303,
                        "test_ood_near":
                            1404,
                        "test_ood_far":
                            1505,
                    }[
                        split
                    ],
                steps=
                    args.steps,
                total_time=
                    args.total_time,
                sigma=
                    args.sigma,
                tilt_scale=
                    args.tilt_scale,
                finite_delta=
                    args.finite_delta,
            )
        )

    # Save endpoints.
    for split, obj in objects.items():
        torch.save(
            obj,
            out_dir
            / f"{split}.pt",
        )

    print(
        "\nGenerating anchor response supervision..."
    )

    anchor_response = (
        make_anchor_response(
            objects["train"],
            n_response=
                args.n_anchor_response,
            device=device,
            mc=
                args.oracle_mc_response,
            batch_size=
                args.oracle_batch_size,
            seed=
                args.seed
                + 2001,
            steps=
                args.steps,
            total_time=
                args.total_time,
            sigma=
                args.sigma,
            tilt_scale=
                args.tilt_scale,
            finite_delta=
                args.finite_delta,
        )
    )

    torch.save(
        anchor_response,
        out_dir
        / "anchor_response.pt",
    )

    print(
        "anchor_response:",
        tuple(
            anchor_response[
                "J_star_mean"
            ].shape
        ),
    )

    print(
        "\nGenerating continuous response-only collocation..."
    )

    response_collocation = (
        make_response_collocation(
            n=
                args.n_response_collocation,
            device=device,
            mc=
                args.oracle_mc_response,
            batch_size=
                args.oracle_batch_size,
            seed=
                args.seed
                + 3001,
            steps=
                args.steps,
            total_time=
                args.total_time,
            sigma=
                args.sigma,
            tilt_scale=
                args.tilt_scale,
            x0_std=
                args.x0_std,
            finite_delta=
                args.finite_delta,
        )
    )

    torch.save(
        response_collocation,
        out_dir
        / "response_collocation.pt",
    )

    print(
        "response_collocation:",
        tuple(
            response_collocation[
                "J_star_mean"
            ].shape
        ),
        "| u range =",
        (
            float(
                response_collocation[
                    "u"
                ].min()
            ),
            float(
                response_collocation[
                    "u"
                ].max()
            ),
        ),
    )

    print(
        "\nRunning central finite-difference tangent sanity check..."
    )

    fd = finite_difference_check(
        device=device,
        seed=
            args.seed
            + 4001,
        n=512,
        steps=
            args.steps,
        total_time=
            args.total_time,
        sigma=
            args.sigma,
        tilt_scale=
            args.tilt_scale,
        x0_std=
            args.x0_std,
        delta=1e-3,
    )

    print(
        "Max central-FD error vs tangent:",
        f"{fd['max_abs_error']:.6e}",
    )

    print(
        "Mean central-FD error vs tangent:",
        f"{fd['mean_abs_error']:.6e}",
    )

    # Aggregate simulator diagnostics.
    diagnostics = {}

    for split in [
        "test_seen",
        "test_id",
        "test_ood_near",
        "test_ood_far",
    ]:
        obj = objects[
            split
        ]

        diagnostics[
            split
        ] = {
            "mean_xT":
                float(
                    obj[
                        "true_conditional_mean"
                    ]
                    .mean()
                ),
            "mean_right_well_prob":
                float(
                    obj[
                        "true_right_well_prob"
                    ]
                    .mean()
                ),
            "mean_expected_J":
                float(
                    obj[
                        "J_star_mean"
                    ]
                    .mean()
                ),
            "mean_abs_expected_J":
                float(
                    obj[
                        "J_star_mean"
                    ]
                    .abs()
                    .mean()
                ),
        }

    metadata = {
        "seed":
            args.seed,
        "state_dim":
            1,
        "intervention_dim":
            1,

        "sde":
            (
                "dX = [X - X^3 + tilt_scale*tanh(u)] dt "
                "+ sigma dW"
            ),

        "potential":
            (
                "V(x;u)=x^4/4-x^2/2-tilt_scale*tanh(u)*x"
            ),

        "tangent":
            (
                "dR/dt=(1-3X^2)R+tilt_scale*sech^2(u), R0=0"
            ),

        "response_semantics":
            (
                "expected conditional path response "
                "E[dX_T/du | x0,u]"
            ),

        "steps":
            args.steps,
        "total_time":
            args.total_time,
        "dt":
            args.total_time
            / args.steps,
        "sigma":
            args.sigma,
        "tilt_scale":
            args.tilt_scale,
        "x0_std":
            args.x0_std,
        "finite_delta":
            args.finite_delta,

        "train_u_anchors":
            [
                -1.0,
                0.0,
                1.0,
            ],

        "response_collocation_u_support":
            [
                -1.3,
                1.3,
            ],

        "test_id_u_support":
            [
                -1.0,
                1.0,
            ],

        "test_ood_near_abs_u_support":
            [
                1.10,
                1.30,
            ],

        "test_ood_far_abs_u_support":
            [
                1.40,
                1.70,
            ],

        "oracle_mc_eval":
            args.oracle_mc_eval,
        "oracle_mc_response":
            args.oracle_mc_response,

        "finite_difference_sanity":
            fd,

        "diagnostics":
            diagnostics,
    }

    with open(
        out_dir
        / "metadata.json",
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\nSimulator diagnostics:"
    )

    for split, d in diagnostics.items():
        print(
            split,
            "| mean xT =",
            f"{d['mean_xT']:.4f}",
            "| mean P(right) =",
            f"{d['mean_right_well_prob']:.4f}",
            "| mean |J*| =",
            f"{d['mean_abs_expected_J']:.4f}",
        )

    print(
        "\nSaved dataset to:"
    )

    print(
        out_dir.resolve()
    )


if __name__ == "__main__":
    main()
