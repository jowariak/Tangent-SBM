#!/usr/bin/env python3
"""
gaussian_nonlinear_data.py

Nonlinear conditional-Gaussian benchmark for Tangent-SBM.

The state/noise model remains Gaussian and analytically controlled:

    x0  ~ N(mu0, Sigma0)
    eps ~ N(0, Sigma_eps)

    xT = A x0
         + B1 u
         + B2 u^2
         + B3 sin(pi u)
         + eps

Hence

    X_T | x0, u ~ N(mu(x0,u), Sigma_eps)

with exact response

    J*(u) = d mu / du
          = B1 + 2 B2 u + pi B3 cos(pi u)

for the default scalar intervention u.

WHY THIS VERSION
----------------
Training endpoint observations are provided only at the three intervention
anchors u in {-1, 0, +1}.  At all three anchors sin(pi u) = 0, so the B3
term is invisible from endpoint values at the observed conditions.

However, its derivative is NOT invisible:
    d/du [B3 sin(pi u)] = pi B3 cos(pi u).

Thus two conditional models can fit the observed endpoint distributions while
having different intervention responses.  Tangent-SBM receives J*(u) at the
same observed training conditions and can use that response information.

Splits
------
train:
    exact anchors {-1, 0, +1}

test_seen:
    fresh samples at the same anchors

test_id:
    continuous u ~ Uniform[-1, 1]
    (in-range interpolation at mostly unseen intervention values)

test_ood_near:
    |u| ~ Uniform[1.15, 1.50]

test_ood_far:
    |u| ~ Uniform[1.75, 2.25]

Outputs
-------
    runs/gaussian_nonlinear_data/
        train.pt
        test_seen.pt
        test_id.pt
        test_ood_near.pt
        test_ood_far.pt
        metadata.json
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

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Ground-truth system
# ============================================================

def default_system():
    mu0 = torch.tensor(
        [0.0, 0.0],
        dtype=torch.float32,
    )

    sigma0 = torch.tensor(
        [
            [1.00, 0.35],
            [0.35, 0.80],
        ],
        dtype=torch.float32,
    )

    A = torch.tensor(
        [
            [0.80, -0.30],
            [0.25,  0.65],
        ],
        dtype=torch.float32,
    )

    # Smooth "visible" intervention component.
    B1 = torch.tensor(
        [
            [ 0.70],
            [-0.40],
        ],
        dtype=torch.float32,
    )

    # Smooth curvature visible at +/-1.
    B2 = torch.tensor(
        [
            [0.25],
            [0.15],
        ],
        dtype=torch.float32,
    )

    # Crucial hidden-response component:
    # sin(pi*u)=0 at u=-1,0,+1, so endpoint values at the training
    # anchors cannot identify B3.  J*(u), however, contains
    # pi*B3*cos(pi*u), which is nonzero at all three anchors.
    B3 = torch.tensor(
        [
            [ 0.55],
            [-0.45],
        ],
        dtype=torch.float32,
    )

    sigma_eps = torch.tensor(
        [
            [0.25, 0.08],
            [0.08, 0.18],
        ],
        dtype=torch.float32,
    )

    return mu0, sigma0, A, B1, B2, B3, sigma_eps


def oracle_mean(
    x0: torch.Tensor,
    u: torch.Tensor,
    A: torch.Tensor,
    B1: torch.Tensor,
    B2: torch.Tensor,
    B3: torch.Tensor,
) -> torch.Tensor:
    """
    Default benchmark has scalar u with shape [N,1].
    """
    if u.ndim != 2 or u.shape[1] != 1:
        raise ValueError("This benchmark currently expects scalar u with shape [N,1].")

    return (
        x0 @ A.T
        + u @ B1.T
        + (u ** 2) @ B2.T
        + torch.sin(math.pi * u) @ B3.T
    )


def oracle_jacobian(
    u: torch.Tensor,
    B1: torch.Tensor,
    B2: torch.Tensor,
    B3: torch.Tensor,
) -> torch.Tensor:
    """
    Returns samplewise J*(u) with shape [N, state_dim, 1].
    """
    if u.ndim != 2 or u.shape[1] != 1:
        raise ValueError("This benchmark currently expects scalar u with shape [N,1].")

    # [N, state_dim]
    deriv = (
        B1.T
        + 2.0 * u * B2.T
        + math.pi * torch.cos(math.pi * u) * B3.T
    )

    return deriv.unsqueeze(-1)


# ============================================================
# Sampling
# ============================================================

def sample_mvn(
    mean: torch.Tensor,
    cov: torch.Tensor,
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    L = torch.linalg.cholesky(cov)
    z = torch.randn(
        n,
        mean.numel(),
        generator=generator,
    )
    return mean[None, :] + z @ L.T


def sample_anchor_u(
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Balanced exact anchors {-1,0,+1}.
    """
    anchors = torch.tensor(
        [-1.0, 0.0, 1.0],
        dtype=torch.float32,
    )

    idx = torch.randint(
        0,
        len(anchors),
        (n,),
        generator=generator,
    )

    return anchors[idx].unsqueeze(1)


def sample_continuous_u(
    n: int,
    split: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if split == "test_id":
        return (
            2.0
            * torch.rand(
                n,
                1,
                generator=generator,
            )
            - 1.0
        )

    if split == "test_ood_near":
        lo, hi = 1.15, 1.50
    elif split == "test_ood_far":
        lo, hi = 1.75, 2.25
    else:
        raise ValueError(split)

    magnitude = (
        lo
        + (hi - lo)
        * torch.rand(
            n,
            1,
            generator=generator,
        )
    )

    signs = torch.where(
        torch.rand(
            n,
            1,
            generator=generator,
        ) < 0.5,
        -torch.ones(n, 1),
        torch.ones(n, 1),
    )

    return signs * magnitude


def make_split(
    n: int,
    split: str,
    seed: int,
    mu0: torch.Tensor,
    sigma0: torch.Tensor,
    A: torch.Tensor,
    B1: torch.Tensor,
    B2: torch.Tensor,
    B3: torch.Tensor,
    sigma_eps: torch.Tensor,
):
    g = torch.Generator().manual_seed(seed)

    x0 = sample_mvn(
        mu0,
        sigma0,
        n,
        g,
    )

    if split in {"train", "test_seen"}:
        u = sample_anchor_u(
            n,
            g,
        )
    else:
        u = sample_continuous_u(
            n,
            split,
            g,
        )

    eps = sample_mvn(
        torch.zeros(
            A.shape[0],
            dtype=torch.float32,
        ),
        sigma_eps,
        n,
        g,
    )

    true_mean = oracle_mean(
        x0,
        u,
        A,
        B1,
        B2,
        B3,
    )

    J_star = oracle_jacobian(
        u,
        B1,
        B2,
        B3,
    )

    xT = true_mean + eps

    return {
        "x0": x0.float(),
        "u": u.float(),
        "xT": xT.float(),
        "eps": eps.float(),
        "true_conditional_mean": true_mean.float(),

        # IMPORTANT: unlike the original globally-linear benchmark,
        # J* now varies by sample through u.
        # Shape: [N, state_dim, intervention_dim].
        "J_star": J_star.float(),

        "split": split,
    }


# ============================================================
# Sanity checks
# ============================================================

def finite_difference_jacobian_check(
    x0: torch.Tensor,
    u: torch.Tensor,
    A: torch.Tensor,
    B1: torch.Tensor,
    B2: torch.Tensor,
    B3: torch.Tensor,
    delta: float = 1e-3,
):
    mu_plus = oracle_mean(
        x0,
        u + delta,
        A,
        B1,
        B2,
        B3,
    )

    mu_minus = oracle_mean(
        x0,
        u - delta,
        A,
        B1,
        B2,
        B3,
    )

    fd = (
        mu_plus - mu_minus
    ) / (
        2.0 * delta
    )

    exact = oracle_jacobian(
        u,
        B1,
        B2,
        B3,
    ).squeeze(-1)

    return (
        fd - exact
    ).abs().max().item()


def empirical_noise_covariance(
    eps: torch.Tensor,
):
    centered = (
        eps
        - eps.mean(
            dim=0,
            keepdim=True,
        )
    )

    return (
        centered.T
        @ centered
        / eps.shape[0]
    )


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--out_dir",
        type=str,
        default="runs/gaussian_nonlinear_data",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=32,
    )

    p.add_argument(
        "--n_train",
        type=int,
        default=20000,
    )

    p.add_argument(
        "--n_test_seen",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--n_test_id",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--n_test_ood_near",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--n_test_ood_far",
        type=int,
        default=5000,
    )

    args = p.parse_args()

    set_seed(
        args.seed
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        mu0,
        sigma0,
        A,
        B1,
        B2,
        B3,
        sigma_eps,
    ) = default_system()

    # Positive-definiteness checks.
    torch.linalg.cholesky(
        sigma0
    )

    torch.linalg.cholesky(
        sigma_eps
    )

    specs = {
        "train": (
            args.n_train,
            args.seed + 101,
        ),
        "test_seen": (
            args.n_test_seen,
            args.seed + 202,
        ),
        "test_id": (
            args.n_test_id,
            args.seed + 303,
        ),
        "test_ood_near": (
            args.n_test_ood_near,
            args.seed + 404,
        ),
        "test_ood_far": (
            args.n_test_ood_far,
            args.seed + 505,
        ),
    }

    files = {}

    for split, (
        n,
        split_seed,
    ) in specs.items():

        obj = make_split(
            n=n,
            split=split,
            seed=split_seed,
            mu0=mu0,
            sigma0=sigma0,
            A=A,
            B1=B1,
            B2=B2,
            B3=B3,
            sigma_eps=sigma_eps,
        )

        path = (
            out_dir
            / f"{split}.pt"
        )

        torch.save(
            obj,
            path,
        )

        files[split] = str(
            path
        )

        print(
            f"{split:18s} | "
            f"N={n:6d} | "
            f"x0={tuple(obj['x0'].shape)} | "
            f"u={tuple(obj['u'].shape)} | "
            f"xT={tuple(obj['xT'].shape)} | "
            f"J*={tuple(obj['J_star'].shape)}"
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    check = torch.load(
        out_dir
        / "test_id.pt",
        map_location="cpu",
        weights_only=False,
    )

    k = min(
        512,
        check["x0"].shape[0],
    )

    fd_err = finite_difference_jacobian_check(
        x0=check["x0"][:k],
        u=check["u"][:k],
        A=A,
        B1=B1,
        B2=B2,
        B3=B3,
        delta=1e-3,
    )

    emp_cov = empirical_noise_covariance(
        check["eps"]
    )

    # Exact J* at the three observed training conditions.
    anchor_u = torch.tensor(
        [
            [-1.0],
            [ 0.0],
            [ 1.0],
        ],
        dtype=torch.float32,
    )

    anchor_J = oracle_jacobian(
        anchor_u,
        B1,
        B2,
        B3,
    )

    metadata = {
        "seed": args.seed,
        "state_dim": 2,
        "intervention_dim": 1,
        "train_u_anchors": [
            -1.0,
            0.0,
            1.0,
        ],
        "test_id_support": [
            -1.0,
            1.0,
        ],
        "ood_near_abs_u_support": [
            1.15,
            1.50,
        ],
        "ood_far_abs_u_support": [
            1.75,
            2.25,
        ],
        "system": {
            "equation":
                "xT = A x0 + B1 u + B2 u^2 + B3 sin(pi u) + eps",
            "conditional_distribution":
                "X_T | x0,u ~ N(mu(x0,u), Sigma_eps)",
            "exact_response":
                "J*(u) = B1 + 2 B2 u + pi B3 cos(pi u)",
            "identifiability_design":
                "Training endpoints use u in {-1,0,1}. "
                "Since sin(pi u)=0 at all three anchors, the B3 term "
                "is invisible to endpoint values there, while its derivative "
                "pi B3 cos(pi u) remains visible to response supervision.",
        },
        "mu0": mu0.tolist(),
        "Sigma0": sigma0.tolist(),
        "A": A.tolist(),
        "B1": B1.tolist(),
        "B2": B2.tolist(),
        "B3": B3.tolist(),
        "Sigma_eps": sigma_eps.tolist(),
        "anchor_J_star": {
            "-1": anchor_J[0].tolist(),
            "0": anchor_J[1].tolist(),
            "1": anchor_J[2].tolist(),
        },
        "files": files,
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

    print("\nGround-truth matrices")
    print("---------------------")
    print("A =")
    print(A)

    print("\nB1 =")
    print(B1)

    print("\nB2 =")
    print(B2)

    print("\nB3 =")
    print(B3)

    print("\nExact J* at observed training anchors")
    print("-------------------------------------")

    for i, uval in enumerate(
        [-1.0, 0.0, 1.0]
    ):
        print(
            f"u={uval:+.1f}:"
        )
        print(
            anchor_J[i]
        )

    print("\nSanity checks")
    print("-------------")

    print(
        "Max central-FD error against exact samplewise J*: "
        f"{fd_err:.6e}"
    )

    print(
        "Empirical test noise covariance:"
    )

    print(
        emp_cov
    )

    print("\nSaved dataset to:")
    print(
        out_dir.resolve()
    )


if __name__ == "__main__":
    main()
