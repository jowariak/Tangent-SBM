#!/usr/bin/env python3
"""
gaussian_response_collocation.py

Generate RESPONSE-ONLY supervision for the nonlinear conditional-Gaussian
benchmark.

This DOES NOT add any endpoint observations.

Existing endpoint training remains:
    u in {-1, 0, +1}

This script creates additional tuples:
    (x0, u, J_star(u))

for u sampled across a broader response-supervision domain, default:
    u ~ Uniform[-1.5, 1.5]

No xT is generated or saved for these collocation samples.

Why this is useful
------------------
Tangent-SBM is designed to use mechanistic response information even where
endpoint observations are unavailable.  This lets us test whether broader
coverage of J*(u) improves extrapolation beyond the sparse endpoint anchors.

Output:
    runs/gaussian_nonlinear_data/response_collocation.pt

Run:
    python gaussian_response_collocation.py \
        --data-dir runs/gaussian_nonlinear_data \
        --n-collocation 12000 \
        --u-min -1.5 \
        --u-max 1.5 \
        --seed 32
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_mvn(mean, cov, n, generator):
    L = torch.linalg.cholesky(cov)
    z = torch.randn(
        n,
        mean.numel(),
        generator=generator,
    )
    return mean[None, :] + z @ L.T


def oracle_jacobian(
    u,
    B1,
    B2,
    B3,
):
    """
    Scalar-u benchmark.
    Returns [N, state_dim, 1].
    """
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


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        type=str,
        default="runs/gaussian_nonlinear_data",
    )

    p.add_argument(
        "--n-collocation",
        type=int,
        default=12000,
    )

    p.add_argument(
        "--u-min",
        type=float,
        default=-1.5,
    )

    p.add_argument(
        "--u-max",
        type=float,
        default=1.5,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=32,
    )

    p.add_argument(
        "--output-name",
        type=str,
        default="response_collocation.pt",
    )

    args = p.parse_args()

    if args.u_max <= args.u_min:
        raise ValueError("u-max must be larger than u-min.")

    set_seed(args.seed)

    data_dir = Path(args.data_dir)

    metadata_path = (
        data_dir
        / "metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. "
            "Run gaussian_nonlinear_data.py first."
        )

    with open(
        metadata_path,
        "r",
    ) as f:
        metadata = json.load(f)

    mu0 = torch.tensor(
        metadata["mu0"],
        dtype=torch.float32,
    )

    sigma0 = torch.tensor(
        metadata["Sigma0"],
        dtype=torch.float32,
    )

    B1 = torch.tensor(
        metadata["B1"],
        dtype=torch.float32,
    )

    B2 = torch.tensor(
        metadata["B2"],
        dtype=torch.float32,
    )

    B3 = torch.tensor(
        metadata["B3"],
        dtype=torch.float32,
    )

    # Positive-definite sanity check.
    torch.linalg.cholesky(
        sigma0
    )

    g = torch.Generator().manual_seed(
        args.seed + 700001
    )

    x0 = sample_mvn(
        mu0,
        sigma0,
        args.n_collocation,
        g,
    )

    u = (
        args.u_min
        + (
            args.u_max
            - args.u_min
        )
        * torch.rand(
            args.n_collocation,
            1,
            generator=g,
        )
    )

    J_star = oracle_jacobian(
        u,
        B1,
        B2,
        B3,
    )

    out = {
        "x0": x0.float(),
        "u": u.float(),
        "J_star": J_star.float(),

        # Deliberately explicit: there are NO endpoint labels here.
        "response_only": True,

        "metadata": {
            "seed": args.seed,
            "n_collocation":
                args.n_collocation,
            "u_min": args.u_min,
            "u_max": args.u_max,
            "target":
                "J_star(u) = B1 + 2 B2 u + pi B3 cos(pi u)",
            "endpoint_labels_included":
                False,
        },
    }

    output_path = (
        data_dir
        / args.output_name
    )

    torch.save(
        out,
        output_path,
    )

    print(
        "Response-only collocation dataset"
    )
    print(
        "---------------------------------"
    )
    print(
        "N:",
        args.n_collocation
    )
    print(
        "u range:",
        (
            float(u.min()),
            float(u.max()),
        )
    )
    print(
        "x0 shape:",
        tuple(x0.shape)
    )
    print(
        "J* shape:",
        tuple(J_star.shape)
    )
    print(
        "Contains endpoint xT:",
        "xT" in out
    )
    print(
        "\nSaved to:"
    )
    print(
        output_path.resolve()
    )


if __name__ == "__main__":
    main()
