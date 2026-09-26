#!/usr/bin/env python3


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
