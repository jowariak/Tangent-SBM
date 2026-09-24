#!/usr/bin/env python3
"""
pdebench_reaction_diffusion_dsbm_unconditional.py

Plain DSBM baseline for the official PDEBench 2D diffusion-reaction benchmark.

This uses the same spatial U-Net bridge architecture as the conditional baseline
but REMOVES access to the physical intervention a=[a_Du,a_Dv,a_k].

Implementation detail:
- The network architecture is kept matched.
- All conditioning channels are forced to zero in training and sampling.
- Therefore the model can use X_t and bridge time t, but cannot use PDE parameters.
- Its directional response with respect to a is identically zero by construction,
  so Jv relative error is exactly the zero-response baseline.

Use this script to train/evaluate the no-conditioning DSBM baseline.

Recommended first run:
    python pdebench_reaction_diffusion_dsbm_unconditional.py \
      --data-dir runs/pdebench_reaction_diffusion_official \
      --run-root runs/pdebench_rd_dsbm \
      --seed 32 \
      --total-imf 3
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

import pdebench_reaction_diffusion_conditional_dsbm as base


def zero_condition(a):
    return torch.zeros_like(a)


class UnconditionalFieldDSBM(base.ConditionalFieldDSBM):
    """
    Same network capacity as ConditionalFieldDSBM, but the intervention
    channels are always zero. Thus the bridge has no access to PDE parameters.
    """

    def get_train_tuple(self, z0, z1, a, fb):
        zt, _, t, target = super().get_train_tuple(
            z0,
            z1,
            zero_condition(a),
            fb,
        )
        return zt, zero_condition(a), t, target

    @torch.no_grad()
    def sample_sde(
        self,
        xstart,
        a,
        fb="f",
        noise_bank=None,
    ):
        # Match the actual baseline API exactly:
        # sample_sde(xstart, a, fb="f", noise_bank=None)
        return super().sample_sde(
            xstart,
            zero_condition(a),
            fb=fb,
            noise_bank=noise_bank,
        )

    def tangent_direction_rollout(
        self,
        x0,
        a,
        direction,
        noise_bank=None,
    ):
        # Model is independent of a by construction.
        return torch.zeros_like(x0)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        default="runs/pdebench_reaction_diffusion_official",
    )

    p.add_argument(
        "--run-root",
        default="runs/pdebench_rd_dsbm",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=32,
    )

    p.add_argument(
        "--total-imf",
        type=int,
        default=3,
    )

    p.add_argument(
        "--inner-steps",
        type=int,
        default=800,
    )

    p.add_argument(
        "--num-steps",
        type=int,
        default=20,
    )

    p.add_argument(
        "--reference-sigma",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--base-channels",
        type=int,
        default=32,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    p.add_argument(
        "--eval-mc",
        type=int,
        default=8,
    )

    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--device",
        default=None,
    )

    return p.parse_args()


def main():
    args = parse_args()

    cfg = base.Config(
        seed=args.seed,
        num_steps=args.num_steps,
        reference_sigma=args.reference_sigma,
        base_channels=args.base_channels,
        total_imf=args.total_imf,
        fork_imf=1,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_mc=args.eval_mc,
        eval_sens_mc=1,
        eval_batch_size=args.eval_batch_size,
    )

    device = torch.device(
        args.device
        if args.device
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    base.set_seed(cfg.seed)

    (
        endpoint,
        response,
        metadata,
        state_shape,
        intervention_dim,
    ) = base.load_dataset(
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
        / f"seed_{cfg.seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base.setup_logging(
        run_dir
        / "dsbm_unconditional.log"
    )

    base.log("=" * 80)
    base.log("PDEBENCH REACTION-DIFFUSION PLAIN DSBM")
    base.log("=" * 80)
    base.log("Device:", device)
    base.log("State shape:", state_shape)
    base.log("Physical intervention hidden from model.")
    base.log(json.dumps(asdict(cfg), indent=2))

    model = UnconditionalFieldDSBM(
        cfg,
        state_shape,
        intervention_dim,
        device,
    )

    history = []
    start = time.time()

    for imf in range(
        1,
        cfg.total_imf + 1,
    ):
        base.log("")
        base.log(
            f"IMF {imf}/{cfg.total_imf} BACKWARD"
        )

        b = model.train_pass(
            endpoint["train"],
            "b",
        )

        base.log(
            f"IMF {imf}/{cfg.total_imf} FORWARD"
        )

        f = model.train_pass(
            endpoint["train"],
            "f",
        )

        # Evaluate train endpoint fidelity after each IMF.
        conv = base.convergence(
            model,
            endpoint["train"],
            cfg,
            device,
        )

        history.append(
            {
                "imf":
                    imf,
                "backward_bridge_loss_last100":
                    b["bridge_loss_last100"],
                "forward_bridge_loss_last100":
                    f["bridge_loss_last100"],
                "train_endpoint_rmse_norm":
                    conv,
            }
        )

        with open(
            run_dir
            / "convergence.json",
            "w",
        ) as fp:
            json.dump(
                history,
                fp,
                indent=2,
            )

        torch.save(
            {
                "model":
                    model.state_dict(),
                "config":
                    asdict(cfg),
                "imf":
                    imf,
                "state_shape":
                    list(state_shape),
                "intervention_dim":
                    intervention_dim,
            },
            run_dir
            / f"imf_{imf}.pt",
        )

        base.log(
            f"IMF {imf} train endpoint RMSE(norm)=",
            f"{conv:.6f}",
        )

    train_seconds = (
        time.time()
        - start
    )

    # Evaluate only final IMF here. If an earlier IMF is better, rerun with
    # --total-imf set to that selected checkpoint for the final 3-seed table.
    results = {}

    for split in [
        "test_seen",
        "test_id",
        "test_ood_near",
        "test_ood_far",
    ]:
        # base.evaluate_split calls model.tangent_direction_rollout;
        # ours returns exactly zero sensitivity.
        results[split] = base.evaluate_split(
            split,
            model,
            endpoint[split],
            response[split],
            cfg,
            device,
            metadata,
        )

    payload = {
        "method":
            "dsbm_unconditional",
        "seed":
            cfg.seed,
        "total_imf":
            cfg.total_imf,
        "train_seconds":
            train_seconds,
        "metrics":
            results,
    }

    with open(
        run_dir
        / "metrics.json",
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
                "method":
                    "dsbm_unconditional",
                "seed":
                    cfg.seed,
                "split":
                    split,
                **m,
            }
        )

    with open(
        run_dir
        / "metrics.csv",
        "w",
        newline="",
    ) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    base.log("")
    base.log("=" * 80)
    base.log("FINAL SUMMARY")
    base.log("=" * 80)

    for split, m in results.items():
        base.log(
            split,
            "| field_rel",
            f"{m['field_rel_l2']:.4f}",
            "| Jv_rel",
            f"{m['directional_j_rel_error']:.4f}",
            "| finite_rel",
            f"{m['finite_response_rel_l2']:.4f}",
        )


if __name__ == "__main__":
    main()
