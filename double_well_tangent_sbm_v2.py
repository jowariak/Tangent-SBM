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

import double_well_conditional_dsbm as base


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


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_response_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing response file: {path}")

    obj = base.safe_torch_load(path, map_location="cpu")

    if not bool(obj.get("response_only", False)):
        raise RuntimeError(f"{path} is not response-only.")

    if bool(obj.get("endpoint_labels_included", True)):
        raise RuntimeError(f"{path} unexpectedly contains endpoint labels.")

    return {
        "x0": obj["x0"].float(),
        "u": obj["u"].float(),
        "J_star": obj["J_star_mean"].float(),
    }


class TangentDoubleWellDSBM(base.ConditionalDSBM):
    def __init__(
        self,
        cfg,
        state_dim,
        intervention_dim,
        device,
        *,
        lambda_sens,
        sens_every,
        sens_batch_size,
        sens_steps,
        anchor_fraction,
        sens_seed,
    ):
        super().__init__(
            cfg=cfg,
            state_dim=state_dim,
            intervention_dim=intervention_dim,
            device=device,
        )

        self.lambda_sens = float(lambda_sens)
        self.sens_every = int(sens_every)
        self.sens_batch_size = int(sens_batch_size)
        self.sens_steps = int(sens_steps)
        self.anchor_fraction = float(anchor_fraction)

        if not 0.0 <= self.anchor_fraction <= 1.0:
            raise ValueError("anchor_fraction must be in [0,1].")

        self.sens_generator = torch.Generator(
            device="cpu"
        ).manual_seed(int(sens_seed))

    def _idx(self, n, bsz):
        return torch.randint(
            0,
            n,
            (bsz,),
            generator=self.sens_generator,
            device="cpu",
        )

    def _sens_noise_bank(self, bsz, dtype):
        return [
            torch.randn(
                bsz,
                self.state_dim,
                generator=self.sens_generator,
                dtype=dtype,
                device="cpu",
            ).to(self.device)
            for _ in range(self.sens_steps)
        ]

    def _response_batch(self, source, bsz):
        if bsz <= 0:
            return None

        idx = self._idx(
            int(source["x0"].shape[0]),
            bsz,
        )

        return {
            "x0": source["x0"][idx].to(self.device),
            "u": source["u"][idx].to(self.device),
            "J_star": source["J_star"][idx].to(self.device),
        }

    def tangent_rollout_train(self, x0, u, noise_bank):
        
        if self.intervention_dim != 1:
            raise NotImplementedError("This benchmark uses scalar u.")

        dt = 1.0 / float(self.sens_steps)

        x = x0
        R = torch.zeros_like(x)
        v = torch.ones_like(u)

        for k in range(self.sens_steps):
            t = torch.full(
                (x.shape[0], 1),
                k / float(self.sens_steps),
                device=x.device,
                dtype=x.dtype,
            )

            def drift_fn(x_in, u_in):
                return self.net_f(
                    x_in,
                    u_in,
                    t,
                )

            drift, tangent_drift = torch.autograd.functional.jvp(
                drift_fn,
                (x, u),
                (R, v),
                create_graph=True,
                strict=False,
            )

            x = (
                x
                + dt * drift
                + self.cfg.reference_sigma
                * math.sqrt(dt)
                * noise_bank[k]
            )

            R = R + dt * tangent_drift

        return R.unsqueeze(-1)

    def expected_response_loss(
        self,
        anchor_response,
        collocation_response,
    ):
        total = self.sens_batch_size

        n_anchor = int(
            round(
                total
                * self.anchor_fraction
            )
        )

        n_colloc = total - n_anchor

        pieces = []

        a = self._response_batch(
            anchor_response,
            n_anchor,
        )
        if a is not None:
            pieces.append(a)

        c = self._response_batch(
            collocation_response,
            n_colloc,
        )
        if c is not None:
            pieces.append(c)

        x0 = torch.cat(
            [p["x0"] for p in pieces],
            dim=0,
        )
        u = torch.cat(
            [p["u"] for p in pieces],
            dim=0,
        )
        J_star = torch.cat(
            [p["J_star"] for p in pieces],
            dim=0,
        )

        perm = torch.randperm(
            x0.shape[0],
            generator=self.sens_generator,
            device="cpu",
        ).to(self.device)

        x0 = x0[perm]
        u = u[perm]
        J_star = J_star[perm]

        
        noise_1 = self._sens_noise_bank(
            x0.shape[0],
            x0.dtype,
        )
        noise_2 = self._sens_noise_bank(
            x0.shape[0],
            x0.dtype,
        )

        J1 = self.tangent_rollout_train(
            x0,
            u,
            noise_1,
        )
        J2 = self.tangent_rollout_train(
            x0,
            u,
            noise_2,
        )

        loss = (
            (J1 - J_star)
            * (J2 - J_star)
        ).mean()

        return loss, {
            "n_anchor": n_anchor,
            "n_collocation": n_colloc,
        }

    def train_pass_tangent(
        self,
        data,
        fb,
        anchor_response,
        collocation_response,
    ):
        z0, z1, u = self.regenerate_coupling(data)

        net = self.nets[fb]
        net.train()

        opt = torch.optim.AdamW(
            net.parameters(),
            lr=self.cfg.lr,
            weight_decay=1e-5,
        )

        n = z0.shape[0]

        recent_bridge = []
        recent_sens = []
        recent_total = []

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

            bz0 = z0[idx]
            bz1 = z1[idx]
            bu = u[idx]

            zt, bu, t, target = self.get_train_tuple(
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
                ((pred - target) ** 2)
                .sum(dim=1)
                .mean()
            )

            sens_loss = None

            if (
                fb == "f"
                and self.lambda_sens > 0.0
                and step % self.sens_every == 0
            ):
                sens_loss, _ = self.expected_response_loss(
                    anchor_response,
                    collocation_response,
                )

                total_loss = (
                    bridge_loss
                    + self.lambda_sens
                    * sens_loss
                )
            else:
                total_loss = bridge_loss

            opt.zero_grad(
                set_to_none=True
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                self.cfg.grad_clip,
            )

            opt.step()

            recent_bridge.append(
                float(
                    bridge_loss
                    .detach()
                    .cpu()
                )
            )

            recent_total.append(
                float(
                    total_loss
                    .detach()
                    .cpu()
                )
            )

            if sens_loss is not None:
                recent_sens.append(
                    float(
                        sens_loss
                        .detach()
                        .cpu()
                    )
                )

            recent_bridge = recent_bridge[-100:]
            recent_total = recent_total[-100:]
            recent_sens = recent_sens[-100:]

            report_every = max(
                100,
                self.cfg.inner_steps // 5,
            )

            if (
                step == 1
                or step % report_every == 0
                or step == self.cfg.inner_steps
            ):
                sens_text = (
                    f"{np.mean(recent_sens):.6f}"
                    if recent_sens
                    else "n/a"
                )

                log(
                    f"{fb} step {step:5d}/{self.cfg.inner_steps}",
                    f"bridge={float(bridge_loss.detach().cpu()):.6f}",
                    f"expected_response={sens_text}",
                    f"total={float(total_loss.detach().cpu()):.6f}",
                )

        self.prev_fb = fb

        return {
            "bridge_loss_last100":
                float(np.mean(recent_bridge)),
            "sensitivity_loss_last100":
                (
                    float(np.mean(recent_sens))
                    if recent_sens
                    else None
                ),
            "total_loss_last100":
                float(np.mean(recent_total)),
        }


def gradient_sanity_check(
    model,
    anchor_response,
    collocation_response,
):
    old_bsz = model.sens_batch_size
    rng_state = model.sens_generator.get_state()

    model.sens_batch_size = min(
        32,
        old_bsz,
    )

    model.net_f.zero_grad(
        set_to_none=True
    )
    model.net_b.zero_grad(
        set_to_none=True
    )

    try:
        loss, mix = model.expected_response_loss(
            anchor_response,
            collocation_response,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Expected-response sanity loss is non-finite."
            )

        loss.backward()

        f_sq = 0.0
        for p in model.net_f.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise RuntimeError(
                        "Non-finite gradient on net_f."
                    )

                f_sq += float(
                    (p.grad.detach() ** 2)
                    .sum()
                    .cpu()
                )

        b_sq = 0.0
        for p in model.net_b.parameters():
            if p.grad is not None:
                b_sq += float(
                    (p.grad.detach() ** 2)
                    .sum()
                    .cpu()
                )

        f_norm = f_sq ** 0.5
        b_norm = b_sq ** 0.5

        if f_norm <= 0.0:
            raise RuntimeError(
                "Expected-response loss gives zero net_f gradient."
            )

        if b_norm > 0.0:
            raise RuntimeError(
                "Forward response loss unexpectedly touched net_b."
            )

        return {
            "loss": float(
                loss.detach().cpu()
            ),
            "forward_grad_norm": f_norm,
            "backward_grad_norm": b_norm,
            **mix,
        }

    finally:
        model.net_f.zero_grad(
            set_to_none=True
        )
        model.net_b.zero_grad(
            set_to_none=True
        )
        model.sens_batch_size = old_bsz
        model.sens_generator.set_state(
            rng_state
        )


def save_checkpoint(
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
            "model": model.state_dict(),
            "config": asdict(cfg),
            "tangent_config": tangent_config,
            "imf": imf,
            "state_dim": state_dim,
            "intervention_dim": intervention_dim,
            "rng_state": base.capture_rng_state(),
        },
        path,
    )


def write_comparison(
    baseline_metrics,
    tangent_results,
    out_path,
    seed,
):
    if not baseline_metrics.exists():
        return

    with open(
        baseline_metrics,
        "r",
    ) as f:
        bm = json.load(f)["metrics"]

    rows = []

    for split, tm in tangent_results.items():
        b = bm[split]

        rows.append(
            {
                "seed": seed,
                "split": split,

                "conditional_mean_rmse":
                    b["mean_rmse"],
                "tangent_mean_rmse":
                    tm["mean_rmse"],

                "conditional_right_well_prob_rmse":
                    b["right_well_prob_rmse"],
                "tangent_right_well_prob_rmse":
                    tm["right_well_prob_rmse"],

                "conditional_j_rel_error":
                    b["jacobian_rel_error"],
                "tangent_j_rel_error":
                    tm["jacobian_rel_error"],

                "conditional_finite_response_rmse":
                    b["finite_response_rmse"],
                "tangent_finite_response_rmse":
                    tm["finite_response_rmse"],
            }
        )

    with open(
        out_path,
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        default="runs/double_well_data",
    )
    p.add_argument(
        "--baseline-run-root",
        default="runs/double_well_conditional",
    )
    p.add_argument(
        "--run-root",
        default="runs/double_well_tangent",
    )
    p.add_argument(
        "--anchor-response",
        default="runs/double_well_data/anchor_response.pt",
    )
    p.add_argument(
        "--response-collocation",
        default="runs/double_well_data/response_collocation.pt",
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
        default=7,
    )
    p.add_argument(
        "--lambda-sens",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--sens-every",
        type=int,
        default=1,
    )
    p.add_argument(
        "--sens-batch-size",
        type=int,
        default=256,
    )
    p.add_argument(
        "--sens-steps",
        type=int,
        default=None,
    )
    p.add_argument(
        "--anchor-fraction",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--device",
        default=None,
    )
    p.add_argument(
        "--skip-gradient-sanity-check",
        action="store_true",
    )
    p.add_argument(
        "--no-rng-replay",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
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
        Path(args.baseline_run_root)
        / f"seed_{args.seed}"
    )

    fork = (
        baseline_seed_dir
        / f"imf_{args.fork_imf}.pt"
    )

    if not fork.exists():
        raise FileNotFoundError(
            f"Missing conditional fork checkpoint: {fork}"
        )

    ckpt = base.safe_torch_load(
        fork,
        map_location="cpu",
    )

    cfg = base.Config(
        **ckpt["config"]
    )
    cfg.fork_imf = args.fork_imf
    cfg.total_imf = args.total_imf

    (
        datasets,
        metadata,
        state_dim,
        intervention_dim,
    ) = base.load_dataset(
        data_dir
    )

    cfg.finite_delta = float(
        metadata.get(
            "finite_delta",
            cfg.finite_delta,
        )
    )

    sens_steps = (
        int(args.sens_steps)
        if args.sens_steps is not None
        else int(cfg.num_steps)
    )

    anchor_response = load_response_file(
        args.anchor_response
    )

    collocation_response = load_response_file(
        args.response_collocation
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

    tangent_cfg = {
        "response_semantics":
            "expected conditional path response",
        "estimator":
            "two-independent-rollout cross estimator",
        "lambda_sens":
            args.lambda_sens,
        "sens_every":
            args.sens_every,
        "sens_batch_size":
            args.sens_batch_size,
        "sens_steps":
            sens_steps,
        "anchor_fraction":
            args.anchor_fraction,
        "sens_seed":
            args.seed + 910001,
    }

    log("=" * 80)
    log("DOUBLE-WELL FORWARD TANGENT-SBM")
    log("=" * 80)
    log("Fork:", fork)
    log("Tangent config:")
    log(
        json.dumps(
            tangent_cfg,
            indent=2,
        )
    )

    model = TangentDoubleWellDSBM(
        cfg=cfg,
        state_dim=state_dim,
        intervention_dim=intervention_dim,
        device=device,
        lambda_sens=args.lambda_sens,
        sens_every=args.sens_every,
        sens_batch_size=args.sens_batch_size,
        sens_steps=sens_steps,
        anchor_fraction=args.anchor_fraction,
        sens_seed=tangent_cfg["sens_seed"],
    )

    model.load_state_dict(
        ckpt["model"]
    )

    restore_rng_state(
        ckpt["rng_state"]
    )

    train_data = datasets["train"]

    if not args.no_rng_replay:
        log(
            "Replaying IMF-3 convergence diagnostic for RNG alignment..."
        )
        base.convergence_metrics(
            model,
            train_data,
            cfg,
            device,
        )

    if not args.skip_gradient_sanity_check:
        check = gradient_sanity_check(
            model,
            anchor_response,
            collocation_response,
        )

        log(
            "Gradient sanity PASSED | "
            f"loss={check['loss']:.6f} | "
            f"net_f={check['forward_grad_norm']:.6e} | "
            f"net_b={check['backward_grad_norm']:.6e} | "
            f"anchor={check['n_anchor']} | "
            f"collocation={check['n_collocation']}"
        )

    history = []
    start = time.time()

    for imf in range(
        cfg.fork_imf + 1,
        cfg.total_imf + 1,
    ):
        log("")
        log("=" * 80)
        log(
            f"IMF {imf}/{cfg.total_imf} - BACKWARD"
        )
        log("=" * 80)

        b = model.train_pass_tangent(
            train_data,
            "b",
            anchor_response,
            collocation_response,
        )

        log("")
        log("=" * 80)
        log(
            f"IMF {imf}/{cfg.total_imf} - FORWARD + EXPECTED RESPONSE"
        )
        log("=" * 80)

        f = model.train_pass_tangent(
            train_data,
            "f",
            anchor_response,
            collocation_response,
        )

        save_checkpoint(
            run_dir / f"imf_{imf}.pt",
            model,
            cfg,
            imf,
            state_dim,
            intervention_dim,
            tangent_cfg,
        )

        conv = base.convergence_metrics(
            model,
            train_data,
            cfg,
            device,
        )

        history.append(
            {
                "imf": imf,
                "backward_bridge_loss_last100":
                    b["bridge_loss_last100"],
                "forward_bridge_loss_last100":
                    f["bridge_loss_last100"],
                "forward_expected_response_loss_last100":
                    f["sensitivity_loss_last100"],
                "train_empirical_endpoint_rmse":
                    conv["train_empirical_endpoint_rmse"],
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
        time.time()
        - start
    )

    model.net_f.eval()
    model.net_b.eval()

    results = {
        split:
            base.evaluate_split(
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

    final_model = (
        run_dir
        / "final_model.pt"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "tangent_config": tangent_cfg,
            "state_dim": state_dim,
            "intervention_dim": intervention_dim,
            "metadata": metadata,
            "fork_checkpoint": str(fork),
        },
        final_model,
    )

    payload = {
        "method": "tangent_sbm",
        "benchmark": "double_well",
        "seed": args.seed,
        "config": asdict(cfg),
        "tangent_config": tangent_cfg,
        "fork_checkpoint": str(fork),
        "train_seconds_after_fork": train_seconds,
        "metrics": results,
        "final_checkpoint": str(final_model),
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
                "method": "tangent_sbm",
                "seed": args.seed,
                "split": split,
                "mean_rmse": m["mean_rmse"],
                "right_well_prob_rmse":
                    m["right_well_prob_rmse"],
                "jacobian_rel_error":
                    m["jacobian_rel_error"],
                "jacobian_rmse":
                    m["jacobian_rmse"],
                "finite_response_rmse":
                    m["finite_response_rmse"],
                "train_seconds_after_fork":
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

    write_comparison(
        baseline_seed_dir / "metrics.json",
        results,
        run_dir / "comparison_vs_conditional.csv",
        args.seed,
    )

    log("")
    log("=" * 80)
    log("DOUBLE-WELL TANGENT-SBM FINAL SUMMARY")
    log("=" * 80)

    for split, m in results.items():
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


if __name__ == "__main__":
    main()
