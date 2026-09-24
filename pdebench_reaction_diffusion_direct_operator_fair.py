#!/usr/bin/env python3
"""
Direct and derivative-informed deterministic operator baselines for the
PDEBench reaction-diffusion intervention benchmark.

This script deliberately reuses the exact ConditionalUNetDrift architecture
and dataset loader from pdebench_reaction_diffusion_conditional_dsbm.py.

Modes
-----
1) Direct conditional operator:
       lambda_sens = 0
       (X0, a) -> XT
       endpoint supervision only

2) Derivative-informed conditional operator:
       lambda_sens > 0
       endpoint loss + directional JVP response supervision during a fixed
       final sensitivity window. By default, 800 final steps with
       sens_every=10 gives 80 response-loss updates, matching the PDEBench
       Tangent-SBM continuation budget.

The response-only supervision files are the same anchor/collocation files used
by the Tangent-SBM PDEBench experiment. No endpoint labels are read from those
files.

Evaluation follows the existing PDEBench benchmark conventions:
- terminal-field relative L2 / RMSE
- directional sensitivity relative error E_J
- finite-response error at metadata["finite_delta"] (default 0.25)

The direct model is deterministic, so no Monte Carlo averaging is needed.
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdebench_reaction_diffusion_conditional_dsbm as base


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def setup_logging(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_response_only(path):
    obj = safe_torch_load(path, map_location="cpu")

    if not bool(obj.get("response_only", False)):
        raise RuntimeError(f"{path}: expected response_only=True")

    if bool(obj.get("endpoint_labels_included", True)):
        raise RuntimeError(f"{path}: endpoint labels unexpectedly included")

    required = ["x0_norm", "a", "direction", "Jv_star_norm"]
    for k in required:
        if k not in obj:
            raise RuntimeError(f"{path}: missing {k}")

    return {
        "x0": obj["x0_norm"].float(),
        "a": obj["a"].float(),
        "direction": obj["direction"].float(),
        "Jv_star": obj["Jv_star_norm"].float(),
    }


def norm_stats(metadata, device):
    mean = torch.tensor(
        metadata["normalization"]["state_mean_per_channel"],
        device=device,
        dtype=torch.float32,
    ).view(1, 2, 1, 1)

    std = torch.tensor(
        metadata["normalization"]["state_std_per_channel"],
        device=device,
        dtype=torch.float32,
    ).view(1, 2, 1, 1)

    return mean, std


def vector_rel_error(pred, true):
    n = pred.shape[0]
    num = torch.linalg.vector_norm((pred - true).reshape(n, -1), dim=1)
    den = torch.linalg.vector_norm(true.reshape(n, -1), dim=1).clamp_min(1e-8)
    return (num / den).mean()


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class DirectConditionalOperator(nn.Module):
    """
    Uses exactly the same U-Net architecture as one DSBM drift network.

    base.ConditionalUNetDrift expects (field, intervention, bridge_time).
    For this deterministic endpoint operator we give it a fixed time channel
    t=1, so parameter count / convolutional architecture stays identical to
    one conditional DSBM drift network.
    """

    def __init__(self, base_channels=32):
        super().__init__()
        self.net = base.ConditionalUNetDrift(base_channels)

    def forward(self, x0, a):
        t = torch.ones(
            (x0.shape[0], 1),
            device=x0.device,
            dtype=x0.dtype,
        )
        return self.net(x0, a, t)

    def directional_jvp(self, x0, a, direction, create_graph=False):
        # x0 is held fixed. We differentiate only with respect to intervention a.
        def f(ai):
            return self.forward(x0, ai)

        _, jv = torch.autograd.functional.jvp(
            f,
            (a,),
            (direction,),
            create_graph=create_graph,
            strict=False,
        )
        return jv


# ---------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------

def cpu_randint(n, b, generator):
    return torch.randint(
        0,
        n,
        (b,),
        generator=generator,
        device="cpu",
    )


def sample_endpoint_batch(train, batch_size, generator, device):
    b = min(batch_size, train["x0"].shape[0])
    idx = cpu_randint(train["x0"].shape[0], b, generator)
    return (
        train["x0"][idx].to(device),
        train["a"][idx].to(device),
        train["x1"][idx].to(device),
    )


def sample_response(src, b, generator, device):
    if b <= 0:
        return None

    idx = cpu_randint(src["x0"].shape[0], b, generator)
    return {
        "x0": src["x0"][idx].to(device),
        "a": src["a"][idx].to(device),
        "direction": src["direction"][idx].to(device),
        "Jv_star": src["Jv_star"][idx].to(device),
    }


def response_batch(anchor, colloc, total, anchor_fraction, generator, device):
    na = int(round(total * anchor_fraction))
    if total >= 2:
        na = max(1, min(total - 1, na))
    else:
        na = total
    nc = total - na

    pieces = [
        p for p in [
            sample_response(anchor, na, generator, device),
            sample_response(colloc, nc, generator, device),
        ]
        if p is not None
    ]

    out = {}
    for k in ["x0", "a", "direction", "Jv_star"]:
        out[k] = torch.cat([p[k] for p in pieces], dim=0)

    # Shuffle with the sensitivity RNG only.
    perm_cpu = torch.randperm(
        out["x0"].shape[0],
        generator=generator,
        device="cpu",
    )
    perm = perm_cpu.to(device)
    for k in out:
        out[k] = out[k][perm]

    return out


def gradient_sanity(model, anchor, colloc, sens_batch_size,
                    anchor_fraction, generator, device):
    state = generator.get_state()

    model.zero_grad(set_to_none=True)
    try:
        batch = response_batch(
            anchor,
            colloc,
            min(2, sens_batch_size),
            anchor_fraction,
            generator,
            device,
        )
        pred = model.directional_jvp(
            batch["x0"],
            batch["a"],
            batch["direction"],
            create_graph=True,
        )
        loss = F.mse_loss(pred, batch["Jv_star"])
        loss.backward()

        sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                sq += float((p.grad.detach() ** 2).sum().cpu())

        if sq <= 0:
            raise RuntimeError("zero sensitivity gradient")

        return {
            "loss": float(loss.detach().cpu()),
            "grad_norm": sq ** 0.5,
        }
    finally:
        model.zero_grad(set_to_none=True)
        generator.set_state(state)


def train_model(
    model,
    train,
    anchor,
    colloc,
    device,
    seed,
    steps,
    batch_size,
    lr,
    grad_clip,
    lambda_sens,
    sens_every,
    sens_batch_size,
    anchor_fraction,
    sens_start_step,
):
    # Separate generators ensure the endpoint minibatch sequence is identical
    # between lambda=0 and lambda>0 runs with the same seed.
    endpoint_gen = torch.Generator(device="cpu").manual_seed(seed + 500001)
    sens_gen = torch.Generator(device="cpu").manual_seed(seed + 700001)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    hist = []
    recent_endpoint = []
    recent_sens = []
    start = time.time()

    for step in range(1, steps + 1):
        x0, a, x1 = sample_endpoint_batch(
            train,
            batch_size,
            endpoint_gen,
            device,
        )

        pred = model(x0, a)
        endpoint_loss = F.mse_loss(pred, x1)

        sens_loss = None
        if (
            lambda_sens > 0
            and step >= sens_start_step
            and step % sens_every == 0
        ):
            sb = response_batch(
                anchor,
                colloc,
                sens_batch_size,
                anchor_fraction,
                sens_gen,
                device,
            )

            jv = model.directional_jvp(
                sb["x0"],
                sb["a"],
                sb["direction"],
                create_graph=True,
            )
            sens_loss = F.mse_loss(jv, sb["Jv_star"])
            total = endpoint_loss + lambda_sens * sens_loss
        else:
            total = endpoint_loss

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
        )
        opt.step()

        recent_endpoint.append(float(endpoint_loss.detach().cpu()))
        recent_endpoint = recent_endpoint[-100:]

        if sens_loss is not None:
            recent_sens.append(float(sens_loss.detach().cpu()))
            recent_sens = recent_sens[-100:]

        report_every = max(50, steps // 10)
        if step == 1 or step % report_every == 0 or step == steps:
            rec = {
                "step": step,
                "endpoint_loss_last100": float(np.mean(recent_endpoint)),
                "sensitivity_loss_last100":
                    float(np.mean(recent_sens)) if recent_sens else None,
                "total_loss": float(total.detach().cpu()),
            }
            hist.append(rec)
            log(
                f"step {step:5d}/{steps}",
                f"endpoint={rec['endpoint_loss_last100']:.6f}",
                f"sens={rec['sensitivity_loss_last100'] if rec['sensitivity_loss_last100'] is not None else float('nan'):.6f}",
                f"total={rec['total_loss']:.6f}",
            )

    return hist, time.time() - start


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def endpoint_metrics(model, data, eval_batch_size, device, metadata):
    preds = []
    n = data["x0"].shape[0]

    for s in range(0, n, eval_batch_size):
        e = min(n, s + eval_batch_size)
        x0 = data["x0"][s:e].to(device)
        a = data["a"][s:e].to(device)
        preds.append(model(x0, a).cpu())

    pred = torch.cat(preds, dim=0)
    true = data["x1"]
    diff = pred - true

    rel = vector_rel_error(pred, true)
    rmse = torch.sqrt((diff ** 2).mean())

    mean, std = norm_stats(metadata, "cpu")
    pred_raw = pred * std + mean
    raw = torch.sqrt(((pred_raw - data["x1_raw"]) ** 2).mean())

    return {
        "field_rmse_norm": float(rmse),
        "field_rel_l2": float(rel),
        "field_rmse_raw": float(raw),
    }


def sensitivity_metrics(model, data, eval_batch_size, device, metadata):
    # Match the existing DSBM evaluator: sensitivity is evaluated on the
    # first min(eval_batch_size, N) examples because it is the expensive metric.
    n = min(eval_batch_size, data["x0"].shape[0])

    x0 = data["x0"][:n].to(device)
    a = data["a"][:n].to(device)
    direction = data["direction"][:n].to(device)
    true = data["Jv_star"][:n].to(device)

    with torch.enable_grad():
        pred = model.directional_jvp(
            x0,
            a,
            direction,
            create_graph=False,
        ).detach()

    diff = pred - true
    rel = vector_rel_error(pred, true)
    rmse = torch.sqrt((diff ** 2).mean())

    _, std = norm_stats(metadata, device)
    raw = torch.sqrt(((diff * std) ** 2).mean())

    return {
        "directional_j_rel_error": float(rel.cpu()),
        "directional_j_rmse_norm": float(rmse.cpu()),
        # Deterministic operator: pathwise and mean response coincide.
        "directional_j_pathwise_rmse_norm": float(rmse.cpu()),
        "directional_j_rmse_raw": float(raw.cpu()),
    }


@torch.no_grad()
def finite_metrics(model, data, eval_batch_size, finite_delta,
                   device, metadata):
    n = min(eval_batch_size, data["x0"].shape[0])

    x0 = data["x0"][:n].to(device)
    a = data["a"][:n].to(device)
    direction = data["direction"][:n].to(device)
    true = data["finite_response"][:n].to(device)

    y0 = model(x0, a)
    y1 = model(x0, a + finite_delta * direction)
    pred = y1 - y0

    diff = pred - true
    rel = vector_rel_error(pred, true)
    rmse = torch.sqrt((diff ** 2).mean())

    _, std = norm_stats(metadata, device)
    raw = torch.sqrt(((diff * std) ** 2).mean())

    return {
        "finite_response_rmse_norm": float(rmse.cpu()),
        "finite_response_rel_l2": float(rel.cpu()),
        "finite_response_rmse_raw": float(raw.cpu()),
    }


def evaluate_split(
    split,
    model,
    endpoint,
    response,
    eval_batch_size,
    finite_delta,
    device,
    metadata,
):
    out = endpoint_metrics(
        model,
        endpoint,
        eval_batch_size,
        device,
        metadata,
    )
    out.update(
        sensitivity_metrics(
            model,
            response,
            eval_batch_size,
            device,
            metadata,
        )
    )
    out.update(
        finite_metrics(
            model,
            response,
            eval_batch_size,
            finite_delta,
            device,
            metadata,
        )
    )

    log(split, json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        default="runs/pdebench_reaction_diffusion_official",
    )
    p.add_argument(
        "--run-root",
        default="runs/pdebench_rd_direct_operator",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=32,
    )

    p.add_argument(
        "--steps",
        type=int,
        default=3200,
        help="Endpoint optimization steps. Same value should be used for direct and derivative-informed runs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    p.add_argument(
        "--base-channels",
        type=int,
        default=32,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--lambda-sens",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--sens-every",
        type=int,
        default=10,
    )
    p.add_argument(
        "--sens-batch-size",
        type=int,
        default=4,
    )
    p.add_argument(
        "--anchor-fraction",
        type=float,
        default=0.25,
    )
    p.add_argument(
        "--sens-window",
        type=int,
        default=800,
        help=(
            "Number of final optimization steps during which sensitivity "
            "updates are enabled. With 3200 total steps, sens_every=10, "
            "and sens_window=800, this gives exactly 80 sensitivity updates, "
            "matching the PDEBench Tangent-SBM continuation."
        ),
    )
    p.add_argument(
        "--anchor-response",
        default=None,
    )
    p.add_argument(
        "--response-collocation",
        default=None,
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
    p.add_argument(
        "--quick",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.lambda_sens < 0:
        raise ValueError("--lambda-sens must be >= 0")

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    endpoint, response_eval, metadata, state_shape, intervention_dim = (
        base.load_dataset(data_dir)
    )

    if state_shape != (2, 128, 128):
        raise RuntimeError(f"Expected state shape (2,128,128), got {state_shape}")
    if intervention_dim != 3:
        raise RuntimeError(f"Expected intervention_dim=3, got {intervention_dim}")

    finite_delta = float(metadata.get("finite_delta", 0.25))

    steps = args.steps
    batch_size = args.batch_size
    base_channels = args.base_channels
    eval_batch_size = args.eval_batch_size
    sens_batch_size = args.sens_batch_size

    if args.quick:
        steps = min(steps, 20)
        batch_size = min(batch_size, 2)
        base_channels = min(base_channels, 16)
        eval_batch_size = 1
        sens_batch_size = 1

    sens_window = min(int(args.sens_window), steps)
    sens_start_step = max(1, steps - sens_window + 1)

    method = (
        "direct_operator"
        if args.lambda_sens == 0
        else "derivative_informed_operator"
    )

    lambda_tag = str(args.lambda_sens).replace(".", "p")
    run_dir = (
        Path(args.run_root)
        / f"lambda_{lambda_tag}"
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir / "train.log")

    log("=" * 80)
    log("PDEBENCH REACTION-DIFFUSION DETERMINISTIC OPERATOR")
    log("=" * 80)
    log("method =", method)
    log("seed =", args.seed)
    log("device =", device)
    log("steps =", steps)
    log("batch_size =", batch_size)
    log("base_channels =", base_channels)
    log("lambda_sens =", args.lambda_sens)
    log("sens_every =", args.sens_every)
    log("sens_batch_size =", sens_batch_size)
    log("anchor_fraction =", args.anchor_fraction)
    log("sens_window =", sens_window)
    log("sens_start_step =", sens_start_step)
    if args.lambda_sens > 0:
        n_sens_updates = sum(
            1 for s in range(sens_start_step, steps + 1)
            if s % args.sens_every == 0
        )
        log("planned_sensitivity_updates =", n_sens_updates)
        log("planned_response_examples =", n_sens_updates * sens_batch_size)
    log("finite_delta =", finite_delta)

    model = DirectConditionalOperator(
        base_channels=base_channels,
    ).to(device)

    num_parameters = sum(p.numel() for p in model.parameters())
    log("num_parameters =", num_parameters)

    anchor = None
    colloc = None

    if args.lambda_sens > 0:
        anchor_path = (
            Path(args.anchor_response)
            if args.anchor_response is not None
            else data_dir / "anchor_response.pt"
        )
        colloc_path = (
            Path(args.response_collocation)
            if args.response_collocation is not None
            else data_dir / "response_collocation.pt"
        )

        anchor = load_response_only(anchor_path)
        colloc = load_response_only(colloc_path)

        sanity_gen = torch.Generator(device="cpu").manual_seed(args.seed + 700001)
        sanity = gradient_sanity(
            model,
            anchor,
            colloc,
            sens_batch_size,
            args.anchor_fraction,
            sanity_gen,
            device,
        )
        log("Gradient sanity PASSED:", sanity)

    history, train_seconds = train_model(
        model=model,
        train=endpoint["train"],
        anchor=anchor,
        colloc=colloc,
        device=device,
        seed=args.seed,
        steps=steps,
        batch_size=batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        lambda_sens=args.lambda_sens,
        sens_every=args.sens_every,
        sens_batch_size=sens_batch_size,
        anchor_fraction=args.anchor_fraction,
        sens_start_step=sens_start_step,
    )

    with open(run_dir / "convergence.json", "w") as f:
        json.dump(history, f, indent=2)

    model.eval()
    results = {}
    for split in [
        "test_seen",
        "test_id",
        "test_ood_near",
        "test_ood_far",
    ]:
        results[split] = evaluate_split(
            split=split,
            model=model,
            endpoint=endpoint[split],
            response=response_eval[split],
            eval_batch_size=eval_batch_size,
            finite_delta=finite_delta,
            device=device,
            metadata=metadata,
        )

    config = {
        "method": method,
        "seed": args.seed,
        "steps": steps,
        "batch_size": batch_size,
        "base_channels": base_channels,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "lambda_sens": args.lambda_sens,
        "sens_every": args.sens_every,
        "sens_batch_size": sens_batch_size,
        "anchor_fraction": args.anchor_fraction,
        "sens_window": sens_window,
        "sens_start_step": sens_start_step,
        "planned_sensitivity_updates": (
            sum(
                1 for s in range(sens_start_step, steps + 1)
                if s % args.sens_every == 0
            )
            if args.lambda_sens > 0 else 0
        ),
        "eval_batch_size": eval_batch_size,
        "finite_delta": finite_delta,
        "quick": bool(args.quick),
        "num_parameters": num_parameters,
    }

    ckpt_path = run_dir / "final_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
        },
        ckpt_path,
    )

    payload = {
        "method": method,
        "benchmark": "pdebench_reaction_diffusion",
        "seed": args.seed,
        "train_seconds": train_seconds,
        "config": config,
        "metrics": results,
        "final_checkpoint": str(ckpt_path),
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    rows = []
    for split, metrics in results.items():
        rows.append({
            "method": method,
            "seed": args.seed,
            "lambda_sens": args.lambda_sens,
            "split": split,
            **metrics,
        })

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)

    log("=" * 80)
    log("FINAL SUMMARY")
    log("=" * 80)
    for split, m in results.items():
        log(
            split,
            "| field_rel", f"{m['field_rel_l2']:.6f}",
            "| Jv_rel", f"{m['directional_j_rel_error']:.6f}",
            "| finite_rel", f"{m['finite_response_rel_l2']:.6f}",
        )
    log("Saved:", run_dir)


if __name__ == "__main__":
    main()
