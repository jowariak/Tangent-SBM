#!/usr/bin/env python3
"""SNS target-correctness control: shuffled_targets.
Derived from the uploaded multipair_memfix parent. Only training Jv_star labels
are transformed. Free learned-SDE rollouts and independent-pair cross loss remain
unchanged. Evaluate checkpoints using the original locked FINAL2 evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import spdebench_sns_conditional_dsbm_v2 as base


TARGET_CONTROL = "shuffled_targets"

def transform_training_targets(anchor, colloc):
    """Transform only normalized response labels; never mutate input records."""
    transformed = []
    manifest = {"target_control": TARGET_CONTROL, "sources": {}}
    for name, src, seed in [("anchor", anchor, 19092026),
                            ("collocation", colloc, 19092027)]:
        target = src["Jv_star"]
        n = target.shape[0]
        record = {"records": int(n)}
        if TARGET_CONTROL == "shuffled_targets":
            if n < 2:
                raise ValueError(f"{name}: shuffling needs at least two records")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            identity = torch.arange(n)
            # Uniform random permutation conditioned on no fixed points.
            # This moves entire response fields, never individual pixels.
            for _ in range(10000):
                permutation = torch.randperm(n, generator=generator)
                if not torch.any(permutation == identity):
                    break
            else:
                raise RuntimeError("Could not generate a derangement")
            changed = target[permutation.to(target.device)]
            record.update(shuffle_seed=seed,
                          target_source_indices=permutation.tolist(),
                          fixed_points=0)
        elif TARGET_CONTROL == "sign_reversed_targets":
            changed = -target
        else:
            raise ValueError(TARGET_CONTROL)
        transformed.append({**src, "Jv_star": changed})
        manifest["sources"][name] = record
    return transformed[0], transformed[1], manifest


def setup_logging(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, mode="a"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def log(*args):
    logging.info(" ".join(str(x) for x in args))


def sigma_tag(x: float) -> str:
    return str(float(x)).replace(".", "p")


def lambda_tag(x: float) -> str:
    # Stable filesystem-friendly tag, e.g. 0.1 -> 0p1, 1000 -> 1000
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def _first_present(obj, names):
    for n in names:
        if n in obj:
            return obj[n]
    return None


def load_response_file(path: Path, state_mean: float, state_std: float):
    """
    Load a response-only training file robustly.

    Supports either:
      raw keys:  x0, a, direction, Jv_star
      normalized keys: x0_norm, a, direction, Jv_star_norm

    No endpoint labels are used.  If an endpoint-like key is present, fail.
    """
    obj = base.safe_load(path)

    forbidden = {
        "xT", "x1", "xT_samples", "endpoint", "endpoint_labels",
        "finite_response_star", "finite_response_samples",
    }
    bad = sorted(k for k in forbidden if k in obj)
    if bad:
        raise RuntimeError(
            f"{path}: response-training file unexpectedly contains endpoint/"
            f"finite-response labels: {bad}"
        )
    if "endpoint_labels_included" in obj and bool(obj["endpoint_labels_included"]):
        raise RuntimeError(f"{path}: endpoint_labels_included=True")

    x0n = _first_present(obj, ["x0_norm"])
    if x0n is None:
        x0raw = _first_present(obj, ["x0"])
        if x0raw is None:
            raise RuntimeError(f"{path}: missing x0/x0_norm")
        x0n = base.normalize_state(x0raw.float(), state_mean, state_std)
    else:
        x0n = x0n.float()

    a = _first_present(obj, ["a"])
    if a is None:
        raise RuntimeError(f"{path}: missing a")
    a = a.float()
    if a.ndim == 1:
        a = a[:, None]

    d = _first_present(obj, ["direction"])
    if d is None:
        d = torch.ones_like(a)
    d = d.float()
    if d.ndim == 1:
        d = d[:, None]

    jn = _first_present(obj, ["Jv_star_norm"])
    if jn is None:
        jraw = _first_present(obj, ["Jv_star"])
        if jraw is None:
            raise RuntimeError(f"{path}: missing Jv_star/Jv_star_norm")
        jn = jraw.float() / float(state_std)
    else:
        jn = jn.float()

    if x0n.ndim != 4 or tuple(x0n.shape[1:]) != (1, 64, 64):
        raise RuntimeError(f"{path}: expected x0 [N,1,64,64], got {tuple(x0n.shape)}")
    if jn.shape != x0n.shape:
        raise RuntimeError(f"{path}: Jv_star shape {tuple(jn.shape)} != x0 {tuple(x0n.shape)}")
    if a.shape[0] != x0n.shape[0] or d.shape[0] != x0n.shape[0]:
        raise RuntimeError(f"{path}: inconsistent number of response conditions")

    return {"x0": x0n, "a": a, "direction": d, "Jv_star": jn}


class TangentFieldDSBM(base.ConditionalFieldDSBM):
    def __init__(
        self,
        cfg,
        device,
        lambda_sens: float,
        sens_every: int,
        sens_batch_size: int,
        anchor_fraction: float,
        sens_pairs: int,
        sens_seed: int,
    ):
        super().__init__(cfg, device)
        self.lambda_sens = float(lambda_sens)
        self.sens_every = int(sens_every)
        self.sens_batch_size = int(sens_batch_size)
        self.anchor_fraction = float(anchor_fraction)
        self.sens_pairs = int(sens_pairs)
        if self.sens_every <= 0:
            raise ValueError("sens_every must be positive")
        if self.sens_batch_size <= 0:
            raise ValueError("sens_batch_size must be positive")
        if self.sens_pairs <= 0:
            raise ValueError("sens_pairs must be positive")
        if not (0.0 <= self.anchor_fraction <= 1.0):
            raise ValueError("anchor_fraction must be in [0,1]")
        self.sens_generator = torch.Generator(device="cpu").manual_seed(int(sens_seed))

    def _idx(self, n: int, b: int):
        return torch.randint(0, n, (b,), generator=self.sens_generator, device="cpu")

    def _sens_noise_bank(self, batch: int, dtype):
        # Use a dedicated CPU generator so sensitivity sampling does not consume
        # the global training RNG used by ordinary bridge matching.
        return [
            torch.randn(
                batch, 1, 64, 64,
                generator=self.sens_generator,
                dtype=dtype,
                device="cpu",
            ).to(self.device)
            for _ in range(self.cfg.num_steps)
        ]

    def _sample_response(self, src, b: int):
        if b <= 0:
            return None
        idx = self._idx(src["x0"].shape[0], b)
        return {
            k: src[k][idx].to(self.device)
            for k in ["x0", "a", "direction", "Jv_star"]
        }

    def _response_batch(self, anchor, colloc):
        total = self.sens_batch_size
        na = int(round(total * self.anchor_fraction))
        if total >= 2 and anchor["x0"].shape[0] > 0 and colloc["x0"].shape[0] > 0:
            na = max(1, min(total - 1, na))
        elif colloc["x0"].shape[0] == 0:
            na = total
        nc = total - na

        pieces = [
            p for p in [
                self._sample_response(anchor, na),
                self._sample_response(colloc, nc),
            ]
            if p is not None
        ]
        x0 = torch.cat([p["x0"] for p in pieces], 0)
        a = torch.cat([p["a"] for p in pieces], 0)
        d = torch.cat([p["direction"] for p in pieces], 0)
        target = torch.cat([p["Jv_star"] for p in pieces], 0)

        perm = torch.randperm(
            x0.shape[0], generator=self.sens_generator, device="cpu"
        ).to(self.device)
        return x0[perm], a[perm], d[perm], target[perm]

    def tangent_training_rollout(self, x0, a, direction, noise_bank):
        """
        Differentiable Euler-Maruyama rollout of state and directional tangent.
        Additive intervention-independent diffusion contributes no tangent-noise term.
        """
        dt = 1.0 / float(self.cfg.num_steps)
        x = x0
        R = torch.zeros_like(x0)

        for k in range(self.cfg.num_steps):
            t = torch.full(
                (x.shape[0], 1),
                k / float(self.cfg.num_steps),
                device=x.device,
                dtype=x.dtype,
            )

            def drift_fn(xi, ai):
                return self.net_f(xi, ai, t)

            drift, tangent_drift = torch.autograd.functional.jvp(
                drift_fn,
                (x, a),
                (R, direction),
                create_graph=True,
                strict=False,
            )
            x = (
                x
                + dt * drift
                + self.cfg.reference_sigma * math.sqrt(dt) * noise_bank[k]
            )
            R = R + dt * tangent_drift

        return R

    def _single_cross_pair_loss(self, x0, a, d, target):
        """One unbiased cross-rollout estimate for a fixed response batch."""
        nb1 = self._sens_noise_bank(x0.shape[0], x0.dtype)
        nb2 = self._sens_noise_bank(x0.shape[0], x0.dtype)

        r1 = self.tangent_training_rollout(x0, a, d, nb1)
        r2 = self.tangent_training_rollout(x0, a, d, nb2)

        e1 = r1 - target
        e2 = r2 - target
        return torch.mean(e1 * e2)

    def conditional_mean_response_loss(self, anchor, colloc):
        """
        Diagnostic/evaluation form of the K-pair estimator.

        Training uses sequential backward passes (see train_pass_tangent) so
        only ONE pair's autograd graph is resident at a time.  This function
        remains useful for small-batch gradient sanity checks.
        """
        x0, a, d, target = self._response_batch(anchor, colloc)
        vals = [
            self._single_cross_pair_loss(x0, a, d, target)
            for _ in range(self.sens_pairs)
        ]
        return torch.stack(vals).mean()

    def train_pass_tangent(self, data, fb, anchor, colloc):
        z0, z1, a = self.regenerate_coupling(data)
        net = self.nets[fb]
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr, weight_decay=1e-5)

        n = z0.shape[0]
        bridge_recent, sens_recent = [], []
        sens_updates = 0

        for step in range(1, self.cfg.inner_steps + 1):
            b = min(self.cfg.batch_size, n)
            idx = torch.randint(0, n, (b,), device=self.device)
            zt, ba, t, target = self.get_train_tuple(
                z0[idx], z1[idx], a[idx], fb
            )
            pred = net(zt, ba, t)
            bridge = F.mse_loss(pred, target)

            do_sens = (
                fb == "f"
                and self.lambda_sens > 0
                and step % self.sens_every == 0
            )

            opt.zero_grad(set_to_none=True)

            if do_sens:
                # Backprop the ordinary bridge term first; its graph can be
                # released immediately.
                bridge.backward()

                # Sample ONE response minibatch and reuse it for all K pairs,
                # exactly matching the intended K-pair estimator.
                x0_s, a_s, d_s, target_s = self._response_batch(anchor, colloc)

                pair_vals = []
                scale = self.lambda_sens / float(self.sens_pairs)
                for _ in range(self.sens_pairs):
                    pair_loss = self._single_cross_pair_loss(
                        x0_s, a_s, d_s, target_s
                    )
                    # Each backward frees this pair's rollout/JVP graph before
                    # the next pair is constructed.  Gradient accumulation is
                    # mathematically identical to backpropagating the average.
                    (scale * pair_loss).backward()
                    pair_vals.append(float(pair_loss.detach().cpu()))

                sens_value = float(np.mean(pair_vals))
                total_value = float(bridge.detach().cpu()) + self.lambda_sens * sens_value
                sens_updates += 1
            else:
                bridge.backward()
                sens_value = None
                total_value = float(bridge.detach().cpu())

            torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
            opt.step()

            bridge_recent.append(float(bridge.detach().cpu()))
            bridge_recent = bridge_recent[-100:]
            if sens_value is not None:
                sens_recent.append(sens_value)
                sens_recent = sens_recent[-100:]

            rep = max(50, self.cfg.inner_steps // 5)
            if step == 1 or step % rep == 0 or step == self.cfg.inner_steps:
                log(
                    f"{fb} step {step:5d}/{self.cfg.inner_steps}",
                    f"bridge={float(bridge.detach().cpu()):.6f}",
                    f"sens_cross={(np.mean(sens_recent) if sens_recent else float('nan')):.8f}",
                    f"total={total_value:.6f}",
                )

        self.prev_fb = fb
        return {
            "bridge_loss_last100": float(np.mean(bridge_recent)),
            "sensitivity_cross_loss_last100": (
                float(np.mean(sens_recent)) if sens_recent else None
            ),
            "sensitivity_updates": int(sens_updates),
        }


def gradient_sanity(model, anchor, colloc):
    old_bs = model.sens_batch_size
    old_state = model.sens_generator.get_state()
    model.sens_batch_size = min(2, old_bs)

    model.net_f.zero_grad(set_to_none=True)
    model.net_b.zero_grad(set_to_none=True)
    try:
        x0, a, d, target = model._response_batch(anchor, colloc)
        vals = []
        for _ in range(model.sens_pairs):
            loss_k = model._single_cross_pair_loss(x0, a, d, target)
            (loss_k / float(model.sens_pairs)).backward()
            vals.append(float(loss_k.detach().cpu()))

        fsq = sum(
            float((p.grad.detach() ** 2).sum().cpu())
            for p in model.net_f.parameters()
            if p.grad is not None
        )
        bsq = sum(
            float((p.grad.detach() ** 2).sum().cpu())
            for p in model.net_b.parameters()
            if p.grad is not None
        )
        if not math.isfinite(fsq) or fsq <= 0:
            raise RuntimeError("zero/nonfinite forward sensitivity gradient")
        if bsq > 0:
            raise RuntimeError("sensitivity unexpectedly touched backward net")
        return {
            "cross_loss": float(np.mean(vals)),
            "forward_grad_norm": fsq ** 0.5,
        }
    finally:
        model.net_f.zero_grad(set_to_none=True)
        model.net_b.zero_grad(set_to_none=True)
        model.sens_batch_size = old_bs
        model.sens_generator.set_state(old_state)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("runs/spdebench_sns_response"))
    p.add_argument(
        "--baseline-run-root",
        type=Path,
        default=Path("runs/spdebench_sns_conditional_final"),
    )
    p.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/sns_target_ablations/production/shuffled_targets"),
    )
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--reference-sigma", type=float, default=1.4)
    p.add_argument("--fork-imf", type=int, default=3)
    p.add_argument("--total-imf", type=int, default=5)
    p.add_argument("--lambda-sens", type=float, required=True)
    p.add_argument("--sens-every", type=int, default=10)
    p.add_argument("--sens-batch-size", type=int, default=8)
    p.add_argument(
        "--sens-pairs",
        type=int,
        default=1,
        help="Independent cross-rollout pairs per sampled response condition.",
    )
    p.add_argument("--anchor-fraction", type=float, default=1.0 / 3.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-gradient-sanity-check", action="store_true")
    p.add_argument(
        "--skip-final-test-eval",
        action="store_true",
        help="Train/save checkpoints without loading/evaluating Seen/ID/Near/Far.",
    )
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = base.resolve_device(args.device)

    fork_path = (
        args.baseline_run_root
        / f"sigma_{sigma_tag(args.reference_sigma)}"
        / f"seed_{args.seed}"
        / f"imf_{args.fork_imf}.pt"
    )
    if not fork_path.exists():
        raise FileNotFoundError(f"Missing Conditional fork checkpoint: {fork_path}")

    ckpt = base.safe_load(fork_path)
    cfg = base.Config(**ckpt["config"])
    cfg.fork_imf = int(args.fork_imf)
    cfg.total_imf = int(args.total_imf)

    if abs(float(cfg.reference_sigma) - float(args.reference_sigma)) > 1e-12:
        raise RuntimeError(
            f"Checkpoint sigma={cfg.reference_sigma} but requested {args.reference_sigma}"
        )
    if int(ckpt.get("imf", -1)) != args.fork_imf:
        raise RuntimeError(
            f"Checkpoint says IMF {ckpt.get('imf')} but --fork-imf={args.fork_imf}"
        )
    if cfg.total_imf <= cfg.fork_imf:
        raise ValueError("total-imf must be > fork-imf")

    # IMPORTANT: base.load_dataset loads the fixed test files.  For tuning runs
    # we intentionally avoid it and load endpoint train + metadata directly.
    metadata = json.load(open(args.data_dir / "metadata.json"))
    state_mean, state_std = base.normalization_from_metadata(metadata)
    tr = base.safe_load(args.data_dir / "endpoint_train.pt")
    train = {
        "x0": base.normalize_state(tr["x0"].float(), state_mean, state_std),
        "x1": base.normalize_state(tr["xT"].float(), state_mean, state_std),
        "x0_raw": tr["x0"].float(),
        "x1_raw": tr["xT"].float(),
        "a": tr["a"].float(),
    }

    anchor = load_response_file(
        args.data_dir / "anchor_response.pt", state_mean, state_std
    )
    colloc = load_response_file(
        args.data_dir / "response_collocation.pt", state_mean, state_std
    )

    if args.quick:
        cfg.inner_steps = min(cfg.inner_steps, 10)
        cfg.num_steps = min(cfg.num_steps, 3)
        train = {k: (v[:32] if torch.is_tensor(v) else v) for k, v in train.items()}
        anchor = {k: v[:4] for k, v in anchor.items()}
        colloc = {k: v[:4] for k, v in colloc.items()}

    anchor, colloc, target_manifest = transform_training_targets(anchor, colloc)

    run_dir = (
        args.run_root
        / f"lambda_{lambda_tag(args.lambda_sens)}"
        / f"seed_{args.seed}"
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "target_control.json").write_text(json.dumps(target_manifest, indent=2))
    setup_logging(run_dir / "tangent_sbm.log")

    log("=" * 80)
    log("SPDEBench stochastic Navier-Stokes Tangent-SBM")
    log("=" * 80)
    log("target control =", TARGET_CONTROL)
    log("device =", device)
    log("fork =", fork_path)
    log("fork IMF =", args.fork_imf, "final IMF =", args.total_imf)
    log("reference_sigma =", cfg.reference_sigma)
    log("lambda_sens =", args.lambda_sens)
    log("sens_every =", args.sens_every)
    log("sens_batch_size =", args.sens_batch_size)
    log("sens_pairs =", args.sens_pairs)
    log("anchor_fraction =", args.anchor_fraction)
    log("anchor response conditions =", anchor["x0"].shape[0])
    log("collocation response conditions =", colloc["x0"].shape[0])
    log(
        "response objective = unbiased conditional-mean cross loss averaged over",
        args.sens_pairs,
        "independent rollout pair(s) per condition",
    )
    log("multipair backward = sequential (one pair graph resident at a time)")

    model = TangentFieldDSBM(
        cfg=cfg,
        device=device,
        lambda_sens=args.lambda_sens,
        sens_every=args.sens_every,
        sens_batch_size=args.sens_batch_size,
        anchor_fraction=args.anchor_fraction,
        sens_pairs=args.sens_pairs,
        sens_seed=args.seed + 700001,
    )
    model.load_state_dict(ckpt["model"])
    base.restore_rng_state(ckpt.get("rng_state"))

    if not args.skip_gradient_sanity_check and args.lambda_sens > 0:
        sanity = gradient_sanity(model, anchor, colloc)
        log("Gradient sanity PASSED:", json.dumps(sanity))

    history = []
    start = time.time()
    total_sens_updates = 0

    for imf in range(args.fork_imf + 1, args.total_imf + 1):
        t0 = time.time()

        log("")
        log("=" * 80)
        log(f"IMF {imf}/{args.total_imf} BACKWARD")
        log("=" * 80)
        bs = model.train_pass_tangent(train, "b", anchor, colloc)

        log("")
        log("=" * 80)
        log(f"IMF {imf}/{args.total_imf} FORWARD")
        log("=" * 80)
        fs = model.train_pass_tangent(train, "f", anchor, colloc)
        total_sens_updates += fs["sensitivity_updates"]

        conv = base.convergence_metric(model, train, cfg, device)
        row = {
            "imf": imf,
            "backward_bridge_loss_last100": bs["bridge_loss_last100"],
            "forward_bridge_loss_last100": fs["bridge_loss_last100"],
            "forward_sensitivity_cross_loss_last100": fs["sensitivity_cross_loss_last100"],
            "sensitivity_updates": fs["sensitivity_updates"],
            "train_endpoint_rmse_norm": conv,
            "seconds": time.time() - t0,
        }
        history.append(row)
        (run_dir / "convergence.json").write_text(json.dumps(history, indent=2))

        save_obj = {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "imf": int(imf),
            "fork_imf": int(args.fork_imf),
            "lambda_sens": float(args.lambda_sens),
            "sens_every": int(args.sens_every),
            "sens_batch_size": int(args.sens_batch_size),
            "sens_pairs": int(args.sens_pairs),
            "anchor_fraction": float(args.anchor_fraction),
            "target_control": TARGET_CONTROL,
        "response_objective": "conditional_mean_multi_pair_unbiased_cross_average",
            "state_shape": [1, 64, 64],
            "intervention_dim": 1,
            "state_mean": float(state_mean),
            "state_std": float(state_std),
            "rng_state": base.capture_rng_state(),
        }
        torch.save(save_obj, run_dir / f"imf_{imf}.pt")
        log("IMF summary =", json.dumps(row, indent=2))
        log("Saved", run_dir / f"imf_{imf}.pt")

    training_summary = {
        "method": "tangent_sbm_" + TARGET_CONTROL,
        "benchmark": "spdebench_sns_response",
        "seed": args.seed,
        "fork_checkpoint": str(fork_path),
        "fork_imf": args.fork_imf,
        "total_imf": args.total_imf,
        "lambda_sens": args.lambda_sens,
        "sens_every": args.sens_every,
        "sens_batch_size": args.sens_batch_size,
        "sens_pairs": args.sens_pairs,
        "anchor_fraction": args.anchor_fraction,
        "target_control": TARGET_CONTROL,
        "response_objective": "conditional_mean_multi_pair_unbiased_cross_average",
        "total_sensitivity_updates": total_sens_updates,
        "sampled_response_conditions": total_sens_updates * args.sens_batch_size,
        "cross_pairs_evaluated": total_sens_updates * args.sens_batch_size * args.sens_pairs,
        "sensitivity_rollouts": total_sens_updates * args.sens_batch_size * args.sens_pairs * 2,
        "train_seconds_after_fork": time.time() - start,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2)
    )

    if args.skip_final_test_eval:
        log("")
        log("=" * 80)
        log("TRAINING COMPLETE -- TEST EVALUATION SKIPPED")
        log("=" * 80)
        log("Use spdebench_sns_select_tangent.py with validation.pt to select lambda.")
        return

    # Only final, frozen runs reach here.
    _, eval_data, _, _, _ = base.load_dataset(args.data_dir)
    results = {
        split: base.evaluate_split(
            split, model, eval_data[split], cfg, device, state_mean, state_std, log
        )
        for split in base.EVAL_FILES
    }

    payload = {**training_summary, "metrics": results}
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    rows = [{"split": s, **m} for s, m in results.items()]
    if rows:
        with (run_dir / "metrics.csv").open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    log("")
    log("=" * 80)
    log("FINAL SUMMARY")
    log("=" * 80)
    for split, m in results.items():
        log(
            split,
            "| mean_rel", f"{m['field_mean_rel_l2']:.6f}",
            "| spread_rel", f"{m['spread_rel_l2']:.6f}",
            "| energy", f"{m['energy_distance_per_sqrt_pixel']:.6f}",
            "| Jv_rel", f"{m['directional_j_rel_error']:.6f}",
            "| finite_rel", f"{m['finite_response_rel_l2']:.6f}",
        )


if __name__ == "__main__":
    main()
