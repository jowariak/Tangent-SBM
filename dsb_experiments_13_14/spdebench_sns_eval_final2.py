#!/usr/bin/env python3
"""
spdebench_sns_eval_final2.py

ONE-SHOT evaluation of the frozen stochastic-NS models on FINAL2.

Frozen protocol:
  reference_sigma = 1.4
  Conditional final = IMF 5
  Tangent shared through IMF 3, constrained IMF 4--5
  lambda_sens = 1000
  sens_batch_size = 8
  sens_pairs = 1
  sens_every = 10
  seeds = 32,42,52

No training. No validation. No old test/final files are loaded.

In addition to endpoint-distribution and finite-response metrics, this script
predeclares response diagnostics:
  * raw J RMSE
  * zero-response RMSE
  * mean per-condition EJ
  * mean per-condition cosine similarity
  * predicted/true response-norm ratio

Response diagnostics use 64 model rollouts/condition by default to reduce MC
noise in the estimated conditional-mean tangent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import spdebench_sns_conditional_dsbm_v2 as base


FINAL2_FILES = {
    "final2_seen": "final2_seen.pt",
    "final2_id": "final2_id.pt",
    "final2_ood_near": "final2_ood_near.pt",
    "final2_ood_far": "final2_ood_far.pt",
}


def sigma_tag(x):
    return str(float(x)).replace(".", "p")


def lambda_tag(x):
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def parse_seeds(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def mean_std(vals):
    x = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
    }


def load_final2_data(data_dir, mean, std):
    manifest_path = data_dir / "final2_eval_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Missing final2_eval_manifest.json. Generate FINAL2 first."
        )
    manifest = json.load(open(manifest_path))
    if not bool(manifest.get("locked", False)):
        raise RuntimeError("FINAL2 manifest is not locked=True")

    out = {}
    for split, fn in FINAL2_FILES.items():
        path = data_dir / fn
        o = base.safe_load(path)
        if not bool(o.get("locked_final_evaluation_only", False)):
            raise RuntimeError(f"{path} is not marked locked_final_evaluation_only")
        if bool(o.get("model_selection_allowed", True)):
            raise RuntimeError(f"{path} incorrectly allows model selection")

        required = [
            "x0", "a", "xT_samples", "direction", "Jv_star",
            "finite_response_star", "enstrophy_samples",
        ]
        missing = [k for k in required if k not in o]
        if missing:
            raise RuntimeError(f"{path} missing {missing}")

        out[split] = {
            "x0": base.normalize_state(o["x0"].float(), mean, std),
            "x0_raw": o["x0"].float(),
            "a": o["a"].float(),
            "xT_samples_raw": o["xT_samples"].float(),
            "direction": o["direction"].float(),
            "Jv_star_raw": o["Jv_star"].float(),
            "finite_response_raw": o["finite_response_star"].float(),
            "enstrophy_samples": o["enstrophy_samples"].float(),
        }
    return manifest, out


def load_model(cp, device, expected_sigma):
    ck = base.safe_load(cp)
    cfg = base.Config(**ck["config"])
    if not math.isclose(
        float(cfg.reference_sigma), float(expected_sigma),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{cp}: sigma={cfg.reference_sigma}, expected={expected_sigma}"
        )
    model = base.ConditionalFieldDSBM(cfg, device)
    model.load_state_dict(ck["model"])
    model.net_f.eval()
    model.net_b.eval()
    return model, cfg, ck


def evaluate_standard(model, cfg, data, device, mean, std, seed_base):
    res = {}
    for j, split in enumerate(FINAL2_FILES):
        base.set_seed(int(seed_base + 1000 * j))
        res[split] = base.evaluate_split(
            split, model, data[split], cfg, device,
            mean, std, lambda *args: None
        )
    return res


def predict_mean_j(model, split_data, cfg, device, state_std, mc, seed):
    base.set_seed(int(seed))
    chunks = []
    n = split_data["x0"].shape[0]

    for s in range(0, n, cfg.eval_batch_size):
        e = min(n, s + cfg.eval_batch_size)
        x0 = split_data["x0"][s:e].to(device)
        a = split_data["a"][s:e].to(device)
        d = split_data["direction"][s:e].to(device)

        draws = []
        with torch.enable_grad():
            for _ in range(mc):
                draws.append(
                    model.tangent_direction_rollout(x0, a, d).detach().cpu()
                )
        chunks.append(torch.stack(draws, 0).mean(0))

    return torch.cat(chunks, 0) * float(state_std)


def response_diag(pred, true):
    p = pred.reshape(pred.shape[0], -1)
    t = true.reshape(true.shape[0], -1)
    err = p - t

    target_rms = float(torch.sqrt(torch.mean(t ** 2)))
    rmse = float(torch.sqrt(torch.mean(err ** 2)))

    pnorm = torch.linalg.vector_norm(p, dim=1)
    tnorm = torch.linalg.vector_norm(t, dim=1)
    enorm = torch.linalg.vector_norm(err, dim=1)

    rel = enorm / tnorm.clamp_min(1e-12)
    cos = torch.sum(p * t, dim=1) / (pnorm * tnorm).clamp_min(1e-12)
    ratio = pnorm / tnorm.clamp_min(1e-12)

    return {
        "target_rms_raw": target_rms,
        "zero_response_rmse_raw": target_rms,
        "model_rmse_raw": rmse,
        "global_relative_rmse": rmse / max(target_rms, 1e-12),
        "mean_condition_EJ": float(rel.mean()),
        "median_condition_EJ": float(rel.median()),
        "mean_condition_cosine": float(cos.mean()),
        "median_condition_cosine": float(cos.median()),
        "mean_pred_true_norm_ratio": float(ratio.mean()),
        "median_pred_true_norm_ratio": float(ratio.median()),
    }


def aggregate(per_seed, method, section):
    if section == "standard":
        keys = [
            "field_mean_rel_l2",
            "spread_rel_l2",
            "energy_distance_per_sqrt_pixel",
            "sliced_wasserstein_raw",
            "enstrophy_w1",
            "directional_j_rel_error",
            "directional_j_rmse_raw",
            "finite_response_rel_l2",
            "finite_response_rmse_raw",
        ]
    else:
        keys = [
            "model_rmse_raw",
            "global_relative_rmse",
            "mean_condition_EJ",
            "mean_condition_cosine",
            "mean_pred_true_norm_ratio",
        ]

    out = {}
    seeds = sorted(int(s) for s in per_seed)
    for split in FINAL2_FILES:
        out[split] = {}
        for k in keys:
            vals = [
                per_seed[str(seed)][method][section][split][k]
                for seed in seeds
            ]
            out[split][k] = mean_std(vals)
    return out


def pct_reduction(old, new):
    return 100.0 * (old - new) / old


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("runs/spdebench_sns_response"))
    p.add_argument(
        "--conditional-root",
        type=Path,
        default=Path("runs/spdebench_sns_conditional_final"),
    )
    p.add_argument(
        "--tangent-root",
        type=Path,
        default=Path("runs/spdebench_sns_tangent_stability_b8"),
    )
    p.add_argument("--reference-sigma", type=float, default=1.4)
    p.add_argument("--lambda-sens", type=float, default=1000.0)
    p.add_argument("--conditional-imf", type=int, default=5)
    p.add_argument("--tangent-imf", type=int, default=5)
    p.add_argument("--seeds", default="32,42,52")
    p.add_argument("--response-mc", type=int, default=64)
    p.add_argument("--eval-seed-base", type=int, default=880000)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("runs/spdebench_sns_FINAL2_comparison.json"),
    )
    args = p.parse_args()

    device = base.resolve_device(args.device)
    seeds = parse_seeds(args.seeds)

    metadata = json.load(open(args.data_dir / "metadata.json"))
    mean, std = base.normalization_from_metadata(metadata)
    manifest, data = load_final2_data(args.data_dir, mean, std)

    print("=" * 92)
    print("FINAL2 -- ONE-SHOT FROZEN EVALUATION")
    print("NO TRAINING / NO VALIDATION / NO MODEL SELECTION")
    print("=" * 92)
    print("generation seed:", manifest["generation_seed"])
    print("reference_sigma:", args.reference_sigma)
    print("lambda_sens:", args.lambda_sens)
    print("Tangent sens_batch_size: 8 (frozen)")
    print("Tangent sens_pairs: 1 (frozen)")
    print("response diagnostic MC:", args.response_mc)
    print("seeds:", seeds)
    print()

    sig = sigma_tag(args.reference_sigma)
    lam = lambda_tag(args.lambda_sens)
    per_seed = {}

    for seed in seeds:
        cond_cp = (
            args.conditional_root / f"sigma_{sig}"
            / f"seed_{seed}" / f"imf_{args.conditional_imf}.pt"
        )
        tang_cp = (
            args.tangent_root / f"lambda_{lam}"
            / f"seed_{seed}" / f"imf_{args.tangent_imf}.pt"
        )
        if not cond_cp.exists():
            raise FileNotFoundError(cond_cp)
        if not tang_cp.exists():
            raise FileNotFoundError(tang_cp)

        cm, ccfg, cck = load_model(cond_cp, device, args.reference_sigma)
        tm, tcfg, tck = load_model(tang_cp, device, args.reference_sigma)

        # Audit the frozen Tangent configuration when metadata are available.
        if float(tck.get("lambda_sens", args.lambda_sens)) != float(args.lambda_sens):
            raise RuntimeError(f"{tang_cp}: wrong lambda")
        if int(tck.get("fork_imf", 3)) != 3:
            raise RuntimeError(f"{tang_cp}: expected fork_imf=3")
        if "sens_batch_size" in tck and int(tck["sens_batch_size"]) != 8:
            raise RuntimeError(f"{tang_cp}: expected sens_batch_size=8")
        if "sens_pairs" in tck and int(tck["sens_pairs"]) != 1:
            raise RuntimeError(f"{tang_cp}: expected sens_pairs=1")

        for name in [
            "num_steps", "eval_mc", "eval_sens_mc", "eval_finite_mc",
            "eval_batch_size", "swd_projections", "finite_delta",
        ]:
            if getattr(ccfg, name) != getattr(tcfg, name):
                raise RuntimeError(
                    f"Seed {seed}: eval config mismatch {name}: "
                    f"{getattr(ccfg,name)} vs {getattr(tcfg,name)}"
                )

        seed_base = args.eval_seed_base + seed * 100000
        cstd = evaluate_standard(cm, ccfg, data, device, mean, std, seed_base)
        tstd = evaluate_standard(tm, tcfg, data, device, mean, std, seed_base)

        cdiag, tdiag = {}, {}
        for j, split in enumerate(FINAL2_FILES):
            # Same MC stream for Conditional and Tangent.
            diag_seed = seed_base + 50000 + j * 1000
            cpred = predict_mean_j(
                cm, data[split], ccfg, device, std,
                args.response_mc, diag_seed
            )
            tpred = predict_mean_j(
                tm, data[split], tcfg, device, std,
                args.response_mc, diag_seed
            )
            true = data[split]["Jv_star_raw"]
            cdiag[split] = response_diag(cpred, true)
            tdiag[split] = response_diag(tpred, true)

        per_seed[str(seed)] = {
            "conditional_checkpoint": str(cond_cp),
            "tangent_checkpoint": str(tang_cp),
            "conditional": {"standard": cstd, "response_diag": cdiag},
            "tangent": {"standard": tstd, "response_diag": tdiag},
        }

        print(f"================ SEED {seed} ================")
        for split in FINAL2_FILES:
            c, t = cstd[split], tstd[split]
            cd, td = cdiag[split], tdiag[split]
            print(
                f"{split:16s} | "
                f"mean {c['field_mean_rel_l2']:.4f}->{t['field_mean_rel_l2']:.4f} | "
                f"spread {c['spread_rel_l2']:.4f}->{t['spread_rel_l2']:.4f} | "
                f"EJ {cd['mean_condition_EJ']:.4f}->{td['mean_condition_EJ']:.4f} | "
                f"cos {cd['mean_condition_cosine']:.3f}->{td['mean_condition_cosine']:.3f} | "
                f"ratio {cd['mean_pred_true_norm_ratio']:.3f}->{td['mean_pred_true_norm_ratio']:.3f} | "
                f"finite {c['finite_response_rel_l2']:.4f}->{t['finite_response_rel_l2']:.4f}"
            )
        print()

    acs = aggregate(per_seed, "conditional", "standard")
    ats = aggregate(per_seed, "tangent", "standard")
    acd = aggregate(per_seed, "conditional", "response_diag")
    atd = aggregate(per_seed, "tangent", "response_diag")

    payload = {
        "protocol": {
            "FINAL2": True,
            "one_shot_frozen_evaluation": True,
            "no_training": True,
            "no_validation": True,
            "no_model_selection": True,
            "reference_sigma": args.reference_sigma,
            "lambda_sens": args.lambda_sens,
            "tangent_sens_batch_size": 8,
            "tangent_sens_pairs": 1,
            "tangent_sens_every": 10,
            "conditional_imf": args.conditional_imf,
            "tangent_fork_imf": 3,
            "tangent_imf": args.tangent_imf,
            "seeds": seeds,
            "response_diag_mc": args.response_mc,
        },
        "manifest": manifest,
        "per_seed": per_seed,
        "aggregate": {
            "conditional": {"standard": acs, "response_diag": acd},
            "tangent": {"standard": ats, "response_diag": atd},
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    def fs(d):
        return f"{d['mean']:.4f}±{d['std']:.4f}"

    print("=" * 92)
    print("FINAL2 AGGREGATE mean ± sample std")
    print("=" * 92)
    for split in FINAL2_FILES:
        cs, ts = acs[split], ats[split]
        cd, td = acd[split], atd[split]
        ej_red = pct_reduction(
            cd["mean_condition_EJ"]["mean"],
            td["mean_condition_EJ"]["mean"],
        )
        fin_red = pct_reduction(
            cs["finite_response_rel_l2"]["mean"],
            ts["finite_response_rel_l2"]["mean"],
        )
        print(
            f"{split:16s} | "
            f"mean {fs(cs['field_mean_rel_l2'])}->{fs(ts['field_mean_rel_l2'])} | "
            f"spread {fs(cs['spread_rel_l2'])}->{fs(ts['spread_rel_l2'])} | "
            f"EJ {fs(cd['mean_condition_EJ'])}->{fs(td['mean_condition_EJ'])} "
            f"({ej_red:.1f}% reduction) | "
            f"cos {fs(cd['mean_condition_cosine'])}->{fs(td['mean_condition_cosine'])} | "
            f"norm {fs(cd['mean_pred_true_norm_ratio'])}->{fs(td['mean_pred_true_norm_ratio'])} | "
            f"finite {fs(cs['finite_response_rel_l2'])}->{fs(ts['finite_response_rel_l2'])} "
            f"({fin_red:.1f}% reduction)"
        )

    print()
    print("Saved:", args.output)
    print("FINAL2 is now consumed. Do not retune from these results.")


if __name__ == "__main__":
    main()
