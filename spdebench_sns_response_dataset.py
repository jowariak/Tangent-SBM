#!/usr/bin/env python3
"""
SPDEBench stochastic Navier--Stokes intervention-response dataset generator.

Uses the official SPDEBench 2D stochastic Navier--Stokes solver. The intervention is
    a = log2(nu / nu0),   nu(a) = nu0 * 2**a
with nominal nu0=1e-4. The physical Q-Wiener path remains latent: models see only
(x0, a), so repeated simulator calls define p(x_T | x0, a).

Modes:
  pilot     viscosity stability sweep
  generate  full endpoint/response/evaluation dataset

Expected SPDEBench checkout:
  <root>/data_gen/src/generator_sns.py
  <root>/data_gen/src/random_forcing.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


@dataclass
class SimConfig:
    # Official SPDEBench NS defaults.
    nu0: float = 1e-4
    alpha: float = 3.0
    tau: float = 3.0
    alpha_Q: float = 0.005
    kappa: int = 10
    sigma: float = 0.005
    truncation: int = 128
    s: int = 64
    T: float = 1.0
    delta_t: float = 1e-3

    # Response benchmark design.
    eps_a: float = 0.03
    finite_delta_a: float = 0.25


ANCHOR_A = [-1.0, 0.0, 1.0]
ID_A = [-0.75, -0.25, 0.25, 0.75]
NEAR_A = [-1.25, 1.25]
FAR_A = [-1.5, 1.5]
PILOT_A = sorted(set(ANCHOR_A + ID_A + NEAR_A + FAR_A))


def log(*args):
    print(*args, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def load_spdebench(root: Path):
    root = root.resolve()
    required = [
        root / "data_gen" / "src" / "generator_sns.py",
        root / "data_gen" / "src" / "random_forcing.py",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing SPDEBench files:\n  " + "\n  ".join(map(str, missing))
            + "\nClone https://github.com/DeepIntoStreams/SPDE_hackathon "
              "and pass its root with --spdebench-root."
        )
    sys.path.insert(0, str(root))
    from data_gen.src.generator_sns import navier_stokes_2d
    from data_gen.src.random_forcing import GaussianRF
    return navier_stokes_2d, GaussianRF


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def capture_rng_state(device: torch.device):
    out = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng_state(state, device: torch.device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def nu_from_a(a: torch.Tensor, nu0: float) -> torch.Tensor:
    a = a.reshape(-1)
    two = torch.tensor(2.0, device=a.device, dtype=a.dtype)
    return (nu0 * torch.pow(two, a)).view(-1, 1, 1)


def add_channel(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(1)


def cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().contiguous()


def stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.float()
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


class SNSimulator:
    def __init__(self, cfg, navier_stokes_2d, GaussianRF, device):
        self.cfg = cfg
        self.solver = navier_stokes_2d
        self.device = device
        self.grf = GaussianRF(2, cfg.s, alpha=cfg.alpha, tau=cfg.tau, device=device)

        t = torch.linspace(0, 1, cfg.s + 1, device=device)[:-1]
        X, Y = torch.meshgrid(t, t, indexing="ij")
        self.f = 0.1 * (
            torch.sin(2 * math.pi * (X + Y))
            + torch.cos(2 * math.pi * (X + Y))
        )
        self.stochastic_forcing = {
            "alpha": cfg.alpha_Q,
            "kappa": cfg.kappa,
            "sigma": cfg.sigma,
            "truncation": cfg.truncation,
        }

    def sample_initial_conditions(self, n: int):
        # Matches the varying-u0 construction in the official generator.
        w_star = self.grf.sample(1)
        return w_star + self.grf.sample(n), w_star

    @torch.no_grad()
    def endpoint(self, w0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        w0 = w0.to(self.device)
        a = a.to(self.device).reshape(-1)
        if len(a) != len(w0):
            raise ValueError("x0/a batch mismatch")
        visc = nu_from_a(a, self.cfg.nu0)
        sol, _, _ = self.solver(
            [1, 1], w0, self.f, visc,
            self.cfg.T, self.cfg.delta_t, 1,
            self.stochastic_forcing,
        )
        return sol[..., -1]

    @torch.no_grad()
    def endpoint_batched(self, w0, a, batch_size):
        out = []
        for s in range(0, len(w0), batch_size):
            e = min(s + batch_size, len(w0))
            log(f"    simulator batch {s}:{e}/{len(w0)}")
            out.append(self.endpoint(w0[s:e], a[s:e]).cpu())
        return torch.cat(out, 0)

    @torch.no_grad()
    def centered_fd(self, x0, a, eps, mc, batch_size):
        """Common-noise centered FD. Returns mean [C,H,W], samples [C,mc,H,W]."""
        C = len(x0)
        xr = x0.repeat_interleave(mc, 0)
        ar = a.reshape(-1).repeat_interleave(mc)
        out = []
        for s in range(0, len(xr), batch_size):
            e = min(s + batch_size, len(xr))
            xb, ab = xr[s:e].to(self.device), ar[s:e].to(self.device)
            rng = capture_rng_state(self.device)
            yp = self.endpoint(xb, ab + eps)
            restore_rng_state(rng, self.device)
            ym = self.endpoint(xb, ab - eps)
            out.append(((yp - ym) / (2 * eps)).cpu())
            log(f"    centered-FD batch {s}:{e}/{len(xr)}")
        samples = torch.cat(out, 0).reshape(C, mc, self.cfg.s, self.cfg.s)
        return samples.mean(1), samples

    @torch.no_grad()
    def finite_response(self, x0, a, delta, mc, batch_size):
        """Common-noise E[X_T(a+delta)-X_T(a)]."""
        C = len(x0)
        xr = x0.repeat_interleave(mc, 0)
        ar = a.reshape(-1).repeat_interleave(mc)
        out = []
        for s in range(0, len(xr), batch_size):
            e = min(s + batch_size, len(xr))
            xb, ab = xr[s:e].to(self.device), ar[s:e].to(self.device)
            rng = capture_rng_state(self.device)
            y0 = self.endpoint(xb, ab)
            restore_rng_state(rng, self.device)
            y1 = self.endpoint(xb, ab + delta)
            out.append((y1 - y0).cpu())
            log(f"    finite-response batch {s}:{e}/{len(xr)}")
        samples = torch.cat(out, 0).reshape(C, mc, self.cfg.s, self.cfg.s)
        return samples.mean(1), samples


def run_pilot(sim, output_dir, seed, n_ics, reps, batch_size, max_abs_thr, growth_thr):
    log("\n=== STABILITY PILOT ===")
    set_seed(seed)
    x0_unique, _ = sim.sample_initial_conditions(n_ics)
    rows = []

    for av in PILOT_A:
        x0 = x0_unique.repeat_interleave(reps, 0)
        a = torch.full((len(x0),), av, device=sim.device, dtype=x0.dtype)
        nu = sim.cfg.nu0 * (2.0 ** av)
        log(f"\na={av:+.2f}  nu={nu:.6e}  n={len(x0)}")
        y = sim.endpoint_batched(x0, a, batch_size)
        finite = torch.isfinite(y)
        finite_fraction = float(finite.float().mean())

        if bool(finite.all()):
            max_abs = float(y.abs().max())
            x_rms = torch.sqrt((x0.cpu() ** 2).mean((1, 2))).clamp_min(1e-12)
            y_rms = torch.sqrt((y ** 2).mean((1, 2)))
            growth = y_rms / x_rms
            ens = 0.5 * (y ** 2).mean((1, 2))
            max_growth = float(growth.max())
            mean_growth = float(growth.mean())
            ens_mean = float(ens.mean())
            ens_std = float(ens.std(unbiased=False))
        else:
            max_abs = max_growth = mean_growth = float("inf")
            ens_mean = ens_std = float("nan")

        stable = finite_fraction == 1.0 and max_abs < max_abs_thr and max_growth < growth_thr
        row = {
            "a": av,
            "nu": nu,
            "finite_fraction": finite_fraction,
            "max_abs_terminal": max_abs,
            "mean_rms_growth": mean_growth,
            "max_rms_growth": max_growth,
            "terminal_enstrophy_mean": ens_mean,
            "terminal_enstrophy_std": ens_std,
            "stable_flag": stable,
        }
        rows.append(row)
        log(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stability_pilot.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {
        "seed": seed,
        "pilot_a": PILOT_A,
        "n_initial_conditions": n_ics,
        "replicates_per_ic": reps,
        "max_abs_threshold": max_abs_thr,
        "rms_growth_threshold": growth_thr,
        "all_stable": all(r["stable_flag"] for r in rows),
        "rows": rows,
    }
    (output_dir / "stability_pilot.json").write_text(json.dumps(summary, indent=2))
    log("\nALL STABLE:", summary["all_stable"])
    return summary["all_stable"]


def build_conditions(x0_bank: torch.Tensor, a_values: Iterable[float]):
    xs, aa = [], []
    for av in a_values:
        xs.append(x0_bank)
        aa.append(torch.full((len(x0_bank),), av, device=x0_bank.device, dtype=x0_bank.dtype))
    return torch.cat(xs, 0), torch.cat(aa, 0)


def save_response_train(path, x0, a, jmean, jsamples, cfg, kind):
    torch.save({
        "benchmark": "spdebench_sns_response",
        "kind": kind,
        "response_only": True,
        "endpoint_labels_included": False,
        "x0": add_channel(cpu(x0)),
        "a": cpu(a.reshape(-1, 1)),
        "direction": torch.ones((len(x0), 1), dtype=torch.float32),
        "Jv_star": add_channel(cpu(jmean)),
        "Jv_samples": cpu(jsamples).unsqueeze(2),
        "eps_a": cfg.eps_a,
        "response_target": "conditional_mean_common_noise_centered_fd",
    }, path)


def generate_eval_split(sim, name, eval_x0, a_values, endpoint_mc, response_mc,
                        finite_mc, batch_size, output_dir):
    log(f"\n=== EVAL {name} ===")
    x0, a = build_conditions(eval_x0, a_values)
    C = len(x0)

    xr = x0.repeat_interleave(endpoint_mc, 0)
    ar = a.repeat_interleave(endpoint_mc)
    y = sim.endpoint_batched(xr, ar, batch_size)
    ys = y.reshape(C, endpoint_mc, sim.cfg.s, sim.cfg.s)

    jmean, js = sim.centered_fd(x0, a, sim.cfg.eps_a, response_mc, batch_size)
    fmean, fs = sim.finite_response(x0, a, sim.cfg.finite_delta_a, finite_mc, batch_size)
    ens = 0.5 * (ys ** 2).mean((-2, -1))

    torch.save({
        "benchmark": "spdebench_sns_response",
        "split": name,
        "x0": add_channel(cpu(x0)),
        "a": cpu(a.reshape(-1, 1)),
        "nu": cpu(nu_from_a(a, sim.cfg.nu0).reshape(-1, 1)),
        "xT_samples": cpu(ys).unsqueeze(2),
        "endpoint_mc": endpoint_mc,
        "direction": torch.ones((C, 1), dtype=torch.float32),
        "Jv_star": add_channel(cpu(jmean)),
        "Jv_samples": cpu(js).unsqueeze(2),
        "response_mc": response_mc,
        "finite_response_star": add_channel(cpu(fmean)),
        "finite_response_samples": cpu(fs).unsqueeze(2),
        "finite_mc": finite_mc,
        "eps_a": sim.cfg.eps_a,
        "finite_delta_a": sim.cfg.finite_delta_a,
        "enstrophy_samples": cpu(ens),
    }, output_dir / f"eval_{name}.pt")


def generate_dataset(sim, root, output_dir, seed, train_ics, endpoint_reps,
                     anchor_ics_per_a, collocation_count, sens_mc, eval_ics,
                     eval_endpoint_mc, eval_response_mc, eval_finite_mc, batch_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    log("\n=== INITIAL CONDITIONS ===")
    w_star = sim.grf.sample(1)
    all_x0 = w_star + sim.grf.sample(train_ics + eval_ics)
    train_x0, eval_x0 = all_x0[:train_ics], all_x0[train_ics:]
    torch.save({
        "w_star": cpu(w_star),
        "train_x0": add_channel(cpu(train_x0)),
        "eval_x0": add_channel(cpu(eval_x0)),
    }, output_dir / "initial_conditions.pt")

    log("\n=== ENDPOINT TRAIN ===")
    bx, ba = build_conditions(train_x0, ANCHOR_A)
    xtrain = bx.repeat_interleave(endpoint_reps, 0)
    atrain = ba.repeat_interleave(endpoint_reps)
    ytrain = sim.endpoint_batched(xtrain, atrain, batch_size)
    torch.save({
        "benchmark": "spdebench_sns_response",
        "x0": add_channel(cpu(xtrain)),
        "a": cpu(atrain.reshape(-1, 1)),
        "nu": cpu(nu_from_a(atrain, sim.cfg.nu0).reshape(-1, 1)),
        "xT": add_channel(cpu(ytrain)),
        "anchor_a": ANCHOR_A,
        "endpoint_reps": endpoint_reps,
    }, output_dir / "endpoint_train.pt")

    if anchor_ics_per_a > train_ics:
        raise ValueError("anchor_ics_per_a > train_ics")
    perm = torch.randperm(train_ics, device=sim.device)

    log("\n=== ANCHOR RESPONSE ===")
    ax, aa = build_conditions(train_x0[perm[:anchor_ics_per_a]], ANCHOR_A)
    aj, ajs = sim.centered_fd(ax, aa, sim.cfg.eps_a, sens_mc, batch_size)
    save_response_train(output_dir / "anchor_response.pt", ax, aa, aj, ajs, sim.cfg, "anchor_response")

    log("\n=== RESPONSE COLLOCATION ===")
    nbase = min(train_ics, max(1, math.ceil(collocation_count / 3)))
    ids = perm[:nbase].repeat(math.ceil(collocation_count / nbase))[:collocation_count]
    cx = train_x0[ids]
    ca = 2 * torch.rand(collocation_count, device=sim.device) - 1
    cj, cjs = sim.centered_fd(cx, ca, sim.cfg.eps_a, sens_mc, batch_size)
    save_response_train(output_dir / "response_collocation.pt", cx, ca, cj, cjs, sim.cfg, "response_collocation")

    generate_eval_split(sim, "seen", eval_x0, ANCHOR_A, eval_endpoint_mc,
                        eval_response_mc, eval_finite_mc, batch_size, output_dir)
    generate_eval_split(sim, "id", eval_x0, ID_A, eval_endpoint_mc,
                        eval_response_mc, eval_finite_mc, batch_size, output_dir)
    generate_eval_split(sim, "ood_near", eval_x0, NEAR_A, eval_endpoint_mc,
                        eval_response_mc, eval_finite_mc, batch_size, output_dir)
    generate_eval_split(sim, "ood_far", eval_x0, FAR_A, eval_endpoint_mc,
                        eval_response_mc, eval_finite_mc, batch_size, output_dir)

    state_stats = torch.cat([xtrain.cpu(), ytrain.cpu()], 0)
    meta = {
        "benchmark": "spdebench_sns_response",
        "spdebench_root": str(root.resolve()),
        "spdebench_git_commit": git_commit(root),
        "sim_config": asdict(sim.cfg),
        "intervention": {
            "name": "log2_viscosity",
            "definition": "a = log2(nu / nu0)",
            "nu0": sim.cfg.nu0,
            "anchor_a": ANCHOR_A,
            "id_a": ID_A,
            "near_ood_a": NEAR_A,
            "far_ood_a": FAR_A,
        },
        "counts": {
            "train_initial_conditions": train_ics,
            "endpoint_reps": endpoint_reps,
            "endpoint_training_pairs": int(len(xtrain)),
            "anchor_response_conditions": int(len(ax)),
            "collocation_response_conditions": int(len(cx)),
            "sensitivity_mc": sens_mc,
            "eval_initial_conditions": eval_ics,
            "eval_endpoint_mc": eval_endpoint_mc,
            "eval_response_mc": eval_response_mc,
            "eval_finite_mc": eval_finite_mc,
        },
        "normalization": {
            "state_scalar": stats(state_stats),
            "a_train_scalar": stats(atrain.cpu()),
        },
        "response_target": {
            "type": "conditional_mean_directional_derivative",
            "direction": 1.0,
            "fd_epsilon_in_a": sim.cfg.eps_a,
            "common_noise": True,
        },
        "finite_response": {
            "delta_a": sim.cfg.finite_delta_a,
            "common_noise": True,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    log("\n=== COMPLETE ===")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spdebench-root", type=Path, default=Path("SPDE_hackathon"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/spdebench_sns_response"))
    p.add_argument("--mode", choices=["pilot", "generate"], default="pilot")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)

    p.add_argument("--pilot-ics", type=int, default=4)
    p.add_argument("--pilot-reps", type=int, default=2)
    p.add_argument("--pilot-max-abs-threshold", type=float, default=1e3)
    p.add_argument("--pilot-rms-growth-threshold", type=float, default=100.0)
    p.add_argument("--skip-pilot", action="store_true")

    p.add_argument("--train-ics", type=int, default=100)
    p.add_argument("--endpoint-reps", type=int, default=4)
    p.add_argument("--anchor-ics-per-a", type=int, default=36)
    p.add_argument("--collocation-count", type=int, default=216)
    p.add_argument("--sens-mc", type=int, default=4)
    p.add_argument("--eval-ics", type=int, default=16)
    p.add_argument("--eval-endpoint-mc", type=int, default=32)
    p.add_argument("--eval-response-mc", type=int, default=16)
    p.add_argument("--eval-finite-mc", type=int, default=16)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    log("device =", device)
    solver, GaussianRF = load_spdebench(args.spdebench_root)
    cfg = SimConfig()
    sim = SNSimulator(cfg, solver, GaussianRF, device)

    if args.quick:
        args.pilot_ics = 1
        args.pilot_reps = 1
        args.train_ics = 4
        args.endpoint_reps = 1
        args.anchor_ics_per_a = 2
        args.collocation_count = 4
        args.sens_mc = 1
        args.eval_ics = 1
        args.eval_endpoint_mc = 2
        args.eval_response_mc = 1
        args.eval_finite_mc = 1
        args.batch_size = min(args.batch_size, 4)

    if args.mode == "pilot":
        ok = run_pilot(sim, args.output_dir, args.seed, args.pilot_ics,
                       args.pilot_reps, args.batch_size,
                       args.pilot_max_abs_threshold,
                       args.pilot_rms_growth_threshold)
        raise SystemExit(0 if ok else 2)

    if not args.skip_pilot:
        ok = run_pilot(sim, args.output_dir, args.seed, args.pilot_ics,
                       args.pilot_reps, args.batch_size,
                       args.pilot_max_abs_threshold,
                       args.pilot_rms_growth_threshold)
        if not ok:
            raise RuntimeError("Pilot failed; inspect stability_pilot.csv before full generation")

    generate_dataset(
        sim, args.spdebench_root, args.output_dir, args.seed,
        args.train_ics, args.endpoint_reps, args.anchor_ics_per_a,
        args.collocation_count, args.sens_mc, args.eval_ics,
        args.eval_endpoint_mc, args.eval_response_mc, args.eval_finite_mc,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
