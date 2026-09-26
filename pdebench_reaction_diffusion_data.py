#!/usr/bin/env python3


import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = torch.tensor([1e-3, 5e-3, 5e-3], dtype=torch.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def physical_params(a):
    base = BASE.to(device=a.device, dtype=a.dtype)[None, :]
    p = base * torch.pow(torch.tensor(2.0, device=a.device, dtype=a.dtype), a)
    return p[:, 0, None, None], p[:, 1, None, None], p[:, 2, None, None]


def lap_neumann(x, dx):
    z = F.pad(x[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    c = z[:, 1:-1, 1:-1]
    return (
        z[:, 1:-1, 2:] + z[:, 1:-1, :-2]
        + z[:, 2:, 1:-1] + z[:, :-2, 1:-1]
        - 4.0 * c
    ) / (dx * dx)


def rhs(state, a, dx):
    u, v = state[:, 0], state[:, 1]
    Du, Dv, k = physical_params(a)
    du = u - u**3 - k - v + Du * lap_neumann(u, dx)
    dv = u - v + Dv * lap_neumann(v, dx)
    return torch.stack([du, dv], dim=1)


@torch.no_grad()
def solve(x0, a, total_time=5.0, dt=0.02, domain_length=2.0):
    nsteps = int(math.ceil(total_time / dt))
    h = total_time / nsteps
    dx = domain_length / x0.shape[-1]
    x = x0.clone()
    
    for _ in range(nsteps):
        k1 = rhs(x, a, dx)
        k2 = rhs(x + h * k1, a, dx)
        x = x + 0.5 * h * (k1 + k2)
        if not torch.isfinite(x).all():
            raise RuntimeError("Non-finite PDE state. Reduce --dt.")
    return x


def anchor_grid(device):
    z = torch.tensor([-1.0, 0.0, 1.0], device=device)
    return torch.cartesian_prod(z, z, z).reshape(-1, 3)


def sample_anchor(n, g, device):
    A = anchor_grid(device)
    idx = torch.randint(0, len(A), (n,), generator=g, device=device)
    return A[idx]


def sample_box(n, lo, hi, g, device):
    return lo + (hi - lo) * torch.rand(n, 3, generator=g, device=device)


def sample_shell(n, inner, outer, g, device):
    chunks = []
    left = n
    while left:
        m = max(256, left * 2)
        z = 2 * outer * torch.rand(m, 3, generator=g, device=device) - outer
        r = z.abs().amax(dim=1)
        keep = z[(r >= inner) & (r <= outer)]
        take = min(left, len(keep))
        if take:
            chunks.append(keep[:take])
            left -= take
    return torch.cat(chunks, dim=0)


def sample_a(n, split, g, device):
    if split in ("train", "test_seen"):
        return sample_anchor(n, g, device)
    if split == "test_id":
        return sample_box(n, -1.0, 1.0, g, device)
    if split == "test_ood_near":
        return sample_shell(n, 1.10, 1.30, g, device)
    if split == "test_ood_far":
        return sample_shell(n, 1.40, 1.70, g, device)
    if split == "response_collocation":
        return sample_box(n, -1.30, 1.30, g, device)
    raise ValueError(split)


def unit_dirs(n, g, device):
    r = torch.randn(n, 3, generator=g, device=device)
    return r / torch.linalg.vector_norm(r, dim=1, keepdim=True).clamp_min(1e-8)


def run_chunks(x0_cpu, a_cpu, device, batch, **solver_kw):
    out = []
    for s in range(0, len(x0_cpu), batch):
        e = min(len(x0_cpu), s + batch)
        out.append(solve(x0_cpu[s:e].to(device), a_cpu[s:e].to(device), **solver_kw).cpu())
    return torch.cat(out, dim=0).float()


def endpoint_split(n, split, seed, grid, device, **solver_kw):
    g = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.randn(n, 2, grid, grid, generator=g, device=device)
    a = sample_a(n, split, g, device)
    xT = solve(x0, a, **solver_kw)
    return {"x0": x0.cpu().float(), "a": a.cpu().float(), "xT": xT.cpu().float(), "split": split}


def directional_truth(x0, a, seed, device, batch, fd_delta, finite_delta, **solver_kw):
    g = torch.Generator(device=device).manual_seed(seed)
    r = unit_dirs(len(x0), g, device).cpu().float()
    yp = run_chunks(x0, a + fd_delta * r, device, batch, **solver_kw)
    ym = run_chunks(x0, a - fd_delta * r, device, batch, **solver_kw)
    jv = (yp - ym) / (2.0 * fd_delta)
    y0 = run_chunks(x0, a, device, batch, **solver_kw)
    ybig = run_chunks(x0, a + finite_delta * r, device, batch, **solver_kw)
    return {
        "direction": r,
        "Jv_star": jv.float(),
        "finite_response": (ybig - y0).float(),
    }


def make_anchor_response(train, n, seed, device, batch, fd_delta, finite_delta, **solver_kw):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(len(train["x0"]), generator=g)[:min(n, len(train["x0"]))]
    x0, a = train["x0"][idx].clone(), train["a"][idx].clone()
    d = directional_truth(x0, a, seed + 1, device, batch, fd_delta, finite_delta, **solver_kw)
    return {
        "x0": x0, "a": a, **d,
        "response_only": True,
        "endpoint_labels_included": False,
        "source": "PDEBench-2D-diffusion-reaction|anchor-response",
    }


def make_collocation(n, seed, grid, device, batch, fd_delta, finite_delta, **solver_kw):
    g = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.randn(n, 2, grid, grid, generator=g, device=device).cpu()
    a = sample_a(n, "response_collocation", g, device).cpu()
    d = directional_truth(x0, a, seed + 1, device, batch, fd_delta, finite_delta, **solver_kw)
    return {
        "x0": x0.float(), "a": a.float(), **d,
        "response_only": True,
        "endpoint_labels_included": False,
        "source": "PDEBench-2D-diffusion-reaction|continuous-response-collocation",
    }


def normalization(train):
    z = torch.cat([train["x0"], train["xT"]], dim=0)
    mean = z.mean(dim=(0, 2, 3), keepdim=True)
    std = z.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    return mean.float(), std.float()


def normalize_endpoint(obj, mean, std):
    obj["x0_norm"] = (obj["x0"] - mean) / std
    obj["xT_norm"] = (obj["xT"] - mean) / std
    if "Jv_star" in obj:
        obj["Jv_star_norm"] = obj["Jv_star"] / std
        obj["finite_response_norm"] = obj["finite_response"] / std


def normalize_response(obj, mean, std):
    obj["x0_norm"] = (obj["x0"] - mean) / std
    obj["Jv_star_norm"] = obj["Jv_star"] / std
    obj["finite_response_norm"] = obj["finite_response"] / std


def fd_check(grid, device, seed, **solver_kw):
    g = torch.Generator(device=device).manual_seed(seed)
    n = 24
    x0 = torch.randn(n, 2, grid, grid, generator=g, device=device)
    a = sample_box(n, -1.0, 1.0, g, device)
    r = unit_dirs(n, g, device)
    def cfd(eps):
        return (solve(x0, a + eps*r, **solver_kw) - solve(x0, a - eps*r, **solver_kw)) / (2*eps)
    j1, j2 = cfd(0.04), cfd(0.02)
    num = torch.linalg.vector_norm((j1-j2).reshape(n, -1), dim=1)
    den = torch.linalg.vector_norm(j2.reshape(n, -1), dim=1).clamp_min(1e-8)
    rel = num / den
    return {"mean_relative_difference": float(rel.mean().cpu()), "max_relative_difference": float(rel.max().cpu())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/pdebench_reaction_diffusion_data")
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--total-time", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--n-anchor-response", type=int, default=1000)
    p.add_argument("--n-response-collocation", type=int, default=2000)
    p.add_argument("--fd-delta", type=float, default=0.03)
    p.add_argument("--finite-delta", type=float, default=0.25)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    solver_kw = dict(total_time=args.total_time, dt=args.dt, domain_length=2.0)

    print("Device:", device)
    print("Grid:", args.grid, "x", args.grid, "state_dim=", 2*args.grid*args.grid)
    print("Intervention: a=[log2(Du/Du0), log2(Dv/Dv0), log2(k/k0)]")
    print("Running finite-difference consistency check...")
    check = fd_check(args.grid, device, args.seed+99001, **solver_kw)
    print(json.dumps(check, indent=2))

    specs = {
        "train": (args.n_train, args.seed+101),
        "test_seen": (args.n_test, args.seed+202),
        "test_id": (args.n_test, args.seed+303),
        "test_ood_near": (args.n_test, args.seed+404),
        "test_ood_far": (args.n_test, args.seed+505),
    }
    data = {}
    for split, (n, seed) in specs.items():
        print("Generating", split, "N=", n)
        data[split] = endpoint_split(n, split, seed, args.grid, device, **solver_kw)

    for i, split in enumerate(["test_seen", "test_id", "test_ood_near", "test_ood_far"]):
        print("Response truth:", split)
        data[split].update(directional_truth(
            data[split]["x0"], data[split]["a"], args.seed+1001+i,
            device, args.batch_size, args.fd_delta, args.finite_delta, **solver_kw
        ))

    print("Generating anchor-response set...")
    anchor = make_anchor_response(
        data["train"], args.n_anchor_response, args.seed+2001,
        device, args.batch_size, args.fd_delta, args.finite_delta, **solver_kw
    )
    print("Generating response-collocation set...")
    colloc = make_collocation(
        args.n_response_collocation, args.seed+3001, args.grid,
        device, args.batch_size, args.fd_delta, args.finite_delta, **solver_kw
    )

    mean, std = normalization(data["train"])
    for obj in data.values():
        normalize_endpoint(obj, mean, std)
    normalize_response(anchor, mean, std)
    normalize_response(colloc, mean, std)

    for split, obj in data.items():
        torch.save(obj, out/f"{split}.pt")
    torch.save(anchor, out/"anchor_response.pt")
    torch.save(colloc, out/"response_collocation.pt")

    meta = {
        "benchmark": "PDEBench 2D diffusion-reaction parameter-swept reduced-grid benchmark",
        "pde": {
            "u_t": "u-u^3-k-v+Du*Laplacian(u)",
            "v_t": "u-v+Dv*Laplacian(v)",
            "boundary": "zero-flux Neumann",
        },
        "pdebench_defaults": {"Du": 1e-3, "Dv": 5e-3, "k": 5e-3, "T": 5.0, "grid": 128},
        "workshop_grid": args.grid,
        "state_shape": [2, args.grid, args.grid],
        "state_dim": 2*args.grid*args.grid,
        "intervention_dim": 3,
        "parameter_mapping": "physical=default*2**a",
        "train_anchor_values": [-1.0, 0.0, 1.0],
        "test_id_support": [-1.0, 1.0],
        "response_collocation_support": [-1.30, 1.30],
        "near_ood_max_abs": [1.10, 1.30],
        "far_ood_max_abs": [1.40, 1.70],
        "response_semantics": "deterministic pathwise directional field response J*v",
        "fd_delta": args.fd_delta,
        "finite_delta": args.finite_delta,
        "fd_check": check,
        "state_mean": mean.reshape(-1).tolist(),
        "state_std": std.reshape(-1).tolist(),
        "sizes": {
            "train": args.n_train,
            "test_per_split": args.n_test,
            "anchor_response": len(anchor["x0"]),
            "response_collocation": len(colloc["x0"]),
        },
    }
    json.dump(meta, open(out/"metadata.json", "w"), indent=2)

    print("Saved:", out.resolve())
    for split in ["test_seen", "test_id", "test_ood_near", "test_ood_far"]:
        q = data[split]["Jv_star_norm"].reshape(args.n_test, -1)
        print(split, "mean ||J*v|| =", float(torch.linalg.vector_norm(q, dim=1).mean()))


if __name__ == "__main__":
    main()
