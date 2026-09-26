#!/usr/bin/env python3


from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import pdebench_reaction_diffusion_official as gen


def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def latin_hypercube(n: int, d: int, lo: float, hi: float, rng):
    out = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        jitter = rng.uniform(0.0, 1.0, size=n)
        q = (perm + jitter) / float(n)
        out[:, j] = lo + (hi - lo) * q
    return out


def nearest_anchor_linf(a):
    anchors = np.asarray(
        list(itertools.product([-1.0, 0.0, 1.0], repeat=3)),
        dtype=np.float64,
    )
    
    return np.min(
        np.max(np.abs(a[:, None, :] - anchors[None, :, :]), axis=2),
        axis=1,
    )


def sample_spread_points(n, lo, hi, min_anchor_dist, seed):
    rng = np.random.default_rng(seed)

    
    a = latin_hypercube(n, 3, lo, hi, rng)

    
    bad = nearest_anchor_linf(a) < min_anchor_dist
    rounds = 0
    while bad.any():
        rounds += 1
        repl = rng.uniform(lo, hi, size=(int(bad.sum()), 3))
        a[bad] = repl
        bad = nearest_anchor_linf(a) < min_anchor_dist
        if rounds > 1000:
            raise RuntimeError("Could not satisfy anchor-distance constraint.")

    return a


def collect_existing_seeds(source: Path):
    names = [
        "train.pt",
        "test_seen.pt",
        "test_id.pt",
        "test_ood_near.pt",
        "test_ood_far.pt",
        "anchor_response.pt",
        "response_collocation.pt",
        "response_eval_seen.pt",
        "response_eval_id.pt",
        "response_eval_ood_near.pt",
        "response_eval_ood_far.pt",
    ]
    seeds = []
    for name in names:
        p = source / name
        if not p.exists():
            continue
        obj = safe_load(p)
        if "pdebench_seed" in obj:
            vals = obj["pdebench_seed"]
            if torch.is_tensor(vals):
                seeds.extend([int(x) for x in vals.reshape(-1).tolist()])
    return seeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-dir",
        default="runs/pdebench_reaction_diffusion_official",
    )
    p.add_argument(
        "--out-dir",
        default="runs/pdebench_reaction_diffusion_spread_budget",
    )
    p.add_argument(
        "--pdebench-root",
        default="third_party/PDEBench",
    )
    p.add_argument("--n-extra", type=int, default=648)
    p.add_argument("--a-lo", type=float, default=-1.30)
    p.add_argument("--a-hi", type=float, default=1.30)
    p.add_argument(
        "--min-anchor-linf",
        type=float,
        default=0.15,
        help="Reject new a values within this L_inf distance of any {-1,0,1}^3 anchor.",
    )
    p.add_argument("--sample-seed", type=int, default=20260904)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    source = Path(args.source_dir)
    out = Path(args.out_dir)
    cache = out / "sim_cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    with open(source / "metadata.json", "r") as f:
        metadata = json.load(f)

    train = safe_load(source / "train.pt")
    original_n = int(train["x0"].shape[0])

    a_extra = sample_spread_points(
        args.n_extra,
        args.a_lo,
        args.a_hi,
        args.min_anchor_linf,
        args.sample_seed,
    )

    existing = collect_existing_seeds(source)
    seed0 = (max(existing) + 10000) if existing else 500000
    sim_seeds = np.arange(seed0, seed0 + args.n_extra, dtype=np.int64)

    specs = [
        (int(seed), a_extra[i])
        for i, seed in enumerate(sim_seeds)
    ]

    print("=" * 72)
    print("SPREAD EQUAL-BUDGET ENDPOINT CONTROL")
    print("=" * 72)
    print("Original train pairs:", original_n)
    print("Extra official simulator calls:", args.n_extra)
    print("a domain:", (args.a_lo, args.a_hi))
    print(
        "nearest-anchor L_inf:",
        "min=", float(nearest_anchor_linf(a_extra).min()),
        "mean=", float(nearest_anchor_linf(a_extra).mean()),
    )
    print("new simulator seed range:", (int(sim_seeds.min()), int(sim_seeds.max())))
    print()

    gen.simulate_many(
        specs,
        pdebench_root=args.pdebench_root,
        cache_dir=cache,
        workers=args.workers,
    )

    recs = [
        gen.load_cached(cache, int(sim_seeds[i]), a_extra[i])
        for i in range(args.n_extra)
    ]

    x0 = torch.stack([r["x0"] for r in recs], 0).float()
    xT = torch.stack([r["xT"] for r in recs], 0).float()
    a = torch.stack([r["a"] for r in recs], 0).float()
    physical = torch.stack([r["physical"] for r in recs], 0).float()
    pde_seed = torch.tensor([r["seed"] for r in recs], dtype=torch.long)

    norm = metadata["normalization"]
    mean = torch.tensor(
        norm["state_mean_per_channel"], dtype=torch.float32
    ).reshape(1, 2, 1, 1)
    std = torch.tensor(
        norm["state_std_per_channel"], dtype=torch.float32
    ).reshape(1, 2, 1, 1)

    extra = {
        "x0": x0,
        "xT": xT,
        "a": a,
        "physical_params": physical,
        "pdebench_seed": pde_seed,
        "x0_norm": (x0 - mean) / std,
        "xT_norm": (xT - mean) / std,
    }

    required = [
        "x0", "xT", "a", "physical_params",
        "pdebench_seed", "x0_norm", "xT_norm",
    ]
    augmented = {}
    for key in required:
        if key not in train:
            raise RuntimeError(f"Original train.pt missing key: {key}")
        augmented[key] = torch.cat([train[key], extra[key]], dim=0)

    augmented["spread_budget_control"] = True
    augmented["original_train_size"] = original_n
    augmented["extra_endpoint_pairs"] = args.n_extra

    torch.save(augmented, out / "train.pt")

    
    for name in [
        "test_seen.pt",
        "test_id.pt",
        "test_ood_near.pt",
        "test_ood_far.pt",
        "anchor_response.pt",
        "response_collocation.pt",
        "response_eval_seen.pt",
        "response_eval_id.pt",
        "response_eval_ood_near.pt",
        "response_eval_ood_far.pt",
    ]:
        shutil.copy2(source / name, out / name)

    meta2 = dict(metadata)
    meta2["spread_budget_control"] = {
        "original_train_pairs": original_n,
        "extra_endpoint_pairs": args.n_extra,
        "augmented_train_pairs": original_n + args.n_extra,
        "same_simulator_call_budget_as_tangent_response_targets": True,
        "allocation": "Latin-hypercube-like spread over response domain",
        "a_lo": args.a_lo,
        "a_hi": args.a_hi,
        "min_anchor_linf": args.min_anchor_linf,
        "sample_seed": args.sample_seed,
        "fresh_pdebench_seeds": True,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta2, f, indent=2)

    np.save(out / "spread_a.npy", a_extra.astype(np.float32))

    print()
    print("DONE")
    print("Output:", out)
    print("Original endpoint pairs:", original_n)
    print("Extra endpoint pairs:", args.n_extra)
    print("Total endpoint pairs:", int(augmented["x0"].shape[0]))
    print("Simulator-budget check PASSED:", args.n_extra == 648)


if __name__ == "__main__":
    main()
