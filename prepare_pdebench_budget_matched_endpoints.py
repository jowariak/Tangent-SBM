#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_cache_index(cache_dir: Path):
    index = {}
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in {cache_dir}")

    print(f"Indexing {len(files)} cached PDEBench trajectories...")

    for i, path in enumerate(files, 1):
        with np.load(path) as z:
            seed = int(z["seed"])
            a = np.asarray(z["a"], dtype=np.float32).reshape(3)

        index.setdefault(seed, []).append((a, path))

        if i % 500 == 0 or i == len(files):
            print(f"  indexed {i}/{len(files)}")

    return index


def find_cached(index, seed: int, target_a: np.ndarray, tol=5e-5):
    if seed not in index:
        raise KeyError(f"No cached trajectories for PDEBench seed {seed}")

    target_a = np.asarray(target_a, dtype=np.float32).reshape(3)
    best = None
    best_dist = float("inf")

    for a, path in index[seed]:
        dist = float(np.max(np.abs(a - target_a)))
        if dist < best_dist:
            best_dist = dist
            best = path

    if best is None or best_dist > tol:
        raise RuntimeError(
            f"Could not match cache for seed={seed}, "
            f"target a={target_a.tolist()}, nearest error={best_dist:.3e}"
        )

    return best, best_dist


def load_cache_record(path: Path):
    with np.load(path) as z:
        return {
            "x0": torch.from_numpy(np.asarray(z["x0"], dtype=np.float32).copy()),
            "xT": torch.from_numpy(np.asarray(z["xT"], dtype=np.float32).copy()),
            "a": torch.from_numpy(np.asarray(z["a"], dtype=np.float32).copy()),
            "physical_params": torch.from_numpy(
                np.asarray(z["physical"], dtype=np.float32).copy()
            ),
            "pdebench_seed": int(z["seed"]),
        }


def collect_extra_pairs(response_obj, index, fd_delta, label):
    a = response_obj["a"].float()
    direction = response_obj["direction"].float()
    seeds = response_obj["pdebench_seed"].long()

    extras = []
    max_match_error = 0.0

    for i in range(a.shape[0]):
        seed = int(seeds[i])
        ai = a[i].numpy().astype(np.float32)
        vi = direction[i].numpy().astype(np.float32)

        for sign in (+1.0, -1.0):
            target = ai + np.float32(sign * fd_delta) * vi

            path, err = find_cached(
                index=index,
                seed=seed,
                target_a=target,
            )
            max_match_error = max(max_match_error, err)
            rec = load_cache_record(path)

            if "x0" in response_obj:
                ref_x0 = response_obj["x0"][i].float()
                if not torch.allclose(rec["x0"], ref_x0, atol=1e-6, rtol=1e-6):
                    raise RuntimeError(
                        f"{label}[{i}] cache trajectory has a different X0"
                    )

            extras.append(rec)

    print(
        f"{label}: {a.shape[0]} response points -> {len(extras)} endpoint pairs; "
        f"max cache match error={max_match_error:.3e}"
    )
    return extras


def stack_records(records):
    return {
        "x0": torch.stack([r["x0"] for r in records], dim=0).float(),
        "xT": torch.stack([r["xT"] for r in records], dim=0).float(),
        "a": torch.stack([r["a"] for r in records], dim=0).float(),
        "physical_params": torch.stack(
            [r["physical_params"] for r in records], dim=0
        ).float(),
        "pdebench_seed": torch.tensor(
            [r["pdebench_seed"] for r in records], dtype=torch.long
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-dir",
        default="runs/pdebench_reaction_diffusion_official",
    )
    p.add_argument(
        "--out-dir",
        default="runs/pdebench_reaction_diffusion_budget_matched",
    )
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    source = Path(args.source_dir)
    out = Path(args.out_dir)
    cache = Path(args.cache_dir) if args.cache_dir else source / "sim_cache"

    out.mkdir(parents=True, exist_ok=True)

    with open(source / "metadata.json", "r") as f:
        metadata = json.load(f)

    fd_delta = float(metadata["fd_delta"])

    train = safe_torch_load(source / "train.pt")
    anchor = safe_torch_load(source / "anchor_response.pt")
    colloc = safe_torch_load(source / "response_collocation.pt")

    original_n = int(train["x0"].shape[0])

    print(f"Original endpoint training pairs: {original_n}")
    print(f"fd_delta: {fd_delta}")

    index = build_cache_index(cache)

    anchor_extra = collect_extra_pairs(
        anchor, index, fd_delta, "anchor_response"
    )
    colloc_extra = collect_extra_pairs(
        colloc, index, fd_delta, "response_collocation"
    )

    extra = stack_records(anchor_extra + colloc_extra)

    mean = torch.tensor(
        metadata["normalization"]["state_mean_per_channel"],
        dtype=torch.float32,
    ).reshape(1, 2, 1, 1)

    std = torch.tensor(
        metadata["normalization"]["state_std_per_channel"],
        dtype=torch.float32,
    ).reshape(1, 2, 1, 1)

    extra["x0_norm"] = (extra["x0"] - mean) / std
    extra["xT_norm"] = (extra["xT"] - mean) / std

    keys = [
        "x0",
        "xT",
        "a",
        "physical_params",
        "pdebench_seed",
        "x0_norm",
        "xT_norm",
    ]

    augmented = {}
    for key in keys:
        if key not in train:
            raise RuntimeError(f"Original train.pt missing key {key}")
        augmented[key] = torch.cat([train[key], extra[key]], dim=0)

    augmented["simulator_budget_matched_control"] = True
    augmented["original_train_size"] = original_n
    augmented["extra_endpoint_pairs"] = int(extra["x0"].shape[0])

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

    budget_metadata = dict(metadata)
    budget_metadata["simulator_budget_matched_control"] = {
        "original_train_pairs": original_n,
        "response_operating_points": int(
            anchor["a"].shape[0] + colloc["a"].shape[0]
        ),
        "simulator_calls_per_response_target_used_as_endpoints": 2,
        "extra_endpoint_pairs": int(extra["x0"].shape[0]),
        "augmented_train_pairs": int(augmented["x0"].shape[0]),
        "uses_finite_response_eval_calls": False,
    }

    with open(out / "metadata.json", "w") as f:
        json.dump(budget_metadata, f, indent=2)

    expected = 2 * int(anchor["a"].shape[0] + colloc["a"].shape[0])
    actual = int(extra["x0"].shape[0])

    print("\nDONE")
    print("Original endpoint pairs:", original_n)
    print("Extra endpoint pairs:", actual)
    print("Total endpoint pairs:", int(augmented["x0"].shape[0]))

    if actual != expected:
        raise RuntimeError(f"Expected {expected} extra pairs, created {actual}")

    print(
        "Simulator-budget check PASSED: exactly two extra endpoint labels "
        "per Tangent response-supervision operating point."
    )


if __name__ == "__main__":
    main()
