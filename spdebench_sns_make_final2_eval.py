#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


SPLITS = {
    "final2_seen": [-1.0, 0.0, 1.0],
    "final2_id": [-0.75, -0.25, 0.25, 0.75],
    "final2_ood_near": [-1.25, 1.25],
    "final2_ood_far": [-1.50, 1.50],
}


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("runs/spdebench_sns_response"))
    p.add_argument("--spdebench-root", type=Path, default=Path("SPDE_hackathon"))
    p.add_argument(
        "--generator-script-dir",
        type=Path,
        default=Path("."),
        help="Directory containing spdebench_sns_response_dataset.py",
    )
    p.add_argument("--seed", type=int, default=2026091702)
    p.add_argument("--final-ics", type=int, default=16)
    p.add_argument("--endpoint-mc", type=int, default=32)
    p.add_argument("--response-mc", type=int, default=16)
    p.add_argument("--finite-mc", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def unique_existing_ics(data_dir: Path):
    ic = safe_load(data_dir / "initial_conditions.pt")
    required = ["train_x0", "eval_x0"]
    missing = [k for k in required if k not in ic]
    if missing:
        raise RuntimeError(f"initial_conditions.pt missing {missing}")

    xs = [
        ic["train_x0"].float().reshape(ic["train_x0"].shape[0], -1),
        ic["eval_x0"].float().reshape(ic["eval_x0"].shape[0], -1),
    ]

    val_path = data_dir / "validation.pt"
    if val_path.exists():
        v = safe_load(val_path)
        if "x0" in v:
            vx = v["x0"].float()
            xs.append(vx.reshape(vx.shape[0], -1))

    return torch.cat(xs, dim=0)


def main():
    args = parse_args()
    data_dir = args.data_dir
    manifest_path = data_dir / "final2_eval_manifest.json"

    if manifest_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"{manifest_path} already exists. The FINAL2 locked set has already "
            "been generated. Refusing to regenerate it. Do NOT use --overwrite "
            "after inspecting final results."
        )

    sys.path.insert(0, str(args.generator_script_dir.resolve()))
    import spdebench_sns_response_dataset as gen

    required_api = [
        "set_seed", "load_spdebench", "SimConfig", "SNSimulator",
        "build_conditions", "add_channel", "cpu", "nu_from_a",
    ]
    missing = [name for name in required_api if not hasattr(gen, name)]
    if missing:
        raise RuntimeError(
            "spdebench_sns_response_dataset.py missing expected API: "
            + ", ".join(missing)
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.quick:
        args.final_ics = 1
        args.endpoint_mc = 2
        args.response_mc = 1
        args.finite_mc = 1
        args.batch_size = min(args.batch_size, 4)

    gen.set_seed(args.seed)
    navier_stokes_2d, GaussianRF = gen.load_spdebench(args.spdebench_root)
    cfg = gen.SimConfig()
    sim = gen.SNSimulator(cfg, navier_stokes_2d, GaussianRF, device)

    for name in ["endpoint_batched", "centered_fd", "finite_response"]:
        if not hasattr(sim, name):
            raise RuntimeError(
                f"SNSimulator missing '{name}'. Use the same generator module "
                "that produced the benchmark."
            )

    ic_obj = safe_load(data_dir / "initial_conditions.pt")
    if "w_star" not in ic_obj:
        raise RuntimeError("initial_conditions.pt missing w_star")
    w_star = ic_obj["w_star"].to(device)

    
    final_x0 = w_star + sim.grf.sample(args.final_ics)

    existing = unique_existing_ics(data_dir)
    fresh = final_x0.detach().cpu().reshape(args.final_ics, -1)
    min_rms = []
    for i in range(fresh.shape[0]):
        d = torch.sqrt(torch.mean((existing - fresh[i:i+1]) ** 2, dim=1))
        min_rms.append(float(d.min()))
    if min(min_rms) < 1e-8:
        raise RuntimeError(
            "A final-evaluation initial condition exactly duplicates an "
            "existing train/original-test/validation IC."
        )

    print("device =", device, flush=True)
    print("FINAL2 locked evaluation seed =", args.seed, flush=True)
    print("fresh FINAL2 ICs =", args.final_ics, flush=True)
    print("minimum RMS distance to any train/original-test/validation IC =",
          min(min_rms), flush=True)
    print("endpoint_mc =", args.endpoint_mc, flush=True)
    print("response_mc =", args.response_mc, flush=True)
    print("finite_mc =", args.finite_mc, flush=True)

    file_map = {}

    for split, a_values in SPLITS.items():
        print("\n" + "=" * 80, flush=True)
        print(split, "a =", a_values, flush=True)
        print("=" * 80, flush=True)

        x0_cond, a_cond = gen.build_conditions(final_x0, a_values)
        C = x0_cond.shape[0]
        print("conditions =", C, flush=True)

        print("Generating endpoint samples ...", flush=True)
        x_rep = x0_cond.repeat_interleave(args.endpoint_mc, dim=0)
        a_rep = a_cond.repeat_interleave(args.endpoint_mc)
        y = sim.endpoint_batched(
            x_rep, a_rep, batch_size=args.batch_size
        )
        y_samples = y.reshape(C, args.endpoint_mc, cfg.s, cfg.s)

        print("Generating centered-FD response targets ...", flush=True)
        j_mean, j_samples = sim.centered_fd(
            x0_cond,
            a_cond,
            eps=cfg.eps_a,
            mc=args.response_mc,
            batch_size=args.batch_size,
        )

        print("Generating finite-response targets ...", flush=True)
        finite_mean, finite_samples = sim.finite_response(
            x0_cond,
            a_cond,
            delta=cfg.finite_delta_a,
            mc=args.finite_mc,
            batch_size=args.batch_size,
        )

        enstrophy_samples = 0.5 * torch.mean(
            y_samples ** 2, dim=(-2, -1)
        )

        obj = {
            "benchmark": "spdebench_sns_response",
            "split": split,
            "locked_final_evaluation_only": True,
            "model_selection_allowed": False,
            "generation_seed": args.seed,
            "a_values": list(a_values),
            "num_fresh_initial_conditions": args.final_ics,
            "x0": gen.add_channel(gen.cpu(x0_cond)),
            "a": gen.cpu(a_cond.reshape(-1, 1)),
            "nu": gen.cpu(gen.nu_from_a(a_cond, cfg.nu0).reshape(-1, 1)),
            "xT_samples": gen.cpu(y_samples).unsqueeze(2),
            "endpoint_mc": args.endpoint_mc,
            "direction": torch.ones((C, 1), dtype=torch.float32),
            "Jv_star": gen.add_channel(gen.cpu(j_mean)),
            "Jv_samples": gen.cpu(j_samples).unsqueeze(2),
            "response_mc": args.response_mc,
            "finite_response_star": gen.add_channel(gen.cpu(finite_mean)),
            "finite_response_samples": gen.cpu(finite_samples).unsqueeze(2),
            "finite_mc": args.finite_mc,
            "eps_a": cfg.eps_a,
            "finite_delta_a": cfg.finite_delta_a,
            "enstrophy_samples": gen.cpu(enstrophy_samples),
            "provenance": {
                "same_ic_law_as_main_benchmark": True,
                "fresh_grf_perturbations": True,
                "disjoint_from_train_original_test_validation_ics": True,
                "min_rms_distance_to_existing_ic": min(min_rms),
                "purpose": "single_untouched_FINAL2_evaluation_after_batch8_model_frozen",
            },
        }

        out = data_dir / f"{split}.pt"
        tmp = data_dir / f".{split}.pt.tmp"
        torch.save(obj, tmp)
        tmp.replace(out)
        file_map[split] = str(out)

        print("Saved", out, flush=True)
        print("x0", tuple(obj["x0"].shape), flush=True)
        print("xT_samples", tuple(obj["xT_samples"].shape), flush=True)
        print("Jv_samples", tuple(obj["Jv_samples"].shape), flush=True)
        print("finite_response_samples",
              tuple(obj["finite_response_samples"].shape), flush=True)

    
    final_ic_path = data_dir / "final2_eval_initial_conditions.pt"
    torch.save(
        {
            "seed": args.seed,
            "x0": gen.add_channel(gen.cpu(final_x0)),
            "locked_final_evaluation_only": True,
        },
        final_ic_path,
    )

    manifest = {
        "benchmark": "spdebench_sns_response",
        "locked": True,
        "generation_seed": args.seed,
        "final_ics": args.final_ics,
        "split_interventions": SPLITS,
        "endpoint_mc": args.endpoint_mc,
        "response_mc": args.response_mc,
        "finite_mc": args.finite_mc,
        "eps_a": cfg.eps_a,
        "finite_delta_a": cfg.finite_delta_a,
        "min_rms_distance_to_train_original_test_validation_ic": min(min_rms),
        "files": file_map,
        "initial_conditions_file": str(final_ic_path),
        "protocol_note": (
            "Generated once after freezing sigma=1.4, IMF3->5, "
            "lambda_sens=1000. This set must not be used for further "
            "hyperparameter/model selection."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("\nFINAL2 LOCKED MANIFEST SAVED:", manifest_path, flush=True)
    print("Do not regenerate this set after inspecting results.", flush=True)


if __name__ == "__main__":
    main()
