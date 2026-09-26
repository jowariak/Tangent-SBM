#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


BASE = np.asarray(
    [1.0e-3, 5.0e-3, 5.0e-3],
    dtype=np.float64,
)

PARAM_NAMES = [
    "Du",
    "Dv",
    "k",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_pdebench_to_path(root: str | None) -> None:
    if root:
        root = str(
            Path(root).resolve()
        )

        if root not in sys.path:
            sys.path.insert(
                0,
                root,
            )


def import_official_simulator(root: str | None):
    add_pdebench_to_path(
        root
    )

    try:
        from pdebench.data_gen.src.sim_diff_react import Simulator
    except Exception as exc:
        raise RuntimeError(
            "Could not import the official PDEBench simulator. "
            "Clone/install PDEBench, or pass --pdebench-root /path/to/PDEBench."
        ) from exc

    return Simulator


def a_to_physical(a: np.ndarray) -> np.ndarray:
    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(
        3,
    )

    return (
        BASE
        * np.power(
            2.0,
            a,
        )
    )


def cache_key(
    seed: int,
    a: np.ndarray,
) -> str:
    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(
        3,
    )

    payload = (
        f"seed={int(seed)}|"
        f"a={a[0]:+.12f},{a[1]:+.12f},{a[2]:+.12f}|"
        "grid=128|T=5|tdim=101"
    )

    digest = hashlib.sha1(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()[
        :16
    ]

    return (
        f"seed_{int(seed):07d}_{digest}.npz"
    )


def run_official_one(
    pdebench_root: str | None,
    seed: int,
    a_tuple: tuple[float, float, float],
    cache_dir: str,
) -> dict:
    
    a = np.asarray(
        a_tuple,
        dtype=np.float64,
    )

    cache_dir_p = Path(
        cache_dir
    )

    cache_dir_p.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        cache_dir_p
        / cache_key(
            seed,
            a,
        )
    )

    if path.exists():
        return {
            "path":
                str(
                    path
                ),
            "cached":
                True,
        }

    Simulator = import_official_simulator(
        pdebench_root
    )

    Du, Dv, k = (
        a_to_physical(
            a
        )
    )

    sim = Simulator(
        Du=float(
            Du
        ),
        Dv=float(
            Dv
        ),
        k=float(
            k
        ),
        t=5,
        tdim=101,
        x_left=-1.0,
        x_right=1.0,
        xdim=128,
        y_bottom=-1.0,
        y_top=1.0,
        ydim=128,
        n=1,
        seed=int(
            seed
        ),
    )

    data = sim.generate_sample()

    if data.shape != (
        101,
        128,
        128,
        2,
    ):
        raise RuntimeError(
            f"Unexpected PDEBench sample shape: {data.shape}"
        )

    x0 = np.moveaxis(
        data[
            0
        ],
        -1,
        0,
    ).astype(
        np.float32,
        copy=False,
    )

    xT = np.moveaxis(
        data[
            -1
        ],
        -1,
        0,
    ).astype(
        np.float32,
        copy=False,
    )

    tmp = path.with_suffix(
        ".tmp.npz"
    )

    np.savez_compressed(
        tmp,
        x0=x0,
        xT=xT,
        a=a.astype(
            np.float32
        ),
        physical=np.asarray(
            [
                Du,
                Dv,
                k,
            ],
            dtype=np.float32,
        ),
        seed=np.asarray(
            seed,
            dtype=np.int64,
        ),
    )

    os.replace(
        tmp,
        path,
    )

    return {
        "path":
            str(
                path
            ),
        "cached":
            False,
    }


def simulate_many(
    specs: list[
        tuple[
            int,
            np.ndarray,
        ]
    ],
    *,
    pdebench_root: str | None,
    cache_dir: Path,
    workers: int,
) -> None:
    
    unique = {}

    for seed, a in specs:
        key = (
            int(
                seed
            ),
            tuple(
                float(
                    z
                )
                for z
                in np.asarray(
                    a
                ).reshape(
                    3,
                )
            ),
        )

        unique[
            key
        ] = (
            seed,
            np.asarray(
                a,
                dtype=np.float64,
            )
        )

    todo = []

    for seed, a in unique.values():
        path = (
            cache_dir
            / cache_key(
                seed,
                a,
            )
        )

        if not path.exists():
            todo.append(
                (
                    seed,
                    a,
                )
            )

    print(
        f"Requested {len(unique)} unique official trajectories; "
        f"{len(todo)} need simulation and "
        f"{len(unique)-len(todo)} are already cached."
    )

    if not todo:
        return

    if workers <= 1:
        for i, (
            seed,
            a,
        ) in enumerate(
            todo,
            start=1,
        ):
            t0 = time.time()

            run_official_one(
                pdebench_root,
                int(
                    seed
                ),
                tuple(
                    float(
                        z
                    )
                    for z
                    in a
                ),
                str(
                    cache_dir
                ),
            )

            print(
                f"[{i}/{len(todo)}] seed={seed} "
                f"a={np.round(a,4).tolist()} "
                f"{time.time()-t0:.1f}s",
                flush=True,
            )

        return

    with ProcessPoolExecutor(
        max_workers=
            workers
    ) as pool:
        futures = {}

        for seed, a in todo:
            fut = pool.submit(
                run_official_one,
                pdebench_root,
                int(
                    seed
                ),
                tuple(
                    float(
                        z
                    )
                    for z
                    in a
                ),
                str(
                    cache_dir
                ),
            )

            futures[
                fut
            ] = (
                seed,
                a,
            )

        done = 0

        for fut in as_completed(
            futures
        ):
            seed, a = (
                futures[
                    fut
                ]
            )

            fut.result()

            done += 1

            print(
                f"[{done}/{len(todo)}] seed={seed} "
                f"a={np.round(a,4).tolist()}",
                flush=True,
            )


def load_cached(
    cache_dir: Path,
    seed: int,
    a: np.ndarray,
) -> dict:
    path = (
        cache_dir
        / cache_key(
            seed,
            a,
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            str(
                path
            )
        )

    with np.load(
        path
    ) as z:
        return {
            "x0":
                torch.from_numpy(
                    z[
                        "x0"
                    ].copy()
                ),
            "xT":
                torch.from_numpy(
                    z[
                        "xT"
                    ].copy()
                ),
            "a":
                torch.from_numpy(
                    z[
                        "a"
                    ].copy()
                ),
            "physical":
                torch.from_numpy(
                    z[
                        "physical"
                    ].copy()
                ),
            "seed":
                int(
                    z[
                        "seed"
                    ]
                ),
        }


def anchor_grid() -> np.ndarray:
    return np.asarray(
        list(
            itertools.product(
                [
                    -1.0,
                    0.0,
                    1.0,
                ],
                repeat=3,
            )
        ),
        dtype=np.float64,
    )


def balanced_anchor_settings(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    anchors = anchor_grid()

    reps = int(
        math.ceil(
            n
            / anchors.shape[
                0
            ]
        )
    )

    arr = np.tile(
        anchors,
        (
            reps,
            1,
        ),
    )[
        :n
    ].copy()

    rng.shuffle(
        arr,
        axis=0,
    )

    return arr


def sample_box(
    n: int,
    lo: float,
    hi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.uniform(
        lo,
        hi,
        size=(
            n,
            3,
        ),
    )


def sample_shell(
    n: int,
    inner: float,
    outer: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = []

    while sum(
        z.shape[
            0
        ]
        for z
        in out
    ) < n:
        z = rng.uniform(
            -outer,
            outer,
            size=(
                max(
                    256,
                    3
                    * n,
                ),
                3,
            ),
        )

        r = np.max(
            np.abs(
                z
            ),
            axis=1,
        )

        keep = z[
            (
                r
                >= inner
            )
            & (
                r
                <= outer
            )
        ]

        out.append(
            keep
        )

    return np.concatenate(
        out,
        axis=0,
    )[
        :n
    ]


def unit_directions(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    v = rng.normal(
        size=(
            n,
            3,
        )
    )

    v /= np.maximum(
        np.linalg.norm(
            v,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )

    return v


def build_endpoint_specs(
    *,
    seed_start: int,
    n: int,
    mode: str,
    rng: np.random.Generator,
):
    seeds = np.arange(
        seed_start,
        seed_start
        + n,
        dtype=np.int64,
    )

    if mode in {
        "train",
        "seen",
    }:
        aa = (
            balanced_anchor_settings(
                n,
                rng,
            )
        )
    elif mode == "id":
        aa = sample_box(
            n,
            -1.0,
            1.0,
            rng,
        )
    elif mode == "near":
        aa = sample_shell(
            n,
            1.10,
            1.30,
            rng,
        )
    elif mode == "far":
        aa = sample_shell(
            n,
            1.40,
            1.70,
            rng,
        )
    else:
        raise ValueError(
            mode
        )

    return [
        (
            int(
                s
            ),
            aa[
                i
            ],
        )
        for i, s
        in enumerate(
            seeds
        )
    ]


def endpoint_object(
    specs,
    cache_dir,
):
    x0 = []
    xT = []
    a = []
    physical = []
    seeds = []

    for seed, aa in specs:
        z = load_cached(
            cache_dir,
            seed,
            aa,
        )

        x0.append(
            z[
                "x0"
            ]
        )

        xT.append(
            z[
                "xT"
            ]
        )

        a.append(
            z[
                "a"
            ]
        )

        physical.append(
            z[
                "physical"
            ]
        )

        seeds.append(
            seed
        )

    return {
        "x0":
            torch.stack(
                x0,
                dim=0,
            ).float(),
        "xT":
            torch.stack(
                xT,
                dim=0,
            ).float(),
        "a":
            torch.stack(
                a,
                dim=0,
            ).float(),
        "physical_params":
            torch.stack(
                physical,
                dim=0,
            ).float(),
        "pdebench_seed":
            torch.tensor(
                seeds,
                dtype=torch.long,
            ),
    }


def response_specs_and_object(
    operating_specs,
    *,
    rng,
    cache_dir,
    fd_delta,
    finite_delta,
):
    n = len(
        operating_specs
    )

    dirs = unit_directions(
        n,
        rng,
    )

    needed = []

    for i, (
        seed,
        a,
    ) in enumerate(
        operating_specs
    ):
        v = dirs[
            i
        ]

        needed.extend(
            [
                (
                    seed,
                    a,
                ),
                (
                    seed,
                    a
                    + fd_delta
                    * v,
                ),
                (
                    seed,
                    a
                    - fd_delta
                    * v,
                ),
                (
                    seed,
                    a
                    + finite_delta
                    * v,
                ),
            ]
        )

    return (
        dirs,
        needed,
    )


def assemble_response(
    operating_specs,
    dirs,
    *,
    cache_dir,
    fd_delta,
    finite_delta,
):
    x0 = []
    a_out = []
    phys = []
    v_out = []
    Jv = []
    finite = []
    seeds = []

    for i, (
        seed,
        a,
    ) in enumerate(
        operating_specs
    ):
        direction = (
            dirs[
                i
            ]
        )

        base = load_cached(
            cache_dir,
            seed,
            a,
        )

        plus = load_cached(
            cache_dir,
            seed,
            a
            + fd_delta
            * direction,
        )

        minus = load_cached(
            cache_dir,
            seed,
            a
            - fd_delta
            * direction,
        )

        fin = load_cached(
            cache_dir,
            seed,
            a
            + finite_delta
            * direction,
        )

        x0.append(
            base[
                "x0"
            ]
        )

        a_out.append(
            base[
                "a"
            ]
        )

        phys.append(
            base[
                "physical"
            ]
        )

        v_out.append(
            torch.from_numpy(
                direction.astype(
                    np.float32
                )
            )
        )

        Jv.append(
            (
                plus[
                    "xT"
                ]
                - minus[
                    "xT"
                ]
            )
            / (
                2.0
                * float(
                    fd_delta
                )
            )
        )

        finite.append(
            fin[
                "xT"
            ]
            - base[
                "xT"
            ]
        )

        seeds.append(
            seed
        )

    return {
        "x0":
            torch.stack(
                x0
            ).float(),
        "a":
            torch.stack(
                a_out
            ).float(),
        "physical_params":
            torch.stack(
                phys
            ).float(),
        "direction":
            torch.stack(
                v_out
            ).float(),
        "Jv_star":
            torch.stack(
                Jv
            ).float(),
        "finite_response":
            torch.stack(
                finite
            ).float(),
        "pdebench_seed":
            torch.tensor(
                seeds,
                dtype=torch.long,
            ),
        "response_only":
            True,
        "endpoint_labels_included":
            False,
    }


def attach_normalization(
    endpoint_objs,
    response_objs,
):
    train = endpoint_objs[
        "train"
    ]

    state = torch.cat(
        [
            train[
                "x0"
            ],
            train[
                "xT"
            ],
        ],
        dim=0,
    )

    mean = state.mean(
        dim=(
            0,
            2,
            3,
        ),
        keepdim=True,
    )

    std = state.std(
        dim=(
            0,
            2,
            3,
        ),
        keepdim=True,
    ).clamp_min(
        1e-6
    )

    for obj in endpoint_objs.values():
        obj[
            "x0_norm"
        ] = (
            obj[
                "x0"
            ]
            - mean
        ) / std

        obj[
            "xT_norm"
        ] = (
            obj[
                "xT"
            ]
            - mean
        ) / std

    for obj in response_objs.values():
        obj[
            "x0_norm"
        ] = (
            obj[
                "x0"
            ]
            - mean
        ) / std

        obj[
            "Jv_star_norm"
        ] = (
            obj[
                "Jv_star"
            ]
            / std
        )

        obj[
            "finite_response_norm"
        ] = (
            obj[
                "finite_response"
            ]
            / std
        )

    return (
        mean.float(),
        std.float(),
    )


def verify_public_h5(
    public_h5: str,
    cache_dir: Path,
    seeds: list[int],
):
    import h5py

    errors = []

    a0 = np.zeros(
        3,
        dtype=np.float64,
    )

    with h5py.File(
        public_h5,
        "r",
    ) as f:
        for seed in seeds:
            key = str(
                seed
            ).zfill(
                4
            )

            if key not in f:
                continue

            public = np.asarray(
                f[
                    f"{key}/data"
                ],
                dtype=np.float32,
            )

            ours = load_cached(
                cache_dir,
                seed,
                a0,
            )

            pub0 = np.moveaxis(
                public[
                    0
                ],
                -1,
                0,
            )

            pubT = np.moveaxis(
                public[
                    -1
                ],
                -1,
                0,
            )

            e0 = float(
                np.max(
                    np.abs(
                        pub0
                        - ours[
                            "x0"
                        ].numpy()
                    )
                )
            )

            eT = float(
                np.max(
                    np.abs(
                        pubT
                        - ours[
                            "xT"
                        ].numpy()
                    )
                )
            )

            errors.append(
                {
                    "seed":
                        seed,
                    "max_abs_x0":
                        e0,
                    "max_abs_xT":
                        eT,
                }
            )

    return errors


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pdebench-root",
        default=None,
        help="Path to cloned PDEBench repository. Not needed if pdebench is installed.",
    )

    p.add_argument(
        "--out-dir",
        default=
            "runs/pdebench_reaction_diffusion_official",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=32,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--n-train",
        type=int,
        default=270,
        help="Default = 10 endpoint ICs per each of 27 anchor parameter settings.",
    )

    p.add_argument(
        "--n-test",
        type=int,
        default=108,
    )

    p.add_argument(
        "--n-anchor-response",
        type=int,
        default=108,
    )

    p.add_argument(
        "--n-response-collocation",
        type=int,
        default=216,
    )

    p.add_argument(
        "--n-eval-response",
        type=int,
        default=64,
        help="Response truth subset per endpoint test split.",
    )

    p.add_argument(
        "--fd-delta",
        type=float,
        default=0.03,
    )

    p.add_argument(
        "--finite-delta",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--public-h5",
        default=None,
        help="Optional official released PDEBench .h5 for reproduction check at a=0.",
    )

    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one default-parameter official trajectory and exit.",
    )

    args = p.parse_args()

    set_seed(
        args.seed
    )

    
    import_official_simulator(
        args.pdebench_root
    )

    out_dir = Path(
        args.out_dir
    )

    cache_dir = (
        out_dir
        / "sim_cache"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.smoke_test:
        print(
            "Running one exact official PDEBench default-parameter trajectory..."
        )

        t0 = time.time()

        simulate_many(
            [
                (
                    0,
                    np.zeros(
                        3,
                        dtype=np.float64,
                    ),
                )
            ],
            pdebench_root=
                args.pdebench_root,
            cache_dir=
                cache_dir,
            workers=1,
        )

        z = load_cached(
            cache_dir,
            0,
            np.zeros(
                3,
                dtype=np.float64,
            ),
        )

        print(
            "x0 shape:",
            tuple(
                z[
                    "x0"
                ].shape
            ),
        )

        print(
            "xT shape:",
            tuple(
                z[
                    "xT"
                ].shape
            ),
        )

        print(
            "physical params:",
            z[
                "physical"
            ].tolist(),
        )

        print(
            f"elapsed={time.time()-t0:.2f}s"
        )

        return

    rng = np.random.default_rng(
        args.seed
    )

    
    cursor = 0

    train_specs = (
        build_endpoint_specs(
            seed_start=
                cursor,
            n=
                args.n_train,
            mode="train",
            rng=rng,
        )
    )

    cursor += (
        args.n_train
    )

    seen_specs = (
        build_endpoint_specs(
            seed_start=
                cursor,
            n=
                args.n_test,
            mode="seen",
            rng=rng,
        )
    )

    cursor += (
        args.n_test
    )

    id_specs = (
        build_endpoint_specs(
            seed_start=
                cursor,
            n=
                args.n_test,
            mode="id",
            rng=rng,
        )
    )

    cursor += (
        args.n_test
    )

    near_specs = (
        build_endpoint_specs(
            seed_start=
                cursor,
            n=
                args.n_test,
            mode="near",
            rng=rng,
        )
    )

    cursor += (
        args.n_test
    )

    far_specs = (
        build_endpoint_specs(
            seed_start=
                cursor,
            n=
                args.n_test,
            mode="far",
            rng=rng,
        )
    )

    cursor += (
        args.n_test
    )

    colloc_a = sample_box(
        args.n_response_collocation,
        -1.30,
        1.30,
        rng,
    )

    colloc_specs = [
        (
            cursor
            + i,
            colloc_a[
                i
            ],
        )
        for i
        in range(
            args.n_response_collocation
        )
    ]

    cursor += (
        args.n_response_collocation
    )

    endpoint_specs = {
        "train":
            train_specs,
        "test_seen":
            seen_specs,
        "test_id":
            id_specs,
        "test_ood_near":
            near_specs,
        "test_ood_far":
            far_specs,
    }

    
    all_specs = []

    for specs in endpoint_specs.values():
        all_specs.extend(
            specs
        )

    
    anchor_specs = (
        train_specs[
            :min(
                args.n_anchor_response,
                len(
                    train_specs
                ),
            )
        ]
    )

    response_operating = {
        "anchor_response":
            anchor_specs,
        "response_collocation":
            colloc_specs,
        "response_eval_seen":
            seen_specs[
                :min(
                    args.n_eval_response,
                    len(
                        seen_specs
                    ),
                )
            ],
        "response_eval_id":
            id_specs[
                :min(
                    args.n_eval_response,
                    len(
                        id_specs
                    ),
                )
            ],
        "response_eval_ood_near":
            near_specs[
                :min(
                    args.n_eval_response,
                    len(
                        near_specs
                    ),
                )
            ],
        "response_eval_ood_far":
            far_specs[
                :min(
                    args.n_eval_response,
                    len(
                        far_specs
                    ),
                )
            ],
    }

    response_dirs = {}

    for name, specs in response_operating.items():
        dirs, needed = (
            response_specs_and_object(
                specs,
                rng=rng,
                cache_dir=
                    cache_dir,
                fd_delta=
                    args.fd_delta,
                finite_delta=
                    args.finite_delta,
            )
        )

        response_dirs[
            name
        ] = dirs

        all_specs.extend(
            needed
        )

    print(
        "Generating/caching official PDEBench trajectories..."
    )

    simulate_many(
        all_specs,
        pdebench_root=
            args.pdebench_root,
        cache_dir=
            cache_dir,
        workers=
            args.workers,
    )

    endpoint_objs = {}

    for name, specs in endpoint_specs.items():
        print(
            "Assembling",
            name,
        )

        endpoint_objs[
            name
        ] = endpoint_object(
            specs,
            cache_dir,
        )

    response_objs = {}

    for name, specs in response_operating.items():
        print(
            "Assembling",
            name,
        )

        response_objs[
            name
        ] = assemble_response(
            specs,
            response_dirs[
                name
            ],
            cache_dir=
                cache_dir,
            fd_delta=
                args.fd_delta,
            finite_delta=
                args.finite_delta,
        )

    mean, std = (
        attach_normalization(
            endpoint_objs,
            response_objs,
        )
    )

    for name, obj in endpoint_objs.items():
        torch.save(
            obj,
            out_dir
            / f"{name}.pt",
        )

    for name, obj in response_objs.items():
        torch.save(
            obj,
            out_dir
            / f"{name}.pt",
        )

    verification = None

    if args.public_h5:
        
        verify_seeds = [
            0,
            1,
            2,
        ]

        simulate_many(
            [
                (
                    s,
                    np.zeros(
                        3,
                        dtype=np.float64,
                    ),
                )
                for s
                in verify_seeds
            ],
            pdebench_root=
                args.pdebench_root,
            cache_dir=
                cache_dir,
            workers=
                args.workers,
        )

        verification = (
            verify_public_h5(
                args.public_h5,
                cache_dir,
                verify_seeds,
            )
        )

        print(
            "Public-H5 reproduction check:"
        )

        print(
            json.dumps(
                verification,
                indent=2,
            )
        )

    metadata = {
        "source":
            "official PDEBench pdebench.data_gen.src.sim_diff_react.Simulator",

        "system":
            "2D diffusion-reaction",

        "official_configuration":
            {
                "grid":
                    [
                        128,
                        128,
                    ],
                "tdim":
                    101,
                "T":
                    5.0,
                "domain":
                    [
                        -1.0,
                        1.0,
                        -1.0,
                        1.0,
                    ],
                "Du0":
                    1e-3,
                "Dv0":
                    5e-3,
                "k0":
                    5e-3,
            },

        "pde":
            {
                "u_t":
                    "u - u^3 - k - v + Du * Laplacian(u)",
                "v_t":
                    "u - v + Dv * Laplacian(v)",
            },

        "intervention":
            {
                "a":
                    [
                        "a_Du",
                        "a_Dv",
                        "a_k",
                    ],
                "mapping":
                    "physical = default * 2**a",
                "dimension":
                    3,
            },

        "state_shape":
            [
                2,
                128,
                128,
            ],

        "state_dim":
            32768,

        "train_anchors_each_coordinate":
            [
                -1.0,
                0.0,
                1.0,
            ],

        "train_anchor_count":
            27,

        "test_id":
            "a in [-1,1]^3",

        "test_ood_near":
            "max(abs(a)) in [1.10,1.30]",

        "test_ood_far":
            "max(abs(a)) in [1.40,1.70]",

        "response_collocation":
            "a in [-1.30,1.30]^3",

        "response_semantics":
            "pathwise directional field response J*r",

        "fd_delta":
            args.fd_delta,

        "finite_delta":
            args.finite_delta,

        "sizes":
            {
                "train":
                    len(
                        train_specs
                    ),
                "test_seen":
                    len(
                        seen_specs
                    ),
                "test_id":
                    len(
                        id_specs
                    ),
                "test_ood_near":
                    len(
                        near_specs
                    ),
                "test_ood_far":
                    len(
                        far_specs
                    ),
                "anchor_response":
                    len(
                        anchor_specs
                    ),
                "response_collocation":
                    len(
                        colloc_specs
                    ),
                "response_eval_per_split":
                    args.n_eval_response,
            },

        "normalization":
            {
                "state_mean_per_channel":
                    mean.reshape(
                        -1
                    ).tolist(),
                "state_std_per_channel":
                    std.reshape(
                        -1
                    ).tolist(),
            },

        "public_h5_reproduction_check":
            verification,
    }

    with open(
        out_dir
        / "metadata.json",
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\nDONE"
    )

    print(
        "Output:",
        out_dir.resolve(),
    )

    print(
        "State shape:",
        metadata[
            "state_shape"
        ],
    )

    for name in [
        "response_eval_seen",
        "response_eval_id",
        "response_eval_ood_near",
        "response_eval_ood_far",
    ]:
        obj = response_objs[
            name
        ]

        norms = torch.linalg.vector_norm(
            obj[
                "Jv_star_norm"
            ].reshape(
                obj[
                    "Jv_star_norm"
                ].shape[
                    0
                ],
                -1,
            ),
            dim=1,
        )

        print(
            name,
            "| mean normalized ||J*r|| =",
            f"{float(norms.mean()):.6f}",
        )


if __name__ == "__main__":
    main()
