import torch
import gaussian_reference as base

def gaussian_reverse_gain(metadata, device, dtype):
    sigma0 = torch.tensor(
        metadata["Sigma0"],
        device=device,
        dtype=dtype,
    )

    A = torch.tensor(
        metadata["A"],
        device=device,
        dtype=dtype,
    )

    sigma_eps = torch.tensor(
        metadata["Sigma_eps"],
        device=device,
        dtype=dtype,
    )

    sigma_y = (
        A
        @ sigma0
        @ A.T
        + sigma_eps
    )

    
    
    K = torch.linalg.solve(
        sigma_y.T,
        (sigma0 @ A.T).T,
    ).T

    return K

def exact_forward_jacobian(u, metadata):
    
    return base.oracle_jacobian(
        u,
        metadata,
    )

def exact_backward_mean_jacobian(u, metadata):
    
    Jf = exact_forward_jacobian(
        u,
        metadata,
    )

    K = gaussian_reverse_gain(
        metadata,
        device=u.device,
        dtype=u.dtype,
    )

    return -torch.einsum(
        "ij,bjk->bik",
        K,
        Jf,
    )

def evaluate_backward_response(
    model,
    data,
    metadata,
    cfg,
    device,
):
    n = min(
        cfg.eval_batch_size,
        data["x1"].shape[0],
    )

    xT = (
        data["x1"][:n]
        .to(device)
    )

    u = (
        data["u"][:n]
        .to(device)
    )

    J_true = (
        exact_backward_mean_jacobian(
            u,
            metadata,
        )
    )

    Js = []

    
    with torch.enable_grad():
        for _ in range(
            cfg.eval_sens_mc
        ):
            noise_bank = [
                torch.randn_like(
                    xT
                )
                for _ in range(
                    model.sens_steps
                )
            ]

            J = (
                model.tangent_training_rollout_direction(
                    xstart=xT,
                    u=u,
                    noise_bank=
                        noise_bank,
                    fb="b",
                )
            )

            Js.append(
                J.detach()
            )

    J_pred = (
        torch.stack(
            Js,
            dim=0,
        )
        .mean(dim=0)
    )

    diff = (
        J_pred
        - J_true
    )

    rel = (
        torch.linalg.vector_norm(
            diff.reshape(-1)
        )
        / torch.linalg.vector_norm(
            J_true.reshape(-1)
        ).clamp_min(1e-8)
    )

    rmse = torch.sqrt(
        (diff ** 2)
        .mean()
    )

    return {
        "backward_jacobian_rel_error":
            float(
                rel.detach()
                .cpu()
            ),
        "backward_jacobian_rmse":
            float(
                rmse.detach()
                .cpu()
            ),
        "mean_predicted_backward_jacobian":
            base.tensor_to_list(
                J_pred.mean(
                    dim=0
                )
            ),
        "mean_true_backward_jacobian":
            base.tensor_to_list(
                J_true.mean(
                    dim=0
                )
            ),
    }