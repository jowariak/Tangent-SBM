#!/usr/bin/env python3


import torch
import double_well_tangent_sbm_v2 as impl


class NaiveSingleRolloutTangent(impl.TangentDoubleWellDSBM):
    def expected_response_loss(
        self,
        anchor_response,
        collocation_response,
    ):
        total = self.sens_batch_size
        n_anchor = int(round(total * self.anchor_fraction))
        n_colloc = total - n_anchor

        pieces = []
        a = self._response_batch(anchor_response, n_anchor)
        if a is not None:
            pieces.append(a)
        c = self._response_batch(collocation_response, n_colloc)
        if c is not None:
            pieces.append(c)

        x0 = torch.cat([p["x0"] for p in pieces], dim=0)
        u = torch.cat([p["u"] for p in pieces], dim=0)
        J_star = torch.cat([p["J_star"] for p in pieces], dim=0)

        perm = torch.randperm(
            x0.shape[0],
            generator=self.sens_generator,
            device="cpu",
        ).to(self.device)

        x0 = x0[perm]
        u = u[perm]
        J_star = J_star[perm]

        noise = self._sens_noise_bank(x0.shape[0], x0.dtype)
        J = self.tangent_rollout_train(x0, u, noise)

        loss = ((J - J_star) ** 2).mean()

        return loss, {
            "n_anchor": n_anchor,
            "n_collocation": n_colloc,
            "loss_mode": "naive_single_rollout_mse",
        }



impl.TangentDoubleWellDSBM = NaiveSingleRolloutTangent

if __name__ == "__main__":
    impl.main()
