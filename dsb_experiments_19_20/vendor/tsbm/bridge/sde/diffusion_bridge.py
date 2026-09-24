import torch
from tqdm import trange


def get_sde_timesteps(t_final, num_steps, eps, device):
    """
    Returns constant dt and two time grids used in simulation of discretized controlled SDEs:
    - sde_ts['f'], (num_steps+1,): 2*eps -> t_final-2*eps (forward times)
    - sde_ts['b'], (num_steps+1,): t_final-2*eps -> 2*eps (backward times)
     """
    dt = (t_final - 4 * eps) / num_steps
    ts = torch.arange(num_steps + 1) * dt + 2 * eps
    sde_ts = {'f': ts, 'b': t_final - ts}
    sde_ts['f'] = sde_ts['f'].to(device)
    sde_ts['b'] = sde_ts['b'].to(device)
    return sde_ts, dt


class DenoiserToVelocityReparamWrapper(torch.nn.Module):
    """Class to reparameterize denoiser net into velocity net with Brownian bridge convention."""

    def __init__(self, net, t_final, sample_direction):
        super().__init__()
        self.net = net
        self.T = t_final
        self.fb = sample_direction

    def forward(self, x, t):
        if self.fb == 'f':
            return (self.net(x, t) - x) / (self.T - t)
        else:
            return (self.net(x, t) - x) / t


class Twisted_BM:
    def __init__(self, t_final, sigma, shape_x, loss_param, device):
        self.device = device

        self.T = t_final
        self.sigma = sigma
        self.d_x = shape_x
        self.loss_param = loss_param  # 'velocity', 'denoiser'

    @torch.no_grad()
    def sample_sde(self, x_start,
                   sample_net,
                   sample_direction,
                   sde_timesteps,
                   id_to_log,
                   save_pred=False,
                   verbose=False):
        """
        - x_start: (B, shape_x)
        - sample_net: net that approximates the drift of the SDE (velocity) or the predicted terminal state (denoiser)
        - sample_direction: 'f' or 'b'
        - sde_timesteps: (num_steps+1, ) -> associated times
            - if sample_direction=='f': increasing times
            - if sample_direction=='b': decreasing times
        - id_to_log: (n,) where 0 < n <= num_steps+1 -> array of indexes between 0 (time=0) and num_steps (time=T) to be saved
        - save_pred : boolean -> whether to save the velocity at the trajectory times

        Returns:
        - log_x: (B, n, shape_x)
        - log_pred: (B, n, shape_x) if save_pred else None
        - ts: (n,) -> corresponding times
        """
        sde_ts = sde_timesteps[sample_direction].to(self.device)  # (num_steps + 1,)
        num_steps = len(sde_ts) - 1
        if sample_direction == 'f':
            dt = sde_ts[1:] - sde_ts[:-1]  # (num_steps,)
        else:
            dt = sde_ts[:-1] - sde_ts[1:]  # (num_steps,)
            id_to_log = torch.sort(num_steps - id_to_log).values  # reverse indexes in 'b'
        assert torch.all(dt > 0), 'Step size should be positive.'

        assert 0 < len(id_to_log) <= num_steps + 1
        assert id_to_log.min() >= 0 and id_to_log.max() <= num_steps

        b = x_start.shape[0]
        log_x = torch.empty((b, len(id_to_log), *self.d_x), device=self.device)
        log_pred = torch.empty((b, len(id_to_log), *self.d_x), device=self.device) if save_pred else None

        # Reparameterize into velocity if necessary
        if self.loss_param == 'denoiser':
            velocity_net = DenoiserToVelocityReparamWrapper(sample_net, self.T, sample_direction)
        else:
            velocity_net = sample_net

        # Use Euler method
        x = x_start.clone()
        index = 0
        bar = trange(num_steps + 1) if verbose else range(num_steps + 1)
        for i in bar:
            t = torch.ones((b, 1), device=self.device) * sde_ts[i]
            pred = velocity_net(x, t)
            if i in id_to_log:
                log_x[:, index] = x.clone()
                if save_pred: log_pred[:, index] = pred
                index += 1
            if i < num_steps:
                x += pred * dt[i]
                x += self.sigma * torch.randn_like(x, device=self.device) * torch.sqrt(dt[i])
        return log_x, log_pred, sde_ts[id_to_log]


class Twisted_BM_ZeroCost(Twisted_BM):
    """SB class associated to DSBM (zero-potential case)."""

    def __init__(self, t_final, sigma, shape_x, loss_param, device):
        super().__init__(t_final, sigma, shape_x, loss_param, device)

    def get_train_target_from_bridge(self, x0, x1, z_noise_t, t, fb):
        """
        Computes the target of the bridge matching loss.

        - x0: (B, dim)
        - x1: (B, dim)
        - z_t: (B, dim)
        - z_noise_t: (B, dim)
        - t: (B, 1)
        - fb: 'f' or 'b'

        Returns:
            - target : (B, dim)
        """

        if self.loss_param == 'denoiser':
            target = x1 if fb == 'f' else x0
        else:
            if fb == 'f':
                # (x1-z_t)/(T-t) : compact expression
                target = (x1 - x0) / self.T
                target = target - self.sigma * torch.sqrt((t / self.T) / (self.T - t)) * z_noise_t
            else:
                # (x0-z_t)/t : compact expression
                target = (x0 - x1) / self.T
                target = target - self.sigma * torch.sqrt((1. - t / self.T) / t) * z_noise_t
        return target


class Twisted_BM_GeneralCost(Twisted_BM):
    """SB class associated to TSBM and GSBM."""

    def __init__(self, state_cost, method, t_final, sigma, shape_x, loss_param, device):
        super().__init__(t_final, sigma, shape_x, loss_param, device)
        self.V = state_cost
        assert method in ['tsbm', 'gsbm']
        self.method = method
        self.max_t_cost = self.V.max_t_cost
        self.min_t_cost = self.V.min_t_cost

    def get_train_target_from_cond_sampler(self, x0, x1, z_t, t, fb, cond_sampler,
                                           z_s=None,
                                           s=None,
                                           s_times=None,
                                           mask_s=None,
                                           z_cv_s=None,
                                           cv_s_times=None,
                                           coeff_cv_s=None,
                                           cv_net=None,
                                           factor_state_cost=1.0,
                                           ):
        """
        Computes the target of the bridge matching loss.

        - x0: (B, dim)
        - x1: (B, dim)
        - z_t: (B, dim)
        - t: (B, 1)
        - fb: 'f' or 'b'
        - cond_sampler: spline learned upon the B pairs (x0,x1)
        - z_s: (B, S, dim) : particles associated to cost times s
        - s: (B, S, 1) : indexes if discrete-time cost // times if continuous-time cost
        - s_times: (B, S, 1) : times associated to s (s=s_times if continuous-time cost)
        - mask_s: (B, S, 1) : whether to s is a valid time for computing grad cost
        - z_cv_s: (B, S, dim) : CV particles associated to cost times cv_s_times
        - cv_s_times: (B, S, 1) : CV times
        - cv_s_times: (B, S, 1) : loss coefficients associated to CV times
        - cv_net: CV net
        - factor_state_cost : float in [0,1] : annealing parameter used to scale the cost term in control variate target

        - target : (B, dim) if method is 'gsbm' else (B*S, dim)
        """
        B = x0.shape[0]
        size_no_expand = (-1,) * len(self.d_x)

        # compute GSBM target
        with torch.no_grad():
            z_t_mult = z_t.view(B, 1, 1, *self.d_x).expand(-1, -1, B, *size_no_expand)
            target_gsbm = cond_sampler.drift(t.squeeze(-1), z_t_mult, fb)[torch.arange(B), 0, torch.arange(B)]

        if self.method == 'gsbm':
            target = target_gsbm
            assert target.shape == (B, *self.d_x)
        else:
            assert z_s is not None and s is not None and mask_s is not None
            assert z_s.shape[:2] == s.shape[:2] == mask_s.shape[:2]
            S = z_s.shape[1]

            # expand all quantities to B*S
            minus_ones = (-1,) * len(self.d_x)
            x0 = x0.unsqueeze(1).expand(-1, S, *minus_ones).reshape(B * S, *self.d_x)
            x1 = x1.unsqueeze(1).expand(-1, S, *minus_ones).reshape(B * S, *self.d_x)
            z_t = z_t.unsqueeze(1).expand(-1, S, *minus_ones).reshape(B * S, *self.d_x)
            t = t.unsqueeze(1).expand(-1, S, -1).reshape(B * S, 1)
            z_s = z_s.reshape(B * S, *self.d_x)
            s = s.reshape(B * S, 1)  # times (continuous time V) or indexes (discrete time V)
            s_times = s_times.reshape(B * S, 1)
            mask_s = mask_s.reshape(B * S, 1).to(x0.dtype)
            if cv_s_times is not None:
                # discrete-time cost & CV
                assert z_cv_s is not None and coeff_cv_s is not None
                cv_s_times = cv_s_times.reshape(B * S, 1)
                coeff_cv_s = coeff_cv_s.reshape(B * S, 1)
                z_cv_s = z_cv_s.reshape(B * S, *self.d_x)
                coeff_cv_target_s = self.V.get_num_cost_times_per_t(t)
                disc_cost_times = self.V.y_times
            else:
                # for continuous-time cost, s=cv_s_times
                cv_s_times = s_times.clone()
                z_cv_s = z_s.clone()
                coeff_cv_s = self.T - t if fb == 'f' else t
                coeff_cv_target_s = self.T - t if fb == 'f' else t
                disc_cost_times = None

            # auto-diff to compute the gradient of the potential
            z_s_inp = z_s.detach().requires_grad_(True)
            cost_eval = self.V(xt=z_s_inp,
                               t=s.squeeze(-1),
                               cond_sampler=cond_sampler,
                               imf_train=True)
            grad_cost = torch.autograd.grad(cost_eval.sum(), z_s_inp, create_graph=False, retain_graph=False, )[0]
            assert x0.shape == x1.shape == z_t.shape == grad_cost.shape
            grad_cost = grad_cost.detach()
            if cv_net is not None:
                # evaluate cv_net in s and t
                A_cv, B_cv, C_cv = cv_net(t,
                                          cv_s_times,
                                          fb=fb,
                                          target_s=s,
                                          disc_cost_times=disc_cost_times)  # each (B*S,1)
                # continuous time : cv_s_times=s: times
                # discrete time : cv_s_times: times, s: indexes
            if self.loss_param == 'denoiser':
                if fb == 'f':
                    mask_cost = (t / self.T < self.max_t_cost).to(x1.dtype)
                    target = x1
                    target -= mask_cost * factor_state_cost * mask_s * (self.T - s_times) * (self.T - t) * grad_cost
                    if cv_net is not None:
                        # first cv term : - A_cv * (T-t) * (x1-z_t)
                        target_cv = -A_cv * (self.T - t) * (x1 - z_t)

                        # second cv term : - B_cv * (T-t)**2 * (x1-z_s)
                        target_cv -= B_cv * coeff_cv_s * (self.T - t) * (x1 - z_cv_s)

                        # third term : -C_cv * valid_mask_s * (T-t)**2 * (T-s_cost) * nabla V_s_cost(z_s_cost)
                        target_cv -= mask_cost * factor_state_cost * mask_s * C_cv * coeff_cv_target_s * (self.T - t) * (
                                self.T - s_times) * grad_cost

                        # add to target
                        target += target_cv
                else:
                    mask_cost = (t / self.T > self.min_t_cost).to(x0.dtype)
                    target = x0
                    target -= mask_cost * factor_state_cost * mask_s * s_times * t * grad_cost
                    if cv_net is not None:
                        # first cv term : - A_cv * t * (x0-z_t)
                        target_cv = -A_cv * t * (x0 - z_t)

                        # second cv term : + B_cv * t**2 * (x0-z_s)
                        target_cv += B_cv * coeff_cv_s * t * (x0 - z_cv_s)

                        # third term : + C_cv * valid_mask_s * t**2 * s_cost * nabla V_s_cost(z_s_cost)
                        target_cv += mask_cost * factor_state_cost * C_cv * coeff_cv_target_s * t * mask_s * s_times * grad_cost

                        # add to target
                        target += target_cv
            else:
                target_gsbm = target_gsbm.unsqueeze(1).expand(-1, S, *size_no_expand).reshape(B * S, *self.d_x)
                if fb == 'f':
                    mask_cost = (t / self.T < self.max_t_cost)
                    target = (x1 - z_t) / (self.T - t)
                    target -= mask_cost.to(x1.dtype) * factor_state_cost * mask_s * (self.T - s_times) * grad_cost
                    if cv_net is not None:
                        # first cv term : - A_cv * (x1-z_t)
                        target_cv = - A_cv * (x1 - z_t)

                        # second cv term : - B_cv * (T-t) * (x1-z_s)
                        target_cv -= B_cv * coeff_cv_s * (x1 - z_cv_s)

                        # third term : - C_cv * valid_mask_s * (T-t) * (T-s_cost) * nabla V_s_cost(z_s_cost)
                        target_cv -= mask_cost.to(x1.dtype) * factor_state_cost * C_cv * mask_s * coeff_cv_target_s * (
                                self.T - s_times) * grad_cost

                        # add to target
                        target += target_cv

                else:
                    mask_cost = (t / self.T > self.min_t_cost)
                    target = (x0 - z_t) / t
                    target -= mask_cost.to(x0.dtype) * factor_state_cost * mask_s * s_times * grad_cost

                    if cv_net is not None:
                        # first cv term : - A_cv * (x0-z_t)
                        target_cv = - A_cv * (x0 - z_t)

                        # second cv term : + B_cv * t * (x0-z_s)
                        target_cv += B_cv * coeff_cv_s * (x0 - z_cv_s)

                        # third term : + C_cv * t * s_cost * nabla V_s_cost(z_s_cost)
                        target_cv += mask_cost.to(
                            x0.dtype) * factor_state_cost * C_cv * mask_s * coeff_cv_target_s * s_times * grad_cost

                        # add to target
                        target += target_cv
                mask_cost = mask_cost.squeeze(-1)
                # replace TSBM target by GSBM target outside the mask
                target[~mask_cost] = target_gsbm[~mask_cost]
            assert target.shape == (B * S, *self.d_x)
        return target
