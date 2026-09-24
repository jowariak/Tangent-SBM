import copy
import numpy as np
from tqdm import trange

from .interp1d import *

# Inspired from https://github.com/facebookresearch/generalized-schrodinger-bridge-matching/blob/main/gsbm/gaussian_path.py


class EndPointSpline(torch.nn.Module):
    def __init__(self, t, xt, spline_type="linear"):
        """
        t: (T,)
        xt: (B, T, D)
        """
        super(EndPointSpline, self).__init__()
        B, T, D = xt.shape
        assert t.shape == (T,) and T > 2, "Need at least 3 points"
        assert t.device == xt.device

        t = t.detach().clone()
        xt = xt.permute(1, 0, 2).detach().clone()

        # fix
        self.B = B  # number of (x0,x1) pairs
        self.T = T  # number controlled points / time steps
        self.D = D  # dimension
        self.spline_type = spline_type

        self.register_buffer("t", t)
        self.register_buffer("t_epd", t.view(-1, 1).expand(-1, B))
        self.register_buffer("x0", xt[0].view(1, B, D))
        self.register_buffer("x1", xt[-1].view(1, B, D))
        self.register_parameter("knots", torch.nn.Parameter(xt[1:-1]))

    @property
    def device(self):
        return self.parameters().__next__().device

    @property
    def xt(self):  # (B, T, D)
        return torch.cat([self.x0, self.knots, self.x1], dim=0).permute(1, 0, 2)

    def interp(self, query_t):
        """
        query_t: (S,) --> yt: (B, S, D)
        """

        (S,) = query_t.shape
        query_t = query_t.view(-1, 1).expand(-1, self.B)
        assert query_t.shape == (S, self.B)

        mask = None
        xt = torch.cat([self.x0, self.knots, self.x1], dim=0)  # (T, B, D)
        if self.spline_type == "linear":
            # print(self.t_epd.shape, self.t_epd.device)
            yt = linear_interp1d(self.t_epd, xt, mask, query_t)
        elif self.spline_type == "cubic":
            yt = cubic_interp1d(self.t_epd, xt, mask, query_t)
        yt = yt.permute(1, 0, 2)
        assert yt.shape == (self.B, S, self.D), yt.shape
        return yt

    def forward(self, t):
        """
        t: (S,) --> yt: (B, S, D)
        """
        return self.interp(t)


class StdSpline(EndPointSpline):
    def __init__(self, t, xt, sigma, t_final, spline_type="linear", grid=None):
        """
        t: (T,) : control timesteps
        xt: (B, T, 1) : control points for std (taken over batch of size B)
        """
        super(StdSpline, self).__init__(t, xt, spline_type=spline_type)
        assert self.D == 1
        self.sigma = sigma
        self.t_final = t_final
        self.softplus = torch.nn.Softplus()
        self.eps = 1e-4  # clamping value for 1/gamma^2 evaluation on the grid
        self.grid = grid

        if grid is not None:
            self.n_grid = grid.shape[0]

    def forward(self, t):
        """
        t: (S,) --> yt: (B, S, 1)
        """
        base = self.sigma * (t * (self.t_final - t) / self.t_final).sqrt()
        xt = self.interp(t)
        return base.view(1, -1, 1) * self.softplus(xt) / self.softplus(torch.zeros_like(xt))

    @torch.no_grad()
    def build_grid(self):
        """
        Builds in place
        - self.F_grid (B,n_grid) : the cumulative increments of 1/StdSpline(u)^2 for u in the grid

        Call this once after gamma_module parameters are set.
        Rebuild if gamma_module changes.
        """
        assert self.grid is not None, 'Time grid is not built.'

        gamma = self.forward(self.grid).squeeze(-1)  # (B,G)
        B, G = gamma.shape

        gamma = gamma.clamp_min(self.eps)
        f = 1 / (gamma.square())  # (B, G)
        dt = self.grid[1:] - self.grid[:-1]  # (G-1,)
        trap = 0.5 * (f[:, :-1] + f[:, 1:]) * dt.unsqueeze(0)  # (B, G-1)

        F_grid = torch.empty((B, G), device=self.device)
        F_grid[:, 0] = 0.0
        F_grid[:, 1:] = torch.cumsum(trap, dim=1)

        self.F_grid = F_grid  # (B,G)

    def _interp_prefix(self, t):
        """
        Args:
            - t : (S,)

        Computes J(0, t) = ∫_s^t 1/StdSpline(u)^2 du over the batch.

        Returns:
            - J : (B,*S)
        """
        if self.F_grid is None:
            raise RuntimeError("Call self.build_grid() before evaluating integrals.")

        t = t.clamp(0.0, self.t_final)

        q_shape = t.shape
        t_flat = t.reshape(-1)  # (N,)

        # Find enclosing grid interval
        idx = torch.searchsorted(self.grid, t_flat, right=False)  # (N,)
        idx = idx.clamp(1, self.n_grid - 1)

        t0 = self.grid[idx - 1]  # (N,)
        t1 = self.grid[idx]  # (N,)
        w = (t_flat - t0) / (t1 - t0)  # (N,)

        # Interpolate each batch row at the same indices
        y0 = self.F_grid[:, idx - 1]  # (B, N)
        y1 = self.F_grid[:, idx]  # (B, N)
        y = y0 + (y1 - y0) * w.unsqueeze(0)  # (B, N)

        return y.reshape(self.B, *q_shape)

    def compute_integral_inv_square(self, t, s):
        """
        Args:
            - t : (S,)
            - s : (S,)

        Computes J(s, t) = ∫_t^s 1/StdSpline(u)^2 du over the batch.

        Returns:
            - J : (B,*S)
        """
        assert t.shape == s.shape, 'Input times should have the same shape.'
        Fs = self._interp_prefix(s)  # (B, *S)
        Ft = self._interp_prefix(t)  # (B, *S)
        return Fs - Ft


################################################################################################


class EndPointGaussianPath(torch.nn.Module):
    def __init__(self, t, xt, s, ys, sigma, t_final, basedrift, device='cpu', grid_ys=None, F_grid_ys=None):
        super(EndPointGaussianPath, self).__init__()

        (B, T, D), (S,) = xt.shape, s.shape
        assert t.shape == (T,) and ys.shape == (B, S, 1)

        self.B = B  # number of (x0,x1) pairs
        self.T = T  # number controlled points for mean spline (includes boundaries)
        self.S = S  # number controlled points for std spline (includes boundaries)
        self.D = D  # dimension

        self.sigma = sigma
        self.t_final = t_final
        self.mean = EndPointSpline(t.to(device), xt.to(device))
        if grid_ys is not None:
            grid_ys = grid_ys.to(device)
        self.gamma = StdSpline(s.to(device), ys.to(device), sigma, t_final, grid=grid_ys)
        if F_grid_ys is not None:
            F_grid_ys = F_grid_ys.to(device)
        self.gamma.F_grid = F_grid_ys

        self.basedrift = basedrift

    @property
    def device(self):
        return self.parameters().__next__().device

    @property
    def mean_ctl_pts(self):
        return self.mean.xt.detach().cpu()

    @property
    def std_ctl_pts(self):
        return self.gamma(self.gamma.t).detach().cpu()

    def sample_xt(self, t, N):
        """
        N: number of xt for each (x0,x1)
        t: (T,) --> xt: (B, N, T, D)
        """

        mean_t = self.mean(t)  # (B, T, D)
        B, T, D = mean_t.shape

        assert t.shape == (T,)
        std_t = self.gamma(t).view(B, 1, T, 1)  # (B, 1, T, 1)

        noise = torch.randn(B, N, T, D, device=t.device)  # (B, N, T, D)

        xt = mean_t.unsqueeze(1) + std_t * noise
        assert xt.shape == noise.shape
        return xt

    def sample_s_given_t(self, t, zt, s, fb='f'):
        """
        Sample zs|t, zt (s>t if fb=='f', s<t if fb=='b')

        t: (B, N)
        zt: (B, N, dim)
        s: (B, N, S) -> multiple arrival times per each (t,z_t), can have the same value as t
                     -> B*N*S different times
        fb: 'f' or 'b'

        --> zs: (B, N, S, dim)
        """
        assert fb in ['f', 'b', 'fb']
        B, N, D = zt.shape
        assert s.shape[:2] == (B, N) and t.shape == (B, N)
        S = s.shape[2]

        t = t.unsqueeze(-1).expand(-1, -1, S).reshape(B * N * S)  # (B*N*S,)
        s = s.reshape(B * N * S)  # (B*N*S,)
        zt = zt.view(B, 1, N, 1, D).expand(-1, B, -1, S, -1).reshape(B, B * N * S, D)  # (B,B*N*S,D)

        mean_t, mean_s = self.mean(t), self.mean(s)  # (B, B*N*S, D)
        gamma_t, gamma_s = self.gamma(t), self.gamma(s)  # (B, B*N*S, 1)
        time_pair = (t, s) if fb == 'f' else (s, t)
        J_t_s = -0.5 * self.sigma ** 2 * self.gamma.compute_integral_inv_square(*time_pair)  # (B, B*N*S)
        J_t_s = J_t_s.unsqueeze(-1)  # (B, B*N*S, 1)

        # mean
        if self.sigma > 0:
            mean_t_s = mean_s + gamma_s * ((zt - mean_t) / gamma_t) * torch.exp(J_t_s)
        else:
            mean_t_s = mean_s + (zt - mean_t)
        # (B, B*N*S, D)
        mean_t_s = mean_t_s.reshape(B, B, N * S, D)
        mean_t_s = mean_t_s[torch.arange(B), torch.arange(B)].reshape(B, N, S, D)
        assert mean_t_s.shape == (B, N, S, D)

        # std
        std_t_s = gamma_s
        coeff_std_sq = -torch.expm1(2 * J_t_s)
        std_t_s *= torch.sqrt(torch.clamp(coeff_std_sq, min=0.0))
        # (B, B*N*S, 1)
        std_t_s = std_t_s.reshape(B, B, N * S, 1)
        std_t_s = std_t_s[torch.arange(B), torch.arange(B)].reshape(B, N, S, 1)
        assert std_t_s.shape == (B, N, S, 1)

        # compute zs|zt
        zs = mean_t_s
        zs += std_t_s * torch.randn_like(mean_t_s)

        return zs

    def ft(self, t, xt, fb):
        """
        Base drift in the direction fb evaluated at (t,xt).

        t: (T,)
        xt: (B, N, T, D)
        fb: 'f' or 'b'
        ===
        ft: (B, N, T, D)
        """
        B, N, T, D = xt.shape
        assert t.shape == (T,)

        x0 = self.mean.x0.view(B, 1, 1, D)
        x1 = self.mean.x1.view(B, 1, 1, D)

        t = t.view(1, 1, T, 1)

        ft = self.basedrift(xt, t, fb, x0, x1, self.sigma, self.t_final)
        return ft

    def drift(self, t, xt, fb):
        """
        Drift of stochastic interpolants in the direction fb evaluated at (t,xt).

        t: (T,)
        xt: (B, N, T, D)
        fb: 'f' or 'b'
        ===
        drift: (B, N, T, D)
        """
        assert (t > 0).all() and (t < self.t_final).all()

        B, N, T, D = xt.shape
        assert t.shape == (T,)

        mean, dmean = torch.autograd.functional.jvp(
            self.mean, t, torch.ones_like(t), create_graph=self.training
        )
        assert mean.shape == dmean.shape == (B, T, D)

        dmean = dmean.view(B, 1, T, D)
        mean = mean.view(B, 1, T, D)

        std, dstd = torch.autograd.functional.jvp(
            self.gamma, t, torch.ones_like(t), create_graph=self.training
        )
        assert std.shape == dstd.shape == (B, T, 1)

        if fb == 'f':
            # u = ∂m + a (x - m),
            # a = (\dot γ - σ^2 / 2γ) / γ
            a = (dstd - self.sigma ** 2 / (2. * std)) / std
            if self.sigma == 0:
                a = torch.zeros_like(a)  # handle deterministic cases
            drift = dmean + a.view(B, 1, T, 1) * (xt - mean)
        else:
            # u = -∂m + a (x - m),
            # a = (-\dot γ - σ^2 / 2γ) / γ
            a = (-dstd - self.sigma ** 2 / (2. * std)) / std
            if self.sigma == 0:
                a = torch.zeros_like(a)  # handle deterministic cases
            drift = -dmean + a.view(B, 1, T, 1) * (xt - mean)

        assert drift.shape == xt.shape
        return drift

    def ut(self, t, xt, fb):
        """
        Additive control to estimate in the direction fb, evaluated at (t,xt).

        t: (T,)
        xt: (B, N, T, D)
        fb: 'f' or 'b'
        ===
        ut: (B, N, T, D)
        """
        ft = self.ft(t, xt, fb)
        drift = self.drift(t, xt, fb)
        assert drift.shape == ft.shape == xt.shape
        return drift - ft

    def forward(self, t, N, fb):
        """
        Samples xt and returns (xt, ut(t,xt)).

        t: (T,)
        fb: 'f' or 'b'
        ===
        xt: (B, N, T, D)
        ut: (B, N, T, D)
        """
        xt = self.sample_xt(t, N)

        B, N, T, D = xt.shape
        assert t.shape == (T,)

        ut = self.ut(t, xt, fb)
        assert ut.shape == xt.shape

        return xt, ut


################################################################################################

def build_loss_fn(gpath, V, discrete_setting=None):
    if discrete_setting is None:
        def loss_fn(t, xt, ut):
            B, N, T, D = xt.shape
            assert t.shape == (T,) and ut.shape == (B, N, T, D)

            cost_s = V(xt, t, cond_sampler=gpath).view(B, N, T)
            cost_c = 0.5 * (ut ** 2).sum(dim=-1)
            assert cost_s.shape == cost_c.shape == (B, N, T)
            return (cost_s + cost_c).mean()
    else:
        def loss_fn(s, xs, ut):
            B, N, T, D = ut.shape
            assert s.shape == (V.K,) and xs.shape == (B, N, V.K, D)

            # all cost times are visited exactly once
            index_s = torch.arange(V.K)
            cost_s = V(xs, index_s, cond_sampler=gpath).view(B, N, V.K).sum(dim=-1)
            cost_c = 0.5 * (ut ** 2).sum(dim=-1).mean(dim=-1)
            assert cost_s.shape == cost_c.shape == (B, N)
            return (cost_s + cost_c).mean()

    return loss_fn


def build_optim(gpath, ccfg):
    if ccfg.optim == "sgd":
        return torch.optim.SGD(
            [
                {"params": gpath.mean.parameters(), "lr": ccfg.lr_mean},
                {"params": gpath.gamma.parameters(), "lr": ccfg.lr_gamma},
            ],
            momentum=ccfg.momentum,
        )
    elif ccfg.optim == "adam":
        return torch.optim.Adam(
            [
                {"params": gpath.mean.parameters(), "lr": ccfg.lr_mean},
                {"params": gpath.gamma.parameters(), "lr": ccfg.lr_gamma},
            ],
        )
    else:
        raise ValueError(f"Unsupported Spline optimizer {ccfg.optim}!")


def fit(ccfg, gpath, fb, loss_fn, cost_name, discrete_cfg=None, first_it=False, eps=0.001, verbose=False):
    """
    V: xt: (*, T, D), t: (T,), gpath --> (*, T)
    """
    assert fb in ['f', 'b', 'fb']

    results = {"name": cost_name}
    results["init_mean"] = gpath.mean_ctl_pts.cpu()
    results["init_gamma"] = gpath.std_ctl_pts.cpu()

    B, D, N, T, device = gpath.B, gpath.D, ccfg.N, ccfg.T, gpath.device
    optim = build_optim(gpath, ccfg)

    gpath.train()
    num_steps = ccfg.nitr_first_it if first_it else ccfg.nitr
    losses = np.zeros(num_steps)
    bar = trange(num_steps) if verbose else range(num_steps)
    for itr in bar:
        optim.zero_grad()

        t = torch.linspace(2 * eps, gpath.t_final - 2 * eps, T, device=device)
        if fb == "fb":
            xt_f, ut_f = gpath(t, N, "f")
            xt_b, ut_b = gpath(t, N, "b")

            if discrete_cfg is not None:
                s = discrete_cfg['cost_times']
                K = discrete_cfg['num_cost_times']
                xs_f, _ = gpath(s, N, "f")
                xs_b, _ = gpath(s, N, "b")
                assert xs_f.shape == xs_b.shape == (B, N, K, D)
                loss = 0.5 * loss_fn(s, xs_f, ut_f) + 0.5 * loss_fn(s, xs_b, ut_b)
            else:
                loss = 0.5 * loss_fn(t, xt_f, ut_f) + 0.5 * loss_fn(t, xt_b, ut_b)

        else:
            xt, ut = gpath(t, N, fb)
            assert xt.shape == ut.shape == (B, N, T, D)

            if discrete_cfg is not None:
                s = discrete_cfg['cost_times']
                K = discrete_cfg['num_cost_times']
                xs, _ = gpath(s, N, fb)
                assert xs.shape == (B, N, K, D)
                loss = loss_fn(s, xs, ut)
            else:
                loss = loss_fn(t, xt, ut)

        loss.backward()
        optim.step()
        losses[itr] = loss.cpu().item()
        if verbose:
            bar.set_description(f"spline loss={losses[itr]}")

    gpath.eval()

    results["final_mean"] = gpath.mean_ctl_pts.cpu()
    results["final_gamma"] = gpath.std_ctl_pts.cpu()
    results["gpath"] = copy.deepcopy(gpath).cpu()
    results["losses"] = losses

    return results
