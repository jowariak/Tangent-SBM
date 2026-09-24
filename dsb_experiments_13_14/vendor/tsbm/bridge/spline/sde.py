import torch


class ZeroBaseDrift(torch.nn.Module):
    def __init__(self):
        super(ZeroBaseDrift, self).__init__()

    def forward(self, xt, t, fb, x0, x1, sigma, t_final):
        return torch.zeros_like(xt)


class BrownianBridgeDrift(torch.nn.Module):
    def __init__(self):
        super(BrownianBridgeDrift, self).__init__()

    def forward(self, xt, t, fb, x0, x1, sigma, t_final):
        if fb == 'f':
            return (x1 - xt) / (t_final - t)
        else:
            return (x0 - xt) / t
