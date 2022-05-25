import torch
from torch import nn
from . utils import safe_log


class RadialFlow(nn.Module):
    """
    Adapted from https://medium.com/swlh/normalizing-flows-are-not-magic-22752d0c924
    and from https://github.com/tonyduan/normalizing-flows
    z = f(x) = = x + β h(α, r)(z − z0)
    """

    def __init__(self, dim):
        super().__init__()

        self.z0 = nn.Parameter(torch.empty(dim))
        self.unc_alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))

        self.dim = dim

        self.softplus = torch.nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.uniform_(self.z0, -0.01, 0.01)
        nn.init.uniform_(self.unc_alpha, -0.01, 0.01)
        nn.init.uniform_(self.beta, -0.01, 0.01)

    def forward(self, x):

        dz = x - self.z0
        r = torch.norm(dz, dim=1).unsqueeze(1)
        alpha = self.softplus(self.unc_alpha)
        h = 1 / (alpha + r)
        bh = self.beta * h
        f = x + bh*dz

        hp = - 1 / (alpha + r) ** 2
        ld = \
            (self.dim - 1) * safe_log(torch.abs(1 + bh)) \
            + safe_log(torch.abs(1 + bh + self.beta*hp*r))
        ld = ld.squeeze(-1)
        return f, ld