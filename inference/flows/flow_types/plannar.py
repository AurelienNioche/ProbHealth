import torch
from torch import nn
from . utils import safe_log


class PlanarFlow(nn.Module):
    """
    Planar flow.
    z = f(x) = x + u h(wᵀx + b)
    """

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(1, dim))
        self.bias = nn.Parameter(torch.empty(1))
        self.scale = nn.Parameter(torch.empty(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.uniform_(self.weight, -0.01, 0.01)
        nn.init.uniform_(self.scale, -0.01, 0.01)
        nn.init.uniform_(self.bias, -0.01, 0.01)

    def forward(self, x):

        activation = self.tanh(x@self.weight.t() + self.bias)

        f = x + self.scale * activation

        psi = (1 - activation ** 2) * self.weight
        det_grad = 1 + psi@self.scale.t()
        ld = safe_log(det_grad.abs()).squeeze(-1)

        return f, ld



