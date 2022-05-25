import torch
from torch import nn


class FCNN(nn.Module):
    """
    Fully Connected Neural Network
    From https://github.com/tonyduan/normalizing-flows
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


def safe_log(z):
    return torch.log(z + 1e-7)
