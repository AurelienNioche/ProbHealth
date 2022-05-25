import os
import torch
import torch.distributions as dist
from torch import nn
from typing import Any, Dict

from . flow_types.plannar import PlanarFlow
from . flow_types.radial import RadialFlow


def safe_log(z):
    return torch.log(z + 1e-7)


class NormalizingFlows(nn.Module):
    """
    Adapted from https://github.com/ex4sperans/variational-inference-with-normalizing-flows

    [Rezende and Mohamed, 2015]

    """

    BKP_DIR = "bkp"

    def __init__(self, dim, flow_length, flow_model=None):
        super().__init__()
        if flow_model is None:
            self.flow_model = PlanarFlow
        elif isinstance(flow_model, str):
            self.flow_model = {
                cls.__name__: cls for cls in [PlanarFlow, RadialFlow]
            }[flow_model]
        else:
            self.flow_model = flow_model

        self.transforms = nn.ModuleList([
            self.flow_model(dim) for _ in range(flow_length)
        ])

        self.dim = dim
        self.flow_length = flow_length

        self.mu = nn.Parameter(torch.zeros(dim).uniform_(-0.01, 0.01))
        self.log_var = nn.Parameter(torch.zeros(dim).uniform_(-0.01, 0.01))

    def sample_base_dist(self, batch_size):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn((batch_size, self.dim))
        return self.mu + eps * std

    def log_prob_base_dist(self, x):
        std = torch.exp(0.5 * self.log_var)
        return dist.Normal(self.mu, std).log_prob(x).sum(axis=-1)

    def forward(self, x):

        log_prob_base_dist = self.log_prob_base_dist(x)

        log_det = torch.zeros(x.shape[0])

        for i in range(self.flow_length):
            x, ld = self.transforms[i](x)
            log_det += ld

        return x, log_prob_base_dist, log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.transforms[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.sample_base_dist(n_samples)
        x, _ = self.inverse(z)
        return x

    def _get_constructor_parameters(self) -> Dict[str, Any]:

        return dict(dim=self.dim,
                    flow_length=self.flow_length,
                    flow_model=self.flow_model)

    def save(self, path: str) -> None:
        """
        Save model to a given location.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                    "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str):
        """
        Load model from path.
        """
        saved_variables = torch.load(path)
        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        return model
