import torch

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

import numpy as np


pyro.set_rng_seed(0)


def model(x, y):

    b = pyro.sample("b", dist.Normal(torch.zeros(2), torch.ones(2)))

    X = x.reshape((len(x), 1))
    X = torch.hstack([torch.ones_like(X), X])

    theta = X@b

    sigma = pyro.sample("sigma", dist.HalfCauchy(0.02))

    return pyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def get_data():

    n_obs = 500

    theta = np.array([2., 3.])

    x = np.random.random(n_obs)
    X = x.reshape((n_obs, 1))
    X = np.hstack([np.ones_like(X), X])
    y = X@theta

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return x, y


def main():

    x, y = get_data()

    num_samples = 500
    warmup_steps = 500
    num_chains = 1

    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )

    mcmc.run(x, y)
    mcmc.summary(prob=0.5)


if __name__ == "__main__":
    main()
