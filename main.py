import torch

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

import numpy as np


pyro.set_rng_seed(0)


def model(X, y):

    theta1 = pyro.sample("theta1", dist.Normal(torch.zeros(6), torch.ones(6))).reshape((2, 3))
    theta2 = pyro.sample("theta2", dist.Normal(torch.zeros(6), torch.ones(6))).reshape((2, 3))

    n_obs = X.shape[0]

    m = X @ theta1
    M = m.reshape((n_obs, 3, 1))
    M = torch.cat([torch.ones_like(M), M], axis=-1)

    y_hat = torch.zeros(n_obs)
    for i in range(len(y)):
        y_hat[i] = torch.diag(M[i] @ theta2).sum()

    # sigma = pyro.sample("sigma", dist.LogNormal(torch.zeros(1), torch.ones(1)))

    return pyro.sample("obs", dist.Normal(y_hat, 0.001), obs=y)


def main():

    x = np.random.randint(13,  size=200)
    X = x.reshape((len(x), 1))
    X = np.hstack([np.ones_like(X), X])
    print("X shape", X.shape)

    theta1 = np.array([[1, 2, 4], [-3, -2, 4]])
    theta2 = np.array([[3, 4, 5], [5, -3, 4]])

    print("theta1 shape", theta1.shape)

    m = X@theta1
    M = m.reshape((len(x), 3, 1))
    M = np.concatenate([np.ones_like(M), M], axis=-1)

    y = np.zeros(len(x))
    for i in range(len(y)):
        y[i] = np.diag(M[i]@theta2).sum()

    num_samples = 1000
    warmup_steps = 1000
    num_chains = 1

    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    mcmc.run(X, y)
    mcmc.summary(prob=0.5)


if __name__ == "__main__":
    main()
