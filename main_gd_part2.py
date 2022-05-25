import torch
from tqdm import tqdm
import time
import numpy as np


torch.manual_seed(0)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones(3))
        self.cst = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):

        n_obs = x.shape[0]

        y_hat = torch.zeros(n_obs) + self.cst
        for j in range(3):
            y_hat[:] += x[:, j] * self.theta[j]

        return y_hat


def get_data():

    n_obs = 1000

    theta = np.array([5., -3., 4.])
    cst = -3.

    x = np.random.random((n_obs, 3))

    y = np.zeros(n_obs) + cst
    for j in range(3):
        y[:] += x[:, j] * theta[j]

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    return x, y


def main():

    epochs = 5000

    x, y = get_data()

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with tqdm(total=epochs) as pbar:

        for e in range(epochs):

            optimizer.zero_grad()

            y_hat = model(x)
            loss = torch.mean((y_hat - y)**2)

            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})
            pbar.update()

    print(model.theta)
    print(model.cst)


if __name__ == "__main__":
    main()
