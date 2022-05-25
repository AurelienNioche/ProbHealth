import torch
from tqdm import tqdm
import numpy as np


torch.manual_seed(0)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones((2, 3)))

    def forward(self, x):

        n_obs = x.shape[0]

        y_hat = torch.zeros((n_obs, 3))
        for j in range(3):
            y_hat[:, j] += self.theta[0, j] + x[:] * self.theta[1, j]

        return y_hat


def get_data():

    n_obs = 1000

    theta = np.array([[1., 2., 4.], [-3., -2., 4.]])

    x = np.random.random(size=n_obs)

    y = np.zeros((n_obs, 3))
    for j in range(3):
        y[:, j] = theta[0, j] + x * theta[1, j]

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


if __name__ == "__main__":
    main()
