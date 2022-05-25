import torch
from tqdm import tqdm
import numpy as np


torch.manual_seed(0)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones((2, )))

    def forward(self, x):

        y_hat = self.theta[0] + x * self.theta[1]
    #     n_obs = x.shape[0]
    #     x_ext = torch.cat([torch.ones((n_obs, 1)), x.reshape((n_obs, 1))], axis=-1)
    #     y_hat = x_ext@self.theta
        return y_hat


def get_data():

    n_obs = 500

    theta = np.array([2., 3.])

    x = np.random.random(size=n_obs)
    # x_ext = np.hstack([np.ones((n_obs, 1)), x.reshape((n_obs, 1))])
    # y = x_ext@theta

    y = theta[0] + x * theta[1]

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

    print(model.theta.detach().numpy())


if __name__ == "__main__":
    main()
