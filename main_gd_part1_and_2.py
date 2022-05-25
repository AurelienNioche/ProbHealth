import torch
from tqdm import tqdm
import numpy as np


torch.manual_seed(123)
np.random.seed(123)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # self.theta1 = torch.nn.Parameter(torch.ones((2, 3)))
        self.theta2 = torch.nn.Parameter(torch.ones(3))
        # self.cst = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, m):

        n_obs = m.shape[0]

        # m_hat = torch.zeros((n_obs, 3))
        # for j in range(3):
        #     m_hat[:, j] += self.theta1[0, j] + x * self.theta1[1, j]

        y_hat = torch.zeros(n_obs) # + self.cst
        for j in range(3):
            y_hat[:] += m[:, j] * self.theta2[j]

        return None, y_hat


def get_data():

    n_obs = 1000

    theta1 = np.array([[1., 2., 4.], [-3., -2., 4.]])
    theta2 = np.array([5., -3., 4.])
    # cst = -3.

    # m = np.random.random(size=(n_obs, 3))

    x = np.random.random(size=n_obs)

    m = np.zeros((n_obs, 3))
    for j in range(3):
        m[:, j] = theta1[0, j] + x * theta1[1, j]

    y = np.zeros(n_obs)  # + cst
    for j in range(3):
        y[:] += m[:, j] * theta2[j]

    x = None # torch.from_numpy(x).float()
    m = torch.from_numpy(m).float()
    y = torch.from_numpy(y).float()

    return x, m, y


def main():

    epochs = 10000

    x, m, y = get_data()

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # with tqdm(total=epochs) as pbar:

    for e in range(epochs):

        optimizer.zero_grad()

        m_hat, y_hat = model(x, m)
        loss = torch.mean((y_hat - y)**2)

        loss.backward()
        optimizer.step()

        print(model.theta2.detach().numpy())

            # pbar.set_postfix({'loss': loss.item()})
            # pbar.update()

    # print(model.theta1)
    print(model.theta2)
    # print(model.cst)


if __name__ == "__main__":
    main()
