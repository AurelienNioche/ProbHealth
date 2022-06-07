import numpy as np
import pandas as pd

n_obs = 1000

theta = np.array([[-3., -2., 4.], [5., -3., 4.]])

x = np.random.random(size=n_obs)

noise = np.random.normal(0, 0.01, size=(n_obs, 3))
noise2 = np.random.normal(0, 0.1, size=n_obs)

z = np.zeros((n_obs, 3))
for j in range(3):
    z[:, j] = x * theta[0, j] + noise[:, j]

y = np.zeros(n_obs) + noise2
for j in range(3):
    y[:] += z[:, j] * theta[1, j]

h, m, e = z[:, 0], z[:, 1], z[:, 2]

df = pd.DataFrame(dict(x=x, y=y, m=m, h=h, e=e))
df.to_csv('fake_no_int.csv', index=False)
