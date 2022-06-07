import semopy
import pandas as pd

# data = pd.read_csv("fake.csv")
#
# # theta1 = np.array([[1., 2., 4.], [-3., -2., 4.]])
# # theta2 = np.array([5., -3., 4.])
# # cst = -3.
#
# print(data)
#
# desc = \
#   "h ~ x\n" \
#   "m ~ x\n" \
#   "e ~ x\n" \
#   "y ~ h + m + e" \

data = pd.read_csv('worland5.csv')

desc = \
    "read ~ ppsych + motiv" + "\n" + \
    "arith ~ motiv"

model = semopy.Model(desc, mimic_lavaan=False)
res = model.fit(data, obj='MLW', solver='SLSQP')
print(res)
print(model.inspect())

for name in model.matrices_names:
    print(name)
    print(getattr(model, f"mx_{name}"))
