import tensorsem as ts

mod : "
  # three-factor model
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"

opts = ts.SemOptions()

model = ts.StructuralEquationModel(opt = opts)  # opts are of class SemOptions
optim = torch.optim.Adam(model.parameters())  # init the optimizer
for epoch in range(1000):
    optim.zero_grad()  # reset the gradients of the parameters
    Sigma = model()  # compute the model-implied covariance matrix
    loss = ts.mvn_negloglik(dat, Sigma)  # compute the negative log-likelihood, dat tensor should exist
    loss.backward()  # compute the gradients and store them in the parameter tensors
    optim.step()  # take a step in the negative gradient direction using adam