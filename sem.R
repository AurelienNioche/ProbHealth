library("lavaan")

dat <- read.csv("fake.csv")
cov(dat)
mdl <- '
  h ~ 1 + x 
  m ~ 1 + x
  e ~ 1 + x
  y ~ 1 + h + m + e
'
fit <- sem(mdl, data=dat)
summary(fit)
