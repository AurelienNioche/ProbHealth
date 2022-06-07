library("lavaan")

dat <- read.csv("fake_no_int.csv")
cov(dat)
mdl <- '
  h ~ x 
  m ~ x
  e ~ x
  y ~ h + m + e
'
fit <- sem(mdl, data=dat, likelihood="wishart")
summary(fit)
