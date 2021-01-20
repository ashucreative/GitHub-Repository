## Implement bootstrap for inference

## Simulated data

n = 100
x = rnorm(n,1,1)
x

# Objective: to obtain a confidence interval for the underlying population mean.
xbar = mean(x)
xbar

# to estimate the variance of xbar we use bootstrap
boot = 3000
rxbar = c()
for(i in 1: boot){
  rs = sample(x,n, replace = T)
  rxbar[i] = mean(rs)
}

sdboot = sd(rxbar)
#then the CI is given by the usual expression
CI  = c(xbar - 1.96*sdboot, xbar + 1.96*sdboot)
CI

## Verifying coverage in a sumulation
nsim = 100
CI =matrix(0, nsim,2)
for(j in 1:nsim){
  x = rnorm(n,1,1)
  xbar = mean(x)
  rxbar = c()
  for(i in 1:boot){
    rs = sample(x, n, replace = T)
    rxbar[i] = mean(rs)
    }
  sdboot = sd(rxbar)
  #then the CI is given by the usual expression
  CI[j,] = c(xbar-1.96*sdboot, xbar+1.96*sdboot)
  }
## If coverage is 95% then 95% of all intervals should contain the population mean(i.e. 1)
sum(CI[,1]<1 & CI[,2]>1)

# We sad that our sample captured the population mean 95% of the times.
# Now lets try with other distribution

nsim = 1000
CI =matrix(0, nsim,2)
for(j in 1:nsim){
  x = rbinom(n,1,0.5)
  xbar = mean(x)
  rxbar = c()
  for(i in 1:boot){
    rs = sample(x, n, replace = T)
    rxbar[i] = mean(rs)
  }
  sdboot = sd(rxbar)
  #then the CI is given by the usual expression
  CI[j,] = c(xbar-1.96*sdboot, xbar+1.96*sdboot)
  }
## If coverage is 95% then 95% of all intervals should contain the population mean(i.e. 1)
sum(CI[,1]<0.5 & CI[,2]>0.5)

