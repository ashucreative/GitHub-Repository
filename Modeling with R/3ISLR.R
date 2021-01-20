##simple Linear Regression
##Example
install.packages("ISLR")
library(ISLR)

data(Credit)
## here we are looking at two variables
#1. balance on credit cards
#2.Income.
# We are interested in trying to predict the balance from income

plot(Credit$Income, Credit$Balance)

model = lm(Balance~Income, data = Credit)
s = summary (model)
s

## general comment: to check all available entries in an object

names(model)
names(s)

## Estimated coefficients (least sq estimates)

model$coefficients

##pval's
# pval for the income coefficient
s$coefficients
s$coefficients [2,4]

# interpret this number?

# the probability of observing the given data from a population
# where income and balance are unrelated (~0). Since, this probability is small (<0.05) thus income
# has a significant association with balance (b1 != 0)

plot (Credit$Income, Credit$Balance)
abline(model)
abline(model, lwd = 3, col = "red")
?abline

#confidence intervals for b0 and b1 (confint())

confint(model)
confint(model, level = 0.9)

## you have to run the plot first before 
# check if this matches up with our class results
# CI for b1: [hb1-t*se, hb1+t*se]

x = Credit$Income; y = Credit$Balance

hb1 = (t(x-mean(x))%*%(y-mean(y)))/(t(x-mean(x))%*%(x-mean(x)))
       
hb0=mean(y)-hb1*mean(x)
hb1; hb0
s_xx = (t(x-mean(x))%*%(x-mean(x)))
n = length(y)
MSE = (sum((y-c(hb0) - c(hb1)*x)^2))/(n-2)
alpha = .05
t_q = qt(1-alpha/2, df = 398) #use qt() for quantiles of t distribution
SE_hb1 = sqrt(MSE/s_xx)
CI = c(hb1-t_q*SE_hb1, hb1+t_q*SE_hb1)
CI

## Detour:
# Sample from a t distribution

k = 1000
a = rt(k, df=15)
hist (a, breaks = 30)
qt(0.025, df = 15)

#### COming back to our exampls

# PRediction intervals
# Now that we have a model predicting balance from income of an individual
# we can predict the balance for any new individual whose income we know
# e.g. suppose botir's income is $200000, what is his cc balance

predict(model, data.frame(Income = 200), interval = "prediction", data = Credit) #Interval for one person

# when you use the argument interval = "prediction" we obtain a confidence interval for a single
# prediction

predict(model, data.frame(Income = 200), interval = "confidence", data = Credit) #Interval for average of everybody

## alternative

library(MASS)
data(Boston)
?Boston

#medv as the response
#lstat as the predictor

plot(Boston$medv, Boston$lstat)

model = lm(medv~lstat, data = Boston)
s = summary (model)
s
s$coefficients
s$coefficients [2,4]

abline(model)
confint(model)
confint(model, level = 0.95)

### R^2: coefficient of determination

#simulated data
# case 1:
n = 100; b0 = 1; b1 = 1
x=rnorm(n); ep = rnorm(n)
y = b0 + b1*x + ep
plot(x,y)
m2 = lm(y~x)
summary(m2)

# case 2:
n = 100; b0 = 1; b1 = 0
x=rnorm(n); ep = rnorm(n)
y = b0 + b1*x + ep
plot(x,y)
m2 = lm(y~x)
summary(m2)

#case 3:
n = 100; b0 = 1; b1 = 3
x=rnorm(n); ep = rnorm(n)
y = b0 + b1*(x^2) + ep
plot(x,y)
m2 = lm(y~x)
summary(m2)

