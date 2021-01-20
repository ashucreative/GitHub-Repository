##Multiple Linear Regression

library(MASS)
data (Boston)
?Boston

##response y
##predictors: all remaining
#variables in the data
model = lm(medv~.,data = Boston)
summary(model)

##
names(Boston)
mod2 = lm(medv~crim+lstat+zn, data = Boston)
summary(mod2)

### Diagnostics (Checking for potential problems)

## non linearity
library(ISLR)
data(Auto)
mod3 = lm(mpg~horsepower, data = Auto)
summary(mod3)

plot(Auto$horsepower, Auto$mpg)
residual = mod3$residuals
predicted = mod3$fitted.values

##residual plot
plot(predicted, residual)
abline(h=0)
## here we see that there might be non-linearity in error terms

## Lets try to introduce a quadratic term
# model now is y=b0+b1+b2*x^2+eps
mod4 = lm(mpg~horsepower+I(horsepower^2), data = Auto)
summary(mod4)
resid_quad = mod4$residuals
pred_quad = mod4$fitted.values
plot(pred_quad, resid_quad)
abline(h=0)
## Now it looks like problem of non-linearity is taken care of
# It still looks like we have to test for heteroscedasticity

# Breusche-Pagan Test
install.packages("lmtest")
library(lmtest)
bptest(mod4)
data (Auto)
lmpg = log(Auto$mpg)
Auto$lmpg = lmpg

mod5 = lm(lmpg~horsepower+I(horsepower^2), data = Auto)
resd5 = mod5$residuals
pred5 = mod5$fitted.values
plot(pred5, resd5)
abline (h=0)
bptest(mod5)
summary(mod5)

#### Checking normality

qqnorm(resd5)
qqline (resd5)

shapiro.test(resd5)
## We get non-normal but qq-plot was enough for normality.
## We check for outliers to alleviate this problem
# Now we can check by studentized residuals

stres = studres(mod5)
plot(pred5,stres)

# identify potential outliers
idx_outliers = which (abs(stres)>3)

## remove these three data points and create a new data set
newdat = Auto[-idx_outliers,]
mod6 = lm(lmpg ~ horsepower + I(horsepower^2), data = newdat)
summary(mod6)
shapiro.test(mod6$residuals)
resd6 = mod6$residuals
qqnorm(resd6)
qqline (resd6)

## To check for high leverage (Unusual x observations)
# Use cooks.distance
cooks = cooks.distance(mod6)
plot(cooks)

## none of these values are above 0.5 or 1 so there is no leverage problem here
#########################
#Now we want to see the effect of correlation on residual plots
# Get some data
install.packages("quantmod")
library(quantmod)
tickrs=new.env()
hist = 365
today = Sys.Date()
start_date = as.Date(today)-hist
####
tickr = c("AMZN", "GOOGLE")
getSymbols(tickr, env=tickrs, src="yahoo", from = start_date, to = today)

Amazon_stock = getElement(tickrs, "AMZN")
google_stock = getElement(tickrs, "GOOGL")

sdata = data.frame(AM)