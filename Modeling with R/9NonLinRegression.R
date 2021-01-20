library(ISLR)
data(Wage)
?Wage
plot(Wage$age, Wage$wage) # This tells us about non-linear association
#Now our objective is to do analysis to describe about population
## Polynomial regression
fit = lm(wage~poly(age, 3), data = Wage)
summary(fit)

fit2 = lm(wage~poly(age, 3, raw = T), data = Wage) # Now we standardized all x variables by dividing with standard dev
summary(fit2)

# another equivalent way of implementing polynomial regression
fit3 = lm(wage~age + I(age^2) + I(age^3), data = Wage)
summary(fit3)

# visualize the model fit and obtain confidence bands
attach(Wage) # Now all variables are in the environment
agelims = range(age)
age.grid = seq(from = agelims[1], to = agelims[2], length.out = 100)
preds = predict(fit,newdata = list(age = age.grid), se = T)
se.bands = cbind(preds$fit-2*preds$se.fit, preds$fit+2*preds$se.fit)
plot(age,wage,xlim = agelims)
lines(age.grid, preds$fit, col="blue", lwd = 3)

matlines(age.grid, se.bands, col = "blue", lty = 3, lwd = 3)
# This tells us that the model is not doing a good job as the y value sharply declines as age increases to high level

fit1= lm(wage~age, data = Wage)
fit2= lm(wage~poly(age,2), data = Wage)
fit3= lm(wage~poly(age,3), data = Wage)
fit4= lm(wage~poly(age,4), data = Wage)
fit5= lm(wage~poly(age,5), data = Wage)

# LIkelihood ratio test
anova(fit1, fit2, fit3, fit4, fit5)
# only valid if we have nested models like these
anova(fit3, fit5)

# step function regression
# use cut() function to split the range of x into a certain number of pieces
cut(age,4)
table(cut(age,4))

fit = lm(wage~cut(age,4), data = Wage)
summary(fit)

#Splines
install.packages("splines")
library(splines)
# regression splines
fit = lm(wage~bs(age, knots = c(25,40, 60)), data = Wage) # 3 knots
summary(fit)

#alternatively
fit = lm(wage~bs(age, df = 6), data = Wage) # By default there are cubic settings, hence 3 parameters and 3 knots, that leads to dof = 6
summary(fit)
attr(bs(age, df = 6), "knots")


# Now we would like to see how well our predictions are

agelims = range(age)
age.grid = seq(from = agelims[1], to = agelims[2], length.out = 100)
preds = predict(fit,newdata = list(age = age.grid), se = T)
se.bands = cbind(preds$fit-2*preds$se.fit, preds$fit+2*preds$se.fit)
plot(age,wage,xlim = agelims)
lines(age.grid, preds$fit, col="blue", lwd = 3)
matlines(age.grid, se.bands, col = "blue", lty = 3, lwd = 3)

# Natural splines
fit = lm(wage~ns(age, df = 6), data = Wage)
summary(fit)

agelims = range(age)
age.grid = seq(from = agelims[1], to = agelims[2], length.out = 100)
preds = predict(fit,newdata = list(age = age.grid), se = T)
se.bands = cbind(preds$fit-2*preds$se.fit, preds$fit+2*preds$se.fit)
plot(age,wage,xlim = agelims)
lines(age.grid, preds$fit, col="blue", lwd = 3)
matlines(age.grid, se.bands, col = "blue", lty = 3, lwd = 3)
