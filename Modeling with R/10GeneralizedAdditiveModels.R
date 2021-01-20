library(ISLR)
data(Wage)

# predict wage using age
?loess

fit = loess(wage~age, span = 0.2, data = Wage)
# to predict a a new observation

predict(fit, newdata = list(age = 20))
attach(Wage)
age.range = range(age)
age.grid = seq(age.range[1], age.range[2])

pred = predict(fit, data.frame(age = age.grid))
plot(age, wage)
lines(age.grid, pred, col = "blue", lwd = 3)               

## Generalized additive models
?Wage
library(splines)
gaml = lm(wage~ns(age, df = 6)+ns(year, df = 6) + education, data = Wage) # Categorical var like education is non-linear by default.
summary(gaml)
installed.packages("gam")
library(gam)

#gam.m2 = gam(wage~ns(age, df = 6) + )

# Non-linear logistic regression
# Binary response and continuous predictor
install.packages("ISLR")
library(ISLR)
data(Wage)

fit.logistic = glm(I(wage>250)~poly(age,4), data = Wage, family = "binomial")
summary(fit.logistic)

## obtain a binary variable to work with
attach(Wage)
resp = I(wage>250)
resp[1:10]

fit.logistic = glm(I(wage>250)~poly(age,4), data = Wage, family = "binomial")
summary(fit.logistic)
library(splines)
fit.logistic.spline = glm(I(wage>250)~ns(age,df = 4), data = Wage, family = "binomial")
summary(fit.logistic.spline)

# Question from last class

install.packages("gam")
library(gam)
gam.model.lm = lm(wage~ns(age,6)+ns(year,6)+education, data = Wage)
summary(gam.model.lm)
gam.model = gam(wage~ns(age,6)+ns(year,6)+education, data = Wage)
summary(gam.model)

plot.Gam(gam.model)

data(Auto)
write.csv(Auto, "auto.csv")
