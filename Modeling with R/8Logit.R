### Logistic Regression

install.packages("ISLR")
library(ISLR)
data("Default")
?Default

data("iris")

da
## Simulate a data set
n = 100
y = sample(c(rep(0,n/2), rep(1,n/2)),n)

x1 = c(rep("A", n/4), rep("B",n/4), rep("C", n/4), rep("D", n/4))
p = 10
x2 = matrix(rnorm(n*p), n, p)
x = data.frame(x1, as.matrix(x2))
dat = data.frame (resp=y, pred = x)

str(dat) # It tells the structure of data


## Wald's Test
mod1 = glm(resp~., data = dat, family = "binomial")
summary(mod1)

## The zvalue reported in the summary is the wald statistic which evaluates the hypothesis test

#H0: beta_k = 0 vs HA: beta_k !=0

# To test the categorical predictor (represented by three dummy variables and thus three coefficients)
# we can test H0: beta1 = beta2=beta3 = 0 vs HA: at least one is different
# this can be done using a LRT

install.packages ('lmtest')
library(lmtest)

mod.reduced = glm (resp ~. -pred.x1, data = dat, family = "binomial")
summary(mod.reduced)

# syntax: reduced model, full model
lrtest(mod.reduced, mod1)# we reject the null of reduced model is true.


## ROC curve
data(Smarket)
mod3 = glm (Direction ~ Lag1+Lag2+Volume, data = Smarket, family = "binomial")
probs = predict (mod3, type = "response")
probs[1:10]

t = 0.5
n = dim(Smarket)[1]
pred.class = rep("Down", n)
pred.class[probs>t] = "Up"

table(pred.class, Smarket$Direction)

TPM = (455 +131)/n
TPM

#Sensitivity (How many times I correctly classified in postive class) and
# specificity (How many times I correctly classified in negative class)
# Here Up is positive

p.ind = which(Smarket$Direction == "Up")
sens = mean(pred.class[p.ind] == Smarket$Direction[p.ind])
sens

neg.ind = which(Smarket$Direction == "Down")
sp = mean(pred.class[neg.ind]==Smarket$Direction[neg.ind])
sp

## ROC curve (x-axis: 1-sp, y-axis: sens)
# we need to compute sens and sp for each value of the threshold

tseq = seq(0.25, 0.75, length.out = 100)

sens = c(); sp = c()

for (j in 1: length(tseq)){
  t = tseq[j]
  pred.class = rep("Down", n)
  pred.class [probs>t]="Up"
  #Sensitivity and specificity
  p.ind = which(Smarket$Direction == "Up")
  sens[j] = mean(pred.class[p.ind] == Smarket$Direction[p.ind])
  neg.ind = which(Smarket$Direction == "Down")
  sp[j] = mean(pred.class[neg.ind] == Smarket$Direction[neg.ind])
  }
plot(1-sp, sens, type = "l")
abline(a = 0, b = 1)