##Cross validation to select the best model, Forward and Backward selection

## Simulate data
set.seed(123)
n = 200
x = rnorm(n)
y = -1- 0.5*x + 1*x^2 + rnorm(n)
plot(x,y)

## Since the relationship is clearly nonlinear, we try polynomial terms X^2, x^3, x^4...

# M1 : y~x +x^2,....,M7:y~x + x^2...x^7

# We shall use cross validation to choose the best model from M1...M7
dat = data.frame(y = y, x = x)

k=5 # no. of folds. Though we are creating a user defined function below where we will take k as input.

cv.function = function(dat, k){

n = length(dat$y)
if (k>n) {stop("check no. of folds")}
ncv = ceiling(n/k);
#ncv we commented out ncv
# Ceiling function is the smallest integer > x
# e.g. 4.2 ---> 5

cv.ind.f = rep(seq(1:k), ncv) # purpose of this is to assign fold to each observation
#cv.ind.f we commented this out as we wanted to make a function that would retrun cv everytime

cv.ind = cv.ind.f[1:n] # we want our fold to be random
#length(cv.ind)

cv.ind.random = sample(cv.ind, n, replace = F)
#cv.ind.random

train = dat[cv.ind.random!=1, ] # trying to extract the subset of the data
# that forms the training set that has all the fold but first one.
MSE = c() # We added this later to store our mse for each training and testing we do.
for (j in 1:k){
train = dat[cv.ind.random!=j,]
response = train$y
design = as.matrix (train [, names(dat)!="y"])
mod = lm(response~design)
hb = coef(mod)

test = dat[cv.ind.random ==1,] # we had both response and predictor in our testing data
resp.test = test$y

#names(test) ##test has both x and y, so we will remove y and use coeff of training set

test[,names(dat)!="y"]

fitted.values = cbind(1,as.matrix(test [,names(dat)!="y"]))%%hb
MSE [j] = mean((resp.test-fitted.values)^2)}

# We earlier did only for first fold, then we introduced the loop in top, repeating it for every other fold.

cv.err = mean(MSE) # Now we will store MSE in cv.err
return (cv.err)}

cv.function (dat, k = 20)

cvm =c()
dat1 = data.frame(y = y, x1 =x)
cvm[1]=cv.function(dat1, k = 5)

dat2 = data.frame(y=y, x1 = x, x2 = x^2)
cvm[2] = cv.function(dat2, k = 5)

dat3 = data.frame(y=y, x1 = x, x2 = x^2, x3 = x^3)
cvm[3] = cv.function(dat2, k = 5)

dat4 = data.frame(y=y, x1 = x, x2 = x^2, x3 = x^3, x4 = x^4)
cvm[4] = cv.function(dat2, k = 5)

dat5 = data.frame(y=y, x1 = x, x2 = x^2, x3 = x^3, x4 = x^4, x5 = x^5)
cvm[5] = cv.function(dat2, k = 5)

dat6 = data.frame(y=y, x1 = x, x2 = x^2, x3 = x^3, x4 = x^4, x5 = x^5, x6 = x^6)
cvm[6] = cv.function(dat2, k = 5)

cvm

# Now above we did different cross validation to know the best model.

## Forward and backward selection
dat6 # contains y and x.. x^6, and we want to choose the best model using forward selections

install.packages("leaps")
library(leaps)
regfit = regsubsets(y~.,data = dat6, method = "forward", nvmax = 10)
s = summary (regfit)
s$adjr2

# in backward selection, we just change it to backward.
