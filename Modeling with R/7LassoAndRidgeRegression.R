## Lasso and RIdge regression

### Simulate data
n=200; p = 500
x = matrix(rnorm(n*p), n, p)
b = c(rep(1,5), rep(0, p-5))
b
y = 1+x%*%b+rnorm(n)
length(y)
dim(x)
x

## Here y is the response and x is the design matrix

install.packages("glmnet")
library(glmnet)
model.lasso = glmnet(x,y)
#by default the function glmnet will fit a lasso estimate
hb = coef(model.lasso, s=0.2)
#the tuning parameter lambda is represented by 's' in the above function
hb

# Implementing
model.ridge = glmnet(x,y,alpha = 0) # alpha=0 indicates the ridge penalty
hb = coef(model.ridge)
hb

## choosing lambda via cross validation
# begin with la = 0.01 and compute the corresponding cv.err

dat = data.frame(x=x, y = y)
la.grid = seq(0.01,1,length.out = 100)
la.grid

k = 5 #no. of fols
n = length(dat$y)
ncv = ceiling(n/k)
cv.ind.f = rep(seq(1:k), ncv)
cv.ind = cv.ind.f[1:n]
cv.ind.random = sample(cv.ind, n, replace = F)

cv.err=c() # we created this cv error later
for(l in 1:length(la.grid)){ # we added this later
  MSE = c()
  for(j in 1:k){
    train = dat[cv.ind.random!=j,]
    response = train$y
    design = as.matrix(train[,names(dat)!="y"])
    mod = glmnet(design, response)
    hb = coef(mod, s = la.grid[l])
    test = dat[cv.ind.random == j,]
    resp.test = test$y
    fitted.values = cbind(1, as.matrix(test[,names(dat)!="y"]))%*%hb
    MSE[j] = mean((resp.test-fitted.values)^2)
  }
  cv.err[l]= mean(MSE)
}

plot(la.grid, cv.err, type = "l")
#

best.la = la.grid[which.min(cv.err)]

best.la

model = glmnet(x,y)
final.hb = coef(model, s = best.la)
final.hb


#### SHort cut to cross validation
## For lasso itself, we can use cv.glmnet too 
cv.mod = cv.glmnet(x,y,nfolds = 5, lambda = seq(0.01, 1, length.out = 100))
cv.mod$lambda.min

## implementing lasso on hitters dataset
library(ISLR)
data("Hitters")

x = Hitters[, 1:13]
x
y = Hitters$Salary


dat = data.frame(x=x, y = y)
dat.complete = dat[complete.cases(dat),]
dat.complete

x.new = as.matrix(dat.complete[,names(dat.complete)!="y"])
y = dat.complete$y
y
length(x.new)
length(y)
dim (x.new)
model = glmnet(x.new,y)
hb = coef(model, s=10) # if we change s, our coefficients change
hb


# Logistic regression

data("Smarket")
?Smarket
logit.model = glm(Direction~Lag1+Lag2+Lag3+Lag4+Volume, data= Smarket, family = binomial)
summary(logit.model)

glm.probs = predict(logit.model, type = "response")
glm.probs[1:10]

## Confusion matrix
glm.pred = rep("Down", 1250)
glm.pred[glm.probs>0.5]="Up"

glm.pred

# Confusion matrix
table(glm.pred, Smarket$Direction)

