#visualization of the CLT
# poulation: bernoulli dist

n = 100; p=0.5
#sample of n coin tosses from this dist is
x = rbinom(n,1,p)
x
mx = mean(x)

#clt says that the sample mean will always be normally distributed, irrespective of the population

#repeat for m = 2000 times using for loop

m = 2000; mx = c()
for (j in 1:m){
  x=rbinom(n,1,p)
  mx[j]=mean(x)
}

mx[1:50]

hist(mx)
hist(mx,breaks=5)
#Doing the same on uniform distribution
m = 2000; mx = c()
for (j in 1:m){
  x=runif(n,-1,1)
  mx[j]=mean(x)
}

mx[1:50]
hist(mx)
hist(mx,breaks=20)

# Simple linear regresion
# simulating data
n = 100; b0 = 1; b1 = 1
x = rnorm(n) #default is mu = 0 and sigma = 1
eps = rnorm(n)
y = b0 + b1*x + eps

plot(x,y) ##scatterplot

#visualizing the sampling distribution of least squares estimates
est_slr=function(x,y){
  hb1 = ((t(x-mean(x))%*%(y-mean(y)))/(t(x-mean(x)))%*%(x-mean(x)))
  hb0 = mean(y)-hb1*mean(x)
  return(list(intercept = hb0, slope=hb1))
}
est_slr(x,y)

###we know hb1 is a random number: sampling distribution:
#hb1~N(b1, sigma^2/s_xx);

s_xx = t(x-mean(x))%*%(x-mean(x))
s_xx

#repeat experiment m = 5000 times
m = 5000; hb1_list=c()
for(j in 1:m){
  eps = rnorm(n)
  y = b0+b1*x+eps
  est = est_slr(x,y)
  hb1_list[j] = est$slope
}

hist(hb1_list , breaks = 25)
#each j is estimating on a training sample of size n

#theoretical resul says the variance should be
vhb1 = 1/s_xx ##Theoretical method of knowing variance of Beta1

hist(hb1_list, breaks = 25)
var(hb1_list) ## Variance calculated by simulation
vhb1


##usefulness

#what proportion of times do we observe a hb1>1.2 or hb1<0.8

(sum(hb1_list>1.2 | hb1_list<0.8))/m

## so the chances of observing a sample with a slope estimate < 0.8 or >1.2
# from a population with the true unknown b1=1, is ~0.0784

## in practice we only have 1 training data set so we do not
# have access to this histogram. Hence, we need to compute this probability
# using the sampling distribution
#pval = p(z>|z|) = p(Z>z)+p(Z<-z), where z is N(0,1)
pval = pnorm(0.8, 1, sqrt(vhb1))+ (1-pnorm(1.2, 1, sqrt(vhb1))) ## pnorm is cdf of normal dist
pval                                  

#the probability of observing a sample where the estimate is 0.8 or lower or 1.2 or higher
#from a population with true unknown coefficient being 1 is about 7.8 %

## Everything we did before this line was the illustration of the idea behind sampling distribution
# and its connection with hypothesis testing
# In application, we dont need to do any of this

# data x: predictor, y: response
# to compute p-values, we can use the function lm()

model = lm(y ~ x)
summary(model)

install.packages("ISLR")
library(ISLR)
data("Carseats")

x = Carseats$Advertising
y = Carseats$Sales

model = lm(y ~ x)
summary(model)

model = lm(Sales~Advertising, data = Carseats) #Another way of writing the same.
summary(model)
