## Introduction to R
x = c(1,2,3,4) ## define a vector
x #print x
y = c(2,3,4,5)

##To get rid of y
#rm(y)
#rm(list=ls()) ##removes everything

z = x+y ##adding two vectors

## vector multiplication
t(x)%*%y #product to transpose of x times y
x*y ## multiplies vectors componentwise
2*x

##some basic operations
mu=mean(x)
v1=var(x)
s1=sd(x); s1 = sd(x)
v1;s1
log(x)
max(x)
length(x) ##size of x

### to access help files
?var ##to get help of a function

##defining matrices
m1 = matrix(c(1,2,3,4), nrow=2, ncol=2); m1 ##fills entries columnwise
fix(m1)
## making a matrix by joining vectors
x1= rep(0,5) ##repeats the first argument
x2= rep(1,5)

m2 = rbind(x1,x2) ##bind rowwise
m2
m3 = cbind(x1,x2) ##bind columnwise

##matrix multiplication
m2*m3
2*m2

#generating random variables
#bernoulii dist
n=100; p=0.5
x=rbinom(n,1,p)
x
##normal distribution
mu=2; sigma = 1; n = 100 ##mu is the mean and sigma is the standard deviation
x=rnorm(n,mu,sigma)
x

## for reproducibility use a seed number
set.seed(123)
n=100; p=0.5
x=rbinom(n,1,p)
x

#using libraries and importing/exporting data
install.packages("ISLR")
library(ISLR)
data(Auto) ##Extract a dataset called "Auto"
Auto
?Auto
dim(Auto) ##dimensions of the data
head(Auto) ##to look at the initial part of this data

#Graphics

#histogram
hist(Auto$mpg)
?hist

#Scatterplot (xy plot)

plot(Auto$displacement, Auto$mpg)

#3d plots

##build a custom function
x=seq(-3,3,by = 0.1) #sequence from -3 to 3 with 0.1 increments
y=x

h=function(x,y){
  z=cos(y)/(1+x^2)
  return (z)
}

h(1,2)
h(2,3)

#perspective plot
g = outer(x,y,h)
persp(x,y,g,theta=0,phi = 60)

#contour plot (2 dim representation of a 3 d plot)
contour(x,y,g)

###importing and exporting data
#exporting
#set working directory
setwd("~/Downloads/Spring 2019/Stat 530")
write.csv(Auto,"exported_data.csv")

##importing data


###Rmarkdown
library(rmarkdown)
\section{Introduction}
This is an example. To write a math equation use ordinary latex codetools
\begin{eqnarray}
(a+b)^2 = a^2+b^2+2ab
\end{eqnarray}
