library(timeSeries)
library(nlme)
library(rstudioapi)

mydata_dengue = read.csv('/home/weile/Datathon/Dengue/Wu/test_results.csv')
par(mfrow=c(2,1))
plot(mydata_dengue[,1], type = 'l', main="index plot of dengue cases")
plot(mydata_dengue[,2], type = 'l', main="line plot of prediction of dengue cases")
lines(mydata_dengue[,2], col='green')

ts_residuals = as.timeSeries(mydata_dengue[,3])
r = acf(mydata_dengue[,3])

par(mfrow=c(2,1))
plot(mydata_dengue[,2], type = 'l', main="line plot of dengue cases")
r = acf(mydata_dengue[,2], lag.max=500, main="acf plot of dengue cases")
r$acf