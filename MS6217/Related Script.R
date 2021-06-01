# Class 1 script

library(quantmod)

getSymbols(c("^VIX"),from="1990-01-01",periodicity = "monthly")

VIX.rtn = diff(log(VIX$VIX.Adjusted))[-1]

plot(VIX$VIX.Adjusted)
hist(VIX.rtn,breaks=50)
abline(v=mean(VIX.rtn),col="green") #add one more straight line into current page


