library(this.path)
library(plm)


dataframe <- read.csv2(paste(this.dir(),"/simulation.csv", sep = ""))

dataframe$product1 <- dataframe$x1_obs * dataframe$x5_obs
dataframe$product2 <- dataframe$x2_obs * dataframe$x6_obs

oracle <- plm(formula = y_fixed_obs ~ product1 + product2,
              data = dataframe,
              model = "within",
              index = c("individus","temps"))

summary <- summary(oracle)
print(sqrt(mean(summary$residuals^2)))
