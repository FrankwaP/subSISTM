library(lcmm)

RSLT_DIR <- 'results'
R_RDS_BEST <- paste(RSLT_DIR , "/best_random_hlme.Rds", sep = '')
PY_CSV_PRED <- paste(RSLT_DIR, "/dataframe_predict.csv", sep = '')
R_CSV_PRED <- paste(RSLT_DIR , "/random_effect_predict.csv", sep = '')


####
random_hlme <- readRDS(R_RDS_BEST)

data <- read.csv2(PY_CSV_PRED, sep = ',', dec = '.')

# https://www.rdocumentation.org/packages/lcmm/versions/2.1.0/topics/predictY
# prediction <- predictY(random_hlme, newdata = data, marg = FALSE)
# write.table(
#   prediction$pred,
#   R_CSV_PRED,
#   sep = ",",
#   dec = ".",
#   row.names = FALSE
# )
# write.table(
#   prediction$pred,
#   R_CSV_PRED,
#   sep = ",",
#   dec = ".",
#   row.names = FALSE
# )


# NO MORE CHEATING FOR 2025
xnames = paste0("x", 1:8)
temps <- unique(data$temps)

# initialization with the (intercept) marginal effect
data$pred <- as.vector(predictY(random_hlme, newdata = data, marg = TRUE)$pred)
for (t in temps[-1:-1]) {
  # a verification has been done with "prev_data <- data"
  # we get the same results as with the original calculation (except for t=0 of course) 
  prev_data <- data[data$temps < t, ]
  ui <- predictRE(random_hlme, prev_data)

  # checking the order so we can simply multiply the x's and the ui's
  if (any(data[data$temps == t,]$individus != reffect$individus)) {
    stop("This is an error message")
  }
  # addition of the intercept random effect
  data$pred[data$temps == t] <- data$pred[data$temps == t] + reffect$intercept
  data$pred[data$temps == t] <- data$pred[data$temps == t] + rowSums(data[data$temps == t, xnames] *  ui[, xnames])
}

write.table(
  data$pred,
  R_CSV_PRED,
  sep = ",",
  dec = ".",
  row.names = FALSE
)
