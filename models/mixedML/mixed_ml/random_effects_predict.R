library(lcmm)

RSLT_DIR <- 'results'
R_RDS_BEST <- paste(RSLT_DIR , "/best_random_hlme.Rds", sep = '')
PY_CSV_PRED <- paste(RSLT_DIR, "/dataframe_predict.csv", sep = '')
R_CSV_PRED <- paste(RSLT_DIR , "/random_effect_predict.csv", sep = '')


####
random_hlme <- readRDS(R_RDS_BEST)

data <- read.csv2(PY_CSV_PRED, sep = ',', dec = '.')

# https://www.rdocumentation.org/packages/lcmm/versions/2.1.0/topics/predictY
prediction <- predictY(random_hlme, newdata = data, marg = FALSE)

write.table(
  prediction$pred,
  R_CSV_PRED,
  sep = ",",
  dec = ".",
  row.names = FALSE
)
