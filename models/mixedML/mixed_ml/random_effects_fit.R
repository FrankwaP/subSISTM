library(lcmm)

RSLT_DIR <- 'results'
PY_CSV_FIT_RESID <- paste(RSLT_DIR , "/fixed_effect_fit_residuals.csv", sep = '')
R_CSV_FIT <- paste(RSLT_DIR , "/random_effect_fit.csv", sep = '')
R_RDS <- paste(RSLT_DIR , "/random_hlme.Rds", sep = '')

######
data <- read.csv2(PY_CSV_FIT_RESID, sep = ',', dec = '.')

random_hlme <- hlme(
  # SAME AS PYTHON !!!
  e_fixed ~ 1,  # + x2_x5 + x4_x7 + x6_x8,
  random = ~  1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8,
  ###
  # cor = AR(temps) ou BM(temps)
  idiag = TRUE,
  data = data,
  subject = 'individus',
  var.time = 'temps'
)

write.table(
  random_hlme$pred,
  R_CSV_FIT,
  sep = ",",
  dec = ".",
  row.names = FALSE
)

saveRDS(random_hlme, R_RDS)
