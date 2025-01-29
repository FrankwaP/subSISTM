library(lcmm)

TSTEP <- "temps"
SUBJECT <- "individus"
INTRCPT <- "intercept"

PY_CSV_FIT_RESID <- "fixed_effect_fit_residuals.csv"
R_RDS <-  "random_hlme.Rds"
R_CSV_FIT <- "random_effect_fit.csv"

######
data_train <- read.csv(PY_CSV_FIT_RESID)

random_hlme <- hlme(
  # SAME AS PYTHON !!!
  e_fixed ~ 1,
  random = ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8,
  ###
  # cor = AR(temps) ou BM(temps)
  idiag = TRUE,
  data = data_train,
  subject = SUBJECT,
  var.time = TSTEP
)

write.csv(
  random_hlme$pred['pred_ss'],
  R_CSV_FIT,
  row.names = FALSE
)

saveRDS(random_hlme, R_RDS)
