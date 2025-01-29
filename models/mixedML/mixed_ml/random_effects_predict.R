library(lcmm)

TSTEP <- "temps"
SUBJECT <- "individus"
INTRCPT <- "intercept"

PY_CSV_PRED <- "dataframe_predict.csv"
R_RDS_BEST <- "best_random_hlme.Rds"
R_CSV_PRED <- "random_effect_predict.csv"

forecast <- function(model, data) {
  temps <- unique(data[, TSTEP])
  x_labels <- model$Xnames
  data[, INTRCPT] <- 1
  # initialization with the (intercept) marginal effect
  pred <- as.vector(predictY(model, newdata = data, marg = TRUE)$pred)
  #############
  # Checked that it does not change anything
  # (SUBJECT from test not matched with SUBJECT from train)
  # youpi1 <- predictY(model, newdata = data, marg = FALSE)$pred
  # data[SUBJECT] <- data[SUBJECT] * 1000
  # youpi2 <- predictY(model, newdata = data, marg = FALSE)$pred
  # if (any(youpi1['pred_ss1'] != youpi2['pred_ss1'])) {
  #   stop("youhou!")
  # }
  ###########
  for (t in temps[-1]) {
    # a verification has been done with "prev_data <- data"
    # to use all the data as with predictY
    # and… we get the same results as with predictY
    # (except for t=0 of course)
    prev_data <- data[data[TSTEP] < t, ]
    ui <- predictRE(model, prev_data)
    #########
    if (nrow(data[data[TSTEP] == t, ]) != nrow(ui)) {
      stop("youhou")
    }
    #############
    # checking the order so we can simply multiply the x's and the ui's
    if (any(data[data[TSTEP] == t, ][SUBJECT] != ui[SUBJECT])) {
      stop("This is an error message!")
    }
    # addition of the random effects
    reffects <- rowSums(data[data[TSTEP] == t, x_labels] * ui[, x_labels])
    pred[data[TSTEP] == t] <- pred[data[TSTEP] == t] + reffects
  }
  return(pred)
}

####
random_hlme <- readRDS(R_RDS_BEST)
data_test <- read.csv(PY_CSV_PRED)
preds <- forecast(random_hlme, data_test)
write.csv(preds, R_CSV_PRED, row.names = FALSE)
