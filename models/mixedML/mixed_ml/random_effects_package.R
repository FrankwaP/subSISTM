suppressWarnings(suppressMessages(library(this.path)))
library(ini)
library(lcmm)

TSTEP <- "temps"
SUBJECT <- "individus"

config <- read.ini(paste0(dirname(this.path()), "/interoperability_config.ini"))

fit <- function(random, subject, time, idiag = FALSE, cor = NULL, maxiter = 50) {
  fixed_effects_csv <- config$TRAINING$fixed_effects_csv
  random_effects_results <- config$TRAINING$random_effects_csv
  mixed_model_rds <- config$TRAINING$mixed_model_rds
  data <- read.csv(fixed_effects_csv)

  B_init <- tryCatch(
    # a previous iteration exist
    {
      previous_hlme <- readRDS(mixed_model_rds)
      print("Loading previously trained model.")
      B_init <- previous_hlme$best
      B_init
    },
    # first iteration: we call HLME to get the default B_init
    # "warning" works better than "error"
    warning = function(e) {
      print("First iteration with a new model.")
      random_hlme <- hlme(
        e_fixed ~ 1,
        random = random,
        idiag = idiag,
        data = data,
        subject = subject,
        maxiter = 0
      )
      B_init <- c(0, random_hlme$best[-1])
      return(B_init)
    }
  )
  print("B_init:")
  print(round(B_init,2))

  command <- substitute(hlme(
    e_fixed ~ 1,
    random = random,
    # cor = cor, # AR(temps) ou BM(temps)
    idiag = idiag,
    data = data,
    subject = subject,
    var.time = time,
    B = B_init,
    posfix = c(1),
    maxiter = maxiter
  ))
  random_hlme <- eval(command)

  write.csv(
    random_hlme$pred["pred_ss"],
    random_effects_results,
    row.names = FALSE
  )
  saveRDS(random_hlme, mixed_model_rds)
}

forecast <- function(subject, time) {
  fixed_effects_csv <- config$PREDICTION$fixed_effects_csv
  random_effects_results <- config$PREDICTION$random_effects_csv
  mixed_model_rds <- config$PREDICTION$mixed_model_rds

  data <- read.csv(fixed_effects_csv)
  tsteps <- unique(data[, time])
  model <- readRDS(mixed_model_rds)
  x_labels <- model$Xnames
  INTRCPT <- model$Xnames[1]

  data[, INTRCPT] <- 1 # trick to simplify the RE 'rowSums' calculation
  # computing the marginal effects
  # !!!!!!!!!!!! Erreur dans x$call$random[2] : objet de type 'symbol' non indiçable
  pred_marg <- as.vector(predictY(model, newdata = data, marg = TRUE)$pred)
  # initializing the prediction with the marginal effect
  # R does not work like Python: the memory will be change when the vector is modified
  pred_me <- pred_marg
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
  for (t in tsteps[-1]) {
    # a verification has been done with "prev_data <- data"
    # to use all the data as with predictY
    # and… we get the same results as with predictY
    # (except for t=0 of course)
    prev_data <- data[data[time] < t, ]
    ui <- predictRE(model, prev_data)
    #########
    if (nrow(data[data[time] == t, ]) != nrow(ui)) {
      stop("youhou")
    }
    #############
    # checking the order so we can simply multiply the x's and the ui's
    if (any(data[data[time] == t, ][subject] != ui[subject])) {
      stop("This is an error message!")
    }
    # addition of the random effects
    reffects <- rowSums(data[data[time] == t, x_labels] * ui[, x_labels])
    pred_me[data[time] == t] <- pred_me[data[time] == t] + reffects
  }
  data[, "pred_mixed"] <- pred_me
  data[, "pred_fixed"] <- pred_marg
  write.csv(data, random_effects_results, row.names = FALSE)
}



#################### TEST
fit(
  random = ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8,
  subject = SUBJECT, time = TSTEP, idiag = TRUE,
)
forecast(subject = SUBJECT, time = TSTEP)
