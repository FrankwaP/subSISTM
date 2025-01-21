## Import des bilbiothèques

# library(devtools)
# load_all("/home/francois/Téléchargements/lcmm-master/R/")
library(lcmm)

# library(doParallel)
library(doSNOW)
library(foreach)

set.seed(0)


########
# fake value for linter it is actually imported by the source command
X_LABELS <- c("")
Y_LABEL <- ""
source(file = "study_config.R")
#######
DATA_FOLDER <- "../../../data/synthetic_bph_1/"
PATTERN_SIMU <- "simulation\\d+\\.csv"
CSV_NAME <- "predictions.csv"
SUBJECT <- "individus"
TSTEP <- "temps"
MOD <- "model"

# same as in the Python script
SIM <- "simulation"
DSET <- "dataset"
TRAIN <- "train"
TEST <- "test"


load_csv <- function(path) {
  data <- read.csv2(file = path)
  data["x2_x5"] <- data["x2"] * data["x5"]
  data["x4_x7"] <- data["x4"] * data["x7"]
  data["x6_x8"] <- data["x6"] * data["x8"]
  data["temps__2"] <- data["temps"]**2
  data["temps__3"] <- data["temps"]**3
  data["temps__4"] <- data["temps"]**4
  return(data)
}

train_hlme <- function(data_train) {
  #   oracle_mixed <- hlme(y_mixed_obs ~ x2_x5 + x4_x7 + x6_x8,
  #     random = ~ x2_x5 + x4_x7 + x6_x8,
  #     idiag = TRUE,
  #     data = Dtrain, subject = "individus"
  #   )
  str_re <- paste0("~", paste0(X_LABELS, collapse = "+"))
  str_fe <- paste0(Y_LABEL, str_re)
  form_re <- as.formula(str_re)
  form_fe <- as.formula(str_fe)
  # trick to avoid problem with `match.call` in lcmm predictY.lcmm.R
  command <- substitute(
    hlme(
      fixed = form_fe,
      random = form_re,
      idiag = TRUE,
      data = data_train,
      subject = SUBJECT
    ),
    env = c(as.list(globalenv()), as.list(environment()))
  )
  model <- eval(command)
  return(model)
}

forecast <- function(model, data) {
  temps <- unique(data[, TSTEP])
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
    reffects <- ui$intercept + rowSums(data[data[TSTEP] == t, X_LABELS] * ui[, X_LABELS])
    pred[data[TSTEP] == t] <- pred[data[TSTEP] == t] + reffects
  }
  return(pred)
}

get_predict_dataframe <- function(model, data, dset_name) {
  data["dataset"] <- dset_name
  data["pred"] <- forecast(model, data)
  return(data[c(SUBJECT, TSTEP, "dataset", "pred")])
}

get_predict_simulation <- function(simu_name, data_train, data_test) {
  model <- train_hlme(data_train)
  pred_train <- get_predict_dataframe(model, data_train, "train")
  pred_test <- get_predict_dataframe(model, data_test, "test")
  pred <- rbind(pred_train, pred_test)
  pred["simulation"] <- simu_name
  return(pred)
}

get_predict_model <- function() {
  data_test <- load_csv(paste0(DATA_FOLDER, "01_test.csv"))
  simu_files <- list.files(path = DATA_FOLDER, pattern = PATTERN_SIMU, full.names = TRUE)

  #########
  # MULTIPROC
  # cl <- makeCluster(10, outfile = "cluster.txt"))
  # # registerDoParallel(cl)
  # registerDoSNOW(cl)
  # pb <- txtProgressBar(max = length(simu_files), style = 3)
  # progress <- function(n) setTxtProgressBar(pb, n)
  # clusterExport(cl, c(
  #   "load_csv", "get_predict_simulation", "train_hlme",
  #   "X_LABELS", "Y_LABEL", "hlme", "SUBJECT", "get_predict_dataframe",
  # ))
  # # pred <- foreach(simu_file = simu_files, .combine = rbind) %dopar% {
  # pred <- foreach(
  #   simu_file = simu_files, .combine = rbind,
  #   .options.snow = list(progress = progress)
  # ) %dopar% {
  #   simu_name <- basename(simu_file)
  #   data_train <- load_csv(simu_file)
  #   return(get_predict_simulation(simu_name, data_train, data_test))
  # }
  ###############
  # SINGLE PROC
  pred <- data.frame()
  for (simu_file in simu_files) {
    simu_name <- basename(simu_file)
    print(simu_name)
    data_train <- load_csv(simu_file)
    ########################################################################
    # data_train <- data_train[data_train[SUBJECT] < 5, ]
    ########################################################################
    pred_simu <- get_predict_simulation(simu_name, data_train, data_test)
    pred <- rbind(pred, pred_simu)
  }
  #############
  write.csv(pred, CSV_NAME, row.names = FALSE)
  return(pred)
}


# get_predict_model("Oracle", c("x2_x5", "x4_x7", "x6_x8"))
# get_predict_model("Linear", c("x1", "x2" ,"x3" ,"x4" ,"x5","x6", "x7", "x8"))
# get_predict_model("time-polynom", c("temps", "temps__2", "temps__3", "temps__4"))
