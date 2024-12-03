## Import des bilbiothèques

library(rockchalk)
library(dplyr)
library(lcmm)
library(doParallel)
library(foreach)
library(plyr)
library(reshape)
library(ggplot2)
set.seed(0)
## Définition des variables

cl <- makeCluster(10)
registerDoParallel(cl)

ind <- 1:500
time <- 0:25
l <- list(ind, time)
dataframe <- rev(expand.grid(rev(l)))
colnames(dataframe) <- c("individus", "temps")
# Epsilon
sigma_epsilon <- c(0.5, 0.1, 0.1, 0.1, 0.5, 1, 0.5, 1)

# X
mu0 <- runif(7, -10, 10)
sig0 <- diag(c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
mu1 <- runif(7, -1, 1) 
sig1 <- diag(c(0.5, 0.5, 0.1, 0.5, 1, 0.1, 0.5))
truthX <- data.frame("µ0"=mu0, "µ1" = mu1, "sigma0"=diag(sig0), "sigma1"=diag(sig1))
rownames(truthX) <- c('X1','X2','X3','X4','X5','X6','X7')

# Y
µ_gamma <- runif(4, -1, 1)
sdgamma <- diag(c(0.5, 0.5, 0.05, 0.1))
truthY <- data.frame('µ' = µ_gamma, 'sigma²' = sdgamma, 'var_eps' = sigma_epsilon[8])

## Fonction simul()


simul <- function(
    individus = ind, df = dataframe, µ0 = mu0, µ1 = mu1, sigma0 = sig0,
    sigma1 = sig1, µg = µ_gamma, sigmag = sdgamma, sigeps = sigma_epsilon) {
  
  alpha0 <- mvrnorm(length(individus), µ0, sigma0)
  alpha1 <- mvrnorm(length(individus), µ1, sigma1)
  g_mixed <- mvrnorm(length(ind), µg, sigmag)
  g_fixed <- mvrnorm(length(ind), µg, diag(numeric(4)))
  
  df$x1 <- alpha0[df$individus, 1] + alpha1[df$individus, 1] * df$temps
  df$x1_obs <- df$x1 + rnorm(length(df$x1), 0, sigeps[1])
  df$x2 <- alpha0[df$individus, 2] + alpha1[df$individus, 2] * log(df$temps + 1)
  df$x2_obs <- df$x2 + rnorm(length(df$x2), 0, sigeps[2])
  df$x3 <- alpha0[df$individus, 3] + 0.001 * alpha1[df$individus, 3] * (df$temps^2)
  df$x3_obs <- df$x3 + rnorm(length(df$x3), 0, sigeps[3])
  df$x4 <- alpha0[df$individus, 4] + alpha1[df$individus, 4] * exp(-0.1 * df$temps)
  df$x4_obs <- df$x4 + rnorm(length(df$x4), 0, sigeps[4])
  df$x5 <- alpha0[df$individus, 5] / (1 + exp(-alpha1[df$individus, 5] * df$temps))
  df$x5_obs <- df$x5 + rnorm(length(df$x5), 0, sigeps[5])
  df$x6 <- pmax(0, (alpha0[df$individus, 6] + 0.1 * alpha1[df$individus, 6] * df$temps)^2)
  df$x6_obs <- df$x6 + rnorm(length(df$x6), 0, sigeps[6])
  df$x7 <- alpha0[df$individus, 7] / (1 + exp(alpha1[df$individus, 7] * df$temps))
  df$x7_obs <- df$x7 + rnorm(length(df$x7), 0, sigeps[7])
  df <- df %>%
    mutate(x8 = ifelse(df$individus %% 2 == 0, 1, 0))
  
  df$x2_x5 <- df$x2 * df$x5
  df$x4_x7 <- df$x4 * df$x7
  df$x6_x8 <- df$x6 * df$x8
  
  df$y_mixed <- g_mixed[df$individus, 1] +
    g_mixed[df$individus, 2] * df$x2 * df$x5 +
    g_mixed[df$individus, 3] * df$x4 * df$x7 +
    g_mixed[df$individus, 4] * df$x6 * df$x8
  
  df$y_mixed_obs <- df$y_mixed + rnorm(length(df$y_mixed), 0, sigeps[8])
  
  
  df$y_fixed <- g_fixed[df$individus, 1] + 
    g_fixed[df$individus, 2] * df$x2 * df$x5 + 
    g_fixed[df$individus, 3] * df$x4 * df$x7 +
    g_fixed[df$individus, 4] * df$x6 * df$x8
  
  
  df$y_fixed_obs <- df$y_fixed + rnorm(length(df$y_fixed), 0, sigeps[8])
  
  return(df)
}


## Génération des données de test


Dtest <- simul()

boucle <- foreach(i=1:100, 
                  .combine=cbind, 
                  .packages=c("rockchalk", "dplyr", "lcmm", "doParallel", "foreach")) %dopar%
{
  Dtrain <- simul()
  k = as.character(i)
  #Modèle oracle sur les Y à effet mixed
  oracle_mixed <- hlme(y_mixed_obs ~ x2_x5 + x4_x7 + x6_x8,
                       random=~ x2_x5 + x4_x7 + x6_x8,
                       idiag = TRUE,
                       data= Dtrain, subject='individus')
  save(oracle_mixed, file = paste("oracle_mix", k ,".rda", sep = ""))
  
  #beta_fix <- oracle_mixed$best[1:4]
  #sigma_k <- oracle_mixed$best[c('varcov 1','varcov 3','varcov 6')]
  #eps_k <- oracle_mixed$best['stderr']
  
  #biais_beta <- beta_fix - µ_gamma
  #biais_sigma <- sigma_k - c(0.5, 0.5, 0.05)
  #res_mixed <- c(beta_fix, biais_beta, sigma_k, biais_sigma, eps_k)
  
  pred_train_mixed <- predictY(oracle_mixed, newdata = Dtrain, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_train_mixed_truth <- mean(abs(pred_train_mixed$pred[,'pred_ss'] - Dtrain$y_mixed))
  mse_train_mixed_truth <- mean((pred_train_mixed$pred[,'pred_ss'] - Dtrain$y_mixed)^2)
  mae_train_mixed_obs <- mean(abs(pred_train_mixed$pred[,'pred_ss'] - Dtrain$y_mixed_obs))
  mse_train_mixed_obs <- mean((pred_train_mixed$pred[,'pred_ss'] - Dtrain$y_mixed_obs)^2)
  
  pred_test_mixed <-predictY(oracle_mixed, newdata = Dtest, var.time = 'temps', marg = FALSE, subject = 'individus' )
  mse_test_mixed_truth <- mean((pred_test_mixed$pred[,'pred_ss'] - Dtest$y_mixed)^2)
  mae_test_mixed_truth <- mean(abs(pred_test_mixed$pred[,'pred_ss'] - Dtest$y_mixed))
  mse_test_mixed_obs <- mean((pred_test_mixed$pred[,'pred_ss'] - Dtest$y_mixed_obs)^2)
  mae_test_mixed_obs <- mean(abs(pred_test_mixed$pred[,'pred_ss'] - Dtest$y_mixed_obs))
  
  #Regression linéaire mixte avec toutes les variables pertinentes pour y_mixed
  naif_mixed <- hlme(y_mixed_obs ~ temps + temps**2 + temps**3 + temps**4,
                     idiag = TRUE,
                     random=~ temps + temps**2 + temps**3 + temps**4,
                     data = Dtrain, subject='individus')
  
  pred_train_naif_mixed <- predictY(naif_mixed, newdata = Dtrain, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_train_naif_mixed_truth <- mean(abs(pred_train_naif_mixed$pred[,'pred_ss'] - Dtrain$y_mixed))
  mse_train_naif_mixed_truth <- mean((pred_train_naif_mixed$pred[,'pred_ss'] - Dtrain$y_mixed)^2)
  mae_train_naif_mixed_obs <- mean(abs(pred_train_naif_mixed$pred[,'pred_ss'] - Dtrain$y_mixed_obs))
  mse_train_naif_mixed_obs <- mean((pred_train_naif_mixed$pred[,'pred_ss'] - Dtrain$y_mixed_obs)^2)
  
  pred_test_naif_mixed <- predictY(naif_mixed, newdata = Dtest, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_test_naif_mixed_truth <- mean(abs(pred_test_naif_mixed$pred[,'pred_ss'] - Dtest$y_mixed))
  mse_test_naif_mixed_truth <- mean((pred_test_naif_mixed$pred[,'pred_ss'] - Dtest$y_mixed)^2)
  mae_test_naif_mixed_obs <- mean(abs(pred_test_naif_mixed$pred[,'pred_ss'] - Dtest$y_mixed_obs))
  mse_test_naif_mixed_obs <- mean((pred_test_naif_mixed$pred[,'pred_ss'] - Dtest$y_mixed_obs)^2)
  
  #regression linéaire mixe sur chaque variable sans interraction
  lin_mixed <- hlme(y_mixed_obs ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8,
                    idiag = TRUE,
                    random=~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8,
                    data = Dtrain, subject='individus')
  
  pred_train_lin_mixed <- predictY(lin_mixed, newdata = Dtrain, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_train_lin_mixed_truth <- mean(abs(pred_train_lin_mixed$pred[,'pred_ss'] - Dtrain$y_mixed))
  mse_train_lin_mixed_truth <- mean((pred_train_lin_mixed$pred[,'pred_ss'] - Dtrain$y_mixed)^2)
  mae_train_lin_mixed_obs <- mean(abs(pred_train_lin_mixed$pred[,'pred_ss'] - Dtrain$y_mixed_obs))
  mse_train_lin_mixed_obs <- mean((pred_train_lin_mixed$pred[,'pred_ss'] - Dtrain$y_mixed_obs)^2)
  
  pred_test_lin_mixed <- predictY(lin_mixed, newdata = Dtest, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_test_lin_mixed_truth <- mean(abs(pred_test_lin_mixed$pred[,'pred_ss'] - Dtest$y_mixed))
  mse_test_lin_mixed_truth <- mean((pred_test_lin_mixed$pred[,'pred_ss'] - Dtest$y_mixed)^2)
  mae_test_lin_mixed_obs <- mean(abs(pred_test_lin_mixed$pred[,'pred_ss'] - Dtest$y_mixed_obs))
  mse_test_lin_mixed_obs <- mean((pred_test_lin_mixed$pred[,'pred_ss'] - Dtest$y_mixed_obs)^2)
  
  #Modèle oracle sur les Y à effet fixes
  oracle_fixed <- lm(y_fixed_obs ~ x2_x5 + x4_x7 + x6_x8, data=Dtrain)
  sum_fix <- summary(oracle_fixed)
  save(oracle_fixed, file = paste("oracle_fix", k ,".rda", sep = ""))
  
  #beta_fix <- sum_fix$coefficients[,'Estimate']
  #biais_beta <- beta_fix - µ_gamma
  #res_fixed <- c(beta_fix, biais_beta)
  
  pred_train_fixed <- predict(oracle_fixed, newdata = Dtrain)
  mae_train_fixed_truth <- mean(abs(pred_train_fixed - Dtrain$y_fixed))
  mse_train_fixed_truth <- mean((pred_train_fixed - Dtrain$y_fixed)^2)
  mae_train_fixed_obs <- mean(abs(pred_train_fixed - Dtrain$y_fixed_obs))
  mse_train_fixed_obs <- mean((pred_train_fixed - Dtrain$y_fixed_obs)^2)
  
  pred_test_fixed <-predict(oracle_fixed, newdata = Dtest)
  mse_test_fixed_truth <- mean((pred_test_fixed - Dtest$y_fixed)^2)
  mae_test_fixed_truth <- mean(abs(pred_test_fixed - Dtest$y_fixed))
  mse_test_fixed_obs <- mean((pred_test_fixed - Dtest$y_fixed_obs)^2)
  mae_test_fixed_obs <- mean(abs(pred_test_fixed - Dtest$y_fixed_obs))
  
  #Regression Linéaire avec toutes les variables pour y_fixed
  naif_fixed <- hlme(y_fixed_obs ~ temps + temps**2 + temps**3 + temps**4,
                        idiag = TRUE,
                        random=~ temps + temps**2 + temps**3 + temps**4,
                        data = Dtrain, subject='individus')
  
  pred_train_naif_fixed <- predictY(naif_fixed, newdata = Dtrain, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_train_naif_fixed_truth <- mean(abs(pred_train_naif_fixed$pred[,'pred_ss'] - Dtrain$y_fixed))
  mse_train_naif_fixed_truth <- mean((pred_train_naif_fixed$pred[,'pred_ss'] - Dtrain$y_fixed)^2)
  mae_train_naif_fixed_obs <- mean(abs(pred_train_naif_fixed$pred[,'pred_ss'] - Dtrain$y_fixed_obs))
  mse_train_naif_fixed_obs <- mean((pred_train_naif_fixed$pred[,'pred_ss'] - Dtrain$y_fixed_obs)^2)
  
  pred_test_naif_fixed <- predictY(naif_fixed, newdata = Dtest, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_test_naif_fixed_truth <- mean(abs(pred_test_naif_fixed$pred[,'pred_ss'] - Dtest$y_fixed))
  mse_test_naif_fixed_truth <- mean((pred_test_naif_fixed$pred[,'pred_ss'] - Dtest$y_fixed)^2)
  mae_test_naif_fixed_obs <- mean(abs(pred_test_naif_fixed$pred[,'pred_ss'] - Dtest$y_fixed_obs))
  mse_test_naif_fixed_obs <- mean((pred_test_naif_fixed$pred[,'pred_ss'] - Dtest$y_fixed_obs)^2)
  
  lin_fixed <- hlme(y_fixed_obs ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8,
                    idiag = TRUE,
                    random=~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8,
                    data = Dtrain, subject='individus')
  
  pred_train_lin_fixed <- predictY(lin_fixed, newdata = Dtrain, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_train_lin_fixed_truth <- mean(abs(pred_train_lin_fixed$pred[,'pred_ss'] - Dtrain$y_fixed))
  mse_train_lin_fixed_truth <- mean((pred_train_lin_fixed$pred[,'pred_ss'] - Dtrain$y_fixed)^2)
  mae_train_lin_fixed_obs <- mean(abs(pred_train_lin_fixed$pred[,'pred_ss'] - Dtrain$y_fixed_obs))
  mse_train_lin_fixed_obs <- mean((pred_train_lin_fixed$pred[,'pred_ss'] - Dtrain$y_fixed_obs)^2)
  
  pred_test_lin_fixed <- predictY(lin_fixed, newdata = Dtest, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_test_lin_fixed_truth <- mean(abs(pred_test_lin_fixed$pred[,'pred_ss'] - Dtest$y_fixed))
  mse_test_lin_fixed_truth <- mean((pred_test_lin_fixed$pred[,'pred_ss'] - Dtest$y_fixed)^2)
  mae_test_lin_fixed_obs <- mean(abs(pred_test_lin_fixed$pred[,'pred_ss'] - Dtest$y_fixed_obs))
  mse_test_lin_fixed_obs <- mean((pred_test_lin_fixed$pred[,'pred_ss'] - Dtest$y_fixed_obs)^2)
  
  #Aggregating results
  res <- c(mae_train_mixed_truth, mse_train_mixed_truth, mae_test_mixed_truth, mse_test_mixed_truth,
           mae_train_naif_mixed_truth, mse_train_naif_mixed_truth, mae_test_naif_mixed_truth, mse_test_naif_mixed_truth,
           mae_train_mixed_obs, mse_train_mixed_obs, mae_test_mixed_obs, mse_test_mixed_obs,
           mae_train_naif_mixed_obs, mse_train_naif_mixed_obs, mae_test_naif_mixed_obs, mse_test_naif_mixed_obs,
           mae_train_lin_mixed_truth, mse_train_lin_mixed_truth, mae_test_lin_mixed_truth, mse_test_lin_mixed_truth,
           mae_train_lin_mixed_obs, mse_train_lin_mixed_obs, mae_test_lin_mixed_obs, mse_test_lin_mixed_obs,
           mae_train_fixed_truth, mse_train_fixed_truth, mae_test_fixed_truth, mse_test_fixed_truth,
           mae_train_naif_fixed_truth, mse_train_naif_fixed_truth, mae_test_naif_fixed_truth, mse_test_naif_fixed_truth,
           mae_train_fixed_obs, mse_train_fixed_obs, mae_test_fixed_obs, mse_test_fixed_obs,
           mae_train_naif_fixed_obs, mse_train_naif_fixed_obs, mae_test_naif_fixed_obs, mse_test_naif_fixed_obs,
           mae_train_lin_fixed_truth, mse_train_lin_fixed_truth, mae_test_lin_fixed_truth, mse_test_lin_fixed_truth,
           mae_train_lin_fixed_obs, mse_train_lin_fixed_obs, mae_test_lin_fixed_obs, mse_test_lin_fixed_obs)
  res <- data.frame(res)
  rownames(res) <- c("mae_train_mixed_truth", "mse_train_mixed_truth", "mae_test_mixed_truth", "mse_test_mixed_truth",
                     "mae_train_naif_mixed_truth", "mse_train_naif_mixed_truth", "mae_test_naif_mixed_truth", "mse_test_naif_mixed_truth",
                     "mae_train_mixed_obs", "mse_train_mixed_obs", "mae_test_mixed_obs", "mse_test_mixed_obs",
                     "mae_train_naif_mixed_obs", "mse_train_naif_mixed_obs", "mae_test_naif_mixed_obs", "mse_test_naif_mixed_obs",
                     "mae_train_lin_mixed_truth", "mse_train_lin_mixed_truth", "mae_test_lin_mixed_truth", "mse_test_lin_mixed_truth",
                     "mae_train_lin_mixed_obs", "mse_train_lin_mixed_obs", "mae_test_lin_mixed_obs", "mse_test_lin_mixed_obs",
                     "mae_train_fixed_truth", "mse_train_fixed_truth", "mae_test_fixed_truth", "mse_test_fixed_truth",
                     "mae_train_naif_fixed_truth", "mse_train_naif_fixed_truth", "mae_test_naif_fixed_truth", "mse_test_naif_fixed_truth",
                     "mae_train_fixed_obs", "mse_train_fixed_obs", "mae_test_fixed_obs", "mse_test_fixed_obs",
                     "mae_train_naif_fixed_obs", "mse_train_naif_fixed_obs", "mae_test_naif_fixed_obs", "mse_test_naif_fixed_obs",
                     "mae_train_lin_fixed_truth", "mse_train_lin_fixed_truth", "mae_test_lin_fixed_truth", "mse_test_lin_fixed_truth",
                     "mae_train_lin_fixed_obs", "mse_train_lin_fixed_obs", "mae_test_lin_fixed_obs", "mse_test_lin_fixed_obs")
  
  Dtrain[,"pred_mixed"] <- pred_train_mixed$pred[,'pred_ss']
  Dtrain[,"pred_fixed"] <- pred_train_fixed
  Dtrain[,"pred_naif_mixed"] <- pred_train_naif_mixed$pred[,'pred_ss']
  Dtrain[,"pred_naif_fixed"] <- pred_train_naif_fixed
  Dtrain[,"pred_lin_mixed"] <- pred_train_lin_mixed$pred[,'pred_ss']
  Dtrain[,"pred_lin_fixed"] <- pred_train_lin_fixed$pred[,'pred_ss']
  
  Pred_test_k <- Dtest[,c("individus", "temps")]
  Pred_test_k[,paste("pred_mixed", k, sep="_")] <- pred_test_mixed$pred[,'pred_ss']
  Pred_test_k[,paste("pred_fixed", k, sep="_")] <- pred_test_fixed
  Pred_test_k[,paste("pred_naif_mixed", k, sep="_")] <- pred_test_naif_mixed$pred[,'pred_ss']
  Pred_test_k[,paste("pred_lin_mixed", k, sep="_")] <- pred_test_lin_mixed$pred[,'pred_ss']
  Pred_test_k[,paste("pred_lin_fixed", k, sep="_")] <- pred_test_lin_fixed$pred[,'pred_ss']
  Pred_test_k[,paste("pred_naif_fixed", k, sep="_")] <- pred_test_naif_fixed
  
  #sortie
  write.csv2(x = Dtrain, file = paste("simulation", k ,".csv", sep = ""), row.names = FALSE)
  
  list(res, Pred_test_k)
}

results <- bind_cols(boucle[1,])
results <- as.data.frame(t(results))
predictions <- join_all(boucle[2,], by=c('individus','temps'))

#val_moy <- colMeans(results[c('µ_1','µ_2','µ_3','sigma_1','sigma_2','sigma_3','sigma_eps','beta_1','beta_2','beta_3')])
#val_moy <- as.data.frame(t(val_moy))
scores <- results[,c("mae_train_mixed_truth", "mse_train_mixed_truth", "mae_test_mixed_truth", "mse_test_mixed_truth",
                     "mae_train_naif_mixed_truth", "mse_train_naif_mixed_truth", "mae_test_naif_mixed_truth", "mse_test_naif_mixed_truth",
                     "mae_train_mixed_obs", "mse_train_mixed_obs", "mae_test_mixed_obs", "mse_test_mixed_obs",
                     "mae_train_naif_mixed_obs", "mse_train_naif_mixed_obs", "mae_test_naif_mixed_obs", "mse_test_naif_mixed_obs",
                     "mae_train_lin_mixed_truth", "mse_train_lin_mixed_truth", "mae_test_lin_mixed_truth", "mse_test_lin_mixed_truth",
                     "mae_train_lin_mixed_obs", "mse_train_lin_mixed_obs", "mae_test_lin_mixed_obs", "mse_test_lin_mixed_obs",
                     "mae_train_fixed_truth", "mse_train_fixed_truth", "mae_test_fixed_truth", "mse_test_fixed_truth",
                     "mae_train_naif_fixed_truth", "mse_train_naif_fixed_truth", "mae_test_naif_fixed_truth", "mse_test_naif_fixed_truth",
                     "mae_train_fixed_obs", "mse_train_fixed_obs", "mae_test_fixed_obs", "mse_test_fixed_obs", "mse_test_fixed_obs",
                     "mae_train_naif_fixed_obs", "mse_train_naif_fixed_obs", "mae_test_naif_fixed_obs", "mse_test_naif_fixed_obs",
                     "mae_train_lin_fixed_truth", "mse_train_lin_fixed_truth", "mae_test_lin_fixed_truth", "mse_test_lin_fixed_truth",
                     "mae_train_lin_fixed_obs", "mse_train_lin_fixed_obs", "mae_test_lin_fixed_obs", "mse_test_lin_fixed_obs")]
scores_moy <- colMeans(scores)                          
scores_moy <- as.data.frame(t(scores_moy))

## Ecrire les résultats
write.csv2(x = Dtest, file = "01_test.csv", row.names = FALSE)
write.csv(x = predictions, file = "Predictions.csv")
write.csv(x = results, "Résultats simulation.csv")
write.csv(x = truthY, "valeurs Y.csv")
write.csv(x = truthX, "valeurs X.csv")
#write.csv(x = val_moy, file = "valeurs_moyennes.csv")
write.csv(x = scores_moy, file = "Performances_moyennes.csv")

stopCluster(cl)
q("no")
