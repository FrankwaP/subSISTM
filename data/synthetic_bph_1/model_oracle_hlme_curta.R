## Import des bilbiothèques
  
  
library(rockchalk)
library(dplyr)
library(lcmm)
library(doParallel)
library(foreach)

set.seed(0)
## Définition des variables

cl <- makeCluster(10)
registerDoParallel(cl)

ind <- 1:500
time <- 0:25
l <- list(ind, time)
dataframe <- rev(expand.grid(rev(l)))
colnames(dataframe) <- c("individus", "temps")
k <- 7
# Epsilon
sigma_epsilon <- c(0.5, 0.1, 0.1, 0.1, 0.002, 0.05, 0.005, 0.1)

# X
mu0 <- runif(k, -10, 10)
sig0 <- diag(c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
mu1 <- runif(k, -1, 1) 
sig1 <- diag(c(0.5, 0.5, 0.1, 0.5, 1, 0.1, 0.5))
truthX <- data.frame("µ0"=mu0, "µ1" = mu1, "sigma0"=diag(sig0), "sigma1"=diag(sig1))
rownames(truthX) <- c('X1','X2','X3','X4','X5','X6','X7')

# Y
µ_gamma <- runif(3, -1, 1)
sdgamma <- diag(c(0.5, 0.5, 0.05))
truthY <- data.frame('µ' = µ_gamma, 'sigma²' = sdgamma, 'var_eps' = sigma_epsilon[8])

## Fonction simul()


simul <- function(
    individus = ind, df = dataframe, µ0 = mu0, µ1 = mu1, sigma0 = sig0,
    sigma1 = sig1, µg = µ_gamma, sigmag = sdgamma, sigeps = sigma_epsilon) {
  
  alpha0 <- mvrnorm(length(individus), µ0, sigma0)
  alpha1 <- mvrnorm(length(individus), µ1, sigma1)
  g_mixed <- mvrnorm(length(ind), µg, sigmag)
  g_fixed <- mvrnorm(length(ind), µg, diag(numeric(3)))
  
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
  
  df$y_mixed <- g_mixed[df$individus, 1] +
    g_mixed[df$individus, 2] * df$x2 * df$x5 +
    g_mixed[df$individus, 3] * df$x4 * df$x7
  
  df$y_mixed_obs <- df$y_mixed + rnorm(length(df$y_mixed), 0, sigeps[8])
  
  
  df$y_fixed <- g_fixed[df$individus, 1] + 
    g_fixed[df$individus, 2] * df$x2 * df$x5 + 
    g_fixed[df$individus, 3] * df$x4 * df$x7
  
  
  df$y_fixed_obs <- df$y_fixed + rnorm(length(df$y_fixed), 0, sigeps[8])
  
  return(df)
}


## Génération des données de test


Dtest <- simul()
write.csv2(x = Dtest, file = "01_test.csv", row.names = FALSE)


## Génération des datasets d'entrainement


boucle <- foreach(i=1:100, 
                  .combine=cbind, 
                  .packages=c("rockchalk", "dplyr", "lcmm", "doParallel", "foreach")) %dopar%
{
  Dtrain <- simul()
  
  #Modèle oracle sur les Y à effet mixed
  oracle_mixed <- hlme(y_mixed_obs ~ x2_x5 + x4_x7,
                       random=~ x2_x5 + x4_x7,
                       data= Dtrain, subject='individus')
  save(oracle_mixed, file = paste("oracle_mix", as.character(i) ,".rda", sep = ""))
  
  beta_k <- oracle_mixed$best[1:3]
  sigma_k <- oracle_mixed$best[c('varcov 1','varcov 3','varcov 6')]
  
  biais_beta <- beta_k - µ_gamma
  biais_sigma <- sigma_k - c(0.5, 0.5, 0.05)
  res_mixed <- c(beta_k, biais_beta, sigma_k, biais_sigma)
  
  pred_train_mixed <- predictY(oracle_mixed, newdata = Dtrain, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_train_mixed <- mean(abs(pred_train_mixed$pred[,'pred_ss'] - Dtrain$y_mixed))
  mse_train_mixed <- mean((pred_train_mixed$pred[,'pred_ss'] - Dtrain$y_mixed)^2)
  
  pred_test_mixed <-predictY(oracle_mixed, newdata = Dtest, var.time = 'temps', marg = FALSE, subject = 'individus' )
  mse_test_mixed <- mean((pred_test_mixed$pred[,'pred_ss'] - Dtest$y_mixed)^2)
  mae_test_mixed <- mean(abs(pred_test_mixed$pred[,'pred_ss'] - Dtest$y_mixed))
  
  #Modèle oracle sur les Y à effet fixes
  oracle_fixed <- lm(y_fixed_obs ~ x2_x5 + x4_x7, data=Dtrain)
  sum_fix <- summary(oracle_fixed)
  save(oracle_fixed, file = paste("oracle_fix", as.character(i) ,".rda", sep = ""))
  
  beta_k <- sum_fix$coefficients[,'Estimate']
  
  biais_beta <- beta_k - µ_gamma
  res_fixed <- c(beta_k, biais_beta)
  
  pred_train_fixed <- predict(oracle_fixed, newdata = Dtrain)
  mae_train_fixed <- mean(abs(pred_train_fixed - Dtrain$y_fixed))
  mse_train_fixed <- mean((pred_train_fixed - Dtrain$y_fixed)^2)
  
  pred_test_fixed <-predict(oracle_fixed, newdata = Dtest)
  mse_test_fixed <- mean((pred_test_fixed - Dtest$y_fixed)^2)
  mae_test_fixed <- mean(abs(pred_test_fixed - Dtest$y_fixed))
  
  res <- c(res_mixed, mae_train_mixed, mse_train_mixed, mae_test_mixed, mse_test_mixed, 
                    res_fixed, mae_train_fixed, mse_train_fixed, mae_test_fixed, mse_test_fixed)
  res <- data.frame(res)
  rownames(res) <- c("µ_1", "µ_2", "µ_3", "biais µ_1", "biais µ_2", " biais µ_3",
                     "sigma_1", "sigma_2", "sigma_3", "biais sigma_1", "biais sigma_2", " biais sigma_3",
                     "mae_train_mixed", "mse_train_mixed", "mae_test_mixed", "mse_test_mixed", 
                     "beta_1", "beta_2", "beta_3", "biais beta_1", "biais beta_2", "biais beta_3",
                     "mae_train_fixed", "mse_train_fixed", "mae_test_fixed", "mse_test_fixed")
  #sortie
  write.csv2(x = Dtrain, file = paste("simulation", as.character(i) ,".csv", sep = ""), row.names = FALSE)
  res
}
boucle <-  as.data.frame(t(boucle))

## Ecrire les résultats

write.csv(x = boucle, "Résultats simulation.csv")

write.csv(x = truthY, "valeurs Y.csv")
write.csv(x = truthX, "valeurs X.csv")

stopCluster(cl)
q("no")
