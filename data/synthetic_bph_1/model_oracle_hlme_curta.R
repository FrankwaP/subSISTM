## Import des bilbiothèques
  
  
library(rockchalk)
library(dplyr)
library(lcmm)
library(doParallel)
library(foreach)

## Définition des variables

cl <- makeCluster(20)
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
sig1 <- diag(c(0.5, 0.5, 0.1, 0.5, 1, 0.2, 0.5))

# Y
µ_gamma <- runif(3, -1, 1)
sdgamma <- diag(c(0.5, 0.5, 0.05))
truthX <- data.frame("µ0"=mu0, "µ1" = mu1, "sigma0"=diag(sig0), "sigma1"=diag(sig1))
rownames(truthX) <- c('X1','X2','X3','X4','X5','X6','X7')

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


num_simulations <- 100
res <- list()
mse_train_oracle <- list()
mse_test_oracle <- list()
mae_train_oracle <- list()
mae_test_oracle <- list()


boucle <- foreach(i=1:100, .combine=cbind) %dopar%
{
  Dtrain <- simul()
  write.csv2(x = Dtrain, file = paste("simulation", as.character(k) ,".csv", sep = ""), row.names = FALSE)
  
  oracle_mixed <- hlme(y_mixed_obs ~ x2_x5 + x4_x7,
                       random=~ x2_x5 + x4_x7,
                       data= Dtrain, subject='individus',
                       nproc = 15)
  save(oracle_mixed, file = paste("oracle", as.character(k) ,".rda", sep = ""))
  
  beta_k <- oracle_mixed$best[1:3]
  sigma_k <- oracle_mixed$best[c('varcov 1','varcov 3','varcov 6')]
  biais_beta <- beta_k - µ_gamma
  biais_sigma <- sigma_k - c(0.5, 0.5, 0.05)
  res[[k]] <- c(beta_k, biais_beta, sigma_k, biais_sigma)
  
  pred_train <- predictY(oracle_mixed, newdata = Dtrain, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_train_oracle[k] <- mean(abs(pred_train$pred[,'pred_ss'] - Dtrain$y_mixed))
  mse_train_oracle[k] <- mean((pred_train$pred[,'pred_ss'] - Dtrain$y_mixed)^2)
  
  pred <-predictY(oracle_mixed, newdata = Dtest, var.time = 'temps', marg = FALSE, subject = 'individus' )
  mse_test_oracle[k] <- mean((pred$pred[,'pred_ss'] - Dtest$y_mixed)^2)
  mae_test_oracle[k] <- mean(abs(pred$pred[,'pred_ss'] - Dtest$y_mixed))
}


## Ecrire les résultats



res <- data.frame(matrix(unlist(res), nrow=length(res), byrow=TRUE))
colnames(res) <- c('µ Gamma1','µ Gamma2', 'µ Gamma3',
                   'Biais µ Gamma 1', 'Biais µ Gamma 2', 'Biais µ Gamma3',
                   'sigma Gamma1', 'sigma Gamma2', 'sigma Gamma3',
                   'Biais sigma Gamma1', 'Biais sigma Gamma 2', 'Biais sigma Gamma3')


mse_train_oracle <- data.frame('MSE' = unlist(mse_train_oracle, use.names= FALSE))
mae_train_oracle <- data.frame('MAE' = unlist(mae_train_oracle, use.names= FALSE))
mse_test_oracle <- data.frame('MSE' = unlist(mse_test_oracle, use.names= FALSE))
mae_test_oracle <- data.frame('MAE' = unlist(mae_test_oracle, use.names= FALSE))


write.csv(x = mse_train_oracle, file = paste( "MSE train.csv", sep = ""))
write.csv(x = res, file = paste( "Valeurs et Biais.csv", sep = ""))
write.csv(x = mae_train_oracle, file = paste( "MAE train.csv", sep = ""))
write.csv(x = mae_test_oracle, file = paste( "MAE test.csv", sep = ""))
write.csv(x = mse_test_oracle, file = paste( "MSE test.csv", sep = ""))

stopCluster(cl)
q("no")
