## Import des bilbiothèques
  
  
library(rockchalk)
library(dplyr)
library(this.path)
library(ggplot2)
library(lcmm)

set.seed(0)

## Définition des variables
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
µ_gamma <- runif(3, -1, 1)
sdgamma <- diag(c(0.5, 0.5, 0.05))
truthY <- data.frame('µ' = µ_gamma, 'sigma²' = sdgamma, 'var_eps' = sigma_epsilon[8])

## Fonction simul()


simul <- function(
    individus = ind, k = 5, df = dataframe, µ0 = mu0, µ1 = mu1, sigma0 = sig0,
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
write.csv2(x = Dtest, file = paste(this.dir(), "/Simulations/01_test.csv", sep=""), row.names = FALSE)


## Génération des datasets d'entrainement


num_simulations <- 2
res_mixed <- list()
mse_train_mixed <- list()
mse_test_mixed <- list()
mae_train_mixed <- list()
mae_test_mixed <- list()
res_fixed <- list()
mse_train_fixed <- list()
mse_test_fixed <- list()
mae_train_fixed <- list()
mae_test_fixed <- list()


for (k in 1:num_simulations) {
  Dtrain <- simul()
  write.csv2(x = Dtrain, file = paste(this.dir(), "/Simulations/simulation", as.character(k) ,".csv", sep = ""), row.names = FALSE)
  
  #Modèle oracle sur les Y à effet mixed
  oracle_mixed <- hlme(y_mixed_obs ~ x2_x5 + x4_x7,
                       random=~ x2_x5 + x4_x7,
                       data= Dtrain, subject='individus',
                       nproc = 6)
  save(oracle_mixed, file = paste(this.dir(), "/Models_oracles/oracle_mix", as.character(k) ,".rda", sep = ""))
  
  beta_k <- oracle_mixed$best[1:3]
  sigma_k <- oracle_mixed$best[c('varcov 1','varcov 3','varcov 6')]
  
  biais_beta <- beta_k - µ_gamma
  biais_sigma <- sigma_k - c(0.5, 0.5, 0.05)
  res_mixed[[k]] <- c(beta_k, biais_beta, sigma_k, biais_sigma)
  
  pred_train_mixed <- predictY(oracle_mixed, newdata = Dtrain, var.time = 'temps', marg = FALSE, subject = 'individus')
  mae_train_mixed[k] <- mean(abs(pred_train_mixed$pred[,'pred_ss'] - Dtrain$y_mixed))
  mse_train_mixed[k] <- mean((pred_train_mixed$pred[,'pred_ss'] - Dtrain$y_mixed)^2)
  
  pred_test_mixed <-predictY(oracle_mixed, newdata = Dtest, var.time = 'temps', marg = FALSE, subject = 'individus' )
  mse_test_mixed[k] <- mean((pred_test_mixed$pred[,'pred_ss'] - Dtest$y_mixed)^2)
  mae_test_mixed[k] <- mean(abs(pred_test_mixed$pred[,'pred_ss'] - Dtest$y_mixed))
  
  #Modèle oracle sur les Y à effet fixes
  oracle_fixed <- lm(y_fixed_obs ~ x2_x5 + x4_x7, data=Dtrain)
  sum_fix <- summary(oracle_fixed)
  save(oracle_fixed, file = paste(this.dir(), "/Models_oracles/oracle_fix", as.character(k) ,".rda", sep = ""))
  
  beta_k <- sum_fix$coefficients[,'Estimate']
  
  biais_beta <- beta_k - µ_gamma
  res_fixed[[k]] <- c(beta_k, biais_beta)
  
  pred_train_fixed <- predict(oracle_fixed, newdata = Dtrain)
  mae_train_fixed[k] <- mean(abs(pred_train_fixed - Dtrain$y_fixed))
  mse_train_fixed[k] <- mean((pred_train_fixed - Dtrain$y_fixed)^2)
  
  pred_test_fixed <-predict(oracle_fixed, newdata = Dtest)
  mse_test_fixed[k] <- mean((pred_test_fixed - Dtest$y_fixed)^2)
  mae_test_fixed[k] <- mean(abs(pred_test_fixed - Dtest$y_fixed))
}

## Spaguetti plot
ggplot(Dtest, aes(y=x6, x=temps, colour=individus))+
  geom_point()+
  geom_line(aes(group=individus))+
  scale_x_continuous(breaks=0:9) 

## Ecrire les résultats

res_mixed <- data.frame(matrix(unlist(res_mixed), nrow=length(res_mixed), byrow=TRUE))
colnames(res_mixed) <- c('µ Gamma1','µ Gamma2', 'µ Gamma3',
                   'Biais µ Gamma 1', 'Biais µ Gamma 2', 'Biais µ Gamma3',
                   'sigma Gamma1', 'sigma Gamma2', 'sigma Gamma3',
                   'Biais sigma Gamma1', 'Biais sigma Gamma 2', 'Biais sigma Gamma3')
res_fixed <- data.frame(matrix(unlist(res_fixed), nrow=length(res_fixed), byrow=TRUE))
colnames(res_fixed) <- c('µ Gamma1','µ Gamma2', 'µ Gamma3',
                         'Biais µ Gamma 1', 'Biais µ Gamma 2', 'Biais µ Gamma3')


mse_train_mixed <- data.frame('MSE_train' = unlist(mse_train_mixed, use.names= FALSE))
mae_train_mixed <- data.frame('MAE_train' = unlist(mae_train_mixed, use.names= FALSE))
mse_test_mixed <- data.frame('MSE_train' = unlist(mse_test_mixed, use.names= FALSE))
mae_test_mixed <- data.frame('MAE_train' = unlist(mae_test_mixed, use.names= FALSE))
mse_train_fixed <- data.frame('MSE_train' = unlist(mse_train_fixed, use.names= FALSE))
mae_train_fixed <- data.frame('MAE_train' = unlist(mae_train_fixed, use.names= FALSE))
mse_test_fixed <- data.frame('MSE_train' = unlist(mse_test_fixed, use.names= FALSE))
mae_test_fixed <- data.frame('MAE_train' = unlist(mae_test_fixed, use.names= FALSE))

write.csv(x = truthY, file = paste(this.dir(), "/Résultats R script/valeurs Y.csv", sep = ""))
write.csv(x = truthX, file = paste(this.dir(), "/Résultats R script/valeurs X.csv", sep = ""))
write.csv(x = res_mixed, file = paste(this.dir(), "/Résultats R script/Valeurs et Biais mixed.csv", sep = ""))
write.csv(x = mse_train_mixed, file = paste(this.dir(), "/Résultats R script/MSE train mixed.csv", sep = ""))
write.csv(x = mae_train_mixed, file = paste(this.dir(), "/Résultats R script/MAE train mixed.csv", sep = ""))
write.csv(x = mae_test_mixed, file = paste(this.dir(), "/Résultats R script/MAE test mixed.csv", sep = ""))
write.csv(x = mse_test_mixed, file = paste(this.dir(), "/Résultats R script/MsE test mixed.csv", sep = ""))
write.csv(x = res_fixed, file = paste(this.dir(), "/Résultats R script/Valeurs et Biais fixed.csv", sep = ""))
write.csv(x = mse_train_fixed, file = paste(this.dir(), "/Résultats R script/MSE train fixed.csv", sep = ""))
write.csv(x = mae_train_fixed, file = paste(this.dir(), "/Résultats R script/MAE train fixed.csv", sep = ""))
write.csv(x = mae_test_fixed, file = paste(this.dir(), "/Résultats R script/MAE test fixed.csv", sep = ""))
write.csv(x = mse_test_fixed, file = paste(this.dir(), "/Résultats R script/MsE test fixed.csv", sep = ""))
