library(rockchalk)
library(dplyr)
library(this.path)

set.seed(13092001)

N_ind <- 500
N_time <- 50
N_ind_per_grp <- 10

ind <- 1:N_ind
time <- 0:N_time

l <- list(ind, time)
dataframe <- rev(expand.grid(rev(l)))
colnames(dataframe) <- c("individus", "temps")

# 10 individus par group
dataframe$groupes <- ceiling(dataframe$individus / N_ind_per_grp)
N_grps <- length(unique(dataframe$group))

# Epsilon
sigma_epsilon <- c(1.5, 0.1, 0.1, 0.1, 0.02, 0.5, 0.01, 0.1)

# X
k <- 8
mu0 <- runif(k, -1, 1)
sig0 <- diag(c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
mu1 <- runif(k, -1, 1)
sig1 <- diag(c(0.5, 0.5, 0.1, 0.5, 1, 1, 0.5, 0.5))

# Y
mgamma <- c(0, 0, 0)
sdgamma <- diag(c(0.5, 0.5, 0.05))
g <- mvrnorm(N_grps, mgamma, sdgamma)

simul <- function(
    individus = ind, k = 5, df = dataframe,
    µ0 = mu0, µ1 = mu1, sigma0 = sig0,
    sigma1 = sig1, gamma = g, sigeps = sigma_epsilon) {
    ####
    alpha0 <- mvrnorm(length(individus), µ0, sigma0)
    alpha1 <- mvrnorm(length(individus), µ1, sigma1)

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

    df$y_mixed <- gamma[df$groupes, 1, drop = FALSE] +
        gamma[df$groupes, 2, drop = FALSE] * df$x1 * df$x5 +
        gamma[df$groupes, 3, drop = FALSE] * df$x2 * df$x6
    df$y_mixed_obs <- df$y_mixed + rnorm(length(df$y_mixed), 0, sigeps[8])

    df$y_fixed <- gamma[1, 1] +
        gamma[1, 2] * df$x1 * df$x5 +
        gamma[1, 3] * df$x2 * df$x6
    df$y_fixed_obs <- df$y_fixed + rnorm(length(df$y_fixed), 0, sigeps[8])

    return(df)
}

dataframe <- simul()

write.csv2(x = dataframe, file = paste(this.dir(), "/simulation.csv", sep = ""), row.names = FALSE)
