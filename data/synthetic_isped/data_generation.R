library(rockchalk)
library(dplyr)
library(this.path)

set.seed(13092001)

ind <- 1:500
time <- 0:50
l = list(ind, time)
dataframe <- rev(expand.grid(rev(l)))
colnames(dataframe) <- c("individus", "temps")
k <- 8
# Epsilon
sigma_epsilon <- c(1.5, 0.1, 0.1, 0.1, 0.02, 0.5, 0.01, 0.1)
# X
mu0 <- runif(k, -1, 1)
sig0 <- diag(c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))

mu1 <- runif(k, -1, 1)
sig1 <- diag(c(0.5, 0.5, 0.1, 0.5, 1, 1, 0.5, 0.5))

# Y
mgamma <- numeric(3)
sdgamma <- diag(c(0.5, 0.5, 0.05))
g <- mvrnorm(length(ind), mgamma, sdgamma)

simul <- function(individus = ind, k = 5, df = dataframe, µ0 = mu0, µ1 = mu1, sigma0 = sig0,
    sigma1 = sig1, gamma = g, sigeps = sigma_epsilon) {
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
    df$x5 <- alpha0[df$individus, 5]/(1 + exp(-alpha1[df$individus, 5] * df$temps))
    df$x5_obs <- df$x5 + rnorm(length(df$x5), 0, sigeps[5])
    df$x6 <- pmax(0, (alpha0[df$individus, 6] + 0.1 * alpha1[df$individus, 6] * df$temps)^2)
    df$x6_obs <- df$x6 + rnorm(length(df$x6), 0, sigeps[6])
    df$x7 <- alpha0[df$individus, 7]/(1 + exp(alpha1[df$individus, 7] * df$temps))
    df$x7_obs <- df$x7 + rnorm(length(df$x7), 0, sigeps[7])
    df <- df %>%
        mutate(x8 = ifelse(df$individus%%2 == 0, 1, 0))

    df$y <- gamma[df$individus, 1] + gamma[df$individus, 2] * df$x1 * df$x5 + gamma[df$individus,
        3] * df$x2 * df$x6
    df$y_obs <- df$y + rnorm(length(df$y), 0, sigeps[8])
    return(df)
}
dataframe <- simul()
plot(time, dataframe[dataframe$individus == 4, ]$y_obs, type = "l")
write.csv(dataframe, paste(this.dir(), "/simulation.csv", sep = ""))
