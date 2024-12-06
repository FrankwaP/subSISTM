library(lcmm)

data <- read.csv2('ml_pred.csv', sep = ',', dec = '.')
data$x6_x8 <- as.numeric(data$x6_x8)


data$individus <- as.numeric(data$individus)
random_hlme <- hlme(
  e_fixed ~ 1,
  random = ~  x2_x5 + x4_x7 + x6_x8,
  # cor = AR(temps) ou BM(temps)
  idiag = TRUE,
  data = data,
  subject = 'individus',
)

write.table(
  random_hlme$pred,
  "random_preds.csv",
  sep = ",",
  dec = ".",
  row.names = FALSE
)

saveRDS(random_hlme, 'random_hlme.Rds')
