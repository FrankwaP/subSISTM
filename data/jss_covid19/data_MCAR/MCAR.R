library(dplyr)


data <- readRDS("df_obfuscated_epidemio.rds")
percentage <- 30

num_delete <- ceiling(nrow(data) * (percentage / 100))

rows_to_delete <- sample(nrow(data), num_delete)

set_missing <- function(df, rows, date_col) {
  df[rows, -which(names(df) == date_col)] <- NA
  return(df)
}

data_modified <- set_missing(data, rows_to_delete, "START_DATE")
print(head(data_modified))
saveRDS(data_modified, "missing_30.rds")