library(zoo)

data_missing <- readRDS("MNAR_30.rds")

impute_all_columns <- function(data) {
  numeric_cols <- sapply(data, is.numeric)

  data[, numeric_cols] <- lapply(data[, numeric_cols], function(col) {
    na.approx(col, x = data$START_DATE, na.rm = FALSE)
  })
  
  return(data)
}

imputed_data <- impute_all_columns(data_missing)

saveRDS(imputed_data, "single_linear_imputed_MNAR_30.rds")

