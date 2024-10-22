library(mice)

# Load your dataset with missing values
missing_data <- readRDS("MNAR_30.rds")

# Perform multiple imputations using the "mean" method with a specified maxit
imputed_data <- mice(missing_data, m = 10, method = "cart")

for (i in 1:10) {
  complete_data <- complete(imputed_data, action = i)
  file_name <- paste0("mice_cart/imputed_dataset_", i, ".rds")
  saveRDS(complete_data, file = file_name)
}