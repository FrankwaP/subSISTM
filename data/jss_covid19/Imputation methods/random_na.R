library(dplyr)

# Load the RDS dataset
data <- readRDS("MNAR_30.rds")

# Replace NA values with random values from the same column, excluding date columns
data_filled <- data %>%
  mutate(across(where(~ !inherits(., "Date")), 
                 ~ ifelse(is.na(.), sample(na.omit(.), sum(is.na(.)), replace = TRUE), .)))

# Save the modified dataset back to an RDS file
saveRDS(data_filled, "random_filled_30.rds")
