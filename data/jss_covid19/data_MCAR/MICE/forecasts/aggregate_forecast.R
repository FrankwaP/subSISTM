# Load necessary libraries
library(dplyr)
library(readr)

# Function to load and prepare data
load_and_prepare_data <- function(file_path) {
  data <- readRDS(file_path)
  return(data)
}

# List of RDS file paths
file_paths <- c("forecast_1.rds", "forecast_2.rds", "forecast_3.rds", 
                "forecast_4.rds", "forecast_5.rds", "forecast_6.rds", 
                "forecast_7.rds", "forecast_8.rds", "forecast_9.rds", 
                "forecast_10.rds")


# Load all datasets
datasets <- lapply(file_paths, load_and_prepare_data)

# Combine datasets into a single data frame
combined_data <- bind_rows(datasets)

mean_forecasts <- combined_data %>%
  group_by(START_DATE, model) %>%
  summarise(mean_forecast = mean(forecast, na.rm = TRUE), .groups = 'drop')

median_forecasts <- combined_data %>%
  group_by(START_DATE, model) %>%
  summarise(median_forecast = median(forecast, na.rm = TRUE), .groups = 'drop')

print(head(mean_forecasts))
  # Replace forecast column with mean forecasts
data_with_mean <- combined_data %>%
  left_join(mean_forecasts, by = c("START_DATE", "model")) %>%
  mutate(forecast = mean_forecast) %>%
  select(-mean_forecast)

# Replace forecast column with median forecasts
data_with_median <- combined_data %>%
  left_join(median_forecasts, by = c("START_DATE", "model")) %>%
  mutate(forecast = median_forecast) %>%
  select(-median_forecast)


  # Save the resulting datasets
saveRDS(data_with_mean, "mean_forecasts.rds")
saveRDS(data_with_median, "median_forecasts.rds")

# Print the first few rows of the datasets to verify
print(head(data_with_mean))
print(head(data_with_median))