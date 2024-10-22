library(dplyr)

# Step 1: Load all datasets
# Assuming your datasets are named ds1, ds2, ..., ds10

ds1 <- readRDS("forecast_1.rds")
ds2 <- readRDS("forecast_2.rds")
ds3 <- readRDS("forecast_3.rds")
ds4 <- readRDS("forecast_4.rds")
ds5 <- readRDS("forecast_5.rds")
ds6 <- readRDS("forecast_6.rds")
ds7 <- readRDS("forecast_7.rds")
ds8 <- readRDS("forecast_8.rds")
ds9 <- readRDS("forecast_9.rds")
ds10 <- readRDS("forecast_10.rds")

dataset_list <- list(ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10)

# Step 2: Combine all datasets
combined_data <- bind_rows(dataset_list)

# Step 3 & 4: Group by date and model, then calculate mean forecast
averaged_data <- combined_data %>%
  group_by(START_DATE, outcomeDate, model) %>%
  summarize(
    forecast = median(forecast, na.rm = TRUE),
    outcome = first(outcome),
    hosp= first(hosp)
  ) %>%
  ungroup()

print(averaged_data)
saveRDS(averaged_data, "median_forecasts.rds")