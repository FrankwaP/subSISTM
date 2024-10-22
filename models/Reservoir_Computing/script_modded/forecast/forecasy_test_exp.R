##### load packages and functions
library(dplyr)
library(ggplot2)
library(scales)


df_forecast_aggregated <- readRDS("final_dataset_with_all_models.rds")


table_perf_esn <- df_forecast_aggregated %>%
  dplyr::mutate(forecast = if_else(forecast < 10, 10, forecast),
                hosp = if_else(hosp < 10, 10, hosp),
                outcome = if_else(outcome < 10, 10, outcome)) %>%
  dplyr::mutate(absolute_error = abs(forecast - outcome),
                relative_error = absolute_error/outcome,
                baseline_absolute_error = abs(hosp - outcome),
                absolute_error_delta_baseline = absolute_error - baseline_absolute_error,
                absolute_error_relative_baseline = absolute_error/baseline_absolute_error) %>%
  group_by(model) %>%
  summarise(mae = mean(absolute_error),
            mre = median(relative_error, na.rm = TRUE),
            mae_baseline = mean(absolute_error_delta_baseline),
            mre_baseline = median(absolute_error_relative_baseline, na.rm = TRUE),
            .groups = "drop")

print(table_perf_esn)
write.table(table_perf_esn, file = "table_base.txt", sep = "\t", row.names = FALSE, quote = FALSE)
