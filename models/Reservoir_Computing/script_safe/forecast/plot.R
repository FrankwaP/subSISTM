##### load packages and functions
library(dplyr)
library(ggplot2)
library(scales)

source(here::here("functions/fct_smoothing_derivative.R"))

df_obfuscated_epidemio <- readRDS("data/scaled_df.rds")

data_i <- fct_smoothing_derivative(data = df_obfuscated_epidemio,
                                   maxOutcomeDate = df_obfuscated_epidemio$START_DATE %>% max,
                                   forecast_days = 14)
feature_labels <- conservation_status <- c(
  "outcome" = "Hospitalisations t+14",
  "outcomeDeriv" = "Outcome",
  "input_scaling" = "Input scaling",
  "leaking_rate" = "Leaking rate",
  "ridge" = "Ridge",
  "spectral_radius" = "Spectral radius",
  "hosp" = "Hospitalizations",
  "P_TOUS_AGES" = "RT-PCR+",
  "P_60_90_PLUS_ANS" = "RT-PCR+ 60 yo+",
  "FRACP_TOUS_AGES" = "% RT-PCR+",
  "FRACP_60_90_PLUS_ANS" = "% RT-PCR+ 60 yo+",
  "URG_covid_19_COUNT" = "Emergency",
  "IPTCC.mean" = "IPTCC",
  "Vaccin_1dose" = "Vaccine",
  "hosp_rolDeriv7" = "Hospitalizations (d)",
  "P_TOUS_AGES_rolDeriv7" = "RT-PCR+ (d)",
  "P_60_90_PLUS_ANS_rolDeriv7" = "RT-PCR+ 60 yo+ (d)",
  "FRACP_TOUS_AGES_rolDeriv7" = "% RT-PCR+ (d)",
  "FRACP_60_90_PLUS_ANS_rolDeriv7" = "% RT-PCR+ 60 yo+ (d)",
  "URG_covid_19_COUNT_rolDeriv7" = "Emergency (d)",
  "IPTCC.mean_rolDeriv7" = "IPTCC (d)",
  "Vaccin_1dose_rolDeriv7" = "Vaccine (d)"
)
plot_present_data <- data_i %>%
  select(-START_DATE, -outcomeHosp, -ends_with("rolDeriv7")) %>%
  tidyr::pivot_longer(cols = -"outcomeDate") %>%
  mutate(name = factor(name, levels = names(feature_labels), labels = feature_labels)) %>%
  ggplot(mapping = aes(x = outcomeDate, y = value, color = name)) +
  geom_line() +
  geom_vline(xintercept = as.Date("2021-03-01"), linetype = 2) +
  scale_color_manual(values = c(rep("#FB8500",2), rep("#023047", 16))) +
  scale_x_date(date_breaks = "5 months", date_labels =  "%m-%y") +
  facet_wrap(name ~ ., scales = "free",
             ncol = 2) +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(x = "Date",
       y = "Value")
print("data_i loaded")

df_forecast_aggregated <- readRDS("data/forecasts/median_forecasts.rds")

table_perf_esn <- df_forecast_aggregated %>%
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
print("table done")
plot_forecast <- df_forecast_aggregated %>%
  tidyr::pivot_longer(cols = c(outcome, forecast, hosp)) %>%
  mutate(name = factor(name,
                       levels = c("outcome",
                                  "hosp",
                                  "forecast"),
                       labels = c("Outcome",
                                  "Baseline",
                                  "Model forecast"))) %>%
  ggplot(mapping = aes(x = outcomeDate, y = value, color = name)) +
  geom_line() +
  scale_color_manual(values = c("#023047", "#8ECAE6", "#FB8500")) +
  scale_x_date(date_breaks = "3 months", date_labels =  "%m-%y") +
  facet_wrap(model ~ ., ncol = 2) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  labs(x = "Date (month-year)",
       y = "Hospitalizations",
       color = "")
print("plot done")



saveRDS(object = list(plot_present_data = plot_present_data,
                      table_perf_esn = table_perf_esn,
                      plot_forecast = plot_forecast),
        file = "data/precomputed_results_median.rds")