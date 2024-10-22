library(dplyr)
library(knitr)
library(ggplot2)


bool_precompute <- file.exists(here::here("data/precomputed_results.rds"))

if(bool_precompute){
  ls_use_case <- readRDS(file = here::here("data/precomputed_results.rds"))
  
# Generate the formatted table
formatted_table <- ls_use_case$table_perf_esn %>%
  knitr::kable(caption = "Model performance with several reservoir configurations.",
               col.names = c("Model", "MAE", "MRE", "MAEB", "MREB"),
               booktabs = TRUE,
               linesep = c('', '\\addlinespace'), digits = 2)

# Save the formatted table to a text file
writeLines(as.character(formatted_table), "data/plot/table_performance.txt")
  

# Save each plot to a file
# Example: plot_present_data
ggsave(file = "data/plot/plot_present_data.png", plot = ls_use_case$plot_present_data, width = 8, height = 8)

# Example: plot_reservoir_hp
ggsave(file = "data/plot/plot_reservoir_hp.png", plot = ls_use_case$plot_reservoir_hp, width = 8, height = 8)

# Example: plot_feature_input_scaling
ggsave(file = "data/plot/plot_feature_input_scaling.png", plot = ls_use_case$plot_feature_input_scaling, width = 8, height = 8)

# Example: plot_forecast
ggsave(file = "data/plot/plot_forecast.png", plot = ls_use_case$plot_forecast, width = 8, height = 8)

# Example: plot_before_aggregation_forecast
ggsave(file = "data/plot/plot_before_aggregation_forecast.png", plot = ls_use_case$plot_before_aggregation_forecast, width = 8, height = 6)

# Example: plot_aggregation
ggsave(file = "data/plot/plot_aggregation.png", plot = ls_use_case$plot_aggregation, width = 8, height = 6)

# Example: plot_reservoir_importance
ggsave(file = "data/plot/plot_reservoir_importance.png", plot = ls_use_case$plot_reservoir_importance, width = 8, height = 6)

# Example: plot_importance
ggsave(file = "data/plot/plot_importance.png", plot = ls_use_case$plot_importance, width = 8, height = 6)

}

print("done")