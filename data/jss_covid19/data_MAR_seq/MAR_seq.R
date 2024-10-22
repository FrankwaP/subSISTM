library(dplyr)

a <- 1
b <- 2

data <- readRDS("df_obfuscated_epidemio.rds")

# Shift the URG_covid_19_COUNT values by one position
data$urg <- lag(data$URG_covid_19_COUNT)

# Handle the first entry (since lag introduces an NA)
data$urg[is.na(data$urg)] <- min(data$URG_covid_19_COUNT)  # Or some other appropriate value

# Normalize the urg_prev values
data$urg <- (data$urg - min(data$urg)) / (max(data$urg) - min(data$urg))

data$missing_prob <- exp(b * data$urg) / (a + exp(b*data$urg))

set.seed(34)  # For reproducibility

# Define the range for sequence lengths
min_sequence_length <- 1
max_sequence_length <- 4

# Create missing sequences
data$missing_indicator <- rep(0, nrow(data))
i <- 1
while (i <= nrow(data)) {
  if (runif(1) > data$missing_prob[i]) {
    # Randomly determine the length of the missing sequence
    seq_length <- sample(min_sequence_length:max_sequence_length, 1)
    print(seq_length)
    seq_end <- min(i + seq_length - 1, nrow(data))
    data$missing_indicator[i:seq_end] <- 1
    i <- seq_end + 5  # Skip to the end of the missing sequence
  } else {
    i <- i + 1
  }
}

# Create the final dataset with NA values for missing entries
data_with_na <- data %>%
  mutate(across(-c(START_DATE, missing_prob, missing_indicator), ~ ifelse(missing_indicator == 1, NA, .)))

data_with_na <- data_with_na %>% select(-missing_prob, -missing_indicator, -urg)

# Display the first few rows of the resulting dataset
print(data_with_na)

saveRDS(data_with_na, "data_MAR_30_seq.rds")
