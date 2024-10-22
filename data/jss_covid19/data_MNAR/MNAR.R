# Read the RDS file
data <- readRDS("df_obfuscated_epidemio.rds")

# Calculate the 70th percentile of URG
threshold <- quantile(data$URG_covid_19_COUNT, 0.7, na.rm = TRUE)

rows_to_modify <- data$URG_covid_19_COUNT > threshold

# Step 4: Set all columns except START_DATE to NA for those rows
data[rows_to_modify, !(names(data) %in% "START_DATE")] <- NA
# Save the modified dataset back to RDS if needed
print(head(data))
saveRDS(data, "MNAR_30.rds")