#' fct_load_data_find_hp
#'
#' @description Function to load data and get the list of dates and the loaded data
#'
#' @param max_date_eval The max date to be evaluated
#' @param data_file The data path
#'
#' @return A list with the vector of dates to be evaluated and the data
fct_load_data_find_hp <- function(max_date_eval = "2021-03-01",
                                  data_directory = "data/mice_scaled/"){

  data_files <- list.files(path = data_directory, pattern = "*.rds", full.names = TRUE)
  results <- list()
  # data_covid <- readRDS(file = data_file) %>%
  #   dplyr::filter(START_DATE <= max_date_eval)
  # ##### SELECT TRAINING DATES
  # # skip first 90 days before learning start
  # # (only one date over 2 to speed up hp selection)
  # vecDates <- data_covid$START_DATE[90:length(data_covid$START_DATE)]
  # vecDates <- vecDates[as.numeric(vecDates) %% 2 == 0]

  for (data_file in data_files) {
    # Load the dataset
    data_covid <- readRDS(file = data_file) %>%
      dplyr::filter(START_DATE <= max_date_eval)
    
    # Skip first 90 days before learning start and select dates (one date over 2)
    vecDates <- data_covid$START_DATE[90:length(data_covid$START_DATE)]
    vecDates <- vecDates[as.numeric(vecDates) %% 2 == 0]
    
    # Store the result in the list with the file name as the key
    results[[basename(data_file)]] <- list(data_covid = data_covid, vecDates = vecDates)
  }
  
  return(results)
}