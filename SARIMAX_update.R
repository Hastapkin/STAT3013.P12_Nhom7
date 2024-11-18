# Load necessary libraries
library(forecast)  # For time series forecasting
library(tseries)   # For time series analysis
library(dplyr)     # For data manipulation

# Step 1: Load and merge data
train <- read.csv(choose.files(), header = TRUE)
meal_info <- read.csv(choose.files(), header = TRUE)
center_info <- read.csv(choose.files(), header = TRUE)

# Merge the training data with center and meal information
train_center_merged <- merge(train, center_info, by = "center_id")
full_data <- merge(train_center_merged, meal_info, by = "meal_id")

# Step 2: Aggregate data weekly
weekly_data <- full_data %>%
  group_by(week, meal_id) %>%
  summarise(
    total_orders = sum(num_orders),
    mean_checkout_price = mean(checkout_price),
    .groups = 'drop'
  )

# Step 3: Define the specific meal IDs
specific_least_meal_id <- 2104  # Least ordered meal ID
specific_most_meal_id <- 2290   # Most ordered meal ID

# Step 4: Filter data for specific meals
least_meal_data <- subset(weekly_data, meal_id == specific_least_meal_id)
most_meal_data <- subset(weekly_data, meal_id == specific_most_meal_id)

# Step 5: Split the data into training (80%) and testing (20%) sets
train_least_data <- least_meal_data[1:floor(0.8 * nrow(least_meal_data)), ]
test_least_data <- least_meal_data[(floor(0.8 * nrow(least_meal_data)) + 1):nrow(least_meal_data), ]
train_most_data <- most_meal_data[1:floor(0.8 * nrow(most_meal_data)), ]
test_most_data <- most_meal_data[(floor(0.8 * nrow(most_meal_data)) + 1):nrow(most_meal_data), ]

# Step 6: Create time series objects for total orders (weekly frequency)
train_least_ts <- ts(train_least_data$total_orders, frequency = 52)
train_most_ts <- ts(train_most_data$total_orders, frequency = 52)


# Step 7: Perform Augmented Dickey-Fuller (ADF) test for stationarity
adf_least_result <- adf.test(train_least_ts)
print(adf_least_result)
adf_most_result <- adf.test(train_most_ts)
print(adf_most_result)

# Step 8: Difference the data if necessary to achieve stationarity
differenced_least_orders <- diff(train_least_ts)
adf_least_result <- adf.test(differenced_least_orders)
print(adf_least_result)

# Plot ACF and PACF to identify potential ARIMA parameters
acf(train_least_data$total_orders)
pacf(train_least_data$total_orders)
acf(train_most_data$total_orders)
pacf(train_most_data$total_orders)

# Step 9: Fit ARIMAX models using mean_checkout_price as the exogenous variable
# ARIMAX model for least ordered meal
modified_sarimax_model_least <- Arima(
  train_least_ts,
  order = c(1, 1, 1),
  xreg = cbind(train_least_data$mean_checkout_price)
)

# ARIMAX model for most ordered meal
modified_sarimax_model_most <- Arima(
  train_most_ts,
  order = c(1, 0, 1),
  xreg = cbind(train_most_data$mean_checkout_price)
)

# Step 10: Forecast for the test data period
# Forecast for least ordered meal
forecast_least_values <- forecast(
  modified_sarimax_model_least,
  xreg = cbind(test_least_data$mean_checkout_price),
  h = nrow(test_least_data)
)
plot(forecast_least_values)

# Forecast for most ordered meal
forecast_most_values <- forecast(
  modified_sarimax_model_most,
  xreg = cbind(test_most_data$mean_checkout_price),
  h = nrow(test_most_data)
)
plot(forecast_most_values)



# Step 11: Evaluate the forecast performance using various metrics
library(Metrics)

# Evaluation for least ordered meal
actual_least_values <- test_least_data$total_orders
forecasted_least_values <- forecast_least_values$mean
mae_least <- mae(actual_least_values, forecasted_least_values)
mse_least <- mse(actual_least_values, forecasted_least_values)
rmse_least <- rmse(actual_least_values, forecasted_least_values)
mape_least <- mape(actual_least_values, forecasted_least_values)
cat("Least Ordered Meal Metrics:\n")
cat("MAE:", mae_least, "\n")
cat("MSE:", mse_least, "\n")
cat("RMSE:", rmse_least, "\n")
cat("MAPE:", mape_least, "%\n")

# Evaluation for most ordered meal
actual_most_values <- test_most_data$total_orders
forecasted_most_values <- forecast_most_values$mean
mae_most <- mae(actual_most_values, forecasted_most_values)
mse_most <- mse(actual_most_values, forecasted_most_values)
rmse_most <- rmse(actual_most_values, forecasted_most_values)
mape_most <- mape(actual_most_values, forecasted_most_values)
cat("Most Ordered Meal Metrics:\n")
cat("MAE:", mae_most, "\n")
cat("MSE:", mse_most, "\n")
cat("RMSE:", rmse_most, "\n")
cat("MAPE:", mape_most, "%\n")

# Step 12: Residual analysis
residuals_least <- actual_least_values - forecasted_least_values
residuals_most <- actual_most_values - forecasted_most_values

# Print residuals
print(residuals_least)
print(residuals_most)

