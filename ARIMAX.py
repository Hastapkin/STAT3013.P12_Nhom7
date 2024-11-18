# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load and merge data
train = pd.read_csv('train.csv')
meal_info = pd.read_csv('meal_info.csv')
center_info = pd.read_csv('fulfilment_center_info.csv')

# Merge the training data with center and meal information
train_center_merged = pd.merge(train, center_info, on='center_id')
full_data = pd.merge(train_center_merged, meal_info, on='meal_id')

weekly_data = full_data.groupby(['week', 'meal_id']).agg(
    total_orders=('num_orders', 'sum'),
    mean_checkout_price=('checkout_price', 'mean')
).reset_index()

#Define the specific meal IDs
specific_least_meal_id = 2104  # Least ordered meal ID
specific_most_meal_id = 2290   # Most ordered meal ID

#Filter data for specific meals
least_meal_data = weekly_data[weekly_data['meal_id'] == specific_least_meal_id]
most_meal_data = weekly_data[weekly_data['meal_id'] == specific_most_meal_id]

#Split the data into training (80%) and testing (20%) sets
split_ratio = 0.8
train_least_data = least_meal_data[:int(split_ratio * len(least_meal_data))]
test_least_data = least_meal_data[int(split_ratio * len(least_meal_data)):]
train_most_data = most_meal_data[:int(split_ratio * len(most_meal_data))]
test_most_data = most_meal_data[int(split_ratio * len(most_meal_data)):]

#Difference the data if necessary to achieve stationarity
def adf_test(series):
    result = adfuller(series,autolag ='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

train_least_data_diff = train_least_data['total_orders'].diff().dropna()
adf_test(train_least_data_diff)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(train_least_data['total_orders'], ax=axes[0])
axes[0].set_title("ACF for Least Ordered Meal")
plot_pacf(train_least_data['total_orders'], ax=axes[1])
axes[1].set_title("PACF for Least Ordered Meal")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(train_most_data['total_orders'], ax=axes[0])
axes[0].set_title("ACF for Most Ordered Meal")
plot_pacf(train_most_data['total_orders'], ax=axes[1])
axes[1].set_title("PACF for Most Ordered Meal")
plt.show()

#Fit ARIMA models with exogenous variable mean_checkout_price
#ARIMA model for least ordered meal
model_least = ARIMA(
    train_least_data['total_orders'], order=(1, 1, 1), 
    exog=train_least_data[['mean_checkout_price']]
).fit()

#ARIMA model for most ordered meal
model_most = ARIMA(
    train_most_data['total_orders'], order=(1, 0, 1), 
    exog=train_most_data[['mean_checkout_price']]
).fit()

#Forecast for the test data period
forecast_least = model_least.forecast(steps=len(test_least_data), exog=test_least_data[['mean_checkout_price']])
forecast_most = model_most.forecast(steps=len(test_most_data), exog=test_most_data[['mean_checkout_price']])


#Plot forecasts
plt.plot(test_least_data['week'], test_least_data['total_orders'], label="Actual", color="blue")
plt.plot(test_least_data['week'], forecast_least, label="Forecast", color="red")

plt.xlabel('Week')
plt.ylabel('Total Orders')
plt.legend()
plt.title('Actual vs Forecasted Least Orders')
plt.show()

plt.plot(test_most_data['week'], test_most_data['total_orders'], label="Actual", color="blue")
plt.plot(test_most_data['week'], forecast_most, label="Forecast", color="red")

plt.xlabel('Week')
plt.ylabel('Total Orders')
plt.legend()
plt.title('Actual vs Forecasted Most Orders')
plt.show()

#Evaluate the forecast performance using various metrics
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mae, mse, rmse, mape

mae_least, mse_least, rmse_least, mape_least = calculate_metrics(test_least_data['total_orders'], forecast_least)
mae_most, mse_most, rmse_most, mape_most = calculate_metrics(test_most_data['total_orders'], forecast_most)
mae_medium, mse_medium, rmse_medium, mape_medium = calculate_metrics(test_medium_data['total_orders'], forecast_medium)

print(f"Least Ordered Meal Metrics:\nMAE: {mae_least}, MSE: {mse_least}, RMSE: {rmse_least}, MAPE: {mape_least}%")
print(f"Most Ordered Meal Metrics:\nMAE: {mae_most}, MSE: {mse_most}, RMSE: {rmse_most}, MAPE: {mape_most}%")

#Residual analysis
residuals_least = test_least_data['total_orders'].values - forecast_least
residuals_most = test_most_data['total_orders'].values - forecast_most


print("Residuals (Least Ordered Meal):", residuals_least)
print("Residuals (Most Ordered Meal):", residuals_most)
