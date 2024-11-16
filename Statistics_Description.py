import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
train = pd.read_csv('train.csv')
meal_info = pd.read_csv('meal_info.csv')
center_info = pd.read_csv('fulfilment_center_info.csv')

# Merge data
train_center_merged = pd.merge(train, center_info, on='center_id')
full_data = pd.merge(train_center_merged, meal_info, on='meal_id')

# Calculate Orders per Meal
meal_orders = full_data.groupby('meal_id')['num_orders'].sum().reset_index()
meal_orders = meal_orders.sort_values(by='num_orders')
meal_orders.columns = ['meal_id', 'total_orders']

# Count unique weeks for each meal
meal_order_frequency = full_data.groupby('meal_id')['week'].nunique().reset_index()
meal_order_frequency.columns = ['meal_id', 'weeks_ordered']

# Merge with meal_orders to create meal_summary
meal_summary = pd.merge(meal_orders, meal_order_frequency, on='meal_id')

# Add average orders per week
meal_summary['average_orders_per_week'] = meal_summary['total_orders'] / meal_summary['weeks_ordered']

# View the meal summary
print(meal_summary)

# Descriptive statistics
print("Mean:", meal_summary['average_orders_per_week'].mean())
print("Median:", meal_summary['average_orders_per_week'].median())
print("Quantiles:", meal_summary['average_orders_per_week'].quantile([0.25, 0.5, 0.75]))
print("Range:", meal_summary['average_orders_per_week'].max() - meal_summary['average_orders_per_week'].min())
print("Variance:", meal_summary['average_orders_per_week'].var())
print("Standard Deviation:", meal_summary['average_orders_per_week'].std())

# Create histogram of average orders per week
plt.figure(figsize=(8, 6))
plt.hist(meal_summary['average_orders_per_week'], bins=30, edgecolor='black', color='skyblue')
plt.title("Distribution of Average Orders per Week per Meal")
plt.xlabel("Average Orders per Week")
plt.ylabel("Number of Meals")
plt.show()

# Average checkout price per meal
meal_check_price = full_data.groupby('meal_id')['checkout_price'].mean().reset_index()
meal_check_price.columns = ['meal_id', 'Average_meal_checkout_price']
meal_summary = pd.merge(meal_check_price, meal_summary, on='meal_id')
meal_summary = meal_summary.sort_values(by='total_orders')

