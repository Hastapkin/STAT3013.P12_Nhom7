# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import data
train = pd.read_csv('train.csv')
meal_info = pd.read_csv('meal_info.csv')
center_info = pd.read_csv('fulfilment_center_info.csv')

# Merge data
train_center_merged = pd.merge(train, center_info, on='center_id')
full_data = pd.merge(train_center_merged, meal_info, on='meal_id')

# Discount calculation
full_data['discount'] = full_data['base_price'] - full_data['checkout_price']

# Average discount and orders per meal
discount_orders_summary = full_data.groupby('meal_id')['discount'].mean().reset_index()
meal_orders = full_data.groupby('meal_id')['num_orders'].sum().reset_index()
discount_orders_summary = pd.merge(discount_orders_summary, meal_orders, on='meal_id')

# Sort by discount
discount_orders_summary = discount_orders_summary.sort_values(by='discount')

# Plot: Discount vs. Number of Orders
plt.figure(figsize=(8, 6))
sns.scatterplot(x='discount', y='num_orders', data=full_data, alpha=0.5)
plt.title("Discount vs. Number of Orders")
plt.xlabel("Discount")
plt.ylabel("Number of Orders")
plt.show()

# Categorize the discount
discount_orders_summary['discount_category'] = pd.cut(discount_orders_summary['discount'],
                                                     bins=[-np.inf, discount_orders_summary['discount'].quantile(1/3),
                                                           discount_orders_summary['discount'].quantile(2/3), np.inf],
                                                     labels=["Low Discount", "Medium Discount", "High Discount"])

discount_orders_summary['num_orders_category'] = pd.cut(discount_orders_summary['num_orders'],
                                                        bins=[-np.inf, discount_orders_summary['num_orders'].quantile(1/3),
                                                              discount_orders_summary['num_orders'].quantile(2/3), np.inf],
                                                        labels=["Low Order", "Medium Order", "High Order"])

# Contingency table
contingency_table = pd.crosstab(discount_orders_summary['discount_category'], discount_orders_summary['num_orders_category'])
print(contingency_table)

# ANOVA test
anova_result = stats.f_oneway(discount_orders_summary[discount_orders_summary['discount_category'] == 'Low Discount']['num_orders'],
                              discount_orders_summary[discount_orders_summary['discount_category'] == 'Medium Discount']['num_orders'],
                              discount_orders_summary[discount_orders_summary['discount_category'] == 'High Discount']['num_orders'])
print(anova_result)

# Chi-square test
chi_square_test = stats.chi2_contingency(contingency_table)
print(chi_square_test)

# Confidence Interval Function
def calculate_ci(data, conf_level=0.95):
    n = len(data)  # Sample size
    mean_val = np.mean(data)  # Sample mean
    sd_val = np.std(data, ddof=1)  # Sample standard deviation
    se = sd_val / np.sqrt(n)  # Standard error
    
    # t-score for the desired confidence level
    t_score = stats.t.ppf(1 - (1 - conf_level) / 2, df=n-1)
    
    # Confidence Interval Calculation
    margin_error = t_score * se
    ci_lower = mean_val - margin_error
    ci_upper = mean_val + margin_error
    
    return ci_lower, ci_upper

# Calculate confidence intervals for each discount category
ci_results = pd.DataFrame({'discount_category': ['Low Discount', 'Medium Discount', 'High Discount'],
                           'ci_lower': [np.nan, np.nan, np.nan],
                           'ci_upper': [np.nan, np.nan, np.nan]})

for category in ['Low Discount', 'Medium Discount', 'High Discount']:
    # Filter data for each discount category
    category_data = discount_orders_summary[discount_orders_summary['discount_category'] == category]['num_orders']
    
    # Apply the CI function
    ci_values = calculate_ci(category_data)
    
    # Store the results
    ci_results.loc[ci_results['discount_category'] == category, ['ci_lower', 'ci_upper']] = ci_values

# Print the confidence intervals
print(ci_results)
