#Import data 
train <- read.csv(choose.files(),header = TRUE)
meal_info <- read.csv(choose.files(),header = TRUE)
center_info <- read.csv(choose.files(),header = TRUE)

#Merge data 
train_center_merged <- merge(train, center_info, by = "center_id")
full_data <- merge(train_center_merged, meal_info, by = "meal_id")

#Discount 
full_data$discount <- full_data$base_price - full_data$checkout_price
full_data

# Average discount and orders per meal
discount_orders_summary <- aggregate(discount ~ meal_id, data = full_data, FUN = mean)
meal_orders <- aggregate(num_orders ~ meal_id, data = full_data, sum, na.rm = TRUE) 
discount_orders_summary <- merge(discount_orders_summary,meal_orders, by = "meal_id")  
discount_orders_summary[order(discount_orders_summary$discount),]

library(ggplot2)
ggplot(full_data, aes(x = discount, y = num_orders)) +
  geom_point(alpha = 0.5) +
  labs(title = "Discount vs. Number of Orders", x = "Discount", y = "Number of Orders") +
  theme_minimal()

#Categorize the discount
discount_orders_summary$discount_category <- cut(discount_orders_summary$discount,
                          breaks = quantile(discount_orders_summary$discount, probs = c(0, 1/3, 1/6, 1), na.rm = TRUE),
                          labels = c("Low Discount", "Medium Discount", "High Discount"),
                          include.lowest = TRUE)
discount_orders_summary$num_orders_category <- cut(discount_orders_summary$num_orders, 
                             breaks = quantile(discount_orders_summary$num_orders, probs = c(0, 1/3, 1/6, 1)), 
                             labels = c("Low Order", "Medium Order", "High Order"), 
                             include.lowest = TRUE)

contingency_table <- table(discount_orders_summary$discount_category, discount_orders_summary$num_orders_category)
print(contingency_table)

anova_result <- aov(num_orders ~ discount_category, data = discount_orders_summary)
summary(anova_result)
chi_square_test <- chisq.test(contingency_table)

# View the results
print(chi_square_test)

# Assuming `discount_orders_summary` is already available

# Function to calculate confidence interval for each category
calculate_ci <- function(data, conf_level = 0.95) {
  n <- length(data)  # Sample size
  mean_val <- mean(data)  # Sample mean
  sd_val <- sd(data)  # Sample standard deviation
  se <- sd_val / sqrt(n)  # Standard error
  
  # t-score for the desired confidence level
  t_score <- qt(1 - (1 - conf_level) / 2, df = n - 1)
  
  # Confidence Interval Calculation
  margin_error <- t_score * se
  ci_lower <- mean_val - margin_error
  ci_upper <- mean_val + margin_error
  
  return(c(ci_lower, ci_upper))
}

# Calculate confidence intervals for each discount category
ci_results <- data.frame(discount_category = levels(discount_orders_summary$discount_category),
                         ci_lower = NA,
                         ci_upper = NA)

for (category in levels(discount_orders_summary$discount_category)) {
  # Filter data for each discount category
  category_data <- subset(discount_orders_summary, discount_category == category)$num_orders
  
  # Apply the CI function
  ci_values <- calculate_ci(category_data)
  
  # Store the results
  ci_results[ci_results$discount_category == category, c("ci_lower", "ci_upper")] <- ci_values
}

# Print the confidence intervals
print(ci_results)

