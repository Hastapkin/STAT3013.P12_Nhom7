#Import data 
train <- read.csv(choose.files(),header = TRUE)
meal_info <- read.csv(choose.files(),header = TRUE)
center_info <- read.csv(choose.files(),header = TRUE)

#Merge data 
train_center_merged <- merge(train, center_info, by = "center_id")
full_data <- merge(train_center_merged, meal_info, by = "meal_id")

# Calculate Orders per Meal
meal_orders <- aggregate(num_orders ~ meal_id, data = full_data, sum, na.rm = TRUE)
meal_orders <- meal_orders[order(meal_orders$num_orders),]
colnames(meal_orders)[2] <- "total_orders"
meal_orders

# Count unique weeks for each meal
meal_order_frequency <- aggregate(week ~ meal_id, data = full_data, FUN = function(x) length(unique(x)))
colnames(meal_order_frequency)[2] <- "weeks_ordered"

# Merge with meal_orders to create meal_summary
meal_summary <- merge(meal_orders, meal_order_frequency, by = "meal_id")

# Add average orders per week
meal_summary$average_orders_per_week <- meal_summary$total_orders / meal_summary$weeks_ordered
meal_summary <- meal_summary[order(meal_summary$total_orders),]
# View the meal summary
mean(meal_summary$average_orders_per_week)
median(meal_summary$total_orders)
quantile(meal_summary$average_orders_per_week)
range(meal_summary$average_orders_per_week)
var(meal_summary$average_orders_per_week)
sd(meal_summary$average_orders_per_week)

library(ggplot2)

# Create histogram of average orders per week
ggplot(meal_summary, aes(x = average_orders_per_week)) +
  geom_histogram(binwidth = 5000, color = "black", fill = "skyblue") +
  labs(title = "Distribution of Average Orders per Week per Meal", 
       x = "Average Orders per Week", 
       y = "Number of Meals") +
  theme_minimal()



#Average checkout price per meal
meal_check_price <- aggregate(checkout_price ~ meal_id, data = full_data, mean, na.rm = TRUE)
colnames(meal_check_price)[2]<- "Average_meal_checkout_price"
meal_summary <- merge(meal_check_price, meal_summary, by = "meal_id")
meal_summary <- meal_summary[order(meal_summary$total_orders), ]
meal_summary





