# ==============================================================================
# End-to-End Data Science Project in R
# Google Play Store Apps and Reviews Analysis
# ==============================================================================

# Load necessary libraries
# Note: If any package is missing, install it using install.packages("package_name")
library(dplyr)
library(ggplot2)
library(caret)
library(tidyr)
library(stringr)
library(randomForest)

# ------------------------------------------------------------------------------
# 1. Load the datasets
# ------------------------------------------------------------------------------
# We assume 'googleplaystore.csv' and 'googleplaystore_user_reviews.csv' 
# are in the current working directory.
apps_data <- read.csv("D:/ADYPU/Sem6/IAR/googleplaystore.csv", stringsAsFactors = FALSE)
reviews_data <- read.csv("D:/ADYPU/Sem6/IAR/googleplaystore_user_reviews.csv", stringsAsFactors = FALSE)

# ------------------------------------------------------------------------------
# 2. Data Cleaning
# ------------------------------------------------------------------------------
cat("\n--- Starting Data Cleaning ---\n")

# -- Clean apps_data --
# Remove duplicate apps computationally, keeping the one with the maximum number of reviews
apps_data <- apps_data %>%
  mutate(Reviews = as.numeric(Reviews)) %>%
  arrange(desc(Reviews)) %>%
  distinct(App, .keep_all = TRUE)

# Clean 'Installs': remove '+', ',', and convert to numeric format
apps_data$Installs <- str_replace_all(apps_data$Installs, "[+,]", "")
apps_data$Installs <- as.numeric(apps_data$Installs)

# Clean 'Price': remove '$' and convert to numeric
apps_data$Price <- str_replace_all(apps_data$Price, "[$]", "")
apps_data$Price <- as.numeric(apps_data$Price)

# Clean 'Size': convert 'M' to megabytes, 'k' to kilobytes, handle 'Varies with device'
apps_data$Size <- str_replace_all(apps_data$Size, "Varies with device", NA_character_)
size_cleaner <- function(size_str) {
  if (is.na(size_str)) return(NA)
  if (grepl("M", size_str)) {
    return(as.numeric(str_replace_all(size_str, "M", "")))
  } else if (grepl("k", size_str)) {
    return(as.numeric(str_replace_all(size_str, "k", "")) / 1024)
  } else {
    return(as.numeric(size_str))
  }
}
apps_data$Size <- sapply(apps_data$Size, size_cleaner)

# Convert categorical variables into factors
apps_data$Category <- as.factor(apps_data$Category)
apps_data$Type <- as.factor(apps_data$Type)
apps_data$Content.Rating <- as.factor(apps_data$Content.Rating)

# Handle Missing Values in apps_data by dropping NAs in crucial analytical columns
apps_data <- drop_na(apps_data, Rating, Installs, Size, Type)

# -- Clean reviews_data --
# Clean and handle missing sentiment values
# Ensure Sentiment_Polarity and Sentiment_Subjectivity are numeric
reviews_data <- reviews_data %>%
  filter(Sentiment != "nan" & !is.na(Sentiment)) %>%
  mutate(
    Sentiment_Polarity = as.numeric(Sentiment_Polarity),
    Sentiment_Subjectivity = as.numeric(Sentiment_Subjectivity)
  ) %>%
  drop_na(Sentiment_Polarity, Sentiment_Subjectivity)

# ------------------------------------------------------------------------------
# 3. Data Integration
# ------------------------------------------------------------------------------
cat("\n--- Integrating Data ---\n")

# Aggregate reviews data by App to get average sentiment polarity and subjectivity
app_sentiment <- reviews_data %>%
  group_by(App) %>%
  summarize(
    Avg_Sentiment_Polarity = mean(Sentiment_Polarity, na.rm = TRUE),
    Avg_Sentiment_Subjectivity = mean(Sentiment_Subjectivity, na.rm = TRUE)
  )

# Merge datasets using the "App" column
# Left join to keep all apps, effectively adding sentiment metrics where available
merged_data <- left_join(apps_data, app_sentiment, by = "App")

# ------------------------------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
cat("\n--- Exploratory Data Analysis Summary ---\n")
print(summary(apps_data %>% select(Rating, Reviews, Size, Installs, Price)))

# ------------------------------------------------------------------------------
# 5. Data Visualization
# ------------------------------------------------------------------------------
cat("\n--- Generating Visualizations ---\n")

# Visualization 1: Histogram of Ratings
# Shows the overall distribution of app ratings across the store
p1 <- ggplot(apps_data, aes(x = Rating)) +
  geom_histogram(binwidth = 0.1, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of App Ratings", x = "Rating", y = "Count")
print(p1)

# Visualization 2: Boxplot of Log10 Installs by Top 10 Categories
top_categories <- apps_data %>%
  count(Category, sort = TRUE) %>%
  head(10) %>%
  pull(Category)

p2 <- ggplot(filter(apps_data, Category %in% top_categories), 
             aes(x = reorder(Category, Installs, FUN = median), y = log10(Installs + 1))) +
  geom_boxplot(fill = "lightgreen") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Log10 Installs by Top Categories", x = "Category", y = "Log10(Installs)")
print(p2)

# Visualization 3: Rating vs Log10 Installs Scatter Plot
# Explores if higher rated apps tend to have more installs
p3 <- ggplot(apps_data, aes(x = Rating, y = log10(Installs + 1))) +
  geom_point(alpha = 0.3, color = "blue") +
  theme_minimal() +
  labs(title = "Rating vs Log10(Installs)", x = "Rating", y = "Log10(Installs)")
print(p3)

# Visualization 4: Average Sentiment Polarity vs App Rating
# Visualizes how actual text review sentiments relate to the star rating
plot_data_sentiment <- drop_na(merged_data, Avg_Sentiment_Polarity, Rating)
p4 <- ggplot(plot_data_sentiment, aes(x = Rating, y = Avg_Sentiment_Polarity)) +
  geom_point(alpha = 0.4, color = "darkorange") +
  geom_smooth(method = "lm", color = "red") +
  theme_minimal() +
  labs(title = "Average Sentiment Polarity vs App Rating", x = "Rating", y = "Sentiment Polarity")
print(p4)

# ------------------------------------------------------------------------------
# 6. Feature Engineering
# ------------------------------------------------------------------------------
# Create a log(installs) feature to handle heavily right-skewed installation numbers
merged_data$Log_Installs <- log10(merged_data$Installs + 1)

# We map the continuous problem into a Classification problem 
# Let's consider apps with 1 Million or more installs as "Successful"
merged_data$Is_Successful <- as.factor(ifelse(merged_data$Installs >= 1000000, "Yes", "No"))

# Prepare the final dataset for machine learning model development
# Selecting predictive features and dropping any missing data (Complete Cases)
ml_data <- merged_data %>%
  select(Is_Successful, Rating, Reviews, Size, Type, Price, Avg_Sentiment_Polarity) %>%
  drop_na()

# ------------------------------------------------------------------------------
# 7. Machine Learning Model (Classification Model to predict App Success)
# ------------------------------------------------------------------------------
cat("\n--- Building Machine Learning Model ---\n")

# Perform Train-Test Split (80% training data, 20% test data)
set.seed(123) 
trainIndex <- createDataPartition(ml_data$Is_Successful, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
data_train <- ml_data[ trainIndex,]
data_test  <- ml_data[-trainIndex,]

# Build a Random Forest Model using the caret package
# 5-fold Cross-Validation for robust performance estimation
fitControl <- trainControl(method = "cv", number = 5)

cat("Training Random Forest Classifier (this may take a moment)...\n")
rf_model <- train(Is_Successful ~ ., data = data_train, 
                 method = "rf", 
                 trControl = fitControl, 
                 importance = TRUE,
                 ntree = 100) # Standard count for balance between robustness and training time

print(rf_model)

# ------------------------------------------------------------------------------
# 8. Model Evaluation
# ------------------------------------------------------------------------------
cat("\n--- Evaluating Model on Test Set ---\n")

# Execute predictions against the unseen test partition
predictions <- predict(rf_model, newdata = data_test)

# Evaluate using Confusion Matrix (provides Accuracy, Sens/Spec, etc.)
conf_matrix <- confusionMatrix(predictions, data_test$Is_Successful)
print(conf_matrix)

# ------------------------------------------------------------------------------
# 9. Interpretation
# ------------------------------------------------------------------------------
# Feature Importance extraction
cat("\n--- Feature Importance Insight ---\n")
importance_scores <- varImp(rf_model, scale = FALSE)
print(importance_scores)
plot(importance_scores, main="Feature Importance for Predicting App Success")

# Summary and Interpretation string:
# 1. We engineered a classification target indicating whether an app achieved 1M+ installs.
# 2. Random Forest identifies 'Reviews' as a dominant feature. High review count naturally 
#    correlates heavily with a massive existing install base.
# 3. 'Size', 'Rating', and 'Avg_Sentiment_Polarity' add predictive nuances on usability constraints 
#    and textual enjoyment indicators.
# 4. Our model's Accuracy on the holdout sets points out how securely these initial metrics 
#    estimate broad market uptake or target hit-ratios.
