# Multiple Linear Regression

# Data Preprocessing

# Importing the dataset
dataset = read.csv('50_Startups.csv')
# dataset = dataset[, 2:3]

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training Set and Test Set
# install.packages('caTools')
library(caTools) # Select library
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8) # (Dependen var, %)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting Multiple Linear Regression to the Training set
# R understands (.) and that you want to express 
# the profit as a linear combination of all the independent variables
regressor = lm(formula = Profit ~ .,
               data = training_set)

# R&D Spend is the only cariable that causes a real impact on profit

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Automatic Backward Elimination
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)) {
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl) {
      j= which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars -1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1, 2, 3, 4, 5)]
backwardElimination(training_set, SL)