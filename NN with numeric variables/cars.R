
# MLP with Categorical(qualititive) variables
# Dataset: cars.csv, 5 variables, 154 obstacle
# Target variable: price

data_set <- read.csv("cars.csv")
View(data_set)

data_set_scaled <- as.data.frame(scale(data_set[,-5]))
View(data_set_scaled)

min_price <- min(data_set$price)
max_price <- max(data_set$price)

#scale by this formula: (price - min_price) / (max_price - min_price)
data_set_scaled$price <- scale(data_set$price, center = min_price,
                               scale = max_price - min_price)

set.seed(123)
n <- sample(154, 90)
train_set <- data_set_scaled[n,] #90
test_set <- data_set_scaled[-n,] #remining 64

test_set_unscaled <- data_set[-n,]
View(test_set_unscaled)

require(neuralnet)
require(caret)
model_formula <- price ~ engine + hp + wgt + fuelcap
nn_model <- neuralnet(model_formula, train_set, algorithm = "rprop+", hidden = 5,
                      err.fct = "sse", act.fct = "logistic", rep = 1, stepmax = 1e06,
                      threshold = 0.01, linear.output = TRUE)
plot(nn_model)
plot(nn_model, show.weights = FALSE)
nn_model$result.matrix
nn_model$weights

predictions <- compute(nn_model, test_set[,-5])
predicted_probabilities <- predictions$net.result
head(predicted_probabilities, 10)

predictions_unscaled <- min_price + (max_price - min_price) * predicted_probabilities
head(predictions_unscaled, 10)

cor(predictions_unscaled, data_set$price[-n])^2


#---


rsquared_list <- NULL
for (i in 1:20) {
  print(i)
  net <- neuralnet(model_formula, train_set, algorithm = "rprop+", hidden = i,
                   err.fct = "sse", act.fct = "logistic", rep = 1, stepmax = 1e06,
                   threshold = 0.01, linear.output = TRUE)
  pred <- compute(net, test_set[,-5])
  pred_prob <- pred$net.result
  pred_unscaled <- min_price + (max_price - min_price) * pred_prob
  rsquared <- cor(pred_unscaled, data_set$price[-n])^2
  rsquared_list <- c(rsquared_list, rsquared)
}

rsquared_list
plot(rsquared_list)
which.max(rsquared_list)
max(rsquared_list)


#---


n_folds <- 5                                #K-fold cross-validation
folds <- createFolds(data_set_scaled$price, k = n_folds, list = FALSE)
class(folds)
View(folds)
table(folds)

rsquared_list <- NULL
for (i in 1:n_folds) {
  print(i)
  indextest <- which(folds==i)
  test_set <- data_set_scaled[indextest,]          #fold number i
  train_set <- data_set_scaled[-indextest,]        #other folds
  net <- neuralnet(model_formula, train_set, algorithm = "rprop+", hidden = 12,
                   err.fct = "sse", act.fct = "logistic", rep = 1, stepmax = 1e06,
                   threshold = 0.01, linear.output = FALSE)
  pred <- compute(net, test_set[,-5])
  pred_prob <- pred$net.result
  pred_unscaled <- min_price + (max_price - min_price) * pred_prob
  rsquared <- cor(pred_unscaled, data_set$price[indextest])^2
  rsquared_list <- c(rsquared_list, rsquared)
}

rsquared_list
plot(rsquared_list)
which.max(rsquared_list)
max(rsquared_list)

