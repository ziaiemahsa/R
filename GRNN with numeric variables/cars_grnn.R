
# Generalized Regression Neural Network
# Dataset: cars.csv, 5 variables, 154 obstacle
# Target variable: price

data_set <- read.csv("cars.csv")
View(data_set)

target <- log(data_set$price)
View(response)

predictors <- log(data_set)
predictors <- predictors[,-5] 
View(predictors)

set.seed(123)
n <- sample(154, 90)

require(grnn)
grnn_model <- learn(data.frame(target[n], predictors[n,]))
net <- smooth(grnn_model, sigma = 1)

predictions_train <- NULL
for (i in 1:90) {
  predictions_train[i] <- guess(net, as.matrix(predictors[n,][i,]))
}
  
head(predictions_train)
rsquared <- cor(predictions_train, target[n])^2
rsquared

predictions_test <- NULL
for (i in 1:64) {
  predictions_test[i] <- guess(net, as.matrix(predictors[-n,][i,]))
}

head(predictions_test)
rsquared <- cor(predictions_test, target[-n])^2
rsquared


#---


sigmas <- seq(1, 20, 1)
sigmas

rsquared_list <- NULL
for (j in 1:20) {
  print(j)
  predtest <- NULL
  net <- smooth(grnn_model, sigma = sigmas[j])
  for (i in 1:64) {
    predtest[i] <- guess(net, as.matrix(predictors[-n,][i,]))
  }
  rsquared_list[j] <- cor(predictions_test, target[-n])^2
}

rsquared_list
plot(rsquared_list)
which.max(rsquared_list)
max(rsquared_list)


#try with smaller sigmas


sigmas <- seq(0.1, 1, 0.1)
sigmas

rsquared_list <- NULL
for (j in 1:10) {
  print(j)
  predictions_test <- NULL
  net <- smooth(grnn_model, sigma = sigmas[j])
  for (i in 1:64) {
    predictions_test[i] <- guess(net, as.matrix(predictors[-n,][i,]))
  }
  rsquared_list[j] <- cor(predictions_test, target[-n])^2
}

rsquared_list
plot(rsquared_list)
which.max(rsquared_list)
max(rsquared_list)

