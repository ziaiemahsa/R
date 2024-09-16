
# probabilistic neural networks
# Dataset: bank.csv, 7 variables, 700 obstacle
# Target variable: default

data_set <- read.csv("bank.csv")
View(data_set)

target <- data_set$default
new_dataset <- data_set[,-7]                                  #remove target
new_dataset <- as.data.frame(scale(new_dataset))              #scale 6 vat=riables
new_dataset <- cbind(default=as.factor(target), new_dataset)  #add target

View(new_dataset)
class(new_dataset)

set.seed(123)
n <- sample(700, 500)
train_set <- new_dataset[n,] #500
test_set <- new_dataset[-n,] #remining 200

require(pnn)
pnn_model <- learn(train_set)
View(pnn_model)

net <- smooth(pnn_model, sigma = 0.5)

performance <- perf(net)
class(performance)
View(performance)

performance$k                                       #num of variables
performance$observed                                #num of targets
performance$guessed                                 #predicted of targets
performance$fails                                   #wrong predictions
performance$success                                 #correct predictions
performance$success_rate                            #accuracy
table(performance$observed, performance$guessed)    #confusion matrix
performance$bic                                     #bayesian information criterion 
                                                       #the smaller bic, the better

require(caret)
guess(net, as.matrix(test_set[1, -1]))              #extract the first row without target value
guess(net, as.matrix(test_set[1, -1]))$category

predictions_list <- NULL
for (i in 1:nrow(test_set)) {
  predictions_list[i] <- guess(net, as.matrix(test_set[i, -1]))$category
}

head(predictions_list)

accuracy <- mean(predictions_list == test_set[,1])
accuracy

confusion_matrix <- table(predictions_list, test_set[,1])
confusion_matrix

precision(confusion_matrix)
recall(confusion_matrix)


#---


sigmas <- seq(0.5, 30, 0.5)
sigmas

success_rate_list <- NULL
for (j in 1:60) {
  print(j)
  net <- smooth(pnn_model, sigma = sigmas[j])
  success_rate_list[j] <- perf(net)$success_rate
}

max(success_rate_list)
sigmas[which.max(success_rate_list)]

net_optimal <- smooth(model, sigma = 0.5)
performance_optimal <- perf(net_optimal)

predictions_list <- NULL
for (i in 1:nrow(test_set)) {
  predictions_list[i] <- guess(net_optimal, as.matrix(test_set[i, -1]))$category
}

head(predictions_list)

confusion_matrix <- table(test_set[,1], predictions_list)
confusion_matrix

accuracy <- mean(predictions_list == test_set[,1])
accuracy

precision(confusion_matrix)
recall(confusion_matrix)


#---


n_folds <- 5                                #K-fold cross-validation
folds <- createFolds(data_set$income, k = n_folds, list = FALSE)
class(folds)
View(folds)
table(folds)

accuracy_list <- NULL
for (i in 1:n_folds) {
  print(i)
  indextest <- which(folds == i)
  test_set <- new_dataset[indextest,]
  train_set <- new_dataset[-indextest,]
  model <- learn(train_set)
  net <- smooth(model, sigma = 5.5)
  predictions <- NULL
  for (j in 1:nrow(test_set)) {
    predictions[j] <- guess(net, as.matrix(test_set[j, -1]))$category
  }
  accuracy <- mean(predictions == test_set[,1])
  accuracy_list <- c(accuracy_list, accuracy)
}

accuracy_list
mean(accuracy_list)
max(accuracy_list)
