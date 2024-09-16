
# MLP with Categorical(qualititive) variables
# Dataset: bank.csv, 7 variables, 700 obstacle
# Target variable: default

data_set <- read.csv("bank.csv")
View(data_set)

set.seed(123)
n <- sample(700, 500)
train_set <- data_set[n,] #500
test_set <- data_set[-n,] #remining 200

require(neuralnet)
require(caret)
model_formula <- default ~ age + educ + employ + income + creddebt + othdebt
nn_model <- neuralnet(
  formula = model_formula,
  data = train_set,
  algorithm = "rprop+",   #resiliant backpropagation with weights backtracking (without is rprop-)
  hidden = 10,            #single hidden layer with 5 nodes(neurons)
  err.fct = "sse",        #sum of square error
  act.fct = "logistic",   #acrivation function
  rep = 1,                #the algorithm repeats only once
  stepmax = 1e06,         #max iteration
  threshold = 0.01,       #if error value goes under threshold the algorithm stops
  linear.output = FALSE   #must be false because variables are categorical
)

plot(nn_model)
plot(nn_model, show.weights = FALSE)
nn_model$result.matrix
nn_model$weights

predictions <- compute(nn_model, test_set[,-7])
                          #select all columns except the 7th column
predicted_probabilities <- predictions$net.result
head(predicted_probabilities, 10)

predicted_classes <- ifelse(predicted_probabilities < 0.5, 0, 1)
                          #if the value of predicted_probabilities is less than 0.5 then it is 0, else 1
head(predicted_classes, 10)

confusion_matrix <- table(predicted_classes, test_set$default)
confusion_matrix

accuracy <- mean(predicted_classes == test_set$default)
print(paste("Accuracy:", accuracy))

recall <- sensitivity(confusion_matrix)
print(paste("Recall:", recall))

precision <- posPredValue(confusion_matrix) #precision <- precision(confusion_matrix)
print(paste("Precision:", precision))       #print(paste("Precision:", precision))

specificity <- specificity(confusion_matrix)
print(paste("Specificity:", specificity))

require(ROCR)
prediction <- prediction(predicted_probabilities, test_set$default)
performance <- performance(prediction, x.measure = "fpr", measure = "tpr")
plot(performance, col = "purple", main = "ROC Curve (ROCR)", xlab = "False Positive Rate", ylab = "True Positive Rate")

auc <- performance(prediction, measure = "auc")
View(auc)
auc@y.name
auc@y.values

#require(pROC)
#predicted_probabilities <- as.numeric(predicted_probabilities)
#roc_curve <- roc(test_set$default, predicted_probabilities)
#dev.off()
#plot(roc_curve, col = "purple", main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")
#auc_value <- auc(roc_curve)
#print(paste("AUC:", auc_value))


#---


accuracy_list <- NULL

for (i in 1:10) {
  print(i)
  net <- neuralnet(model_formula, train_set, algorithm = "rprop+", hidden = i,
                   err.fct = "sse", act.fct = "logistic", rep = 1, stepmax = 1e06,
                   threshold = 0.01, linear.output = FALSE)
  pred <- compute(net, test_set[,-7])
  pred_prob <- pred$net.result
  pred_categ <- ifelse(pred_prob<0.5, 0, 1)
  acc <- mean(pred_categ == test_set$default)
  accuracy_list <- c(accuracy_list, acc)
}

accuracy_list
plot(accuracy_list)
which.max(accuracy_list)
max(accuracy_list)


#---


n_folds <- 5                                #K-fold cross-validation
folds <- createFolds(data_set$income, k = n_folds, list = FALSE)
class(folds)
View(folds)
table(folds)

accuracy_list <- NULL
for (i in 1:n_folds) {
  print(i)
  indextest <- which(folds==i)
  test_set <- data_set[indextest,]          #fold number i
  train_set <- data_set[-indextest,]        #other folds
  net <- neuralnet(model_formula, train_set, algorithm = "rprop+", hidden = 5,
                   err.fct = "sse", act.fct = "logistic", rep = 1, stepmax = 1e06,
                   threshold = 0.01, linear.output = FALSE)
  pred <- compute(net, test_set[,-7])
  pred_prob <- pred$net.result
  pred_categ <- ifelse(pred_prob<0.5, 0, 1)
  acc <- mean(pred_categ == test_set$default)
  accuracy_list <- c(accuracy_list, acc)
}

accuracy_list
mean(accuracy_list)
max(accuracy_list)

