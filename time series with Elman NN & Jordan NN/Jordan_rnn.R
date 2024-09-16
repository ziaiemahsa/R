
# Jordan (recurrent) neural networks
# Dataset: temperatures.csv, 730 obstacles

data_set <- read.csv("temperatures.csv")
View(data_set)

require(RSNNS)
require(quantmod)
require(Metrics)
require(MLmetrics)
require(zoo)

#dependent variable target
target <- as.ts(data_set$Temp)
View(target)
class(target)

target <- log(target)
target <- as.zoo(target)
class(target)

#independent variables x1 to x30
x1 <- Lag(target, k = 1)
x2 <- Lag(target, k = 2)
x3 <- Lag(target, k = 3)
x4 <- Lag(target, k = 4)
x5 <- Lag(target, k = 5)
x6 <- Lag(target, k = 6)
x7 <- Lag(target, k = 7)
x8 <- Lag(target, k = 8)
x9 <- Lag(target, k = 9)
x10 <- Lag(target, k = 10)
x11 <- Lag(target, k = 11)
x12 <- Lag(target, k = 12)
x13 <- Lag(target, k = 13)
x14 <- Lag(target, k = 14)
x15 <- Lag(target, k = 15)
x16 <- Lag(target, k = 16)
x17 <- Lag(target, k = 17)
x18 <- Lag(target, k = 18)
x19 <- Lag(target, k = 19)
x20 <- Lag(target, k = 20)
x21 <- Lag(target, k = 21)
x22 <- Lag(target, k = 22)
x23 <- Lag(target, k = 23)
x24 <- Lag(target, k = 24)
x25 <- Lag(target, k = 25)
x26 <- Lag(target, k = 26)
x27 <- Lag(target, k = 27)
x28 <- Lag(target, k = 28)
x29 <- Lag(target, k = 29)
x30 <- Lag(target, k = 30)
View(x1)
View(x8)
View(x30)

data <- cbind(target, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
              x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
              x21, x22, x23, x24, x25, x26, x27, x28, x29, x30)
View(data)

data <- data[-(1:30),]      #remove the empty cells (the first 30 rows)
View(data)

splitter <- splitForTrainingAndTest(data[,2:31], data[,1], ratio = 0.05)
View(splitter)

set.seed(123)
net_jordan <- jordan(splitter$inputsTrain, splitter$targetsTrain,
                   size=c(60), maxit = 10000, learnFuncParams=c(0.1),
                   inputsTest = splitter$inputsTest,
                   targetsTest = splitter$targetsTest)
plotIterativeError(net_jordan)

predictions_test <- predict(net_jordan, splitter$inputsTest)
rsquared <- cor(splitter$targetsTest, predictions_test)^2
rsquared

plot(splitter$targetsTest, type = "l")
lines(predictions_test, type = "l", col = "red")

temp_test_initial <- as.vector(exp(data[666:700, 1]))
prediction_antilog <- as.vector(exp(predictions_test))

mase(temp_test_initial, prediction_antilog)
mape(temp_test_initial, prediction_antilog)

