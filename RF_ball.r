library(randomForest)
library(caTools)
library(ggplot2)
library(gridExtra)

## Load Data and Split into Training and Testing Sets
args <- commandArgs(trailingOnly = TRUE)
input_path <- "/Users/lin/Downloads/ball.txt"
plot_path <- if (length(args) >= 2L) args[2] else "ball_plots.pdf"
ball <- read.table(input_path, header = TRUE, stringsAsFactors = TRUE)
# Drop player identifiers and the untransformed salary to avoid target leakage.
# Pos86 contains a test-only level (CD) under the original split.
ball <- subset(ball, select = -c(Id, Name, Salary, Pos86))
dim(ball)
summary(ball)
str(ball)
head(ball, n = 10)

set.seed(123)
split <- sample.split(ball$Logsalary, SplitRatio = 0.8)
train_data <- subset(ball, split)
test_data <- subset(ball, !split)
dim(train_data)
dim(test_data)

pdf(plot_path, width = 10, height = 6)

## Multiple Linear Regression
########################################################################

lm.fit <- glm(
  Logsalary ~ .,
  family = gaussian(link = "identity"),
  data = train_data
)

summary(lm.fit)

## Prediction
pred.lm.fit <- predict(
  lm.fit,
  newdata = test_data
)

## Mean Squared Error
mse.lm.fit <- mean(
  (pred.lm.fit - test_data$Logsalary)^2
)
mse.lm.fit

## Root Mean Squared Error
rmse.lm.fit <- sqrt(mse.lm.fit)
rmse.lm.fit

## Predicted vs. Observed Values
data.frame(
  pred.Logsalary = pred.lm.fit,
  true.Logsalary = test_data$Logsalary
)

plot_limits <- range(c(ball$Logsalary, pred.lm.fit))
plot(pred.lm.fit, test_data$Logsalary,
     xlim = plot_limits,
     ylim = plot_limits,
     xlab = "Predicted Values",
     ylab = "Observed Values",
     main = "Multiple Linear Regression",
     pch = 19)

abline(a = 0, b = 1, col = "red", lty = 2)


########################################################################
## Random Forest Regression
########################################################################

## Random Forest Regression Model
set.seed(123)

rf.fit <- randomForest(
  Logsalary ~ .,
  data = train_data,
  ntree = 500,
  mtry = 4,
  nodesize = 5,
  importance = TRUE
)

## Model Summary
print(rf.fit)
summary(rf.fit)

## OOB Mean Squared Error
rf.fit$mse

## OOB Error Plot
plot(rf.fit,
     main = "Random Forest Regression: OOB MSE")

## Variable Importance
importance(rf.fit)

varImpPlot(rf.fit,
           main = "Variable Importance: Baseball")

## Prediction
pred.rf.fit <- predict(
  rf.fit,
  newdata = test_data
)

## Mean Squared Error
mse.rf.fit <- mean(
  (pred.rf.fit - test_data$Logsalary)^2
)
mse.rf.fit

## Root Mean Squared Error
rmse.rf.fit <- sqrt(mse.rf.fit)
rmse.rf.fit

## Predicted vs. Observed Values
data.frame(
  pred.Logsalary = pred.rf.fit,
  true.Logsalary = test_data$Logsalary
)

plot_limits <- range(c(plot_limits, pred.rf.fit))
plot(pred.rf.fit, test_data$Logsalary,
     xlim = plot_limits,
     ylim = plot_limits,
     xlab = "Predicted Values",
     ylab = "Observed Values",
     main = "Random Forest Regression",
     pch = 19)

abline(a = 0, b = 1, col = "red", lty = 2)


########################################################################
## Model Comparison
## Multiple Linear Regression vs. Random Forest
########################################################################

## MSE and RMSE Comparison
results <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  MSE = c(mse.lm.fit, mse.rf.fit),
  RMSE = c(rmse.lm.fit, rmse.rf.fit)
)

print(results)

## Visualization of Prediction Results
p1 <- ggplot(
  data.frame(
    Predicted = pred.lm.fit,
    Observed = test_data$Logsalary
  ),
  aes(x = Predicted, y = Observed)
) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1,
              color = "red", linetype = 2) +
  coord_fixed(xlim = plot_limits, ylim = plot_limits) +
  labs(title = "Linear Regression") +
  theme_bw()

p2 <- ggplot(
  data.frame(
    Predicted = pred.rf.fit,
    Observed = test_data$Logsalary
  ),
  aes(x = Predicted, y = Observed)
) +
  geom_point(color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1,
              color = "red", linetype = 2) +
  coord_fixed(xlim = plot_limits, ylim = plot_limits) +
  labs(title = "Random Forest") +
  theme_bw()

grid.arrange(p1, p2, ncol = 2)

########################################################################
## Effect of mtry on Random Forest
########################################################################

mtry.values <- seq_len(ncol(train_data) - 1)
oob.mse <- numeric(length(mtry.values))

for (i in seq_along(mtry.values)) {

  set.seed(123)

  fit <- randomForest(
    Logsalary ~ .,
    data = train_data,
    ntree = 500,
    mtry = mtry.values[i]
  )

  oob.mse[i] <- tail(fit$mse, 1)
}

plot(mtry.values, oob.mse,
     type = "b",
     pch = 19,
     xlab = "Number of Variables (mtry)",
     ylab = "OOB Mean Squared Error",
     main = "Effect of mtry on Random Forest")

print(data.frame(mtry = mtry.values, OOB_MSE = oob.mse))
invisible(dev.off())
