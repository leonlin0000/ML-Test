library(rpart)
library(rpart.plot)

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args)) args[1] else "~/Downloads/ball.txt"
ball <- read.table(input, header = TRUE, quote = '"', comment.char = "")

print(dim(ball))
print(summary(ball))
str(ball)
print(head(ball, n = 10))

# 預測 Salary。使用 1986 年及生涯資料，排除薪資衍生欄位和球員識別欄位。
# League86 和 Div86 已提供球隊所屬分組，因此不再加入 Team86。
excluded <- c("Id", "Name", "Logsalary", "Team86", "League87", "Team87")
model_data <- ball[, !names(ball) %in% excluded, drop = FALSE]
for (column in names(model_data)) {
  if (is.character(model_data[[column]])) {
    model_data[[column]] <- factor(model_data[[column]])
  }
}

set.seed(123)
# 每個類別至少保留一筆訓練樣本，避免測試時出現模型未見過的類別。
required_rows <- integer()
for (values in Filter(is.factor, model_data)) {
  required_rows <- union(required_rows, vapply(
    split(seq_along(values), values),
    function(rows) rows[sample.int(length(rows), 1L)],
    integer(1)
  ))
}
train_size <- floor(0.8 * nrow(model_data))
if (length(required_rows) > train_size) stop("Too many categories for the training split.")
remaining <- setdiff(seq_len(nrow(model_data)), required_rows)
train_rows <- c(required_rows, remaining[
  sample.int(length(remaining), train_size - length(required_rows))
])
train_data <- droplevels(model_data[train_rows, , drop = FALSE])
test_data <- model_data[-train_rows, , drop = FALSE]
print(c(training = nrow(train_data), testing = nrow(test_data)))

lm_fit <- lm(Salary ~ ., data = train_data)
print(summary(lm_fit))
pred_lm <- predict(lm_fit, newdata = test_data)

tree_fit <- rpart(Salary ~ ., method = "anova", data = train_data)
print(tree_fit$variable.importance)
pred_tree <- predict(tree_fit, newdata = test_data)

print(c(
  linear_regression_mse = mean((pred_lm - test_data$Salary)^2),
  regression_tree_mse = mean((pred_tree - test_data$Salary)^2)
))
print(data.frame(
  Name = ball$Name[-train_rows],
  observed = test_data$Salary,
  linear_regression = pred_lm,
  regression_tree = pred_tree,
  row.names = NULL
))

plot_predictions <- function(predicted, observed, title) {
  limits <- range(predicted, observed)
  plot(predicted, observed, xlim = limits, ylim = limits, asp = 1,
       xlab = "Predicted Salary", ylab = "Observed Salary", main = title)
  abline(0, 1, col = "red", lty = 2)
}

if (!interactive()) pdf("ball_regression_plots.pdf", width = 10, height = 7)
plot_predictions(pred_lm, test_data$Salary, "Multiple Linear Regression")
rpart.plot(tree_fit, extra = 101)
plot_predictions(pred_tree, test_data$Salary, "Regression Tree")
if (!interactive()) invisible(dev.off())
