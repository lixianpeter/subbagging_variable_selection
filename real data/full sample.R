setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
library(Matrix)
library(glmnet)
library(speedglm)

# Read data
income_data <- fread("combined PUMAs from 2014-2018 all states.csv", data.table = FALSE)

# Pre process of the data
income_data$income <- as.numeric(income_data$income == ">50K")
income_data$workclass <- relevel(factor(income_data$workclass), ref = "Private")
income_data$sex <- relevel(factor(income_data$sex), ref = "Female")

# Convert remaining character variables to factors
for (j in names(income_data)) {
  if (is.character(income_data[[j]])) {
    income_data[[j]] <- factor(income_data[[j]])
  }
}

y <- income_data$income
X <- sparse.model.matrix(income ~ ., data = income_data)
X <- X[, colnames(X) != "(Intercept)", drop = FALSE]

# Initial unpenalised logistic estimator for adaptive weights
init_fit <- speedglm(
  income ~ .,
  data = income_data,
  family = binomial(link = "logit")
)

beta0 <- coef(init_fit)[-1]
beta0 <- beta0[colnames(X)]
beta0[is.na(beta0)] <- 0

# Scale-adjusted adaptive weights
X_mean <- Matrix::colMeans(X)
X_sd <- sqrt(pmax(Matrix::colMeans(X^2) - X_mean^2, 1e-8))

w <- 1 / (abs(beta0 * X_sd) + 1e-4)^1
w <- pmin(w, 1e5) # avoid 0 penalty

# Adaptive lasso
adlasso <- cv.glmnet(
  x = X,
  y = y,
  family = "binomial",
  alpha = 1,
  penalty.factor = w,
  nfolds = 3
)

# Choose the best lambda 
fit <- adlasso$glmnet.fit
coef_path <- as.matrix(coef(fit))

num_selected <- apply(coef_path[-1, ] != 0, 2, sum)
lambda_use <- fit$lambda[which.min(abs(num_selected - 12))]

# Final adaptive-lasso coefficients
coef_adlasso <- coef(fit, s = lambda_use)

adlasso_result <- data.frame(
  variable = rownames(coef_adlasso),
  coefficient = as.numeric(coef_adlasso)
)


write.csv(adlasso_result, "adaptive_lasso_full_coefficients.csv", row.names = FALSE)
