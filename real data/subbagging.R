# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Packages
library(data.table)
library(Matrix)
library(glmnet)


# Read and preprocess data


income_data <- fread(
  "combined PUMAs from 2014-2018 all states.csv",
  data.table = FALSE
)

income_data$income <- as.numeric(income_data$income == ">50K")
income_data$workclass <- relevel(factor(income_data$workclass), ref = "Private")
income_data$sex <- relevel(factor(income_data$sex), ref = "Female")

for (j in names(income_data)) {
  if (is.character(income_data[[j]])) {
    income_data[[j]] <- factor(income_data[[j]])
  }
}

y <- income_data$income

X <- sparse.model.matrix(income ~ ., data = income_data)
X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
X <- as.matrix(X)

N <- nrow(X)
p <- ncol(X)


# Subbagging logistic LSA adaptive LASSO


subbag_logistic_lsa_glmnet <- function(
    X, y,
    k_N,
    m_N,
    eps_weight = 1e-6,
    nlambda = 100,
    seed = 123
) {
  set.seed(seed)
  
  X <- as.matrix(X)
  y <- as.numeric(y)
  
  N <- nrow(X)
  p <- ncol(X)
  
  
  var_names <- colnames(X)
  if (is.null(var_names)) {
    var_names <- paste0("V", seq_len(p))
  }
  
  beta_subsample_list <- vector("list", m_N)
  second_derivative_subsample_list <- vector("list", m_N)
  
  t0 <- Sys.time()
  
  
  # Fit logistic regression on each subsample
  
  
  for (i in seq_len(m_N)) {
    subsample_idx <- sample(N, size = k_N, replace = FALSE)
    
    y_subsample <- y[subsample_idx]
    x_subsample <- X[subsample_idx, , drop = FALSE]
    
    fit <- tryCatch(
      # Some subsample may be incomplete skip
      glm.fit(
        x = x_subsample,
        y = y_subsample,
        family = binomial(link = "logit"),
        intercept = FALSE,
        control = glm.control(maxit = 50)
      ),
      error = function(e) NULL
    )
    
    if (is.null(fit)) {
      beta_subsample <- rep(0, p)
    } else {
      beta_subsample <- as.numeric(coef(fit))
      beta_subsample[!is.finite(beta_subsample)] <- 0
    }
    
    prob <- plogis(as.numeric(x_subsample %*% beta_subsample))
    
    w <- pmax(prob * (1 - prob), 1e-8)
    
    second_derivative_subsample <- crossprod(x_subsample * sqrt(w)) / k_N
    
    beta_subsample_list[[i]] <- beta_subsample
    second_derivative_subsample_list[[i]] <- second_derivative_subsample
    
    cat("Finished subsample", i, "of", m_N, "\n")
  }
  
  
  # Initial estimator: average of beta_subsample
  
  
  beta_average <- Reduce("+", beta_subsample_list) / m_N
  
  

  # LSA(beta) =
  # (1 / m_N) sum_s (beta - beta_s)' H_s (beta - beta_s)
  
  
  glm_X <- NULL
  glm_Y <- NULL
  
  for (k in seq_len(m_N)) {
    eig <- eigen(second_derivative_subsample_list[[k]], symmetric = TRUE)
    
    eig_values <- pmax(eig$values, 0)
    
    Vk_half <- eig$vectors %*%
      diag(sqrt(eig_values), nrow = p, ncol = p) %*%
      t(eig$vectors)
    
    glm_Y <- rbind(glm_Y, Vk_half %*% beta_subsample_list[[k]])
    glm_X <- rbind(glm_X, Vk_half)
  }
  
  glm_X <- glm_X / sqrt(m_N)
  glm_Y <- glm_Y / sqrt(m_N)
  
  glm_Y_vec <- as.numeric(glm_Y)
  
  cy <- sd(glm_Y_vec) * sqrt(length(glm_Y_vec) - 1) / sqrt(length(glm_Y_vec))
  
  if (!is.finite(cy) || cy <= 0) {
    cy <- 1
  }
  
  
  # Adaptive LASSO penalty factor
  
  
  penalty_factor <- 1 / (
    pmax(abs(beta_average), eps_weight) /
      cy^2 / 2 / length(glm_Y_vec)
  )
  
  
  # Lambda grid
  
  
  gridLambda <- 10^seq(
    0,
    log10(log(N) / N * 1e-1),
    length = nlambda
  )
  
  
  # Fit adaptive LASSO using glmnet
  
  
  glmnet_fit <- glmnet(
    x = glm_X / cy,
    y = glm_Y_vec / cy,
    family = "gaussian",
    alpha = 1,
    standardize = FALSE,
    intercept = FALSE,
    penalty.factor = penalty_factor,
    lambda = gridLambda
  )
  
  beta_path <- as.matrix(coef(glmnet_fit))[-1, , drop = FALSE]
  
  
  # Select lambda by actual LSA SBIC
  # SBIC(lambda) = k_N * LSA(beta_lambda) + df * log(N)
  
  
  BIC_vec <- apply(beta_path, 2, FUN = function(beta_lambda) {
    df <- sum(beta_lambda != 0)
    
    LSA <- 0
    
    for (k in seq_len(m_N)) {
      diff <- beta_lambda - beta_subsample_list[[k]]
      
      LSA <- LSA +
        as.numeric(
          t(diff) %*%
            second_derivative_subsample_list[[k]] %*%
            diff
        )
    }
    
    LSA <- LSA / m_N
    
    k_N * LSA + df * log(N)
  })
  
  best <- which.min(BIC_vec)
  
  beta_hat <- beta_path[, best]
  names(beta_hat) <- var_names
  rownames(beta_path) <- var_names
  
  selected <- which(beta_hat != 0)
  
  time <- as.numeric(Sys.time() - t0, units = "secs")
  
  
  # Output
  list(
    beta_hat = beta_hat,
    selected = selected,
    selected_names = var_names[selected],
    lambda_best = glmnet_fit$lambda[best],
    BIC_min = min(BIC_vec),
    beta_path = beta_path,
    lambda_path = glmnet_fit$lambda,
    BIC_vec = BIC_vec,
    beta_average = beta_average,
    beta_subsample_list = beta_subsample_list,
    second_derivative_subsample_list = second_derivative_subsample_list,
    k_N = k_N,
    m_N = m_N,
    time = time
  )
}



# Set subbagging parameters
# k_N <- floor(N^(1 / 2 + delta))
# m_N <- floor(alpha_subbag * N / k_N)
k_N <- 252569
m_N <- 6

cat("Subsample size k_N:", k_N, "\n")
cat("Number of subsamples m_N:", m_N, "\n")



# Run method
fit <- subbag_logistic_lsa_glmnet(
  X = X,
  y = y,
  k_N = k_N,
  m_N = m_N,
  eps_weight = 1e-6,
  nlambda = 100,
  seed = 123
)


# Save outputs
coef_out <- data.frame(
  variable = names(fit$beta_hat),
  beta_hat = as.numeric(fit$beta_hat),
  selected = names(fit$beta_hat) %in% fit$selected_names
)

path_out <- data.frame(
  lambda = fit$lambda_path,
  BIC = fit$BIC_vec,
  df = apply(fit$beta_path, 2, function(x) sum(x != 0))
)


write.csv(coef_out, "subbagging_coefficients.csv", row.names = FALSE)
write.csv(path_out, "SBIC_lambda.csv", row.names = FALSE)
