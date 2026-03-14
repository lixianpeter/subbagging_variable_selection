# ============================================================
# Meinshausen & Buehlmann (2010) using stabs
# Paper version: original stability selection
# ============================================================

# install.packages(c("stabs", "glmnet"))
suppressPackageStartupMessages({
  library(stabs)
  library(glmnet)
})

set.seed(1)

# ----------------------------
# User inputs
# ----------------------------
# In stabs::stabsel():
# - sampling.type = "MB" means Meinshausen & Buehlmann (2010)
# - the package uses B = number of subsampling replicates
# Here we write m_N instead.
#
# In stabs, subsample size is effectively fixed at about N / 2.
# So k_N is not a free tuning parameter here.
#
# IMPORTANT:
# In stabs, only two of cutoff, q, and PFER can be specified.
# Here we specify cutoff and q, and let stabs compute the implied PFER.
m_N <- 100
q <- 8
cutoff <- 0.75

# ----------------------------
# Toy example of logistic regression data
# ----------------------------
N <- 200
p <- 50
s <- 8

X <- matrix(rnorm(N * p), N, p)

beta <- rep(0, p)
beta[1:s] <- 1

eta <- drop(X %*% beta)
prob <- 1 / (1 + exp(-eta))
Y <- rbinom(N, size = 1, prob = prob)

# ----------------------------
# Optional: inspect parameter combination
# ----------------------------
stabsel_parameters(
  p = p,
  q = q,
  cutoff = cutoff,
  sampling.type = "MB",
  B = m_N
)

# ----------------------------
# Run Meinshausen & Buehlmann (2010): logistic version
# ----------------------------
res_logistic <- stabsel(
  x = X,
  y = Y,
  fitfun = glmnet.lasso,
  cutoff = cutoff,
  q = q,
  sampling.type = "MB",
  B = m_N,
  papply = lapply
)

cat("True signals (logistic):\n")
print(1:s)

cat("Selected variables (Meinshausen & Buehlmann 2010, stabs::stabsel, logistic):\n")
print(selected(res_logistic))

cat("Selection probabilities (logistic):\n")
print(head(sort(res_logistic$phat, decreasing = TRUE), 15))


# ----------------------------
# Toy example of linear regression data
# ----------------------------
N <- 200
p <- 50
s <- 8

X <- matrix(rnorm(N * p), N, p)

beta <- rep(0, p)
beta[1:s] <- 1

Y <- drop(X %*% beta + rnorm(N, sd = 1))

# ----------------------------
# Run Meinshausen & Buehlmann (2010): linear version
# ----------------------------
res_linear <- stabsel(
  x = X,
  y = Y,
  fitfun = glmnet.lasso,
  cutoff = cutoff,
  q = q,
  sampling.type = "MB",
  B = m_N,
  papply = lapply
)

cat("True signals (linear):\n")
print(1:s)

cat("Selected variables (Meinshausen & Buehlmann 2010, stabs::stabsel, linear):\n")
print(selected(res_linear))

cat("Selection probabilities (linear):\n")
print(head(sort(res_linear$phat, decreasing = TRUE), 15))

