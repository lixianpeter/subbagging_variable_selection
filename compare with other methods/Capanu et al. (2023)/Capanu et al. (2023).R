# ============================================================
# Capanu et al. (2023) using STOPES
# Paper version: opts(...)
# ============================================================

# install.packages(c("STOPES", "MASS"))
suppressPackageStartupMessages({
  library(STOPES)
  library(MASS)
})


# ----------------------------
# User inputs
# ----------------------------
# In STOPES::opts():
# - m = number of subsamples
# - prop_split = subsample fraction
# so k_N = N * prop_split
m_N <- 20
k_N <- floor(N / 2)
prop_split <- k_N / N

# The following is from paper's simulation settings
crit <- "aic"
cutoff <- 0.75

# ----------------------------
# Toy example of logistic regression data
# ----------------------------
N <- 200
p <- 50
s <- 8

Sigma <- diag(p)
X <- mvrnorm(N, mu = rep(0, p), Sigma = Sigma)

beta <- rep(0, p)
beta[1:s] <- 1

eta <- drop(X %*% beta)
prob <- 1 / (1 + exp(-eta))
Y <- rbinom(N, size = 1, prob = prob)



# ----------------------------
# Run Capanu et al. (2023) OPTS
# ----------------------------
res_OPTS <- opts(
  X = X,
  Y = Y,
  m = m_N,
  crit = crit,
  prop_split = prop_split,
  cutoff = cutoff,
  family = "binomial"
)

cat("True signals:\n")
print(1:s)

cat("Selected variables (Capanu et al. 2023, STOPES::opts):\n")
print(which(res_OPTS$Jhat))

cat("Selection frequencies:\n")
print(head(sort(res_OPTS$freqs, decreasing = TRUE), 15))




# ----------------------------
# Toy example of linear regression data
# ----------------------------
N <- 200
p <- 50
s <- 8

Sigma <- diag(p)
X <- mvrnorm(N, mu = rep(0, p), Sigma = Sigma)

beta <- rep(0, p)
beta[1:s] <- 1

Y <- drop(X %*% beta + rnorm(N, sd = 1))

# ----------------------------
# Run Capanu et al. (2023) OPTS
# ----------------------------
res_OPTS <- opts(
  X = X,
  Y = Y,
  m = m_N,
  crit = crit,
  prop_split = prop_split,
  cutoff = cutoff,
  family = "gaussian"
)

cat("True signals:\n")
print(1:s)

cat("Selected variables (Capanu et al. 2023, STOPES::opts):\n")
print(which(res_OPTS$Jhat))

cat("Selection frequencies:\n")
print(head(sort(res_OPTS$freqs, decreasing = TRUE), 15))

