# ============================================================
# Shah & Samworth (2013) Complementary Pairs Stability Selection (CPSS)
# Notation: N (full sample size), k_N (subsample size), m_N (number of complementary pairs)
# You provide the base selector as a function.
#
# References:
# - Shah, R.D. & Samworth, R.J. (2013), JRSSB 75(1):55–80 (CPSS definition + bounds)
# - stabs package documents CPSS as sampling.type="SS"
# ============================================================

cpss_shah_samworth <- function(
    X, Y,
    k_N = floor(nrow(as.matrix(X)) / 2),
    m_N = 50,
    base_selector,          # function(X_sub, Y_sub, ...) -> integer vector of selected variable indices in 1:p
    tau = 0.9,              # CPSS frequency threshold (called τ in the paper)
    seed = NULL,
    return_frequencies = TRUE,
    ...
) {
  if (!is.null(seed)) set.seed(seed)
  
  X <- as.matrix(X)
  N <- nrow(X)
  p <- ncol(X)
  if (length(Y) != N) stop("Y length must equal nrow(X).")
  if (k_N <= 1 || k_N > N) stop("k_N must satisfy 1 < k_N <= N.")
  if (m_N < 1) stop("m_N must be >= 1.")
  if (!is.function(base_selector)) stop("base_selector must be a function.")
  if (tau <= 0 || tau >= 1) stop("tau must be in (0,1).")
  
  # CPSS uses complementary pairs (A_{2j-1}, A_{2j}) disjoint, each of size k_N
  # If k_N = floor(N/2), then A_{2j} is naturally the complement of A_{2j-1} (up to 1 obs if N odd).
  # If k_N is set differently, we still enforce disjointness by sampling 2*k_N indices and splitting.
  counts <- integer(p)           # counts over 2*m_N subsamples
  sel_sizes <- integer(2 * m_N)  # for q-hat
  
  selected_sets <- vector("list", 2 * m_N)
  
  for (j in seq_len(m_N)) {
    if (2 * k_N <= N) {
      pool <- sample.int(N, size = 2 * k_N, replace = FALSE)
      A1 <- pool[1:k_N]
      A2 <- pool[(k_N + 1):(2 * k_N)]
    } else {
      # fallback: if 2*k_N > N, can't make disjoint pairs; stop (CPSS requires disjoint halves)
      stop("Need 2*k_N <= N to form complementary (disjoint) pairs. Use k_N <= floor(N/2).")
    }
    
    # run base selector on A1
    sel1 <- unique(as.integer(base_selector(X[A1, , drop = FALSE], Y[A1], ...)))
    sel1 <- sel1[sel1 >= 1 & sel1 <= p]
    selected_sets[[2*j - 1]] <- sel1
    sel_sizes[2*j - 1] <- length(sel1)
    if (length(sel1) > 0) counts[sel1] <- counts[sel1] + 1L
    
    # run base selector on A2 (complementary subsample)
    sel2 <- unique(as.integer(base_selector(X[A2, , drop = FALSE], Y[A2], ...)))
    sel2 <- sel2[sel2 >= 1 & sel2 <= p]
    selected_sets[[2*j]] <- sel2
    sel_sizes[2*j] <- length(sel2)
    if (length(sel2) > 0) counts[sel2] <- counts[sel2] + 1L
  }
  
  freq <- counts / (2 * m_N)
  stable_set <- which(freq >= tau)
  
  # q-hat = average number selected per (half) subsample
  q_hat <- mean(sel_sizes)
  
  # “MB-style” worst-case bound that they recover (under exchangeability etc.)
  # E(false selections) <= q^2 / ((2*tau - 1) * p) for tau > 1/2
  mb_bound <- NA_real_
  if (tau > 0.5) mb_bound <- (q_hat^2) / ((2 * tau - 1) * p)
  
  out <- list(
    N = N, p = p, k_N = k_N, m_N = m_N,
    tau = tau,
    stable_set = stable_set,
    q_hat = q_hat,
    mb_bound = mb_bound,
    selected_sets = selected_sets
  )
  if (return_frequencies) out$freq <- freq
  out
}

# ============================================================
# Example (copy/paste): linear regression + lasso base selector (glmnet)
# ============================================================

# install.packages("glmnet")  # uncomment if needed
suppressPackageStartupMessages(library(glmnet))

set.seed(1)
N <- 200
p <- 60
s <- 8

X <- matrix(rnorm(N * p), N, p)
beta <- rep(0, p); beta[1:s] <- 1.0
Y <- as.vector(X %*% beta + rnorm(N))

# Base selector: lasso at a fixed lambda choice rule inside each subsample.
# Here: choose lambda.1se from CV within the subsample (simple but a bit slower).
lasso_cv_selector <- function(X_sub, Y_sub, nfolds = 5) {
  cvfit <- cv.glmnet(X_sub, Y_sub, family = "gaussian", alpha = 1, nfolds = nfolds, standardize = TRUE)
  b <- as.matrix(coef(cvfit, s = "lambda.1se"))[-1, 1]  # drop intercept
  which(b != 0)
}

# Run CPSS
k_N <- floor(N / 2)
m_N <- 50
tau <- 0.9

res <- cpss_shah_samworth(
  X, Y,
  k_N = k_N,
  m_N = m_N,
  base_selector = lasso_cv_selector,
  tau = tau,
  seed = 123,
  nfolds = 5
)

cat("True signals:", 1:s, "\n")
cat("CPSS stable set:", res$stable_set, "\n")
cat("q_hat:", res$q_hat, "\n")
cat("MB-style bound (only meaningful for tau>0.5):", res$mb_bound, "\n")

cat("Top 15 frequencies:\n")
print(head(sort(res$freq, decreasing = TRUE), 15))