# ============================================================
# Capanu et al. (2023)-style OPT-STABS simulation driver
# Rewritten with subbagging notation: N, k_N, m_N
# ============================================================

suppressPackageStartupMessages({
  library(MASS)     # mvrnorm
  library(STOPES)   # opts
})

# ----- helper: Toeplitz correlation -----
toeplitz_cor <- function(p, rho) {
  mat <- matrix(0, p, p)
  for (i in 1:p) for (j in 1:p) mat[i, j] <- rho^(abs(i - j))
  mat
}

# ----- helper: generate logistic data -----
gen_logit_data <- function(N, p, s, beta_signal = 1.0, rho = 0.5, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  Sigma <- toeplitz_cor(p, rho)
  X <- MASS::mvrnorm(n = N, mu = rep(0, p), Sigma = Sigma)
  beta <- rep(0, p)
  beta[1:s] <- beta_signal
  eta <- drop(X %*% beta)
  pr <- 1 / (1 + exp(-eta))
  Y <- rbinom(N, size = 1, prob = pr)
  list(X = X, Y = Y, beta = beta)
}

# ----- helper: evaluate variable selection -----
eval_selection <- function(selected, s, p) {
  selected <- sort(unique(selected))
  tp <- sum(selected %in% 1:s)
  fp <- sum(selected %in% (s + 1):p)
  tpr <- tp / s
  exact <- as.integer(tp == s && fp == 0)
  list(tp = tp, fp = fp, tpr = tpr, exact = exact)
}

# ----- one replicate for one scenario -----
# N   = full sample size
# k_N = subsample size
# m_N = number of subsamples
one_run_opts_subbagging <- function(N, p, s, beta_signal, rho,
                                    k_N, m_N,
                                    cutoff = 0.75,
                                    crit = "aic",
                                    family = binomial(),
                                    seed = NULL) {
  if (k_N <= 1 || k_N > N) stop("k_N must satisfy 1 < k_N <= N")
  if (m_N < 1) stop("m_N must be >= 1")
  
  dat <- gen_logit_data(N = N, p = p, s = s, beta_signal = beta_signal, rho = rho, seed = seed)
  
  # STOPES::opts uses prop_split = subsample fraction
  prop_split <- k_N / N
  
  fit <- STOPES::opts(
    X = dat$X, Y = dat$Y,
    m = m_N,
    crit = crit,
    prop_split = prop_split,
    cutoff = cutoff,
    family = family
  )
  
  # fit$Jhat: selected variable indices in {1, ..., p}
  eval_selection(fit$Jhat, s = s, p = p)
}

# ----- simulation loop -----
run_sim_opts_subbagging <- function(R = 1000,
                                    N = 200, p = 25, s = 5,
                                    beta_signal = 1.0,
                                    rho = 0.5,
                                    k_N = 100,
                                    m_N = 100,
                                    cutoff = 0.75,
                                    crit = "aic",
                                    seed0 = 1) {
  out <- data.frame(tp = integer(R), fp = integer(R), tpr = numeric(R), exact = integer(R))
  
  for (r in 1:R) {
    res <- one_run_opts_subbagging(
      N = N, p = p, s = s,
      beta_signal = beta_signal, rho = rho,
      k_N = k_N, m_N = m_N,
      cutoff = cutoff, crit = crit,
      seed = seed0 + r
    )
    out[r, ] <- c(res$tp, res$fp, res$tpr, res$exact)
  }
  
  list(
    settings = list(
      R = R, N = N, p = p, s = s,
      beta_signal = beta_signal, rho = rho,
      k_N = k_N, m_N = m_N,
      cutoff = cutoff, crit = crit
    ),
    summary = list(
      avg_fp = mean(out$fp),
      avg_tpr = mean(out$tpr),
      pr_exact = mean(out$exact)
    ),
    raw = out
  )
}

# ----- example run -----
# Example: N=200, k_N=100 (half-subsamples), m_N=100 subsamples, 200 replicates
res <- run_sim_opts_subbagging(
  R = 200,
  N = 200, p = 25, s = 5,
  beta_signal = 1.0, rho = 0.5,
  k_N = 100, m_N = 100,
  cutoff = 0.75,
  crit = "aic",
  seed0 = 123
)

print(res$settings)
print(res$summary)