# ============================================================
# 1) Stability selection function (Meinshausen & Bühlmann style)
# ============================================================

stability_selection_MB2010 <- function(
    X, Y,
    k_N, m_N,
    base_selector,          # function(X_sub, Y_sub, ...) -> integer vector in 1:p
    pi_thr = 0.9,
    replace = FALSE,
    seed = NULL,
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
  
  counts <- integer(p)
  selected_list <- vector("list", m_N)
  
  for (b in seq_len(m_N)) {
    idx <- sample.int(N, size = k_N, replace = replace)
    
    sel <- base_selector(X[idx, , drop = FALSE], Y[idx], ...)
    sel <- unique(as.integer(sel))
    sel <- sel[sel >= 1 & sel <= p]
    
    selected_list[[b]] <- sel
    if (length(sel) > 0) counts[sel] <- counts[sel] + 1L
  }
  
  freq <- counts / m_N
  stable_set <- which(freq >= pi_thr)
  
  list(
    N = N, p = p, k_N = k_N, m_N = m_N, pi_thr = pi_thr,
    freq = freq,
    stable_set = stable_set,
    selected_list = selected_list
  )
}

# ============================================================
# 2) Example: simulate linear regression data
# ============================================================

set.seed(1)

N <- 200
p <- 20
s <- 5

X <- matrix(rnorm(N * p), N, p)

beta <- rep(0, p)
beta[1:s] <- 1.0

Y <- as.vector(X %*% beta + rnorm(N, sd = 1.0))

# ============================================================
# 3) Base selector for linear regression:
#    select variables with p-value <= alpha in lm()
# ============================================================

lm_pvalue_selector <- function(X_sub, Y_sub, alpha = 0.05) {
  dat <- data.frame(Y = Y_sub, X_sub)
  fit <- lm(Y ~ ., data = dat)
  sm <- summary(fit)
  
  # p-values for coefficients excluding intercept
  pvals <- sm$coefficients[-1, 4]
  which(pvals <= alpha)
}

# ============================================================
# 4) Run stability selection in your notation (N, k_N, m_N)
# ============================================================

k_N <- floor(N / 2)   # half-subsampling
m_N <- 100            # number of subsamples
pi_thr <- 0.8         # frequency threshold (tune as you like)

res <- stability_selection_MB2010(
  X, Y,
  k_N = k_N,
  m_N = m_N,
  base_selector = lm_pvalue_selector,
  pi_thr = pi_thr,
  seed = 123,
  alpha = 0.05        # forwarded into lm_pvalue_selector(...)
)

cat("True signals:", 1:s, "\n")
cat("Stable set  :", res$stable_set, "\n")
cat("Top 10 freqs:\n")
print(head(sort(res$freq, decreasing = TRUE), 10))