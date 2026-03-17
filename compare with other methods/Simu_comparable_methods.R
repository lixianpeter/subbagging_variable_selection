Simu <- function(a) {
  # install.packages(c("stabs", ""STOPES""))
  library(stabs)
  library(STOPES)
  library(MASS)
  library(glmnet)
  ## Input Parameters
  id <- a
  list <- read.csv("Paralist.csv", stringsAsFactors = FALSE)
  
  para <- list[id, ]
  
  ### ---------------------------- --
  #  For test only -------
  ### ---------------------------- --
  # Comment out when not testing
  # para = list()
  # para$model = "Linear"
  # para$p = 20
  # para$N = 100000
  # para$delta = 1/4
  # para$alpha = 1
  # para$SNR = 0.5
  # para$LoopStart = 1
  # para$LoopEnd = 50
  # loop = 1
  ### ---------------------------- --
  
  
  model <- as.character(para$model)
  p <- para$p
  N <- para$N
  delta <- para$delta
  alpha <- para$alpha
  SNR <- para$SNR
  p0 <- 12
  
  LoopStart <- para$LoopStart
  LoopEnd <- para$LoopEnd
  
  rm(list, para)
  
  if(model == 'Linear'){
    filename1 <- file.path("result", paste0(model, "_Capanu_estimate_SNR", SNR, "_p", p, "_N", sprintf("%.0f", N), "_delta", sprintf("%.2f", delta), "_alpha", alpha, ".csv"))
    filename2 <- file.path("result", paste0(model, "_Capanu_SNR", SNR, "_p", p, "_N", sprintf("%.0f", N), "_delta", sprintf("%.2f", delta), "_alpha", alpha, ".csv"))
    filename3 <- file.path("result", paste0(model, "_SS_SNR", SNR, "_p", p, "_N", sprintf("%.0f", N), "_delta", sprintf("%.2f", delta), "_alpha", alpha, ".csv"))
    filename4 <- file.path("result", paste0(model, "_MB_SNR", SNR, "_p", p, "_N", sprintf("%.0f", N), "_delta", sprintf("%.2f", delta), "_alpha", alpha, ".csv"))
  }else{
    filename1 <- file.path("result", paste0(model, "_Capanu_estimate_p", p, "_N", sprintf("%.0f", N), "_delta", sprintf("%.2f", delta), "_alpha", alpha, ".csv"))
    filename2 <- file.path("result", paste0(model, "_Capanu_p", p, "_N", sprintf("%.0f", N), "_delta", sprintf("%.2f", delta), "_alpha", alpha, ".csv"))
    filename3 <- file.path("result", paste0(model, "_SS_p", p, "_N", sprintf("%.0f", N), "_delta", sprintf("%.2f", delta), "_alpha", alpha, ".csv"))
    filename4 <- file.path("result", paste0(model, "_MB_p", p, "_N", sprintf("%.0f", N), "_delta", sprintf("%.2f", delta), "_alpha", alpha, ".csv"))
  }
  beta_true <- c(seq(-1, -0.5, length.out = p0/2), seq(0.5, 1, length.out = p0/2), rep(0, p - p0))
  
  for (loop in LoopStart:LoopEnd) {
    set.seed(loop)
    
    rho <- 0.5
    idx <- 0:(p - 1)
    
    Sigma <- rho ^ abs(outer(idx, idx, "-"))
    
    ## Generate correlated covariates
    x <- mvrnorm(
      n = N,
      mu = rep(0, p),
      Sigma = Sigma
    )   # N x p
    
    e_sigma <- t(beta_true) %*% Sigma %*% beta_true / SNR
      
    if (model == "Linear") {
      
      # code for Linear model
      error <- rnorm(n = N, mean = 0, sd = sqrt(e_sigma))
      y <- x %*% beta_true + error
      
    } else {
      
      # code for non-Linear model
      y <- rbinom(n = N, 1, plogis(x %*% beta_true))
    }
    
    # The following settings used in our method
    kn <- floor(N^(1/2+delta)) 
    mn <- floor(alpha*N/kn)
    
   
    
    ### ---------------------------- --
    #  Capanu et al. (2023) OPTS -------
    ### ---------------------------- --
    # In STOPES::opts():
    # - m = number of subsamples
    # - prop_split = subsample fraction
    # so k_N = N * prop_split
    k_N <- floor(N / 2)
    prop_split <- k_N / N
    # The following is from paper's simulation settings
    crit <- "aic"
    cutoff <- 0.75
    
    t0 <- Sys.time() 
    res_Capanu <- opts(
      X = x,
      Y = y,
      m = mn,
      crit = crit,
      prop_split = prop_split,
      cutoff = cutoff,
      family = if (model == "Linear") "gaussian" else "binomial"
    )
    time <- as.numeric(Sys.time() - t0, units = "secs")
    CI_indicator <- as.numeric((res_Capanu$betahat[1:p0] + qnorm(0.975) * res_Capanu$SE[1:p0] > beta_true[1:p0]) & (res_Capanu$betahat[1:p0] - qnorm(0.975) * res_Capanu$SE[1:p0] < beta_true[1:p0]))
    result <- c(loop,  res_Capanu$betahat, res_Capanu$SE, CI_indicator, time)
    write.table(t(result), filename1, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
    result <- c(loop,  res_Capanu$Jhat, time)
    write.table(t(result), filename2, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
    
    
    ### ---------------------------- --
    #  Shah & Samworth (2013) -------
    ### ---------------------------- --
    # In stabs::stabsel():
    # - sampling.type = "SS" means Shah & Samworth (2013)
    # - the package uses B = number of complementary pairs
    # Here we write m_N instead.
    # So m_N is not exactly the total number of fitted subsamples;
    # total number of fitted subsamples is 2 * m_N.
    #
    q <- as.integer(1.25*p0)
    cutoff <- max(0.75, q/p+0.1) # in most cases, it is just 0.75; this is needed to avoid error when p is relatively small
    
    t0 <- Sys.time()     
    res_SS <- stabsel(
      x = x,
      y = y,
      fitfun = glmnet.lasso,
      args.fitfun = list(family = if (model == "Linear") "gaussian" else "binomial"),
      cutoff = cutoff,
      q = q,
      sampling.type = "SS",
      B = as.integer(mn/2),
      papply = lapply
    )
    time <- as.numeric(Sys.time() - t0, units = "secs")
    selected_variable <- rep(0,p)
    selected_variable[res_SS$selected] <- 1
    result <- c(loop,  selected_variable, time)
    write.table(t(result), filename3, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
    
    ### ---------------------------- --
    # Meinshausen & Buehlmann (2010) -------
    ### ---------------------------- --
    # In stabs::stabsel():
    # - sampling.type = "SS" means Shah & Samworth (2013)
    # - the package uses B = number of complementary pairs
    # Here we write m_N instead.
    # So m_N is not exactly the total number of fitted subsamples;
    # total number of fitted subsamples is 2 * m_N.
    #
    q <- as.integer(1.25*p0)
    cutoff <- max(0.75, q/p+0.1) # in most cases, it is just 0.75; this is needed to avoid error when p is relatively small
    
    t0 <- Sys.time()  
    res_MB <- stabsel(
      x = x,
      y = y,
      fitfun = glmnet.lasso,
      args.fitfun = list(family = if (model == "Linear") "gaussian" else "binomial"),
      cutoff = cutoff,
      q = q,
      sampling.type = "MB",
      B = mn,
      papply = lapply
    )
    time <- as.numeric(Sys.time() - t0, units = "secs")
    selected_variable <- rep(0,p)
    selected_variable[res_SS$selected] <- 1
    result <- c(loop,  selected_variable, time)
    write.table(t(result), filename4, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
    

  }
}
