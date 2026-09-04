# Instrumented copy of Code_choice_set_6.R. Same model specifications and
# control settings as the original; the only additions are timing around each
# phase and a PNG device so the plots do not need an interactive display.

.libPaths(Sys.getenv("R_LIB", unset = "../../rlib"))
suppressMessages({library(ergm); library(statnet.common)})

# Resolved relative to the working directory the script is launched from
# (results/r), so the run's outputs land beside its timings.
DATA_DIR <- Sys.getenv("DATA_DIR", unset = "../../reference")

TIMING_LOG <- "timings.tsv"
cat("phase\tseconds\n", file = TIMING_LOG)
cat(sprintf("MAXIT_CAP=%s PRED_N=%s\n", Sys.getenv("MAXIT_CAP", "none"),
            Sys.getenv("PRED_N", "5000")), file = "run_settings.txt")

# Runs expr, appends its wall-clock duration to TIMING_LOG, and returns its value.
timeit <- function(label, expr) {
  t0 <- proc.time()[["elapsed"]]
  val <- force(expr)
  dt <- proc.time()[["elapsed"]] - t0
  cat(sprintf("%s\t%.2f\n", label, dt), file = TIMING_LOG, append = TRUE)
  message(sprintf("[%s] %s: %.2f s", format(Sys.time(), "%H:%M:%S"), label, dt))
  flush(stderr())
  val
}

set_attr <- function(df, net, attr_product){
  df$src <- df$rspd_id
  df$dest <- df$model_id
  set.vertex.attribute(net, attrname = names(attr_product)[1:(ncol(attr_product)-1)],
                       value = attr_product[1:(ncol(attr_product)-1)], v = attr_product$dest)
  return(net)
}

make_network <- function(df){
  df$src <- df$rspd_id
  df$dest <- df$model_id
  unique_rspd <- unique(df$rspd_id)
  unique_model <- unique(df$model_id)
  num_rspd <- length(unique_rspd)
  num_model <- length(unique_model)
  for(i in 1:nrow(df)){
    df$src[i] <- which(unique_rspd == df$src[i])
    df$dest[i] <- which(unique_model == df$dest[i]) + num_rspd
  }
  attr_product <- unique(df[,c("V1", "V2", "V3", "V4", "dest")])
  el = as.matrix(df[c('src','dest')])
  net_consideration <- network(x = el, matrix.type = "edgelist", directed = F, bipartite = num_rspd)
  mat_inv <- 1 - as.matrix.network(net_consideration)
  el_purchase = as.matrix(df[df$purchase == 1, c('src','dest')])
  attr(el_purchase,'n') = length(unique_model) + length(unique_rspd)
  net_purchase <- network(x = el_purchase, matrix.type = "edgelist", directed = F, bipartite = num_rspd)
  net_purchase <- set_attr(df, net_purchase, attr_product)
  net_consideration <- set_attr(df, net_consideration, attr_product)
  return(list("net_purchase" = net_purchase, "net_consideration" = net_consideration, "mat_inv" = mat_inv))
}

# Part 1 -----------------------------------------------------------------
data_train <- timeit("01_read_train_csv", {
  d <- read.csv(file.path(DATA_DIR, "Sampled_data_to_share.csv"), header = TRUE)
  d$V4 <- relevel(as.factor(d$V4), "A")
  d
})
newList <- timeit("02_make_network_train", make_network(data_train))
net_purchase      <- newList$net_purchase
net_consideration <- newList$net_consideration
mat_inv           <- newList$mat_inv

# Part 2 -----------------------------------------------------------------
timeit("03_plot_networks", {
  png("net_purchase.png", width = 1200, height = 1200)
  mat_purchase <- as.matrix(net_purchase)
  plot(net_purchase,
       vertex.cex = ifelse(network.vertex.names(net_purchase) > 5000, apply(mat_purchase,2,sum)/100+1, 1),
       usearrows = FALSE,
       vertex.col = ifelse(network.vertex.names(net_purchase) > 5000, "#3399FF", "#FF9900"))
  dev.off()
  invisible(NULL)
})

# Part 3 -----------------------------------------------------------------
# MAXIT_CAP and PRED_N shorten a run for timing projections without changing
# any statistical setting: MCMLE cost is roughly linear in iterations, and the
# prediction loop does identical work per customer, so both extrapolate.
# Defaults reproduce the original script exactly.
MAXIT_CAP <- as.integer(Sys.getenv("MAXIT_CAP", unset = "1000000"))
# MAXIT replaces each fit's iteration limit rather than capping it. The script
# as published sets 30 for the star model, but the authors' published output
# reports MCMLE.maxit = 200, and at 30 ergm reports "MCMLE estimation did not
# converge". Set MAXIT=200 to match the published run.
MAXIT <- as.integer(Sys.getenv("MAXIT", unset = "0"))
# Which fits to run, comma separated. "degree" is available but excluded from
# the default: b2degrange(25) does not estimate on this data -- ergm reports
# "b2deg25+ not varying" and the fit runs indefinitely without completing an
# MCMLE iteration. Pass FITS=degree to reproduce that.
# Part 4 needs the star fit and no other, so FITS=star still produces the
# prediction timings.
FITS <- strsplit(Sys.getenv("FITS", unset = "null,star,both"), ",")[[1]]
run_fit <- function(name) name %in% FITS
CTRL <- function(maxit) control.ergm(MCMC.samplesize = 1250, MCMC.interval = 1000000,
                                     MCMLE.maxit = min(if (MAXIT > 0) MAXIT else maxit,
                                                       MAXIT_CAP),
                                     parallel = 4,
                                     parallel.type = "PSOCK", seed = 123)

if (run_fit("null")) {
  ergm_choice6_null <- timeit("04_fit_null", ergm(
    net_purchase ~ edges + offset(edgecov(mat_inv)) +
      b2cov("V1") + b2cov("V2") + b2cov("V3") + b2factor("V4"),
    offset.coef = -Inf, constraints = ~b1degrees,
    control = CTRL(100), eval.loglik = FALSE))
  print(summary(ergm_choice6_null))
  saveRDS(ergm_choice6_null, "fit_null.rds")
}

if (run_fit("degree")) {
  ergm_choice6_degree <- timeit("05_fit_degree", ergm(
    net_purchase ~ edges + offset(edgecov(mat_inv)) +
      b2cov("V1") + b2cov("V2") + b2cov("V3") + b2factor("V4") + b2degrange(25),
    offset.coef = -Inf, constraints = ~b1degrees,
    control = CTRL(30), eval.loglik = FALSE))
  print(summary(ergm_choice6_degree))
  saveRDS(ergm_choice6_degree, "fit_degree.rds")
}

if (run_fit("star")) {
  ergm_choice6_star <- timeit("06_fit_star", ergm(
    net_purchase ~ edges + offset(edgecov(mat_inv)) +
      b2cov("V1") + b2cov("V2") + b2cov("V3") + b2factor("V4") + b2star(2),
    offset.coef = -Inf, constraints = ~b1degrees,
    control = CTRL(30), eval.loglik = FALSE))
  print(summary(ergm_choice6_star))
  saveRDS(ergm_choice6_star, "fit_star.rds")
}

if (run_fit("both")) {
  ergm_choice6_both <- timeit("07_fit_both", ergm(
    net_purchase ~ edges + offset(edgecov(mat_inv)) +
      b2cov("V1") + b2cov("V2") + b2cov("V3") + b2factor("V4") + b2star(2) + b2degrange(25),
    offset.coef = -Inf, constraints = ~b1degrees,
    control = CTRL(50), eval.loglik = FALSE))
  print(summary(ergm_choice6_both))
  saveRDS(ergm_choice6_both, "fit_both.rds")
}

# Part 4 -----------------------------------------------------------------
data_test <- timeit("08_read_test_csv", {
  d <- read.csv(file.path(DATA_DIR, "test_data_to_share.csv"), header = TRUE)
  d$V4 <- relevel(as.factor(d$V4), "A")
  d
})
newList_test <- timeit("09_make_network_test", make_network(data_test))
net_purchase_test      <- newList_test$net_purchase
net_consideration_test <- newList_test$net_consideration
mat_inv_test           <- newList_test$mat_inv

considernet_test <- as.matrix.network(net_consideration_test)
purchasenet_test <- as.matrix.network(net_purchase_test)
n1 <- nrow(purchasenet_test); n2 <- ncol(purchasenet_test)
prob_y_star <- matrix(0.0, n1, n2)
model_train_star <- ergm_choice6_star

ergm_choice6_star_test <- timeit("10_fit_test_structure", ergm(
  net_purchase_test ~ edges + offset(edgecov(mat_inv_test)) +
    b2cov("V1") + b2cov("V2") + b2cov("V3") + b2factor("V4") + b2star(2),
  offset.coef = -Inf, constraints = ~b1degrees,
  control = CTRL(1), eval.loglik = FALSE))

xAlt <- ergm_choice6_star_test
xAlt$formula <- nonsimp_update.formula(xAlt$formula, xAlt$network ~ .)
z <- summary(xAlt$formula)

sample_size <- as.integer(Sys.getenv("PRED_N", unset = "5000"))
timeit("11_prediction_loop", {
  for (i in 1:sample_size){
    zAlt <- list(); n <- 1
    k <- which(purchasenet_test[i,] == 1) + 5000
    for(j in (which(considernet_test[i,] == 1) + 5000)){
      if(k == j){ zAlt[[n]] <- z } else {
        xAlt$network[i,j] <- 1; xAlt$network[i,k] <- 0
        zAlt[[n]] <- summary(xAlt$formula)
        xAlt$network[i,j] <- 0; xAlt$network[i,k] <- 1
      }
      n <- n + 1
    }
    n <- 1
    for(j in (which(considernet_test[i,] == 1) + 5000)){
      expsum <- 0
      for(num in 1:6){
        temp <- zAlt[[num]] - zAlt[[n]]
        expsum <- exp(sum(model_train_star$coef[-2]*temp[-2])) + expsum
      }
      prob_y_star[i, j-n1] <- 1.0/expsum
      n <- n + 1
    }
    if (i %% 25 == 0) { message(sprintf("  prediction %d/%d @ %s", i, sample_size, format(Sys.time(), "%H:%M:%S"))); flush(stderr()) }
  }
  invisible(NULL)
})
saveRDS(prob_y_star, "prob_y_star.rds")

right_wrong_ratio <- array(data = 0, dim = sample_size)
timeit("12_topn_eval", {
  for(i in 1:sample_size){
    temp_set <- sort(prob_y_star[i, which(considernet_test[i,] == 1)], decreasing = TRUE)
    for(k in which(considernet_test[i,] == 1)){
      for (m in 1:3){
        if (prob_y_star[i,k] == temp_set[m] && purchasenet_test[i,k] == 1) right_wrong_ratio[i] <- 1
      }
    }
  }
  invisible(NULL)
})
cat("top-3 accuracy:", sum(right_wrong_ratio)/length(right_wrong_ratio), "\n")
