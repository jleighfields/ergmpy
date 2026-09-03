# Fits the star model by ergm's own maximum pseudo-likelihood and writes the
# coefficients to results/r/mple_{train,test}.csv.
#
# This is NOT the estimator ergmpy.choice.mple implements, and the comparison
# is the point. ergm forms its pseudo-likelihood dyad by dyad, as a binary
# logit over consideration-set dyads, which discards the one-purchase-per-
# customer constraint that defines the sample space. On this model it returns
# linear coefficients whose signs contradict ergm's own MCMLE fit, and warns
# that the GLM may be separable. ergmpy conditions on the other customers
# instead, leaving a multinomial choice over each consideration set.
#
# Runs in about a second per dataset -- no MCMC is involved.
#
# Run from the repo root:  Rscript benchmarks/r/fit_mple.R

.libPaths(Sys.getenv("R_LIB", unset = "rlib"))
suppressMessages({library(ergm); library(statnet.common)})

DATA_DIR <- Sys.getenv("DATA_DIR", unset = "reference")
OUT_DIR <- "results/r"

# make_network and set_attr are defined once, in bench.R, and sourced here so
# the two paths cannot build the network differently.
bench <- readLines("benchmarks/r/bench.R")
first <- grep("^set_attr <- function", bench)
last <- grep("^# Part 1", bench) - 1
eval(parse(text = paste(bench[first:last], collapse = "\n")))

for (tag in c("train", "test")) {
  file <- if (tag == "train") "Sampled_data_to_share.csv" else "test_data_to_share.csv"
  d <- read.csv(file.path(DATA_DIR, file))
  d$V4 <- relevel(as.factor(d$V4), "A")
  nets <- make_network(d)
  # ergm builds the offset term's label from the expression passed to edgecov,
  # and that label is a lookup key on the Python side. Binding a plain local
  # first yields `edgecov.mat_inv`, matching what bench.R produces; passing
  # `nets$mat_inv` would key it as `edgecov.nets$mat_inv` instead.
  net_purchase <- nets$net_purchase
  mat_inv <- nets$mat_inv

  fit <- ergm(net_purchase ~ edges + offset(edgecov(mat_inv)) +
                b2cov("V1") + b2cov("V2") + b2cov("V3") + b2factor("V4") +
                b2star(2),
              offset.coef = -Inf, constraints = ~b1degrees,
              estimate = "MPLE", eval.loglik = FALSE)

  s <- summary(fit)$coefficients
  write.csv(data.frame(term = rownames(s), estimate = s[, 1], std_error = s[, 2]),
            file.path(OUT_DIR, sprintf("mple_%s.csv", tag)), row.names = FALSE)
  cat(sprintf("wrote %s/mple_%s.csv\n", OUT_DIR, tag))
}
