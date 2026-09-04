# Converts the fitted ergm objects bench.R saved into the CSVs the Python
# tests and benchmarks compare against, plus a metadata table recording how
# each fit was run.
#
# bench.R writes .rds files, which are gitignored -- they are large and
# regenerable. The CSVs written here are small, are tracked, and are what
# tests/test_predict.py and tests/test_convex_hull.py read, so the suite needs
# no R installation.
#
# Depends on bench.R having run first:
#   cd results/r && FITS=star MAXIT_CAP=2 PRED_N=200 Rscript ../../benchmarks/r/bench.R
#   cd results/r && FITS=star MAXIT=200 PRED_N=1 Rscript ../../benchmarks/r/bench.R
#
# Run from the repo root:  Rscript benchmarks/r/export_fits.R

.libPaths(Sys.getenv("R_LIB", unset = "rlib"))
suppressMessages(library(ergm))

OUT_DIR <- "results/r"
metadata <- list()

# Reads one fit and returns its coefficients plus a row describing the run.
# ergm adapts MCMC.samplesize and MCMC.interval away from what control.ergm
# requested, so the values recorded here are what the fit ended up using, not
# what was asked for.
export_fit <- function(rds, csv, phase) {
  path <- file.path(OUT_DIR, rds)
  if (!file.exists(path)) {
    cat(sprintf("SKIP %-28s (%s not found -- run bench.R first)\n", csv, rds))
    return(NULL)
  }
  fit <- readRDS(path)
  s <- summary(fit)$coefficients
  write.csv(data.frame(term = rownames(s), estimate = s[, 1], std_error = s[, 2]),
            file.path(OUT_DIR, csv), row.names = FALSE)

  # bench.R records each phase's wall clock in timings.tsv.
  seconds <- NA_real_
  timings <- file.path(OUT_DIR, "timings.tsv")
  if (!is.na(phase) && file.exists(timings)) {
    t <- read.delim(timings)
    hit <- t$phase == phase
    if (any(hit)) seconds <- t$seconds[which(hit)[1]]
  }

  cat(sprintf("wrote %s\n", csv))
  data.frame(
    fit = sub("\\.csv$", "", csv),
    maxit = fit$control$MCMLE.maxit,
    iterations = fit$iterations,
    # ergm stops early when its confidence test rules out non-convergence at
    # MCMLE.confidence; reaching maxit means it never did. The run log records
    # the p-value that decided each iteration.
    converged = fit$iterations < fit$control$MCMLE.maxit,
    seconds = seconds,
    mcmc_samplesize = fit$control$MCMC.samplesize,
    mcmc_interval = fit$control$MCMC.interval,
    ergm_version = as.character(packageVersion("ergm"))
  )
}

metadata[["star"]] <- export_fit("fit_star.rds", "mcmle_star.csv", "06_fit_star")
# timings.tsv holds only the most recent bench.R run, so a fit from an earlier
# run has no timing there. Passing NA is honest; passing "06_fit_star" would
# record whichever run wrote the file last.
metadata[["star_maxit2"]] <- export_fit("fit_star_maxit2.rds",
                                        "coef_star_maxit2.csv", NA_character_)

rows <- Filter(Negate(is.null), metadata)
if (length(rows) > 0) {
  table <- do.call(rbind, rows)
  write.csv(table, file.path(OUT_DIR, "fit_metadata.csv"), row.names = FALSE)
  cat("\nwrote fit_metadata.csv\n")
  print(table, row.names = FALSE)
}

# The probability matrix is scored for only the first PRED_N customers, so the
# rows beyond that are zero and the Python comparison reads just the scored ones.
prob_path <- file.path(OUT_DIR, "prob_y_star_maxit2_n200.rds")
if (file.exists(prob_path)) {
  probabilities <- readRDS(prob_path)
  scored <- sum(rowSums(probabilities) > 0)
  write.csv(probabilities[seq_len(scored), ],
            file.path(OUT_DIR, "prob_star_maxit2_n200.csv"), row.names = FALSE)
  cat(sprintf("wrote prob_star_maxit2_n200.csv (%d scored customers)\n", scored))
} else {
  cat("SKIP prob_star_maxit2_n200.csv (prob_y_star_maxit2_n200.rds not found)\n")
}
