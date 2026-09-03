# Converts the fitted ergm objects bench.R saved into the CSVs the Python
# tests and benchmarks compare against.
#
# bench.R writes .rds files, which are gitignored -- they are large and
# regenerable. The CSVs written here are small, are tracked, and are what
# tests/test_predict.py and tests/test_convex_hull.py read, so the suite needs
# no R installation.
#
# Depends on bench.R having run first:
#   cd results/r && FITS=star MAXIT_CAP=2 PRED_N=200 Rscript ../../benchmarks/r/bench.R
#   cd results/r && FITS=star Rscript ../../benchmarks/r/bench.R      # converged
#
# Run from the repo root:  Rscript benchmarks/r/export_fits.R

.libPaths(Sys.getenv("R_LIB", unset = "rlib"))
suppressMessages(library(ergm))

OUT_DIR <- "results/r"

# Writes one fit's coefficient table, or says why it could not.
export_coefficients <- function(rds, csv) {
  path <- file.path(OUT_DIR, rds)
  if (!file.exists(path)) {
    cat(sprintf("SKIP %-28s (%s not found -- run bench.R first)\n", csv, rds))
    return(invisible(FALSE))
  }
  s <- summary(readRDS(path))$coefficients
  write.csv(data.frame(term = rownames(s), estimate = s[, 1], std_error = s[, 2]),
            file.path(OUT_DIR, csv), row.names = FALSE)
  cat(sprintf("wrote %s\n", csv))
  invisible(TRUE)
}

export_coefficients("fit_star_maxit2.rds", "coef_star_maxit2.csv")
export_coefficients("fit_star.rds", "mcmle_star_converged.csv")

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
