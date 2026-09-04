# Writes the control settings a fitted ergm object actually used, so the
# Python side can be matched against them rather than against the script's
# source. ergm adapts MCMC.samplesize and MCMC.interval upward to reach its
# MCMLE.effectiveSize target, so what a fit requested and what it used differ.
#
# Run from the repo root:  Rscript benchmarks/r/export_control_settings.R

.libPaths(Sys.getenv("R_LIB", unset = "rlib"))
suppressMessages(library(ergm))

OUT <- "results/r/control_settings.csv"
# MCMC.effectiveSize is the figure ergm computes for the model and adapts
# toward; MCMLE.effectiveSize is the base it starts from. Both are exported,
# because a control that names one as its counterpart has to be checkable
# against it.
KEYS <- c("MCMLE.maxit", "MCMC.samplesize", "MCMC.interval", "MCMC.burnin",
          "MCMLE.termination", "MCMLE.confidence", "MCMLE.effectiveSize",
          "MCMC.effectiveSize", "MCMC.base.effectiveSize",
          "MCMLE.effectiveSize.interval_drop", "MCMC.effectiveSize.maxruns",
          "parallel", "seed", "MCMLE.steplength", "MCMLE.steplength.margin",
          "CD.maxit", "CD.nsteps")

rows <- list()
for (rds in c("fit_star.rds", "fit_star_maxit2.rds")) {
  path <- file.path("results/r", rds)
  if (!file.exists(path)) next
  control <- readRDS(path)$control
  for (k in KEYS) {
    v <- control[[k]]
    if (!is.null(v)) {
      rows[[length(rows) + 1]] <- data.frame(
        fit = sub("\\.rds$", "", rds), setting = k,
        value = paste(v, collapse = ",")
      )
    }
  }
}

if (length(rows) > 0) {
  table <- do.call(rbind, rows)
  write.csv(table, OUT, row.names = FALSE)
  cat("wrote", OUT, "\n")
  print(table, row.names = FALSE)
}
