# Generates the convex-hull comparison cases under results/r/convex_hull.
#
# ergm's shrink_into_CH is what .Hummel.steplength calls to find how far the
# observed statistics can move toward the simulated mean while staying inside
# the sample's convex hull. ergmpy.convex_hull reimplements it; these cases let
# that comparison run with no R installed.
#
# Cases alternate on purpose: odd-numbered test points sit well outside the
# hull (shrink factor below 1), even-numbered ones just off the centroid
# (factor above 1, since ergm does not clamp). Sizes cover the model's real
# shape, 1250 sampled points in 8 dimensions, plus smaller ones where a
# mistake is easier to read.
#
# Run from the repo root:  Rscript benchmarks/r/gen_convex_hull_cases.R

.libPaths(Sys.getenv("R_LIB", unset = "rlib"))
suppressMessages(library(ergm))

shrink <- get("shrink_into_CH", envir = asNamespace("ergm"))
out <- "results/r/convex_hull"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

set.seed(42)
sizes <- c(200, 1250, 1250, 400, 1250, 800)
dims  <- c(3, 8, 8, 5, 8, 2)

for (case in seq_along(sizes)) {
  n <- sizes[case]
  d <- dims[case]
  M <- matrix(rnorm(n * d), n, d)
  offset <- if (case %% 2 == 1) 6 else 0.05
  p <- colMeans(M) + offset * rnorm(d)

  gamma <- suppressMessages(shrink(p, M, solver = "lpsolve"))

  write.csv(M, file.path(out, sprintf("ch_M_%d.csv", case)), row.names = FALSE)
  write.csv(rbind(p), file.path(out, sprintf("ch_p_%d.csv", case)), row.names = FALSE)
  cat(sprintf("case%d n=%4d d=%d  gamma=%.10f\n", case, n, d, gamma))
}

cat("\nThese factors are the expected values in tests/test_convex_hull.py\n")
cat("and benchmarks/python/verify_ch.py; update both if this is regenerated.\n")
