#!/usr/bin/env bash
# Installs the R baseline's only dependency, ergm, into ./rlib at the project
# root, and records what it installed.
#
# Installs from a DATED Posit Package Manager snapshot rather than `latest`.
# ergm is this project's oracle -- every Python estimate is checked against
# its output -- so an ergm that silently moves underneath the comparison would
# invalidate it without anything failing. Pinning the snapshot freezes the
# whole dependency set, not just ergm's version number. To deliberately move
# to a newer ergm, set SNAPSHOT to a later date, rerun, and re-record the
# comparison outputs (see benchmarks/README.md).
#
# The snapshot also serves precompiled binaries. That matters: from CRAN
# source, lpSolveAPI and robustbase both need a Fortran compiler, and without
# gfortran they fail, ergm fails with them, and install.packages() still exits
# 0 -- it downgrades the failure to a trailing warning. The verification step
# at the end is what actually reports whether this worked.
#
# The HTTPUserAgent option is load bearing: without it the same URL serves
# source tarballs and the Fortran requirement comes back.
set -euo pipefail

SNAPSHOT="${SNAPSHOT:-2026-09-03}"
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME:-noble}}")"
mkdir -p "$P/rlib" "$P/results/r"

echo "Installing ergm from the ${SNAPSHOT} snapshot for ${CODENAME}..."

Rscript --vanilla - "$P/rlib" "$CODENAME" "$SNAPSHOT" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]
repo <- sprintf("https://packagemanager.posit.co/cran/__linux__/%s/%s", args[2], args[3])
options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
        paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))
install.packages("ergm", lib = lib, repos = repo, Ncpus = 8)
RS

# install.packages() cannot be trusted to report its own failure, so load the
# package and let a non-zero exit say so. The versions are written beside the
# measurements, because a timing without its environment cannot be compared.
R_LIBS_USER="$P/rlib" SNAPSHOT="$SNAPSHOT" Rscript --vanilla - "$P/results/r/environment.txt" <<'RS'
suppressMessages(library(ergm))
out <- commandArgs(trailingOnly = TRUE)[1]
lines <- c(
  sprintf("recorded:  %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  sprintf("snapshot:  %s", Sys.getenv("SNAPSHOT")),
  sprintf("R:         %s", R.version.string),
  sprintf("platform:  %s", R.version$platform),
  "packages:",
  sprintf("  %-16s %s", c("ergm", "network", "statnet.common", "coda"),
          vapply(c("ergm", "network", "statnet.common", "coda"),
                 function(p) as.character(packageVersion(p)), character(1)))
)
writeLines(lines, out)
cat(paste(lines, collapse = "\n"), "\n")
RS
