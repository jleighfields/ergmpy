#!/usr/bin/env bash
# Installs the R baseline's only dependency, ergm, into ./rlib at the project
# root. ergm pulls network, statnet.common, coda, trust, lpSolveAPI and
# robustbase.
#
# Installs from Posit Package Manager rather than CRAN, which serves
# precompiled binaries for this platform. From CRAN source, lpSolveAPI and
# robustbase both need a Fortran compiler; without gfortran they fail, ergm
# fails with them, and install.packages() still exits 0 -- it downgrades the
# failure to a trailing warning. The verification step at the end is what
# actually reports whether this worked.
#
# The HTTPUserAgent option is load bearing: without it the same URL serves
# source tarballs and the Fortran requirement comes back.
set -euo pipefail

P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME:-noble}}")"
mkdir -p "$P/rlib"

Rscript --vanilla - "$P/rlib" "$CODENAME" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]
repo <- sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", args[2])
options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
        paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))
install.packages("ergm", lib = lib, repos = repo, Ncpus = 8)
RS

# install.packages() cannot be trusted to report its own failure, so load the
# package and let a non-zero exit say so.
R_LIBS_USER="$P/rlib" Rscript --vanilla -e \
  'suppressMessages(library(ergm)); cat("ergm", as.character(packageVersion("ergm")), "installed\n")'
