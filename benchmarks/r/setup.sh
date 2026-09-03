#!/usr/bin/env bash
# Installs the R baseline's only dependency into ./rlib at the project root.
# ergm pulls network, statnet.common, coda, trust, DEoptimR and friends, all
# compiled from source, so this takes a few minutes on a first run.
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$P/rlib"
R_LIBS_USER="$P/rlib" Rscript -e \
  'install.packages("ergm", lib = Sys.getenv("R_LIBS_USER"), repos = "https://cloud.r-project.org", Ncpus = 8)'
