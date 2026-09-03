"""Times the Python equivalent of the R script's make_network() index mapping.

The R version assigns contiguous node ids with a `which(unique_rspd == ...)`
linear scan inside a per-row loop, so it does O(rows x unique_ids) comparisons.
Both implementations here produce the same 1-based ids the R code produces:
customers 1..num_rspd, products num_rspd+1..num_rspd+num_model.
"""

import time
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).resolve().parents[2] / "reference" / "Sampled_data_to_share.csv"


def build_ids_polars(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Maps respondent and model ids to contiguous node ids via polars.

    Args:
        path: Path to the input CSV.

    Returns:
        The src (customer) and dest (product) node id arrays.
    """
    df = pl.read_csv(path)
    # rank("dense") over the sorted unique values reproduces which(unique == x)
    # only if the R uniques were sorted; they are not, so match first-appearance
    # order explicitly instead.
    rspd_order = df["rspd_id"].unique(maintain_order=True)
    model_order = df["model_id"].unique(maintain_order=True)
    rspd_map = {v: i + 1 for i, v in enumerate(rspd_order)}
    model_map = {v: i + 1 for i, v in enumerate(model_order)}
    n_rspd = len(rspd_order)
    src = df["rspd_id"].replace_strict(rspd_map).to_numpy()
    dest = df["model_id"].replace_strict(model_map).to_numpy() + n_rspd
    return src, dest


def main() -> None:
    """Times the polars implementation over several repeats and prints the best."""
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        src, dest = build_ids_polars(DATA)
        times.append(time.perf_counter() - t0)
    print(f"rows={len(src)}  customers={src.max()}  products={dest.max() - src.max()}")
    print(f"polars read+factorize: best {min(times) * 1000:.1f} ms  "
          f"(median {sorted(times)[2] * 1000:.1f} ms of {len(times)} runs)")


if __name__ == "__main__":
    main()
