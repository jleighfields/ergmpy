"""Reads the estimates the authors published, for benchmarks to compare against.

They exist only as an image -- `reference/Plots/`'s screenshot -- so
`results/r/published_estimates.csv` is the transcription, and its accompanying
README says so. Every other file in `results/r/` is written by a script; this
one is the exception and is marked as such.

Kept out of `ergmpy/` deliberately. These are measured output from one run in
someone else's paper, not a property of the model, and no package code reads
them -- only the scripts that compare against them.
"""

import csv
from pathlib import Path

PUBLISHED = (Path(__file__).resolve().parents[2] / "results" / "r"
             / "published_estimates.csv")


def published_estimates() -> dict[str, float]:
    """Returns the published estimate for each term.

    Returns:
        Term name to estimate.

    Raises:
        FileNotFoundError: If the transcription is missing, rather than
            returning an empty comparison that would silently pass.
    """
    if not PUBLISHED.exists():
        raise FileNotFoundError(
            f"{PUBLISHED} not found; it is transcribed from the authors' "
            "screenshot and is not regenerable by any script here"
        )
    return {row["term"]: float(row["estimate"])
            for row in csv.DictReader(PUBLISHED.open())}
