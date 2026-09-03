# notebooks

marimo notebooks — plain Python files, not JSON, so they diff and review like
any other source.

| Notebook | What it covers |
|---|---|
| `01_replicate_r_script.py` | The authors' R script end to end, with the reasoning at each step |

Numbered sparsely (01, 10, 20) so a step can be inserted without renumbering
everything after it.

## Running

```bash
uv sync --group notebooks          # a plain `uv sync` leaves marimo out
uv run marimo edit notebooks/01_replicate_r_script.py
```

`uv run marimo run <file>` opens it read-only as an app instead, and
`uv run marimo export html <file> -o out.html` executes it headless, which is
how a notebook gets checked without a browser.

## Conventions

Notebooks import from `ergmpy` and define no modeling logic. A function here
that computes a statistic, a probability or an estimate belongs in the package
— the notebook's job is to explain and demonstrate, so that the explanation
cannot drift from the code it describes.
