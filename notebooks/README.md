# notebooks

marimo notebooks — plain Python files, so they diff and review like any other
source. Rendered outputs are cached under `__marimo__/`, which is gitignored.

| Notebook | What it covers |
|---|---|
| `01_replicate_r_script.py` | The authors' R script end to end, with the reasoning at each step |

## Running

See [Getting started](../README.md#the-notebook) in the root README for the
`edit` / `run` / `export` commands and what each is for.

## Conventions

`CLAUDE.md`'s **Notebooks** section is the authority: numbering, the
one-cell-per-name rule marimo's reactive graph imposes, and the rule that
notebooks import from `ergmpy` and define no modeling logic.
