---
name: edit-notebook
description: Safely modify notebook cells (preserving outputs) or re-run an entire notebook headlessly. Every situation that touches a .ipynb goes through this procedure: use an nbformat script that replaces only cell["source"] (NotebookEdit wipes the outputs of the whole notebook, and training logs / solver results / figures are expensive to regenerate); for headless re-runs call nbclient directly (both nbconvert installations on this machine are broken).
---

# Safe Notebook Editing and Headless Re-runs

## Why not NotebookEdit

NotebookEdit clears the cell outputs of the entire notebook, and in a research notebook the training logs, numerical solver results and figures are expensive to regenerate (ten minutes to hours). Use an `nbformat.read/write` script that replaces only `cell["source"]` — the round-trip is verified lossless, only the edited lines change, and outputs/metadata are preserved as-is.

## Editing cell source (preserving outputs)

Template (write it into the scratchpad and run it there; it does not go into the repo):

```python
import nbformat

p = r"<absolute path to project root>/notebooks/<notebook to edit>.ipynb"
nb = nbformat.read(p, as_version=4)

hits = [c for c in nb.cells if "some unique locating substring" in c["source"]]
assert len(hits) == 1, f"expected 1 hit, got {len(hits)}"

old = "the original text being replaced"
assert hits[0]["source"].count(old) == 1
hits[0]["source"] = hits[0]["source"].replace(old, "the new content")

nbformat.write(nb, p)
```

Key points:

- **Locate via a unique substring plus an assert on the hit count**, and before replacing, `assert source.count(old) == 1` — the same discipline the Edit tool enforces, except here you have to write it yourself in the script.
- To add a cell, use `nbformat.v4.new_code_cell(...)` / `new_markdown_cell(...)` and `nb.cells.insert(i, cell)` to place it at a specific position.
- Having an edited source with stale outputs is an acceptable state; when new output is needed, re-run the corresponding cell or the whole notebook.
- When editing a plotting cell, also follow the plotting conventions in the project `CLAUDE.md` (the shared style function, no hard-coded fontsize, saving figures into the project's designated artifact directory, plotting cells re-runnable independently of upstream computation). **The specific path conventions are governed by the project `CLAUDE.md`; this skill does not legislate them.**

## Headless re-run (calling nbclient directly)

Both nbconvert installations on this machine are broken — do **not** use `jupyter nbconvert --execute`. Use nbclient:

```python
import nbformat
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(r"<absolute path to project root>")
p = ROOT / "notebooks" / "<notebook to re-run>.ipynb"

nb = nbformat.read(p, as_version=4)
client = NotebookClient(
    nb,
    timeout=None,
    resources={"metadata": {"path": str(ROOT / "notebooks")}},  # kernel cwd = notebooks/
)
client.execute()
nbformat.write(nb, p)
```

Key points:

- **A re-run overwrites all outputs — commit a snapshot of the current notebook before running.**
- Do not specify `kernel_name`; use the kernelspec that comes with the notebook metadata.
- A full re-run is a ten-minutes-to-hours job: always run it in the background and monitor the log output; never let it occupy the foreground.
- Before running, check that the notebook's heavy-job switches (`RUN_*` / `FORCE_*`, the checkpoint `resume` flag) are in the intended state: `resume=True` will silently skip already-completed training **without raising an error**; after changing the model / data / structure you must set `resume=False` or delete the old checkpoints, otherwise stale results get taken for new ones (see CLAUDE.md, "once anything upstream changes, all existing derived artifacts are void"). The staleness guard will report STALE, but the switches are your own responsibility.
- If the notebook imports torch and uses a scipy/MKL solver in the same kernel, it depends on the `KMP_DUPLICATE_LIB_OK` line in the setup cell (Windows + Anaconda MKL + pip torch ship two copies of `libiomp5md.dll`; without it you get `OMP: Error #15` and the process aborts outright). **Do not delete that line while editing.**
