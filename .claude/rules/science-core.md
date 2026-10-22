---
paths: ["core/**/*.py", "mazes/**/*.py", "output.py", "checkpoint_manager.py", "roi_manager.py"]
---
# Rules for analysis code (loaded when these files are touched)

- Analysis numbers must not change unless the task explicitly says they should.
  - Before refactoring or optimizing, keep a reference copy of the old path, or reuse an existing reference (`tests/tst_reference_engine.py`).
  - Assert identical outputs: `==` on results and byte-equal CSV/JSON.
  - If exactness is impossible, stop and say so. Don't ship "close enough".
- Float subtleties matter. Examples: `statistics.mean` vs `fsum` differs in the last bit, and `12` vs `12.0` changes the settings hash.
- Don't pass numpy scalars into OpenCV point APIs. Convert with `float(...)`.
- New outputs are additive. Document every new file or column in `OUTPUTS.md` in the same change.
- Run the byte-identity tests after touching the pipeline: `tests/test_cli.py`, `tests/test_parallel_batch.py`, `tests/test_workspace_run_tab.py` and `tests/test_workspace_trial_run.py`.
