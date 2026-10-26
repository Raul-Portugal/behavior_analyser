# Behavioral Maze Analyzer — project guide for Claude

PyQt6 + OpenCV desktop app that scores mouse behaviour videos (Y-maze, EPM,
Freestyle/Open Field with arena & thigmotaxis, Tail Suspension Test). Used by a
neuroscience lab; numbers end up in papers, so **correctness and reproducibility
beat everything else**.

**Before planning or starting a phase, read `docs/ROADMAP.md`** (current
status, priorities, decisions, open follow-ups). Update it at the end of every
task that changes status (see "Finishing a task").

## Commands (Windows; use the build venv, not system Python)
```
# full suite (~2500 tests, 10-15 min) — offscreen Qt + throwaway settings file
QT_QPA_PLATFORM=offscreen BEHAVIORAL_MAZE_ANALYZER_SETTINGS_FILE=$TEMP/bma_test.ini .build-venv/Scripts/python.exe -B -m unittest discover -s tests
# one module (prefer this while iterating)
QT_QPA_PLATFORM=offscreen BEHAVIORAL_MAZE_ANALYZER_SETTINGS_FILE=$TEMP/bma_test.ini .build-venv/Scripts/python.exe -B -m unittest tests.test_workspace_trials_stage
.build-venv/Scripts/python.exe -m ruff check .
.build-venv/Scripts/python.exe -B gui_main.py --startup-check      # builds both windows, exit 0
.build-venv/Scripts/python.exe gui_main.py [--classic] [study.bmaproj]
.build-venv/Scripts/python.exe -m core.cli run --plan plan.json --out DIR [--workers N]
.\build_release.ps1 -DistPath build\some-dist                       # tests + PyInstaller exe
```
- `gh` lives at `"C:\Program Files\GitHub CLI\gh.exe"` (not on PATH in Git Bash).
- Run long suites in the background; don't chain `git push` after a test command
  with `&&` + `grep` — check the result explicitly before pushing.

## Layout
- `core/` — Qt-free science and data: `pipeline.py` (per-video run), `batch_runner.py`,
  `cli.py`, `detection.py`, `tst_motion_engine.py` (TST engine v3), `project.py` /
  `project_io.py` (`.bmaproj`, schema 2, template + per-video trial/zone overrides),
  `setup_state.py`, `time_bins.py`, `agreement_stats.py` / `positional_validation.py`
  (validation), `roi_alignment.py`, `models.py` (`BatchSettings`).
- `mazes/` — paradigms; paradigm-specific behaviour goes in **Maze hooks**
  (`setup_steps`, `analysis_kind`, `validate_settings`, …), not `isinstance` checks.
- `gui/workspace/` — the 4.0 single-window UI: `window.py` + `stages/`
  (videos, arena, detection, trials, review, results), design system in `theme.py`,
  `icons.py`, `ui_kit.py`. Reusable widgets: `gui/roi_editor/`, `gui/tuners/`,
  `gui/setup_panels/`, `gui/video_player.py`.
- `gui/main_window.py` — the **classic** wizard UI (`--classic`); keep it working.
- `tools/ui_gallery.py` — renders widgets/stages to PNG for visual QA.
- Docs: `readme.md`, `quickstart.md`, `OUTPUTS.md` (every output file/column),
  `CHANGELOG.md`, `TST_ENGINE.md`, `VALIDATION_PROTOCOL.md`, `docs/ROADMAP.md`.

## Invariants (do not break)
- **Analysis outputs are sacred.** A refactor or speed-up must give byte-identical
  CSV/JSON outputs; prove it with a test against the old path. New metrics are
  additive; never rename existing output columns. Byte-identity tests exist for
  CLI vs workspace vs parallel runs — keep them passing.
- `BatchSettings.to_dict()` feeds the checkpoint settings hash: don't add keys to it
  for non-analysis options (metadata, bins, UI state). Write new optional keys only
  when set.
- Bump `.bmaproj` `SCHEMA_VERSION` only with a migration + tests.
- Cross-platform CI (Windows + Ubuntu offscreen) must stay green.

## Code rules
- Match the surrounding style; no new dependencies without asking.
- Qt: **never use `QObject.sender()`** (caused segfaults); timers must be owned by a
  widget (`QTimer(self)`), not free `singleShot`s into objects that may die; join
  every QThread and release every `cv2.VideoCapture` on hide/close/shutdown.
- Workspace dialogs go through `window.prompts` so tests never block on a modal.
- Edits in workspace stages apply live (no Apply buttons); debounce, flush on leave.

## UI work
- Verify visually: render with `tools/ui_gallery.py` (default Windows platform +
  `WA_DontShowOnScreen`; `offscreen` has no fonts) at 1366×768 and 1920×1080,
  light and dark, and **look at the PNGs**. Never pop windows on the user's screen.
- Keep canvases (ROI editor, tuner, player) ≥ ~65% of the window width.

## Git
- `main` = released. Work happens on `release/x.y.0` integration branches; each
  piece of work on `feat/*`, `fix/*`, `chore/*` off it; PR the release branch to
  `main`; tag `vX.Y.Z` → `.github/workflows/release.yml` builds & attaches the exe.
- Commit after each passing task; push often (usage limits can cut a session).
- Never force-push shared branches, never `--no-verify`.

## Finishing a task
1. Tests for the change + full suite green, ruff clean, startup check passes.
2. Update `OUTPUTS.md` / `CHANGELOG.md` / `quickstart.md` if behaviour or outputs changed.
3. Update `docs/ROADMAP.md`: tick the item, add follow-ups, set "Next step".
4. Commit and push.
